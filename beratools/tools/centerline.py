"""
Copyright (C) 2025 Applied Geospatial Research Group.

This script is licensed under the GNU General Public License v3.0.
See <https://gnu.org/licenses/gpl-3.0> for full license details.

---------------------------------------------------------------------------
Author: Richard Zeng

Description:
    This script is part of the BERA Tools.
    Webpage: https://github.com/appliedgrg/beratools

    The purpose of this script is to provide main interface for centerline tool.
"""

import logging
import time

from pathlib import Path
import rasterio
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString
from beratools.core.logger import Logger
import beratools.core.algo_centerline as bt_centerline
import beratools.core.algo_dijkstra as bt_dijkstra
import beratools.core.constants as bt_const
import beratools.tools.common as bt_common
import beratools.core.algo_common as algo_common

from beratools.core.tool_base import execute_multiprocessing

log = Logger("centerline", file_level=logging.INFO)
logger = log.get_logger()
print = log.print


class SeedLine:
    def __init__(self, line_gdf, ras_file, proc_segments, line_radius):
        self.line = line_gdf
        self.raster = ras_file
        self.line_radius = line_radius
        self.lc_path = None
        self.centerline = None
        self.corridor_poly_gpd = None

    def compute(self):
        line = self.line.geometry[0]
        line_radius = self.line_radius
        in_raster = self.raster
        seed_line = line  # LineString
        default_return = (seed_line, seed_line, None)

        ras_clip, out_meta = bt_common.clip_raster(in_raster, seed_line, line_radius)
        cost_clip, _ = algo_common.cost_raster(ras_clip, out_meta)

        lc_path = line
        try:
            if bt_const.CL_USE_SKIMAGE_GRAPH:
                # skimage shortest path
                lc_path = bt_dijkstra.find_least_cost_path_skimage(
                    cost_clip, out_meta, seed_line
                )
            else:
                lc_path = bt_dijkstra.find_least_cost_path(
                    cost_clip, out_meta, seed_line
                )
        except Exception as e:
            print(e)
            return default_return

        if lc_path:
            lc_path_coords = lc_path.coords
        else:
            lc_path_coords = []

        self.lc_path = lc_path

        # search for centerline
        if len(lc_path_coords) < 2:
            print("No least cost path detected, use input line.")
            self.line["status"] = bt_const.CenterlineStatus.FAILED.value
            return default_return

        # get corridor raster
        lc_path = LineString(lc_path_coords)
        ras_clip, out_meta = bt_common.clip_raster(
            in_raster, lc_path, line_radius * 0.9
        )
        cost_clip, _ = algo_common.cost_raster(ras_clip, out_meta)

        out_transform = out_meta["transform"]
        transformer = rasterio.transform.AffineTransformer(out_transform)
        cell_size = (out_transform[0], -out_transform[4])

        x1, y1 = lc_path_coords[0]
        x2, y2 = lc_path_coords[-1]
        source = [transformer.rowcol(x1, y1)]
        destination = [transformer.rowcol(x2, y2)]
        corridor_thresh_cl = algo_common.corridor_raster(
            cost_clip,
            out_meta,
            source,
            destination,
            cell_size,
            bt_const.FP_CORRIDOR_THRESHOLD,
        )

        # find contiguous corridor polygon and extract centerline
        df = gpd.GeoDataFrame(geometry=[seed_line], crs=out_meta["crs"])
        corridor_poly_gpd = bt_centerline.find_corridor_polygon(
            corridor_thresh_cl, out_transform, df
        )
        center_line, status = bt_centerline.find_centerline(
            corridor_poly_gpd.geometry.iloc[0], lc_path
        )
        self.line["status"] = status.value

        self.lc_path = self.line.copy()
        self.lc_path.geometry = [lc_path]

        self.centerline = self.line.copy()
        self.centerline.geometry = [center_line]

        self.corridor_poly_gpd = corridor_poly_gpd


def generate_line_class_list(
    in_vector, in_raster, line_radius, layer=None, proc_segments=True
) -> list:
    line_classes = []
    line_list = algo_common.prepare_lines_gdf(in_vector, layer, proc_segments)

    for item in line_list:
        line_classes.append(SeedLine(item, in_raster, proc_segments, line_radius))

    return line_classes


def process_single_line_class(seed_line):
    seed_line.compute()
    return seed_line


def centerline(
    in_line,
    in_raster,
    line_radius,
    proc_segments,
    out_line,
    processes,
    verbose,
    callback=print,
    parallel_mode=bt_const.ParallelMode.SEQUENTIAL,
    in_layer=None,
    out_layer=None,
):
    if not bt_common.compare_crs(
        bt_common.vector_crs(in_line), bt_common.raster_crs(in_raster)
    ):
        print("Line and CHM have different spatial references, please check.")
        return

    line_class_list = generate_line_class_list(
        in_line,
        in_raster,
        line_radius=float(line_radius),
        layer=in_layer,
        proc_segments=proc_segments,
    )

    print("{} lines to be processed.".format(len(line_class_list)))

    lc_path_list = []
    centerline_list = []
    corridor_poly_list = []
    result = execute_multiprocessing(
        process_single_line_class,
        line_class_list,
        "Centerline",
        processes,
        verbose=verbose,
        mode=parallel_mode,
    )

    for item in result:
        lc_path_list.append(item.lc_path)
        centerline_list.append(item.centerline)
        corridor_poly_list.append(item.corridor_poly_gpd)

    # Concatenate the lists of GeoDataFrames into single GeoDataFrames
    lc_path_list = pd.concat(lc_path_list, ignore_index=True)
    centerline_list = pd.concat(centerline_list, ignore_index=True)
    corridor_polys = pd.concat(corridor_poly_list, ignore_index=True)

    # Save the concatenated GeoDataFrames to the shapefile/gpkg
    centerline_list.to_file(out_line, layer=out_layer)

    # Check if the output file is a shapefile
    out_line_path = Path(out_line)

    if out_line_path.suffix == ".shp":
        # Generate the new file name for the GeoPackage with '_aux' appended
        aux_file = out_line_path.with_name(out_line_path.stem + "_aux.gpkg")
        print(f"Saved auxiliary data to: {aux_file}")
    else:
        aux_file = out_line  # continue using out_line (gpkg)

    # Save lc_path_list and corridor_polys to the new GeoPackage with '_aux' suffix
    lc_path_list.to_file(aux_file, layer="least_cost_path")
    corridor_polys.to_file(aux_file, layer="corridor_polygon")


# TODO: fix geometries when job done
if __name__ == "__main__":
    in_args, in_verbose = bt_common.check_arguments()
    start_time = time.time()
    centerline(**in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print("Elapsed time: {}".format(time.time() - start_time))
