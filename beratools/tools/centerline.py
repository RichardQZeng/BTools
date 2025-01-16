import logging
import time

import sys
from pathlib import Path
# from inspect import getsourcefile
#
# if __name__ == "__main__":
#     current_file = Path(getsourcefile(lambda: 0)).resolve()
#     btool_dir = current_file.parents[2]
#     sys.path.insert(0, btool_dir.as_posix())

import rasterio
import pandas as pd
import geopandas as gpd
from shapely.geometry import LineString, MultiLineString
from beratools.core.logger import Logger
import beratools.core.algo_centerline as bt_centerline
import beratools.core.dijkstra_algorithm as bt_dijkstra
import beratools.core.constants as bt_const
import beratools.tools.common as bt_common

from beratools.core.tool_base import execute_multiprocessing

log = Logger('centerline', file_level=logging.INFO)
logger = log.get_logger()
print = log.print

class SeedLine:
    def __init__(self, line_gdf, ras_file, proc_segments, line_radius) :
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

        cost_clip, out_meta = bt_common.clip_raster(in_raster, seed_line, line_radius)

        if not bt_const.HAS_COST_RASTER:
            cost_clip, _ = bt_common.cost_raster(cost_clip, out_meta)

        lc_path = line
        try:
            if bt_const.CL_USE_SKIMAGE_GRAPH:
                # skimage shortest path
                lc_path = bt_dijkstra.find_least_cost_path_skimage(cost_clip, out_meta, seed_line)
            else:
                lc_path = bt_dijkstra.find_least_cost_path(cost_clip, out_meta, seed_line)
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
            print('No least cost path detected, use input line.')
            self.line["status"] = bt_const.CenterlineStatus.FAILED.value
            return default_return

        # get corridor raster
        lc_path = LineString(lc_path_coords)
        cost_clip, out_meta = bt_common.clip_raster(in_raster, lc_path, line_radius * 0.9)
        if not bt_const.HAS_COST_RASTER:
            cost_clip, _ = bt_common.cost_raster(cost_clip, out_meta)

        out_transform = out_meta['transform']
        transformer = rasterio.transform.AffineTransformer(out_transform)
        cell_size = (out_transform[0], -out_transform[4])

        x1, y1 = lc_path_coords[0]
        x2, y2 = lc_path_coords[-1]
        source = [transformer.rowcol(x1, y1)]
        destination = [transformer.rowcol(x2, y2)]
        corridor_thresh_cl = bt_common.corridor_raster(cost_clip, out_meta, source, destination,
                                             cell_size, bt_const.FP_CORRIDOR_THRESHOLD)

        # find contiguous corridor polygon and extract centerline
        df = gpd.GeoDataFrame(geometry=[seed_line], crs=out_meta['crs'])
        corridor_poly_gpd = bt_centerline.find_corridor_polygon(corridor_thresh_cl, out_transform, df)
        center_line, status = bt_centerline.find_centerline(corridor_poly_gpd.geometry.iloc[0], lc_path)
        self.line ['status'] = status.value

        self.lc_path = self.line.copy()
        self.lc_path.geometry = [lc_path]

        self.centerline = self.line.copy()
        self.centerline.geometry = [center_line]

        self.corridor_poly_gpd = corridor_poly_gpd


def clean_geometries(gdf):
    """
    Removes rows with invalid, None, or empty geometries from the GeoDataFrame.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame to clean.

    Returns:
        GeoDataFrame: The cleaned GeoDataFrame with valid, non-null, and non-empty geometries.
    """
    # Remove rows where the geometry is invalid, None, or empty
    gdf = gdf[gdf.geometry.is_valid]  # Only keep valid geometries
    gdf = gdf[~gdf.geometry.isna()]  # Remove rows with None geometries
    gdf = gdf[gdf.geometry.apply(lambda geom: not geom.is_empty)]  # Remove empty geometries
    return gdf


def read_geospatial_file(file_path, layer=None):
    """
    Reads a geospatial file and returns a cleaned GeoDataFrame.

    Args:
        file_path (str): The path to the geospatial file (e.g., .shp, .gpkg).
        layer (str, optional): The specific layer to read if the file is multi-layered (e.g., GeoPackage).

    Returns:
        GeoDataFrame: The cleaned GeoDataFrame containing the data from the file with valid geometries only.
        None: If there is an error reading the file or layer.
    """
    try:
        if layer is None:
            gdf = gpd.read_file(file_path)  # Read the file without specifying a layer
        else:
            gdf = gpd.read_file(file_path, layer=layer)  # Read the file with the specified layer

        # Clean the geometries in the GeoDataFrame
        gdf = clean_geometries(gdf)
        gdf['BT_UID'] = range(len(gdf))  # assign temporary UID
        return gdf

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


def has_multilinestring(gdf):
    """Check if any geometry is a MultiLineString."""
    # Filter out None values (invalid geometries) from the GeoDataFrame
    valid_geometries = gdf.geometry
    return any(isinstance(geom, MultiLineString) for geom in valid_geometries)


def prepare_lines_gdf(file_path, layer=None, proc_segments=True):
    """Split lines at vertices or return original rows, with handling for MultiLineString."""
    # Check if there are any MultiLineString geometries
    gdf = read_geospatial_file(file_path, layer=layer)

    if has_multilinestring(gdf):
        gdf = gdf.explode(index_parts=False)  # Explode MultiLineStrings into individual LineStrings

    split_gdf_list = []

    for row in gdf.itertuples(index=False):  # Use itertuples to iterate
        line = row.geometry  # Access geometry directly via the named tuple

        # If proc_segment is True, split the line at vertices
        if proc_segments:
            coords = list(line.coords)  # Extract the list of coordinates (vertices)

            # For each LineString, split the line into segments by the vertices
            for i in range(len(coords) - 1):
                segment = LineString([coords[i], coords[i + 1]])

                # Copy over all non-geometry columns from the parent row (excluding 'geometry')
                attributes = {col: getattr(row, col) for col in gdf.columns if col != 'geometry'}
                single_row_gdf = gpd.GeoDataFrame([attributes], geometry=[segment], crs=gdf.crs)
                split_gdf_list.append(single_row_gdf)

        else:
            # If proc_segment is False, just add the original row as a single-row GeoDataFrame
            attributes = {col: getattr(row, col) for col in gdf.columns if col != 'geometry'}
            single_row_gdf = gpd.GeoDataFrame([attributes], geometry=[line])
            split_gdf_list.append(single_row_gdf)

    return split_gdf_list

def generate_line_class_list(in_vector, in_raster, line_radius,  layer=None, proc_segments=True)-> list:
    line_classes = []
    line_list = prepare_lines_gdf(in_vector, layer, proc_segments)

    for item in line_list :
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
    parallel_mode=bt_const.PARALLEL_MODE,
    in_layer=None,
    out_layer=None
):
    if not bt_common.compare_crs(bt_common.vector_crs(in_line), bt_common.raster_crs(in_raster)):
        print("Line and CHM have different spatial references, please check.")
        return

    line_class_list = generate_line_class_list(in_line, in_raster,
                                               line_radius=float(line_radius),
                                               layer=in_layer,
                                               proc_segments=proc_segments)

    print('{} lines to be processed.'.format(len(line_class_list)))

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

    # save features
    lc_path_list = pd.concat(lc_path_list)
    centerline_list = pd.concat(centerline_list)
    corridor_polys = pd.concat(corridor_poly_list)
    # lc_path_list.to_file(out_line, layer='least_cost_path')
    centerline_list.to_file(out_line, layer='centerline')
    # corridor_polys.to_file(out_line, layer='corridor_polygon')


# TODO: fix geometries when job done
if __name__ == '__main__':
    in_args, in_verbose = bt_common.check_arguments()
    start_time = time.time()
    centerline(**in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Elapsed time: {}'.format(time.time() - start_time))
