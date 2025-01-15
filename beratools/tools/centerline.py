import logging
import time

import sys
from pathlib import Path
from inspect import getsourcefile

if __name__ == "__main__":
    current_file = Path(getsourcefile(lambda: 0)).resolve()
    btool_dir = current_file.parents[2]
    sys.path.insert(0, btool_dir.as_posix())

import fiona
import rasterio
import pandas as pd
import geopandas as gpd
from geopandas import GeoDataFrame
from shapely.geometry import shape, LineString, MultiLineString
from beratools.core.logger import Logger
from beratools.core.algo_centerline import find_corridor_polygon, find_centerline
from beratools.core.dijkstra_algorithm import (
    find_least_cost_path_skimage,
    find_least_cost_path,
)
from beratools.core.constants import (
    HAS_COST_RASTER,
    CL_USE_SKIMAGE_GRAPH,
    CenterlineStatus,
    FP_CORRIDOR_THRESHOLD,
    PARALLEL_MODE,
)
from beratools.core.tool_base import execute_multiprocessing
from beratools.tools.common import (
    clip_raster,
    cost_raster,
    corridor_raster,
    compare_crs,
    vector_crs,
    raster_crs,
    segments,
    save_features_to_file,
    check_arguments,
)

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
        # prop = line_args[0][1]
        line_radius = self.line_radius
        in_raster = self.raster

        seed_line = line  # LineString
        # line_radius = float(line_radius)

        default_return = (seed_line, seed_line, None)

        cost_clip, out_meta = clip_raster(in_raster, seed_line, line_radius)

        if not HAS_COST_RASTER:
            cost_clip, _ = cost_raster(cost_clip, out_meta)

        lc_path = line
        try:
            if CL_USE_SKIMAGE_GRAPH:
                # skimage shortest path
                lc_path = find_least_cost_path_skimage(cost_clip, out_meta, seed_line)
            else:
                lc_path = find_least_cost_path(cost_clip, out_meta, seed_line)
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
            self.line["status"] = CenterlineStatus.FAILED.value
            return default_return

        # get corridor raster
        lc_path = LineString(lc_path_coords)
        cost_clip, out_meta = clip_raster(in_raster, lc_path, line_radius * 0.9)
        if not HAS_COST_RASTER:
            cost_clip, _ = cost_raster(cost_clip, out_meta)

        out_transform = out_meta['transform']
        transformer = rasterio.transform.AffineTransformer(out_transform)
        cell_size = (out_transform[0], -out_transform[4])

        x1, y1 = lc_path_coords[0]
        x2, y2 = lc_path_coords[-1]
        source = [transformer.rowcol(x1, y1)]
        destination = [transformer.rowcol(x2, y2)]
        corridor_thresh_cl = corridor_raster(cost_clip, out_meta, source, destination,
                                             cell_size, FP_CORRIDOR_THRESHOLD)

        # find contiguous corridor polygon and extract centerline
        df = GeoDataFrame(geometry=[seed_line], crs=out_meta['crs'])
        corridor_poly_gpd = find_corridor_polygon(corridor_thresh_cl, out_transform, df)
        center_line, status = find_centerline(corridor_poly_gpd.geometry.iloc[0], lc_path)
        self.line ['status'] = status.value

        self.lc_path = self.line.copy()
        self.lc_path.geometry = [lc_path]

        self.centerline = self.line.copy()
        self.centerline.geometry = [center_line]

        self.corridor_poly_gpd = corridor_poly_gpd

        # print(" Searching centerline: line {} ".format(line_id), flush=True)
        # return center_line, lc_path, corridor_poly_gpd


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

    # List to hold the resulting single-row GeoDataFrames
    split_gdf_list = []

    for row in gdf.itertuples(index=False):  # Use itertuples to iterate
        line = row.geometry  # Access geometry directly via the named tuple

        # If proc_segment is True, split the line at vertices
        if proc_segments:
            coords = list(line.coords)  # Extract the list of coordinates (vertices)

            # For each LineString, split the line into segments by the vertices
            for i in range(len(coords) - 1):
                # Create a new segment (LineString) from each pair of consecutive coordinates
                segment = LineString([coords[i], coords[i + 1]])

                # Copy over all non-geometry columns from the parent row (excluding 'geometry')
                attributes = {col: getattr(row, col) for col in gdf.columns if col != 'geometry'}

                # Create a single-row GeoDataFrame for the segment
                single_row_gdf = gpd.GeoDataFrame([attributes], geometry=[segment])

                # Append this single-row GeoDataFrame to the list
                split_gdf_list.append(single_row_gdf)

        else:
            # If proc_segment is False, just add the original row as a single-row GeoDataFrame
            attributes = {col: getattr(row, col) for col in gdf.columns if col != 'geometry'}
            single_row_gdf = gpd.GeoDataFrame([attributes], geometry=[line])
            split_gdf_list.append(single_row_gdf)

    # Return the list of single-row GeoDataFrames, but merge them back into a single GeoDataFrame
    return split_gdf_list

def generate_line_class_list(in_vector, in_raster, line_radius,  layer=None, proc_segments=True):
    line_class_list = []
    line_list = prepare_lines_gdf(in_vector, layer, proc_segments)

    for item in line_list :
        line_class_list.append(SeedLine(item, in_raster, proc_segments, line_radius))

    return line_class_list

def process_single_line_class(seed_line):
    seed_line.compute()
    return seed_line

def process_single_line(line_args):
    line = line_args[0][0]
    prop = line_args[0][1]
    line_radius = line_args[1]
    in_raster = line_args[2]
    line_id = line_args[3]
    seed_line = shape(line)  # LineString
    line_radius = float(line_radius)

    default_return = (seed_line, seed_line, prop, None)

    cost_clip, out_meta = clip_raster(in_raster, seed_line, line_radius)

    if not HAS_COST_RASTER:
        cost_clip, _ = cost_raster(cost_clip, out_meta)

    lc_path = line
    try:
        if CL_USE_SKIMAGE_GRAPH:
            # skimage shortest path
            lc_path = find_least_cost_path_skimage(cost_clip, out_meta, seed_line)
        else:
            lc_path = find_least_cost_path(cost_clip, out_meta, seed_line)
    except Exception as e:
        print(e)
        return default_return

    if lc_path:
        lc_path_coords = lc_path.coords
    else:
        lc_path_coords = []

    # search for centerline
    if len(lc_path_coords) < 2:
        print('No least cost path detected, use input line.')
        prop['status'] = CenterlineStatus.FAILED.value
        return default_return

    # get corridor raster
    lc_path = LineString(lc_path_coords)
    cost_clip, out_meta = clip_raster(in_raster, lc_path, line_radius * 0.9)
    if not HAS_COST_RASTER:
        cost_clip, _ = cost_raster(cost_clip, out_meta)

    out_transform = out_meta['transform']
    transformer = rasterio.transform.AffineTransformer(out_transform)
    cell_size = (out_transform[0], -out_transform[4])

    x1, y1 = lc_path_coords[0]
    x2, y2 = lc_path_coords[-1]
    source = [transformer.rowcol(x1, y1)]
    destination = [transformer.rowcol(x2, y2)]
    corridor_thresh_cl = corridor_raster(cost_clip, out_meta, source, destination,
                                         cell_size, FP_CORRIDOR_THRESHOLD)

    # find contiguous corridor polygon and extract centerline
    df = GeoDataFrame(geometry=[seed_line], crs=out_meta['crs'])
    corridor_poly_gpd = find_corridor_polygon(corridor_thresh_cl, out_transform, df)
    center_line, status = find_centerline(corridor_poly_gpd.geometry.iloc[0], lc_path)
    prop['status'] = status.value

    # print(" Searching centerline: line {} ".format(line_id), flush=True)
    return center_line, lc_path, prop, corridor_poly_gpd


def centerline(
    in_line,
    in_raster,
    line_radius,
    proc_segments,
    out_line,
    processes,
    verbose,
    callback=print,
    parallel_mode=PARALLEL_MODE,
    in_layer=None,
    out_layer=None
):
    if not compare_crs(vector_crs(in_line), raster_crs(in_raster)):
        print("Line and CHM have different spatial references, please check.")
        return

    # Read input line features
    layer_crs = None
    schema = None
    input_lines = []

    with fiona.open(in_line) as open_line_file:
        layer_crs = open_line_file.crs
        schema = open_line_file.meta['schema']

        if 'BT_UID' not in schema['properties']:
            schema['properties']['BT_UID'] = 'int'

        uid = 0
        for line in open_line_file:
            line.properties['BT_UID'] = uid
            if line.geometry:
                if line.geometry['type'] != 'MultiLineString':
                    input_lines.append([line.geometry, line.properties])
                else:
                    print('MultiLineString found.')
                    geoms = shape(line['geometry']).geoms
                    for item in geoms:
                        line_part = fiona.Geometry.from_dict(item)
                        if line_part:
                            input_lines.append([line_part, line.properties])
            else:
                print(f'Line {line.id} has empty geometry.')

            uid += 1

    if proc_segments:
        # split line segments at vertices
        input_lines_temp = []
        for line in input_lines:
            line_seg = line[0]
            line_prop = line[1]
            line_segs = segments(line_seg.coordinates)
            line_feats = [(line, line_prop) for line in line_segs]
            if line_segs:
                input_lines_temp.extend(line_feats)

        input_lines = input_lines_temp

    # Process lines
    all_lines = []
    i = 0
    for line in input_lines:
        all_lines.append((line, line_radius, in_raster, i))
        i += 1

    print('{} lines to be processed.'.format(len(all_lines)))

    line_class_list = generate_line_class_list(in_line, in_raster,
                                               line_radius=float(line_radius),
                                               layer=in_layer,
                                               proc_segments=proc_segments)

    lc_path_list = []
    # feat_props = []
    centerline_list = []
    corridor_poly_list = []
    # result = execute_multiprocessing(
    #     process_single_line,
    #     all_lines,
    #     "Centerline",
    #     processes,
    #     verbose=verbose,
    #     mode=parallel_mode,
    # )
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

    # out_centerline_path = Path(out_line)
    # schema['properties']['status'] = 'int'
    # save_features_to_file(out_centerline_path.as_posix(), layer_crs, center_line_geoms, feat_props, schema)
    #
    # driver = 'GPKG'
    # layer = 'least_cost_path'
    # out_aux_gpkg = out_centerline_path.with_stem(out_centerline_path.stem + '_aux').with_suffix('.gpkg')
    # save_features_to_file(out_aux_gpkg.as_posix(), layer_crs, lc_path_geoms, feat_props,
    #                       schema, driver=driver, layer=layer)

    # save features
    lc_path_list = pd.concat(lc_path_list)
    centerline_list = pd.concat(centerline_list)
    corridor_polys = pd.concat(corridor_poly_list)
    lc_path_list.to_file(out_line, layer='least_cost_path')
    centerline_list.to_file(out_line, layer='centerline')
    corridor_polys.to_file(out_line, layer='corridor_polygon')


# TODO: fix geometries when job done
if __name__ == '__main__':
    in_args, in_verbose = check_arguments()
    start_time = time.time()
    centerline(**in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Elapsed time: {}'.format(time.time() - start_time))
