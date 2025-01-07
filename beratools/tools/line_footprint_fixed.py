import time
from itertools import chain

import sys
from pathlib import Path
from inspect import getsourcefile

if __name__ == "__main__":
    current_file = Path(getsourcefile(lambda: 0)).resolve()
    btool_dir = current_file.parents[2]
    sys.path.insert(0, btool_dir.as_posix())

from math import isclose
import numpy as np
import pandas as pd
import geopandas as gpd
import shapely.ops
from shapely.geometry import Polygon, MultiPolygon, MultiLineString, mapping, Point
from beratools.tools.common import (
    generate_perpendicular_line_precise,
    check_arguments,
)
from beratools.core.tool_base import execute_multiprocessing
from beratools.core.constants import FP_FIXED_WIDTH_DEFAULT, ParallelMode
from beratools.core.linegrouping import LineGrouping


def prepare_line_args(line_gdf, poly_gdf, n_samples, offset):
    """
    Args:
        line_gdf
        poly_gdf
        n_samples
        offset

    Returns:
        line_args : list
            row :
            inter_poly :
            n_samples :
            offset :
            i :  line ID

    """
    spatial_index = poly_gdf.sindex
    line_args = []

    for row in line_gdf.itertuples():
        line = row.geometry

        # Skip rows where geometry is None
        if line is None:
            print(row)
            continue

        inter_poly = poly_gdf.loc[spatial_index.query(line)]
        try: 
            line_args.append(
                [line_gdf.loc[[row.Index]], inter_poly, n_samples, offset, row.Index]
            )
        except Exception as e:
            print(e)

    return line_args


# Calculating Line Widths
def generate_sample_points(line, n_samples=10):
    """
    Generate evenly spaced points along a line.

    Args:
        line (LineString): The line along which to generate points.
        n_samples (int): The number of points to generate (default is 10).

    Returns
        list:  List of shapely Point objects.
    """
    # TODO: determine line type
    try:
        pts = line.coords
    except Exception as e:
        line = shapely.ops.linemerge(line)
        tuple_coord = mapping(line)['coordinates']
        pts = list(chain(*tuple_coord))

    return [Point(item) for item in pts]


def process_single_line(line_arg):
    row = line_arg[0]
    inter_poly = line_arg[1]
    n_samples = line_arg[2]
    offset = line_arg[3]
    line_id = line_arg[4]

    widths, line, perp_lines, perp_lines_original = calculate_average_width(row.iloc[0].geometry, inter_poly, offset, n_samples)

    # Calculate the 75th percentile width
    # filter zeros in width array
    arr_filter = [False if isclose(i, 0.0) else True for i in widths]
    widths = widths[arr_filter]

    q3_width = FP_FIXED_WIDTH_DEFAULT
    q4_width = FP_FIXED_WIDTH_DEFAULT
    try:
        q3_width = np.percentile(widths, 40)
        q4_width = np.percentile(widths, 90)
    except Exception as e:
        print(e)

    # Store the 75th percentile width as a new attribute
    row['avg_width'] = q3_width
    row['max_width'] = q4_width
    # hist, bins = np.histogram(widths)
    # bins = pd.Series(bins).rolling(2).mean()[1:].to_numpy()  # mid-points of bins
    # row['width_hist'] = str(hist)
    # row['width_bins'] = str(bins)

    row['geometry'] = line
    try:
        row['perp_lines'] = perp_lines
        row['perp_lines_original'] = perp_lines_original
    except Exception as e:
        print(e)

    # print('line processed: {}'.format(line_id))

    return row


def generate_fixed_width_footprint(line_gdf, max_width=False):
    """
    Creates a buffer around each line in the GeoDataFrame using its 'max_width' attribute and
    saves the resulting polygons in a new shapefile.

    Parameters:
    - line_gdf: A GeoDataFrame containing LineString geometries with 'max_width' attribute.
    - output_file_path: The path where the output shapefile will be stored.
    """
    # Create a new GeoDataFrame with the buffer polygons
    buffer_gdf = line_gdf.copy(deep=True)

    mean_avg_width = line_gdf['avg_width'].mean()
    mean_max_width = line_gdf['max_width'].mean()

    line_gdf['avg_width'].fillna(mean_avg_width, inplace=True)
    line_gdf['max_width'].fillna(mean_max_width, inplace=True)

    line_gdf['avg_width'].replace(0.0, mean_avg_width, inplace=True)
    line_gdf['max_width'].replace(0.0, mean_max_width, inplace=True)

    if not max_width:
        print('Using quantile 75% width')
        buffer_gdf['geometry'] = line_gdf.apply(
            lambda row: row.geometry.buffer(row.avg_width / 2) if row.geometry is not None else None, axis=1)
    else:
        print('Using quantile 90% + 20% width')
        buffer_gdf['geometry'] = line_gdf.apply(
            lambda row: row.geometry.buffer(row.max_width * 1.2 / 2) if row.geometry is not None else None, axis=1)

    return buffer_gdf


def smooth_linestring(line, tolerance=1.0):
    """
    Smooths a LineString geometry using the Ramer-Douglas-Peucker algorithm.

    Parameters:
    - line: The LineString geometry to smooth.
    - tolerance: The maximum distance from a point to a line for the point to be considered part of the line.

    Returns:
    The smoothed LineString geometry.
    """
    # simplified_line = line.simplify(tolerance)
    simplified_line = line
    return simplified_line


def calculate_average_width(line, polygon, offset, n_samples):
    """
    Calculates the average width of a polygon perpendicular to the given line.
    """
    # Smooth the line
    line = smooth_linestring(line, tolerance=1.0)

    valid_widths = 0
    sample_points = generate_sample_points(line, n_samples=n_samples)
    sample_points_pairs = list(zip(sample_points[:-2], sample_points[1:-1], sample_points[2:]))
    widths = np.zeros(len(sample_points_pairs))
    perp_lines = []
    perp_lines_original = []

    # remove polygon holes
    poly_list = []
    for geom in polygon.geometry:
        if type(geom) is MultiPolygon:
            for item in geom.geoms:
                poly_list.append(Polygon(list(item.exterior.coords)))
        else:
            poly_list.append(Polygon(list(geom.exterior.coords)))

    polygon_no_holes = gpd.GeoDataFrame(geometry=poly_list, crs=polygon.crs)

    for i, points in enumerate(sample_points_pairs):
        perp_line = generate_perpendicular_line_precise(points, offset=offset)
        perp_lines_original.append(perp_line)

        polygon_intersect = polygon_no_holes.iloc[polygon_no_holes.sindex.query(perp_line)]
        intersections = polygon_intersect.intersection(perp_line)

        line_list = []
        try:
            for inter in intersections:
                if not inter.is_empty:
                    if type(inter) is MultiLineString:
                        line_list += list(inter.geoms)
                    else:
                        line_list.append(inter)

            perp_lines += line_list
        except Exception as e:
            print(e)

        try:
            for item in line_list:
                widths[i] = max(widths[i], item.length)
                valid_widths += 1
        except Exception as e:
            print(e)

    return widths, line, MultiLineString(perp_lines), MultiLineString(perp_lines_original)


def line_footprint_fixed(callback, in_line, in_footprint, n_samples, offset, max_width,
                         out_footprint, processes, verbose):
    n_samples = int(n_samples)
    offset = float(offset)
    line_gdf = gpd.read_file(in_line)
    poly_gdf = gpd.read_file(in_footprint)

    lg = LineGrouping(line_gdf)
    lg.run_grouping()
    merged_line_gdf = LineGrouping.run_line_merge(line_gdf)

    line_args = prepare_line_args(merged_line_gdf, poly_gdf, n_samples, offset)
    out_lines = execute_multiprocessing(process_single_line, line_args, 'Fixed footprint', processes)
    line_attr = pd.concat(out_lines)

    # create fixed width footprint
    buffer_gdf = generate_fixed_width_footprint(line_attr, max_width=max_width)

    # Save the lines with attributes and polygons to a new file
    perp_lines_gdf = buffer_gdf.copy(deep=True)
    perp_lines_origianl_gdf = buffer_gdf.copy(deep=True)

    # save fixed width footprint
    buffer_gdf = buffer_gdf.drop(columns=['perp_lines'])
    buffer_gdf = buffer_gdf.drop(columns=['perp_lines_original'])
    buffer_gdf.crs = perp_lines_gdf.crs
    buffer_gdf.reset_index(inplace=True, drop=True)

    # save original merged lines
    merged_line_gdf.to_file(out_footprint, layer="merged_lines_original")

    # trim lines and footprints
    lg.run_cleanup(buffer_gdf)
    lg.save_file(out_footprint)

    # perpendicular lines
    layer = 'perp_lines'
    out_footprint = Path(out_footprint)
    out_aux_gpkg = out_footprint.with_stem(out_footprint.stem + '_aux').with_suffix('.gpkg')
    perp_lines_gdf = perp_lines_gdf.set_geometry('perp_lines')
    perp_lines_gdf = perp_lines_gdf.drop(columns=['perp_lines_original'])
    perp_lines_gdf = perp_lines_gdf.drop(columns=['geometry'])
    perp_lines_gdf.crs = buffer_gdf.crs
    perp_lines_gdf.to_file(out_aux_gpkg.as_posix(), layer=layer)

    layer = 'perp_lines_original'
    perp_lines_origianl_gdf = perp_lines_origianl_gdf.set_geometry('perp_lines_original')
    perp_lines_origianl_gdf = perp_lines_origianl_gdf.drop(columns=['perp_lines'])
    perp_lines_origianl_gdf = perp_lines_origianl_gdf.drop(columns=['geometry'])
    perp_lines_origianl_gdf.crs = buffer_gdf.crs
    perp_lines_origianl_gdf.to_file(out_aux_gpkg.as_posix(), layer=layer)

    layer = 'centerline_simplified'
    line_attr = line_attr.drop(columns='perp_lines')
    line_attr.to_file(out_aux_gpkg.as_posix(), layer=layer)

    callback('Fixed width footprint tool finished.')


if __name__ == '__main__':
    in_args, in_verbose = check_arguments()
    start_time = time.time()
    line_footprint_fixed(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Elapsed time: {}'.format(time.time() - start_time))
