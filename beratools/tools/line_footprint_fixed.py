import time
import math
from multiprocessing.pool import Pool

import numpy as np
import geopandas as gpd
import pandas as pd
from shapely.affinity import scale, rotate
from shapely.geometry import Point, LineString

from common import *


class OperationCancelledException(Exception):
    pass


def prepare_line_args(shp_line, shp_poly, n_samples, offset):
    """
    Parameters
    ----------
    shp_line
    shp_poly
    n_samples
    offset

    Returns
    -------
    line_args : list
        row :
        inter_poly :
        n_samples :
        offset :
        i :  line ID

    """
    line_gdf = gpd.read_file(shp_line)
    poly_gdf = gpd.read_file(shp_poly)
    spatial_index = poly_gdf.sindex
    line_args = []

    i = 0
    for i, row in line_gdf.iterrows():
        line = row.geometry

        # Skip rows where geometry is None
        if line is None:
            print(row)
            continue

        inter_poly = poly_gdf.iloc[spatial_index.query(line)]
        line_args.append([line_gdf.iloc[[i]], inter_poly, n_samples, offset, i])

    return line_args


# Calculating Line Widths
def generate_sample_points(line, n_samples=10):
    """
    Generate evenly spaced points along a line.

    Parameters
    ----------
    line : shapely LineString
        The line along which to generate points.
    n_samples : int, optional
        The number of points to generate (default is 10).

    Returns
    -------
    list
        List of shapely Point objects.
    """
    # return [line.interpolate(i / n_samples, normalized=True) for i in range(n_samples)]
    return [Point(item) for item in list(line.coords)]


def generate_perpendicular_line(point, line, offset=20):
    """
    Generate a perpendicular line to the input line at the given point.

    Parameters
    ----------
    point : shapely.geometry.Point
        The point on the line where the perpendicular should be generated.
    line : shapely.geometry.LineString
        The line to which the perpendicular line will be generated.
    offset : float, optional
        The length of the perpendicular line.

    Returns
    -------
    shapely.geometry.LineString
        The generated perpendicular line.
    """
    # Compute the angle of the line
    p1, p2 = line.coords[0], line.coords[-1]  # Modify this line
    angle = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])

    # Compute the angle of the perpendicular line
    angle_perp = angle + np.pi / 2.0  # Perpendicular angle

    # Generate the perpendicular line
    perp_line = LineString([(point.x - offset * np.cos(angle_perp), point.y - offset * np.sin(angle_perp)),
                            (point.x + offset * np.cos(angle_perp), point.y + offset * np.sin(angle_perp))])

    return perp_line


def process_single_line(line_arg):
    row = line_arg[0]
    inter_poly = line_arg[1]
    n_samples = line_arg[2]
    offset = line_arg[3]
    line_id = line_arg[4]

    widths, line, perp_lines = calculate_average_width(row.iloc[0].geometry, inter_poly, offset, n_samples)

    # Calculate the 75th percentile width
    # filter zeros in width array
    arr_filter = [False if math.isclose(i, 0.0) else True for i in widths]
    widths = widths[arr_filter]
    q3_width = np.percentile(widths, 40)
    q4_width = np.percentile(widths, 90)

    # Store the 75th percentile width as a new attribute
    row['avg_width'] = q3_width
    row['max_width'] = q4_width
    row['sampling_widths'] = [widths]
    row['sampling_widths'] = row['sampling_widths'].apply(lambda x: str(x))

    row['geometry'] = line
    try:
        row['perp_lines'] = perp_lines
    except Exception as e:
        print(e)

    print('line processed: {}'.format(line_id))

    return row


def generate_fixed_width_footprint(line_gdf, shp_footprint, max_width=False):
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

    return widths, line, MultiLineString(perp_lines)


def line_footprint_fixed(callback, in_line, in_footprint, n_samples, offset, max_width,
                         out_footprint, processes, verbose):
    n_samples = int(n_samples)
    offset = float(offset)
    line_args = prepare_line_args(in_line, in_footprint, n_samples, offset)

    if PARALLEL_MODE == MODE_MULTIPROCESSING:
        out_lines = execute_multiprocessing(line_args, processes, verbose)
    elif PARALLEL_MODE == MODE_SEQUENTIAL:
        out_lines = execute_multiprocessing(line_args, processes, verbose)

    line_attr = pd.concat(out_lines)

    # create fixed width footprint
    buffer_gdf = generate_fixed_width_footprint(line_attr, in_footprint, max_width=max_width)

    # Save the lines with attributes and polygons to a new shapefile
    # shp_line_attr = r'D:\Temp\test-ecosite\fixed_width_line_attr.shp'
    # print('Line with width attributes saved to... ', shp_line_attr)
    # line_attr.to_file(shp_line_attr)
    perp_lines_gdf = buffer_gdf.copy(deep=True)
    buffer_gdf = buffer_gdf.drop(columns=['perp_lines'])
    buffer_gdf.crs = perp_lines_gdf.crs
    buffer_gdf.to_file(out_footprint)

    # perpendicular lines
    perp_lines_gdf = perp_lines_gdf.set_geometry('perp_lines')
    perp_lines_gdf = perp_lines_gdf.drop(columns=['geometry'])
    perp_lines_gdf.crs = buffer_gdf.crs
    perp_lines_path = Path(out_footprint).with_stem(Path(out_footprint).stem+'_perp_lines')
    perp_lines_gdf.to_file(perp_lines_path)

    geojson_path = Path(out_footprint).with_suffix('.geojson')
    buffer_gpd_4326 = buffer_gdf.to_crs('EPSG:4326')
    buffer_gpd_4326.to_file(geojson_path.as_posix(), driver='GeoJSON')

    gdf_simplified_path = Path(in_line).with_stem(Path(in_line).stem + "_simplified")
    line_attr = line_attr.drop(columns='perp_lines')
    line_attr.to_file(gdf_simplified_path)

    callback('tool_template tool done.')


# protect the entry point
def execute_multiprocessing(line_args, processes, verbose):
    try:
        total_steps = len(line_args)
        features = []
        with Pool(processes) as pool:
            step = 0
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(process_single_line, line_args):
                if BT_DEBUGGING:
                    print('Got result: {}'.format(result), flush=True)

                features.append(result)
                step += 1
                if verbose:
                    print(' "PROGRESS_LABEL Ceterline {} of {}" '.format(step, total_steps), flush=True)
                    print(' %{} '.format(step / total_steps * 100), flush=True)

        return features
    except OperationCancelledException:
        print("Operation cancelled")
        return None


def execute_sequential_processing(line_args, processes, verbose):
    total_steps = len(line_args)
    step = 0
    features = []
    for line in line_args:
        result = process_single_line(line)
        features.append(result)

        step += 1
        if verbose:
            print(' "PROGRESS_LABEL Ceterline {} of {}" '.format(step, total_steps), flush=True)
            print(' %{} '.format(step / total_steps * 100), flush=True)

    return features


if __name__ == '__main__':
    in_args, in_verbose = check_arguments()
    start_time = time.time()
    line_footprint_fixed(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Elapsed time: {}'.format(time.time() - start_time))

