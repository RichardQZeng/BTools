import time
from multiprocessing.pool import Pool

import numpy as np
import geopandas as gpd
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
        line_args.append([row, inter_poly, n_samples, offset, i])

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
    return [line.interpolate(i / n_samples, normalized=True) for i in range(n_samples)]


def generate_perpendicular_line(point, line, offset=10):
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

    widths = calculate_average_width(row.geometry, inter_poly, offset, n_samples)

    # Calculate the 75th percentile width
    q3_width = np.percentile(widths, 75)
    q4_width = np.percentile(widths, 90)

    # Store the 75th percentile width as a new attribute
    row['avg_width'] = q3_width
    row['max_width'] = q4_width

    print('line processed: {}'.format(line_id))

    return row


def calculate_and_store_widths(shp_line, shp_poly, n_samples=40, offset=30):
    """
    Calculate the 75th percentile width for each line in a GeoDataFrame and
    store them as new attributes in the GeoDataFrame.

    Parameters:
    - line_gdf: A GeoDataFrame containing LineString geometries.
    - poly_gdf: A GeoDataFrame containing Polygon geometries that define the areas.
    - n_samples: The number of sample points to use when estimating the width of each line.
    - offset: The length of the perpendicular lines used to estimate the width.

    Returns:
    This function does not return a value. It modifies the input line_gdf in-place.
    """
    line_gdf = gpd.read_file(shp_line)
    poly_gdf = gpd.read_file(shp_poly)
    spatial_index = poly_gdf.sindex
    line_args = []

    for i, row in line_gdf.iterrows():
        line = row.geometry

        # Skip rows where geometry is None
        if line is None:
            print(row)
            continue

        inter_poly = poly_gdf.iloc[spatial_index.query(line)]
        line_args.append([row, inter_poly, n_samples, offset])

    results = []
    i = 0
    for line in line_args:
        result = process_single_line(line)
        results.append(result)
        print('Line processed: {}'.format(i))
        i += 1

    out_lines = gpd.GeoDataFrame(results)
    return out_lines


def generate_fixed_width_footprint(line_gdf, shp_footprint, max_width=True):
    """
    Creates a buffer around each line in the GeoDataFrame using its 'max_width' attribute and
    saves the resulting polygons in a new shapefile.

    Parameters:
    - line_gdf: A GeoDataFrame containing LineString geometries with 'max_width' attribute.
    - output_file_path: The path where the output shapefile will be stored.
    """
    # Create a new GeoDataFrame with the buffer polygons
    buffer_gdf = line_gdf.copy()

    mean_avg_width = line_gdf['avg_width'].mean()
    mean_max_width = line_gdf['max_width'].mean()

    line_gdf['avg_width'].fillna(mean_avg_width, inplace=True)
    line_gdf['max_width'].fillna(mean_max_width, inplace=True)

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
    simplified_line = line.simplify(tolerance)
    return simplified_line


def calculate_average_width(line, polygon, offset, n_samples):
    """
    Calculates the average width of a polygon perpendicular to the given line.
    """
    widths = np.zeros(n_samples)

    # Smooth the line
    line = smooth_linestring(line, tolerance=5)

    valid_widths = 0
    for i, point in enumerate(generate_sample_points(line, n_samples=n_samples)):
        perp_line = generate_perpendicular_line(point, line, offset=offset)
        intersections = polygon.intersection(perp_line)

        for intersection in intersections:
            if intersection.geom_type == 'LineString':
                widths[i] = max(widths[i], intersection.length)
                valid_widths += 1

    #     print(f"Calculated {valid_widths} valid widths")  # Logging the number of valid widths
    return widths


def line_footprint_fixed(callback, in_line, in_footprint, n_samples, offset, max_width,
                         out_footprint, processes, verbose):
    n_samples = int(n_samples)
    offset = float(offset)
    line_args = prepare_line_args(in_line, in_footprint, n_samples, offset)

    if PARALLEL_MODE == MODE_MULTIPROCESSING:
        out_lines = execute_multiprocessing(line_args, processes, verbose)
    elif PARALLEL_MODE == MODE_SEQUENTIAL:
        out_lines = execute_multiprocessing(line_args, processes, verbose)

    line_attr = gpd.GeoDataFrame(out_lines)

    # create fixed width footprint
    buffer_gdf = generate_fixed_width_footprint(line_attr, in_footprint, max_width=True)

    # Save the lines with attributes and polygons to a new shapefile
    # shp_line_attr = r'D:\Temp\test-ecosite\fixed_width_line_attr.shp'
    # print('Line with width attributes saved to... ', shp_line_attr)
    # line_attr.to_file(shp_line_attr)
    buffer_gdf.to_file(out_footprint)

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
                    print(' %{} '.format(step/total_steps*100), flush=True)

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


