#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString


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
    return [line.interpolate(i/n_samples, normalized=True) for i in range(n_samples)]


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
    angle_perp = angle + np.pi/2.0  # Perpendicular angle
    
    # Generate the perpendicular line
    perp_line = LineString([(point.x - offset*np.cos(angle_perp), point.y - offset*np.sin(angle_perp)), 
                            (point.x + offset*np.cos(angle_perp), point.y + offset*np.sin(angle_perp))])
    
    return perp_line


def process_single_line(line_arg):
    row = line_arg[0]
    inter_poly = line_arg[1]
    n_samples = line_arg[2]
    offset = line_arg[3]

    widths = calculate_average_width(row.geometry, inter_poly, n_samples, offset)

    # Calculate the 75th percentile width
    q3_width = np.percentile(widths, 75)
    q4_width = np.percentile(widths, 90)

    # Store the 75th percentile width as a new attribute
    row['avg_width'] = q3_width
    row['max_width'] = q4_width

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
        buffer_gdf['geometry'] = line_gdf.apply(lambda row: row.geometry.buffer(row.avg_width / 2) if row.geometry is not None else None, axis=1)
    else:
        print('Using quantile 90% + 20% width')
        buffer_gdf['geometry'] = line_gdf.apply(lambda row: row.geometry.buffer(row.max_width * 1.2 / 2) if row.geometry is not None else None, axis=1)

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


# In summary, your workflow will be like this:
# 
#     Read the line and polygon shapefiles.
#     For each line in the line shapefile:
#     a. Generate several points along the line using generate_sample_points.
#     b. For each generated point, create a perpendicular line using generate_perpendicular_line.
#     c. Calculate the width of the line at each point by finding the intersection of the perpendicular line with the
#     corresponding polygon and calculating the length of the intersected line.
#     d. Calculate the average width of the line by averaging the widths calculated at each point.
#     Add the average width as a new attribute to the line in the line shapefile.
#     Save the updated line shapefile.
# 
# The calculate_average_width function integrates these steps (2a-d) into one function, making it easier to
# apply to each line in your shapefile.
# 
# The calculate_average_width function is designed to calculate the average width for a single line geometry.
# The calculate_and_store_widths function, on the other hand, is intended to be used for an entire GeoDataFrame,
# where it applies the calculate_average_width function to each line geometry and stores
# the resulting width in the GeoDataFrame.

if __name__ == '__main__':
    shp_poly = r'D:\Temp\test-ecosite\footprint_rel.shp'
    shp_line = r'D:\Temp\test-ecosite\centerline_attr.shp'
    shp_line_attr = r'D:\Temp\test-ecosite\fixed_width_line_attr.shp'
    shp_footprint = r'D:\Temp\test-ecosite\fixed_width_footprint.shp'

    # Read the shapefiles
    line_attr = calculate_and_store_widths(shp_line, shp_poly, n_samples=10, offset=10)

    # create fixed width footprint
    buffer_gdf = generate_fixed_width_footprint(line_attr, shp_footprint, max_width=True)

    print('Line with width attributes saved to... ', shp_line_attr)
    line_attr.to_file(shp_line_attr)

    # Save the buffer polygons to a new shapefile
    buffer_gdf.to_file(shp_footprint)
