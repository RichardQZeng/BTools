#!/usr/bin/env python3
""" This file is intended to be hosting common functions for BERA Tools.
"""

# This script is part of the BERA Tools geospatial library.
# Author: Richard Zeng
# Created: 12/04/2023
# License: MIT

# imports
import sys
import math
import tempfile
from pathlib import Path
from collections import OrderedDict
from itertools import zip_longest, compress

import json
import shlex
import uuid
import argparse
import numpy as np

from dask.distributed import Client, progress, as_completed
import multiprocessing
import ray

import rasterio
import rasterio.mask
from rasterio import features

import fiona
from fiona import Geometry

import shapely
from shapely.affinity import rotate
from shapely.ops import unary_union, snap, split, transform, substring, linemerge, nearest_points
from shapely.geometry import (shape, mapping, Point, LineString,
                              MultiLineString, MultiPoint, Polygon, MultiPolygon)

import pandas as pd
import geopandas as gpd
from osgeo import ogr, gdal, osr
from pyproj import CRS, Transformer
from pyogrio import set_gdal_config_options

from skimage.graph import MCP_Geometric, route_through_array

from label_centerlines import get_centerline
from multiprocessing.pool import Pool

from enum import IntEnum, unique
@unique
class CenterlineStatus(IntEnum):
    SUCCESS = 1
    FAILED = 2
    REGENERATE_SUCCESS = 3
    REGENERATE_FAILED = 4


# constants
MODE_MULTIPROCESSING = 1
MODE_SEQUENTIAL = 2
MODE_DASK = 3
MODE_RAY = 4

PARALLEL_MODE = MODE_MULTIPROCESSING
@unique
class ParallelMode(IntEnum):
    MULTIPROCESSING = 1
    SEQUENTIAL = 2
    DASK = 3
    RAY = 4

USE_SCIPY_DISTANCE = True
USE_NUMPY_FOR_DIJKSTRA = True

NADDatum=['NAD83 Canadian Spatial Reference System', 'North American Datum 1983']

BT_NODATA = -9999
BT_NODATA_COST = np.inf
BT_DEBUGGING = False
BT_MAXIMUM_CPU_CORES = 60  # multiprocessing has limit of 64, consider pathos
BT_BUFFER_RATIO = 0.0  # overlapping ratio of raster when clipping lines
BT_LABEL_MIN_WIDTH = 130
BT_SHOW_ADVANCED_OPTIONS = False
BT_EPSLON = sys.float_info.epsilon  # np.finfo(float).eps
BT_UID = 'BT_UID'

GROUPING_SEGMENT = True
LP_SEGMENT_LENGTH = 500

# centerline
CL_USE_SKIMAGE_GRAPH = True
CL_BUFFER_CLIP = 5.0
CL_BUFFER_CENTROID = 3.0
CL_SNAP_TOLERANCE = 15.0
CL_SEGMENTIZE_LENGTH = 1.0
CL_SIMPLIFY_LENGTH = 0.5
CL_SMOOTH_SIGMA = 0.8
CL_DELETE_HOLES = True
CL_SIMPLIFY_POLYGON = True
CL_CLEANUP_POLYGON_BY_AREA = 1.0
CL_POLYGON_BUFFER = 1e-6

FP_CORRIDOR_THRESHOLD = 2.5
FP_SEGMENTIZE_LENGTH = 2.0
FP_FIXED_WIDTH_DEFAULT = 5.0
FP_PERP_LINE_OFFSET = 30.0

# restore .shx for shapefile for using GDAL or pyogrio
gdal.SetConfigOption('SHAPE_RESTORE_SHX', 'YES')
set_gdal_config_options({'SHAPE_RESTORE_SHX': 'YES'})

# suppress all kinds of warnings
if not BT_DEBUGGING:
    # gdal warning
    gdal.SetConfigOption('CPL_LOG', 'NUL')

    # suppress warnings
    import warnings
    warnings.filterwarnings("ignore")

    # to suppress Pandas UserWarning: Geometry column does not contain geometry when splitting lines
    warnings.simplefilter(action='ignore', category=UserWarning)


def clip_raster(in_raster_file, clip_geom, buffer=0.0, out_raster_file=None, ras_nodata=BT_NODATA):
    out_meta = None
    with (rasterio.open(in_raster_file)) as raster_file:
        out_meta = raster_file.meta
        if out_meta['nodata']:
            ras_nodata = out_meta['nodata']
        else:
            out_meta['nodata'] = ras_nodata

        clip_geo_buffer = [clip_geom.buffer(buffer)]
        out_image: np.ndarray
        out_image, out_transform = rasterio.mask.mask(raster_file, clip_geo_buffer,
                                                      crop=True, nodata=ras_nodata, filled=True)

    height, width = out_image.shape[1:]
    out_meta.update({"driver": "GTiff",
                     "height": height,
                     "width": width,
                     "transform": out_transform,
                     "nodata": ras_nodata})

    if out_raster_file:
        with rasterio.open(out_raster_file, "w", **out_meta) as dest:
            dest.write(out_image)
            print('[Clip raster]: data saved to {}.'.format(out_raster_file))

    return out_image, out_meta


def save_raster_to_file(in_raster_mem, in_meta, out_raster_file):
    """

    Parameters
    ----------
    in_raster_mem: npmpy raster
    in_meta: input meta
    out_raster_file: output raster file

    Returns
    -------

    """
    with rasterio.open(out_raster_file, "w", **in_meta) as dest:
        dest.write(in_raster_mem, indexes=1)


def clip_lines(clip_geom, buffer, in_line_file, out_line_file):
    in_line = gpd.read_file(in_line_file)
    out_line = in_line.clip(clip_geom.buffer(buffer*BT_BUFFER_RATIO))

    if out_line_file and len(out_line) > 0:
        out_line.to_file(out_line_file)
        print('[Clip lines]:  data saved to {}.'.format(out_line_file))

    return out_line


def read_geoms_from_shapefile(in_file):
    geoms = []
    with fiona.open(in_file) as open_file:
        layer_crs = open_file.crs
        for geom in open_file:
            geoms.append(geom['geometry'])

    return geoms


# Read feature from shapefile
def read_feature_from_shapefile(in_file):
    shapes = []
    with fiona.open(in_file) as open_file:
        for feat in open_file:
            shapes.append([shape(feat.geometry), feat.properties])

    return shapes


def generate_raster_footprint(in_raster, latlon=True):
    inter_img = 'myimage.tif'

    #  get raster datasource
    src_ds = gdal.Open(in_raster)
    width, height = src_ds.RasterXSize, src_ds.RasterYSize
    coords_geo = []

    # ensure there is nodata
    # gdal_translate ... -a_nodata 0 ... outimage.vrt
    # gdal_edit -a_nodata 255 somefile.tif

    # gdal_translate -outsize 1024 0 vendor_image.tif myimage.tif
    options = None
    with tempfile.TemporaryDirectory() as tmp_folder:
        if BT_DEBUGGING:
            print('Temporary folder: {}'.format(tmp_folder))

        if max(width, height) <= 1024:
            inter_img = in_raster
        else:
            if width >= height:
                options = gdal.TranslateOptions(width=1024, height=0)
            else:
                options = gdal.TranslateOptions(width=0, height=1024)

            inter_img = Path(tmp_folder).joinpath(inter_img).as_posix()
            gdal.Translate(inter_img, src_ds, options=options)

        coords = None
        with rasterio.open(inter_img) as src:
            msk = src.read_masks(1)
            shapes = features.shapes(msk, mask=msk)
            shapes = list(shapes)
            coords = shapes[0][0]['coordinates'][0]

            for pt in coords:
                pt = rasterio.transform.xy(src.transform, pt[1], pt[0])
                coords_geo.append(pt)

    coords_geo.pop(-1)

    if latlon:
        in_crs = CRS(src_ds.GetSpatialRef().ExportToWkt())
        out_crs = CRS('EPSG:4326')
        transformer = Transformer.from_crs(in_crs, out_crs)

        coords_geo = list(transformer.itransform(coords_geo))
        coords_geo = [list(pt) for pt in coords_geo]

    return coords_geo if len(coords_geo) > 0 else None


def remove_nan_from_array(matrix):
    with np.nditer(matrix, op_flags=['readwrite']) as it:
        for x in it:
            if np.isnan(x[...]):
                x[...] = BT_NODATA_COST


# Split LineString to segments at vertices
def segments(line_coords):
    if len(line_coords) < 2:
        return None
    elif len(line_coords) == 2:
        return [Geometry.from_dict({'type': 'LineString', 'coordinates': line_coords})]
    else:
        seg_list = zip(line_coords[:-1], line_coords[1:])
        line_list = [{'type': 'LineString', 'coordinates': coords} for coords in seg_list]
        return [Geometry.from_dict(line) for line in line_list]


def extract_string_from_printout(str_print, str_extract):
    str_array = shlex.split(str_print)  # keep string in double quotes
    str_array_enum = enumerate(str_array)
    index = 0
    for item in str_array_enum:
        if str_extract in item[1]:
            index = item[0]
            break
    str_out = str_array[index]
    return str_out.strip()


def check_arguments():
    # Get tool arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=json.loads)
    parser.add_argument('-p', '--processes')
    parser.add_argument('-v', '--verbose')
    args = parser.parse_args()

    verbose = True if args.verbose == 'True' else False
    for item in args.input:
        if args.input[item] == 'false':
            args.input[item] = False
        elif args.input[item] == 'true':
            args.input[item] = True

    return args, verbose


def save_features_to_shapefile(out_file, crs, geoms, schema=None, properties=None):
    """

    Parameters
    ----------
    out_file :
    crs :
    geoms : shapely geometry objects
    schema :
    properties :

    Returns
    -------

    """
    # remove all None items
    # TODO: check geom type consistency
    # geoms = [item for item in geoms if item is not None]

    if len(geoms) < 1:
        return

    try:
        geom_type = mapping(geoms[0])['type']
    except Exception as e:
        print(e)

    if not schema:
        props_tuple = zip([], [])  # if lengths are not the same, ValueError raises
        props_schema = [(item, type(value).__name__) for item, value in props_tuple]

        schema = {
            'geometry': geom_type,
            'properties': OrderedDict([])
        }

        properties = None

    driver = 'ESRI Shapefile'
    print('Writing to shapefile {}'.format(out_file))

    try:
        out_line_file = fiona.open(out_file, 'w', driver, schema, crs)
    except Exception as e:
        print(e)
        out_line_file.close()
        return

    if properties:
        feat_tuple = zip_longest(geoms, properties)
    else:  # properties are None
        feat_tuple = [(item, None) for item in geoms]

    try:
        for geom, prop in feat_tuple:
            # prop_zip = {} if prop is None else OrderedDict(list(zip(fields, prop)))

            if geom:
                feature = {
                    'geometry': mapping(geom),
                    'properties': prop
                }

                out_line_file.write(feature)
    except Exception as e:
        print(e)

    out_line_file.close()


def vector_crs(in_vector):
    vec_crs = None
    with ogr.Open(in_vector) as vector_file:
        if vector_file:
            vec_crs = vector_file.GetLayer().GetSpatialRef()

    return vec_crs


def raster_crs(in_raster):
    ras_crs = None
    with gdal.Open(in_raster) as raster_file:
        if raster_file:
            ras_crs = raster_file.GetSpatialRef()

    return ras_crs


def compare_crs(crs_org, crs_dst):
    if crs_org and crs_dst:
        if crs_org.IsSameGeogCS(crs_dst):
            print('Check: Input file Spatial Reference are the same, continue.')
            return True
        else:
            crs_org_norm = CRS(crs_org.ExportToWkt())
            crs_dst_norm = CRS(crs_dst.ExportToWkt())
            if crs_org_norm.is_compound:
                crs_org_proj=crs_org_norm.sub_crs_list[0].coordinate_operation.name
            elif crs_org_norm.name == 'unnamed':
                return False
            else:
                crs_org_proj=crs_org_norm.coordinate_operation.name

            if crs_dst_norm.is_compound:
                crs_dst_proj = crs_dst_norm.sub_crs_list[0].coordinate_operation.name
            elif crs_org_norm.name == 'unnamed':
                return False
            else:
                crs_dst_proj = crs_dst_norm.coordinate_operation.name

            if crs_org_proj == crs_dst_proj:
                print('Checked: Input files Spatial Reference are the same, continue.')
                return True

    return False


def identity_polygon(line_args):
    """
    Return polygon of line segment

    Parameters
    ----------
    line_args : list of geodataframe
        0 : geodataframe line segment, one item
        1 : geodataframe line buffer, one item
        2 : geodataframe polygons returned by spatial search

    Returns
    -------
        line, identity :  tuple of line and associated footprint

    """
    line = line_args[0]
    in_cl_buffer = line_args[1][['geometry', 'OLnFID']]
    in_fp_polygon = line_args[2]

    identity = None
    try:
        # drop polygons not intersecting with line segment
        line_geom = line.iloc[0].geometry
        drop_list = []
        for i in in_fp_polygon.index:
            if not in_fp_polygon.loc[i].geometry.intersects(line_geom):
                drop_list.append(i)
            elif line_geom.intersection(in_fp_polygon.loc[i].geometry).length/line_geom.length < 0.30:
                drop_list.append(i)  # if less the 1/5 of line is inside of polygon, ignore

        # drop all polygons not used
        in_fp_polygon = in_fp_polygon.drop(index=drop_list)

        if not in_fp_polygon.empty:
            identity = in_fp_polygon.overlay(in_cl_buffer, how='intersection')
            # identity = identity.dropna(subset=['OLnSEG_2', 'OLnFID_2'])
            # identity = identity.drop(columns=['OLnSEG_1', 'OLnFID_2'])
            # identity = identity.rename(columns={'OLnFID_1': 'OLnFID', 'OLnSEG_2': 'OLnSEG'})
    except Exception as e:
        print(e)

    return line, identity


def line_split2(in_ln_shp,seg_length):

    # Check the OLnFID column in data. If it is not, column will be created
    if 'OLnFID' not in in_ln_shp.columns.array:
        if BT_DEBUGGING:
            print("Cannot find {} column in input line data")

        print("New column created: {}".format('OLnFID', 'OLnFID'))
        in_ln_shp['OLnFID'] = in_ln_shp.index
    line_seg = split_into_Equal_Nth_segments(in_ln_shp, seg_length)

    return line_seg


def split_into_Equal_Nth_segments(df, seg_length):
    odf = df
    crs = odf.crs
    if 'OLnSEG' not in odf.columns.array:
        df['OLnSEG'] = np.nan
    df = odf.assign(geometry=odf.apply(lambda x: cut_line(x.geometry,seg_length), axis=1))
    # df = odf.assign(geometry=odf.apply(lambda x: cut_line(x.geometry, x.geometry.length), axis=1))
    df=df.explode()

    df['OLnSEG'] = df.groupby('OLnFID').cumcount()
    gdf = gpd.GeoDataFrame(df, geometry=df.geometry, crs=crs)
    gdf = gdf.sort_values(by=['OLnFID', 'OLnSEG'])
    gdf = gdf.reset_index(drop=True)

    if "shape_leng" in gdf.columns.array:
        gdf["shape_leng"] = gdf.geometry.length
    elif "LENGTH" in gdf.columns.array:
        gdf["LENGTH"]=gdf.geometry.length
    else:
        gdf["shape_leng"] = gdf.geometry.length
    return gdf


def split_line_nPart(line,seg_length):
    seg_line = shapely.segmentize(line, seg_length)
    distances = np.arange(seg_length, line.length, seg_length)

    if len(distances) > 0:
        points = [shapely.line_interpolate_point(seg_line, distance) for distance in distances]

        split_points = shapely.multipoints(points)
        mline = split(seg_line, split_points)
    else:
        mline = seg_line

    return mline


def cut_line(line, distance):
    """

    Parameters
    ----------
    line : LineString line to be split by distance along line
    distance : float length of segment to cut

    Returns
    -------
    List of LineString
    """
    lines = list()
    lines = cut(line, distance, lines)
    return lines


def cut(line, distance, lines):
    # Cuts a line in several segments at a distance from its starting point
    if line.has_z:
        line = transform(lambda x, y, z=None: (x, y), line)

    if shapely.is_empty(line) or shapely.is_missing(line):
        return None
    # else:
    if math.fmod(line.length, distance) < 1:
        return [line]
    elif distance >= line.length:
        return [line]
    # else:
    end_pt = None
    line = shapely.segmentize(line, distance)

    while line.length > distance:
        coords = list(line.coords)
        for i, p in enumerate(coords):
            pd = line.project(Point(p))

            if abs(pd - distance) < BT_EPSLON:
                lines.append(LineString(coords[:i+1]))
                line = LineString(coords[i:])
                end_pt = None
                break
            elif pd > distance:
                end_pt = line.interpolate(distance)
                lines.append(LineString(coords[:i] + list(end_pt.coords)))
                line = LineString(list(end_pt.coords) + coords[i:])
                break

    if end_pt:
        lines.append(line)
    return lines


def find_centerline(poly, input_line):
    """
    Parameters
    ----------
    poly : Polygon
    input_line : LineString
        Least cost path or seed line

    Returns
    -------

    """
    default_return = input_line, CenterlineStatus.FAILED
    if not poly:
        print('find_centerline: No polygon found')
        return default_return

    poly = shapely.segmentize(poly, max_segment_length=CL_SEGMENTIZE_LENGTH)

    poly = poly.buffer(CL_POLYGON_BUFFER)  # buffer polygon to reduce MultiPolygons
    if type(poly) is MultiPolygon:
        print('MultiPolygon encountered, skip.')
        return default_return

    exterior_pts = list(poly.exterior.coords)

    if CL_DELETE_HOLES:
        poly = Polygon(exterior_pts)
    if CL_SIMPLIFY_POLYGON:
        poly = poly.simplify(CL_SIMPLIFY_LENGTH)

    try:
        centerline = get_centerline(poly, segmentize_maxlen=1, max_points=3000,
                                    simplification=0.05, smooth_sigma=CL_SMOOTH_SIGMA, max_paths=1)
    except Exception as e:
        print('Exception in get_centerline.')
        return default_return

    if type(centerline) is MultiLineString:
        if len(centerline.geoms) > 1:
            print(" Multiple centerline segments detected, no further processing.")
            return centerline, CenterlineStatus.SUCCESS  # TODO: inspect
        elif len(centerline.geoms) == 1:
            centerline = centerline.geoms[0]
        else:
            return default_return

    cl_coords = list(centerline.coords)

    # trim centerline at two ends
    head_buffer = Point(cl_coords[0]).buffer(CL_BUFFER_CLIP)
    centerline = centerline.difference(head_buffer)

    end_buffer = Point(cl_coords[-1]).buffer(CL_BUFFER_CLIP)
    centerline = centerline.difference(end_buffer)

    if not centerline:
        print('No centerline detected, use input line instead.')
        return default_return
    try:
        if centerline.is_empty:
            print('Empty centerline detected, use input line instead.')
            return default_return
    except Exception as e:
        print(e)

    centerline = snap_end_to_end(centerline, input_line)

    # Check if centerline is valid. If not, regenerate by splitting polygon into two halves.
    if not centerline_is_valid(centerline, input_line):
        try:
            print(f'Regenerating line ...')
            centerline = regenerate_centerline(poly, input_line)
            return centerline, CenterlineStatus.REGENERATE_SUCCESS
        except Exception as e:
            print('find_centerline:  Exception occurred. \n {}'.format(e))
            return input_line, CenterlineStatus.REGENERATE_FAILED

    return centerline, CenterlineStatus.SUCCESS


def find_route(array, start, end, fully_connected,geometric):
    from skimage.graph import route_through_array
    route_list,cost_list = route_through_array(array, start, end,fully_connected,geometric)
    return route_list,cost_list


def find_corridor_polygon(corridor_thresh, in_transform, line_gpd):
    # Threshold corridor raster used for generating centerline
    corridor_thresh_cl = np.ma.where(corridor_thresh == 0.0, 1, 0).data
    corridor_mask = np.where(1 == corridor_thresh_cl, True, False)
    poly_generator = features.shapes(corridor_thresh_cl, mask=corridor_mask, transform=in_transform)
    corridor_polygon = []

    try:
        for poly, value in poly_generator:
            corridor_polygon.append(shape(poly))
    except Exception as e:
        print(e)

    if corridor_polygon:
        corridor_polygon = unary_union(corridor_polygon)
    else:
        corridor_polygon = None

    # create GeoDataFrame for centerline
    corridor_poly_gpd = gpd.GeoDataFrame.copy(line_gpd)
    corridor_poly_gpd.geometry = [corridor_polygon]

    return corridor_poly_gpd


def find_centerlines(poly_gpd, line_seg, processes):
    centerline = None
    centerline_gpd = []
    rows_and_paths = []

    try:
        for i in poly_gpd.index:
            row = poly_gpd.loc[[i]]
            poly = row.geometry.iloc[0]
            line_id = row['OLnFID']
            lc_path = line_seg.loc[line_id].geometry.iloc[0]
            rows_and_paths.append((row, lc_path))
    except Exception as e:
        print(e)

    total_steps = len(rows_and_paths)
    step = 0

    if PARALLEL_MODE == MODE_MULTIPROCESSING:
        with Pool(processes=processes) as pool:
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(find_single_centerline, rows_and_paths):
                centerline_gpd.append(result)
                step += 1
                print(' "PROGRESS_LABEL Centerline {} of {}" '.format(step, total_steps), flush=True)
                print(' %{} '.format(step / total_steps * 100))
                print('Centerline No. {} done'.format(step))
    elif PARALLEL_MODE == MODE_SEQUENTIAL:
        for item in rows_and_paths:
            row_with_centerline = find_single_centerline(item)
            centerline_gpd.append(row_with_centerline)
            step += 1
            print(' "PROGRESS_LABEL Centerline {} of {}" '.format(step, total_steps), flush=True)
            print(' %{} '.format(step / total_steps * 100))
            print('Centerline No. {} done'.format(step))

    return pd.concat(centerline_gpd)


def find_single_centerline(row_and_path):
    """

    Parameters
    ----------
    row_and_path:
        list of row (polygon and props) and least cost path
        first is geopandas row, second is input line, (least cost path)

    Returns
    -------

    """
    row = row_and_path[0]
    lc_path = row_and_path[1]

    poly = row.geometry.iloc[0]
    centerline = find_centerline(poly, lc_path)
    row['centerline'] = centerline

    return row


def line_angle(point_1, point_2):
    """
    Calculates the angle of the line

    Parameters
    ----------
    point_1, point_2: start and end points of shapely line
    """
    delta_y = point_2.y - point_1.y
    delta_x = point_2.x - point_1.x

    angle = math.atan2(delta_y, delta_x)
    return angle


def centerline_is_valid(centerline, input_line):
    """
    Check if centerline is valid
    Parameters
    ----------
    centerline :
    input_line : shapely LineString
        This can be input seed line or least cost path. Only two end points are used.

    Returns
    -------

    """
    if not centerline:
        return False

    # centerline length less the half of least cost path
    if (centerline.length < input_line.length / 2 or
            centerline.distance(Point(input_line.coords[0])) > BT_EPSLON or
            centerline.distance(Point(input_line.coords[-1])) > BT_EPSLON):
        return False

    return True


def generate_perpendicular_line_precise(points, offset=20):
    """
    Generate a perpendicular line to the input line at the given point.

    Parameters
    ----------
    points : shapely.geometry.Point list
        The points on the line where the perpendicular should be generated.
    offset : float, optional
        The length of the perpendicular line.

    Returns
    -------
    shapely.geometry.LineString
        The generated perpendicular line.
    """
    # Compute the angle of the line
    center = points[1]
    perp_line = None

    if len(points) == 2:
        head = points[0]
        tail = points[1]

        delta_x = head.x - tail.x
        delta_y = head.y - tail.y
        angle = 0.0

        if math.isclose(delta_x, 0.0):
            angle = math.pi / 2
        else:
            angle = math.atan(delta_y / delta_x)

        start = [center.x + offset / 2.0, center.y]
        end = [center.x - offset / 2.0, center.y]
        line = LineString([start, end])
        perp_line = rotate(line, angle + math.pi / 2.0, origin=center, use_radians=True)
    elif len(points) == 3:
        head = points[0]
        tail = points[2]

        angle_1 = line_angle(center, head)
        angle_2 = line_angle(center, tail)
        angle_diff = (angle_2 - angle_1) / 2.0
        head_line = LineString([center, head])
        head_new = Point(center.x + offset / 2.0 * math.cos(angle_1), center.y + offset / 2.0 * math.sin(angle_1))
        if head.has_z:
            head_new = shapely.force_3d(head_new)
        try:
            perp_seg_1 = LineString([center, head_new])
            perp_seg_1 = rotate(perp_seg_1, angle_diff, origin=center, use_radians=True)
            perp_seg_2 = rotate(perp_seg_1, math.pi, origin=center, use_radians=True)
            perp_line = LineString([list(perp_seg_1.coords)[1], list(perp_seg_2.coords)[1]])
        except Exception as e:
            print(e)

    return perp_line


def regenerate_centerline(poly, input_line):
    """
    Regenerates centerline when initial
    ----------
    poly : line is not valid
    Parameters
    input_line : shapely LineString
        This can be input seed line or least cost path. Only two end points will be used

    Returns
    -------

    """
    line_1 = substring(input_line, start_dist=0.0, end_dist=input_line.length/2)
    line_2 = substring(input_line, start_dist=input_line.length/2, end_dist=input_line.length)

    pts = shapely.force_2d([Point(list(input_line.coords)[0]),
                            Point(list(line_1.coords)[-1]),
                            Point(list(input_line.coords)[-1])])
    perp = generate_perpendicular_line_precise(pts)

    # MultiPolygon is rare, but need to be dealt with
    # remove polygon of area less than CL_CLEANUP_POLYGON_BY_AREA
    poly = poly.buffer(CL_POLYGON_BUFFER)
    if type(poly) is MultiPolygon:
        poly_geoms = list(poly.geoms)
        poly_valid = [True] * len(poly_geoms)
        for i, item in enumerate(poly_geoms):
            if item.area < CL_CLEANUP_POLYGON_BY_AREA:
                poly_valid[i] = False

        poly_geoms = list(compress(poly_geoms, poly_valid))
        if len(poly_geoms) != 1:  # still multi polygon
            print('regenerate_centerline: Multi or none polygon found, pass.')

        poly = Polygon(poly_geoms[0])

    poly_exterior = Polygon(poly.buffer(CL_POLYGON_BUFFER).exterior)
    poly_split = split(poly_exterior, perp)

    if len(poly_split.geoms) < 2:
        print('regenerate_centerline: polygon split failed, pass.')
        return None

    poly_1 = poly_split.geoms[0]
    poly_2 = poly_split.geoms[1]

    # find polygon and line pairs
    pair_line_1 = line_1
    pair_line_2 = line_2
    if not poly_1.intersects(line_1):
        pair_line_1 = line_2
        pair_line_2 = line_1
    elif poly_1.intersection(line_1).length < line_1.length / 3:
        pair_line_1 = line_2
        pair_line_2 = line_1

    center_line_1 = find_centerline(poly_1, pair_line_1)
    center_line_2 = find_centerline(poly_2, pair_line_2)

    status_1 = center_line_1[1]
    center_line_1 = center_line_1[0]
    status_2 = center_line_2[1]
    center_line_2 = center_line_2[0]

    if not center_line_1 or not center_line_2:
        print('Regenerate line: centerline is None')
        return None

    try:
        if center_line_1.is_empty or center_line_2.is_empty:
            print('Regenerate line: centerline is empty')
            return None
    except Exception as e:
        print(e)

    print(f'Centerline is regenerated.')
    return linemerge(MultiLineString([center_line_1, center_line_2]))


def snap_end_to_end(in_line, line_reference):
    if type(in_line) is MultiLineString:
        in_line = linemerge(in_line)
        if type(in_line) is MultiLineString:
            print(f'MultiLineString found {in_line.centroid}, pass.')
            return None

    pts = list(in_line.coords)
    if len(pts) < 2:
        print('snap_end_to_end: input line invalid.')
        return in_line

    line_start = Point(pts[0])
    line_end = Point(pts[-1])
    ref_ends = MultiPoint([line_reference.coords[0], line_reference.coords[-1]])

    _, snap_start = nearest_points(line_start, ref_ends)
    _, snap_end = nearest_points(line_end, ref_ends)

    if in_line.has_z:
        snap_start = shapely.force_3d(snap_start)
        snap_end = shapely.force_3d(snap_end)
    else:
        snap_start = shapely.force_2d(snap_start)
        snap_end = shapely.force_2d(snap_end)

    pts[0] = snap_start.coords[0]
    pts[-1] = snap_end.coords[0]

    return LineString(pts)


def corridor_raster(raster_clip, out_meta, source, destination, cell_size, corridor_threshold):
    """
    Calculate corridor raster
    Parameters
    ----------
    raster_clip : raster
    out_meta : raster file meta
    source : list of point tuple(s)
        start point in row/col
    destination : list of point tuple(s)
        end point in row/col
    cell_size: tuple
        (cell_size_x, cell_size_y)
    corridor_threshold : double

    Returns
    -------
    corridor raster
    """

    try:
        # change all nan to BT_NODATA_COST for workaround
        raster_clip = np.squeeze(raster_clip, axis=0)
        remove_nan_from_array(raster_clip)

        # generate the cost raster to source point
        mcp_source = MCP_Geometric(raster_clip, sampling=cell_size)
        source_cost_acc = mcp_source.find_costs(source)[0]
        del mcp_source

        # # # generate the cost raster to destination point
        mcp_dest = MCP_Geometric(raster_clip, sampling=cell_size)
        dest_cost_acc = mcp_dest.find_costs(destination)[0]

        # Generate corridor
        corridor = source_cost_acc + dest_cost_acc
        corridor = np.ma.masked_invalid(corridor)

        # Calculate minimum value of corridor raster
        if not np.ma.min(corridor) is None:
            corr_min = float(np.ma.min(corridor))
        else:
            corr_min = 0.5

        # normalize corridor raster by deducting corr_min
        corridor_norm = corridor - corr_min
        corridor_thresh_cl = np.ma.where(corridor_norm >= corridor_threshold, 1.0, 0.0)

    except Exception as e:
        print(e)
        print('corridor_raster: Exception occurred.')
        return None

    # export intermediate raster for debugging
    if BT_DEBUGGING:
        suffix = str(uuid.uuid4())[:8]
        path_temp = Path(r'C:\BERATools\Surmont_New_AOI\test_selected_lines\temp_files')
        if path_temp.exists():
            path_cost = path_temp.joinpath(suffix + '_cost.tif')
            path_corridor = path_temp.joinpath(suffix + '_corridor.tif')
            path_corridor_norm = path_temp.joinpath(suffix + '_corridor_norm.tif')
            path_corridor_cl = path_temp.joinpath(suffix + '_corridor_cl_poly.tif')
            out_cost = np.ma.masked_equal(raster_clip, np.inf)
            save_raster_to_file(out_cost, out_meta, path_cost)
            save_raster_to_file(corridor, out_meta, path_corridor)
            save_raster_to_file(corridor_norm, out_meta, path_corridor_norm)
            save_raster_to_file(corridor_thresh_cl, out_meta, path_corridor_cl)
        else:
            print('Debugging: raster folder not exists.')

    return corridor_thresh_cl


def find_least_cost_path_skimage(cost_clip, in_meta, seed_line):
    lc_path_new = []

    out_transform = in_meta['transform']
    transformer = rasterio.transform.AffineTransformer(out_transform)

    x1, y1 = list(seed_line.coords)[0][:2]
    x2, y2 = list(seed_line.coords)[-1][:2]
    row1, col1 = transformer.rowcol(x1, y1)
    row2, col2 = transformer.rowcol(x2, y2)

    try:
        path_new = route_through_array(cost_clip[0], [row1, col1], [row2, col2])
    except Exception as e:
        print(e)
        return None

    if path_new[0]:
        for row, col in path_new[0]:
            x, y = transformer.xy(row, col)
            lc_path_new.append((x, y))

    if len(lc_path_new) < 2:
        print('No least cost path detected, pass.')
        return None
    else:
        lc_path_new = LineString(lc_path_new)

    return lc_path_new


def execute_multiprocessing(in_func, in_data, processes, workers, verbose):
    out_result = []
    step = 0
    if PARALLEL_MODE == MODE_MULTIPROCESSING:
        pool = multiprocessing.Pool(processes)
        print("Multiprocessing started...")
        print("Using {} CPU cores".format(processes))
        total_steps = len(in_data)
        with Pool(processes) as pool:
            for result in pool.imap_unordered(in_func, in_data):
                if not result:
                    print('No results found.')
                    continue
                if len(result) == 0:
                    print('No results found.')
                    continue
                else:
                    out_result.append(result)

            step += 1
            if verbose:
                print(' "PROGRESS_LABEL Ceterline {} of {}" '.format(step, total_steps), flush=True)
                print(' %{} '.format(step / total_steps * 100), flush=True)

        pool.close()
        pool.join()
    elif PARALLEL_MODE == MODE_DASK:
        dask_client = Client(threads_per_worker=2, n_workers=10)
        print(dask_client)
        try:
            print('start processing')
            result = dask_client.map(in_func, in_data)
            seq = as_completed(result)

            for i in seq:
                out_result.append(i.result())
        except Exception as e:
            dask_client.close()
            print(e)

        dask_client.close()
    elif PARALLEL_MODE == MODE_RAY:
        ray.init(log_to_driver=False)
        process_single_line_ray = ray.remote(in_func)
        result_ids = [process_single_line_ray.remote(item) for item in in_data]

        while len(result_ids):
            done_id, result_ids = ray.wait(result_ids)
            result_item = ray.get(done_id[0])
            out_result.append(result_item)
            print('Done {}'.format(step))
            step += 1
        ray.shutdown()

    elif PARALLEL_MODE == MODE_SEQUENTIAL:
        i = 0
        for line in in_data:
            result_item = in_func(line)
            out_result.append(result_item)

    return out_result