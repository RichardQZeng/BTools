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
from itertools import zip_longest

import json
import shlex
import argparse
import numpy as np

import rasterio
import rasterio.mask
from rasterio import features

import fiona
from fiona import Geometry

import shapely
from shapely.ops import unary_union, snap, split
from shapely.geometry import (shape, mapping, Point, LineString,
                              MultiLineString, MultiPoint, Polygon, MultiPolygon)

import pandas as pd
import geopandas as gpd
from osgeo import ogr, gdal, osr
from pyproj import CRS, Transformer
from pyogrio import set_gdal_config_options

from label_centerlines import get_centerline

from multiprocessing.pool import Pool

# constants
MODE_MULTIPROCESSING = 1
MODE_SEQUENTIAL = 2
MODE_DASK = 3

PARALLEL_MODE = MODE_MULTIPROCESSING

USE_SCIPY_DISTANCE = True
USE_NUMPY_FOR_DIJKSTRA = True

NADDatum = ['NAD83 Canadian Spatial Reference System', 'North American Datum 1983']

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
CL_BUFFER_CLIP = 10
CL_BUFFER_CENTROID = 3
CL_SNAP_TOLERANCE = 10
CL_BUFFER_MULTIPOLYGON = 0.01  # buffer MultiPolygon by 0.01 meter to convert to Polygon
CL_SEGMENTIZE_LENGTH = 1
CL_SIMPLIFY_LENGTH = 0.5
CL_SMOOTH_SIGMA = 0.5
CL_DELETE_HOLES = True
CL_SIMPLIFY_POLYGON = True

FP_CORRIDOR_THRESHOLD = 3.0
FP_SEGMENTIZE_LENGTH = 2

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


def clip_raster(clip_geom, buffer, in_raster_file, out_raster_file):
    ras_nodata = BT_NODATA

    with (rasterio.open(in_raster_file)) as raster_file:
        ras_nodata = raster_file.meta['nodata']
        clip_geo_buffer = [clip_geom.buffer(buffer)]
        out_image, out_transform = rasterio.mask.mask(raster_file, clip_geo_buffer, crop=True, nodata=ras_nodata)

    out_meta = raster_file.meta.copy()

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

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


def vector_crs(vector_file):
    in_line_file = ogr.Open(vector_file)

    # TODO: in_line_file is None
    vec_crs = in_line_file.GetLayer().GetSpatialRef()

    del in_line_file
    return vec_crs


def raster_crs(raster_file):
    cost_raster_file = gdal.Open(raster_file)
    ras_crs = cost_raster_file.GetSpatialRef()

    del cost_raster_file
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
            else:
                crs_org_proj=crs_org_norm.coordinate_operation.name

            if crs_dst_norm.is_compound:
                crs_dst_proj = crs_dst_norm.sub_crs_list[0].coordinate_operation.name
            else:
                crs_dst_proj = crs_dst_norm.coordinate_operation.name

            if crs_org_proj== crs_dst_proj :
                print('Checked: Input files Spatial Reference are the same, continue.')
                return True

        print('Different GCS, please check.')

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
    from shapely.ops import split,snap
    seg_line = shapely.segmentize(line, seg_length)
    distances=np.arange(seg_length,line.length,seg_length)

    if len(distances)>0:
        points = [shapely.line_interpolate_point(seg_line,distance) for distance in distances]

        # snap_points = snap(points, seg_line, 0.001)
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
    # seg_line = shapely.segmentize(line, distance)
    lines=cut(line, distance, lines)
    return lines


def cut(line, distance, lines):
    # Cuts a line in several segments at a distance from its starting point
    from shapely import ops
    if line.has_z:
        line=ops.transform(lambda x,y,z=None:(x,y),line)
    if shapely.is_empty(line) or shapely.is_missing(line):
        return None
    else:
        if math.fmod(line.length , distance)<(1):
            return [line]
        elif distance >= line.length:
            return [line]
        else:
            end_pt = None
            line=shapely.segmentize(line,distance)
            while line.length > distance:

                coords = list(line.coords)
                for i, p in enumerate(coords):
                    pd = line.project(Point(p))
                    # if abs(pd - line.length) < BT_EPSLON:
                    #     lines.append(line)
                    #     return lines

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



def find_centerline(poly, lc_path):
    """
    Parameters
    ----------
    poly
    lc_path

    Returns
    -------

    """
    poly = shapely.segmentize(poly, max_segment_length=CL_SEGMENTIZE_LENGTH)

    exterior_pts = []
    if type(poly) is MultiPolygon:
        poly = poly.buffer(CL_BUFFER_MULTIPOLYGON)
        if type(poly) is MultiPolygon:
            print('MultiPolygon encountered, skip.')
            return None

    exterior_pts = list(poly.exterior.coords)

    if CL_DELETE_HOLES:
        poly = Polygon(exterior_pts)
    if CL_SIMPLIFY_POLYGON:
        poly = poly.simplify(CL_SIMPLIFY_LENGTH)

    centerline = get_centerline(poly, segmentize_maxlen=1, max_points=3000,
                                simplification=0.05, smooth_sigma=CL_SMOOTH_SIGMA, max_paths=1)

    if type(centerline) is MultiLineString:
        if len(centerline.geoms) > 1:
            print(" Multiple centerline segments detected, no further processing.")
            return centerline
        elif len(centerline.geoms) == 1:
            centerline = centerline.geoms[0]
        else:
            return None

    cl_coords = list(centerline.coords)

    # trim centerline at two ends
    head_buffer = Point(cl_coords[0]).buffer(CL_BUFFER_CLIP)
    centerline = centerline.difference(head_buffer)

    end_buffer = Point(cl_coords[-1]).buffer(CL_BUFFER_CLIP)
    centerline = centerline.difference(end_buffer)

    # snap two end vertices to the least cost path ends
    lc_coords = list(lc_path.coords)

    # check if point is 2D or 3D
    lc_start_pt = None
    lc_end_pt = None

    # convert LC end points to 2D or 3D
    if len(cl_coords[0]) == 2:
        lc_start_pt = lc_coords[0][0:2]
        lc_end_pt = lc_coords[-1][0:2]
    elif len(cl_coords[0]) == 3:
        lc_start_pt = lc_coords[0][0:3]
        lc_end_pt = lc_coords[-1][0:3]

    # snap centerline to LC end points
    lc_end_pts = MultiPoint([lc_start_pt, lc_end_pt])
    centerline = snap(centerline, lc_end_pts, CL_SNAP_TOLERANCE)

    return centerline

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

    for poly, value in poly_generator:
        corridor_polygon.append(shape(poly))
    corridor_polygon = unary_union(corridor_polygon)

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
    row_and_path: list of row (polygon and props) and least cost path

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

