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
import argparse
import warnings
import numpy as np

from osgeo import ogr, gdal
import osgeo
from osgeo.osr import SpatialReference
import fiona
import shapely
from shapely.affinity import rotate
from shapely.ops import split, transform
from shapely.geometry import shape, mapping, Point, LineString, box

import pandas as pd
import geopandas as gpd
from pyproj import CRS, Transformer
from pyogrio import set_gdal_config_options

import rasterio
from rasterio import mask

from skimage.graph import MCP_Geometric, MCP_Connect

from scipy import ndimage
import xarray as xr
from xrspatial import focal, convolution

from numpy.lib.stride_tricks import as_strided,sliding_window_view

from pathlib import Path
from inspect import getsourcefile
if __name__ == "__main__":
    current_file = Path(getsourcefile(lambda: 0)).resolve()
    btool_dir = current_file.parents[2]
    sys.path.insert(0, btool_dir.as_posix())

from beratools.core.tool_base import *
from beratools.core.constants import *

# to suppress pandas UserWarning: Geometry column does not contain geometry when splitting lines
warnings.simplefilter(action='ignore', category=UserWarning)

# restore .shx for shapefile for using GDAL or pyogrio
gdal.SetConfigOption('SHAPE_RESTORE_SHX', 'YES')
set_gdal_config_options({'SHAPE_RESTORE_SHX': 'YES'})

# suppress all kinds of warnings
if not BT_DEBUGGING:
    gdal.SetConfigOption('CPL_LOG', 'NUL')  # gdal warning
    warnings.filterwarnings("ignore")  # suppress warnings
    warnings.simplefilter(action='ignore', category=UserWarning)  # suppress Pandas UserWarning


def clip_raster(in_raster_file, clip_geom, buffer=0.0, out_raster_file=None, default_nodata=BT_NODATA):
    out_meta = None
    with (rasterio.open(in_raster_file)) as raster_file:
        out_meta = raster_file.meta
        ras_nodata=out_meta['nodata']
        if ras_nodata is None:
            ras_nodata = default_nodata

        clip_geo_buffer = [clip_geom.buffer(buffer)]
        out_image: np.ndarray
        out_image, out_transform = mask.mask(raster_file, clip_geo_buffer,
                                             crop=True, nodata=ras_nodata, filled=True)
        if np.isnan(ras_nodata):
            out_image[np.isnan(out_image)]=default_nodata

        elif np.isinf(ras_nodata):
            out_image[np.isinf(out_image)] = default_nodata
        else:
            out_image[out_image==ras_nodata] = default_nodata

        out_image = np.ma.masked_where(out_image==default_nodata,out_image)
        out_image.fill_value=default_nodata
        ras_nodata = default_nodata

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
    out_line = in_line.clip(clip_geom.buffer(buffer * BT_BUFFER_RATIO))

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
    inter_img = 'image_overview.tif'

    #  get raster datasource
    src_ds = gdal.Open(in_raster)
    width, height = src_ds.RasterXSize, src_ds.RasterYSize
    src_crs = src_ds.GetSpatialRef().ExportToWkt()

    geom = None
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

            shapes = gdal.Footprint(None, inter_img, dstSRS=src_crs, format='GeoJSON')
            target_feat = shapes['features'][0]
            geom = shape(target_feat['geometry'])

        # coords = None
        # with rasterio.open(inter_img) as src:
        #     if np.isnan(src.nodata):
        #         geom = box(*src.bounds)
        #         coords_geo = list(geom.exterior.coords)
        #     else:
        #         msk = src.read_masks(1)
        #         shapes = features.shapes(msk, mask=msk)
        #         shapes = list(shapes)
        #         coords = shapes[0][0]['coordinates'][0]
        #
        #         for pt in coords:
        #             pt = rasterio.transform.xy(src.transform, pt[1], pt[0])
        #             coords_geo.append(pt)
        #
        #         coords_geo.pop(-1)

    if latlon:
        out_crs = CRS('EPSG:4326')
        transformer = Transformer.from_crs(CRS(src_crs), out_crs)

        geom = transform(transformer.transform, geom)
        # coords_geo = list(transformer.itransform(coords_geo))
        # coords_geo = [list(pt) for pt in coords_geo]

    return geom


def remove_nan_from_array(matrix):
    with np.nditer(matrix, op_flags=['readwrite']) as it:
        for x in it:
            if np.isnan(x[...]):
                x[...] = BT_NODATA_COST


def replace_Nodata2NaN(matrix, nodata):
    with np.nditer(matrix, op_flags=['readwrite']) as it:
        for x in it:
            if (x[...] == nodata):
                x[...] = np.NaN


def replace_Nodata2Inf(matrix, nodata):
    with np.nditer(matrix, op_flags=['readwrite']) as it:
        for x in it:
            if (x[...] == nodata):
                x[...] = np.Inf


# Split LineString to segments at vertices
def segments(line_coords):
    if len(line_coords) < 2:
        return None
    elif len(line_coords) == 2:
        return [fiona.Geometry.from_dict({'type': 'LineString', 'coordinates': line_coords})]
    else:
        seg_list = zip(line_coords[:-1], line_coords[1:])
        line_list = [{'type': 'LineString', 'coordinates': coords} for coords in seg_list]
        return [fiona.Geometry.from_dict(line) for line in line_list]


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


def save_features_to_file(out_file, crs, geoms, properties=None, schema=None,
                          driver='ESRI Shapefile', layer=None):
    """

    Parameters
    ----------
    out_file :
    crs :
    geoms : shapely geometry objects
    schema :
    properties :
    driver:
    layer:

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

    # driver = 'ESRI Shapefile'
    print('Writing to file {}'.format(out_file), flush=True)

    try:
        out_line_file = fiona.open(out_file, 'w', driver, schema, crs, layer=layer)
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
    osr_crs = SpatialReference()
    from pyproj.enums import WktVersion
    vec_crs = None
    # open input vector data as Geodataframe
    gpd_vector=gpd.GeoDataFrame.from_file(in_vector)
    try:
        if gpd_vector.crs != None:
            vec_crs=gpd_vector.crs
            if osgeo.version_info.major < 3:
                osr_crs.ImportFromWkt(vec_crs.to_wkt(WktVersion.WKT1_GDAL))
            else:
                osr_crs.ImportFromEPSG(vec_crs.to_epsg())
            return osr_crs
        else:
            print('No Coordinate Reference System (CRS) find in the input feature, please check!')
            exit()
    except Exception as e:
        print(e)
        exit()

    # with ogr.Open(in_vector) as vector_file:
    #     if vector_file:
    #         vec_crs = vector_file.GetLayer().GetSpatialRef()

    # return vec_crs

def df_crs(in_df):
    vec_crs = None
    osr_crs = SpatialReference()
    from pyproj.enums import WktVersion
    try:
        if in_df.crs != None:
            vec_crs=in_df.crs
            if osgeo.version_info.major < 3:
                osr_crs.ImportFromWkt(vec_crs.to_wkt(WktVersion.WKT1_GDAL))
            else:
                osr_crs.ImportFromEPSG(vec_crs.to_epsg())
            return osr_crs
        else:
            print('No Coordinate Reference System (CRS) find in the input feature, please check!')
            exit()
    except Exception as e:
        print(e)
        exit()

def raster_crs(in_raster):
    ras_crs = None
    osr_crs = SpatialReference()
    with rasterio.open(in_raster) as raster_file:
    # with gdal.Open(in_raster) as raster_file:
        from pyproj.enums import WktVersion
        try:
            if raster_file.crs != None:
                vec_crs = raster_file.crs
                if osgeo.version_info.major < 3:
                    osr_crs.ImportFromWkt(vec_crs.to_wkt(WktVersion.WKT1_GDAL))
                else:
                    osr_crs.ImportFromEPSG(vec_crs.to_epsg())
                return osr_crs
            else:
                print('No Coordinate Reference System (CRS) find in the input feature, please check!')
                exit()
        except Exception as e:
            print(e)
            exit()


def compare_crs(crs_org, crs_dst):
    if crs_org and crs_dst:
        if crs_org.IsSameGeogCS(crs_dst):
            print('Check: Input file Spatial Reference are the same, continue.')
            return True
        else:
            crs_org_norm = CRS(crs_org.ExportToWkt())
            crs_dst_norm = CRS(crs_dst.ExportToWkt())
            if crs_org_norm.is_compound:
                crs_org_proj = crs_org_norm.sub_crs_list[0].coordinate_operation.name
            elif crs_org_norm.name == 'unnamed':
                return False
            else:
                crs_org_proj = crs_org_norm.coordinate_operation.name

            if crs_dst_norm.is_compound:
                crs_dst_proj = crs_dst_norm.sub_crs_list[0].coordinate_operation.name
            elif crs_org_norm.name == 'unnamed':
                return False
            else:
                crs_dst_proj = crs_dst_norm.coordinate_operation.name

            if crs_org_proj == crs_dst_proj:
                if crs_org_norm.name==crs_dst_norm.name:
                    print('Checked: Input files Spatial Reference are the same, continue.')
                    return True
                else:
                    print('Checked: Data are on the same projected Zone but using different Spatial Reference. \n Consider to reproject all data onto same spatial reference system.\n Process Stop.')
                    exit()
            else:
                return False

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
            elif line_geom.intersection(in_fp_polygon.loc[i].geometry).length / line_geom.length < 0.30:
                drop_list.append(i)  # if less the 1/5 of line is inside of polygon, ignore

        # drop all polygons not used
        in_fp_polygon = in_fp_polygon.drop(index=drop_list)

        if not in_fp_polygon.empty:
            identity = in_fp_polygon.overlay(in_cl_buffer, how='intersection')
    except Exception as e:
        print(e)

    return line, identity


def line_split2(in_ln_shp, seg_length):
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
    df = odf.assign(geometry=odf.apply(lambda x: cut_line(x.geometry, seg_length), axis=1))
    # df = odf.assign(geometry=odf.apply(lambda x: cut_line(x.geometry, x.geometry.length), axis=1))
    df = df.explode()

    df['OLnSEG'] = df.groupby('OLnFID').cumcount()
    gdf = gpd.GeoDataFrame(df, geometry=df.geometry, crs=crs)
    gdf = gdf.sort_values(by=['OLnFID', 'OLnSEG'])
    gdf = gdf.reset_index(drop=True)

    if "shape_leng" in gdf.columns.array:
        gdf["shape_leng"] = gdf.geometry.length
    elif "LENGTH" in gdf.columns.array:
        gdf["LENGTH"] = gdf.geometry.length
    else:
        gdf["shape_leng"] = gdf.geometry.length
    return gdf


def split_line_nPart(line, seg_length):
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

            if abs(pd - distance) < BT_EPSILON:
                lines.append(LineString(coords[:i + 1]))
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
        if len(raster_clip.shape) > 2:
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
        if np.ma.min(corridor) is not None:
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

    return corridor_thresh_cl


def LCP_skimage_mcp_connect(cost_clip, in_meta, seed_line):
    lc_path_new = []
    if len(cost_clip.shape) > 2:
        cost_clip = np.squeeze(cost_clip, axis=0)

    out_transform = in_meta['transform']
    transformer = rasterio.transform.AffineTransformer(out_transform)

    x1, y1 = list(seed_line.coords)[0][:2]
    x2, y2 = list(seed_line.coords)[-1][:2]
    source = [transformer.rowcol(x1, y1)]
    destination = [transformer.rowcol(x2, y2)]

    try:

        init_obj1 = MCP_Connect(cost_clip)
        results = init_obj1.find_costs(source, destination)
        # init_obj2 = MCP_Geometric(cost_clip)
        path = []
        for end in destination:
            path.append(init_obj1.traceback(end))
        for row, col in path[0]:
            x, y = transformer.xy(row, col)
            lc_path_new.append((x, y))
    except Exception as e:
        print(e)
        return None

    if len(lc_path_new) < 2:
        print('No least cost path detected, pass.')
        return None
    else:
        lc_path_new = LineString(lc_path_new)

    return lc_path_new


def chk_df_multipart(df, chk_shp_in_string):
    try:
        found = False
        if str.upper(chk_shp_in_string) in [x.upper() for x in df.geom_type.values]:
            found = True
            df = df.explode()
            if type(df) is gpd.geodataframe.GeoDataFrame:
                df['OLnSEG'] = df.groupby('OLnFID').cumcount()
                df = df.sort_values(by=['OLnFID', 'OLnSEG'])
                df = df.reset_index(drop=True)
        else:
            found = False
        return df, found
    except Exception as e:
        print(e)
        return df, True


def dyn_fs_raster_stdmean(canopy_ndarray, kernel, nodata):
    # This function uses xrspatial which can handle large data but slow
    mask=canopy_ndarray.mask
    in_ndarray=np.ma.where(mask==True,np.NaN,canopy_ndarray)
    result_ndarray = focal.focal_stats(xr.DataArray(in_ndarray.data), kernel, stats_funcs=['std', 'mean'])

    # Assign std and mean ndarray (return array contain NaN value)
    reshape_std_ndarray = result_ndarray[0].data
    reshape_mean_ndarray = result_ndarray[1].data

    return reshape_std_ndarray, reshape_mean_ndarray


def dyn_smooth_cost(canopy_ndarray, max_line_dist, sampling):
    mask = canopy_ndarray.mask
    in_ndarray = np.ma.where(mask == True, np.NaN, canopy_ndarray)
    # scipy way to do Euclidean distance transform
    euc_dist_array = ndimage.distance_transform_edt(np.logical_not(np.isnan(in_ndarray.data)), sampling=sampling)
    euc_dist_array[mask==True]=np.NaN
    smooth1 = float(max_line_dist) - euc_dist_array
    smooth1[smooth1 <= 0.0] = 0.0
    smooth_cost_array = smooth1 / float(max_line_dist)

    return smooth_cost_array


def dyn_np_cost_raster(canopy_ndarray, cc_mean, cc_std, cc_smooth, avoidance, cost_raster_exponent):
    aM1a = (cc_mean - cc_std)
    aM1b = (cc_mean + cc_std)
    aM1 = np.divide(aM1a, aM1b, where=aM1b != 0, out=np.zeros(aM1a.shape, dtype=float))
    aM = (1 + aM1) / 2
    aaM = (cc_mean + cc_std)
    bM = np.where(aaM <= 0, 0, aM)
    cM = bM * (1 - avoidance) + (cc_smooth * avoidance)
    dM = np.where(canopy_ndarray.data == 1, 1, cM)
    eM = np.exp(dM)
    result = np.power(eM, float(cost_raster_exponent))

    return result


def dyn_np_cc_map(in_chm, canopy_ht_threshold, nodata):

    canopy_ndarray = np.ma.where(in_chm >= canopy_ht_threshold, 1., 0.).astype(float)
    canopy_ndarray.fill_value=nodata
    # canopy_ndarray = np.ma.filled(canopy_ndarray, nodata)
    # canopy_ndarray[canopy_ndarray==nodata]=np.NaN   # TODO check the code, extra step?

    return canopy_ndarray


def cost_raster(in_raster, meta):
    if len(in_raster.shape) > 2:
        in_raster = np.squeeze(in_raster, axis=0)

    # raster_clip, out_meta = clip_raster(self.in_raster, seed_line, self.line_radius)
    # in_raster = np.squeeze(in_raster, axis=0)
    cell_x, cell_y = meta['transform'][0], -meta['transform'][4]

    kernel = convolution.circle_kernel(cell_x, cell_y, 2.5)
    dyn_canopy_ndarray = dyn_np_cc_map(in_raster, FP_CORRIDOR_THRESHOLD, BT_NODATA)
    cc_std, cc_mean = dyn_fs_raster_stdmean(dyn_canopy_ndarray, kernel, BT_NODATA)
    cc_smooth = dyn_smooth_cost(dyn_canopy_ndarray, 2.5, [cell_x, cell_y])

    # TODO avoidance, re-use this code
    avoidance = max(min(float(0.4), 1), 0)
    cost_clip = dyn_np_cost_raster(dyn_canopy_ndarray, cc_mean, cc_std,
                                   cc_smooth, 0.4, 1.5)

    # TODO use nan or BT_DATA?
    cost_clip[in_raster == BT_NODATA] = np.nan
    dyn_canopy_ndarray[in_raster == BT_NODATA] = np.nan

    return cost_clip, dyn_canopy_ndarray


def cost_raster_2nd_version(
    in_raster,
    meta,
    tree_radius,
    canopy_ht_threshold,
    max_line_dist,
    canopy_avoid,
    cost_raster_exponent,
):
    """
    This is the general version of cost_raster
    To be merged later: variables and consistent nodata solution
    """
    if len(in_raster.shape) > 2:
        in_raster = np.squeeze(in_raster, axis=0)

    # raster_clip, out_meta = clip_raster(self.in_raster, seed_line, self.line_radius)
    # in_raster = np.squeeze(in_raster, axis=0)
    cell_x, cell_y = meta["transform"][0], -meta["transform"][4]

    kernel = convolution.circle_kernel(cell_x, cell_y, tree_radius)
    dyn_canopy_ndarray = dyn_np_cc_map(in_raster, canopy_ht_threshold, BT_NODATA)
    cc_std, cc_mean = dyn_fs_raster_stdmean(dyn_canopy_ndarray, kernel, BT_NODATA)
    cc_smooth = dyn_smooth_cost(dyn_canopy_ndarray, max_line_dist, [cell_x, cell_y])

    # TODO avoidance, re-use this code
    avoidance = max(min(float(canopy_avoid), 1), 0)
    cost_clip = dyn_np_cost_raster(
        dyn_canopy_ndarray, cc_mean, cc_std, cc_smooth, avoidance, cost_raster_exponent
    )

    # TODO use nan or BT_DATA?
    cost_clip[in_raster == BT_NODATA] = np.nan
    dyn_canopy_ndarray[in_raster == BT_NODATA] = np.nan

    return cost_clip, dyn_canopy_ndarray


def morph_raster(corridor_thresh, canopy_raster, exp_shk_cell, cell_size_x):
    # Process: Stamp CC and Max Line Width
    ras_sum = corridor_thresh + canopy_raster
    raster_class = np.ma.where(ras_sum == 0, 1, 0).data

    if exp_shk_cell > 0 and cell_size_x < 1:
        # Process: Expand
        # FLM original Expand equivalent
        cell_size = int(exp_shk_cell * 2 + 1)
        expanded = ndimage.grey_dilation(raster_class, size=(cell_size, cell_size))

        # Process: Shrink
        # FLM original Shrink equivalent
        file_shrink = ndimage.grey_erosion(expanded, size=(cell_size, cell_size))

    else:
        if BT_DEBUGGING:
            print("No Expand And Shrink cell performed.")
        file_shrink = raster_class

    # Process: Boundary Clean
    clean_raster = ndimage.gaussian_filter(file_shrink, sigma=0, mode="nearest")

    return clean_raster


def generate_line_args_NoClipraster(line_seg, work_in_buffer, in_chm_obj, in_chm, tree_radius, max_line_dist,
                                    canopy_avoidance, exponent, canopy_thresh_percentage):
    line_argsC = []

    for record in range(0, len(work_in_buffer)):
        try:
            line_bufferC = work_in_buffer.loc[record, 'geometry']

            nodata = BT_NODATA
            line_argsC.append([in_chm, float(work_in_buffer.loc[record, 'DynCanTh']), float(tree_radius),
                               float(max_line_dist), float(canopy_avoidance), float(exponent), in_chm_obj.res, nodata,
                               line_seg.iloc[[record]], in_chm_obj.meta.copy(), record, 10, 'Center',
                               canopy_thresh_percentage, line_bufferC])
        except Exception as e:

            print(e)

        step = record + 1
        total = len(work_in_buffer)

        print(f' "PROGRESS_LABEL Preparing lines {step} of {total}" ', flush=True)
        print(f' %{step / total * 100} ', flush=True)

    return line_argsC


def generate_line_args_DFP_NoClip(line_seg, work_in_bufferL, work_in_bufferC, in_chm_obj,
                                  in_chm, tree_radius, max_line_dist, canopy_avoidance,
                                  exponent, work_in_bufferR, canopy_thresh_percentage):
    line_argsL = []
    line_argsR = []
    line_argsC = []
    line_id = 0
    for record in range(0, len(work_in_bufferL)):
        line_bufferL = work_in_bufferL.loc[record, 'geometry']
        line_bufferC = work_in_bufferC.loc[record, 'geometry']
        LCut = work_in_bufferL.loc[record, 'LDist_Cut']

        nodata = BT_NODATA
        line_argsL.append([in_chm, float(work_in_bufferL.loc[record, 'DynCanTh']), float(tree_radius),
                           float(max_line_dist), float(canopy_avoidance), float(exponent), in_chm_obj.res, nodata,
                           line_seg.iloc[[record]], in_chm_obj.meta.copy(), line_id, LCut, 'Left',
                           canopy_thresh_percentage, line_bufferL])

        line_argsC.append([in_chm, float(work_in_bufferC.loc[record, 'DynCanTh']), float(tree_radius),
                           float(max_line_dist), float(canopy_avoidance), float(exponent), in_chm_obj.res, nodata,
                           line_seg.iloc[[record]], in_chm_obj.meta.copy(), line_id, 10, 'Center',
                           canopy_thresh_percentage, line_bufferC])

        line_id += 1

    line_id = 0
    for record in range(0, len(work_in_bufferR)):
        line_bufferR = work_in_bufferR.loc[record, 'geometry']
        RCut = work_in_bufferR.loc[record, 'RDist_Cut']
        # clipped_rasterR, out_transformR = rasterio.mask.mask(in_chm, [line_bufferR], crop=True,
        #                                                      nodata=BT_NODATA, filled=True)
        # clipped_rasterR = np.squeeze(clipped_rasterR, axis=0)
        #
        # # make rasterio meta for saving raster later
        # out_metaR = in_chm.meta.copy()
        # out_metaR.update({"driver": "GTiff",
        #                  "height": clipped_rasterR.shape[0],
        #                  "width": clipped_rasterR.shape[1],
        #                  "nodata": BT_NODATA,
        #                  "transform": out_transformR})
        line_bufferC = work_in_bufferC.loc[record, 'geometry']
        # clipped_rasterC, out_transformC = rasterio.mask.mask(in_chm, [line_bufferC], crop=True,
        #                                                      nodata=BT_NODATA, filled=True)
        #
        # clipped_rasterC = np.squeeze(clipped_rasterC, axis=0)
        # out_metaC = in_chm.meta.copy()
        # out_metaC.update({"driver": "GTiff",
        #                   "height": clipped_rasterC.shape[0],
        #                   "width": clipped_rasterC.shape[1],
        #                   "nodata": BT_NODATA,
        #                   "transform": out_transformC})

        nodata = BT_NODATA
        # TODO deal with inherited nodata and BT_NODATA_COST
        # TODO convert nodata to BT_NODATA_COST
        line_argsR.append([in_chm, float(work_in_bufferR.loc[record, 'DynCanTh']), float(tree_radius),
                           float(max_line_dist), float(canopy_avoidance), float(exponent), in_chm_obj.res, nodata,
                           line_seg.iloc[[record]], in_chm_obj.meta.copy(), line_id, RCut, 'Right',
                           canopy_thresh_percentage, line_bufferR])

        step = line_id + 1 + len(work_in_bufferL)
        total = len(work_in_bufferL) + len(work_in_bufferR)
        print(f' "PROGRESS_LABEL Preparing... {step} of {total}" ', flush=True)
        print(f' %{step / total * 100} ', flush=True)

        line_id += 1

    return line_argsL, line_argsR, line_argsC


def chk_null_geometry(in_data):
    find = False
    if isinstance(in_data, gpd.GeoDataFrame):
        if len(in_data[(in_data.is_empty | in_data.isna())]) > 0:
            find = True

    return find

def read_data2gpd(in_data):
    print("Reading data.......")
    out_gpd_obj = gpd.GeoDataFrame.from_file(in_data)
    return out_gpd_obj
