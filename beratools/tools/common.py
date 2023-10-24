#!/usr/bin/env python3
""" This file is intended to be hosting common functions for BERA Tools.
"""

# This script is part of the BERA Tools geospatial library.
# Author: Richard Zeng
# Created: 12/04/2023
# License: MIT

# imports
import sys
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
from shapely.geometry import shape, mapping, Point, LineString
import geopandas as gpd

from osgeo import ogr, gdal, osr
from pyproj import CRS, Transformer


# constants
MODE_MULTIPROCESSING = 1
MODE_SEQUENTIAL = 2
MODE_RAY = 3

PARALLEL_MODE = MODE_SEQUENTIAL

USE_SCIPY_DISTANCE = True
USE_NUMPY_FOR_DIJKSTRA = True

BT_NODATA = -9999
BT_DEBUGGING = False
BT_MAXIMUM_CPU_CORES = 60  # multiprocessing has limit of 64, consider pathos
BT_BUFFER_RATIO = 0.0  # overlapping ratio of raster when clipping lines
BT_LABEL_MIN_WIDTH = 130
BT_SHOW_ADVANCED_OPTIONS = False
BT_EPSLON = sys.float_info.epsilon  # np.finfo(float).eps

BT_UID = 'BT_UID'

GROUPING_SEGMENT = True
LP_SEGMENT_LENGTH = 30

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

    with(rasterio.open(in_raster_file)) as raster_file:
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
                x[...] = -9999


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


def save_features_to_shapefile(out_file, crs, geoms, fields=None, properties=None):
    # remove all None items
    # TODO: check geom type consistency
    geoms = [item for item in geoms if item is not None]

    if len(geoms) < 1:
        return

    if fields is None:
        fields = []
    if properties is None:
        properties = []

    try:
        geom_type = mapping(geoms[0])['type']
    except Exception as e:
        print(e)

    props_tuple = zip(fields, properties, strict=True)  # if lengths are not the same, ValueError raises
    props_schema = [(item, type(value).__name__) for item, value in props_tuple]

    schema = {
        'geometry': geom_type,
        'properties': OrderedDict(props_schema)
    }

    driver = 'ESRI Shapefile'
    print('Writing to shapefile {}'.format(out_file))

    with fiona.open(out_file, 'w', driver, schema, crs) as out_line_file:
        feat_tuple = zip_longest(geoms, properties)

        try:
            for geom, prop in feat_tuple:
                prop_zip = {} if prop is None else OrderedDict(list(zip(fields, prop)))

                if geom:
                    feature = {
                        'geometry': mapping(geom),
                        'properties': prop_zip
                    }

                    out_line_file.write(feature)
        except Exception as e:
            print(e)


def vector_crs(vector_file):
    in_line_file = ogr.Open(vector_file)
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
            if crs_org_norm.name == crs_dst_norm.name:
                print('Same crs, continue.')
                return True

        print('Different GCS, please check.')

    return False
