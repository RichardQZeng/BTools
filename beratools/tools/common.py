#!/usr/bin/env python3
""" This file is intended to be hosting common functions for BERA Tools.
"""

# This script is part of the BERA Tools geospatial library.
# Author: Richard Zeng
# Created: 12/04/2023
# License: MIT

# imports
import rasterio
import rasterio.mask
import geopandas as gpd
import fiona

from osgeo import ogr, gdal, osr
from rasterio import features
import tempfile
from pathlib import Path
from pyproj import CRS, Transformer

# constants
USE_MULTI_PROCESSING = True
USE_SCIPY_DISTANCE = True
USE_PATHOS_MULTIPROCESSING = True

BT_NODATA = -9999
BT_DEBUGGING = False
BT_MAXIMUM_CPU_CORES = 60  # multiprocessing has limit of 64, consider pathos
BT_BUFFER_RATIO = 0.0  # overlapping ratio of raster when clipping lines
BT_LABEL_MIN_WIDTH = 130
BT_SHOW_ADVANCED_OPTIONS = False


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


def read_lines_from_shapefile(in_file):
    lines = []
    with fiona.open(in_file) as open_line_file:
        layer_crs = open_line_file.crs
        for line in open_line_file:
            lines.append(line['geometry'])

    return lines


def generate_raster_footprint(in_raster):
    inter_img = 'myimage.tif'

    #  get raster datasource
    src_ds = gdal.Open(in_raster)
    width, height = src_ds.RasterXSize, src_ds.RasterYSize

    # ensure there is nodata
    # gdal_translate ... -a_nodata 0 ... outimage.vrt
    # gdal_edit -a_nodata 255 somefile.tif

    # gdal_translate -outsize 1024 0 vendor_image.tif myimage.tif
    options = None
    tmp_folder = tempfile.TemporaryDirectory()
    print('Temporary folder: '.format(tmp_folder.name))

    if max(width, height) <= 1024:
        inter_img = in_raster
    else:
        if width >= height:
            options = gdal.TranslateOptions(width=1024, height=0)
        else:
            options = gdal.TranslateOptions(width=0, height=1024)

        inter_img = Path(tmp_folder.name).joinpath(inter_img).as_posix()
        gdal.Translate(inter_img, src_ds, options=options)

    coords = None
    with rasterio.open(inter_img) as src:
        msk = src.read_masks(1)
        shapes = features.shapes(msk, mask=msk)
        shapes = list(shapes)
        coords = shapes[0][0]['coordinates'][0]

        coords_geo = []
        for pt in coords:
            pt = rasterio.transform.xy(src.transform, pt[1], pt[0])
            coords_geo.append(pt)

    # clean temporary folder
    tmp_folder.cleanup()

    in_crs = CRS(src_ds.GetSpatialRef().ExportToWkt())
    out_crs = CRS('EPSG:4326')
    transformer = Transformer.from_crs(in_crs, out_crs)
    coords_geo.pop(-1)
    coords_geo = list(transformer.itransform(coords_geo))
    coords_geo = [list(pt) for pt in coords_geo]

    return coords_geo if len(coords_geo) > 0 else None


