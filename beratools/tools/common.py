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

