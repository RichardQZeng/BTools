# ---------------------------------------------------------------------------
#    Copyright (C) 2021  Applied Geospatial Research Group
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, version 3.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://gnu.org/licenses/gpl-3.0>.
#
# ---------------------------------------------------------------------------
#
# FLM_LineFootprint.py
# Script Author: Gustavo Lopes Queiroz
# Date: 2020-Jan-22
# Refactor to use for produce dynamic footprint from dynamic canopy and cost raster with open source libraries
# Prerequisite:  Line feature class must have the attribute Fields:"CorridorTh" "DynCanTh" "OLnFID"
# dynamic_line_footprint.py
# Maverick Fong
# Date: 2023-Dec
# This script is part of the BERA toolset
# Webpage: https://github.com/
#
# Purpose: Creates dynamic footprint polygons for each input line based on a least
# cost corridor method and individual line thresholds.
#
# ---------------------------------------------------------------------------

from multiprocessing.pool import Pool
import geopandas
import json
import math
import argparse
import time
import rasterio.features
import xarray as xr
import xrspatial.focal
from xrspatial import convolution
import warnings
import pandas
import numpy
from scipy import ndimage
# from numpy.lib.stride_tricks import as_strided
from rasterio import features, mask
from skimage.graph import MCP_Geometric  # ,MCP
import shapely
from shapely import LineString
from common import *
import sys
# to suppress panadas UserWarning: Geometry column does not contain geometry when splitting lines
warnings.simplefilter(action='ignore', category=UserWarning)

class OperationCancelledException(Exception):
    pass


def dyn_np_cc_map(in_array, canopy_ht_threshold, nodata):
    masked_array = numpy.ma.masked_where(in_array == nodata, in_array)
    canopy_ndarray = numpy.ma.where(masked_array >= canopy_ht_threshold, 1., 0.)
    canopy_ndarray = numpy.ma.where(in_array == nodata, -9999, canopy_ndarray).data
    return canopy_ndarray, masked_array


def dyn_fs_raster_stdmean(in_ndarray, kernel, masked_array, nodata):
    # This function uses xrspatial whcih can handle large data but slow
    # print("Calculating Canopy Closure's Focal Statistic-Stand Deviation Raster .......")
    ndarray = numpy.where(in_ndarray == nodata, numpy.nan, in_ndarray)
    result_ndarray = xrspatial.focal.focal_stats(xr.DataArray(ndarray), kernel, stats_funcs=['std', 'mean'])

    # Flattening the array
    flatten_std_result_ndarray = result_ndarray[0].data.reshape(-1)
    flatten_mean_result_ndarray = result_ndarray[1].data.reshape(-1)

    # Re-shaping the array
    reshape_Std_ndarray = flatten_std_result_ndarray.reshape(ndarray.shape[0], ndarray.shape[1])
    reshape_mean_ndarray = flatten_mean_result_ndarray.reshape(ndarray.shape[0], ndarray.shape[1])
    return reshape_Std_ndarray, reshape_mean_ndarray


def dyn_smooth_cost(in_raster, max_line_dist, cell_x, cell_y):
    # print('Generating Cost Raster .......')

    euc_dist_array = None
    # scipy way to do Euclidean distance transform
    euc_dist_array = ndimage.distance_transform_edt(numpy.logical_not(in_raster), sampling=[cell_x, cell_y])

    smooth1 = float(max_line_dist) - euc_dist_array
    # cond_smooth1 = numpy.where(smooth1 > 0, smooth1, 0.0)
    smooth1[smooth1 <= 0.0] = 0.0
    smooth_cost_array = smooth1 / float(max_line_dist)

    return smooth_cost_array

def dyn_np_cost_raster(canopy_ndarray, cc_mean, cc_std, cc_smooth, avoidance, cost_raster_exponent):
    aM1a = (cc_mean - cc_std)
    aM1b = (cc_mean + cc_std)
    aM1 = numpy.divide(aM1a, aM1b, where=aM1b != 0.,out=numpy.zeros(aM1a.shape, dtype=float))
    aM = (1. + aM1) / 2.
    aaM = (cc_mean + cc_std)
    bM = numpy.where(aaM <= 0., 0., aM)
    # aaM[aaM <= 0] = 0
    # numpy.place(aaM, aaM > 0, aM)
    # bM = aaM
    cM = bM * (1. - avoidance) + (cc_smooth * avoidance)
    dM = numpy.where(canopy_ndarray == 1., 1., cM)
    # numpy.place(canopy_ndarray, canopy_ndarray != 1, cM)
    # dM = canopy_ndarray
    eM = numpy.exp(dM)
    result = numpy.power(eM, float(cost_raster_exponent))

    return result

def dyn_canopy_cost_raster(args):
    in_chm = args[0]
    canopy_ht_threshold = args[1]
    Tree_radius = args[2]
    max_line_dist = args[3]
    canopy_avoid = args[4]
    exponent = args[5]
    res = args[6]
    nodata = args[7]
    line_df = args[8]
    out_transform = args[9]

    canopy_ht_threshold = float(canopy_ht_threshold)


    tree_radius = float(Tree_radius)  # get the round up integer number for tree search radius
    max_line_dist = float(max_line_dist)
    canopy_avoid = float(canopy_avoid)
    cost_raster_exponent = float(exponent)

    (cell_x, cell_y) = res

    # print('Loading CHM............')
    band1_ndarray = in_chm

    # print('Preparing Kernel window............')
    kernel = convolution.circle_kernel(cell_x, cell_y, math.ceil(tree_radius))

    # Generate Canopy Raster and return the Canopy array
    dyn_canopy_ndarray, masked_array = dyn_np_cc_map(band1_ndarray, canopy_ht_threshold, nodata)

    # Calculating focal statistic from canopy raster
    cc_std, cc_mean = dyn_fs_raster_stdmean(dyn_canopy_ndarray, kernel, masked_array, nodata)

    # Smoothing raster
    cc_smooth = dyn_smooth_cost(dyn_canopy_ndarray, max_line_dist, cell_x, cell_y)
    avoidance = max(min(float(canopy_avoid), 1), 0)
    dyn_cost_ndarray = dyn_np_cost_raster(dyn_canopy_ndarray, cc_mean, cc_std, cc_smooth, avoidance,
                                          cost_raster_exponent)
    dyn_cost_ndarray[numpy.isnan(dyn_cost_ndarray)]=nodata
    return line_df, dyn_canopy_ndarray, dyn_cost_ndarray, out_transform

def split_line_fc(line):
    return list(map(LineString, zip(line.coords[:-1], line.coords[1:])))

def split_line_nPart(line):
    # Work out n parts for each line (divided by 30m)
    n = math.ceil(line.length / 30)
    if n > 1:
        # divided line into n-1 equal parts;
        distances = numpy.linspace(0, line.length, n)
        points = [line.interpolate(distance) for distance in distances]
        line = shapely.LineString(points)
        mline = split_line_fc(line)
    else:
        mline = line
    return mline

def split_into_segments(df):
    odf = df
    crs = odf.crs
    if not 'OLnSEG' in odf.columns.array:
        df['OLnSEG'] = numpy.nan

    df = odf.assign(geometry=odf.apply(lambda x: split_line_fc(x.geometry), axis=1))
    df = df.explode()

    df['OLnSEG'] = df.groupby('OLnFID').cumcount()
    gdf = geopandas.GeoDataFrame(df, geometry=df.geometry, crs=crs)
    gdf = gdf.sort_values(by=['OLnFID', 'OLnSEG'])
    gdf = gdf.reset_index(drop=True)
    return gdf


def split_into_Equal_Nth_segments(df):
    odf = df
    crs = odf.crs
    if not 'OLnSEG' in odf.columns.array:
        df['OLnSEG'] = numpy.nan
    df = odf.assign(geometry=odf.apply(lambda x: split_line_nPart(x.geometry), axis=1))
    df = df.explode(index_parts=True)

    df['OLnSEG'] = df.groupby('OLnFID').cumcount()
    gdf = geopandas.GeoDataFrame(df, geometry=df.geometry, crs=crs)
    gdf = gdf.sort_values(by=['OLnFID', 'OLnSEG'])
    gdf = gdf.reset_index(drop=True)
    return gdf


def dynamic_line_footprint(callback, in_line, in_chm, max_ln_width, exp_shk_cell, proc_segments, out_footprint,
       tree_radius,max_line_dist, canopy_avoidance, exponent, full_step, processes, verbose):
    line_seg = geopandas.GeoDataFrame.from_file(in_line)
    # Check the Dynamic Corridor threshold column in data. If it is not, new column will be created

    if not 'DynCanTh' in line_seg.columns.array:
        print("Cannot find {} column in input line data.\n "
              "Please run Dynamic Canopy Threshold first".format('DynCanTh'))
        exit()
    # Check the OLnFID column in data. If it is not, column will be created
    if not 'OLnFID' in line_seg.columns.array:
        print(
            "Cannot find {} column in input line data.\n '{}' column will be created".format('OLnFID', 'OLnFID'))
        line_seg['OLnFID'] = line_seg.index

    if not 'CorridorTh' in line_seg.columns.array:
        print(
            "Cannot find {} column in input line data.\n '{}' column will be created".format('CorridorTh',
                                                                                             'CorridorTh'))
        line_seg['CorridorTh'] = 3.0
    else:
        use_CorridorThCol = True

    if not 'OLnSEG' in line_seg.columns.array:
        # print(
        #     "Cannot find {} column in input line data.\n '{}' column will be created base on input Features ID".format('OLnSEG', 'OLnSEG'))
        line_seg['OLnSEG'] = line_seg['OLnFID']

    print('%{}'.format(10))
    # check coordinate systems between line and raster features
    with rasterio.open(in_chm) as raster:
        if line_seg.crs.to_epsg() != raster.crs.to_epsg():
            print("Line and raster spatial references are not same, please check.")
            exit()

        else:
            if proc_segments.lower() == 'true':
                proc_segments = True
                print("Spliting lines into segments...")
                line_seg = split_into_segments(line_seg)
                print("Spliting lines into segments...Done")
            else:
                proc_segments = False
                line_seg = split_into_Equal_Nth_segments(line_seg)
            print('%{}'.format(20))
            worklnbuffer = geopandas.GeoDataFrame.copy((line_seg))
            worklnbuffer['geometry'] = shapely.buffer(worklnbuffer['geometry'], distance=float(max_ln_width),
                                                      cap_style=1)
            line_args = []
            # nodata = raster.nodata
            # in_chm = raster.read(1)
            # results = []
            # index=1

            print("Prepare CHMs for Dynamic cost raster......")
            for record in range(0, len(worklnbuffer)):
                line_buffer = worklnbuffer.loc[record, 'geometry']
                clipped_raster, out_transform = rasterio.mask.mask(raster, [line_buffer], crop=True, nodata=-9999,
                                                                   filled=True)
                clipped_raster = numpy.squeeze(clipped_raster, axis=0)
                nodata = -9999
                line_args.append([clipped_raster, float(worklnbuffer.loc[record, 'DynCanTh']),
                                  float(tree_radius), float(max_line_dist), float(canopy_avoidance),
                                  float(exponent), raster.res, nodata, line_seg.iloc[[record]], out_transform])

            print("Prepare CHMs for Dynamic cost raster......Done")
            print('%{}'.format(30))

            print('Generate Dynamic cost raster.....')
            list_dict_segment_all = multiprocessing_dynamic_CC(line_args, processes)
            print('Generate Dynamic cost raster.....Done')
            print('%{}'.format(50))
        for row in range(0, len(list_dict_segment_all)):
            l=list(list_dict_segment_all[row])
            l.append(float(max_line_dist))
            l.append(use_CorridorThCol)
            list_dict_segment_all[row]=tuple(l)

        # pass center lines for footprint
        print("Generate Dynamic footprint.....")
        footprint_list = []
        # USE_MULTI_PROCESSING = False
        if USE_MULTI_PROCESSING:
            footprint_list = multiprocessing_Dyn_FP(list_dict_segment_all, processes)
        else:
            # Non multi-processing, for debug only
            print("There are {} result to process.".format(len(list_dict_segment_all)))
            index = 0
            for row in list_dict_segment_all:
                footprint_list.append(dyn_process_single_line(row))
                print("FP for line {} is done".format(index))
                index = index + 1
    print('%{}'.format(80))
    print("Generate Dynamic footprint.....Done")
    print('Generating shapefile...........')
    results = geopandas.GeoDataFrame(pandas.concat(footprint_list))
    results = results.sort_values(by=['OLnFID', 'OLnSEG'])
    results = results.reset_index(drop=True)

    # dissolved polygon group by column 'OLnFID'
    dissolved_results = results.dissolve(by='OLnFID', as_index=False)
    dissolved_results = dissolved_results.drop(columns=['OLnSEG'])
    print("Saving output.....")
    dissolved_results.to_file(out_footprint)
    print('%{}'.format(100))


def dyn_process_single_line(segment):
    # this function takes single line to work the line footprint
    # (regardless it process the whole line or individual segment)

    df = segment[0]
    in_canopy_r = segment[1]
    in_cost_r = segment[2]
    if numpy.isnan(in_canopy_r).all():
        print("Canopy raster empty")
    elif numpy.isnan(in_cost_r).all():
        print("Cost raster empty")

    exp_shk_cell = segment[4]
    use_CorridorCol=segment[5]

    if use_CorridorCol:
        corridor_th_value = df.CorridorTh.iloc[0]
        try:
            corridor_th_value=float(corridor_th_value)
            if corridor_th_value<0:
                corridor_th_value = 3.0
        except ValueError:
            corridor_th_value=3.0
    else:
        corridor_th_value= 3.0
    # max_ln_dist=dict_segment.max_ln_dist.iloc[0]
    shapefile_proj = df.crs

    in_transform = segment[3]

    # segment line feature ID
    FID = df['OLnSEG']
    # original line ID for segment line
    OID = df['OLnFID']

    segment_list = []

    feat = df.loc[df.index[0], 'geometry']
    for coord in feat.coords:
        segment_list.append(coord)

    # Find origin and destination coordinates
    x1, y1 = segment_list[0][0], segment_list[0][1]
    x2, y2 = segment_list[-1][0], segment_list[-1][1]

    # Create Point "origin"
    origin_point = shapely.Point([x1, y1])
    origin = [shapes for shapes in geopandas.GeoDataFrame(geometry=[origin_point], crs=shapefile_proj).geometry]

    # Create Point "destination"
    destination_point = shapely.Point([x2, y2])
    destination = [shapes for shapes in
                   geopandas.GeoDataFrame(geometry=[destination_point], crs=shapefile_proj).geometry]

    cell_size_x = in_transform[0]
    cell_size_y = -in_transform[4]

    # Work out the corridor from both end of the centerline
    try:

        # numpy.place(in_cost_r, numpy.isnan(in_cost_r), -9999)
        # numpy.place(in_canopy_r, in_canopy_r == -9999, 1)

        # Rasterize source point
        rasterized_source = features.rasterize(origin, out_shape=in_cost_r.shape
                                               , transform=in_transform,
                                               fill=0, all_touched=True, default_value=1)
        source = numpy.transpose(numpy.nonzero(rasterized_source))

        # generate the cost raster to source point
        mcp_source = MCP_Geometric(in_cost_r, sampling=(cell_size_x, cell_size_y))
        source_cost_acc = mcp_source.find_costs(source)[0]
        del mcp_source

        # Rasterize destination point
        rasterized_destination = features.rasterize(destination, out_shape=in_cost_r.shape,
                                                    transform=in_transform,
                                                    out=None, fill=0, all_touched=True, default_value=1,
                                                    dtype=numpy.dtype('int64'))
        destination = numpy.transpose(numpy.nonzero(rasterized_destination))

        # generate the cost raster to destination point
        mcp_dest = MCP_Geometric(in_cost_r, sampling=(cell_size_x, cell_size_y))
        dest_cost_acc = mcp_dest.find_costs(destination)[0]

        del mcp_dest

        # Generate corridor
        corridor = source_cost_acc + dest_cost_acc
        corridor = numpy.ma.masked_invalid(corridor)

        # Calculate minimum value of corridor raster
        if not numpy.ma.min(corridor) is None:
            corr_min = float(numpy.ma.min(corridor))
        else:
            corr_min = 0.05

        # Set minimum as zero and save minimum file
        corridor_min = numpy.ma.where((corridor - corr_min)> corridor_th_value, 1.,0.)

        # Process: Stamp CC and Max Line Width
        # Original code here
        # RasterClass = SetNull(IsNull(CorridorMin),((CorridorMin) + ((Canopy_Raster) >= 1)) > 0)
        temp1 = (corridor_min+ in_canopy_r)
        raster_class = numpy.ma.where(temp1 == 0, 1, 0).data

        # BERA proposed Binary morphology
        # RasterClass_binary=numpy.where(RasterClass==0,False,True)
        if exp_shk_cell > 0 and cell_size_x < 1:
            # Process: Expand
            # FLM original Expand equivalent
            cell_size = int(exp_shk_cell * 2 + 1)

            expanded = ndimage.grey_dilation(raster_class, size=(cell_size, cell_size))

            # BERA proposed Binary morphology Expand
            # Expanded = ndimage.binary_dilation(RasterClass_binary, iterations=exp_shk_cell,border_value=1)

            # Process: Shrink
            # FLM original Shrink equivalent
            file_shrink = ndimage.grey_erosion(expanded, size=(cell_size, cell_size))

            # BERA proposed Binary morphology Shrink
            # fileShrink = ndimage.binary_erosion((Expanded),iterations=Exp_Shk_cell,border_value=1)
        else:
            if BT_DEBUGGING:
                print('No Expand And Shrink cell performed.')
            file_shrink = raster_class

        # Process: Boundary Clean
        clean_raster = ndimage.gaussian_filter(file_shrink, sigma=0, mode='nearest')

        # creat mask for non-polygon area
        mask = numpy.where(clean_raster == 1, True, False)

        # Process: ndarray to shapely Polygon
        out_polygon = features.shapes(clean_raster, mask=mask, transform=in_transform)

        # create a shapely multipoly
        multi_polygon = []
        for shape, value in out_polygon:
            multi_polygon.append(shapely.geometry.shape(shape))
        poly = shapely.geometry.MultiPolygon(multi_polygon)

        # create a pandas dataframe for the FP
        out_data = pandas.DataFrame({'OLnFID': OID, 'OLnSEG': FID, 'geometry': poly})
        out_gdata = geopandas.GeoDataFrame(out_data, geometry='geometry', crs=shapefile_proj)

        return out_gdata

    except Exception as e:

        print('Exception: {}'.format(e))
    except:
        print(sys.exc_info())

def multiprocessing_Dyn_FP(line_args, processes):
    try:
        total_steps = len(line_args)
        features = []
        chunksize = math.ceil(total_steps / processes)
        with Pool(processes=processes) as pool:
            step = 0
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(dyn_process_single_line, line_args, chunksize=chunksize):
                if BT_DEBUGGING:
                    print('Got result: {}'.format(result), flush=True)
                features.append(result)
                step += 1
                print('%{}'.format(step / total_steps * 100))
        return features
    except OperationCancelledException:
        print("Operation cancelled")
        return None

def multiprocessing_dynamic_CC(line_args, processes):
    try:

        total_steps = len(line_args)

        features = []
        chunksize = math.ceil(total_steps / processes)
        with Pool(processes=int(processes)) as pool:

            step = 0
            # execute tasks in order, process results out of order
            # USE_MULTI_PROCESSING = False
            if USE_MULTI_PROCESSING:
                for result in pool.imap_unordered(dyn_canopy_cost_raster, line_args, chunksize=chunksize):
                    total_steps = len(line_args)
                    if BT_DEBUGGING:
                        print('Got result: {}'.format(result), flush=True)
                    features.append(result)
                    step += 1
                    print('%{}'.format(step / total_steps * 100))
            else:
                index = 0
                for row in line_args:
                    features.append(dyn_canopy_cost_raster(row))
                    print("Dynamic CC for line {} is done".format(index))
                    index = index + 1

        return features

    except OperationCancelledException:
        print("Operation cancelled")


if __name__ == '__main__':
    start_time = time.time()
    print('Starting Dynamic Footprint processing\n @ {}'.format(
        time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=json.loads)
    parser.add_argument('-p', '--processes')
    parser.add_argument('-v', '--verbose')
    args = parser.parse_args()
    args.input['full_step'] = False

    verbose = True if args.verbose == 'True' else False
    dynamic_line_footprint(print, **args.input, processes=int(args.processes), verbose=verbose)

    print('%{}'.format(100))
    print('Finishing Dynamic Footprint processes @ {}\n(or in {} second)'.format(
        time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), round(time.time() - start_time, 5)))
