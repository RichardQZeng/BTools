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
# Refactor to use for produce dynamic footprint from dynamic canopy and cost raster with open source libraries
# Prerequisite:  Line feature class must have the attribute Fields:"CorridorTh" "DynCanTh" "OLnFID"
# line_footprint_function.py
# Maverick Fong
# Date: 2023-Dec
# This script is part of the BERA toolset
# Webpage: https://github.com/
#
# Purpose: Creates dynamic footprint polygons for each input line based on a least
# cost corridor method and individual line thresholds.
#
# ---------------------------------------------------------------------------

import math
import time
from pathlib import Path
import uuid

from multiprocessing.pool import Pool

from scipy import stats
import xarray as xr
from xrspatial import convolution, focal
import os
import pandas as pd
from geopandas import GeoDataFrame

from scipy import ndimage
from rasterio import features, mask

from shapely import buffer
from shapely.geometry import LineString, Point, MultiPolygon, shape

from common import *
import skimage
from skimage.morphology import *
from skimage.graph import MCP_Geometric, MCP_Connect,MCP,MCP_Flexible
from dijkstra_algorithm import *

import numpy as np

class OperationCancelledException(Exception):
    pass


def dyn_np_cc_map(in_array, canopy_ht_threshold, nodata):
    masked_array = np.ma.masked_where(in_array == nodata, in_array)
    canopy_ndarray = np.ma.where(masked_array >= canopy_ht_threshold, 1., 0.)
    canopy_ndarray = np.ma.where(in_array == nodata, BT_NODATA, canopy_ndarray).data  # TODO check the code, extra step?
    return canopy_ndarray, masked_array


def dyn_fs_raster_stdmean(in_ndarray, kernel, nodata):
    # This function uses xrspatial which can handle large data but slow
    # print("Calculating Canopy Closure's Focal Statistic-Stand Deviation Raster ...")
    ndarray = np.where(in_ndarray == nodata, np.nan, in_ndarray)
    result_ndarray = focal.focal_stats(xr.DataArray(ndarray), kernel, stats_funcs=['std', 'mean'])

    # Flattening the array
    flatten_std_result_ndarray = result_ndarray[0].data.reshape(-1)
    flatten_mean_result_ndarray = result_ndarray[1].data.reshape(-1)

    # Re-shaping the array
    reshape_std_ndarray = flatten_std_result_ndarray.reshape(ndarray.shape[0], ndarray.shape[1])
    reshape_mean_ndarray = flatten_mean_result_ndarray.reshape(ndarray.shape[0], ndarray.shape[1])
    return reshape_std_ndarray, reshape_mean_ndarray


def dyn_smooth_cost(in_raster, max_line_dist, cell_x, cell_y):
    # print('Generating Cost Raster ...')

    # scipy way to do Euclidean distance transform
    euc_dist_array = None
    euc_dist_array = ndimage.distance_transform_edt(np.logical_not(in_raster), sampling=[cell_x, cell_y])

    smooth1 = float(max_line_dist) - euc_dist_array
    # cond_smooth1 = np.where(smooth1 > 0, smooth1, 0.0)
    smooth1[smooth1 <= 0.0] = 0.0
    smooth_cost_array = smooth1 / float(max_line_dist)

    return smooth_cost_array


def dyn_np_cost_raster(canopy_ndarray, cc_mean, cc_std, cc_smooth, avoidance, cost_raster_exponent):
    aM1a = (cc_mean - cc_std)
    aM1b = (cc_mean + cc_std)
    aM1 = np.divide(aM1a, aM1b, where=aM1b != 0., out=np.zeros(aM1a.shape, dtype=float))
    aM = (1. + aM1) / 2.
    aaM = (cc_mean + cc_std)
    bM = np.where(aaM <= 0., 0., aM)
    cM = bM * (1. - avoidance) + (cc_smooth * avoidance)
    dM = np.where(canopy_ndarray == 1., 1., cM)
    eM = np.exp(dM)
    result = np.where(np.isnan(eM), BT_NODATA, np.power(eM, float(cost_raster_exponent)))

    return result


def dyn_canopy_cost_raster(args):
    in_chm = args[0]
    canopy_ht_threshold = args[1]
    tree_radius = args[2]
    max_line_dist = args[3]
    canopy_avoid = args[4]
    exponent = args[5]
    res = args[6]
    nodata = args[7]
    line_df = args[8]
    out_meta = args[9]
    line_id = args[10]
    Cut_Dist = args[11]
    Side=args[12]
    canopy_thresh_percentage= float(args[13])/100
    line_buffer=args[14]

    if Side=='Left':
        canopy_ht_threshold=line_df.CL_CutHt#*canopy_thresh_percentage
    elif Side=='Right':
        canopy_ht_threshold = line_df.CR_CutHt #*canopy_thresh_percentage
    else:

        canopy_ht_threshold=1

    canopy_ht_threshold = float(canopy_ht_threshold)
    if canopy_ht_threshold<=0:
        canopy_ht_threshold=0.5
    tree_radius = float(tree_radius)  # get the round up integer number for tree search radius
    max_line_dist = float(max_line_dist)
    canopy_avoid = float(canopy_avoid)
    cost_raster_exponent = float(exponent)

    # print('Loading CHM ...')
    (cell_x, cell_y) = res
    band1_ndarray = in_chm

    # print('Preparing Kernel window ...')
    kernel = convolution.circle_kernel(cell_x, cell_y, math.ceil(tree_radius))

    # Generate Canopy Raster and return the Canopy array
    dyn_canopy_ndarray, masked_array = dyn_np_cc_map(band1_ndarray, canopy_ht_threshold, nodata)

    # Calculating focal statistic from canopy raster
    cc_std, cc_mean = dyn_fs_raster_stdmean(dyn_canopy_ndarray, kernel, nodata)

    # Smoothing raster
    cc_smooth = dyn_smooth_cost(dyn_canopy_ndarray, max_line_dist, cell_x, cell_y)
    avoidance = max(min(float(canopy_avoid), 1), 0)
    dyn_cost_ndarray = dyn_np_cost_raster(dyn_canopy_ndarray, cc_mean, cc_std,
                                          cc_smooth, avoidance, cost_raster_exponent)
    dyn_cost_ndarray[np.isnan(dyn_cost_ndarray)] = BT_NODATA_COST  # TODO was nodata, changed to BT_NODATA_COST
    return (line_df, dyn_canopy_ndarray, dyn_cost_ndarray, out_meta, max_line_dist, nodata, line_id,Cut_Dist,line_buffer)


def split_line_fc(line):
    return list(map(LineString, zip(line.coords[:-1], line.coords[1:])))


def split_line_npart(line):
    # Work out n parts for each line (divided by LP_SEGMENT_LENGTH)
    n = math.ceil(line.length / LP_SEGMENT_LENGTH)
    if n > 1:
        # divided line into n-1 equal parts;
        distances = np.linspace(0, line.length, n)
        points = [line.interpolate(distance) for distance in distances]
        line = LineString(points)
        mline = split_line_fc(line)
    else:
        mline = line
    return mline


def split_into_segments(df):
    odf = df
    crs = odf.crs
    if 'OLnSEG' not in odf.columns.array:
        df['OLnSEG'] = np.nan

    df = odf.assign(geometry=odf.apply(lambda x: split_line_fc(x.geometry), axis=1))
    df = df.explode()

    df['OLnSEG'] = df.groupby('OLnFID').cumcount()
    gdf = GeoDataFrame(df, geometry=df.geometry, crs=crs)
    gdf = gdf.sort_values(by=['OLnFID', 'OLnSEG'])
    gdf = gdf.reset_index(drop=True)
    return gdf


def split_into_equal_nth_segments(df):
    odf = df
    crs = odf.crs
    if 'OLnSEG' not in odf.columns.array:
        df['OLnSEG'] = np.nan
    df = odf.assign(geometry=odf.apply(lambda x: split_line_npart(x.geometry), axis=1))
    df = df.explode(index_parts=True)

    df['OLnSEG'] = df.groupby('OLnFID').cumcount()
    gdf = GeoDataFrame(df, geometry=df.geometry, crs=crs)
    gdf = gdf.sort_values(by=['OLnFID', 'OLnSEG'])
    gdf = gdf.reset_index(drop=True)
    return gdf


def generate_line_args(line_seg, work_in_bufferL,work_in_bufferC, raster, tree_radius, max_line_dist,
                       canopy_avoidance, exponent, work_in_bufferR,canopy_thresh_percentage):
    line_argsL = []
    line_argsR = []
    line_argsC = []
    line_id = 0
    for record in range(0, len(work_in_bufferL)):
        line_bufferL = work_in_bufferL.loc[record, 'geometry']
        line_bufferC = work_in_bufferC.loc[record, 'geometry']
        LCut=work_in_bufferL.loc[record, 'LDist_Cut']

        clipped_rasterL, out_transformL = rasterio.mask.mask(raster, [line_bufferL], crop=True,
                                                           nodata=BT_NODATA, filled=True)
        clipped_rasterL = np.squeeze(clipped_rasterL, axis=0)

        clipped_rasterC, out_transformC = rasterio.mask.mask(raster, [line_bufferC], crop=True,
                                                             nodata=BT_NODATA, filled=True)

        clipped_rasterC=np.squeeze(clipped_rasterC, axis=0)

        # make rasterio meta for saving raster later
        out_metaL = raster.meta.copy()
        out_metaL.update({"driver": "GTiff",
                         "height": clipped_rasterL.shape[0],
                         "width": clipped_rasterL.shape[1],
                         "nodata": BT_NODATA,
                         "transform": out_transformL})

        out_metaC = raster.meta.copy()
        out_metaC.update({"driver": "GTiff",
                          "height": clipped_rasterC.shape[0],
                          "width": clipped_rasterC.shape[1],
                          "nodata": BT_NODATA,
                          "transform": out_transformC})

        nodata = BT_NODATA
        line_argsL.append([clipped_rasterL, float(work_in_bufferL.loc[record, 'DynCanTh']), float(tree_radius),
                          float(max_line_dist), float(canopy_avoidance), float(exponent), raster.res, nodata,
                          line_seg.iloc[[record]], out_metaL, line_id,LCut,'Left',canopy_thresh_percentage,line_bufferC])

        line_argsC.append([clipped_rasterC, float(work_in_bufferC.loc[record, 'DynCanTh']), float(tree_radius),
                           float(max_line_dist), float(canopy_avoidance), float(exponent), raster.res, nodata,
                           line_seg.iloc[[record]], out_metaC, line_id,10,'Center',canopy_thresh_percentage,line_bufferC])

        line_id += 1

    line_id = 0
    for record in range(0, len(work_in_bufferR)):
        line_bufferR = work_in_bufferR.loc[record, 'geometry']
        RCut = work_in_bufferR.loc[record, 'RDist_Cut']
        clipped_rasterR, out_transformR = rasterio.mask.mask(raster, [line_bufferR], crop=True,
                                                             nodata=BT_NODATA, filled=True)
        clipped_rasterR = np.squeeze(clipped_rasterR, axis=0)

        # make rasterio meta for saving raster later
        out_metaR = raster.meta.copy()
        out_metaR.update({"driver": "GTiff",
                         "height": clipped_rasterR.shape[0],
                         "width": clipped_rasterR.shape[1],
                         "nodata": BT_NODATA,
                         "transform": out_transformR})
        line_bufferC = work_in_bufferC.loc[record, 'geometry']
        clipped_rasterC, out_transformC = rasterio.mask.mask(raster, [line_bufferC], crop=True,
                                                             nodata=BT_NODATA, filled=True)

        clipped_rasterC = np.squeeze(clipped_rasterC, axis=0)
        out_metaC = raster.meta.copy()
        out_metaC.update({"driver": "GTiff",
                          "height": clipped_rasterC.shape[0],
                          "width": clipped_rasterC.shape[1],
                          "nodata": BT_NODATA,
                          "transform": out_transformC})

        nodata = BT_NODATA
        # TODO deal with inherited nodata and BT_NODATA_COST
        # TODO convert nodata to BT_NODATA_COST
        line_argsR.append([clipped_rasterR, float(work_in_bufferR.loc[record, 'DynCanTh']), float(tree_radius),
                           float(max_line_dist), float(canopy_avoidance), float(exponent), raster.res, nodata,
                           line_seg.iloc[[record]], out_metaR, line_id,RCut,'Right',canopy_thresh_percentage,line_bufferC])




        line_id += 1

    return line_argsL,line_argsR,line_argsC


def find_corridor_threshold_boundary(canopy_clip, least_cost_path, corridor_raster):
    threshold = -1
    thresholds = [-1] * 10

    # morphological filters to get polygons from canopy raster
    canopy_bin = np.where(np.isclose(canopy_clip, 1.0), True, False)
    clean_holes = remove_small_holes(canopy_bin)
    clean_obj = remove_small_objects(clean_holes)

    polys = features.shapes(skimage.img_as_ubyte(clean_obj), mask=clean_obj)
    polys = [shape(poly).segmentize(FP_SEGMENTIZE_LENGTH) for poly, _ in polys]

    # perpendicular segments intersections with polygons
    size = corridor_raster.shape
    pts = []
    for poly in polys:
        pts.extend(list(poly.exterior.coords))

    index_0 = []
    index_1 = []
    for pt in pts:
        if int(pt[0]) < size[1] and int(pt[1]) < size[0]:
            index_0.append(int(pt[0]))
            index_1.append(int(pt[1]))

    try:
        thresholds = corridor_raster[index_1, index_0]
    except Exception as e:
        print(e)

    # trimmed mean of values at intersections
    threshold = stats.trim_mean(thresholds, 0.3)

    return threshold


def find_corridor_threshold(raster):
    """
    Find the optimal corridor threshold by raster histogram
    Parameters
    ----------
    raster : corridor raster

    Returns
    -------
    corridor_threshold : float

    """
    corridor_threshold = -1.0
    hist, bins = np.histogram(raster.flatten(), bins=100, range=(0, 100))
    CostStd=np.nanstd(raster.flatten())
    half_count = np.sum(hist)/2
    sub_count = 0

    for count, bin_no in zip(hist, bins):
        sub_count += count
        if sub_count > half_count:
            break

        corridor_threshold = bin_no

    return corridor_threshold


# def find_centerlines(poly_gpd, line_seg, processes):
#     centerline = None
#     centerline_gpd = []
#     rows_and_paths = []
#
#     try:
#         for i in poly_gpd.index:
#             row = poly_gpd.loc[[i]]
#             poly = row.geometry.iloc[0]
#             line_id = row.OLnFID[i]
#             lc_path = line_seg.loc[line_seg['OLnFID']==line_id].geometry[i]
#             rows_and_paths.append((row, lc_path))
#     except Exception as e:
#         print(e)
#
#     total_steps = len(rows_and_paths)
#     step = 0
#
#     if PARALLEL_MODE == MODE_MULTIPROCESSING:
#         with Pool(processes=processes) as pool:
#             # execute tasks in order, process results out of order
#             for result in pool.imap_unordered(find_single_centerline, rows_and_paths):
#                 centerline_gpd.append(result)
#                 step += 1
#                 print(' "PROGRESS_LABEL Centerline {} of {}" '.format(step, total_steps), flush=True)
#                 print(' %{} '.format(step / total_steps * 100))
#                 # print('Centerline No. {} done'.format(step))
#     elif PARALLEL_MODE == MODE_SEQUENTIAL:
#         for item in rows_and_paths:
#             row_with_centerline = find_single_centerline(item)
#             centerline_gpd.append(row_with_centerline)
#             step += 1
#             print(' "PROGRESS_LABEL Centerline {} of {}" '.format(step, total_steps), flush=True)
#             print(' %{} '.format(step / total_steps * 100))
#             # print('Centerline No. {} done'.format(step))
#
#     return pd.concat(centerline_gpd)


# def find_single_centerline(row_and_path):
#     """
#
#     Parameters
#     ----------
#         in_canopy_r has np.inf, not masked
#         in_cost_r has BT_NODATA_COST, not masked
#         All other rasters are masked
#     segment
#
#     Returns
#     -------
#
#     """
#     row = row_and_path[0]
#     lc_path = row_and_path[1]
#
#     poly = row.geometry.iloc[0]
#     centerline = find_centerline(poly, lc_path)
#     row['centerline'] = centerline
#
#     return row


def process_single_line_relative(segment):
    # Segment args from mulitprocessing:
    # [clipped_chm, float(work_in_bufferR.loc[record, 'DynCanTh']), float(tree_radius),
    # float(max_line_dist), float(canopy_avoidance), float(exponent), raster.res, nodata,
    # line_seg.iloc[[record]], out_meta, line_id,RCut,Side,canopy_thresh_percentage,line_buffer]

    # this will change segment content, and parameters will be changed
    segment = dyn_canopy_cost_raster(segment)
    # Segement after Clipped Canopy and Cost Raster
    # line_df, dyn_canopy_ndarray, dyn_cost_ndarray, out_meta, max_line_dist, nodata, line_id,Cut_Dist,line_buffer

    # this function takes single line to work the line footprint
    # (regardless it process the whole line or individual segment)
    df = segment[0]
    in_canopy_r = segment[1]
    in_cost_r = segment[2]

    # in_transform = segment[3]
    if np.isnan(in_canopy_r).all():
        print("Canopy raster empty")

    if np.isnan(in_cost_r).all():
        print("Cost raster empty")


    in_meta = segment[3]
    exp_shk_cell = segment[4]
    no_data = segment[5]
    line_id = segment[6]
    Cut_Dist=segment[7]
    line_bufferR=segment[8]

    shapefile_proj = df.crs
    in_transform = in_meta['transform']

    FID = df['OLnSEG']  # segment line feature ID
    OID = df['OLnFID']  # original line ID for segment line

    segment_list = []

    feat = df.geometry.iloc[0]
    for coord in feat.coords:
        segment_list.append(coord)

    # Find origin and destination coordinates
    x1, y1 = segment_list[0][0:2]
    x2, y2 = segment_list[-1][0:2]

    # Create Point "origin"
    origin_point = Point([x1, y1])
    origin = [shapes for shapes in GeoDataFrame(geometry=[origin_point], crs=shapefile_proj).geometry]

    # Create Point "destination"
    destination_point = Point([x2, y2])
    destination = [shapes for shapes in
                   GeoDataFrame(geometry=[destination_point], crs=shapefile_proj).geometry]

    cell_size_x = in_transform[0]
    cell_size_y = -in_transform[4]

    # Work out the corridor from both end of the centerline
    try:
        # Rasterize source point
        # rasterized_source = features.rasterize(origin, out_shape=in_cost_r.shape, transform=in_transform,
        #                                        fill=0, all_touched=True, default_value=1)
        # source = np.transpose(np.nonzero(rasterized_source))

        # Rasterize destination point
        # rasterized_dest = features.rasterize(destination, out_shape=in_cost_r.shape, transform=in_transform,
        #                                        fill=0, all_touched=True, default_value=1)
        # destination= np.transpose(np.nonzero(rasterized_dest))

        # TODO: further investigate and submit issue to skimage
        # There is a severe bug in skimage find_costs
        # when nan is present in clip_cost_r, find_costs cause access violation
        # no message/exception will be caught
        # change all nan to BT_NODATA_COST for workaround
        remove_nan_from_array(in_cost_r)

        # generate the cost raster to source point
        transformer = rasterio.transform.AffineTransformer(in_transform)
        source = [transformer.rowcol(x1, y1)]

        # generate the cost raster to source point
        mcp_source = MCP_Geometric(in_cost_r, sampling=(cell_size_x, cell_size_y))
        source_cost_acc = mcp_source.find_costs(source)[0]
        # del mcp_source

        # generate the cost raster to destination point
        destination = [transformer.rowcol(x2, y2)]


        # # # generate the cost raster to destination point
        mcp_dest = MCP_Geometric(in_cost_r, sampling=(cell_size_x, cell_size_y))
        dest_cost_acc = mcp_dest.find_costs(destination)[0]

        # generate 1m interval points along line
        distance_delta = 1
        distances = np.arange(0, feat.length, distance_delta)

        multipoint_along_line = [feat.interpolate(distance) for distance in distances]
        points_along_line=[]

        for coord in multipoint_along_line:
            points_along_line.append([coord.x, coord.y])
        points=[]
        for pt in range(0,len(points_along_line)):
            points.append([transformer.rowcol(points_along_line[pt][0], points_along_line[pt][1])])

        # Rasterize points along line
        rasterized_points_Alongln = features.rasterize(multipoint_along_line, out_shape=in_cost_r.shape, transform=in_transform,
                                               fill=0, all_touched=True, default_value=1)
        points_Alongln = np.transpose(np.nonzero(rasterized_points_Alongln))


        #Find minimum cost paths through an N-d costs array.
        mcp_flexible1=MCP_Flexible(in_cost_r,sampling=(cell_size_x, cell_size_y),fully_connected=True)
        flex_cost_alongLn, flex_back_alongLn = mcp_flexible1.find_costs(starts=points_Alongln)

        # Generate corridor
        # corridor = source_cost_acc + dest_cost_acc
        corridor=flex_cost_alongLn#+flex_cost_dest #cum_cost_tosource+cum_cost_todestination
        corridor = np.ma.masked_invalid(corridor)

        # Calculate minimum value of corridor raster
        if not np.ma.min(corridor) is None:
            corr_min = float(np.ma.min(corridor))
        else:
            corr_min = 0.5




        # normalize corridor raster by deducting corr_min
        corridor_norm = corridor - corr_min

        # export intermediate raster for debugging
        BT_DEBUGGING=False
        if BT_DEBUGGING:
            suffix = str(uuid.uuid4())[:8]
            path_temp = Path(r"D:\BT_Test\ConcaveHull\test-cost")
            if path_temp.exists():
                path_canopy = path_temp.joinpath(suffix + '_canopy.tif')
                path_cost = path_temp.joinpath(suffix + '_cost.tif')
                path_corridor = path_temp.joinpath(suffix + '_corridor.tif')
                path_corridor_min = path_temp.joinpath(suffix + '_corridor_min.tif')
                path_corridor_norm = path_temp.joinpath(suffix + '_corridor_norm.tif')
                out_canopy = np.ma.masked_equal(in_canopy_r, np.inf)
                out_cost = np.ma.masked_equal(in_cost_r, np.inf)
                save_raster_to_file(out_canopy, in_meta, path_canopy)
                save_raster_to_file(out_cost, in_meta, path_cost)
                save_raster_to_file(corridor, in_meta, path_corridor)
                save_raster_to_file(corridor_norm, in_meta, path_corridor_min)
                save_raster_to_file(corridor_norm*100/df.length.iloc[0], in_meta, path_corridor_norm)
            else:
                print('Debugging: raster folder not exists.')

        # Set minimum as zero and save minimum file
        # corridor_th_value = find_corridor_threshold(corridor_norm)
        corridor_th_value=(Cut_Dist/cell_size_x)
        # corridor_th_value = find_corridor_threshold_boundary(in_canopy_r, feat, corridor_norm)
        if corridor_th_value < 0:  # if no threshold found, use default value
           corridor_th_value = (FP_CORRIDOR_THRESHOLD/ cell_size_x)

        # corridor_th_value = FP_CORRIDOR_THRESHOLD
        corridor_thresh = np.ma.where(corridor_norm > corridor_th_value, 1.0, 0.0)
        corridor_thresh_cl = np.ma.where(corridor_norm > (corridor_th_value+(1/cell_size_x)), 1.0, 0.0)

        # find contiguous corridor polygon for centerline
        corridor_poly_gpd = find_corridor_polygon(corridor_thresh_cl, in_transform, df)

        # Process: Stamp CC and Max Line Width
        # Original code here
        # RasterClass = SetNull(IsNull(CorridorMin),((CorridorMin) + ((Canopy_Raster) >= 1)) > 0)
        temp1 = (corridor_thresh + in_canopy_r)
        raster_class = np.ma.where(temp1 == 0, 1, 0).data

        # BERA proposed Binary morphology
        # RasterClass_binary=np.where(RasterClass==0,False,True)
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
        mask = np.where(clean_raster == 1, True, False)

        # Process: ndarray to shapely Polygon
        out_polygon = features.shapes(clean_raster, mask=mask, transform=in_transform)

        # create a shapely multipolygon
        multi_polygon = []
        for poly, value in out_polygon:
            multi_polygon.append(shape(poly))
        poly = MultiPolygon(multi_polygon)

        # create a pandas dataframe for the FP
        out_data = pd.DataFrame({'OLnFID': OID, 'OLnSEG': FID, 'CorriThresh': corridor_th_value, 'geometry': poly})
        out_gdata = GeoDataFrame(out_data, geometry='geometry', crs=shapefile_proj)


        return out_gdata, corridor_poly_gpd

    except Exception as e:
        print('Exception: {}'.format(e))


def multiprocessing_footprint_relative(line_args, processes):
    try:
        total_steps = len(line_args)

        feats = []
        # chunksize = math.ceil(total_steps / processes)
        with Pool(processes=processes) as pool:
            step = 0
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(process_single_line_relative, line_args):
                if BT_DEBUGGING:
                    print('Got result: {}'.format(result), flush=True)
                feats.append(result)
                step += 1
                print(' "PROGRESS_LABEL Dynamic Line Footprint {} of {}" '.format(step, total_steps), flush=True)
                print(' %{} '.format(step / total_steps * 100))
        return feats
    except OperationCancelledException:
        print("Operation cancelled")
        return None


def main_line_footprint_relative(callback, in_line, in_chm, max_ln_width, exp_shk_cell, out_footprint, out_centerline,
                                 tree_radius, max_line_dist, canopy_avoidance, exponent, full_step,
                                 canopy_thresh_percentage, processes, verbose):
    # use_corridor_th_col = True
    line_seg = GeoDataFrame.from_file(in_line)

    # If Dynamic canopy threshold column not found, create one
    if 'DynCanTh' not in line_seg.columns.array:
        print("Please create field {} first".format('DynCanTh'))
        exit()
    if not float(canopy_thresh_percentage):
        canopy_thresh_percentage=50
    else:
        canopy_thresh_percentage = float(canopy_thresh_percentage)


    if float(canopy_avoidance)<=0.0:
        canopy_avoidance=0.0
    if float(exponent)<=0.0:
        exponent=1.0
    # If OLnFID column is not found, column will be created
    if 'OLnFID' not in line_seg.columns.array:
        print("Created {} column in input line data.".format('OLnFID'))
        line_seg['OLnFID'] = line_seg.index

    if 'OLnSEG' not in line_seg.columns.array:
        line_seg['OLnSEG'] = line_seg['OLnFID']

    print('%{}'.format(10))

    # check coordinate systems between line and raster features
    with rasterio.open(in_chm) as raster:
        line_args = []

        if compare_crs(vector_crs(in_line), raster_crs(in_chm)):
            proc_segments = False
            if proc_segments:
                print("Splitting lines into segments...")
                line_seg_split = split_into_segments(line_seg)
                print("Splitting lines into segments...Done")
            else:
                if full_step:
                    line_seg_split = line_seg
                else:
                    line_seg_split = split_into_Equal_Nth_segments(line_seg,100)

            print('%{}'.format(20))

            work_in_bufferL1 = GeoDataFrame.copy(line_seg_split)
            work_in_bufferL2 = GeoDataFrame.copy(line_seg_split)
            work_in_bufferR1 = GeoDataFrame.copy(line_seg_split)
            work_in_bufferR2 = GeoDataFrame.copy(line_seg_split)
            work_in_bufferC = GeoDataFrame.copy(line_seg_split)
            work_in_bufferL1['geometry'] = buffer(work_in_bufferL1['geometry'], distance=float(max_ln_width)+1,
                                                 cap_style=3,single_sided=True)

            work_in_bufferL2['geometry'] = buffer(work_in_bufferL2['geometry'], distance=-1,
                                                  cap_style=3, single_sided=True)

            work_in_bufferL = GeoDataFrame(pd.concat([work_in_bufferL1,work_in_bufferL2]))
            work_in_bufferL=work_in_bufferL.dissolve(by='OLnFID', as_index=False)

            work_in_bufferR1['geometry'] = buffer(work_in_bufferR1['geometry'], distance=-float(max_ln_width)-1,
                                                 cap_style=3, single_sided=True)
            work_in_bufferR2['geometry'] = buffer(work_in_bufferR2['geometry'], distance=1,
                                                  cap_style=3, single_sided=True)

            work_in_bufferR = GeoDataFrame(pd.concat([work_in_bufferR1, work_in_bufferR2]))
            work_in_bufferR = work_in_bufferR.dissolve(by='OLnFID', as_index=False)

            work_in_bufferC['geometry'] = buffer(work_in_bufferC['geometry'], distance=float(max_ln_width),
                                                 cap_style=3, single_sided=False)
            print("Prepare CHMs for Dynamic cost raster ...")
            line_argsL, line_argsR,line_argsC= generate_line_args(line_seg_split, work_in_bufferL,work_in_bufferC, raster, tree_radius, max_line_dist,
                                           canopy_avoidance, exponent, work_in_bufferR,canopy_thresh_percentage)
        else:
            print("Line and canopy raster spatial references are not same, please check.")
            exit()
        # pass center lines for footprint
        print("Generating Dynamic footprint ...")

        feat_listL = []
        feat_listR = []
        feat_listC = []
        poly_listL = []
        poly_listR = []
        footprint_listL = []
        footprint_listR = []
        footprint_listC = []
        # PARALLEL_MODE = MODE_SEQUENTIAL
        if PARALLEL_MODE == MODE_MULTIPROCESSING:
            # feat_listC = multiprocessing_footprint_relative(line_argsC, processes)
            feat_listL = multiprocessing_footprint_relative(line_argsL, processes)
            feat_listR = multiprocessing_footprint_relative(line_argsR, processes)
        elif PARALLEL_MODE == MODE_SEQUENTIAL:
            print("There are {} result to process.".format(len(line_args)))
            step = 1
            total_steps = len(line_argsL)
            for row in line_argsL:
                feat_listL.append(process_single_line_relative(row))
                print("Footprint for line {} is done".format(step))
                print(' "PROGRESS_LABEL Dynamic Line Footprint {} of {}" '.format(step, total_steps), flush=True)
                print(' %{} '.format((step/total_steps)*100))
                step += 1
            step = 1
            total_steps = len(line_argsR)
            for row in line_argsR:
                feat_listR.append(process_single_line_relative(row))
                print("Footprint for line {} is done".format(step))
                print(' "PROGRESS_LABEL Dynamic Line Footprint {} of {}" '.format(step, total_steps), flush=True)
                print(' %{} '.format((step/total_steps)*100))
                step += 1

            # step = 1
            # total_steps = len(line_argsC)
            # for row in line_argsC:
            #     feat_listC.append(process_single_line_relative(row))
            #     print("Footprint for line {} is done".format(step))
            #     print(' "PROGRESS_LABEL Dynamic Line Footprint {} of {}" '.format(step, total_steps), flush=True)
            #     print(' %{} '.format((step / total_steps) * 100))
            #     step += 1

    print('%{}'.format(80))
    print("Task done.")


    for feat in feat_listL:
        if feat:
            footprint_listL.append(feat[0])
            poly_listL.append(feat[1])

    for feat in feat_listR:
        if feat:
            footprint_listR.append(feat[0])
            poly_listR.append(feat[1])



    print('Writing shapefile ...')
    resultsL = GeoDataFrame(pd.concat(footprint_listL))
    resultsL['geometry'] = resultsL['geometry'].buffer(0.005)
    resultsR = GeoDataFrame(pd.concat(footprint_listR))
    resultsR['geometry'] = resultsR['geometry'].buffer(0.005)
    resultsL = resultsL.sort_values(by=['OLnFID', 'OLnSEG'])
    resultsR = resultsR.sort_values(by=['OLnFID', 'OLnSEG'])
    resultsL = resultsL.reset_index(drop=True)
    resultsR = resultsR.reset_index(drop=True)
    #
    # # dissolved polygon group by column 'OLnFID'
    # dissolved_resultsL = resultsL.dissolve(by='OLnFID', as_index=False)
    # dissolved_resultsR = resultsR.dissolve(by='OLnFID', as_index=False)

    resultsAll=GeoDataFrame(pd.concat([resultsL,resultsR]))
    dissolved_results = resultsAll.dissolve(by='OLnFID', as_index=False)
    dissolved_results['geometry']=dissolved_results['geometry'].buffer(-0.005)
    print("Saving output ...")
    dissolved_results.to_file(out_footprint)
    print("Footprint file saved")

    # dissolved polygon group by column 'OLnFID'
    print("Generating centerlines ...")
    resultsCL = GeoDataFrame(pd.concat(poly_listL))
    resultsCL['geometry'] = resultsCL['geometry'].buffer(0.005)
    resultsCR = GeoDataFrame(pd.concat(poly_listR))
    resultsCR['geometry'] = resultsCR['geometry'].buffer(0.005)
    # resultsCL = resultsCL.sort_values(by=['OLnFID', 'OLnSEG'])
    # resultsCR = resultsCR.sort_values(by=['OLnFID', 'OLnSEG'])
    # resultsCL = resultsCL.reset_index(drop=True)
    # resultsCR = resultsCR.reset_index(drop=True)
    resultsCLR = GeoDataFrame(pd.concat([resultsCL, resultsCR]))
    resultsCLR = resultsCLR.dissolve(by='OLnFID', as_index=False)
    resultsCLR = resultsCLR.sort_values(by=['OLnFID', 'OLnSEG'])
    resultsCLR = resultsCLR.reset_index(drop=True)
    resultsCLR['geometry']=resultsCLR['geometry'].buffer(-0.005)

    # out_centerline=False
    # save lines to file
    if out_centerline:
        poly_centerline_gpd = find_centerlines(resultsCLR, line_seg, processes)
        poly_gpd = poly_centerline_gpd.copy()
        centerline_gpd = poly_centerline_gpd.copy()

        centerline_gpd = centerline_gpd.set_geometry('centerline')
        centerline_gpd = centerline_gpd.drop(columns=['geometry'])
        centerline_gpd.crs = poly_centerline_gpd.crs
        centerline_gpd.to_file(out_centerline)
        print("Centerline file saved")

        # save polygons
        path = Path(out_centerline)
        path = path.with_stem(path.stem+'_poly')
        poly_gpd = poly_gpd.drop(columns=['centerline'])
        poly_gpd.to_file(path)

    print('%{}'.format(100))


if __name__ == '__main__':
    start_time = time.time()
    print('Dynamic Footprint processing started')
    print('Current time: {}'.format(time.strftime("%d %b %Y %H:%M:%S", time.localtime())))

    in_args, in_verbose = check_arguments()
    main_line_footprint_relative(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)

    print('%{}'.format(100))
    print('Dynamic Footprint processing finished')
    print('Current time: {}'.format(time.strftime("%d %b %Y %H:%M:%S", time.localtime())))
    print('Total processing time (seconds): {}'.format(round(time.time() - start_time, 3)))
