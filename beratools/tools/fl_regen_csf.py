import math
import time
import pandas
import geopandas
import numpy
import scipy
import os
import shapely
from shapely.ops import unary_union, split
from rasterio import mask
import argparse
import json
from multiprocessing.pool import Pool

from common import *


class OperationCancelledException(Exception):
    pass
def regen_csf(line_args):
    # (result_identity,attr_seg_lines, area_analysis, change_analysis, in_change,in_tree_shp)
    attr_seg_line = line_args[0]
    result_identity = line_args[1]

    area_analysis = line_args[2]
    change_analysis = line_args[3]
    in_change = line_args[4]
    in_tree=line_args[5]


    has_footprint = True
    if type(result_identity) is geopandas.geodataframe.GeoDataFrame:
        if result_identity.empty:
            has_footprint = False
        else:
            # merge result_identity
            result_identity = result_identity.dissolve()

    elif not result_identity:
        has_footprint = False

    # Check if query result is not empty, if empty input identity footprint will be skipped
    if attr_seg_line.empty:
        return None

    if "AvgWidth" in attr_seg_line.columns.array:
        max_ln_width = math.ceil(attr_seg_line["AvgWidth"])
        if not max_ln_width>=1.0:
            max_ln_width = 0.5
    else:
        if has_footprint:
            #estimate width= (Perimeter -Sqrt(Perimeter^2-16*Area))/4
            #for long and skinny: estimate width = 2*Area / Perimeter
            P=float(result_identity.geometry.length)
            A=float(result_identity.geometry.area)
            # max_ln_width = math.ceil(((P-math.sqrt(math.pow(P,2)-(16*A)))/4)/2)
            max_ln_width = math.ceil((2 * A) / P)
            if not max_ln_width >= 1.0:
                max_ln_width = 0.5
        else:
            max_ln_width = 0.5
    index = 0

    if change_analysis and has_footprint:  # with change raster and footprint

        fp = result_identity.iloc[0].geometry
        line_feat = attr_seg_line.iloc[0].geometry


        # if the selected seg do not have identity footprint geometry
        if shapely.is_empty(fp):
            # use the buffer from the segment line
            line_buffer = shapely.buffer(line_feat, float(max_ln_width)/4)
        else:
            # if identity footprint has geometry, use as a buffer area
            line_buffer = fp
            # check trees
        with rasterio.open(in_change) as in_change_file:
            cell_size_x = in_change_file.transform[0]
            cell_size_y = -in_change_file.transform[4]
            # clipped the change base on polygon of line buffer or footprint
            clipped_change, out_transform = rasterio.mask.mask(in_change_file, [line_buffer], crop=True)

            # drop the ndarray to 2D ndarray
            clipped_change = numpy.squeeze(clipped_change, axis=0)

            # masked all NoData value cells
            clean_change = numpy.ma.masked_where(clipped_change == in_change_file.nodata, clipped_change)

            # Calculate the summary statistics from the clipped change
            change_mean = numpy.nanmean(clean_change)
        #count trees within FP area
        trees_counts = len(in_tree[in_tree.within(line_buffer)])
        if trees_counts>=7:
            reg_class="Advanced"
        elif 3<trees_counts<7:
            reg_class = "Regenerating"
        else: # 0-2 trees counts
            if change_mean>0.06:
                reg_class="Regenerating"
            else:
                reg_class="Arrested"

    elif change_analysis and not has_footprint:  # with change raster but no footprint

        line_feat = attr_seg_line.geometry.iloc[0]
        line_buffer = shapely.buffer(line_feat, float(max_ln_width))


        with rasterio.open(in_change) as in_change_file:
            cell_size_x = in_change_file.transform[0]
            cell_size_y = -in_change_file.transform[4]
            # Calculate the mean changes
            # clipped the change base on polygon of line buffer or footprint
            clipped_change, out_transform = rasterio.mask.mask(in_change_file, [line_buffer], crop=True)

            # drop the ndarray to 2D ndarray
            clipped_change = numpy.squeeze(clipped_change, axis=0)

            # masked all NoData value cells
            clean_change = numpy.ma.masked_where(clipped_change == in_change_file.nodata, clipped_change)

            # Calculate the summary statistics from the clipped change
            change_mean = numpy.nanmean(clean_change)
        # count trees within FP area
        trees_counts = len(in_tree[in_tree.within(line_buffer)])
        if trees_counts >= 7:
            reg_class = "Advanced"
        elif 3 < trees_counts < 7:
            reg_class = "Regenerating"
        else:  # 0-2 trees counts
            if change_mean > 0.06:
                reg_class = "Regenerating"
            else:
                reg_class = "Arrested"


    elif not change_analysis or not has_footprint:  # Either no change_analysis or no footprint

        line_feat = attr_seg_line.geometry.iloc[0]

        # if the selected seg do not have identity footprint geometry
        line_buffer = shapely.buffer(line_feat, float(max_ln_width))

        # count trees within FP area
        trees_counts = len(in_tree[in_tree.within(line_buffer)])
        if trees_counts >= 7:
            reg_class = "Advanced"
        elif 3 < trees_counts < 7:
            reg_class = "Regenerating"
        else:
            reg_class = "Not Available"

        change_mean=numpy.nan

    elif not change_analysis and not has_footprint:  # no change raster and no footprint
        reg_class = "Not Available"
        change_mean = numpy.nan
        trees_counts=numpy.nan

    attr_seg_line["AveChanges"]=change_mean
    attr_seg_line["Num_trees"] = trees_counts
    attr_seg_line["Reg_Class"] = reg_class
    return attr_seg_line


def identity_polygon(line_args):
    line = line_args[0]
    # in_cl_buffer = line_args[1][['geometry', 'OLnFID', 'OLnSEG']]
    in_touched_fp = line_args[1][['geometry', 'OLnFID', 'OLnSEG']]
    in_search_polygon = line_args[2]
    if 'OLnSEG' not in in_search_polygon.columns.array:
        in_search_polygon = in_search_polygon.assign(OLnSEG=0)
    if 'OLnFID' not in in_search_polygon.columns.array:
        in_search_polygon = in_search_polygon.assign(OLnFID=in_search_polygon['OLnFID'].index)
    identity = None
    try:
        # TODO: determine when there is empty polygon
        # TODO: this will produce empty identity
        if not in_search_polygon.empty:
            identity = in_search_polygon.overlay(in_touched_fp, how='identity')
            identity = identity.dropna(subset=['OLnSEG_2', 'OLnFID_2'])
            identity = identity.drop(columns=['OLnSEG_1', 'OLnFID_2'])
            identity = identity.rename(columns={'OLnFID_1': 'OLnFID', 'OLnSEG_2': 'OLnSEG'})
    except Exception as e:
        print(e)

    return line, identity


def execute_multiprocessing_identity(line_args, processes):
    # Multiprocessing identity polygon
    try:
        total_steps = len(line_args)
        features = []
        with Pool(processes) as pool:
            step = 0
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(identity_polygon, line_args):
                if BT_DEBUGGING:
                    print('Got result: {}'.format(result), flush=True)
                features.append(result)
                step += 1
                print('%{}'.format(step / total_steps * 100))

    except OperationCancelledException:
        print("Operation cancelled")
        exit()

    print("Identifies are done.")
    return features


def execute_multiprocessing_csf(line_args, processes):
    try:
        total_steps = len(line_args)
        features = []
        with Pool(processes) as pool:
            step = 0
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(regen_csf, line_args):
                if BT_DEBUGGING:
                    print('Got result: {}'.format(result), flush=True)
                    print('Line processed: {}'.format(step))

                features.append(result)
                step += 1
                print('%{}'.format(step / total_steps * 100))

    except OperationCancelledException:
        print("Operation cancelled")
        exit()

    return features


def fl_restration_csf(callback, in_line, in_footprint,in_trees, in_change, proc_segments, out_line,processes, verbose):
    # assign Tool arguments
    BT_DEBUGGING = False
    in_cl = in_line
    in_fp = in_footprint

    print("Checking input parameters ...")

    try:
        print("loading in shapefile(s) ...")
        # in_line_shp = pyogrio.read_dataframe(in_line)
        # in_tree_shp = pyogrio.read_dataframe(in_trees)
        # in_fp_shp = pyogrio.read_dataframe(in_footprint)
        in_line_shp = geopandas.read_file(in_line,engine="pyogrio")
        in_tree_shp = geopandas.read_file(in_trees,engine="pyogrio")
        in_fp_shp = geopandas.read_file(in_footprint,engine="pyogrio")
    except SystemError:
       print("Invalid input feature, please check!")
       exit()

    #Check datum, at this stage only check input data against NAD 83 datum
    sameDatum = False
    for shp in [in_line_shp,in_tree_shp,in_fp_shp]:
        if shp.crs.datum.name in NADDatum:
            sameDatum=True
        else:
            sameDatum=False
    try:
        #Check projection zone among input data with NAD 83 datum
        if sameDatum:
            if in_line_shp.crs.utm_zone != in_tree_shp.crs.utm_zone !=in_fp_shp.crs.utm_zone:
                print("Input shapefiles are on different project Zone, please check.")
                exit()
        else:
            print("Input shapefiles are not on supported Datum, please check.")
            exit()
    except Exception as error_in_shapefiles:
        print("Input shapefiles are invalid: {} , please check.".format(error_in_shapefiles))
        exit()

    if not os.path.exists(os.path.dirname(out_line)):
        os.makedirs(os.path.dirname(out_line))
    else:
        pass
    print("Checking input parameters ... Done")

    in_fields = list(in_line_shp.columns)



    # check coordinate systems between line and raster features
    try:
        # Check projection zone among input raster with input vector data
        with rasterio.open(in_change) as in_raster:
            if not in_raster.crs.to_epsg() in [in_fp_shp.crs.to_epsg(),in_line_shp.crs.to_epsg(),in_tree_shp.crs.to_epsg(),2956]:
                print("Line and raster spatial references are different , please check.")
                exit()
            else:
                change_analysis = True
    except Exception as error_in_raster:

        print("Invalid input raster: {}, please check!".format(error_in_raster))
        change_analysis = False
        exit()

    HasOLnFID = False

    # determine to do area or/and height analysis
    if len(in_fp_shp) == 0:
        print('No footprints provided, buffer of the input lines will be used instead')
        area_analysis = False
        # AOI_trees = in_tree_shp[in_tree_shp.within(in_line_shp.buffer(50))]
    else:
        area_analysis = True
        # AOI_trees = in_tree_shp[in_tree_shp.within(in_fp_shp)]

    print("Preparing line segments...")

    # Segment lines
    # Return split lines with two extra columns:['OLnFID','OLnSEG']
    # or return whole input line
    print("Input_Lines: {}".format(in_cl))
    if proc_segments == True:
        attr_seg_lines = line_split2(in_line_shp, 10)
    else:
        # copy original line input to another Geodataframe
        attr_seg_lines = geopandas.GeoDataFrame.copy(in_line_shp)

    print('%{}'.format(10))

    print("Line segments preparation done.")
    print("{} footprints to be identified by {} segments ...".format(len(in_fp_shp.index), len(attr_seg_lines)))

    # Prepare line parameters for multiprocessing
    line_args = []

    # prepare line args: list of line, line buffer and footprint polygon
    # footprint spatial searching
    footprint_sindex = in_fp_shp.sindex

    for i in attr_seg_lines.index:
        line = attr_seg_lines.iloc[[i]]
        line_buffer = line.copy()
        if proc_segments:
            line_buffer['geometry'] = line.simplify(tolerance=1, preserve_topology=True).buffer(10, cap_style=shapely.BufferCapStyle.flat)
        else:
            line_buffer['geometry'] = line.buffer(10,cap_style=shapely.BufferCapStyle.flat)
        fp_touched = in_fp_shp.iloc[footprint_sindex.query(line_buffer.iloc[0].geometry,predicate="overlaps",sort=True)]
        if not "OLnFID" in fp_touched.columns.array:
            fp_touched["OLnFID"] = int(line["OLnFID"])
        if not "OLnSEG" in fp_touched.columns.array:
            if proc_segments:
                fp_touched["OLnSEG"] = int(line["OLnSEG"])
            else:
                fp_touched["OLnSEG"] = 0
        fp_intersected=fp_touched.dissolve()
        fp_intersected.geometry = fp_intersected.geometry.clip(line_buffer)
        fp_intersected['geometry'] = fp_intersected.geometry.map(lambda x: unary_union(x))
        list_item = [line, fp_touched, fp_intersected]

        line_args.append(list_item)
    print("Identifying footprint.... ")
    # multiprocessing of identity polygons
    features = []
    if not BT_DEBUGGING:
        features = execute_multiprocessing_identity(line_args, processes)
    else:
    # Debug use
        for index in range(0,len(line_args)):
            result=(identity_polygon(line_args[index]))
            if not len(result)==0:
                features.append(result)

    print("Prepare for classify ...")
    # prepare list of result_identity, Att_seg_lines, areaAnalysis, heightAnalysis, args.input
    AOI_trees=in_tree_shp
    line_args = []
    for index in range(0, len(features)):
        list_item = [features[index][0], features[index][1], area_analysis, change_analysis, in_change,AOI_trees]
        line_args.append(list_item)

        # Linear attributes
    print("Classify lines ...")
    print('%{}'.format(60))

    # Multiprocessing regeneration classifying
    features = []
    BT_DEBUGGING=True
    if not BT_DEBUGGING:
        features = execute_multiprocessing_csf(line_args, processes)
    else:
    # Debug use
        for index in range(0,len(line_args)-1):
            result=(regen_csf(line_args[index]))
            if not len(result)==0:
                features.append(result)
                print(result)

    # Combine identity polygons into once geodataframe
    if len(features) == 0:
        print('No lines found.')
        exit()
    print('Appending output ...')
    result_attr = geopandas.GeoDataFrame(pandas.concat(features, ignore_index=True))
    result_attr.reset_index()

    print('%{}'.format(90))
    print('Saving output ...')

    # Save attributed lines, was output_att_line
    geopandas.GeoDataFrame.to_file(result_attr, out_line)

    print('%{}'.format(100))


if __name__ == '__main__':
    start_time = time.time()
    print('Line regeneration classify started at {}'.format(time.strftime("%b %Y %H:%M:%S", time.localtime())))

    # Get tool arguments

    in_args, in_verbose = check_arguments()
    fl_restration_csf(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)

    print('Current time: {}'.format(time.strftime("%d %b %Y %H:%M:%S", time.localtime())))
    print('Line regeneration classify done in {} seconds'.format(round(time.time() - start_time, 5)))
