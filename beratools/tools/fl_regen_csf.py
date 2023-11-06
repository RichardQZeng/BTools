import math
import time
import pandas
import geopandas
import pyogrio
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
def split_line_nPart(line):
    from shapely.ops import split,snap
    # Work out n parts for each line (divided by 10m)
    n=math.ceil(line.length/10)
    if n>1:
        # divided line into n-1 equal parts;
        distances=numpy.linspace(0,line.length,n)
        points=[line.interpolate(distance) for distance in distances]

        split_points=shapely.multipoints(points)
        # mline=cut_line_at_points(line,points)
        mline = split(line, split_points)
        # mline=split_line_fc(line)
    else:
        mline=line
    return mline
def split_into_Equal_Nth_segments(df):
    odf=df
    crs=odf.crs
    if not 'OLnSEG' in odf.columns.array:
        df['OLnSEG'] = numpy.nan
    df=odf.assign(geometry=odf.apply(lambda x: split_line_nPart(x.geometry), axis=1))
    df=df.explode()

    df['OLnSEG'] = df.groupby('OLnFID').cumcount()
    gdf=geopandas.GeoDataFrame(df,geometry=df.geometry,crs=crs)
    gdf = gdf.sort_values(by=['OLnFID', 'OLnSEG'])
    gdf=gdf.reset_index(drop=True)
    return  gdf

def line_split(callback, HasOLnFID, in_ln_shp, seg_length,proc_segments,in_fields,  processes, verbose):
    # in_ln_shp = geopandas.GeoDataFrame.from_file(in_cl)

    # Check the OLnFID column in data. If it is not, column will be created
    if 'OLnFID' not in in_ln_shp.columns.array:
        if BT_DEBUGGING:
            print("Cannot find {} column in input line data")

        print("New column created: {}".format('OLnFID', 'OLnFID'))
        in_ln_shp['OLnFID'] = in_ln_shp.index
    if proc_segments== True:
        line_seg=split_into_Equal_Nth_segments(in_ln_shp)
    else:
    # copy original line input to another Geodataframe
        in_cl_line = geopandas.GeoDataFrame.copy(in_ln_shp)

    # Prepare line for arbitrary split lines
    if proc_segments:
        return line_seg

    else:  # Return Line as input and create two columns as Primary Key
        return in_cl_line


def find_direction(bearing):
    ori = "N-S"
    if 22.5 <= bearing < 67.5 or 202.5 <= bearing < 247.5:
        ori = "NE-SW"
    elif 67.5 <= bearing < 112.5 or 247.5 <= bearing < 292.5:
        ori = "E-W"
    elif 112.5 <= bearing < 157.5 or 292.5 <= bearing < 337.5:
        ori = "NW-SE"
    return ori


def find_euc_distance(in_feat):
    x1, y1 = in_feat.coords[0]
    x2, y2 = in_feat.coords[-1]
    return scipy.spatial.distance.euclidean([x1, y1], [x2, y2])


def find_bearing(seg):  # Geodataframe
    line_coords = shapely.geometry.mapping(seg[['geometry']])['features'][0]['geometry']['coordinates']

    x1, y1 = line_coords[0]
    x2, y2 = line_coords[-1]
    dx = x2 - x1
    dy = y2 - y1

    bearing = numpy.nan

    if dx == 0.0 and dy < 0.0:
        bearing = 180.0

    elif dx == 0.0 and dy > 0.0:
        bearing = 0.0
    elif dx > 0.0 and dy == 0.0:
        bearing = 90.0
    elif dx < 0.0 and dy == 0.0:
        bearing = 270.0
    elif dx > 0.0:
        angle = math.degrees(math.atan(dy / dx))
        bearing = 90.0 - angle
    elif dx < 0.0:
        angle = math.degrees(math.atan(dy / dx))
        bearing = 270.0 - angle

    return bearing


def restoration_csf(line_args):
    # (result_identity,attr_seg_lines, area_analysis, change_analysis, in_change,in_tree_shp, 50)
    attr_seg_line = line_args[0]
    result_identity = line_args[1]

    area_analysis = line_args[2]
    change_analysis = line_args[3]
    in_change = line_args[4]
    in_tree=line_args[5]
    max_ln_width = line_args[6]

    has_footprint = True
    if type(result_identity) is geopandas.geodataframe.GeoDataFrame:
        if result_identity.empty:
            has_footprint = False

    elif not result_identity:
        has_footprint = False

    # Check if query result is not empty, if empty input identity footprint will be skipped
    if attr_seg_line.empty:
        return None

    index = 0

    if change_analysis and has_footprint:  # with change raster and footprint

        with rasterio.open(in_change) as in_change_file:
            cell_size_x = in_change_file.transform[0]
            cell_size_y = -in_change_file.transform[4]

            # merge result_identity
            result_identity = result_identity.dissolve()

            fp = result_identity.iloc[0].geometry
            line_feat = attr_seg_line.geometry.iloc[0]

            # if the selected seg do not have identity footprint geometry
            if shapely.is_empty(fp):
                # use the buffer from the segment line
                line_buffer = shapely.buffer(line_feat, float(max_ln_width)/4)
            else:
                # if identity footprint has geometry, use as a buffer area
                line_buffer = fp
                # check trees

            #count trees within FP area
            trees_counts = len(in_tree[in_tree.within(line_buffer)])
            if trees_counts>=7:
                reg_class="Advanced"
            elif 3<trees_counts<7:
                reg_class = "Regenerating"
            else:

                # clipped the change base on polygon of line buffer or footprint
                clipped_change, out_transform = rasterio.mask.mask(in_change_file, [line_buffer], crop=True)

                # drop the ndarray to 2D ndarray
                clipped_change = numpy.squeeze(clipped_change, axis=0)

                # masked all NoData value cells
                clean_change = numpy.ma.masked_where(clipped_change == in_change_file.nodata, clipped_change)

                # Calculate the summary statistics from the clipped change
                change_mean = numpy.ma.mean(clean_change)
                if change_mean>0:
                    reg_class="Regenerating"
                else:
                    reg_class="Arrested"
    elif change_analysis and not has_footprint:  # with change raster but no footprint
        with rasterio.open(in_change) as in_change_file:
            cell_size_x = in_change_file.transform[0]
            cell_size_y = -in_change_file.transform[4]

            # merge result_identity
            result_identity = result_identity.dissolve()

            fp = result_identity.iloc[0].geometry
            line_feat = attr_seg_line.geometry.iloc[0]

            # if the selected seg do not have identity footprint geometry
            if shapely.is_empty(fp):
                # use the buffer from the segment line
                line_buffer = shapely.buffer(line_feat, float(max_ln_width) / 4)
            else:
                # if identity footprint has geometry, use as a buffer area
                line_buffer = fp
                # check trees

            # count trees within FP area
            trees_counts = len(in_tree[in_tree.within(line_buffer)])
            if trees_counts >= 7:
                reg_class = "Advanced"
            elif 3 < trees_counts < 7:
                reg_class = "Regenerating"
            else:

                # clipped the change base on polygon of line buffer or footprint
                clipped_change, out_transform = rasterio.mask.mask(in_change_file, [line_buffer], crop=True)

                # drop the ndarray to 2D ndarray
                clipped_change = numpy.squeeze(clipped_change, axis=0)

                # masked all NoData value cells
                clean_change = numpy.ma.masked_where(clipped_change == in_change_file.nodata, clipped_change)

                # Calculate the summary statistics from the clipped change
                change_mean = numpy.ma.mean(clean_change)
                if change_mean > 0:
                    reg_class = "Regenerating"
                else:
                    reg_class = "Arrested"


    elif not change_analysis or not has_footprint:  # Either no change_analysis or no footprint
        # merge result_identity
        result_identity = result_identity.dissolve()

        fp = result_identity.iloc[0].geometry
        line_feat = attr_seg_line.geometry.iloc[0]

        # if the selected seg do not have identity footprint geometry
        if shapely.is_empty(fp):
            # use the buffer from the segment line
            line_buffer = shapely.buffer(line_feat, float(max_ln_width) / 4)
        else:
            # if identity footprint has geometry, use as a buffer area
            line_buffer = fp

        # count trees within FP area
        trees_counts = len(in_tree[in_tree.within(line_buffer)])
        if trees_counts >= 7:
            reg_class = "Advanced"
        elif 3 < trees_counts < 7:
            reg_class = "Regenerating"
        else:
            reg_class = "Not Available"

    elif not change_analysis and not has_footprint:  # no change raster and no footprint
        reg_class = "Not Available"



    return result_identity


def identity_polygon(line_args):
    line = line_args[0]
    in_cl_buffer = line_args[1][['geometry', 'OLnFID', 'OLnSEG']]
    in_fp_polygon = line_args[2]
    if 'OLnSEG' not in in_fp_polygon.columns.array:
        in_fp_polygon = in_fp_polygon.assign(OLnSEG=0)

    identity = None
    try:
        # TODO: determine when there is empty polygon
        # TODO: this will produce empty identity
        if not in_fp_polygon.empty:
            identity = in_fp_polygon.overlay(in_cl_buffer, how='identity')
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
            for result in pool.imap_unordered(restoration_csf, line_args):
                if BT_DEBUGGING:
                    print('Got result: {}'.format(result), flush=True)

                if type(result) is geopandas.geodataframe.GeoDataFrame:
                    if not result.empty:
                        features.append(result)

                step += 1
                print('Line processed: {}'.format(step))
                print('%{}'.format(step / total_steps * 100))

    except OperationCancelledException:
        print("Operation cancelled")
        exit()

    return features


def fl_restration_csf(callback, in_line, in_footprint,in_trees, in_change, proc_segments, out_line,processes, verbose):
    # assign Tool arguments
    in_cl = in_line
    in_fp = in_footprint

    UTMZone=['"Longitude of natural origin",-111']

    print("Checking input parameters ...")

    try:
        in_line_shp = pyogrio.read_dataframe(in_line)
        in_tree_shp = pyogrio.read_dataframe(in_trees)
        in_fp_shp = pyogrio.read_dataframe(in_footprint)
    except SystemError:
       print("Invalid input feature, please check!")
       exit()

    sameDatum = False
    for shp in [in_line_shp,in_tree_shp,in_fp_shp]:
        if shp.crs.datum.name in NADDatum:
            sameDatum=True
        else:
            sameDatum=False
    try:
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



    # Valid input footprint shapefile has geometry
    # in_fp_shp = geopandas.GeoDataFrame.from_file(in_fp)
    # in_ln_shp = geopandas.GeoDataFrame.from_file(in_cl)  # TODO: check projection
    in_fields = list(in_line_shp.columns)

    # check coordinate systems between line and raster features
    try:
        with rasterio.open(in_change) as in_raster:
            if not in_raster.crs.to_epsg() in [in_fp_shp.crs.to_epsg(),in_line_shp.crs.to_epsg(),in_tree_shp.crs.to_epsg()]:
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
    else:
        area_analysis = True

    # try:
    #     with rasterio.open(in_change) as in_change_raster:
    #         change_analysis = True
    # except Exception as error_in_change:
    #     print(error_in_change)
    #     change_analysis = False
    #     exit()


    print("Preparing line segments...")

    # Segment lines
    # Return split lines with two extra columns:['OLnFID','OLnSEG']
    # or return whole input line
    print("Input_Lines: {}".format(in_cl))
    attr_seg_lines = line_split(print, HasOLnFID, in_line_shp, 10,proc_segments,in_fields, processes=int(in_args.processes), verbose=in_args.verbose)

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
        line_buffer['geometry'] = line.buffer(50, cap_style=shapely.BufferCapStyle.flat)
        fp_intersected = in_fp_shp.iloc[footprint_sindex.query(line_buffer.iloc[0].geometry)]
        list_item = [line, line_buffer, fp_intersected]

        line_args.append(list_item)

    # multiprocessing of identity polygons
    features = []
    features = execute_multiprocessing_identity(line_args, processes)

    print("Prepare for filling attributes ...")
    # prepare list of result_identity, Att_seg_lines, areaAnalysis, heightAnalysis, args.input
    line_args = []
    for index in range(0, len(features)):
        list_item = [features[index][0], features[index][1], area_analysis, change_analysis, in_change,in_tree_shp, 50]
        line_args.append(list_item)

        # Linear attributes
    print("Adding attributes ...")
    print('%{}'.format(60))

    # Multiprocessing identity polygon
    features = []
    # features = execute_multiprocessing_csf(line_args, processes)
    for index in range(0,len(line_args)-1):
        result=(restoration_csf(line_args[index]))
        if not len(result)==0:
            features.append(result)
    # Combine identity polygons into once geodataframe
    if len(features) == 0:
        print('No lines found.')
        exit()

    result_attr = geopandas.GeoDataFrame(pandas.concat(features, ignore_index=True))
    result_attr.reset_index()
    print('Attribute processing done.')
    print('%{}'.format(80))

    # Clean the split line attribute columns
    field_list = ['OLnFID', 'OLnSEG', 'geometry', 'LENGTH', 'FP_Area', 'Perimeter', 'Bearing', 'Direction',
                  'Sinuosity', 'AvgWidth', 'AvgHeight', 'Fragment', 'Volume', 'Roughness', 'Disso_ID', 'FP_ID']
    field_list.extend(in_fields)
    del_list = list(col for col in result_attr.columns if col not in field_list)
    result_attr = result_attr.drop(columns=del_list)
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
