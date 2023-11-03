import math
import time
import pandas
import geopandas
import numpy
import scipy
import shapely
from shapely.ops import unary_union, split
from rasterio import mask
import argparse
import json
from multiprocessing.pool import Pool

from common import *


class OperationCancelledException(Exception):
    pass


def line_split(callback, HasOLnFID, in_cl, seg_length, max_ln_width, sampling_type, processes, verbose):
    in_ln_shp = geopandas.GeoDataFrame.from_file(in_cl)

    # Check the OLnFID column in data. If it is not, column will be created
    if 'OLnFID' not in in_ln_shp.columns.array:
        if BT_DEBUGGING:
            print("Cannot find {} column in input line data")

        print("New column created: {}".format('OLnFID', 'OLnFID'))
        in_ln_shp['OLnFID'] = in_ln_shp.index

    # Copy all the input line into geodataframe
    in_cl_line = geopandas.GeoDataFrame.copy(in_ln_shp)

    # copy the input line into split points GoeDataframe
    in_cl_splittpoint = geopandas.GeoDataFrame.copy(in_cl_line)

    # create empty geodataframe for split line and straight line from split points
    in_cl_splitline = geopandas.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=in_ln_shp.crs)
    in_cl_straightline = geopandas.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=in_ln_shp.crs)

    # Generate points along merged input line (with start/ end point) base on user Segment Length
    i = 0
    j = 0

    # Prepare line for arbitrary split lines
    if sampling_type == 'ARBITRARY':
        # loop thought all the record in input centerlines
        for row in range(0, len(in_cl_line)):
            # get geometry from record
            in_ln_feat = in_cl_line.loc[row, 'geometry']

            # get geometry's length from record
            in_line_length = in_cl_line.loc[row, 'geometry'].length

            if 0.0 <= in_line_length <= seg_length:
                # if input line is shorter than the set arbitrary distance, get the start and end points from input line
                points = [shapely.Point(in_ln_feat.coords[0]), shapely.Point(in_ln_feat.coords[-1])]
            else:
                # if input line is longer than the set arbitrary distance,
                # get the arbitrary distance list along input line
                distances = numpy.arange(0, in_line_length, seg_length)

                # append line's end point into numpy array
                if distances[-1] < in_line_length:
                    distances = numpy.append(distances, in_line_length)

                # interpolate distance list and get all the points' coordinates
                points = [in_ln_feat.interpolate(distance) for distance in distances]

            # Make sure points are snapped to input line
            points = shapely.snap(points, in_ln_feat, 0.001)
            points = shapely.multipoints(points)

            lines = split(in_ln_feat, points)

            # replace row record's line geometry of into multipoint geometry
            in_cl_splittpoint.loc[row, 'geometry'] = points

            # Split the input line base on the split points
            # extract points coordinates into list of point GeoSeries
            listofpoint = shapely.geometry.mapping(in_cl_splittpoint.loc[row, 'geometry'])['coordinates']

            # Generate split lines (straight line) from points
            straight_ln = (list(map(shapely.geometry.LineString, zip(shapely.LineString(listofpoint).coords[:-1],
                                                                     shapely.LineString(listofpoint).coords[1:]))))
            seg_i = 1

            buffer_list = []

            for seg in straight_ln:
                for col in in_cl_splittpoint.columns.array:
                    in_cl_straightline.loc[i, col] = in_cl_splittpoint.loc[row, col]
                in_cl_straightline.loc[i, 'geometry'] = seg

                buffer_list.append(seg.buffer(max_ln_width, cap_style=shapely.BufferCapStyle.flat))

                if not HasOLnFID:
                    in_cl_straightline.loc[i, 'OLnFID'] = row

                in_cl_straightline.loc[i, 'OLnSEG'] = seg_i
                seg_i = seg_i + 1
                i = i + 1

            # Split the input lines base on buffer
            segment_list = []
            for polygon in buffer_list:
                segment_list.append(shapely.ops.split(in_cl_line.loc[row, 'geometry'], polygon))

            seg_i = 1
            seg_list_index = 0
            for seg in lines.geoms:
                for col in in_cl_splittpoint.columns.array:
                    in_cl_splitline.loc[j, col] = in_cl_splittpoint.loc[row, col]
                    in_cl_splitline.loc[j, 'geometry'] = seg

                if not HasOLnFID:
                    in_cl_splitline.loc[j, 'OLnFID'] = row

                in_cl_splitline.loc[j, 'OLnSEG'] = seg_i
                seg_i = seg_i + 1
                seg_list_index = seg_list_index + 1
                j = j + 1

            in_cl_splitline = in_cl_splitline.dropna(subset='geometry')

        in_cl_splitline.reset_index()
    elif sampling_type == "LINE-CROSSINGS":
        # create empty geodataframe for lines
        in_cl_dissolved = geopandas.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=in_ln_shp.crs)

        lines = list(line for line in in_cl_line['geometry'])
        in_cl_dissolved['geometry'] = list(shapely.ops.linemerge(lines).geoms)

        identical_segs = in_cl_dissolved.sjoin(in_cl_line, predicate='covered_by')
        identical_segs['Disso_ID'] = identical_segs.index
        columns = list(col for col in identical_segs.columns
                       if col not in ['geometry', 'index_right', 'Shape_Leng', 'Shape_Le_1', 'len'])
        identical_segs = pandas.DataFrame(identical_segs[columns])
        identical_segs.reset_index()

        share_seg = in_cl_line.sjoin(in_cl_dissolved, predicate='covered_by')
        share_seg = share_seg[share_seg.duplicated('index_right', keep=False)]
        share_seg['Disso_ID'] = share_seg['index_right']

        share_seg = pandas.DataFrame(share_seg[columns])
        share_seg.reset_index()

        segs_identity = pandas.concat([identical_segs, share_seg])
        segs_identity.reset_index()

        for seg in range(0, len(in_cl_dissolved.index)):
            in_cl_dissolved.loc[seg, 'Disso_ID'] = seg
            common_segs = segs_identity.query("Disso_ID=={}".format(seg))
            fp_list = common_segs['OLnFID']

            for col in common_segs.columns:
                in_cl_dissolved.loc[seg, col] = common_segs.loc[common_segs.index[0], col]

            in_cl_dissolved.loc[seg, 'OLnSEG'] = seg
            in_cl_dissolved.loc[[seg], 'FP_ID'] = pandas.Series([fp_list], index=in_cl_dissolved.index[[seg]])
            in_cl_dissolved.loc[[seg], 'OLnFID'] = pandas.Series([fp_list], index=in_cl_dissolved.index[[seg]])

        in_cl_dissolved['Disso_ID'].astype(int)

    else:  # Return Line as input and create two columns as Primary Key
        if not HasOLnFID:
            in_cl_line['OLnFID'] = in_cl_line.index

        in_cl_line['OLnSEG'] = 0
        in_cl_line.reset_index()

    if sampling_type == 'IN-FEATURES':
        return in_cl_line
    elif sampling_type == "ARBITRARY":
        return in_cl_splitline
    elif sampling_type == "LINE-CROSSINGS":
        return in_cl_dissolved


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


def fill_attributes(line_args):
    # (result_identity,attr_seg_lines,area_analysis,height_analysis, in_chm, max_lin_width)
    attr_seg_line = line_args[0]
    result_identity = line_args[1]

    area_analysis = line_args[2]
    height_analysis = line_args[3]
    in_chm = line_args[4]
    max_ln_width = line_args[5]

    has_footprint = True
    if type(result_identity) is geopandas.geodataframe.GeoDataFrame:
        if result_identity.empty:
            has_footprint = False
    elif not result_identity:
        has_footprint = False

    # Check if query result is not empty, if empty input identity footprint will be skipped
    if attr_seg_line.empty:
        return None

    index = attr_seg_line.index[0]
    fields = ['LENGTH', 'FP_Area', 'Perimeter', 'Bearing', 'Direction', 'Sinuosity',
              'AvgWidth', 'Fragment', 'AvgHeight', 'Volume', 'Roughness']
    values = dict.fromkeys(fields, numpy.nan)

    if height_analysis and has_footprint:  # with CHM
        with rasterio.open(in_chm) as in_chm_file:
            cell_size_x = in_chm_file.transform[0]
            cell_size_y = -in_chm_file.transform[4]

            # merge result_identity
            result_identity = result_identity.dissolve()

            fp = result_identity.iloc[0].geometry
            line_feat = attr_seg_line.geometry.iloc[0]

            # if the selected seg do not have identity footprint geometry
            if shapely.is_empty(fp):
                # use the buffer from the segment line
                line_buffer = shapely.buffer(line_feat, float(max_ln_width))
            else:
                # if identity footprint has geometry, use as a buffer area
                line_buffer = fp

            # clipped the chm base on polygon of line buffer or footprint
            clipped_chm, out_transform = rasterio.mask.mask(in_chm_file, [line_buffer], crop=True)

            # drop the ndarray to 2D ndarray
            clipped_chm = numpy.squeeze(clipped_chm, axis=0)

            # masked all NoData value cells
            clean_chm = numpy.ma.masked_where(clipped_chm == in_chm_file.nodata, clipped_chm)

            # calculate the Euclidean distance from start to end points of segment line
            eucDistance = find_euc_distance(line_feat)

            # Calculate the summary statistics from the clipped CHM
            chm_mean = numpy.ma.mean(clean_chm)
            chm_std = numpy.ma.std(clean_chm)
            chm_sum = numpy.ma.sum(clean_chm)
            chm_count = numpy.ma.count(clean_chm)
            OnecellArea = cell_size_y * cell_size_x

            sqStdPop = 0.0
            try:
                sqStdPop = math.pow(chm_std, 2) * (chm_count - 1) / chm_count
            except ZeroDivisionError as e:
                sqStdPop = 0.0

            # writing result to feature's attributes
            values['LENGTH'] = line_feat.length
            values['FP_Area'] = result_identity.iloc[0].geometry.area
            values['Perimeter'] = result_identity.iloc[0].geometry.length
            values['Bearing'] = find_bearing(attr_seg_line)
            values['Direction'] = find_direction(values['Bearing'])
            try:
                values['Sinuosity'] = line_feat.length / eucDistance
            except ZeroDivisionError as e:
                values['Sinuosity'] = numpy.nan
            try:
                values["AvgWidth"] = values['FP_Area'] / line_feat.length
            except ZeroDivisionError as e:
                values["AvgWidth"] = numpy.nan
            try:
                values["Fragment"] = values['Perimeter'] / values['FP_Area']
            except ZeroDivisionError as e:
                values["Fragment"] = numpy.nan

            values["AvgHeight"] = chm_mean
            values["Volume"] = chm_sum * OnecellArea
            values["Roughness"] = math.sqrt(math.pow(chm_mean, 2) + sqStdPop)
    elif has_footprint:  # No CHM
        line_feat = attr_seg_line.geometry.iloc[0]
        eucDistance = find_euc_distance(line_feat)
        attr_seg_line.loc[index, 'LENGTH'] = line_feat.length
        attr_seg_line.loc[index, 'FP_Area'] = result_identity.loc[0].geometry.area
        attr_seg_line.loc[index, 'Perimeter'] = result_identity.loc[0].geometry.length
        attr_seg_line.loc[index, 'Bearing'] = find_bearing(attr_seg_line)
        attr_seg_line.loc[index, 'Direction'] = find_direction(values['Bearing'])
        try:
            attr_seg_line.loc[index, 'Sinuosity'] = line_feat.length / eucDistance
        except ZeroDivisionError as e:
            attr_seg_line.loc[index, 'Sinuosity'] = numpy.nan
        try:
            attr_seg_line.loc[index, "AvgWidth"] = values['FP_Area'] / line_feat.length
        except ZeroDivisionError as e:
            attr_seg_line.loc[index, "AvgWidth"] = numpy.nan
        try:
            attr_seg_line.loc[index, "Fragment"] = values['Perimeter'] / values['FP_Area']
        except ZeroDivisionError as e:
            attr_seg_line.loc[index, "Fragment"] = numpy.nan

        attr_seg_line.loc[index, "AvgHeight"] = numpy.nan

        attr_seg_line.loc[index, "Volume"] = numpy.nan
        attr_seg_line.loc[index, "Roughness"] = numpy.nan
    else:  # no footprint
        fields.remove('AvgHeight')
        values.pop('AvgHeight')

    attr_seg_line.loc[index, fields] = values

    if attr_seg_line.empty:
        print('Geometry is empty')

    return attr_seg_line


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


def execute_multiprocessing_attributes(line_args, processes):
    try:
        total_steps = len(line_args)
        features = []
        with Pool(processes) as pool:
            step = 0
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(fill_attributes, line_args):
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


def forest_line_attributes(callback, in_line, in_footprint, in_chm, sampling_type, seg_len,
                           ln_split_tol, max_ln_width, out_line, processes, verbose):
    # assign Tool arguments
    in_cl = in_line
    in_fp = in_footprint
    seg_len = float(seg_len)
    ln_split_tol = float(ln_split_tol)
    max_ln_width = float(max_ln_width)

    # Valid input footprint shapefile has geometry
    in_fp_shp = geopandas.GeoDataFrame.from_file(in_fp)
    # in_ln_shp = geopandas.GeoDataFrame.from_file(in_cl)  # TODO: check projection
    in_ln_shp = geopandas.read_file(in_cl, rows=1)
    in_fields = list(in_ln_shp.columns)

    # check coordinate systems between line and raster features
    with rasterio.open(in_chm) as in_raster:
        if in_fp_shp.crs.to_epsg() != in_raster.crs.to_epsg():
            print("Line and raster spatial references are not the same, please check.")
            exit()

    HasOLnFID = False

    # determine to do area or/and height analysis
    if len(in_fp_shp) == 0:
        print('No footprints provided, buffer of the input lines will be used instead')
        area_analysis = False
    else:
        area_analysis = True

    try:
        with rasterio.open(in_chm) as in_CHM:
            height_analysis = True
    except Exception as error_in_CHM:
        print(error_in_CHM)
        height_analysis = False
        exit()

    # Process the following SamplingType
    sampling_list = ["IN-FEATURES", "LINE-CROSSINGS", "ARBITRARY"]
    if sampling_type not in sampling_list:
        print("SamplingType is not correct, please verify it.")
        exit()

    print("Preparing line segments...")

    # Segment lines
    # Return split lines with two extra columns:['OLnFID','OLnSEG']
    # or return Dissolved whole line
    print("Input_Lines: {}".format(in_cl))
    attr_seg_lines = line_split(print, HasOLnFID, in_cl, seg_len, max_ln_width, sampling_type,
                                processes=int(in_args.processes), verbose=in_args.verbose)

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
        line_buffer['geometry'] = line.buffer(max_ln_width, cap_style=shapely.BufferCapStyle.flat)
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
        list_item = [features[index][0], features[index][1], area_analysis, height_analysis, in_chm, max_ln_width]
        line_args.append(list_item)

        # Linear attributes
    print("Adding attributes ...")
    print('%{}'.format(60))

    # Multiprocessing identity polygon
    features = []
    features = execute_multiprocessing_attributes(line_args, processes)

    # Combine identity polygons into once geodataframe
    if len(features) == 0:
        print('No lines found.')
        exit()

    result_attr = geopandas.GeoDataFrame(pandas.concat(features, ignore_index=True))
    result_attr.reset_index()
    print('Attribute processing done.')
    print('%{}'.format(80))

    # Clean the split line attribute columns
    field_list = ['geometry', 'LENGTH', 'FP_Area', 'Perimeter', 'Bearing', 'Direction',
                  'Sinuosity', 'AvgWidth', 'AvgHeight', 'Fragment', 'Volume', 'Roughness']
    field_list.extend(in_fields)
    del_list = list(col for col in result_attr.columns if col not in field_list)
    result_attr = result_attr.drop(columns=del_list)
    result_attr.reset_index()

    print('%{}'.format(90))
    print('Saving output ...')

    # Save attributed lines, was output_att_line
    result_attr.to_file(out_line)

    print('%{}'.format(100))


if __name__ == '__main__':
    start_time = time.time()
    print('Line Attributes started at {}'.format(time.strftime("%b %Y %H:%M:%S", time.localtime())))

    # Get tool arguments
    in_args, in_verbose = check_arguments()
    forest_line_attributes(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)

    print('Current time: {}'.format(time.strftime("%d %b %Y %H:%M:%S", time.localtime())))
    print('Line Attributes processing done in {} seconds'.format(round(time.time() - start_time, 5)))
