import math
import time
import pandas as pd
import geopandas as gpd
import numpy
import scipy
import shapely
from shapely.ops import unary_union, split
from rasterio import mask
from multiprocessing.pool import Pool

from common import *


class OperationCancelledException(Exception):
    pass


def line_split(callback, HasOLnFID, in_cl, seg_length, max_ln_width, sampling_type, verbose):
    in_ln_shp = gpd.GeoDataFrame.from_file(in_cl)

    # Check the OLnFID column in data. If it is not, column will be created
    if 'OLnFID' not in in_ln_shp.columns.array:
        if BT_DEBUGGING:
            print("Cannot find {} column in input line data")

        print("New column created: {}".format('OLnFID', 'OLnFID'))
        in_ln_shp['OLnFID'] = in_ln_shp.index

    # Copy all the input line into geodataframe
    in_cl_line = gpd.GeoDataFrame.copy(in_ln_shp)

    # Prepare line for arbitrary split lines
    if sampling_type == 'ARBITRARY':
        # copy the input line into split points GoeDataframe
        in_cl_split_point = gpd.GeoDataFrame.copy(in_cl_line)

        # create empty geodataframe for split line and straight line from split points
        in_cl_split_line = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=in_ln_shp.crs)
        line_id = 0

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

            seg_i = 1
            seg_list_index = 0

            for seg in lines.geoms:
                for col in in_cl_split_point.columns.array:
                    in_cl_split_line.loc[line_id, col] = in_cl_split_point.loc[row, col]
                    in_cl_split_line.loc[line_id, 'geometry'] = seg

                if not HasOLnFID:
                    in_cl_split_line.loc[line_id, 'OLnFID'] = row

                in_cl_split_line.loc[line_id, 'OLnSEG'] = seg_i
                seg_i = seg_i + 1
                seg_list_index = seg_list_index + 1
                line_id = line_id + 1

            in_cl_split_line = in_cl_split_line.dropna(subset='geometry')

        in_cl_split_line.reset_index()
        return in_cl_split_line
    elif sampling_type == "LINE-CROSSINGS":
        # create empty geodataframe for lines
        in_cl_dissolved = gpd.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=in_ln_shp.crs)

        lines = list(line for line in in_cl_line['geometry'])
        in_cl_dissolved['geometry'] = list(shapely.ops.linemerge(lines).geoms)

        identical_segs = in_cl_dissolved.sjoin(in_cl_line, predicate='covered_by')
        identical_segs['Disso_ID'] = identical_segs.index
        columns = list(col for col in identical_segs.columns
                       if col not in ['geometry', 'index_right', 'Shape_Leng', 'Shape_Le_1', 'len'])
        identical_segs = pd.DataFrame(identical_segs[columns])
        identical_segs.reset_index()

        share_seg = in_cl_line.sjoin(in_cl_dissolved, predicate='covered_by')
        share_seg = share_seg[share_seg.duplicated('index_right', keep=False)]
        share_seg['Disso_ID'] = share_seg['index_right']

        share_seg = pd.DataFrame(share_seg[columns])
        share_seg.reset_index()

        segs_identity = pd.concat([identical_segs, share_seg])
        segs_identity.reset_index()

        for seg in range(0, len(in_cl_dissolved.index)):
            in_cl_dissolved.loc[seg, 'Disso_ID'] = seg
            common_segs = segs_identity.query("Disso_ID=={}".format(seg))
            fp_list = common_segs['OLnFID']

            for col in common_segs.columns:
                in_cl_dissolved.loc[seg, col] = common_segs.loc[common_segs.index[0], col]

            in_cl_dissolved.loc[seg, 'OLnSEG'] = seg
            in_cl_dissolved.loc[[seg], 'FP_ID'] = pd.Series([fp_list], index=in_cl_dissolved.index[[seg]])
            in_cl_dissolved.loc[[seg], 'OLnFID'] = pd.Series([fp_list], index=in_cl_dissolved.index[[seg]])

        in_cl_dissolved['Disso_ID'].astype(int)
        return in_cl_dissolved
    else:  # Return Line as input and create two columns as Primary Key
        if not HasOLnFID:
            in_cl_line['OLnFID'] = in_cl_line.index

        in_cl_line['OLnSEG'] = 0
        in_cl_line.reset_index()
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


def fill_attributes(line_args):
    # (result_identity,attr_seg_lines,area_analysis,height_analysis, in_chm, max_lin_width)
    attr_seg_line = line_args[0]
    result_identity = line_args[1]

    area_analysis = line_args[2]
    height_analysis = line_args[3]
    in_chm = line_args[4]
    max_ln_width = line_args[5]

    # determine if footprint is empty
    has_footprint = True
    if type(result_identity) is gpd.GeoDataFrame:
        if result_identity.empty:
            has_footprint = False
    elif not result_identity:
        has_footprint = False

    # if line is empty, then skip
    if attr_seg_line.empty:
        return None

    index = attr_seg_line.index[0]
    fields = ['LENGTH', 'FP_Area', 'Perimeter', 'Bearing', 'Direction', 'Sinuosity',
              'AvgWidth', 'Fragment', 'AvgHeight', 'Volume', 'Roughness']
    values = dict.fromkeys(fields, numpy.nan)
    line_feat = attr_seg_line.geometry.iloc[0]
    # default footprint by buffering
    line_buffer = line_feat.buffer(float(max_ln_width), cap_style=shapely.BufferCapStyle.flat)

    # merge result_identity
    if has_footprint:
        result_identity = result_identity.dissolve()
        fp = result_identity.iloc[0].geometry
        line_buffer = fp

    # assign common attributes
    euc_distance = find_euc_distance(line_feat)  # Euclidean distance from start to end points of segment line
    values['LENGTH'] = line_feat.length
    values['FP_Area'] = line_buffer.area
    values['Perimeter'] = line_buffer.length
    values['Bearing'] = find_bearing(attr_seg_line)
    values['Direction'] = find_direction(values['Bearing'])

    try:
        values['Sinuosity'] = line_feat.length / euc_distance
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

    if height_analysis:  # with CHM
        with rasterio.open(in_chm) as in_chm_file:
            cell_size_x = in_chm_file.transform[0]
            cell_size_y = -in_chm_file.transform[4]

            # clipped the chm base on polygon of line buffer or footprint
            clipped_chm, out_transform = rasterio.mask.mask(in_chm_file, [line_buffer], crop=True)

            # drop the ndarray to 2D ndarray
            clipped_chm = numpy.squeeze(clipped_chm, axis=0)

            # masked all NoData value cells
            clean_chm = numpy.ma.masked_where(clipped_chm == in_chm_file.nodata, clipped_chm)

            # Calculate the summary statistics from the clipped CHM
            chm_mean = numpy.ma.mean(clean_chm)
            chm_std = numpy.ma.std(clean_chm)
            chm_sum = numpy.ma.sum(clean_chm)
            chm_count = numpy.ma.count(clean_chm)
            one_cell_area = cell_size_y * cell_size_x

            sq_std_pow = 0.0
            try:
                sq_std_pow = math.pow(chm_std, 2) * (chm_count - 1) / chm_count
            except ZeroDivisionError as e:
                sq_std_pow = 0.0

            values["AvgHeight"] = chm_mean
            values["Volume"] = chm_sum * one_cell_area
            values["Roughness"] = math.sqrt(math.pow(chm_mean, 2) + sq_std_pow)
    else:  # No CHM
        # remove fields not used
        fields.remove('AvgHeight')
        values.pop('AvgHeight')

        fields.remove('Volume')
        values.pop('Volume')

        fields.remove('Roughness')
        values.pop('Roughness')

    attr_seg_line.loc[index, fields] = values
    footprint = gpd.GeoDataFrame({'geometry': [line_buffer]}, crs=attr_seg_line.crs)

    return attr_seg_line, footprint


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
        line_segments = []
        line_footprints = []
        with Pool(processes) as pool:
            step = 0
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(fill_attributes, line_args):
                if BT_DEBUGGING:
                    print('Got result: {}'.format(result), flush=True)

                if type(result[0]) is gpd.GeoDataFrame:
                    if not result[0].empty:
                        line_segments.append(result[0])

                if type(result[1]) is gpd.GeoDataFrame:
                    if not result[1].empty:
                        line_footprints.append(result[1])

                step += 1
                print('Line processed: {}'.format(step))
                print('%{}'.format(step / total_steps * 100))

    except OperationCancelledException:
        print("Operation cancelled")
        exit()

    return line_segments, line_footprints


def forest_line_attributes(callback, in_line, in_footprint, in_chm, sampling_type, seg_len,
                           ln_split_tol, max_ln_width, out_line, processes, verbose):
    # assign Tool arguments
    in_cl = in_line
    in_fp = in_footprint
    seg_len = float(seg_len)
    ln_split_tol = float(ln_split_tol)
    max_ln_width = float(max_ln_width)

    # Valid input footprint shapefile has geometry
    in_fp_shp = gpd.GeoDataFrame.from_file(in_fp)
    in_ln_shp = gpd.read_file(in_cl, rows=1)  # TODO: check projection
    in_fields = list(in_ln_shp.columns)

    # check coordinate systems between line and raster features
    try:
        with rasterio.open(in_chm) as in_raster:
            if in_fp_shp.crs.to_epsg() != in_raster.crs.to_epsg():
                print("Line and raster spatial references are not the same, please check.")
                exit()
    except Exception as e:
        print(e)

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
    attr_seg_lines = line_split(print, HasOLnFID, in_cl, seg_len, max_ln_width, sampling_type, verbose=in_args.verbose)

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
    features = execute_multiprocessing_attributes(line_args, processes)

    # Combine into one geodataframe
    if len(features) == 0:
        print('No lines found.')
        exit()

    line_segments = features[0]
    line_footprints = features[1]

    result_segments = gpd.GeoDataFrame(pd.concat(line_segments, ignore_index=True))
    result_segments.reset_index()

    result_footprints = gpd.GeoDataFrame(pd.concat(line_footprints, ignore_index=True))
    result_footprints.reset_index()

    print('Attribute processing done.')
    print('%{}'.format(80))

    # Clean the split line attribute columns
    field_list = ['geometry', 'LENGTH', 'FP_Area', 'Perimeter', 'Bearing', 'Direction',
                  'Sinuosity', 'AvgWidth', 'AvgHeight', 'Fragment', 'Volume', 'Roughness']
    field_list.extend(in_fields)
    del_list = list(col for col in result_segments.columns if col not in field_list)
    result_segments = result_segments.drop(columns=del_list)
    result_segments.reset_index()

    print('%{}'.format(90))
    print('Saving output ...')

    # Save attributed lines, was output_att_line
    result_segments.to_file(out_line)
    result_footprints.to_file(r'D:\Temp\test-ecosite\footprint_inter.shp')

    print('%{}'.format(100))


if __name__ == '__main__':
    start_time = time.time()
    print('Line Attributes started at {}'.format(time.strftime("%b %Y %H:%M:%S", time.localtime())))

    # Get tool arguments
    in_args, in_verbose = check_arguments()
    forest_line_attributes(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)

    print('Current time: {}'.format(time.strftime("%d %b %Y %H:%M:%S", time.localtime())))
    print('Line Attributes processing done in {} seconds'.format(round(time.time() - start_time, 5)))
