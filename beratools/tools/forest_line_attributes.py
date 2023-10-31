import math
import time
import pandas
import geopandas
import numpy
import scipy
import shapely
from shapely.ops import unary_union
from rasterio import mask
import argparse
import json
from multiprocessing.pool import Pool

from common import *


class OperationCancelledException(Exception):
    pass


def AttLineSplit(callback, HasOLnFID, processes, verbose, **args):
    in_ln_shp = geopandas.GeoDataFrame.from_file(args['in_line'])

    # Check the OLnFID column in data. If it is not, column will be created
    if not 'OLnFID' in in_ln_shp.columns.array:
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

    # Prepare line for arbitrary splitted lines
    if args['sampling_type'] == 'ARBITRARY':
        # loop thought all the record in input centerlines
        for row in range(0, len(in_cl_line)):
            # get geometry from record
            in_ln_feat = in_cl_line.loc[row, 'geometry']

            # get geometry's length from record
            in_line_length = in_cl_line.loc[row, 'geometry'].length

            if 0.0 <= in_line_length <= float(args["seg_len"]):
                # if input line is shorter than the set arbitrary distance, get the start and end points from input line
                points = [shapely.Point(in_ln_feat.coords[0]), shapely.Point(in_ln_feat.coords[-1])]
            else:
                # if input line is longer than the set arbitrary distance,
                # get the arbitrary distance list along input line
                distances = numpy.arange(0, in_line_length, float(args["seg_len"]))

                # append line's end point into numpy array
                if distances[-1] < in_line_length:
                    distances = numpy.append(distances, in_line_length)

                # interpolate distance list and get all the points' coordinates
                points = [in_ln_feat.interpolate(distance) for distance in distances]

            # Make sure points are snapped to input line
            points = shapely.snap(points, in_ln_feat, 0.001)

            # replace row record's line geometry of into multipoint geometry
            in_cl_splittpoint.loc[row, 'geometry'] = shapely.multipoints(points)

            # Split the input line base on the splitted points
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

                buffer_list.append(seg.buffer(float(args['Max_ln_width']), cap_style=shapely.BufferCapStyle.flat))

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
            for seg in segment_list:  # .geoms:
                for col in in_cl_splittpoint.columns.array:
                    in_cl_splitline.loc[j, col] = in_cl_splittpoint.loc[row, col]

                if seg_list_index == 0:
                    in_cl_splitline.loc[j, 'geometry'] = seg.geoms[seg_list_index]  # shapely.union_all(seg)
                elif 0 < seg_list_index < len(segment_list) - 1:
                    in_cl_splitline.loc[j, 'geometry'] = seg.geoms[1]
                else:
                    in_cl_splitline.loc[j, 'geometry'] = seg.geoms[-1]

                # if not HasOLnFID:
                if not HasOLnFID:
                    in_cl_splitline.loc[j, 'OLnFID'] = row

                in_cl_splitline.loc[j, 'OLnSEG'] = seg_i
                seg_i = seg_i + 1
                seg_list_index = seg_list_index + 1
                j = j + 1
            in_cl_splitline = in_cl_splitline.dropna(subset='geometry')

        in_cl_splitline.reset_index()
        in_cl_straightline.reset_index()
    elif args['sampling_type'] == "LINE-CROSSINGS":
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
            fp_list = list(common_segs['OLnFID'])

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

    if args["sampling_type"] == 'IN-FEATURES':
        return in_cl_line, numpy.nan

    elif args["sampling_type"] == "ARBITRARY":
        return in_cl_splitline, in_cl_straightline

    elif args["sampling_type"] == "LINE-CROSSINGS":

        return in_cl_dissolved, numpy.nan

    # elif args["sampling_type"] == "WHOLE-LINE":
    #     return in_cl_line,numpy.nan


def findDirection(bearing):
    ori = "N-S"
    if 22.5 <= bearing < 67.5 or 202.5 <= bearing < 247.5:
        ori = "NE-SW"
    elif 67.5 <= bearing < 112.5 or 247.5 <= bearing < 292.5:
        ori = "E-W"
    elif 112.5 <= bearing < 157.5 or 292.5 <= bearing < 337.5:
        ori = "NW-SE"
    return ori


def findEucDistance(in_feat):
    x1, y1 = in_feat.coords[0]
    x2, y2 = in_feat.coords[-1]
    return scipy.spatial.distance.euclidean([x1, y1], [x2, y2])


def findBearing(seg):  # Geodataframe
    line_coords = shapely.geometry.mapping(seg[['geometry']])['features'][0]['geometry']['coordinates']

    x1, y1 = line_coords[0]
    x2, y2 = line_coords[-1]
    dx = x2 - x1
    dy = y2 - y1

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


def Fill_Attributes(line_args):  # (result_identity,Att_seg_lines,areaAnalysis,heightAnalysis,**args):
    result_identity = line_args[0]
    Att_seg_lines = line_args[1]
    areaAnalysis = line_args[2]
    heightAnalysis = line_args[3]
    args = line_args[4]

    if type(result_identity) is geopandas.geodataframe.GeoDataFrame:
        if result_identity.empty:
            return result_identity
    elif not result_identity:
        return result_identity

    if heightAnalysis:  # with CHM
        with rasterio.open(args['in_chm']) as in_chm_file:
            cell_size_x = in_chm_file.transform[0]
            cell_size_y = -in_chm_file.transform[4]

            for index in result_identity.index:
                fp = result_identity.iloc[index].geometry
                if 'Disso_ID' in result_identity.columns.array and 'Disso_ID' in result_identity.columns.array:
                    query_str = "Disso_ID=={} ".format(result_identity.iloc[index].Disso_ID)
                elif 'OLnFID' in result_identity.columns.array and 'OLnSEG' in result_identity.columns.array:
                    query_str = "OLnFID=={} and OLnSEG=={}".format(result_identity.iloc[index].OLnFID,
                                                                   result_identity.iloc[index].OLnSEG)
                else:
                    query_str = "OLnFID=={}".format(result_identity.iloc[index].OLnFID)

                selected_segLn = Att_seg_lines.query(query_str)

                # Check if query result is not empty, if empty input identity footprint will be skipped
                if len(selected_segLn.index) > 0:
                    line_feat = selected_segLn['geometry'].iloc[0]

                    # if the selected seg do not have identity footprint geometry
                    if shapely.is_empty(fp):
                        # use the buffer from the segment line
                        line_buffer = shapely.buffer(line_feat, float(args['Max_ln_width']))
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
                    eucDistance = findEucDistance(line_feat)

                    # Calculate the summary statistics from the clipped CHM
                    chm_mean = numpy.ma.mean(clean_chm)
                    chm_std = numpy.ma.std(clean_chm)
                    chm_sum = numpy.ma.sum(clean_chm)
                    chm_count = numpy.ma.count(clean_chm)
                    OnecellArea = cell_size_y * cell_size_x

                    try:
                        sqStdPop = math.pow(chm_std, 2) * (chm_count - 1) / chm_count
                    except ZeroDivisionError as e:
                        sqStdPop = 0.0

                    # writing result to feature's attributes
                    result_identity.loc[index, 'LENGTH'] = line_feat.length
                    result_identity.loc[index, 'FP_Area'] = result_identity.loc[index, 'geometry'].area
                    result_identity.loc[index, 'Perimeter'] = result_identity.loc[index, 'geometry'].length
                    result_identity.loc[index, 'Bearing'] = findBearing(selected_segLn)
                    result_identity.loc[index, 'Direction'] = findDirection(result_identity.loc[index, 'Bearing'])
                    try:
                        result_identity.loc[index, 'Sinuosity'] = line_feat.length / eucDistance
                    except ZeroDivisionError as e:
                        result_identity.loc[index, 'Sinuosity'] = numpy.nan
                    try:
                        result_identity.loc[index, "AvgWidth"] = result_identity.loc[
                                                                     index, 'FP_Area'] / line_feat.length
                    except ZeroDivisionError as e:
                        result_identity.loc[index, "AvgWidth"] = numpy.nan
                    try:
                        result_identity.loc[index, "Fragment"] = result_identity.loc[index, 'Perimeter'] / \
                                                                 result_identity.loc[index, 'FP_Area']
                    except ZeroDivisionError as e:
                        result_identity.loc[index, "Fragment"] = numpy.nan

                    result_identity.loc[index, "AvgHeight"] = chm_mean
                    result_identity.loc[index, "Volume"] = chm_sum * OnecellArea
                    result_identity.loc[index, "Roughness"] = math.sqrt(math.pow(chm_mean, 2) + sqStdPop)

                del selected_segLn

    else:
        # No CHM
        for index in result_identity.index:

            query_str = "OLnFID=={} and OLnSEG=={}".format(result_identity.iloc[index].OLnFID,
                                                           result_identity.iloc[index].OLnSEG)
            selected_segLn = Att_seg_lines.query(query_str)

            if len(selected_segLn.index) > 0:
                line_feat = selected_segLn['geometry'].iloc[0]
                eucDistance = findEucDistance(line_feat)
                result_identity.loc[index, 'LENGTH'] = line_feat.length
                result_identity.loc[index, 'FP_Area'] = result_identity.loc[index, 'geometry'].area
                result_identity.loc[index, 'Perimeter'] = result_identity.loc[index, 'geometry'].length
                result_identity.loc[index, 'Bearing'] = findBearing(selected_segLn)
                result_identity.loc[index, 'Direction'] = findDirection(result_identity.loc[index, 'Bearing'])
                try:
                    result_identity.loc[index, 'Sinuosity'] = line_feat.length / eucDistance
                except ZeroDivisionError as e:
                    result_identity.loc[index, 'Sinuosity'] = numpy.nan
                try:
                    result_identity.loc[index, "AvgWidth"] = result_identity.loc[index, 'FP_Area'] / line_feat.length
                except ZeroDivisionError as e:
                    result_identity.loc[index, "AvgWidth"] = numpy.nan
                try:
                    result_identity.loc[index, "Fragment"] = result_identity.loc[index, 'Perimeter'] / \
                                                             result_identity.loc[index, 'FP_Area']
                except ZeroDivisionError as e:
                    result_identity.loc[index, "Fragment"] = numpy.nan

                result_identity.loc[index, "AvgHeight"] = numpy.nan
                result_identity.loc[index, "Volume"] = numpy.nan
                result_identity.loc[index, "Roughness"] = numpy.nan

            del selected_segLn

    return result_identity


def identity_polygon(line_args):
    in_cl_buffer = line_args[0][['geometry', 'OLnFID', 'OLnSEG']]
    in_fp_polygon = line_args[1]
    if not 'OLnSEG' in in_fp_polygon.columns.array:
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

    return identity


if __name__ == '__main__':
    start_time = time.time()
    print('Starting attribute Forest Line Attributes \n@ {}'.format(
        time.strftime("%d %b %Y %H:%M:%S", time.localtime())))

    # Get tool arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=json.loads)
    parser.add_argument('-p', '--processes')
    parser.add_argument('-v', '--verbose')
    args = parser.parse_args()
    if args.verbose == 'True':
        verbose = True
    else:
        verbose = False

    # assign Tool arguments
    in_cl = args.input["in_line"]
    in_fp = args.input["in_footprint"]
    in_chm = args.input["in_chm"]
    sampling_type = args.input["sampling_type"]
    seg_len = float(args.input["seg_len"])
    ln_split_Tol = float(args.input["ln_split_Tol"])
    max_ln_width = float(args.input["max_ln_width"])

    # Valid input footprint shapefile has geometry
    in_fp_shp = geopandas.GeoDataFrame.from_file(in_fp)
    in_ln_shp = geopandas.GeoDataFrame.from_file(in_cl)

    # check coordinate systems between line and raster features
    with rasterio.open(in_chm) as in_raster:
        if in_fp_shp.crs.to_epsg() != in_raster.crs.to_epsg():
            print("Line and raster spatial references are not the same, please check.")
            exit()

    del in_raster

    if len(in_fp_shp) <= 0:
        print('There is no footprint provided, a buffer from the input line is used instead')
        areaAnalysis = False
        HasOLnFID = False
    else:
        areaAnalysis = True
        # Check OLnFID column in input centre line:
        if 'OLnFID' in in_fp_shp.columns.array:
            # make sure the 'OLnFID' column value as integer
            in_fp_shp[['OLnFID']] = in_fp_shp[['OLnFID']].astype(int)
            HasOLnFID = True
        elif 'Fr_Orig_ln' in in_fp_shp.columns.array:
            in_fp_shp = in_fp_shp.rename(columns={'Fr_Orig_ln': 'OLnFID'})
            in_fp_shp[['OLnFID']] = in_fp_shp[['OLnFID']].astype(int)
            HasOLnFID = True
        else:  # future code: get the original centerlines ID and write into subset of centerlines column "OLnFID"
            HasOLnFID = False
            print("Please prepare original line feature's ID (FID) ...")
            exit()

        if 'Fr_Seg_Ln' in in_fp_shp.columns.array:
            in_fp_shp = in_fp_shp.rename(columns={'Fr_Seg_Ln': 'OLnSEG'})
            in_fp_shp[['OLnSEG']] = in_fp_shp[['OLnSEG']].astype(int)

        elif "OLnSEG" not in in_fp_shp.columns.array:
            in_fp_shp['OLnSEG'] = 0
    try:
        with rasterio.open(in_chm) as in_CHM:
            heightAnalysis = True
        del in_CHM
    except Exception as error_in_CHM:
        print(error_in_CHM)
        heightAnalysis = False
        exit()

    # Only process the following SamplingType
    SamplingType = ["IN-FEATURES", "LINE-CROSSINGS", "ARBITRARY"]
    if args.input["sampling_type"] not in SamplingType:
        print("SamplingType is not correct, please verify it.")
        exit()

    # footprintField = FileToField(fileBuffer)
    print("Preparing line segments...")
    # Segment lines
    print("Input_Lines: {}".format(in_cl))

    # Return splitted lines with two extra columns:['OLnFID','OLnSEG'] or return Dissolved whole line
    Att_seg_lines, Straight_lines = AttLineSplit(print, HasOLnFID, processes=int(args.processes),
                                                 verbose=verbose, **args.input)

    print('%{}'.format(10))

    if args.input["sampling_type"] != "LINE-CROSSINGS":
        if areaAnalysis:
            print('%{}'.format(20))

            # Buffer seg straight line or whole ln for identify footprint polygon
            if isinstance(Straight_lines, geopandas.GeoDataFrame):
                in_cl_buffer = geopandas.GeoDataFrame.copy(Straight_lines)
                in_cl_buffer['geometry'] = in_cl_buffer.buffer(max_ln_width, cap_style=shapely.BufferCapStyle.flat)
            else:
                in_cl_buffer = geopandas.GeoDataFrame.copy(Att_seg_lines)
                in_cl_buffer['geometry'] = in_cl_buffer.buffer(max_ln_width, cap_style=shapely.BufferCapStyle.flat)
        else:
            in_cl_buffer = geopandas.GeoDataFrame.copy(Att_seg_lines)
            in_cl_buffer['geometry'] = in_cl_buffer.buffer(max_ln_width, cap_style=shapely.BufferCapStyle.flat)
            in_fp_shp = in_cl_buffer

    else:  # LINE-CROSSINGS
        if areaAnalysis:
            # load in the footprint shapefile
            query_col = 'OLnFID'

            # Query footprints based on OLnFID or Fr_Orig_Ln and prepare a new in fp dataframe
            all_matched_fp = []

            for line_index in Att_seg_lines.index:
                fp_list = Att_seg_lines.iloc[line_index].FP_ID
                selected_fp = in_fp_shp.query(query_col + ' in @fp_list')
                if len(selected_fp) > 1:
                    dissolved_fp = geopandas.GeoDataFrame.dissolve(selected_fp)
                elif len(selected_fp) == 1:
                    dissolved_fp = selected_fp
                else:
                    print('No match!!!')
                    dissolved_fp = geopandas.GeoDataFrame()
                dissolved_line=""
                for line in fp_list:
                    dissolved_line=dissolved_line+str(line)+" "

                dissolved_fp = dissolved_fp.assign(Dis_OLnFID=[dissolved_line])
                dissolved_fp = dissolved_fp.assign(Disso_ID=[line_index])
                all_matched_fp.append(dissolved_fp)
            in_fp_shp = geopandas.GeoDataFrame(pandas.concat(all_matched_fp))
            in_fp_shp=in_fp_shp.reset_index(drop=True)
            print('%{}'.format(20))

            # buffer whole ln for identify footprint polygon
            in_cl_buffer = geopandas.GeoDataFrame.copy(Att_seg_lines)
            in_cl_buffer['geometry'] = in_cl_buffer.buffer(max_ln_width, cap_style=shapely.BufferCapStyle.flat)
        else:
            in_cl_buffer = geopandas.GeoDataFrame.copy(Att_seg_lines)
            in_cl_buffer['geometry'] = in_cl_buffer.buffer(max_ln_width, cap_style=shapely.BufferCapStyle.flat)
            in_fp_shp = in_cl_buffer

    print("Line segments are prepared.")

    # Create a emtpy geodataframe for identity polygon
    result_identity = geopandas.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=in_cl_buffer.crs)
    result_Att = geopandas.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=in_cl_buffer.crs)
    print(
        "There are {} original footprint to be identified by {} segment lines ...".format(len(in_fp_shp.index),
                                                                                          len(in_cl_buffer)))
    print('%{}'.format(30))

    # Prepare line parameters for multiprocessing
    line_args = []

    if args.input["sampling_type"] == 'LINE-CROSSINGS':
        query_col1 = 'Disso_ID'
    elif args.input["sampling_type"] == 'ARBITRARY':
        query_col1 = 'OLnFID'

    elif args.input["sampling_type"] == 'IN-FEATURES':
        query_col1 = 'OLnFID'

    elif 'Fr_Orig_ln' in in_fp_shp.columns.array:
        query_col1 = 'Fr_Orig_ln'

    else:
        print('Could not match footprint to split lines.  Please check primary key: Disso_ID or OlnFID, or Fr_Orig_Ln')
        exit()

    # prepare line args: list of line buffer and fp polygon
    if args.input["sampling_type"] != 'LINE-CROSSINGS':
        in_fp_shp['OLnFID'] = in_fp_shp['OLnFID'].astype(int)
        in_fp_shp['OLnSEG'] = in_fp_shp['OLnSEG'].astype(int)
        in_cl_buffer['OLnFID'] = in_cl_buffer['OLnFID'].astype(int)
        in_cl_buffer['OLnSEG'] = in_cl_buffer['OLnSEG'].astype(int)
    else:

        in_fp_shp['Disso_ID'] = in_fp_shp['Disso_ID'].astype(int)
        in_fp_shp['OLnFID'] = in_fp_shp['OLnFID'].astype(int)
        in_cl_buffer['Disso_ID'] = in_cl_buffer['Disso_ID'].astype(int)
        in_cl_buffer['OLnSEG'] = in_cl_buffer['OLnSEG'].astype(int)

    for row in in_cl_buffer.index:
        list_item = []

        list_item.append(in_cl_buffer.iloc[[row]])
        list_item.append(in_fp_shp.query(query_col1 + "=={}".format(in_cl_buffer.loc[row, query_col1])))

        line_args.append(list_item)

    # Sequence processing identity polygon
    # total_steps = len(line_args)
    # features = []
    # for row in range(0,total_steps):
    #     features.append(identity_polygon(line_args[row]))

    # Multiprocessing identity polygon
    try:
        total_steps = len(line_args)
        features = []
        with Pool(processes=int(args.processes)) as pool:
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

    print("Prepare for filling attributes ...")
    # prepare list of result_identity, Att_seg_lines, areaAnalysis, heightAnalysis, args.input
    line_args = []
    for index in range(0, len(features)):
        list_item = []
        list_item.append(features[index])
        list_item.append(Att_seg_lines)
        list_item.append(areaAnalysis)
        list_item.append(heightAnalysis)
        list_item.append(args.input)
        line_args.append(list_item)

        # ##Linear attributes
    print("Adding attributes ...")
    print('%{}'.format(60))

    # Multiprocessing identity polygon
    try:
        total_steps = len(line_args)
        features = []
        with Pool(processes=int(args.processes)) as pool:
            step = 0
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(Fill_Attributes, line_args):
                if BT_DEBUGGING:
                    print('Got result: {}'.format(result), flush=True)

                if type(result) is geopandas.geodataframe.GeoDataFrame:
                    if not result.empty:
                        features.append(result)

                step += 1
                print('%{}'.format(step / total_steps * 100))

    except OperationCancelledException:
        print("Operation cancelled")
        exit()

    # Combine identity polygons into once geodataframe
    result_Att = geopandas.GeoDataFrame(pandas.concat(features, ignore_index=True))
    result_Att.reset_index()
    print('Attribute done.')
    print('%{}'.format(80))

    footprint_att = pandas.DataFrame(result_Att.drop(columns='geometry'))

    # Clean the split line attribute columns
    del_list = list(col for col in footprint_att.columns if
                    col not in ['OLnFID', 'OLnSEG', 'geometry', 'LENGTH', 'FP_Area', 'Perimeter', 'Bearing',
                                'Direction', 'Sinuosity', 'AvgWidth', 'AvgHeight', "Fragment", "Volume",
                                "Roughness", 'Disso_ID', 'FP_ID'])
    footprint_att = footprint_att.drop(columns=del_list)
    footprint_att.reset_index()

    # Merging the cleaned split line with identity dataframe
    if args.input["sampling_type"] != "LINE-CROSSINGS":
        output_att_line = Att_seg_lines.merge(footprint_att, how='left', on=['OLnFID', 'OLnSEG'])
    else:
        Att_seg_lines['FP_ID'] = Att_seg_lines['FP_ID'].apply(lambda x: str(x))
        Att_seg_lines['OLnFID'] = Att_seg_lines['OLnFID'].apply(lambda x: str(x))
        footprint_att['OLnFID'] = footprint_att['OLnFID'].apply(lambda x: str(x))
        output_att_line = Att_seg_lines.merge(footprint_att, how='left', on=['Disso_ID'])

    print('%{}'.format(90))
    print('Saving output ...')

    # Save attributed lines
    geopandas.GeoDataFrame.to_file(output_att_line, args.input['out_line'])

    print('%{}'.format(100))
    print('Current time: {}'.format(time.strftime("%d %b %Y %H:%M:%S", time.localtime())))
    print('Line Attributes processing done in {} seconds)'.format(round(time.time() - start_time, 5)))
