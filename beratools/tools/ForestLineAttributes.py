import math
import time
import os
import pandas
import geopandas
import numpy
import scipy
import shapely
from shapely.ops import unary_union
import rasterio
from rasterio import  mask
import argparse
import json
from multiprocessing.pool import Pool

from common import *


class OperationCancelledException(Exception):
    pass

def FileToField(filen):
    return ("FID_"+os.path.basename(os.path.splitext(filen)[0]).replace(" ", "_"))[:10]

def PathFileName (path):
    return os.path.basename(path)

def FieldNameList(fc):
    #return a list of column name from shapefile expect "geometry"
    fieldlist=[]
    if isinstance(fc,geopandas.GeoDataFrame):
        list_of_columns=fc.columns.array
    else:
        list_of_columns=geopandas.read_file(fc).columns.array
    for field in list_of_columns:
        if field !="geometry":
            fieldlist.append(field)

    return fieldlist

def HasField(fc, fi):
    #Check column name
    fieldlist=FieldNameList(fc)
    if fi in fieldlist:
        print("column: {} is found".format(fi))
        return True
    elif fi == 'CorridorTh':
        shapefile = geopandas.GeoDataFrame.from_file(fc)
        for row in range(0,len(shapefile)):
            shapefile.loc[row,fi]=3.0
        shapefile.to_file(fc)
        print("Warning: There is no field named {} in the input data".format('CorridorTh'))
        print("Field: '{}' is added and default threshold (i.e.3) is adopted".format('CorridorTh'))
        return True
    else:
        print("Warning: There is no field named {} in the input data".format(fi))

        return False

def SpliteLine2(**args):

    shapefile = geopandas.GeoDataFrame.from_file(args['in_cl'])
    shapefile_proj = shapefile.crs

    KeepFieldName=[]
    for col_name in shapefile.columns:
        if col_name != 'geometry':
            KeepFieldName.append(col_name)

    list_of_segment=[]

    i = 0
    if len(shapefile)>0:  # process when shapefile is not an empty feature class

        for row in range(0,len(shapefile)):
            feat=shapefile.loc[row].geometry # creates a geometry object
            # Split every segments from line
            segment_list = (list(map(shapely.geometry.LineString, zip(feat.coords[:-1], feat.coords[1:]))))
            feature_attributes = {}
            if args['proc_seg']: # process every segments

                for seg in segment_list:
                    feature_attributes = {}
                    feature_attributes['FID'] = i
                    feature_attributes['OID'] = row
                    feature_attributes['Total_Seg'] = len(segment_list)
                    feature_attributes['Seg_leng'] = seg.length
                    feature_attributes['geometry'] = seg
                    feature_attributes['Proj_crs'] = shapefile_proj

                    for col_name in KeepFieldName:
                        feature_attributes[col_name] = shapefile.loc[row, col_name]

                    list_of_segment.append(feature_attributes)
                    i = i + 1


            else: # process on original lines
                feature_attributes['FID']=i
                feature_attributes['OID']= row
                feature_attributes['No_of_Seg']= len(segment_list)
                feature_attributes['Seg_leng']= feat.length
                feature_attributes['geometry']= feat
                feature_attributes['Proj_crs'] = shapefile_proj
                for col_name in KeepFieldName:
                    feature_attributes[col_name]= shapefile.loc[row,col_name]
                list_of_segment.append(feature_attributes)
                i = i + 1

        print("There are {} lines to be processed.".format(len(list_of_segment)))

        # return a list of features Dictionary
        return list_of_segment

    else:
        print("Input line feature is corrupted, exit!")
        exit()


def lineprepare(callback,**args):
    # in_cl, in_CanopyR, in_CostR, CorridorTh_field, CorridorTh_value, Max_ln_width, Exp_Shk_cell,
    # proc_seg, out_footprint

    CorridorTh_value = float(args['CorridorTh_value'])
    Max_ln_width = float(args['Max_ln_width'])
    Exp_Shk_cell = int(args['Exp_Shk_cell'])
    print("Preparing Lines............")
    print("Process every segments: {}".format(args['proc_seg']))
    if args['proc_seg']=='True':
        args['proc_seg']=True
    else:
        args['proc_seg']=False

    #Open shapefile -input centerlines and check existing Corridor threshold field
    # if threshold field is not found, it will be created and populate value of '3'
    print('Check {} field in input feature.'.format(args['CorridorTh_field']))
    if HasField(args['in_cl'],args['CorridorTh_field']):
        pass

    #get the list of orginal columns names
    fieldlist_col=FieldNameList(args['in_cl'])

    #Splite the input centerline and return a list of geodataframe
    print('Split line process.............')
    list_dict_segment_all=SpliteLine2(**args.input)

    # Add tools arguments into geodataframe record
    for record in list_dict_segment_all:
        record['in_CanopyR']=args['in_CanopyR']
        record['in_CostR']=args['in_CostR']
        record['CorridorTh_field']=args['CorridorTh_field']
        record['CorridorTh_value']=CorridorTh_value
        record['Max_ln_width']=Max_ln_width
        record['Exp_Shk_cell']=Exp_Shk_cell
        record['proc_seg']= args['proc_seg']
        record['out_footprint']=args['out_footprint']
        record['Orgi_col']=fieldlist_col


    # return list of geodataframe represents each line or segment
    return list_dict_segment_all


def AttLineSplit(callback,processes,verbose, **args):

    in_ln_shp=geopandas.GeoDataFrame.from_file(args['in_cl'])

    #Check OLnFID column in input centre line:
    if 'OLnFID' in in_ln_shp.columns.array:
        # make sure the 'OLnFID' column value as integer
        in_ln_shp[['OLnFID']]=in_ln_shp[['OLnFID']].astype(int)
        HasOLnFID = True
    elif 'Fr_Orig_ln' in in_ln_shp.columns.array:
        geopandas.GeoDataFrame.rename()
        in_ln_shp.rename(columns={'Fr_Orig_ln':'OLnFID'})
        in_ln_shp[['OLnFID']] = in_ln_shp[['OLnFID']].astype(int)
        HasOLnFID = True
    else: #ToDo:  code: get the orginal centerlines ID and write into subset of centerlines column "OLnFID"
        HasOLnFID = False
        print("Please prepare original line feature's ID (FID) ....")
        exit()

    # Copy all the input line into geodataframe
    in_cl_line = geopandas.GeoDataFrame.copy(in_ln_shp)

    # copy the input line into split points GoeDataframe
    in_cl_splittpoint=geopandas.GeoDataFrame.copy(in_cl_line)

    #create empty geodataframe for split line and straight line from split points
    in_cl_splitline=geopandas.GeoDataFrame(columns=['geometry'],geometry='geometry',crs=in_ln_shp.crs)
    in_cl_straightline = geopandas.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=in_ln_shp.crs)

    # Generate points along merged input line (with start/ end point) base on user Segment Length
    i=0
    j=0
    #Prepare line for arbitrary splited lines #ToDo remove LINE-CROSSINGS
    if args['sampling_type']=='ARBITRARY'or args['sampling_type'] == "LINE-CROSSINGS":
        # loop thought all the record in input centerlines
        for row in range(0, len(in_cl_line)):
            # get geometry from record
            in_ln_feat=in_cl_line.loc[row, 'geometry']
            # get geometry's length from record
            in_line_length=in_cl_line.loc[row, 'geometry'].length
            if  0.0<=in_line_length <=float(args["seg_len"]):
                #if input line is shoter than the set arbitrary distance, get the start and end points from input line
                points=[shapely.Point(in_ln_feat.coords[0]),shapely.Point(in_ln_feat.coords[-1])]
            else:
                # if input line is longer than the set arbitrary distance, get the arbitrary distance list along input line
                distances = numpy.arange(0,in_line_length, float(args["seg_len"]))
                # append line's end point into numpy array
                if distances[-1]<in_line_length:
                    distances=numpy.append(distances,in_line_length)
                #interpolate distance list and get all the points' coordinates
                points = [in_ln_feat.interpolate(distance) for distance in distances]


            #Make sure points are snapped to input line
            points = shapely.snap(points,in_ln_feat, 0.001)

            # replace row record's line geometry of into multipoint geometry
            in_cl_splittpoint.loc[row, 'geometry'] = shapely.multipoints(points)

            # Split the input line base on the splited points
            #extract points coordinates into list of point Geoseries
            listofpoint=shapely.geometry.mapping(in_cl_splittpoint.loc[row,'geometry'])['coordinates']

            #Generate split lines (straight line) from points
            straight_ln = (list(map(shapely.geometry.LineString, zip(shapely.LineString(listofpoint).coords[:-1],
                                                                      shapely.LineString(listofpoint).coords[1:]))))
            seg_i = 1

            buffer_list=[]
            # in_ln_buffer=geopandas.GeoDataFrame(columns=['geometry'],geometry='geometry',crs=in_ln_shp.crs)

            for seg in straight_ln:
                for col in in_cl_splittpoint.columns.array:
                    in_cl_straightline.loc[i,col]=in_cl_splittpoint.loc[row,col]
                in_cl_straightline.loc[i,'geometry']=seg

                buffer_list.append(seg.buffer(float(args['Max_ln_width']),cap_style=2))

                if not HasOLnFID:
                    in_cl_straightline.loc[i,'OLnFID']=row

                in_cl_straightline.loc[i,'OLnSEG']=seg_i
                seg_i=seg_i+1
                i=i+1


            # Split the input lines base on buffer
            segment_list=[]
            for polygon in buffer_list:
                segment_list.append(shapely.ops.split(in_cl_line.loc[row, 'geometry'],polygon))

            # Split the input lines base on points (Alternative)
            # snap_line_to_pt=shapely.ops.snap(in_cl_line.loc[row, 'geometry'],in_cl_splittpoint.loc[row, 'geometry'],
            #                                  tolerance= (float(1.0e-12)))
            # segment_list = shapely.ops.split(snap_line_to_pt, in_cl_splittpoint.loc[row, 'geometry'])

            seg_i=1
            seg_list_index = 0
            for seg in segment_list:#.geoms:

                for col in in_cl_splittpoint.columns.array:
                    in_cl_splitline.loc[j,col]=in_cl_splittpoint.loc[row,col]

                if seg_list_index==0:
                    in_cl_splitline.loc[j,'geometry']=seg.geoms[seg_list_index]# shapely.union_all(seg)
                elif 0<seg_list_index<len(segment_list)-1:
                    in_cl_splitline.loc[j, 'geometry'] = seg.geoms[1]
                else:
                    in_cl_splitline.loc[j, 'geometry'] = seg.geoms[-1]

                # if not HasOLnFID:
                if not HasOLnFID:
                    in_cl_splitline.loc[j, 'OLnFID'] = row

                in_cl_splitline.loc[j,'OLnSEG']=seg_i
                seg_i=seg_i+1
                seg_list_index=seg_list_index+1
                j=j+1
            in_cl_splitline=in_cl_splitline.dropna(subset='geometry')

        in_cl_splitline.reset_index()
        in_cl_straightline.reset_index()
   #ToDo prepare line for Line Crossings
    # elif args['sampling_type'] == "LINE-CROSSINGS":
    #     # create empty geodataframe for lines
    #     in_cl_dissolved = geopandas.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=in_ln_shp.crs)
    #
    #     lines=list(line for line in in_cl_line['geometry'])
    #     in_cl_dissolved['geometry']=list(shapely.ops.linemerge(lines).geoms)
    #
    #     #debugging save
    #     # geopandas.GeoDataFrame.to_file(in_cl_dissolved,r"D:\Maverick\BERATool_Test_Data\LFP_result1\tool_in_cl_dissolved.shp")
    #     #
    #     identical_segs=in_cl_dissolved.sjoin(in_cl_line,predicate='covered_by')
    #     identical_segs['Disso_ID'] = identical_segs.index
    #     identical_segs = pandas.DataFrame(identical_segs[['OLnFID', 'Disso_ID']])
    #     identical_segs.reset_index()
    #
    #
    #     # # debugging save
    #     # geopandas.GeoDataFrame.to_file(identical_segs,r"D:\Maverick\BERATool_Test_Data\LFP_result1\tool_in_cl_identical_segs.shp")
    #
    #     share_seg = in_cl_line.sjoin(in_cl_dissolved, predicate='covered_by')
    #     share_seg=share_seg[share_seg.duplicated('index_right',keep=False)]
    #     share_seg['Disso_ID']=share_seg['index_right']
    #     share_seg=pandas.DataFrame(share_seg[['OLnFID','Disso_ID']])
    #     share_seg.reset_index()
    #
    #     seg_identity=pandas.DataFrame.append(identical_segs,share_seg)
    #     seg_identity.reset_index()
    #
    #     # debugging save
    #     # pandas.DataFrame.to_csv(seg_identity,
    #     #                                r"D:\Maverick\BERATool_Test_Data\LFP_result1\tool_in_cl_seg_identity.csv")
    #
    #     for seg in range(0,len(in_cl_dissolved.index)):
    #         common_segs=seg_identity.query("Disso_ID=={}".format(seg))
    #         for each_seg in range(0,len(common_segs.index)):
    #             common_fp=in_ln_shp.query('OLnFID=={}'.format(common_segs.loc[each_seg,'OLnFID']))
    #             common_fp.append(common_fp)
    #         # debugging save
    #         geopandas.GeoDataFrame.to_file(common_fp,
    #                                        r"D:\Maverick\BERATool_Test_Data\LFP_result1\tool_common_fp.shp")
    #
    #
    #
    #     exit()
        # for row in range(0, len(in_cl_dissolved)):
        #     in_ln_feat = in_cl_line.loc[row, 'geometry']
        #     # in_line_length = in_cl_line.loc[row, 'geometry'].length
        #     points = [shapely.Point(in_ln_feat.coords[0]), shapely.Point(in_ln_feat.coords[-1])]
        #
        #     # Make sure points are snapped to input line
        #     points = shapely.snap(points, in_ln_feat, 0.01)
        #
        #     # replace row record's line geometry of into multipoint geometry
        #     in_cl_splittpoint.loc[row, 'geometry'] = shapely.multipoints(points)

            # Split the input line base on the splited points
            # extract points coordinates into list of point Geoseries
            # listofpoint = shapely.geometry.mapping(in_cl_splittpoint.loc[row, 'geometry'])['coordinates']


            # segment_list = shapely.ops.split(in_cl_line.loc[row, 'geometry'], in_cl_splittpoint.loc[row, 'geometry'])


            # seg_i = 1
            # for seg in segment_list.geoms:
            #     for col in in_cl_splittpoint.columns.array:
            #         in_cl_splitline.loc[j, col] = in_cl_splittpoint.loc[row, col]
            #     in_cl_splitline.loc[j, 'geometry'] = seg
            #     if not HasOLnFID:
            #         in_cl_straightline.loc[i, 'OLnFID'] = row
            #     in_cl_splitline.loc[j, 'OLnSEG'] = seg_i
            #     seg_i = seg_i + 1
            #
            #     j = j + 1
    else: #Return Line as input and create two columns as Primary Key
        if not HasOLnFID:
            in_cl_line['OLnFID'] = in_cl_line.index
        in_cl_line['OLnSEG'] = 0
        in_cl_line.reset_index()

    # debugging save
    # geopandas.GeoDataFrame.to_file(in_cl_splittpoint,
    #                                r"D:\Maverick\BERATool_Test_Data\LFP_result1\tool_in_cl_splittpoint.shp")
    # geopandas.GeoDataFrame.to_file(in_cl_splitline,
    #                                r"D:\Maverick\BERATool_Test_Data\LFP_result1\tool_in_cl_splitline.shp")
    # geopandas.GeoDataFrame.to_file(in_cl_line,
    #                                r"D:\Maverick\BERATool_Test_Data\LFP_result1\tool_in_cl_line.shp")
    # geopandas.GeoDataFrame.to_file(in_cl_straightline,
    #                                r"D:\Maverick\BERATool_Test_Data\LFP_result1\tool_cl_straightline.shp")


    if args["sampling_type"] == 'IN-FEATURES':
        return in_cl_line,numpy.nan

    elif args["sampling_type"] == "ARBITRARY":
        return in_cl_splitline,in_cl_straightline

    elif args["sampling_type"] == "LINE-CROSSINGS":

        return in_cl_splitline,in_cl_straightline

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
    return scipy.spatial.distance.euclidean([x1,y1],[x2,y2])

def findBearing(seg): # Geodataframe

    line_coords=shapely.geometry.mapping(seg[['geometry']])['features'][0]['geometry']['coordinates']

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
    elif dx>0.0:
        angle = math.degrees(math.atan(dy / dx))
        bearing=90.0-angle
    elif dx<0.0:
        angle = math.degrees(math.atan(dy / dx))
        bearing=270.0-angle

    return bearing

def Fill_Attributes(result_identity,Att_seg_lines,areaAnalysis,heightAnalysis,**args):
    if heightAnalysis: #with CHM
        with rasterio.open(args['in_CHMR']) as in_CHM:

            cell_size_x = in_CHM.transform[0]
            cell_size_y = -in_CHM.transform[4]

            print("Attribute footprints....".format(len(result_identity.index)))
            for index in result_identity.index:

                fp=result_identity.iloc[index].geometry
                query_str = "OLnFID=={} and OLnSEG=={}".format(result_identity.iloc[index].OLnFID,
                                                           result_identity.iloc[index].OLnSEG)
                selected_segLn = Att_seg_lines.query(query_str)

                # Check if query result is not empty, if empty input identity footprint will be skipped
                if len(selected_segLn.index) > 0:
                    line_feat = selected_segLn['geometry'].iloc[0]
                    #if the selected seg has not identity footprint gemoetry
                    if shapely.is_empty(fp):
                       #use the buffer from the segment line
                        line_buffer=shapely.buffer(line_feat,float(args['Max_ln_width']))
                    else:
                        # if identity footprint has geometry, use as a buffer area
                        line_buffer=fp
                    # clipped the chm base on polygon of line buffer or footprint
                    clipped_chm, out_transform = rasterio.mask.mask(in_CHM, [line_buffer], crop=True)
                    # drop the ndarray to 2D ndarray
                    clipped_chm = numpy.squeeze(clipped_chm, axis=0)
                    #masked all NoData value cells
                    clean_chm = numpy.ma.masked_where(clipped_chm == in_CHM.nodata, clipped_chm)
                    # calculate the Euclidean distance from start to end points of segment line
                    eucDistance=findEucDistance(line_feat)
                    # Calculate the summary statistics from the clipped CHM
                    chm_mean=numpy.ma.mean(clean_chm)
                    chm_std=numpy.ma.std(clean_chm)
                    chm_sum=numpy.ma.sum(clean_chm)
                    chm_count=numpy.ma.count(clean_chm)
                    cellArea=cell_size_y*cell_size_x
                    try:
                        sqStdPop = math.pow(chm_std, 2) * (chm_count - 1) / chm_count
                    except ZeroDivisionError as e:
                        sqStdPop = 0.0

                    #writing result to feature's attributes
                    result_identity.loc[index, 'LENGTH'] = line_feat.length
                    result_identity.loc[index, 'FP_Area'] = result_identity.loc[index, 'geometry'].area
                    result_identity.loc[index, 'Perimeter'] = result_identity.loc[index, 'geometry'].length
                    result_identity.loc[index, 'Bearing'] = findBearing(selected_segLn)
                    result_identity.loc[index, 'Direction'] = findDirection(result_identity.loc[index, 'Bearing'])
                    try:
                        result_identity.loc[index, 'Sinuosity']=line_feat.length / eucDistance
                    except ZeroDivisionError as e:
                        result_identity.loc[index, 'Sinuosity'] = numpy.nan
                    try:
                        result_identity.loc[index, "AvgWidth"] =  result_identity.loc[index, 'FP_Area']/ line_feat.length
                    except ZeroDivisionError as e:
                        result_identity.loc[index, "AvgWidth"]=numpy.nan
                    try:
                        result_identity.loc[index, "Fragment"] = result_identity.loc[index, 'Perimeter']/result_identity.loc[index, 'FP_Area']
                    except ZeroDivisionError as e:
                        result_identity.loc[index, "Fragment"] = numpy.nan

                    result_identity.loc[index, "AvgHeight"] = chm_mean
                    result_identity.loc[index, "Volume"] = chm_sum*cellArea
                    result_identity.loc[index, "Roughness"] = math.sqrt(math.pow(chm_mean, 2) + sqStdPop)

                del selected_segLn

    else:
        #No CHM
        for index in result_identity.index:

                query_str = "OLnFID=={} and OLnSEG=={}".format(result_identity.iloc[index].OLnFID,
                                                           result_identity.iloc[index].OLnSEG)
                selected_segLn = Att_seg_lines.query(query_str)

                if len(selected_segLn.index) > 0:
                    line_feat = selected_segLn['geometry'].iloc[0]
                    eucDistance=findEucDistance(line_feat)
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
                    result_identity.loc[index, "AvgHeight"] = numpy.nan
                    result_identity.loc[index, "Volume"] = numpy.nan
                    result_identity.loc[index, "Roughness"] = numpy.nan

                del selected_segLn


def identity_polygon(line_args):
    in_cl_buffer=line_args[0]
    in_fp_polygon=line_args[1]
    identity = in_fp_polygon.overlay(in_cl_buffer, how='identity')
    identity = identity.dropna(subset='OLnFID')
    return identity

if __name__ == '__main__':

    start_time = time.time()
    print('Starting attribute line footprint processing @ {}'.format(time.strftime("%a, %d %b %Y %H:%M:%S",time.localtime())))

    #Get tool arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=json.loads)
    parser.add_argument('-p', '--processes')
    parser.add_argument('-v', '--verbose')
    args = parser.parse_args()
    if args.verbose == 'True':
        verbose = True
    else:
        verbose = False

    #assign Tool arguments
    in_cl=args.input["in_cl"]
    in_fp=args.input["in_fp"]
    in_CHMR=args.input["in_CHMR"]
    sampling_type=args.input["sampling_type"]
    seg_len=float(args.input["seg_len"])
    ln_split_Tol=float(args.input["ln_split_Tol"])
    Max_ln_width=float(args.input["Max_ln_width"])
    Out_AttSeg=args.input["Out_AttSeg"]

    #Vaild input footprint shapefile has geometry

    if len(geopandas.GeoSeries.from_file(in_fp))<=0:
        areaAnalysis=False
    else:
        areaAnalysis = True

    try:
        with rasterio.open(in_CHMR) as in_CHM:
            heightAnalysis=True
        del in_CHM
    except Exception as error_in_CHM:
        print(error_in_CHM)
        heightAnalysis=False


    # Only process the following SampleingType
    SampleingType=["IN-FEATURES", "LINE-CROSSINGS", "ARBITRARY"]
    if args.input["sampling_type"] not in SampleingType:
        print("SamplingType is not correct, please verify it.")
        exit()

    # footprintField = FileToField(fileBuffer)
    print("Preparing line segments...")
    # Segment lines
    print("Input_Lines: {}".format(in_cl))


    #Return splited lines with two extra columns:['OLnFID','OLnSEG'] or return Dissolved whole line
    Att_seg_lines,Straight_lines=AttLineSplit(print, processes=int(args.processes), verbose=verbose,**args.input)

    print('%{}'.format(10))
    print("Line segments are prepared.")
    # #debugging save
    # geopandas.GeoDataFrame.to_file(Att_seg_lines, r"D:\Maverick\BERATool_Test_Data\LFP_result1\tool_att_seg_lines.shp")

    # if isinstance(Straight_lines,geopandas.GeoDataFrame):
    #     geopandas.GeoDataFrame.to_file(Straight_lines, r"D:\Maverick\BERATool_Test_Data\LFP_result1\tool_att_Straight_lines.shp")

    if areaAnalysis:
        #load in the footprint shapefile
        in_fp_shp=geopandas.GeoDataFrame.from_file(in_fp)


        print('%{}'.format(20))

        # Buffer seg straigth line or whole ln for identify footprint polygon
        if isinstance(Straight_lines,geopandas.GeoDataFrame):
            in_cl_buffer = geopandas.GeoDataFrame.copy(Straight_lines)
            in_cl_buffer['geometry']=in_cl_buffer.buffer(Max_ln_width,cap_style=2)
        else:
            in_cl_buffer=geopandas.GeoDataFrame.copy(Att_seg_lines)
            in_cl_buffer['geometry']=in_cl_buffer.buffer(Max_ln_width,cap_style=2)
        #
        # # # debugging save
        # geopandas.GeoDataFrame.to_file(in_cl_buffer, r"D:\Maverick\BERATool_Test_Data\LFP_result1\tool_in_cl_buffer.shp")

        # Create a emtpy geodataframe for identity polygon
        result_identity = geopandas.GeoDataFrame(columns=['geometry'], geometry='geometry', crs=in_cl_buffer.crs)
        print(
            "There are {} original footprint to be identified for {} segment lines....".format(len(in_fp_shp.index),len(in_cl_buffer)))
        print('%{}'.format(30))


        #Prepare line parameters for multiprocessing
        line_args=[]

        for row in in_cl_buffer.index:
            list_item=[]

            list_item.append(in_cl_buffer.iloc[[row]])
            for col in in_fp_shp.columns.array:
                if col=='OLnFID':
                    query_col=col
                    list_item.append(in_fp_shp.query(query_col + '=={}'.format(Att_seg_lines.loc[row, 'OLnFID'])))
                    break
                elif col=='Fr_Orig_ln':
                    query_col = col
                    list_item.append(in_fp_shp.query(query_col + '=={}'.format(Att_seg_lines.loc[row, 'OLnFID'])))
                    break
                else:
                    print('Could not match footprint to split lines.  Please check....')
                    exit()

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

        # Combine identity polygons into once geodataframe
        result_identity = geopandas.GeoDataFrame(pandas.concat(features, ignore_index=True))
        result_identity.reset_index()

        # # debugging save
        # geopandas.GeoDataFrame.to_file(result_identity, r"D:\Maverick\BERATool_Test_Data\LFP_result1\tool_in_result_identity.shp")


        print("Identifies are done.\nThere are {} identity polygons in total.".format(len(result_identity.index)))

        # ##Linear attributes
        print("Adding attributes......")
        print('%{}'.format(60))
        Fill_Attributes(result_identity,Att_seg_lines,areaAnalysis,heightAnalysis,**args.input)

        print('%{}'.format(80))

        # # debugging save
        # geopandas.GeoDataFrame.to_file(result_identity, r"D:\Maverick\BERATool_Test_Data\LFP_result1\tool_in_result_buffer.shp")

        #prepare a pandas dataframe from the result identity footprints without "geometry"
        footprint_att=pandas.DataFrame(result_identity.drop(columns='geometry'))

        # Clean the split line attribute columns
        del_list = list(col for col in footprint_att.columns if
                        col not in ['OLnFID', 'OLnSEG', 'geometry', 'LENGTH', 'FP_Area', 'Perimeter', 'Bearing',
                                    'Direction', 'Sinuosity', 'AvgWidth','AvgHeight', "Fragment", "Volume",
                                    "Roughness"])
        footprint_att = footprint_att.drop(columns=del_list)


        # Merging the cleaned split line with identity dataframe
        output_att_line=Att_seg_lines.merge(footprint_att,how='left',on=['OLnFID','OLnSEG'])




        print('%{}'.format(90))

        # Save attributed lines
        geopandas.GeoDataFrame.to_file(output_att_line, args.input['Out_AttSeg'])

        print('%{}'.format(100))
    print('Finishing footprint processing @ {} (or in {} second)'.format(time.strftime("%a, %d %b %Y %H:%M:%S"
                                                        ,time.localtime()),round(time.time() - start_time,5)))




