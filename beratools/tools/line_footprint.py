import time
import os
import pandas
import geopandas
import numpy
from skimage.graph import MCP_Geometric
import shapely
import rasterio
from rasterio import features, mask
from scipy import ndimage
import argparse
import json
from multiprocessing.pool import Pool

USE_MULTI_PROCESSING = True


class OperationCancelledException(Exception):
    pass


def line_footprint(callback, in_cl, in_CanopyR, in_CostR, CorridorTh_field,
                   CorridorTh_value, Max_ln_width, Exp_Shk_cell, proc_seg, out_footprint):

    CorridorTh_value = float(CorridorTh_value)
    Max_ln_width = float(Max_ln_width)
    Exp_Shk_cell = float(Exp_Shk_cell)
    proc_seg = False if proc_seg == 'False' else True

    list_dict_segment_all = lineprepare(callback, in_cl, in_CanopyR, in_CostR, CorridorTh_field,
                   CorridorTh_value, Max_ln_width, Exp_Shk_cell, proc_seg, out_footprint)

    total_steps = len(list_dict_segment_all)

    # pass single line one at a time for footprint
    footprint_list = []
    if USE_MULTI_PROCESSING:
        footprint_list = execute_multiprocessing(list_dict_segment_all)
    else:
        for row in list_dict_segment_all:
            footprint_list.append(process_single_line(row))

    # Old multiprocessing
    # pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    # footprint_list = pool.map(process_single_line, list_dict_segment_all)

    print('Generating shapefile...........')
    results = geopandas.GeoDataFrame(pandas.concat(footprint_list, ignore_index=False))

    # dissolved polygon group by column 'Fr_Orig_ln'
    dissolved_results = results.dissolve('Fr_Orig_ln')
    dissolved_results.to_file(out_footprint)

    print('Finishing footprint processing @ {} (or in {} second)'
          .format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), time.time()-start_time))

    # Old multiprocessing pool call
    # pool.close()
    # pool.join()

def PathFileName(path):
    return os.path.basename(path)


def FieldNameList(fc):
    # return a list of column name from shapefile
    fieldlist = geopandas.read_file(fc).columns.array
    return fieldlist


def HasField(fc, fi):
    # Check column name
    fieldlist = FieldNameList(fc)
    if fi in fieldlist:
        print("column: {} is found".format(fi))
        return True
    elif fi == 'CorridorTh':
        shapefile = geopandas.GeoDataFrame.from_file(fc)
        for row in range(0, len(shapefile)):
            shapefile.loc[row, fi] = 3.0
        shapefile.to_file(fc)
        print("Warning: There is no field named {} in the input data".format('CorridorTh'))
        print("Field: '{}' is added and default threshold (i.e.3) is adopted".format('CorridorTh'))
        return True
    else:
        print("Warning: There is no field named {} in the input data".format(fi))

        return False


def SpliteLine2(in_cl, proc_seg):
    shapefile = geopandas.GeoDataFrame.from_file(in_cl)
    shapefile_proj = shapefile.crs

    KeepFieldName = []
    for col_name in shapefile.columns:
        if col_name != 'geometry':
            KeepFieldName.append(col_name)

    list_of_segment = []

    i = 0
    # process when shapefile is not an empty feature class
    if len(shapefile) > 0:

        for row in range(0, len(shapefile)):
            # creates a geometry object
            feat = shapefile.loc[row].geometry

            # Split every segments from line
            segment_list = (list(map(shapely.geometry.LineString, zip(feat.coords[:-1], feat.coords[1:]))))
            feature_attributes = {}

            # process every segments
            if proc_seg:

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
            else:  # process on original lines
                feature_attributes['FID'] = i
                feature_attributes['OID'] = row
                feature_attributes['No_of_Seg'] = len(segment_list)
                feature_attributes['Seg_leng'] = feat.length
                feature_attributes['geometry'] = feat
                feature_attributes['Proj_crs'] = shapefile_proj
                for col_name in KeepFieldName:
                    feature_attributes[col_name] = shapefile.loc[row, col_name]
                list_of_segment.append(feature_attributes)
                i = i + 1

        print("There are {} lines to be processed.".format(len(list_of_segment)))

        # return a list of features Dictionary
        return list_of_segment

    else:
        print("Input line feature is corrupted, exit!")
        exit()


def process_single_line(dict_segment):
    # this function takes single line to work the line footprint
    # (regardless it process the whole line or individual segment)

    in_CanopyR = dict_segment['in_CanopyR']
    in_CostR = dict_segment['in_CostR']
    # CorridorTh_field=dict_segment['CorridorTh_field']
    CorridorTh_value = dict_segment['CorridorTh_value']
    Max_ln_width = dict_segment['Max_ln_width']
    Exp_Shk_cell = dict_segment['Exp_Shk_cell']
    # out_footprint=dict_segment['out_footprint']
    shapefile_proj = dict_segment['Proj_crs']
    orginal_col_name_list = dict_segment['Orgi_col']

    # segment line feature ID
    FID = dict_segment['FID']
    # original line ID for segment line
    OID = dict_segment['OID']

    segment_list = []

    feat = dict_segment['geometry']
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

    # Buffer around line and clip cost raster and canopy raster
    with rasterio.open(in_CostR) as src1:
        clip_in_CostR, out_transform1 = rasterio.mask.mask(src1, [shapely.buffer(feat, Max_ln_width)], crop=True)
        out_meta = src1.meta
        crs = src1.crs
        crs_code = src1.meta['crs']
        crs_wkt = crs.wkt
        cellSizex = src1.transform[0]
        cellSizey = -src1.transform[4]

    out_meta.update({"driver": "GTiff",
                     "height": clip_in_CostR.shape[1],
                     "width": clip_in_CostR.shape[2],
                     "transform": out_transform1})

    del src1
    with rasterio.open(in_CanopyR) as src:
        clip_in_CanopyR, out_transform2 = rasterio.mask.mask(src, [shapely.buffer(feat, Max_ln_width)], crop=True)
        out_meta = src.meta
        crs = src.crs

    out_meta.update({"driver": "GTiff",
                     "height": clip_in_CanopyR.shape[1],
                     "width": clip_in_CanopyR.shape[2],
                     "transform": out_transform2})
    del src

    # Work out the corridor from both end of the centerline
    try:
        clip_CostR = numpy.squeeze(clip_in_CostR, axis=0)
        clip_CanopyR = numpy.squeeze(clip_in_CanopyR, axis=0)

        numpy.place(clip_CostR, clip_CostR < 1, -9999)

        # Rasterize source point
        rasterized_source = features.rasterize(origin, out_shape=clip_CostR.shape
                                               , transform=out_transform1,
                                               out=None, fill=0, all_touched=True, default_value=1, dtype=None)
        source = numpy.transpose(numpy.nonzero(rasterized_source))

        # generate the cost raster to source point
        mcp_source = MCP_Geometric(clip_CostR, sampling=(cellSizex, cellSizey))
        source_cost_Acc = mcp_source.find_costs(source)[0]

        del mcp_source
        # Rasterize destination point
        rasterized_destination = features.rasterize(destination, out_shape=clip_CostR.shape,
                                                    transform=out_transform1,
                                                    out=None, fill=0, all_touched=True, default_value=1, dtype=None)
        destination = numpy.transpose(numpy.nonzero(rasterized_destination))

        # generate the cost raster to destination point
        mcp_Dest = MCP_Geometric(clip_CostR, sampling=(cellSizex, cellSizey))
        dest_cost_Acc = mcp_Dest.find_costs(destination)[0]

        del mcp_Dest

        # Generate corridor raster
        Corridor = numpy.add(source_cost_Acc, dest_cost_Acc)
        nullcells = numpy.where(numpy.isinf(Corridor), True, False)

        # Calculate minimum value of corridor raster
        if not numpy.min(Corridor) is None:
            corrMin = float(numpy.min(Corridor))
        else:
            corrMin = 0.0

        # Set minimum as zero and save minimum file
        CorridorMin = numpy.where((Corridor - corrMin) > CorridorTh_value, 1, 0)
        numpy.place(CorridorMin, nullcells, -9999)

        # Process: Stamp CC and Max Line Width

        # Original code here
        # RasterClass = SetNull(IsNull(CorridorMin),((CorridorMin) + ((Canopy_Raster) >= 1)) > 0)
        RasterClass = numpy.where(clip_CanopyR + CorridorMin == 0, 0, 1)

        # BERA proposed Binary morphology
        # RasterClass_binary=numpy.where(RasterClass==0,False,True)

        if Exp_Shk_cell > 0 and Exp_Shk_cell * cellSizex < 1:
            # Process: Expand
            # FLM original Expand equivalent
            Expanded = ndimage.grey_dilation(RasterClass, size=(Exp_Shk_cell * 2 + 1, Exp_Shk_cell * 2 + 1))

            # BERA proposed Binary morphology Expand
            # Expanded = ndimage.binary_dilation(RasterClass_binary, iterations=Exp_Shk_cell,border_value=1)

            # Process: Shrink
            # FLM original Shrink equivalent
            fileShrink = ndimage.grey_erosion(Expanded, size=(Exp_Shk_cell * 2 + 1, Exp_Shk_cell * 2 + 1))

            # BERA proposed Binary morphology Shrink
            # fileShrink = ndimage.binary_erosion((Expanded),iterations=Exp_Shk_cell,border_value=1)
        else:
            print('No Expand And Shrink cell perform.')
            fileShrink = RasterClass

        # Process: Boundary Clean
        cleanRaster = ndimage.gaussian_filter(fileShrink, sigma=0, mode='nearest')

        # creat mask for non-polygon area
        mask = numpy.where(cleanRaster == 0, True, False)

        # Process: ndarray to shapely Polygon
        Out_polygon = features.shapes(cleanRaster, mask=mask, transform=out_transform1)

        # create a shapely multipoly
        multi_polygon = []
        for shape, value in Out_polygon:
            # print(shape)
            multi_polygon.append(shapely.geometry.shape(shape))
        poly = shapely.geometry.MultiPolygon(multi_polygon)

        # create a multipoly Geopandas Geodataframe for the input whole line's polygon or segments' polygon
        out_data = geopandas.GeoDataFrame()
        out_data['Fr_Seg_Ln'] = [FID]
        out_data['Fr_Orig_ln'] = [OID]
        out_data['geometry'] = None
        out_data.loc[0, 'geometry'] = poly
        out_data = out_data.set_crs(crs_code, allow_override=True)
        for col in orginal_col_name_list:
            if col != 'geometry':
                out_data[col] = dict_segment[col]

        # out_data.to_file(out_footprint)
        print("Footprint for FID:{} is done".format(FID))

        # return a geodataframe
        return out_data

    except Exception as e:
        print(e)


def lineprepare(callback, in_cl, in_CanopyR, in_CostR, CorridorTh_field, 
                CorridorTh_value, Max_ln_width, Exp_Shk_cell, proc_seg, out_footprint):
    # in_cl, in_CanopyR, in_CostR, CorridorTh_field, CorridorTh_value, Max_ln_width, Exp_Shk_cell,
    # proc_seg, out_footprint

    # CorridorTh_value = float(args['CorridorTh_value'])
    # Max_ln_width = float(args['Max_ln_width'])
    # Exp_Shk_cell = int(args['Exp_Shk_cell'])
    # print("Preparing Lines............")
    # print("Process every segments: {}".format(args['proc_seg']))
    # if args['proc_seg'] == 'True':
    #     args['proc_seg'] = True
    # else:
    #     args['proc_seg'] = False

    # Open shapefile -input centerlines and check existing Corridor threshold field
    # if threshold field is not found, it will be created and populate value of '3'
    # print('Check {} field in input feature.'.format(args['CorridorTh_field']))
    # if HasField(args['in_cl'], args['CorridorTh_field']):
    #     pass

    # get the list of original columns names
    fieldlist_col = FieldNameList(in_cl)

    # Split the input centerline and return a list of geodataframe
    print('Split line process.............')
    list_dict_segment_all = SpliteLine2(in_cl, proc_seg)

    # Add tools arguments into geodataframe record
    for record in list_dict_segment_all:
        record['in_CanopyR'] = in_CanopyR
        record['in_CostR'] = in_CostR
        record['CorridorTh_field'] = CorridorTh_field
        record['CorridorTh_value'] = CorridorTh_value
        record['Max_ln_width'] = Max_ln_width
        record['Exp_Shk_cell'] = Exp_Shk_cell
        record['proc_seg'] = proc_seg
        record['out_footprint'] = out_footprint
        record['Orgi_col'] = fieldlist_col

    # return list of geodataframe represents each line or segment
    return list_dict_segment_all


def execute_multiprocessing(line_args):
    try:
        total_steps = len(line_args)
        features = []
        with Pool() as pool:
            step = 0
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(process_single_line, line_args):
                print('Got result: {}'.format(result), flush=True)
                features.append(result)
                step += 1
                print(step)
                print('%{}'.format(step / total_steps * 100))
                # if result > 0.9:
                #     print('Pool terminated.')
                #     raise OperationCancelledException()
        return features
    except OperationCancelledException:
        print("Operation cancelled")
        return None

if __name__ == '__main__':
    start_time = time.time()
    print('Starting footprint processing @ {}'.format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    # Get tool arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=json.loads)
    args = parser.parse_args()

    line_footprint(print, **args.input)

