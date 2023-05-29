import time
import os
import pandas
import geopandas
import numpy
from skimage.graph import MCP_Geometric
import shapely
from rasterio import features, mask
from scipy import ndimage
import argparse
import json
from multiprocessing.pool import Pool

from common import *


class OperationCancelledException(Exception):
    pass


def line_footprint(callback, in_cl, in_canopy_r, in_cost_r, corridor_th_field, corridor_th_value,
                   max_ln_width, exp_shk_cell, proc_seg, out_footprint, processes, verbose):

    corridor_th_value = float(corridor_th_value)
    max_ln_width = float(max_ln_width)
    exp_shk_cell = int(exp_shk_cell)

    proc_seg = False if proc_seg == 'False' else True

    list_dict_segment_all = line_prepare(callback, in_cl, in_canopy_r, in_cost_r, corridor_th_field,
                                         corridor_th_value, max_ln_width, exp_shk_cell, proc_seg, out_footprint)

    total_steps = len(list_dict_segment_all)

    # pass single line one at a time for footprint
    footprint_list = []
    if USE_MULTI_PROCESSING:
        footprint_list = execute_multiprocessing(list_dict_segment_all, processes)
    else:
        for row in list_dict_segment_all:
            footprint_list.append(process_single_line(row))

    print('Generating shapefile...........')
    results = geopandas.GeoDataFrame(pandas.concat(footprint_list, ignore_index=False))

    # dissolved polygon group by column 'OLnFID'
    dissolved_results = results.dissolve('OLnFID')
    dissolved_results.to_file(out_footprint)

    print('Finishing footprint processing @ {} (or in {} second)'
          .format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), time.time()-start_time))


def path_file_name(path):
    return os.path.basename(path)


def field_name_list(fc):
    # return a list of column name from shapefile
    field_list = geopandas.read_file(fc).columns.array
    return field_list


def has_field(fc, fi):
    # Check column name
    field_list = field_name_list(fc)
    if fi in field_list:
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


def split_line2(in_cl, proc_seg):
    shapefile = geopandas.GeoDataFrame.from_file(in_cl)
    shapefile_proj = shapefile.crs

    keep_field_name = []
    for col_name in shapefile.columns:
        if col_name != 'geometry':
            keep_field_name.append(col_name)

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

                    for col_name in keep_field_name:
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
                for col_name in keep_field_name:
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

    in_canopy_r = dict_segment['in_canopy_r']
    in_cost_r = dict_segment['in_cost_r']
    # CorridorTh_field=dict_segment['CorridorTh_field']
    corridor_th_value = dict_segment['corridor_th_value']
    max_ln_width = dict_segment['max_ln_width']
    exp_shk_cell = dict_segment['exp_shk_cell']
    # out_footprint=dict_segment['out_footprint']
    shapefile_proj = dict_segment['Proj_crs']
    orginal_col_name_list = dict_segment['Orgi_col']

    # segment line feature ID
    FID = dict_segment['OLnSEG']
    # original line ID for segment line
    OID = dict_segment['OLnFID']

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
    # TODO: deal with NODATA
    with rasterio.open(in_cost_r) as src1:
        clip_in_cost_r, out_transform1 = rasterio.mask.mask(src1, [shapely.buffer(feat, max_ln_width)], crop=True)
        out_meta = src1.meta
        crs = src1.crs
        crs_code = src1.meta['crs']
        crs_wkt = crs.wkt
        cell_size_x = src1.transform[0]
        cell_size_y = -src1.transform[4]

    out_meta.update({"driver": "GTiff",
                     "height": clip_in_cost_r.shape[1],
                     "width": clip_in_cost_r.shape[2],
                     "transform": out_transform1})

    del src1
    with rasterio.open(in_canopy_r) as src:
        clip_in_canopy_r, out_transform2 = rasterio.mask.mask(src, [shapely.buffer(feat, max_ln_width)], crop=True)
        out_meta = src.meta
        crs = src.crs

    out_meta.update({"driver": "GTiff",
                     "height": clip_in_canopy_r.shape[1],
                     "width": clip_in_canopy_r.shape[2],
                     "transform": out_transform2})
    del src

    # Work out the corridor from both end of the centerline
    try:
        clip_cost_r = numpy.squeeze(clip_in_cost_r, axis=0)
        clip_canopy_r = numpy.squeeze(clip_in_canopy_r, axis=0)

        numpy.place(clip_cost_r, clip_cost_r < 1, -9999)

        # Rasterize source point
        rasterized_source = features.rasterize(origin, out_shape=clip_cost_r.shape
                                               , transform=out_transform1,
                                               out=None, fill=0, all_touched=True, default_value=1, dtype=None)
        source = numpy.transpose(numpy.nonzero(rasterized_source))

        # generate the cost raster to source point
        mcp_source = MCP_Geometric(clip_cost_r, sampling=(cell_size_x, cell_size_y))
        source_cost_acc = mcp_source.find_costs(source)[0]

        del mcp_source
        # Rasterize destination point
        rasterized_destination = features.rasterize(destination, out_shape=clip_cost_r.shape,
                                                    transform=out_transform1,
                                                    out=None, fill=0, all_touched=True, default_value=1, dtype=None)
        destination = numpy.transpose(numpy.nonzero(rasterized_destination))

        # generate the cost raster to destination point
        mcp_dest = MCP_Geometric(clip_cost_r, sampling=(cell_size_x, cell_size_y))
        dest_cost_acc = mcp_dest.find_costs(destination)[0]

        del mcp_dest

        # Generate corridor raster
        corridor = numpy.add(source_cost_acc, dest_cost_acc)
        # null_cells = numpy.where(numpy.isinf(corridor), True, False)

        # Calculate minimum value of corridor raster
        if not numpy.nanmin(corridor) is None:
            corr_min = float(numpy.nanmin(corridor))
        else:
            corr_min = 0.05

        # Set minimum as zero and save minimum file
        corridor_min = numpy.where((corridor - corr_min) > corridor_th_value, 0, 1)
        masked_corridor_min = numpy.ma.masked_where(corridor_min == 0, corridor_min)

        # Process: Stamp CC and Max Line Width

        # Original code here
        # RasterClass = SetNull(IsNull(CorridorMin),((CorridorMin) + ((Canopy_Raster) >= 1)) > 0)
        temp1 = numpy.ma.add(masked_corridor_min, in_canopy_r)
        raster_class = numpy.where(temp1.data == 1, 1, 0)

        # BERA proposed Binary morphology
        # RasterClass_binary=numpy.where(RasterClass==0,False,True)

        if exp_shk_cell > 0 and cell_size_x < 1:
            # Process: Expand
            # FLM original Expand equivalent
            cell_size = int(exp_shk_cell * 2 + 1)
            expanded = ndimage.grey_dilation(raster_class, size=(exp_shk_cell * 2 + 1, exp_shk_cell * 2 + 1))

            # BERA proposed Binary morphology Expand
            # Expanded = ndimage.binary_dilation(RasterClass_binary, iterations=exp_shk_cell,border_value=1)

            # Process: Shrink
            # FLM original Shrink equivalent
            file_shrink = ndimage.grey_erosion(expanded, size=(exp_shk_cell * 2 + 1, exp_shk_cell * 2 + 1))

            # BERA proposed Binary morphology Shrink
            # fileShrink = ndimage.binary_erosion((Expanded),iterations=Exp_Shk_cell,border_value=1)
        else:
            if BT_DEBUGGING:
                print('No Expand And Shrink cell performed.')
            file_shrink = raster_class

        # Process: Boundary Clean
        clean_raster = ndimage.gaussian_filter(file_shrink, sigma=0, mode='nearest')

        # creat mask for non-polygon area
        mask = numpy.where(clean_raster == 0, True, False)

        # Process: ndarray to shapely Polygon
        out_polygon = features.shapes(clean_raster, mask=mask, transform=out_transform1)

        # create a shapely multipoly
        multi_polygon = []
        for shape, value in out_polygon:
            multi_polygon.append(shapely.geometry.shape(shape))
        poly = shapely.geometry.MultiPolygon(multi_polygon)

        # create a multipoly Geopandas Geodataframe for the input whole line's polygon or segments' polygon
        out_data = geopandas.GeoDataFrame()
        out_data['OLnSEG'] = [FID]
        out_data['OLnFID'] = [OID]
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
        print('Exception: {}'.format(e))


def line_prepare(callback, in_cl, in_canopy_r, in_cost_r, corridor_th_field,
                 corridor_th_value, max_ln_width, exp_shk_cell, proc_seg, out_footprint):
    # Open shapefile -input centerlines and check existing Corridor threshold field
    # if threshold field is not found, it will be created and populate value of '3'
    # print('Check {} field in input feature.'.format(args['CorridorTh_field']))
    # if HasField(args['in_cl'], args['CorridorTh_field']):
    #     pass

    # get the list of original columns names
    field_list_col = field_name_list(in_cl)

    # Split the input centerline and return a list of geodataframe
    print('Split line process.............')
    list_dict_segment_all = split_line2(in_cl, proc_seg)

    # Add tools arguments into geodataframe record
    for record in list_dict_segment_all:
        record['in_canopy_r'] = in_canopy_r
        record['in_cost_r'] = in_cost_r
        record['corridor_th_field'] = corridor_th_field
        record['corridor_th_value'] = corridor_th_value
        record['max_ln_width'] = max_ln_width
        record['exp_shk_cell'] = exp_shk_cell
        record['proc_seg'] = proc_seg
        record['out_footprint'] = out_footprint
        record['Orgi_col'] = field_list_col

    # return list of geodataframe represents each line or segment
    return list_dict_segment_all


def execute_multiprocessing(line_args, processes):
    try:
        total_steps = len(line_args)
        features = []
        with Pool(processes) as pool:
            step = 0
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(process_single_line, line_args):
                if BT_DEBUGGING:
                    print('Got result: {}'.format(result), flush=True)
                features.append(result)
                step += 1
                print('%{}'.format(step / total_steps * 100))
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
    parser.add_argument('-p', '--processes')
    parser.add_argument('-v', '--verbose')
    args = parser.parse_args()

    verbose = True if args.verbose == 'True' else False

    line_footprint(print, **args.input, processes=int(args.processes), verbose=verbose)

