import time
import warnings

import pandas
import geopandas
import numpy
import math
from skimage.graph import MCP_Geometric
import shapely
from shapely import *
from rasterio import features, mask
from scipy import ndimage
import argparse
import json
from multiprocessing.pool import Pool

from common import *

# to suppress pandas UserWarning: Geometry column does not contain geometry when splitting lines
warnings.simplefilter(action='ignore', category=UserWarning)


class OperationCancelledException(Exception):
    pass


def line_footprint(callback, in_line, in_canopy, in_cost, corridor_th_value, max_ln_width,
                   exp_shk_cell, proc_segments, out_footprint, processes, verbose):
    corridor_th_field = 'CorridorTh'
    line_seg = geopandas.GeoDataFrame.from_file(in_line)
    max_ln_width = float(max_ln_width)
    exp_shk_cell = int(exp_shk_cell)

    with rasterio.open(in_canopy) as raster:
        if line_seg.crs.to_epsg() != raster.crs.to_epsg():
            print("Line and raster spatial references are not same, please check.")
            exit()
    del raster
    with rasterio.open(in_cost) as raster:
        if line_seg.crs.to_epsg() != raster.crs.to_epsg():
            print("Line and raster spatial references are not same, please check.")
            exit()
    del raster
    if 'OLnFID' not in line_seg.columns.array:
        print("Cannot find {} column in input line data.\n '{}' column will be created".format('OLnFID', 'OLnFID'))
        line_seg['OLnFID'] = line_seg.index

    if 'CorridorTh' not in line_seg.columns.array:
        print("Cannot find {} column in input line data.\n '{}' "
              "column will be created".format('CorridorTh', 'CorridorTh'))
        line_seg['CorridorTh'] = corridor_th_value
    else:
        corridor_th_value = float(9999999)
    if 'OLnSEG' not in line_seg.columns.array:
        line_seg['OLnSEG'] = 0
        # print("Cannot find {} column in input line data.\n '{}' column will be created "
        #      "base on input Features ID".format('OLnSEG', 'OLnSEG'))

    ori_total_feat = len(line_seg)

    if proc_segments:
        print("Splitting lines into segments...")
        line_seg = split_into_segments(line_seg)
        print("Splitting lines into segments...Done")
    else:
        line_seg = split_into_equal_nth_segments(line_seg)

    list_dict_segment_all = line_prepare(callback, line_seg, in_canopy, in_cost, corridor_th_field, corridor_th_value,
                                         max_ln_width, exp_shk_cell, proc_segments, out_footprint, ori_total_feat)

    # pass single line one at a time for footprint
    footprint_list = []

    if USE_MULTI_PROCESSING:
        footprint_list = execute_multiprocessing(list_dict_segment_all, processes)
    else:
        for row in list_dict_segment_all:
            footprint_list.append(process_single_line(row))
            print("ID:{} is Done".format(row['OLnFID']))

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

    print('Finishing footprint processing @ {}\n (or in {} second)'
          .format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), time.time()-start_time))


def field_name_list(fc):
    # return a list of column name from shapefile
    if isinstance(fc,geopandas.GeoDataFrame):
        field_list = fc.columns.array
    else:
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


def process_single_line(dict_segment):
    # this function takes single line to work the line footprint
    # (regardless it process the whole line or individual segment)
    print('Processing line with ID: {}'.format(dict_segment['id']), flush=True)
    in_canopy_r = dict_segment['in_canopy_r']
    in_cost_r = dict_segment['in_cost_r']
    # CorridorTh_field=dict_segment['CorridorTh_field']
    corridor_th_value = dict_segment['corridor_th_value']

    try:
        corridor_th_value = float(corridor_th_value)
        if corridor_th_value < 0.0:
            corridor_th_value = 3.0
    except ValueError:
        corridor_th_value = 3.0

    max_ln_width = dict_segment['max_ln_width']
    exp_shk_cell = dict_segment['exp_shk_cell']
    # out_footprint=dict_segment['out_footprint']
    shapefile_proj = dict_segment['Proj_crs']
    orginal_col_name_list = dict_segment['orgi_col']

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
    destination = [shapes for shapes in geopandas.GeoDataFrame(geometry=[destination_point],
                                                               crs=shapefile_proj).geometry]

    # Buffer around line and clip cost raster and canopy raster
    # TODO: deal with NODATA
    with rasterio.open(in_cost_r) as src1:
        clip_in_cost_r, out_transform1 = rasterio.mask.mask(src1, [shapely.buffer(feat, max_ln_width)],
                                                            crop=True, nodata=-9999, filled=True)
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
        clip_in_canopy_r, out_transform2 = rasterio.mask.mask(src, [shapely.buffer(feat, max_ln_width)],
                                                              crop=True, nodata=-9999, filled=True)
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

        # Rasterize source point
        rasterized_source = features.rasterize(origin, out_shape=clip_cost_r.shape, transform=out_transform1,
                                               out=None, fill=0, all_touched=True, default_value=1, dtype=None)
        source = numpy.transpose(numpy.nonzero(rasterized_source))

        # generate the cost raster to source point
        mcp_source = MCP_Geometric(clip_cost_r, sampling=(cell_size_x, cell_size_y))
        source_cost_acc = mcp_source.find_costs(source)[0]
        del mcp_source

        # Rasterize destination point
        rasterized_destination = features.rasterize(destination, out_shape=clip_cost_r.shape, transform=out_transform1,
                                                    out=None, fill=0, all_touched=True, default_value=1, dtype=None)
        destination = numpy.transpose(numpy.nonzero(rasterized_destination))

        # generate the cost raster to destination point
        mcp_dest = MCP_Geometric(clip_cost_r, sampling=(cell_size_x, cell_size_y))
        dest_cost_acc = mcp_dest.find_costs(destination)[0]
        del mcp_dest

        # Generate corridor raster
        corridor = source_cost_acc + dest_cost_acc
        corridor = numpy.ma.masked_invalid(corridor)

        # Calculate minimum value of corridor raster
        if numpy.ma.min(corridor) is not None:
            corr_min = float(numpy.ma.min(corridor))
        else:
            corr_min = 0.05

        # corridor[numpy.isinf(corridor)]=-9999
        # masked_corridor = numpy.ma.masked_where(corridor==-9999, corridor)
        # Set minimum as zero and save minimum file
        corridor_min = numpy.ma.where((corridor - corr_min) > corridor_th_value, 1.0, 0.0)

        # Process: Stamp CC and Max Line Width

        # Original code here
        # RasterClass = SetNull(IsNull(CorridorMin),((CorridorMin) + ((Canopy_Raster) >= 1)) > 0)
        temp1 = (corridor_min + clip_canopy_r)
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
        msk = numpy.where(clean_raster == 1, True, False)

        # Process: ndarray to shapely Polygon
        out_polygon = features.shapes(clean_raster, mask=msk, transform=out_transform1)

        # create a shapely multipolygon
        multi_polygon = []
        for shape, value in out_polygon:
            multi_polygon.append(shapely.geometry.shape(shape))
        poly = shapely.geometry.MultiPolygon(multi_polygon)

        # create a pandas dataframe for the FP
        out_data = pandas.DataFrame({'OLnFID': [OID], 'OLnSEG': [FID], 'geometry': poly})
        out_gdata = geopandas.GeoDataFrame(out_data, geometry='geometry', crs=shapefile_proj)

        return out_gdata

    except Exception as e:
        print('Exception: {}'.format(e))


def split_line_fc(line):
    return list(map(LineString, zip(line.coords[:-1], line.coords[1:])))


def split_line_npart(line):
    # Work out n parts for each line (divided by 30m)
    n = math.ceil(line.length/30)
    if n > 1:
        # divided line into n-1 equal parts;
        distances = numpy.linspace(0, line.length, n)
        points = [line.interpolate(dist) for dist in distances]
        line = shapely.LineString(points)
        mline = split_line_fc(line)
    else:
        mline = line

    return mline


def split_into_segments(df):
    odf = df
    crs = odf.crs
    if 'OLnSEG' not in odf.columns.array:
        df['OLnSEG'] = numpy.nan

    df = odf.assign(geometry=odf.apply(lambda x: split_line_fc(x.geometry), axis=1))
    df = df.explode()

    df['OLnSEG'] = df.groupby('OLnFID').cumcount()
    gdf = geopandas.GeoDataFrame(df, geometry=df.geometry, crs=crs)
    gdf = gdf.sort_values(by=['OLnFID', 'OLnSEG'])
    gdf = gdf.reset_index(drop=True)
    return gdf


def split_into_equal_nth_segments(df):
    odf = df
    crs = odf.crs
    if 'OLnSEG' not in odf.columns.array:
        df['OLnSEG'] = numpy.nan
    df = odf.assign(geometry=odf.apply(lambda x: split_line_npart(x.geometry), axis=1))
    df = df.explode()

    df['OLnSEG'] = df.groupby('OLnFID').cumcount()
    gdf = geopandas.GeoDataFrame(df, geometry=df.geometry, crs=crs)
    gdf = gdf.sort_values(by=['OLnFID', 'OLnSEG'])
    gdf = gdf.reset_index(drop=True)
    return gdf


def line_prepare(callback, line_seg, in_canopy_r, in_cost_r, corridor_th_field, corridor_th_value,
                 max_ln_width, exp_shk_cell, proc_seg, out_footprint, ori_total_feat):

    # get the list of original columns names
    field_list_col = field_name_list(line_seg)
    keep_field_name = []
    for col_name in line_seg.columns:
        if col_name != 'geometry':
            keep_field_name.append(col_name)

    list_of_segment = []

    i = 0
    # process when shapefile is not an empty feature class
    if len(line_seg) > 0:

        for row in range(0, len(line_seg)):
            # creates a geometry object
            feat = line_seg.loc[row].geometry

            feature_attributes = {'seg_length': feat.length, 'geometry': feat, 'Proj_crs': line_seg.crs, 'id': i}
            # feature_attributes['seg_length'] = feat.length
            # feature_attributes['geometry'] = feat
            # feature_attributes['Proj_crs'] = line_seg.crs
            # feature_attributes['id'] = i

            for col_name in keep_field_name:
                feature_attributes[col_name] = line_seg.loc[row, col_name]
            list_of_segment.append(feature_attributes)
            i += 1

        print("There are {} lines to be processed.".format(ori_total_feat))
    else:
        print("Input line feature is corrupted, exit!")
        exit()

    # Add tools arguments into geodataframe record
    for record in list_of_segment:
        record['in_canopy_r'] = in_canopy_r
        record['in_cost_r'] = in_cost_r
        record['corridor_th_field'] = corridor_th_field
        record['corridor_th_value'] = record['CorridorTh']
        record['max_ln_width'] = max_ln_width
        record['exp_shk_cell'] = exp_shk_cell
        record['proc_seg'] = proc_seg
        record['out_footprint'] = out_footprint
        record['orgi_col'] = field_list_col

    # return list of geodataframe represents each line or segment
    return list_of_segment


def execute_multiprocessing(line_args, processes):
    try:
        total_steps = len(line_args)
        features = []
        with Pool(processes) as pool:
            # chunksize = math.ceil(total_steps / processes)
            # chunk_size = 1000
            step = 0
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(process_single_line, line_args):  # , chunksize=chunk_size):
                if BT_DEBUGGING:
                    print('Got result: {}'.format(result), flush=True)
                features.append(result)
                step += 1
                print('%{}'.format(step/total_steps*100), flush=True)

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

