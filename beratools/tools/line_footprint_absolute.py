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
from multiprocessing.pool import Pool
import itertools

from common import *
from dijkstra_algorithm import *


# to suppress pandas UserWarning: Geometry column does not contain geometry when splitting lines
warnings.simplefilter(action='ignore', category=UserWarning)


class OperationCancelledException(Exception):
    pass


def line_footprint(callback, in_line, in_canopy, in_cost, corridor_th_value, max_ln_width,
                   exp_shk_cell, out_footprint, processes, verbose):
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
        if BT_DEBUGGING:
            print("Cannot find {} column in input line data".format('CorridorTh'))
        print("New column created: {}".format('CorridorTh'))
        line_seg['CorridorTh'] = corridor_th_value
    else:
        corridor_th_value = float(9999999)
    if 'OLnSEG' not in line_seg.columns.array:
        line_seg['OLnSEG'] = 0

    ori_total_feat = len(line_seg)

    proc_segments = False
    if proc_segments:
        print("Splitting lines into segments...")
        line_seg = split_into_segments(line_seg)
        print("Splitting lines into segments...Done")
    else:
        line_seg = split_into_equal_nth_segments(line_seg)

    line_args = line_prepare(callback, line_seg, in_canopy, in_cost, corridor_th_field, corridor_th_value,
                             max_ln_width, exp_shk_cell, proc_segments, out_footprint, ori_total_feat)

    # pass single line one at a time for footprint
    feat_list = []
    footprint_list = []
    line_list = []

    if PARALLEL_MODE == MODE_MULTIPROCESSING:
        feat_list = execute_multiprocessing(line_args, processes, verbose)
    else:
        process_single_line = process_single_line_segment
        if GROUPING_SEGMENT:
            process_single_line = process_single_line_whole

        total_steps = len(line_args)
        step = 0
        for row in line_args:
            feat_list.append(process_single_line(row))
            step += 1
            if verbose:
                print(' "PROGRESS_LABEL Line Footprint {} of {}" '.format(step, total_steps), flush=True)
                print(' %{} '.format(step / total_steps * 100), flush=True)

    print('Generating shapefile ...')

    if feat_list:
        for i in feat_list:
            footprint_list.append(i[0])
            line_list.append(i[1])
    
    results = geopandas.GeoDataFrame(pandas.concat(footprint_list))
    results = results.sort_values(by=['OLnFID', 'OLnSEG'])
    results = results.reset_index(drop=True)

    # dissolved polygon group by column 'OLnFID'
    dissolved_results = results.dissolve(by='OLnFID', as_index=False)
    dissolved_results = dissolved_results.drop(columns=['OLnSEG'])
    print("Saving output ...")
    dissolved_results.to_file(out_footprint)
    print('%{}'.format(100))

    # save lines to file
    save_features_to_shapefile(r'D:\Temp\lines\new_lines.shp', line_seg.crs, line_list)

    print('Finishing footprint processing in {} seconds)'.format(time.time()-start_time))


def field_name_list(fc):
    # return a list of column name from shapefile
    if isinstance(fc, geopandas.GeoDataFrame):
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


def process_single_line_whole(line):
    footprints = []
    lines = []
    for line_seg in line:
        footprint = process_single_line_segment(line_seg)
        footprints.append(footprint[0])
        lines.append(footprint[1])

    coord_list = []
    if lines:
        for item in lines:
            if item:
                coord_list.append(item[0])

    multi_line = MultiLineString(coord_list)

    if footprints:
        if not all(item is None for item in footprints):
            footprint_merge = pandas.concat(footprints)
            footprint_merge.dissolve()
            footprint_merge.drop(columns=['OLnSEG'])
        else:
            print(f'Empty footprint returned.')
            return None
    else:
        print(f'Empty footprint returned.')
        return None

    if len(line) > 0:
        print('process_single_line_whole: Processing line with ID: {}, done.'
              .format(line[0]['OLnFID']), flush=True)
    return footprint_merge, multi_line


def process_single_line_segment(dict_segment):
    # this function takes single line to work the line footprint
    # (regardless it process the whole line or individual segment)
    in_canopy_r = dict_segment['in_canopy_r']
    in_cost_r = dict_segment['in_cost_r']
    corridor_th_value = dict_segment['corridor_th_value']

    try:
        corridor_th_value = float(corridor_th_value)
        if corridor_th_value < 0.0:
            corridor_th_value = 3.0
    except ValueError as e:
        print(e)
        corridor_th_value = 3.0

    max_ln_width = dict_segment['max_ln_width']
    exp_shk_cell = dict_segment['exp_shk_cell']
    shapefile_proj = dict_segment['Proj_crs']
    orginal_col_name_list = dict_segment['org_col']

    FID = dict_segment['OLnSEG']  # segment line feature ID
    OID = dict_segment['OLnFID']  # original line ID for segment line

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

        # TODO: further investigate and submit issue to skimage
        # There is a severe bug in skimage find_costs
        # when nan is present in clip_cost_r, find_costs cause access violation
        # no message/exception will be caught
        # change all nan to -9999 for workaround
        remove_nan_from_array(clip_cost_r)

        # generate the cost raster to source point
        mcp_source = MCP_Geometric(clip_cost_r, sampling=(cell_size_x, cell_size_y))
        source_cost_acc, traceback = mcp_source.find_costs(source)
        del mcp_source

        # Rasterize destination point
        rasterized_destination = features.rasterize(destination, out_shape=clip_cost_r.shape, transform=out_transform1,
                                                    out=None, fill=0, all_touched=True, default_value=1, dtype=None)
        destination = numpy.transpose(numpy.nonzero(rasterized_destination))

        # generate the cost raster to destination point
        mcp_dest = MCP_Geometric(clip_cost_r, sampling=(cell_size_x, cell_size_y))
        dest_cost_acc, traceback = mcp_dest.find_costs(destination)
        del mcp_dest

        # Generate corridor raster
        corridor = source_cost_acc + dest_cost_acc
        corridor = numpy.ma.masked_invalid(corridor)

        # find least cost path in corridor raster
        mat = corridor.copy()
        lc_path = find_least_cost_path(out_meta, mat, out_transform2, 9999, feat)

        # Calculate minimum value of corridor raster
        if numpy.ma.min(corridor) is not None:
            corr_min = float(numpy.ma.min(corridor))
        else:
            corr_min = 0.05

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
        for shp, value in out_polygon:
            multi_polygon.append(shapely.geometry.shape(shp))
        poly = shapely.geometry.MultiPolygon(multi_polygon)

        # create a pandas dataframe for the FP
        out_data = pandas.DataFrame({'OLnFID': [OID], 'OLnSEG': [FID], 'geometry': poly})
        out_gdata = geopandas.GeoDataFrame(out_data, geometry='geometry', crs=shapefile_proj)

        if not GROUPING_SEGMENT:
            print('LP:PSLS: Processing line ID: {}, done.'.format(dict_segment['OLnSEG']), flush=True)

        return out_gdata, lc_path

    except Exception as e:
        print('Exception: {}'.format(e))
        print('Line footprint: 318')
        return None


def split_line_fc(line):
    return list(map(LineString, zip(line.coords[:-1], line.coords[1:])))


def split_line_npart(line):
    if not line:
        return None
    # Work out n parts for each line (divided by 30m)
    n = math.ceil(line.length/LP_SEGMENT_LENGTH)
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
            if feat:
                feature_attributes = {'seg_length': feat.length, 'geometry': feat, 'Proj_crs': line_seg.crs}

                for col_name in keep_field_name:
                    feature_attributes[col_name] = line_seg.loc[row, col_name]
                list_of_segment.append(feature_attributes)
                i += 1

        print("There are {} lines to be processed.".format(ori_total_feat))
    else:
        print("Input line feature is corrupted, exit!")
        exit()

    # Add tools arguments into GeoDataFrame record
    for record in list_of_segment:
        record['in_canopy_r'] = in_canopy_r
        record['in_cost_r'] = in_cost_r
        record['corridor_th_field'] = corridor_th_field
        record['corridor_th_value'] = record['CorridorTh']
        record['max_ln_width'] = max_ln_width
        record['exp_shk_cell'] = exp_shk_cell
        record['proc_seg'] = proc_seg
        record['out_footprint'] = out_footprint
        record['org_col'] = field_list_col

    # return list of GeoDataFrame represents each line or segment
    if GROUPING_SEGMENT:
        # group line segments by line id
        def key_func(x): return x['OLnFID']
        lines = []
        for key, group in itertools.groupby(list_of_segment, key_func):
            lines.append(list(group))

        return lines
    else:
        return list_of_segment


def execute_multiprocessing(line_args, processes, verbose):
    try:
        total_steps = len(line_args)
        features = []

        with Pool(processes) as pool:
            # chunk_size = 1
            step = 0
            process_single_line = process_single_line_segment
            if GROUPING_SEGMENT:
                process_single_line = process_single_line_whole

            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(process_single_line, line_args):
                if BT_DEBUGGING:
                    print('Got result: {}'.format(result), flush=True)
                features.append(result)
                step += 1
                if verbose:
                    print(' "PROGRESS_LABEL Line Footprint {} of {}" '.format(step, total_steps), flush=True)
                    print(' %{} '.format(step/total_steps*100), flush=True)

        print('Multiprocessing done.')
        return features
    except OperationCancelledException:
        print("Operation cancelled")
        return None


if __name__ == '__main__':
    start_time = time.time()
    print('Footprint processing started')
    print('Current time: {}'.format(time.strftime("%b %Y %H:%M:%S", time.localtime())))

    in_args, in_verbose = check_arguments()
    line_footprint(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Current time: {}'.format(time.strftime("%b %Y %H:%M:%S", time.localtime())))

