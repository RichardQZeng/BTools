from collections import OrderedDict
from multiprocessing.pool import Pool

import numpy as np
import time

import rasterio
import rasterio.mask
import fiona
from fiona import Geometry
from osgeo import gdal, ogr
from shapely.geometry import shape, mapping, LineString, Point
import ray

from dijkstra_algorithm import *
from common import *

# import line_profiler
# profile = line_profiler.LineProfiler()

# from memory_profiler import profile


class OperationCancelledException(Exception):
    pass


def centerline(callback, in_line, in_cost, line_radius,
               proc_segments, out_line, processes, verbose):
    if not compare_crs(vector_crs(in_line), raster_crs(in_cost)):
        print("Line and CHM spatial references are not same, please check.")
        return

    # Read input line features
    layer_crs = None
    input_lines = []
    with fiona.open(in_line) as open_line_file:
        layer_crs = open_line_file.crs
        for line in open_line_file:
            if line.geometry:
                if line.geometry['type'] != 'MultiLineString':
                    input_lines.append(line['geometry'])
                else:
                    print('MultiLineString found.')
                    geoms = shape(line['geometry']).geoms
                    for item in geoms:
                        line_part = Geometry.from_dict(item)
                        if line_part:
                            input_lines.append(line_part)
            else:
                print(f'Line {line.id} has empty geometry.')

    if proc_segments:
        # split line segments at vertices
        input_lines_temp = []
        for line in input_lines:
            line_segs = segments(line.coordinates)
            if line_segs:
                input_lines_temp += line_segs

        input_lines = input_lines_temp

    out_fields_list = ["start_pt_id", "end_pt_id", "total_cost"]

    # Process lines
    fiona_features = []
    all_lines = []
    features = []
    id = 0
    for line in input_lines:
        all_lines.append((line, line_radius, in_cost, id))
        id += 1

    print('{} lines to be processed.'.format(len(all_lines)))
    step = 0

    if PARALLEL_MODE == MODE_MULTIPROCESSING:
        features = execute_multiprocessing(all_lines, processes, verbose)
    elif PARALLEL_MODE == MODE_RAY:
        ray.init(log_to_driver=False)
        total_steps = len(all_lines)
        process_single_line_ray = ray.remote(process_single_line)
        result_ids = [process_single_line_ray.remote(line) for line in all_lines]

        while len(result_ids):
            done_id, result_ids = ray.wait(result_ids) 
            feat_geometry, feat_attributes = ray.get(done_id[0])
            features.append((feat_geometry, feat_attributes))
            print('Done {}'.format(step))
            step += 1

        # ray.shutdown()

    elif PARALLEL_MODE == MODE_SEQUENTIAL:
        for line in all_lines:
            feat_geometry, feat_attributes = process_single_line(line)
            if feat_geometry and feat_attributes:
                features.append((feat_geometry, feat_attributes))
            step += 1
            if verbose:
                print(' "PROGRESS_LABEL Ceterline {} of {}" '.format(step, total_steps), flush=True)
                print(' %{} '.format(step / total_steps * 100), flush=True)

    i = 0
    for feature in features:
        if not feature[0] or not feature[1]:
            continue

        if len(feature[0]) <= 1:
            print('Less than two points in the list {}, ignore'.format(i))
            continue

        i += 1

        # Save lines to shapefile
        single_feature = {
            'geometry': mapping(LineString(feature[0])),
            'properties': OrderedDict(list(zip(out_fields_list, feature[1])))
        }
        fiona_features.append(single_feature)

    schema = {
        'geometry': 'LineString',
        'properties': OrderedDict([
            ('start_pt_id', 'int'),
            ('end_pt_id', 'int'),
            ('total_cost', 'float')
        ])
    }

    driver = 'ESRI Shapefile'
    print('Writing lines to shapefile')

    with fiona.open(out_line, 'w', driver, schema, layer_crs.to_proj4()) as out_line_file:
        for feature in fiona_features:
            out_line_file.write(feature)

    if ray.is_initialized():
        ray.shutdown()


# @profile
def process_single_line(line_args, find_nearest=True, output_linear_reference=False):
    line = line_args[0]
    line_radius = line_args[1]
    in_cost_raster = line_args[2]

    line_buffer = shape(line).buffer(float(line_radius))
    pt_start = line['coordinates'][0]
    pt_end = line['coordinates'][-1]

    # buffer clip
    with(rasterio.open(in_cost_raster)) as raster_file:
        out_image, out_transform = rasterio.mask.mask(raster_file, [line_buffer], crop=True, nodata=BT_NODATA)

    ras_nodata = raster_file.meta['nodata']
    if not ras_nodata:
        ras_nodata = BT_NODATA

    USE_NUMPY_FOR_DIJKSTRA = True
    if USE_NUMPY_FOR_DIJKSTRA:
        matrix, contains_negative = MinCostPathHelper.block2matrix_numpy(out_image[0], ras_nodata)
    else:
        matrix, contains_negative = MinCostPathHelper.block2matrix(out_image[0], ras_nodata)

    if contains_negative:
        raise Exception('ERROR: Raster has negative values.')

    # get row col for points
    ras_transform = rasterio.transform.AffineTransformer(out_transform)

    if type(pt_start[0]) is tuple or type(pt_start[1]) is tuple or type(pt_end[0]) is tuple or type(pt_end[1]) is tuple:
        print("Point initialization error. Input is tuple.")
        return None, None

    start_tuples = []
    end_tuples = []
    start_tuple = []
    try:
        start_tuples = [(ras_transform.rowcol(pt_start[0], pt_start[1]), Point(pt_start[0], pt_start[1]), 0)]
        end_tuples = [(ras_transform.rowcol(pt_end[0], pt_end[1]), Point(pt_end[0], pt_end[1]), 1)]
        start_tuple = start_tuples[0]
        end_tuple = end_tuples[0]

        # regulate end poit coords in case they are out of index of matrix
        mat_size = matrix.shape
        mat_size = (mat_size[0]-1, mat_size[0]-1)
        start_tuple = (min(start_tuple[0], mat_size), start_tuple[1], start_tuple[2])
        end_tuple = (min(end_tuple[0], mat_size), end_tuple[1], end_tuple[2])

    except Exception as e:
        print(e)

    print(" Searching least cost path for line with id: {} ".format(line_args[3]), flush=True)

    if USE_NUMPY_FOR_DIJKSTRA:
        result = dijkstra_np(start_tuple, end_tuple, matrix)
    else:
        # TODO: change end_tuples to end_tuple
        result = dijkstra(start_tuple, end_tuples, matrix, find_nearest)

    if result is None:
        # raise Exception
        return None, None

    if len(result) == 0:
        # raise Exception
        print('No result returned.')
        return None, None

    path_points = None
    for path, costs, end_tuple in result:
        path_points = MinCostPathHelper.create_points_from_path(ras_transform, path,
                                                                start_tuple[1], end_tuple[1])
        if output_linear_reference:
            # TODO: code not reached
            # add linear reference
            for point, cost in zip(path_points, costs):
                point.addMValue(cost)

        total_cost = costs[-1]

    feat_attr = (start_tuple[2], end_tuple[2], total_cost)
    return path_points, feat_attr


def execute_multiprocessing(line_args, processes, verbose):
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
                if verbose:
                    print(' "PROGRESS_LABEL Ceterline {} of {}" '.format(step, total_steps), flush=True)
                    print(' %{} '.format(step/total_steps*100), flush=True)

        return features
    except OperationCancelledException:
        print("Operation cancelled")
        return None


if __name__ == '__main__':
    in_args, in_verbose = check_arguments()
    start_time = time.time()
    centerline(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Elapsed time: {}'.format(time.time() - start_time))
