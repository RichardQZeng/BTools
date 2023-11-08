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
# from common import *

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
    schema = None
    input_lines = []

    with fiona.open(in_line) as open_line_file:
        layer_crs = open_line_file.crs
        schema = open_line_file.meta['schema']
        for line in open_line_file:
            if line.geometry:
                if line.geometry['type'] != 'MultiLineString':
                    input_lines.append([line.geometry, line.properties])
                else:
                    print('MultiLineString found.')
                    geoms = shape(line['geometry']).geoms
                    for item in geoms:
                        line_part = Geometry.from_dict(item)
                        if line_part:
                            input_lines.append([line_part, line.properties])
            else:
                print(f'Line {line.id} has empty geometry.')

    if proc_segments:
        # split line segments at vertices
        input_lines_temp = []
        for line in input_lines:
            line_seg = line[0]
            line_prop = line[1]
            line_segs = segments(line_seg.coordinates)
            line_feats = [(line, line_prop) for line in line_segs]
            if line_segs:
                input_lines_temp.extend(line_feats)

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
    total_steps = len(all_lines)

    feat_geoms = []
    feat_props = []
    if PARALLEL_MODE == MODE_MULTIPROCESSING:
        feat_geoms, feat_props = execute_multiprocessing(all_lines, processes, verbose)

    elif PARALLEL_MODE == MODE_RAY:  # TODO: feature properties are added to return
        ray.init(log_to_driver=False)
        process_single_line_ray = ray.remote(process_single_line)
        result_ids = [process_single_line_ray.remote(line) for line in all_lines]

        while len(result_ids):
            done_id, result_ids = ray.wait(result_ids) 
            feat_geoms, feat_props = ray.get(done_id[0])
            features.append((feat_geoms, feat_props))
            print('Done {}'.format(step))
            step += 1

        # ray.shutdown()

    elif PARALLEL_MODE == MODE_SEQUENTIAL:
        for line in all_lines:
            geom, _, prop = process_single_line(line)
            if geom and prop:
                feat_geoms.append(geom)
                feat_props.append(prop)
            step += 1
            if verbose:
                print(' "PROGRESS_LABEL Ceterline {} of {}" '.format(step, total_steps), flush=True)
                print(' %{} '.format(step / total_steps * 100), flush=True)

    i = 0
    # for feature in features:
    #     if not feature[0] or not feature[1]:
    #         continue
    #
    #     if len(feature[0]) <= 1:
    #         print('Less than two points in the list {}, ignore'.format(i))
    #         continue
    #
    #     i += 1
    #
    #     # Save lines to shapefile
    #     single_feature = {
    #         'geometry': mapping(LineString(feature[0])),
    #         'properties': feature[2]
    #     }
    #     fiona_features.append(single_feature)

    driver = 'ESRI Shapefile'
    print('Writing lines to shapefile')

    save_features_to_shapefile(out_line, layer_crs, feat_geoms, schema, feat_props)

    if ray.is_initialized():
        ray.shutdown()


# @profile
def process_single_line(line_args, find_nearest=True, output_linear_reference=False):
    line = line_args[0][0]
    prop = line_args[0][1]
    line_radius = line_args[1]
    in_cost_raster = line_args[2]

    line_buffer = shape(line).buffer(float(line_radius))

    # buffer clip
    with rasterio.open(in_cost_raster) as raster_file:
        out_image, out_transform = rasterio.mask.mask(raster_file, [line_buffer], crop=True, nodata=BT_NODATA)

    line_id = line_args[3]

    ras_nodata = raster_file.meta['nodata']
    if not ras_nodata:
        ras_nodata = BT_NODATA

    least_cost_path = find_least_cost_path(ras_nodata, out_image, out_transform, line_id, shape(line))
    return least_cost_path[0], least_cost_path[1], prop


def execute_multiprocessing(line_args, processes, verbose):
    try:
        total_steps = len(line_args)
        feat_geoms = []
        feat_props = []
        with Pool(processes) as pool:
            step = 0
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(process_single_line, line_args):
                if BT_DEBUGGING:
                    print('Got result: {}'.format(result), flush=True)

                geom = result[0]
                prop = result[2]
                if geom and prop:
                    feat_geoms.append(LineString(geom))
                    feat_props.append(prop)

                step += 1
                if verbose:
                    print(' "PROGRESS_LABEL Ceterline {} of {}" '.format(step, total_steps), flush=True)
                    print(' %{} '.format(step/total_steps*100), flush=True)

        return feat_geoms, feat_props
    except OperationCancelledException:
        print("Operation cancelled")
        return None


if __name__ == '__main__':
    in_args, in_verbose = check_arguments()
    start_time = time.time()
    centerline(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Elapsed time: {}'.format(time.time() - start_time))
