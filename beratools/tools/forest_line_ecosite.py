from collections import OrderedDict
from multiprocessing.pool import Pool
import time

import fiona
from shapely.geometry import shape, mapping
from shapely.ops import split
from shapely import STRtree
import ray

from common import *

# import line_profiler
# profile = line_profiler.LineProfiler()

# from memory_profiler import profile


class OperationCancelledException(Exception):
    pass


def forest_line_ecosite(callback, in_line, in_ecosite, out_line, processes, verbose):
    if not compare_crs(vector_crs(in_line), vector_crs(in_ecosite)):
        print("Line and CHM spatial references are not same, please check.")
        return

    # Read input line features
    layer_crs = None
    input_lines = []
    with fiona.open(in_line) as open_line_file:
        layer_crs = open_line_file.crs
        for line in open_line_file:
            if line.geometry:
                if line.geometry.type != 'MultiLineString':
                    input_lines.append([line.geometry, line.properties])
                else:
                    print('MultiLineString found.')
                    geoms = shape(line.geometry).geoms
                    for item in geoms:
                        line_part = Geometry.from_dict(item)
                        if line_part:
                            input_lines.append([line_part, line.properties])
            else:
                print(f'Line {line.id} has empty geometry.')

    out_fields_list = ["ecosite"]

    # Create search tree
    feats = read_feature_from_shapefile(in_ecosite)
    geoms = [i[0] for i in feats]
    tree = STRtree(geoms)

    # Process lines
    fiona_features = []
    all_lines = []
    feat_all = []
    id = 0
    for line in input_lines:
        line_geom = shape(line[0])
        line_prop = line[2]
        index_query = tree.query(line_geom)
        geoms_intersected = []
        for i in index_query:
            geoms_intersected.append(geoms[i])
        all_lines.append(([line_geom, line_prop], geoms_intersected, id))
        id += 1

    print('{} lines to be processed.'.format(len(all_lines)))
    step = 0
    total_steps = len(all_lines)

    if PARALLEL_MODE == MODE_MULTIPROCESSING:
        feat_all = execute_multiprocessing(all_lines, processes, verbose)
    elif PARALLEL_MODE == MODE_RAY:
        ray.init(log_to_driver=False)
        process_single_line_ray = ray.remote(process_single_line)
        result_ids = [process_single_line_ray.remote(line) for line in all_lines]

        while len(result_ids):
            done_id, result_ids = ray.wait(result_ids)
            feat_geometry, feat_attributes = ray.get(done_id[0])
            feat_all.append((feat_geometry, feat_attributes))
            print('Done {}'.format(step))
            step += 1

        # ray.shutdown()

    elif PARALLEL_MODE == MODE_SEQUENTIAL:
        for line in all_lines:
            line_collection = process_single_line(line)
            if line_collection:
                feat_all.append(line_collection)
            step += 1
            if verbose:
                print(' "PROGRESS_LABEL Ceterline {} of {}" '.format(step, total_steps), flush=True)
                print(' %{} '.format(step / total_steps * 100), flush=True)

    i = 0
    for feature in feat_all:
        if not feature:
            continue

        i += 1
        for line in feature:
            try:
                single_line = {
                    'geometry': mapping(line),
                    'properties': OrderedDict(list(zip(out_fields_list, ['treed area'])))  # TODO: add attributes
                }
            except Exception as e:
                print(e)
            else:
                fiona_features.append(single_line)

    schema = {
        'geometry': 'LineString',
        'properties': OrderedDict([
            ('ecosite', 'str'),
        ])
    }

    driver = 'ESRI Shapefile'
    print('Writing lines to shapefile')

    # Save lines to shapefile
    with fiona.open(out_line, 'w', driver, schema, layer_crs.to_proj4()) as out_line_file:
        for feature in fiona_features:
            out_line_file.write(feature)

    if ray.is_initialized():
        ray.shutdown()


def split_line_with_polygon(lines, polygon):
    line_list = []
    for line in lines:
        line_collection = split(line, polygon)

        if not line_collection.is_empty:
            for i in line_collection.geoms:
                line_list.append(i)

    return line_list


# param
# @profile
def process_single_line(line_args, find_nearest=True, output_linear_reference=False):
    line = line_args[0]
    geoms = line_args[1]

    lines = [line]
    if len(geoms) == 0:  # none intersecting polygons
        return [line]
    else:
        for geom in geoms:
            lines = split_line_with_polygon(lines, geom)

        if len(lines) == 0:
            pass

    return lines


def execute_multiprocessing(line_args, processes, verbose):
    try:
        total_steps = len(line_args)
        feat_all = []
        with Pool(processes) as pool:
            step = 0
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(process_single_line, line_args):
                if BT_DEBUGGING:
                    print('Got result: {}'.format(result), flush=True)

                feat_all.append(result)
                step += 1
                if verbose:
                    print(' "PROGRESS_LABEL Ecosite {} of {}" '.format(step, total_steps), flush=True)

                print('Line processed: {}'.format(step), flush=True)
                print(' %{} '.format(step/total_steps*100), flush=True)

        return feat_all
    except OperationCancelledException:
        print("Operation cancelled")
        return None


if __name__ == '__main__':
    in_args, in_verbose = check_arguments()
    start_time = time.time()
    forest_line_ecosite(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Elapsed time: {}'.format(time.time() - start_time))
