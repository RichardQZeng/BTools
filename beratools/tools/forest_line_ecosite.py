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
    in_properties = None
    with fiona.open(in_line) as in_file_vector:
        layer_crs = in_file_vector.crs
        in_properties = in_file_vector.meta['schema']['properties']
        for line in in_file_vector:
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

    out_fields_list = OrderedDict(in_properties)
    out_fields_list["ecosite"] = 'str'

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
        line_prop = line[1]
        index_query = tree.query(line_geom)
        geoms_intersected = []
        for i in index_query:
            geoms_intersected.append({'geom': geoms[i], 'prop': feats[i][1]})  # polygon has property
        all_lines.append(({'geom': line_geom, 'prop': line_prop}, geoms_intersected, id))
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
                    'geometry': mapping(line[0]),
                    'properties': line[1]  # TODO: add attributes
                }
            except Exception as e:
                print(e)
            else:
                fiona_features.append(single_line)

    schema = {
        'geometry': 'LineString',
        'properties': out_fields_list
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


# @profile
def process_single_line(line_args, find_nearest=True, output_linear_reference=False):
    """
    Parameters
    ----------
    line_args : tuple
        line_args has three items: {line geometry, line properties}, intersected polygons (list) and line ID

    Returns
    --------
    list
        The return list consist of split lines by intersection with polygons

    """
    in_geom = line_args[0]
    polygons = line_args[1]

    out_geom = [in_geom['geom']]
    if len(polygons) > 0:  # none intersecting polygons
        for poly in polygons:
            out_geom = split_line_with_polygon(out_geom, poly['geom'])

    final_geoms = []
    if len(out_geom) > 0:
        for i in out_geom:
            temp_prop = in_geom['prop']
            for j in polygons:
                if j['geom'].contains(i):
                    temp_prop['ecosite'] = j['prop']['ecosite']  # TODO: specify 'ecosite' field name
                    final_geoms.append([i, temp_prop])


    return final_geoms


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
