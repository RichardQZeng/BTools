from random import random
from multiprocessing.pool import Pool
import json
import argparse

import fiona
import numpy as np
import rasterio
import rasterio.mask
from shapely.geometry import shape, mapping, LineString, Point

from collections import OrderedDict
from fiona.crs import CRS

from dijkstra_algorithm import *

USE_MULTI_PROCESSING = True


class OperationCancelledException(Exception):
    pass


def centerline(callback, in_line, in_cost_raster, line_radius, process_segments, out_center_line):
    # Read input line features
    input_lines = []
    with fiona.open(in_line) as open_line_file:
        for line in open_line_file:
            input_lines.append(line['geometry'])

    if process_segments:
        pass

    out_fields_list = ["start_pt_id", "end_pt_id", "total_cost"]

    # Process lines
    fiona_features = []
    all_lines = []
    features = []
    for line in input_lines:
        all_lines.append((line, line_radius, in_cost_raster))

    if USE_MULTI_PROCESSING:
        features = execute(all_lines)
    else:
        for line in all_lines:
            feat_geometry, feat_attributes = process_algorithm(line)
            if feat_geometry and feat_attributes:
                features.append((feat_geometry, feat_attributes))

    for feature in features:
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

    # TODO correct EPSG code
    layer_crs = CRS.from_epsg(2956)

    driver = 'ESRI Shapefile'

    out_line_file = fiona.open(out_center_line, 'w', driver, schema, layer_crs.to_proj4())
    for feature in fiona_features:
        out_line_file.write(feature)
    del out_line_file

    callback('Centerline tool done.')


class MinCostPathHelper:

    @staticmethod
    def _point_to_row_col(pointxy, ras_transform):
        col, row = ras_transform.rowcol(pointxy.x(), pointxy.y())

        return row, col

    @staticmethod
    def _row_col_to_point(row_col, ras_transform):
        x, y = ras_transform.xy(row_col[0], row_col[1])
        return x, y

    @staticmethod
    def create_points_from_path(ras_transform, min_cost_path, start_point, end_point):
        path_points = list(map(lambda row_col: MinCostPathHelper._row_col_to_point(row_col, ras_transform),
                               min_cost_path))
        path_points[0] = (start_point.x, start_point.y)
        path_points[-1] = (end_point.x, end_point.y)
        return path_points

    @staticmethod
    def create_path_feature_from_points(path_points, attr_vals):
        path_points_raw = [[pt.x, pt.y] for pt in path_points]

        return LineString(path_points_raw), attr_vals

    @staticmethod
    def block2matrix(block):
        contains_negative = False
        matrix = [[None if block.isNoData(i, j) else block.value(i, j) for j in range(block.width())]
                  for i in range(block.height())]

        for l in matrix:
            for v in l:
                if v is not None:
                    if v < 0:
                        contains_negative = True

        return matrix, contains_negative

    @staticmethod
    def block2matrix_numpy(block, nodata):
        contains_negative = False
        width, height = block.shape
        matrix = [[None if np.isclose(block[i][j], nodata) else block[i][j] for j in range(height)]
                  for i in range(width)]

        for l in matrix:
            for v in l:
                if v is not None:
                    if v < 0:
                        contains_negative = True

        return matrix, contains_negative


def process_algorithm(line_args, find_nearest=True, output_linear_reference=False):
    line = line_args[0]
    line_radius = line_args[1]
    in_raster = line_args[2]

    line_buffer = shape(line).buffer(float(line_radius))
    pt_start = line['coordinates'][0]
    pt_end = line['coordinates'][-1]

    # buffer clip
    with(rasterio.open(in_raster)) as raster_file:
        out_image, out_transform = rasterio.mask.mask(raster_file, [line_buffer], crop=True)
    matrix, contains_negative = MinCostPathHelper.block2matrix_numpy(out_image[0], raster_file.meta['nodata'])

    if contains_negative:
        raise Exception('ERROR: Raster has negative values.')

    # get row col for points
    ras_transform = rasterio.transform.AffineTransformer(out_transform)

    # TODO: the last element in tuple is point ID
    start_tuples = [(ras_transform.rowcol(pt_start[0], pt_start[1]), Point(pt_start[0], pt_start[1]), 0)]
    end_tuples = [(ras_transform.rowcol(pt_end[0], pt_end[1]), Point(pt_end[0], pt_end[1]), 1)]
    start_tuple = start_tuples[0]

    print("Searching least cost path...")
    result = dijkstra(start_tuple, end_tuples, matrix, find_nearest)

    if result is None:
        raise Exception

    if len(result) == 0:
        raise Exception

    path_points = None
    for path, costs, end_tuples in result:
        for end_tuple in end_tuples:
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


# task executed in a worker process
def process_line(line, input_raster):
    line_geom = shape(line)
    buffer = line_geom.buffer(line_radius)

    with rasterio.open(input_raster) as src_raster:
        out_image, out_transform = rasterio.mask.mask(src_raster, [buffer], crop=True)
        out_meta = src_raster.meta

    out_meta.update({"driver": "GTiff",
                     "height": out_image.shape[1],
                     "width": out_image.shape[2],
                     "transform": out_transform})

    # with rasterio.open("RGB.byte.masked.tif", "w", **out_meta) as dest:
    #     dest.write(out_image)

    # return the generated value
    value = None
    return value


# protect the entry point
def execute(line_args):
    try:
        total_steps = len(line_args)
        features = []
        with Pool() as pool:
            step = 0
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(process_algorithm, line_args):
                print('Got result: {}'.format(result), flush=True)
                features.append(result)
                step += 1
                print(step)
                print('%{}'.format(step/total_steps*100))
                # if result > 0.9:
                #     print('Pool terminated.')
                #     raise OperationCancelledException()
        return features
    except OperationCancelledException:
        print("Operation cancelled")
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=json.loads)
    args = parser.parse_args()

    centerline(print, **args.input)
