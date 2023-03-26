from random import random
from multiprocessing.pool import Pool
import json
import argparse

import fiona
import numpy as np
import rasterio
import rasterio.mask
from shapely.geometry import shape, mapping

from math import floor
from collections import OrderedDict
from fiona.crs import from_epsg

from PyQt5.QtCore import QVariant
from qgis.core import (
    QgsFeature,
    QgsGeometry,
    QgsPoint,
    QgsPointXY,
    QgsField,
    QgsFields,
    QgsWkbTypes,
    QgsRasterLayer
)

from dijkstra_algorithm import *


class OperationCancelledException(Exception):
    pass


def centerline(callback, in_line, in_cost_raster, line_radius, process_segments, out_center_line):
    cost_raster = QgsRasterLayer(in_cost_raster)
    cost_raster_band = 1

    # Read input line features
    input_lines = []
    with fiona.open(in_line) as open_line_file:
        for line in open_line_file:
            input_lines.append(line['geometry'])

    if process_segments:
        pass

    out_fields = MinCostPathHelper.create_fields()

    # Process lines
    features = []
    for line in input_lines:
        feature = process_algorithm(line, line_radius, in_cost_raster,
                                    cost_raster, cost_raster_band, fields=out_fields)

        if feature:
            features.append(feature)

    # Save lines to shapefile
    out_feature = {
        'geometry': mapping(feature.geometry()),
        'properties': OrderedDict(list(zip(feature.fields().names(), feature.attributes())))
    }

    schema = {
        'geometry': 'LineString',
        'properties': OrderedDict([
            ('start_pt_id', 'int'),
            ('end_pt_id', 'int'),
            ('total_cost', 'float')
        ])
    }
    crs = from_epsg(2956)
    driver = 'ESRI Shapefile'

    out_line_file = fiona.open(out_center_line, 'w', driver, schema, crs)
    for feat in features:
        out_line_file.write(feat)
    del out_line_file

    # execute()
    callback('Centerline tool done.')


class MinCostPathHelper:

    @staticmethod
    def _point_to_row_col(pointxy, raster_layer):
        xres = raster_layer.rasterUnitsPerPixelX()
        yres = raster_layer.rasterUnitsPerPixelY()
        extent = raster_layer.dataProvider().extent()

        col = floor((pointxy.x() - extent.xMinimum()) / xres)
        row = floor((extent.yMaximum() - pointxy.y()) / yres)

        return row, col

    @staticmethod
    def _row_col_to_point(row_col, raster_layer):
        xres = raster_layer.rasterUnitsPerPixelX()
        yres = raster_layer.rasterUnitsPerPixelY()
        extent = raster_layer.dataProvider().extent()

        x = (row_col[1] + 0.5) * xres + extent.xMinimum()
        y = extent.yMaximum() - (row_col[0] + 0.5) * yres
        return QgsPoint(x, y)

    @staticmethod
    def create_points_from_path(cost_raster, min_cost_path, start_point, end_point):
        path_points = list(map(lambda row_col: MinCostPathHelper._row_col_to_point(row_col, cost_raster),
                               min_cost_path))
        path_points[0].setX(start_point.x())
        path_points[0].setY(start_point.y())
        path_points[-1].setX(end_point.x())
        path_points[-1].setY(end_point.y())
        return path_points

    @staticmethod
    def create_fields():
        start_field = QgsField("start_pt_id", QVariant.Int, "int")
        end_field = QgsField("end_pt_id", QVariant.Int, "int")
        cost_field = QgsField("total_cost", QVariant.Double, "double", 10, 3)
        fields = QgsFields()
        fields.append(start_field)
        fields.append(end_field)
        fields.append(cost_field)
        return fields

    @staticmethod
    def create_path_feature_from_points(path_points, attr_vals, fields):
        polyline = QgsGeometry.fromPolyline(path_points)
        feature = QgsFeature(fields)

        start_index = feature.fieldNameIndex("start_pt_id")
        end_index = feature.fieldNameIndex("end_pt_id")
        cost_index = feature.fieldNameIndex("total_cost")
        feature.setAttribute(start_index, attr_vals[0])
        feature.setAttribute(end_index, attr_vals[1])
        feature.setAttribute(cost_index, attr_vals[2])  # cost
        feature.setGeometry(polyline)
        return feature

    @staticmethod
    def features_to_tuples(point_features, raster_layer):
        row_cols = []

        extent = raster_layer.dataProvider().extent()

        for point_feature in point_features:
            if point_feature.hasGeometry():
                point_geom = point_feature.geometry()

                if point_geom.wkbType() == QgsWkbTypes.MultiPoint:
                    multi_points = point_geom.asMultiPoint()
                    for pointxy in multi_points:
                        if extent.contains(pointxy):
                            row_col = MinCostPathHelper._point_to_row_col(pointxy, raster_layer)
                            row_cols.append((row_col, pointxy, point_feature.id()))
                elif point_geom.wkbType() == QgsWkbTypes.Point:
                    pointxy = point_geom.asPoint()
                    if extent.contains(pointxy):
                        row_col = MinCostPathHelper._point_to_row_col(pointxy, raster_layer)
                        row_cols.append((row_col, pointxy, point_feature.id()))

        return row_cols

    @staticmethod
    def get_all_block(raster_layer, band_num):
        provider = raster_layer.dataProvider()
        extent = provider.extent()

        xres = raster_layer.rasterUnitsPerPixelX()
        yres = raster_layer.rasterUnitsPerPixelY()
        width = floor((extent.xMaximum() - extent.xMinimum()) / xres)
        height = floor((extent.yMaximum() - extent.yMinimum()) / yres)
        return provider.block(band_num, extent, width, height)

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


def process_algorithm(line, line_radius, in_raster, cost_raster, cost_raster_band,
                      find_nearest=True, output_linear_reference=False, fields=None):
    if cost_raster is None:
        raise Exception('Cost raster is not valid')
    if cost_raster_band is None:
        raise Exception('Cost raster band is not valid')

    if cost_raster.rasterType() not in [cost_raster.Multiband, cost_raster.GrayOrUndefined]:
        raise Exception

    line_buffer = shape(line).buffer(float(line_radius))
    pt_start = line['coordinates'][0]
    pt_end = line['coordinates'][-1]
    geom_start = QgsGeometry.fromPointXY(QgsPointXY(pt_start[0], pt_start[1]))
    geom_end = QgsGeometry.fromPointXY(QgsPointXY(pt_end[0], pt_end[1]))

    feat_start = QgsFeature(fields)
    feat_start.setId(0)
    feat_start.setGeometry(geom_start)
    feat_end = QgsFeature(fields)
    feat_end.setId(1)
    feat_end.setGeometry(geom_end)

    start_features = [feat_start]
    end_features = [feat_end]

    # start_features = list(start_source.getFeatures())
    # print(str(len(start_features)))
    #
    # end_features = list(end_source.getFeatures())
    # print(str(len(end_features)))

    start_tuples = MinCostPathHelper.features_to_tuples(start_features, cost_raster)
    if len(start_tuples) == 0:
        raise Exception("ERROR: The start-point layer contains no legal point.")
    start_tuple = start_tuples[0]

    end_tuples = MinCostPathHelper.features_to_tuples(end_features, cost_raster)
    if len(end_tuples) == 0:
        raise Exception("ERROR: The end-point layer contains no legal point.")
    # end_tuple = end_tuples[0]

    block = MinCostPathHelper.get_all_block(cost_raster, cost_raster_band)
    # buffer clip
    with(rasterio.open(in_raster)) as raster_file:
        out_image, out_transform = rasterio.mask.mask(raster_file, [line_buffer], crop=True)
    matrix, contains_negative = MinCostPathHelper.block2matrix_numpy(out_image[0], raster_file.meta['nodata'])
    print("The size of cost raster is: {}x{}".format(block.height(), block.width()))

    if contains_negative:
        raise Exception('ERROR: Raster has negative values.')

    # get row col for points
    ras_transform = rasterio.transform.AffineTransformer(out_transform)
    start_tuples = [(ras_transform.rowcol(pt_start[0], pt_start[1]), QgsPointXY(pt_start[0], pt_start[1]), 0)]
    end_tuples = [(ras_transform.rowcol(pt_end[0], pt_end[1]), QgsPointXY(pt_end[0], pt_end[1]), 0)]
    start_tuple = start_tuples[0]

    print("Searching least cost path...")
    result = dijkstra(start_tuple, end_tuples, matrix, find_nearest)

    if result is None:
        raise Exception

    if len(result) == 0:
        raise Exception

    for path, costs, end_tuples in result:
        for end_tuple in end_tuples:
            path_points = MinCostPathHelper.create_points_from_path(cost_raster, path,
                                                                    start_tuple[1], end_tuple[1])
            if output_linear_reference:
                # add linear reference
                for point, cost in zip(path_points, costs):
                    point.addMValue(cost)

            total_cost = costs[-1]
            path_feature = None
            path_feature = MinCostPathHelper.create_path_feature_from_points(path_points,
                                                                             (start_tuple[2], end_tuple[2], total_cost),
                                                                             fields)
    return path_feature


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
def execute():
    # create and configure the process pool
    data = [[random() for n in range(100)] for i in range(300)]
    try:
        total_steps = 300
        with Pool() as pool:
            step = 0
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(process_line, data):
                print('Got result: {}'.format(result), flush=True)
                step += 1
                print(step)
                print('%{}'.format(step/total_steps*100))
                # if result > 0.9:
                #     print('Pool terminated.')
                #     raise OperationCancelledException()

    except OperationCancelledException:
        print("Operation cancelled")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=json.loads)
    args = parser.parse_args()

    centerline(print, **args.input)
