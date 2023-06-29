from collections import OrderedDict
from multiprocessing.pool import Pool

import numpy as np
import time

import pyproj
import rasterio
import rasterio.mask
import fiona
from fiona import Geometry
from osgeo import gdal, ogr
from shapely.geometry import shape, mapping, LineString, Point

from dijkstra_algorithm import *
from common import *

import line_profiler
profile = line_profiler.LineProfiler()


class OperationCancelledException(Exception):
    pass


def compare_crs(in_line, in_cost_raster):
    line_crs = None
    ras_crs = None
    in_line_file = ogr.Open(in_line)
    line_crs = in_line_file.GetLayer().GetSpatialRef()

    cost_raster_file = gdal.Open(in_cost_raster)
    ras_crs = cost_raster_file.GetSpatialRef()

    del in_line_file
    del cost_raster_file

    if line_crs and ras_crs:
        if line_crs.IsSameGeogCS(ras_crs):
            print('Check: Input file Spatial Reference are the same, continue.')
            return True
        else:
            line_crs_norm = pyproj.CRS(line_crs.ExportToWkt())
            ras_crs_norm = pyproj.CRS(ras_crs.ExportToWkt())
            if ras_crs_norm.name == line_crs_norm.name:
                print('Same crs, continue.')
                return True

    return False


def centerline(callback, in_line, in_cost, line_radius,
               proc_segments, out_line, processes, verbose):
    if not compare_crs(in_line, in_cost):
        print("Line and CHM spatial references are not same, please check.")
        return

    # Read input line features
    layer_crs = None
    input_lines = []
    with fiona.open(in_line) as open_line_file:
        layer_crs = open_line_file.crs
        for line in open_line_file:
            if line.geometry['type'] != 'MultiLineString':
                input_lines.append(line['geometry'])
            else:
                print('MultiLineString found.')
                geoms = shape(line['geometry']).geoms
                for item in geoms:
                    line_part = Geometry.from_dict(item)
                    if line_part:
                        input_lines.append(line_part)

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
    if USE_MULTI_PROCESSING:
        features = execute_multiprocessing(all_lines, processes, verbose)
    else:
        total_steps = len(all_lines)
        step = 0
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

    out_line_file = fiona.open(out_line, 'w', driver, schema, layer_crs.to_proj4())
    for feature in fiona_features:
        out_line_file.write(feature)
    del out_line_file


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
    def block2matrix_numpy(block, nodata):
        contains_negative = False
        with np.nditer(block, flags=["refs_ok"], op_flags=['readwrite']) as it:
            for x in it:
                if np.isclose(x, nodata) or np.isnan(x):
                    x[...] = 9999.0
                elif x < 0:
                    contains_negative = True

        return block, contains_negative

    @staticmethod
    def block2matrix(block, nodata):
        contains_negative = False
        width, height = block.shape
        # TODO: deal with nodata
        matrix = [[None if np.isclose(block[i][j], nodata) or np.isclose(block[i][j], BT_NODATA)
                   else block[i][j] for j in range(height)] for i in range(width)]

        for l in matrix:
            for v in l:
                if v is not None:
                    if v < 0 and not np.isclose(v, BT_NODATA):
                        contains_negative = True

        return matrix, contains_negative


# @profile
def process_single_line(line_args, find_nearest=True, output_linear_reference=False):
    line = line_args[0]
    line_radius = line_args[1]
    in_raster = line_args[2]

    line_buffer = shape(line).buffer(float(line_radius))
    pt_start = line['coordinates'][0]
    pt_end = line['coordinates'][-1]

    # buffer clip
    with(rasterio.open(in_raster)) as raster_file:
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
    except Exception as e:
        print(e)

    print(" Searching least cost path for line with id: {} ".format(line_args[3]), flush=True)

    if USE_NUMPY_FOR_DIJKSTRA:
        result = dijkstra_np(start_tuple, end_tuples, matrix)
    else:
        result = dijkstra(start_tuple, end_tuples, matrix, find_nearest)

    if result is None:
        # raise Exception
        return None, None

    if len(result) == 0:
        # raise Exception
        print('No result returned.')
        return None, None

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
