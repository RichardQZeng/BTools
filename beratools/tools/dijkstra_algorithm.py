# -*- coding: utf-8 -*-

"""
/***************************************************************************
 LeastCostPath Algorithm
 This algorithm is adapted from a QGIS plugin:
 Find the least cost path with given cost raster and points
 Original author: FlowMap Group@SESS.PKU
 Source code repository: https://github.com/Gooong/LeastCostPath
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
"""

__author__ = 'Richard Zeng'
__date__ = '2023-03-01'
__copyright__ = '(C) 2023 by AppliedGRG'

# This will get replaced with a git SHA1 when you do a git archive
__revision__ = '$Format:%H$'

import numpy as np
from skimage.graph import route_through_array
from math import sqrt
import queue
import collections
from common import *

sqrt2 = sqrt(2)


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


def dijkstra(start_tuple, end_tuples, block, find_nearest, feedback=None):
    class Grid:
        def __init__(self, matrix):
            self.map = matrix
            self.h = len(matrix)
            self.w = len(matrix[0])
            self.manhattan_boundry = None
            self.curr_boundry = None

        def _in_bounds(self, id):
            x, y = id
            return 0 <= x < self.h and 0 <= y < self.w

        def _passable(self, id):
            x, y = id
            return self.map[x][y] is not None

        def is_valid(self, id):
            return self._in_bounds(id) and self._passable(id)

        def neighbors(self, id):
            x, y = id
            results = [(x + 1, y), (x, y - 1), (x - 1, y), (x, y + 1),
                       (x + 1, y - 1), (x + 1, y + 1), (x - 1, y - 1), (x - 1, y + 1)]
            results = list(filter(self.is_valid, results))
            return results

        @staticmethod
        def manhattan_distance(id1, id2):
            x1, y1 = id1
            x2, y2 = id2
            return abs(x1 - x2) + abs(y1 - y2)

        @staticmethod
        def min_manhattan(curr_node, end_nodes):
            return min(map(lambda node: Grid.manhattan_distance(curr_node, node), end_nodes))

        @staticmethod
        def max_manhattan(curr_node, end_nodes):
            return max(map(lambda node: Grid.manhattan_distance(curr_node, node), end_nodes))

        @staticmethod
        def all_manhattan(curr_node, end_nodes):
            return {end_node: Grid.manhattan_distance(curr_node, end_node) for end_node in end_nodes}

        def simple_cost(self, cur, nex):
            cx, cy = cur
            nx, ny = nex
            currV = self.map[cx][cy]
            offsetV = self.map[nx][ny]
            if cx == nx or cy == ny:
                return (currV + offsetV) / 2
            else:
                return sqrt2 * (currV + offsetV) / 2

    result = []
    grid = Grid(block)

    end_dict = collections.defaultdict(list)
    for end_tuple in end_tuples:
        end_dict[end_tuple[0]].append(end_tuple)
    end_row_cols = set(end_dict.keys())
    end_row_col_list = list(end_row_cols)
    start_row_col = start_tuple[0]

    frontier = queue.PriorityQueue()
    frontier.put((0, start_row_col))
    came_from = {}
    cost_so_far = {}
    decided = set()

    if not grid.is_valid(start_row_col):
        return result

    # init progress
    index = 0
    distance_dic = grid.all_manhattan(start_row_col, end_row_cols)
    if find_nearest:
        total_manhattan = min(distance_dic.values())
    else:
        total_manhattan = sum(distance_dic.values())

    total_manhattan = total_manhattan + 1
    bound = total_manhattan
    if feedback:
        feedback.setProgress(1 + 100 * (1 - bound / total_manhattan))

    came_from[start_row_col] = None
    cost_so_far[start_row_col] = 0

    while not frontier.empty():
        _, current_node = frontier.get()
        if current_node in decided:
            continue
        decided.add(current_node)

        # update the progress bar
        if feedback:
            if feedback.isCanceled():
                return None

            index = (index + 1) % len(end_row_col_list)
            target_node = end_row_col_list[index]
            new_manhattan = grid.manhattan_distance(current_node, target_node)
            if new_manhattan < distance_dic[target_node]:
                if find_nearest:
                    curr_bound = new_manhattan
                else:
                    curr_bound = bound - (distance_dic[target_node] - new_manhattan)

                distance_dic[target_node] = new_manhattan

                if curr_bound < bound:
                    bound = curr_bound
                    if feedback:
                        feedback.setProgress(1 + 100 * (1 - bound / total_manhattan)*(1 - bound / total_manhattan))

        # reacn destination
        if current_node in end_row_cols:
            path = []
            costs = []
            traverse_node = current_node
            while traverse_node is not None:
                path.append(traverse_node)
                costs.append(cost_so_far[traverse_node])
                traverse_node = came_from[traverse_node]
            
            # start point and end point overlaps
            if len(path) == 1:
                path.append(start_row_col)
                costs.append(0.0)
            path.reverse()
            costs.reverse()
            result.append((path, costs, end_dict[current_node]))

            end_row_cols.remove(current_node)
            end_row_col_list.remove(current_node)
            if len(end_row_cols) == 0 or find_nearest:
                break

        # relax distance
        for nex in grid.neighbors(current_node):
            new_cost = cost_so_far[current_node] + grid.simple_cost(current_node, nex)
            if nex not in cost_so_far or new_cost < cost_so_far[nex]:
                cost_so_far[nex] = new_cost
                frontier.put((new_cost, nex))
                came_from[nex] = current_node

    return result

#########################################################
# Numpy based algorithm


def valid_node(node, size_of_grid):
    """Checks if node is within the grid boundaries."""
    if node[0] < 0 or node[0] >= size_of_grid:
        return False
    if node[1] < 0 or node[1] >= size_of_grid:
        return False
    return True


def up(node):
    return node[0] - 1, node[1]


def down(node):
    return node[0] + 1, node[1]


def left(node):
    return node[0], node[1] - 1


def right(node):
    return node[0], node[1] + 1


def backtrack(initial_node, desired_node, distances):
    # idea start at the last node then choose the least number of steps to go back
    # last node
    path = [desired_node]

    size_of_grid = distances.shape[0]

    while True:
        # check up down left right - choose the direction that has the least distance
        potential_distances = []
        potential_nodes = []

        directions = [up, down, left, right]

        for direction in directions:
            node = direction(path[-1])
            if valid_node(node, size_of_grid):
                potential_nodes.append(node)
                potential_distances.append(distances[node[0], node[1]])

        # least_distance_index = np.argmin(potential_distances)
        print(potential_nodes)

        least_distance_index = np.argsort(potential_distances)

        pt_added = False
        for index in least_distance_index:
            p_point = potential_nodes[index]
            if p_point == (1, 6):
                pass
            if p_point not in path:
                path.append(p_point)
                pt_added = True
                break

        if index >= len(potential_distances)-1 and not pt_added:
            print("No best path found.")
            return

        if path[-1][0] == initial_node[0] and path[-1][1] == initial_node[1]:
            break

    return list(reversed(path))


def dijkstra_np(start_tuple, end_tuple, matrix):
    """Dijkstras algorithm for finding the shortest path between two nodes in a graph.

    Args:
        start_node (list): [row,col] coordinates of the initial node
        end_node (list): [row,col] coordinates of the desired node
        matrix (array 2d): 2d numpy array that contains any matrix as 1s and free space as 0s

    Returns:
        list[list]: list of list of nodes that form the shortest path
    """

    # source and destination are free
    start_node = start_tuple[0]
    end_node = end_tuple[0]
    path = None
    costs = None

    try:
        matrix[start_node[0], start_node[1]] = 0
        matrix[end_node[0], end_node[1]] = 0

        path, cost = route_through_array(matrix, start_node, end_node)
        costs = [0.0 for i in range(len(path))]
    except Exception as e:
        print(e)
        return None

    return [(path, costs, end_tuple)]


def find_least_cost_path(ras_nodata, out_image, out_transform, line_id, line,
                         find_nearest=True, output_linear_reference=False):
    pt_start = line.coords[0]
    pt_end = line.coords[-1]

    image_shp = out_image.shape
    image_data = out_image[0]
    if len(image_shp) == 2:
        image_data = out_image

    if USE_NUMPY_FOR_DIJKSTRA:
        matrix, contains_negative = MinCostPathHelper.block2matrix_numpy(image_data, ras_nodata)
    else:
        matrix, contains_negative = MinCostPathHelper.block2matrix(image_data, ras_nodata)

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
        mat_size = (mat_size[0] - 1, mat_size[0] - 1)
        start_tuple = (min(start_tuple[0], mat_size), start_tuple[1], start_tuple[2])
        end_tuple = (min(end_tuple[0], mat_size), end_tuple[1], end_tuple[2])

    except Exception as e:
        print(e)

    print(" Searching least cost path for line with id: {} ".format(line_id), flush=True)

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
