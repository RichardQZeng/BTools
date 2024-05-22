#
#    Copyright (C) 2021  Applied Geospatial Research Group
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, version 3.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://gnu.org/licenses/gpl-3.0>.
#
# ---------------------------------------------------------------------------
#
# FLM_VertexOptimization.py
# Script Author: Richard Zeng
# Date: 2021-Oct-26
#
# This script is part of the Forest Line Mapper (FLM) toolset
# Webpage: https://github.com/appliedgrg/flm
#
# Purpose: Move line vertices to right seismic line courses
#
# ---------------------------------------------------------------------------
# System imports
import os
import numpy as np
import math
import time
from pathlib import Path

import uuid
from shapely.geometry import shape, mapping, Point, LineString, \
     MultiLineString, GeometryCollection, Polygon
from shapely import STRtree
import fiona
import rasterio
import rasterio.mask

from dask.distributed import Client, progress, as_completed
import ray
import multiprocessing

from common import *
from dijkstra_algorithm import *

DISTANCE_THRESHOLD = 2  # 1 meter for intersection neighbourhood
SEGMENT_LENGTH = 20  # Distance (meter) from intersection to anchor points


class Vertex:
    def __init__(self, point, line, line_no, end_no, uid):
        self.cost_footprint = None
        self.pt_optimized = None
        self.centerlines = None
        self.anchors = None
        self.in_cost = None
        self.line_radius = None
        self.vertex = {"point": [point.x, point.y], "lines": []}
        self.add_line(line, line_no, end_no, uid)

    # @staticmethod
    # def create_vertex(point, line, line_no, end_no, uid):
    #     vertex = {"point": [point.x, point.y],
    #               "lines": [[line, end_no, {"line_no": line_no}]]}
    #
    #     vertex = VertexGrouping.add_anchors_to_vertex(vertex, uid)
    #
    #     return vertex

    def add_line(self, line, line_no, end_no, uid):
        item = [line, end_no, {"line_no": line_no}]
        item = self.add_anchors_to_line(item, uid)
        if item:
            self.vertex["lines"].append(item)

    @staticmethod
    def get_angle(line, end_index):
        """
        Calculate the angle of the first or last segment
        line: ArcPy Polyline
        end_index: 0 or -1 of the the line vertices. Consider the multipart.
        """
        pt = points_in_line(line)

        if end_index == 0:
            pt_1 = pt[0]
            pt_2 = pt[1]
        elif end_index == -1:
            pt_1 = pt[-1]
            pt_2 = pt[-2]

        deltaX = pt_2.x - pt_1.x
        deltaY = pt_2.y - pt_1.y
        if math.isclose(pt_1.x, pt_2.x, abs_tol=BT_EPSLON):
            angle = math.pi / 2
            if deltaY > 0:
                angle = math.pi / 2
            elif deltaY < 0:
                angle = -math.pi / 2
        else:
            angle = np.arctan(deltaY / deltaX)

            # arctan is in range [-pi/2, pi/2], regulate all angles to [[-pi/2, 3*pi/2]]
            if deltaX < 0:
                angle += math.pi  # the second or fourth quadrant

        return angle

    def add_anchors_to_line(self, line, uid):
        """
        Append new vertex to vertex group, by calculating distance to existing vertices
        An anchor point will be added together with line
        """
        line[2]["UID"] = uid

        # Calculate anchor point for each vertex
        # point = Point(self.vertex["point"][0], self.vertex["point"][1])
        point = Point(self.point())
        line_string = line[0]
        index = line[1]
        pts = points_in_line(line_string)

        pt_1 = None
        pt_2 = None
        if index == 0:
            pt_1 = point
            pt_2 = pts[1]
        elif index == -1:
            pt_1 = point
            pt_2 = pts[-2]

        # Calculate anchor point
        dist_pt = 0.0
        if pt_1 and pt_2:
            dist_pt = pt_1.distance(pt_2)

        # TODO: check why two points are the same
        if np.isclose(dist_pt, 0.0):
            print('Points are close, return')
            return None

        X = pt_1.x + (pt_2.x - pt_1.x) * SEGMENT_LENGTH / dist_pt
        Y = pt_1.y + (pt_2.y - pt_1.y) * SEGMENT_LENGTH / dist_pt
        line.insert(-1, [X, Y])  # add anchor point to list (the third element)

        return line

    def generate_anchor_pairs(self):
        """
        Extend line following outward direction to length of SEGMENT_LENGTH
        Use the end point as anchor point.
            vertex: input intersection with all related lines
            return: one or two pairs of anchors according to numbers of lines intersected.
                    two pairs anchors return when 3 or 4 lines intersected
                    one pair anchors return when 1 or 2 lines intersected
        """
        lines = self.lines()
        point = self.point()
        slopes = []
        for line in lines:
            line_seg = line[0]
            pt_index = line[1]
            slopes.append(self.get_angle(line_seg, pt_index))

        index = 0  # the index of line which paired with first line.
        pt_start_1 = None
        pt_end_1 = None
        pt_start_2 = None
        pt_end_2 = None

        if len(slopes) == 4:
            # get sort order of angles
            index = np.argsort(slopes)

            # first anchor pair (first and third in the sorted array)
            pt_start_1 = lines[index[0]][2]
            pt_end_1 = lines[index[2]][2]

            pt_start_2 = lines[index[1]][2]
            pt_end_2 = lines[index[3]][2]
        elif len(slopes) == 3:
            # find the largest difference between angles
            angle_diff = [abs(slopes[0]-slopes[1]), abs(slopes[0]-slopes[2]), abs(slopes[1]-slopes[2])]
            angle_diff_norm = [2*math.pi-i if i > math.pi else i for i in angle_diff]
            index = np.argmax(angle_diff_norm)
            pairs = [(0, 1), (0, 2), (1, 2)]
            pair = pairs[index]

            # first anchor pair
            pt_start_1 = lines[pair[0]][2]
            pt_end_1 = lines[pair[1]][2]

            # the rest one index
            remain = list({0, 1, 2}-set(pair))[0]  # the remaining index

            try:
                pt_start_2 = lines[remain][2]
                # symmetry point of pt_start_2 regarding vertex["point"]
                X = point[0] - (pt_start_2[0] - point[0])
                Y = point[1] - (pt_start_2[1] - point[1])
                pt_end_2 = [X, Y]
            except Exception as e:
                print(e)

        # this scenario only use two anchors and find the closest point on least cost path
        elif len(slopes) == 2:
            pt_start_1 = lines[0][2]
            pt_end_1 = lines[1][2]
        elif len(slopes) == 1:
            pt_start_1 = lines[0][2]
            # symmetry point of pt_start_1 regarding vertex["point"]
            X = point[0] - (pt_start_1[0] - point[0])
            Y = point[1] - (pt_start_1[1] - point[1])
            pt_end_1 = [X, Y]

        if not pt_start_1 or not pt_end_1:
            print("Anchors not found")

        # if points are outside of cost footprint, set to None
        points = [pt_start_1, pt_end_1, pt_start_2, pt_end_2]
        for index, pt in enumerate(points):
            if pt:
                if not self.cost_footprint.contains(Point(pt)):
                    points[index] = None

        if len(slopes) == 4 or len(slopes) == 3:
            if None in points:
                return None
            else:
                return points
        elif len(slopes) == 2 or len(slopes) == 1:
            if None in (pt_start_1, pt_end_1):
                return None
            else:
                return pt_start_1, pt_end_1

    def optimize(self):
        try:
            self.anchors = self.generate_anchor_pairs()
        except Exception as e:
            print(e)

        if not self.anchors:
            if BT_DEBUGGING:
                print("No anchors retrieved")
            return None

        centerline_1 = None
        centerline_2 = None
        intersection = None

        if CL_USE_SKIMAGE_GRAPH:
            find_lc_path = find_least_cost_path_skimage
        else:
            find_lc_path = find_least_cost_path

        try:
            if len(self.anchors) == 4:
                seed_line = LineString(self.anchors[0:2])
                cost_clip, out_meta = clip_raster(self.in_cost, seed_line, self.line_radius)
                centerline_1 = find_lc_path(cost_clip, out_meta, seed_line)
                seed_line = LineString(self.anchors[2:4])
                cost_clip, out_meta = clip_raster(self.in_cost, seed_line, self.line_radius)
                centerline_2 = find_lc_path(cost_clip, out_meta, seed_line)

                if centerline_1 and centerline_2:
                    intersection = intersection_of_lines(centerline_1, centerline_2)
            elif len(self.anchors) == 2:
                seed_line = LineString(self.anchors)
                cost_clip, out_meta = clip_raster(self.in_cost, seed_line, self.line_radius)
                centerline_1 = find_lc_path(cost_clip, out_meta, seed_line)

                if centerline_1:
                    intersection = closest_point_to_line(self.point(), centerline_1)
        except Exception as e:
            print(e)

        # Update vertices according to intersection, new center lines are returned
        if type(intersection) is MultiPoint:
            intersection = intersection.centroid

        self.centerlines = [centerline_1, centerline_2]
        self.pt_optimized = intersection
        print(f'Processing vertex {self.point()[0]:.2f}, {self.point()[1]:.2f} done')


    def lines(self):
        return self.vertex["lines"]

    def point(self):
        return self.vertex["point"]


class VertexGrouping:
    def __init__(self, callback, in_line, in_cost, line_radius, out_line):
        self.in_line = in_line
        self.in_cost = in_cost
        self.line_radius = float(line_radius)
        self.out_line = out_line
        self.segment_all = []
        self.in_schema = None  # input shapefile schema
        self.crs = None
        self.vertex_grp = []
        self.sindex = None

        # calculate cost raster footprint
        footprint_coords = generate_raster_footprint(self.in_cost, latlon=False)
        self.cost_footprint = Polygon(footprint_coords)

    @staticmethod
    def segments(line_coords):
        """
        Split LineString to segments at vertices
        Parameters
        ----------
        self :
        line_coords :

        Returns
        -------

        """
        if len(line_coords) == 2:
            line = shape({'type': 'LineString', 'coordinates': line_coords})
            if not np.isclose(line.length, 0.0):
                return [line]
        elif len(line_coords) > 2:
            seg_list = zip(line_coords[:-1], line_coords[1:])
            line_list = [shape({'type': 'LineString', 'coordinates': coords}) for coords in seg_list]
            return [line for line in line_list if not np.isclose(line.length, 0.0)]

        return None

    def split_lines(self):
        with fiona.open(self.in_line) as open_line_file:
            # get input shapefile fields
            self.in_schema = open_line_file.meta['schema']
            self.in_schema['properties']['BT_UID'] = 'int:10'  # add field

            i = 0
            self.crs = open_line_file.crs
            for line in open_line_file:
                props = OrderedDict(line['properties'])
                if not line['geometry']:
                    continue
                if line['geometry']['type'] != 'MultiLineString':
                    props[BT_UID] = i
                    self.segment_all.append([shape(line['geometry']), props])
                    i += 1
                else:
                    print('MultiLineString found.')
                    geoms = shape(line['geometry']).geoms
                    for item in geoms:
                        props[BT_UID] = i
                        self.segment_all.append([shape(item), props])
                        i += 1

        # split line segments at vertices
        input_lines_temp = []
        line_no = 0
        for line in self.segment_all:
            line_segs = self.segments(list(line[0].coords))
            if line_segs:
                for seg in line_segs:
                    input_lines_temp.append({'line': shape(seg), 'line_no': line_no, 'prop': line[1],
                                             'start_visited': False, 'end_visited': False})
                    line_no += 1

            print_msg('Splitting lines', line_no, len(self.segment_all))

        self.segment_all = input_lines_temp

        # create spatial index for all line segments
        self.sindex = STRtree([item['line'] for item in self.segment_all])

    def create_vertex_group(self, point, line, line_no, end_no, uid):
        """

        Parameters
        ----------
        point :
        line :
        end_no : head or tail of line, 0, -1

        Returns
        -------

        """
        # all end points not added will be stay with this vertex
        vertex = Vertex(point, line, line_no, end_no, uid)
        search = self.sindex.query(point.buffer(CL_POLYGON_BUFFER))

        # add more vertices to the new group
        for i in search:
            seg = self.segment_all[i]
            if line_no == seg['line_no']:
                continue

            uid = seg['prop']['BT_UID']
            if not seg['start_visited']:
                if self.points_are_close(point, Point(seg['line'].coords[0])):
                    vertex.add_line(seg['line'], seg['line_no'], 0, uid)
                    seg['start_visited'] = True

            if not seg['end_visited']:
                if self.points_are_close(point, Point(seg['line'].coords[-1])):
                    vertex.add_line(seg['line'], seg['line_no'], -1, uid)
                    seg['end_visited'] = True

        vertex.in_cost = self.in_cost
        vertex.line_radius = self.line_radius
        vertex.cost_footprint = self.cost_footprint
        self.vertex_grp.append(vertex)

    @staticmethod
    def points_are_close(pt1, pt2):
        if abs(pt1.x - pt2.x) < DISTANCE_THRESHOLD and abs(pt1.y - pt2.y) < DISTANCE_THRESHOLD:
            return True
        else:
            return False

    def group_vertices(self):
        try:
            self.split_lines()
            print('split_lines done.')

            i = 0
            for line in self.segment_all:
                pt_list = points_in_line(line['line'])
                if len(pt_list) == 0:
                    print(f"Line {line['line_no']} is empty")
                    continue
                uid = line['prop']['BT_UID']
                if not line['start_visited']:
                    self.create_vertex_group(pt_list[0], line['line'], line['line_no'], 0, uid)
                    line['start_visited'] = True
                    i += 1
                    print_msg('Grouping vertices', i, len(self.segment_all))

                if not line['end_visited']:
                    self.create_vertex_group(pt_list[-1], line['line'], line['line_no'], -1, uid)
                    line['end_visited'] = True
                    i += 1
                    print_msg('Grouping vertices', i, len(self.segment_all))

            print('group_intersections done.')

        except Exception as e:
            print(e)


def points_in_line(line):
    point_list = []
    try:
        for point in list(line.coords):  # loops through every point in a line
            # loops through every vertex of every segment
            if point:  # adds all the vertices to segment_list, which creates an array
                point_list.append(Point(point[0], point[1]))
    except Exception as e:
        print(e)

    return point_list


def update_line_vertex(line, index, point):
    if not line:
        return None

    if index >= len(line.coords) or index < -1:
        return line

    coords = list(line.coords)
    if len(coords[index]) == 2:
        coords[index] = (point.x, point.y)
    elif len(coords[index]) == 3:
        coords[index] = (point.x, point.y, 0.0)

    return LineString(coords)


def intersection_of_lines(line_1, line_2):
    """
     only LINESTRING is dealt with for now
    Parameters
    ----------
    line_1 :
    line_2 :

    Returns
    -------

    """
    # intersection collection, may contain points and lines
    inter = None
    if line_1 and line_2:
        inter = line_1.intersection(line_2)

    # TODO: intersection may return GeometryCollection, LineString or MultiLineString
    if inter:
        if (type(inter) is GeometryCollection or
                type(inter) is LineString or
                type(inter) is MultiLineString):
            return inter.centroid

    return inter


def closest_point_to_line(point, line):
    if not line:
        return None

    pt = line.interpolate(line.project(Point(point)))
    return pt


def process_single_line(vertex):
    """
    It uses memory workspace instead of shapefiles.
    The refactoring is to accelerate the processing speed.
        vertex: intersection with all lines crossed at the intersection
        return: optimized vertex
    """
    vertex.optimize()
    return vertex


def vertex_optimization(callback, in_line, in_cost, line_radius, out_line, processes, verbose):
    if not compare_crs(vector_crs(in_line), raster_crs(in_cost)):
        return

    vg = VertexGrouping(callback, in_line, in_cost, line_radius, out_line)
    vg.group_vertices()

    vertices = execute_multiprocessing(process_single_line, 'Vertex Optimization',
                                       vg.vertex_grp, processes, 1, verbose)

    # No line generated, exit
    if len(vertices) <= 0:
        print("No lines optimized.")
        return

    # Flatten vertices which is a list of list
    anchor_list = []
    leastcost_list = []
    inter_list = []
    cl_list = []

    # Dump all polylines into point array for vertex updates
    feature_all = {}
    for i in vg.segment_all:
        feature = [i['line'], i['prop']]
        feature_all[i['line_no']] = feature

    for vertex in vertices:
        if not vertex:
            continue

        if vertex.anchors:
            for pt in vertex.anchors:
                anchor_list.append(Point(pt))

        if vertex.centerlines:
            for line in vertex.centerlines:
                if line:
                    leastcost_list.append(line)

        if vertex.pt_optimized:
            inter_list.append(vertex.pt_optimized)

        for line in vertex.lines():
            index = line[1]
            line_no = line[3]["line_no"]
            pt_array = feature_all[line_no][0]

            if not pt_array or not vertex.pt_optimized:
                continue

            new_intersection = vertex.pt_optimized

            updated_line = pt_array
            if index == 0 or index == -1:
                try:
                    updated_line = update_line_vertex(pt_array, index, new_intersection)
                except Exception as e:
                    print(e)

            feature_all[line_no][0] = updated_line

    line_path = Path(out_line)
    file_name = line_path.stem
    file_lc = line_path.with_stem(file_name + '_leastcost').as_posix()
    file_anchors = line_path.with_stem(file_name + "_anchors").as_posix()
    file_inter = line_path.with_stem(file_name + "_intersections").as_posix()

    fields = []
    properties = []
    all_lines = [value[0] for key, value in feature_all.items()]
    all_props = [value[1] for key, value in feature_all.items()]
    save_features_to_shapefile(out_line, vg.crs, all_lines, vg.in_schema, all_props)
    save_features_to_shapefile(file_lc, vg.crs, leastcost_list, fields, properties)
    save_features_to_shapefile(file_anchors, vg.crs, anchor_list, fields, properties)
    save_features_to_shapefile(file_inter, vg.crs, inter_list, fields, properties)


if __name__ == '__main__':
    in_args, in_verbose = check_arguments()
    start_time = time.time()
    vertex_optimization(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Elapsed time: {}'.format(time.time() - start_time))
