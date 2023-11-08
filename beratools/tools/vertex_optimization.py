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
import multiprocessing
import numpy as np
import math
import time
from pathlib import Path

import uuid
import shapely.geometry as shgeo
from shapely.geometry import shape, mapping, Point, LineString, \
     MultiLineString, GeometryCollection, Polygon
import fiona
import rasterio
import rasterio.mask

from common import *
from dijkstra_algorithm import *

DISTANCE_THRESHOLD = 2  # 1 meter for intersection neighbourhood
SEGMENT_LENGTH = 20  # Distance (meter) from intersection to anchor points
BT_NODATA = -9999


class VertexOptimization:
    def __init__(self, callback, in_line, in_cost, line_radius, out_line, processes, verbose):
        self.in_line = in_line
        self.in_cost = in_cost
        self.line_radius = line_radius
        self.out_line = out_line
        self.processes = processes
        self.verbose = verbose
        self.segment_all = None
        self.in_schema = None  # input shapefile schema

        # calculate cost raster footprint
        footprint_coords = generate_raster_footprint(in_cost, latlon=False)
        self.cost_footprint = Polygon(footprint_coords)

    def execute(self):
        vertex_grp = []
        centerlines = []
        try:
            self.segment_all = self.split_lines(self.in_line)
        except IndexError:
            print(e)

        try:
            vertex_grp = self.group_intersections(self.segment_all)
        except IndexError:
            print(e)

        if PARALLEL_MODE == MODE_MULTIPROCESSING:
            pool = multiprocessing.Pool(self.processes)
            print("Multiprocessing started...")
            print("Using {} CPU cores".format(self.processes))
            centerlines = pool.map(self.process_single_line, vertex_grp)
            pool.close()
            pool.join()
        elif PARALLEL_MODE == MODE_SEQUENTIAL:
            i = 0
            for line in vertex_grp:
                centerline = self.process_single_line(line)
                centerlines.append(centerline)
                # i += 1
                # if i > 2:
                #     break

        return centerlines

    def least_cost_path(self, in_raster, anchors, line_radius):
        line = shgeo.LineString(anchors)

        line_buffer = shape(line).buffer(float(line_radius))
        pt_start = anchors[0]
        pt_end = anchors[-1]

        # buffer clip
        with(rasterio.open(in_raster)) as raster_file:
            out_image, out_transform = rasterio.mask.mask(raster_file, [line_buffer], crop=True, nodata=BT_NODATA)

        ras_nodata = raster_file.meta['nodata']
        if not ras_nodata:
            ras_nodata = BT_NODATA

        matrix, contains_negative = MinCostPathHelper.block2matrix_numpy(out_image[0], ras_nodata)

        if contains_negative:
            raise Exception('ERROR: Raster has negative values.')

        # get row col for points
        ras_transform = rasterio.transform.AffineTransformer(out_transform)

        if type(pt_start[0]) is tuple or type(pt_start[1]) is tuple or \
           type(pt_end[0]) is tuple or type(pt_end[1]) is tuple:
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
        except Exception as e:
            print(e)

        print(" Searching least cost path for line with id", flush=True)
        result = dijkstra_np(start_tuple, end_tuple, matrix)

        if result is None:
            # raise Exception
            return None, None

        if len(result) == 0:
            # raise Exception
            print('No result returned.')
            return None, None

        path_points = None
        for path, costs, end_tuples in result:
            path_points = MinCostPathHelper.create_points_from_path(ras_transform, path,
                                                                    start_tuple[1], end_tuple[1])
            total_cost = costs[-1]

        feat_attr = (start_tuple[2], end_tuple[2], total_cost)
        return LineString(path_points), feat_attr

    # Split LineString to segments at vertices
    def segments(self, line_coords):
        if len(line_coords) == 2:
            line = shape({'type': 'LineString', 'coordinates': line_coords})
            if not np.isclose(line.length, 0.0):
                return [line]
        elif len(line_coords) > 2:
            seg_list = zip(line_coords[:-1], line_coords[1:])
            line_list = [shape({'type': 'LineString', 'coordinates': coords}) for coords in seg_list]
            return [line for line in line_list if not np.isclose(line.length, 0.0)]

        return None

    def split_lines(self, in_line):
        input_lines = []
        with fiona.open(in_line) as open_line_file:
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
                    input_lines.append([shape(line['geometry']), props])
                    i += 1
                else:
                    print('MultiLineString found.')
                    geoms = shape(line['geometry']).geoms
                    for item in geoms:
                        props[BT_UID] = i
                        input_lines.append([shape(item), props])
                        i += 1

        # split line segments at vertices
        input_lines_temp = []
        line_no = 0
        for line in input_lines:
            line_segs = self.segments(list(line[0].coords))
            if line_segs:
                for seg in line_segs:
                    input_lines_temp.append([seg, line_no, line[1], None])
                    line_no += 1

        input_lines = input_lines_temp

        return input_lines

    def update_line_vertex(self, line, index, point):
        if not line:
            return None

        if index >= len(line.coords) or index < -1:
            return line

        coords = list(line.coords)
        coords[index] = (point.x, point.y, 0.0)
        return LineString(coords)

    # only LINESTRING is dealt with for now
    def intersection_of_lines(self, line_1, line_2):
        # intersection collection, may contain points and lines
        inter = None
        if line_1 and line_2:
            inter = line_1.intersection(line_2)

        # TODO: intersection may return GEOMETRYCOLLECTION, LINESTRIMG or MultiLineString
        if inter:
            if type(inter) is GeometryCollection or type(inter) is LineString or type(inter) is MultiLineString:
                return inter.centroid

        return inter

    def closest_point_to_line(self, point, line):
        if not line:
            return None

        pt = line.interpolate(line.project(shgeo.Point(point)))
        return pt

    def append_to_group(self, vertex, vertex_grp, UID):
        """
        Append new vertex to vertex group, by calculating distance to existing vertices
        An anchor point will be added together with line
        """
        pt_added = False

        vertex["lines"][0][2]["UID"] = UID

        # Calculate anchor point for each vertex
        point = shgeo.Point(vertex["point"][0], vertex["point"][1])
        line = vertex["lines"][0][0]
        index = vertex["lines"][0][1]
        pts = self.points_in_line(line)

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
            return

        X = pt_1.x + (pt_2.x - pt_1.x) * SEGMENT_LENGTH / dist_pt
        Y = pt_1.y + (pt_2.y - pt_1.y) * SEGMENT_LENGTH / dist_pt
        vertex["lines"][0].insert(-1, [X, Y])  # add anchor point to list (the third element)

        for item in vertex_grp:
            if abs(point.x - item["point"][0]) < DISTANCE_THRESHOLD and \
               abs(point.y - item["point"][1]) < DISTANCE_THRESHOLD:
                item["lines"].append(vertex["lines"][0])
                pt_added = True

        # Add the first vertex or new vertex not found neighbour
        if not pt_added:
            vertex_grp.append(vertex)

    def points_in_line(self, line):
        point_list = []
        try:
            for point in list(line.coords):  # loops through every point in a line
                # loops through every vertex of every segment
                if point:  # adds all the vertices to segment_list, which creates an array
                    point_list.append(shgeo.Point(point[0], point[1]))
        except Exception as e:
            print(e)

        return point_list

    def group_intersections(self, lines):
        """
        Identify intersections of 2,3 or 4 lines and group them.
        Each group has all the end vertices, start(0) or end (-1) vertex and the line geometry
        Intersection list format: {["point":intersection_pt, "lines":[[line_geom, pt_index, anchor_geom], ...]], ...}
        pt_index: 0 is start vertex, -1 is end vertex
        """
        vertex_grp = []
        i = 0
        try:
            for line in lines:
                point_list = self.points_in_line(line[0])

                if len(point_list) == 0:
                    print("Line {} is empty".format(line[1]))
                    continue

                # Add line to groups based on proximity of two end points to group
                pt_start = {"point": [point_list[0].x, point_list[0].y], "lines": [[line[0], 0, {"lineNo": line[1]}]]}
                pt_end = {"point": [point_list[-1].x, point_list[-1].y], "lines": [[line[0], -1, {"lineNo": line[1]}]]}
                self.append_to_group(pt_start, vertex_grp, line[2][BT_UID])
                self.append_to_group(pt_end, vertex_grp, line[2][BT_UID])
                # print(i)
                i += 1
        except Exception as e:
            # TODO: test traceback
            print(e)

        return vertex_grp

    def get_angle(self, line, end_index):
        """
        Calculate the angle of the first or last segment
        line: ArcPy Polyline
        end_index: 0 or -1 of the the line vertices. Consider the multipart.
        """

        pt = self.points_in_line(line)

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
                angle = math.pi/2
            elif deltaY < 0:
                angle = -math.pi/2
        else:
            angle = np.arctan(deltaY/deltaX)

            # arctan is in range [-pi/2, pi/2], regulate all angles to [[-pi/2, 3*pi/2]]
            if deltaX < 0:
                angle += math.pi  # the second or fourth quadrant

        return angle

    def generate_anchor_pairs(self, vertex):
        """
        Extend line following outward direction to length of SEGMENT_LENGTH
        Use the end point as anchor point.
            vertex: input intersection with all related lines
            return: one or two pairs of anchors according to numbers of lines intersected.
                    two pairs anchors return when 3 or 4 lines intersected
                    one pair anchors return when 1 or 2 lines intersected
        """
        lines = vertex["lines"]
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
                X = vertex["point"][0] - (pt_start_2[0] - vertex["point"][0])
                Y = vertex["point"][1] - (pt_start_2[1] - vertex["point"][1])
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
            X = vertex["point"][0] - (pt_start_1[0] - vertex["point"][0])
            Y = vertex["point"][1] - (pt_start_1[1] - vertex["point"][1])
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

    def process_single_line(self, vertex):
        """
        New version of worklines. It uses memory workspace instead of shapefiles.
        The refactoring is to accelerate the processing speed.
            vertex: intersection with all lines crossed at the intersection
            return: one or two centerlines
        """
        anchors = []
        try:
            anchors = self.generate_anchor_pairs(vertex)
        except Exception as e:
            print(e)

        if not anchors:
            if BT_DEBUGGING:
                print("No anchors retrieved")
            return None

        centerline_1 = None
        centerline_2 = None
        intersection = None

        try:
            if len(anchors) == 4:
                centerline_1, _ = self.least_cost_path(self.in_cost, anchors[0:2], self.line_radius)
                centerline_2, _ = self.least_cost_path(self.in_cost, anchors[2:4], self.line_radius)

                if centerline_1 and centerline_2:
                    intersection = self.intersection_of_lines(centerline_1, centerline_2)
            elif len(anchors) == 2:
                centerline_1, _ = self.least_cost_path(self.in_cost, anchors, self.line_radius)

                if centerline_1:
                    intersection = self.closest_point_to_line(vertex["point"], centerline_1)
        except Exception as e:
            print(e)

        # Update vertices according to intersection, new center lines are returned
        try:
            temp = [anchors, [centerline_1, centerline_2], intersection, vertex]
            print("Processing vertex {} done".format(vertex["point"]))
        except Exception as e:
            print(e)

        return temp


def vertex_optimization(callback, in_line, in_cost, line_radius, out_line, processes, verbose):
    if not compare_crs(vector_crs(in_line), raster_crs(in_cost)):
        return

    tool_vo = VertexOptimization(callback, in_line, in_cost, line_radius, out_line, processes, verbose)
    centerlines = tool_vo.execute()

    # No line generated, exit
    if len(centerlines) <= 0:
        print("No lines generated, exit")
        return

    # Flatten centerlines which is a list of list
    anchor_list = []
    leastcost_list = []
    inter_list = []
    cl_list = []

    # Dump all polylines into point array for vertex updates
    feature_all = {}
    for i in tool_vo.segment_all:
        feature = [i[0], i[2]]
        feature_all[i[1]] = feature

    for sublist in centerlines:
        if not sublist:
            continue
        if len(sublist) > 0:
            for pt in sublist[0]:
                anchor_list.append(Point(pt))
            for line in sublist[1]:
                leastcost_list.append(line)

            inter_list.append(sublist[2])

            for line in sublist[3]["lines"]:
                index = line[1]
                lineNo = line[3]["lineNo"]
                pt_array = feature_all[lineNo][0]

                if not pt_array or not sublist[2]:
                    continue

                new_intersection = sublist[2]

                updated_line = pt_array
                if index == 0 or index == -1:
                    try:
                        updated_line = tool_vo.update_line_vertex(pt_array, index, new_intersection)
                    except Exception as e:
                        print(e)

                feature_all[lineNo][0] = updated_line

    line_path = Path(out_line)
    file_name = line_path.stem
    file_leastcost = line_path.with_stem(file_name + '_leastcost').as_posix()
    file_anchors = line_path.with_stem(file_name + "_anchors").as_posix()
    file_inter = line_path.with_stem(file_name + "_intersections").as_posix()

    fields = []
    properites = []
    all_lines = [value[0] for key, value in feature_all.items()]
    all_props = [value[1] for key, value in feature_all.items()]
    save_features_to_shapefile(out_line, tool_vo.crs, all_lines, tool_vo.in_schema, all_props)
    save_features_to_shapefile(file_leastcost, tool_vo.crs, leastcost_list, fields, properites)
    save_features_to_shapefile(file_anchors, tool_vo.crs, anchor_list, fields, properites)
    save_features_to_shapefile(file_inter, tool_vo.crs, inter_list, fields, properites)


if __name__ == '__main__':
    in_args, in_verbose = check_arguments()
    start_time = time.time()
    vertex_optimization(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Elapsed time: {}'.format(time.time() - start_time))
