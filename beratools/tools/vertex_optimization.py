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

import uuid
import shapely.geometry as shgeo
from shapely.geometry import shape, mapping, Point, LineString, MultiLineString
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

    def execute(self):
        vertex_grp = []
        centerlines = []
        try:
            self.segment_all = self.split_lines(self.in_line)
        except IndexError:
            print(e)

        try:
            vertex_grp = self.groupIntersections(self.segment_all)
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
                i += 1
                if i > 2:
                    break

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

        print(" Searching least cost path for line with id", flush=True)
        result = dijkstra_np(start_tuple, end_tuples, matrix)

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

                total_cost = costs[-1]

        feat_attr = (start_tuple[2], end_tuple[2], total_cost)
        return LineString(path_points), feat_attr

    # Split LineString to segments at vertices
    def segments(self, line_coords):
        if len(line_coords) < 2:
            return None
        elif len(line_coords) == 2:
            return [shape({'type': 'LineString', 'coordinates': line_coords})]
        else:
            seg_list = zip(line_coords[:-1], line_coords[1:])
            line_list = [{'type': 'LineString', 'coordinates': coords} for coords in seg_list]
            return [shape(line) for line in line_list]


    def split_lines(self, in_line):
        input_lines = []
        with fiona.open(in_line) as open_line_file:
            layer_crs = open_line_file.crs
            for line in open_line_file:
                if line['geometry']['type'] != 'MultiLineString':
                    input_lines.append([shape(line['geometry']), dict(line['properties'])])
                else:
                    print('MultiLineString found.')
                    geoms = shape(line['geometry']).geoms
                    for item in geoms:
                        input_lines.append([shape(item), dict(line['properties'])])

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

        if index >= len(line) or index < 0:
            return line

        coords = list(line.coords)
        coords[index] = point
        return LineString(coords)

    def intersectionOfLines(self, line_1, line_2):
        # intersection collection, may contain points and lines
        inter = None
        if line_1 and line_2:
            inter = line_1.intersection(line_2)

        if inter:
            return inter.centroid.x, inter.centroid.y

        return inter

    def closestPointToLine(self, point, line):
        if not line:
            return None

        pt = line.interpolate(line.project(shgeo.Point(point)))
        return pt

    def appendToGroup(self, vertex, vertex_grp, UID):
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
        pts = self.ptsInLine(line)

        if index == 0:
            pt_1 = point
            pt_2 = pts[1]
        elif index == -1:
            pt_1 = point
            pt_2 = pts[-2]

        # Calculate anchor point
        dist_pt = math.sqrt(sum((px - qx) ** 2.0 for px, qx in zip([pt_1.x, pt_1.y], [pt_2.x, pt_2.y])))
        X = pt_1.x + (pt_2.x - pt_1.x) * SEGMENT_LENGTH / dist_pt
        Y = pt_1.y + (pt_2.y - pt_1.y) * SEGMENT_LENGTH / dist_pt
        vertex["lines"][0].insert(-1, [X, Y])  # add anchor point to list (the third element)

        for item in vertex_grp:
            if abs(point.x - item["point"][0]) < DISTANCE_THRESHOLD and abs(
                    point.y - item["point"][1]) < DISTANCE_THRESHOLD:
                item["lines"].append(vertex["lines"][0])
                pt_added = True

        # Add the first vertex or new vertex not found neighbour
        if not pt_added:
            vertex_grp.append(vertex)

    def ptsInLine(self, line):
        point_list = []
        try:
            for point in list(line.coords):  # loops through every point in a line
                # loops through every vertex of every segment
                if point:  # adds all the vertices to segment_list, which creates an array
                    point_list.append(shgeo.Point(point[0], point[1]))
        except Exception as e:
            print(e)

        return point_list

    def groupIntersections(self, lines):
        """
        Identify intersections of 2,3 or 4 lines and group them.
        Each group has all the end vertices, start(0) or end (-1) vertex and the line geometry
        Intersection list format: {["point":intersection_pt, "lines":[[line_geom, pt_index, anchor_geom], ...]], ...}
        pt_index: 0 is start vertex, -1 is end vertex
        """
        vertex_grp = []
        try:
            for line in lines:
                point_list = self.ptsInLine(line[0])

                if len(point_list) == 0:
                    print("Line {} is empty".format(line[1]))
                    continue

                # Add line to groups based on proximity of two end points to group
                pt_start = {"point": [point_list[0].x, point_list[0].y], "lines": [[line[0], 0, {"lineNo": line[1]}]]}
                pt_end = {"point": [point_list[-1].x, point_list[-1].y], "lines": [[line[0], -1, {"lineNo": line[1]}]]}
                self.appendToGroup(pt_start, vertex_grp, line[2]['Id'])
                self.appendToGroup(pt_end, vertex_grp, line[2]['Id'])
        except Exception as e:
            # TODO: test traceback
            print(e)

        return vertex_grp

    def getAngle(self, line, end_index):
        """
        Calculate the angle of the first or last segment
        line: ArcPy Polyline
        end_index: 0 or -1 of the the line vertices. Consider the multipart.
        """

        pt = self.ptsInLine(line)

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

    def generateAnchorPairs(self, vertex):
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
            slopes.append(self.getAngle(line_seg, pt_index))

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

        if len(slopes) == 4 or len(slopes) == 3:
            return pt_start_1, pt_end_1, pt_start_2, pt_end_2
        elif len(slopes) == 2 or len(slopes) == 1:
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
            anchors = self.generateAnchorPairs(vertex)
        except Exception as e:
            print(e)

        if not anchors:
            print("No anchors retrieved")
            return None

        centerline_1 = [None]
        centerline_2 = [None]
        intersection = None

        try:
            if len(anchors) == 4:
                # centerline_1 = leastCostPath(Cost_Raster, anchors[0:2], line_processing_radius)
                # centerline_2 = leastCostPath(Cost_Raster, anchors[2:4], line_processing_radius)
                centerline_1, _ = self.least_cost_path(self.in_cost, anchors[0:2], self.line_radius)
                centerline_2, _ = self.least_cost_path(self.in_cost, anchors[2:4], self.line_radius)

                if centerline_1 and centerline_2:
                    intersection = self.intersectionOfLines(centerline_1, centerline_2)
            elif len(anchors) == 2:
                # centerline_1 = leastCostPath(cost_raster, anchors, line_processing_radius)
                centerline_1, _ = self.least_cost_path(self.in_cost, anchors, self.line_radius)

                if centerline_1:
                    intersection = self.closestPointToLine(vertex["point"], centerline_1)
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
    # Prepare input lines for multiprocessing
    # fields = flmc.GetAllFieldsFromShp(Forest_Line_Feature_Class)

    tool_vo = VertexOptimization(callback, in_line, in_cost, line_radius, out_line, processes, verbose)
    centerlines = tool_vo.execute()

    # No line generated, exit
    if len(centerlines) <= 0:
        print("No lines generated, exit")
        return

    # # Create output centerline shapefile
    # flmc.log("Create centerline shapefile...")
    # arcpy.CreateFeatureclass_management(os.path.dirname(Out_Centerline), os.path.basename(Out_Centerline),
    #                                     "POLYLINE", Forest_Line_Feature_Class, "DISABLED",
    #                                     "DISABLED", Forest_Line_Feature_Class)
    #
    # # write out new intersections
    # file_name = os.path.splitext(Out_Centerline)
    #
    # file_leastcost = file_name[0] + "_leastcost" + file_name[1]
    # arcpy.CreateFeatureclass_management(os.path.dirname(file_leastcost), os.path.basename(file_leastcost),
    #                                     "POLYLINE", "", "DISABLED", "DISABLED", Forest_Line_Feature_Class)
    #
    # file_anchors = file_name[0] + "_anchors" + file_name[1]
    # arcpy.CreateFeatureclass_management(os.path.dirname(file_anchors), os.path.basename(file_anchors),
    #                                     "POINT", "", "DISABLED", "DISABLED", Forest_Line_Feature_Class)
    # file_inter = file_name[0] + "_intersections" + file_name[1]
    # arcpy.CreateFeatureclass_management(os.path.dirname(file_inter), os.path.basename(file_inter),
    #                                     "POINT", "", "DISABLED", "DISABLED", Forest_Line_Feature_Class)

    # Flatten centerlines which is a list of list
    anchor_list = []
    leastcost_list = []
    inter_list = []
    cl_list = []

    # Dump all polylines into point array for vertex updates
    ptarray_all = {}
    for i in tool_vo.segment_all:
        pt = [i[0], i[2]]
        ptarray_all[i[1]] = pt

    for sublist in centerlines:
        if not sublist:
            continue
        if len(sublist) > 0:
            for pt in sublist[0]:
                anchor_list.append(pt)
            for line in sublist[1]:
                leastcost_list.append(line)

            inter_list.append(sublist[2])

            for line in sublist[3]["lines"]:
                index = line[1]
                lineNo = line[3]["lineNo"]
                pt_array = ptarray_all[lineNo][0]

                if not pt_array or not sublist[2]:
                    continue
                pt_inter = sublist[2]
                new_intersection = [pt_inter[0], pt_inter[1]]

                if index == 0 or index == -1:
                    # the first point of first part
                    # or the last point of the last part
                    replace_index = 0
                    if index == -1:
                        try:
                            replace_index = len(pt_array[index]) - 1
                        except Exception as e:
                            print(e)

                    try:
                        # pt_array[index].replace(replace_index, new_intersection)
                        pt_array = tool_vo.update_line_vertex(pt_array, index, new_intersection)
                    except Exception as e:
                        print(e)

                ptarray_all[lineNo][0] = pt_array

    pass

    # # write all new intersections
    # with arcpy.da.InsertCursor(file_anchors, ["SHAPE@"]) as cursor:
    #     for pt in anchor_list:
    #         if pt:
    #             cursor.insertRow([arcpy.Point(pt[0], pt[1])])
    #
    # with arcpy.da.InsertCursor(file_leastcost, ["SHAPE@"]) as cursor:
    #     for line in leastcost_list:
    #         if line:
    #             cursor.insertRow([line])
    #
    # # write all new intersections
    # with arcpy.da.InsertCursor(file_inter, ["SHAPE@"]) as cursor:
    #     for pt in inter_list:
    #         if pt:
    #             cursor.insertRow([arcpy.Point(pt[0], pt[1])])
    #
    # with arcpy.da.InsertCursor(Out_Centerline, ["SHAPE@"] + fields) as cursor:
    #     for line in ptarray_all.values():
    #         if line:
    #             try:
    #                 if line[0].count > 0:
    #                     row = [arcpy.Polyline(line[0])]
    #                     for i in fields:
    #                         row.append(line[1][i])
    #
    #                     cursor.insertRow(row)
    #             except Exception as e:
    #                 print("Write output lines: {}".format(e))


if __name__ == '__main__':
    in_args, in_verbose = check_arguments()
    start_time = time.time()
    vertex_optimization(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Elapsed time: {}'.format(time.time() - start_time))
