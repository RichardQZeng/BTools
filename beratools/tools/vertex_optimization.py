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
import numpy as np
import time
from pathlib import Path
from collections import OrderedDict

import fiona
import shapely.geometry as sh_geom
from shapely import STRtree

import beratools.core.constants as bt_const
import beratools.tools.common as bt_common
import beratools.core.tool_base as bt_base
from beratools.core import algo_dijkstra
import beratools.core.algo_common as algo_common


class SingleLine:
    def __init__(self, line_gdf, line_no, end_no, search_distance):
        self.line_gdf = line_gdf
        self.line = self.line_gdf.geometry[0]
        self.line_no = line_no
        self.end_no = end_no
        self.search_distance = search_distance
        self.anchor = None

        self.add_anchors_to_line()

    def is_valid(self):
        return self.line.is_valid
    
    def line_coord_list(self):
        return algo_common.line_coord_list(self.line)
    
    def get_end_vertex(self):
        return self.line_coord_list()[self.end_no]
    
    def touches_point(self, vertex):
        return algo_common.points_are_close(vertex, self.get_end_vertex())

    def get_angle(self):
        return algo_common.get_angle(self.line, self.end_no)

    def add_anchors_to_line(self):
        """
        Append new vertex to vertex group, by calculating distance to existing vertices
        An anchor point will be added together with line
        """
        # Calculate anchor point for each vertex
        point = self.get_end_vertex()
        line_string = self.line
        index = self.end_no
        pts = algo_common.line_coord_list(line_string)

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
            print("Points are close, return")
            return None

        X = pt_1.x + (pt_2.x - pt_1.x) * self.search_distance / dist_pt
        Y = pt_1.y + (pt_2.y - pt_1.y) * self.search_distance / dist_pt
        self.anchor = [X, Y]  # add anchor point

class Vertex:
    def __init__(self, line_obj):
        self.vertex = line_obj.get_end_vertex()
        self.search_distance = line_obj.search_distance

        self.cost_footprint = None
        self.pt_optimized = None
        self.centerlines = None
        self.anchors = None
        self.in_raster = None
        self.line_radius = None
        self.lines = []

        self.add_line(line_obj)

    def add_line(self, line_obj):
        self.lines.append(line_obj)

    def add_anchors_to_line(self, line, uid):
        """
        Append new vertex to vertex group, by calculating distance to existing vertices
        An anchor point will be added together with line
        """
        line[2]["UID"] = uid

        # Calculate anchor point for each vertex
        # point = Point(self.vertex["point"][0], self.vertex["point"][1])
        point = sh_geom.Point(self.get_point())
        line_string = line[0]
        index = line[1]
        pts = algo_common.line_coord_list(line_string)

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
            print("Points are close, return")
            return None

        X = pt_1.x + (pt_2.x - pt_1.x) * self.search_distance / dist_pt
        Y = pt_1.y + (pt_2.y - pt_1.y) * self.search_distance / dist_pt
        line.insert(-1, [X, Y])  # add anchor point to list (the third element)

        return line

    def generate_anchor_pairs(self):
        """
        Extend line following outward direction to length of search_distance
        Use the end point as anchor point.
            vertex: input intersection with all related lines
            return: one or two pairs of anchors according to numbers of lines intersected.
                    two pairs anchors return when 3 or 4 lines intersected
                    one pair anchors return when 1 or 2 lines intersected
        """
        lines = self.get_lines()
        point = self.get_point()
        slopes = []
        for line in self.lines:
            slopes.append(line.get_angle())

        index = 0  # the index of line which paired with first line.
        pt_start_1 = None
        pt_end_1 = None
        pt_start_2 = None
        pt_end_2 = None

        if len(slopes) == 4:
            # get sort order of angles
            index = np.argsort(slopes)

            # first anchor pair (first and third in the sorted array)
            pt_start_1 = self.lines[index[0]].anchor
            pt_end_1 = self.lines[index[2]].anchor

            pt_start_2 = self.lines[index[1]].anchor
            pt_end_2 = self.lines[index[3]].anchor
        elif len(slopes) == 3:
            # find the largest difference between angles
            angle_diff = [
                abs(slopes[0] - slopes[1]),
                abs(slopes[0] - slopes[2]),
                abs(slopes[1] - slopes[2]),
            ]
            angle_diff_norm = [2 * np.pi - i if i > np.pi else i for i in angle_diff]
            index = np.argmax(angle_diff_norm)
            pairs = [(0, 1), (0, 2), (1, 2)]
            pair = pairs[index]

            # first anchor pair
            pt_start_1 = self.lines[pair[0]].anchor
            pt_end_1 = self.lines[pair[1]].anchor

            # the rest one index
            remain = list({0, 1, 2} - set(pair))[0]  # the remaining index

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
            pt_start_1 = self.lines[0].anchor
            pt_end_1 = self.lines[1].anchor
        elif len(slopes) == 1:
            pt_start_1 = self.lines[0].anchor
            # symmetry point of pt_start_1 regarding vertex["point"]
            X = point.x - (pt_start_1[0] - point.x)
            Y = point.y - (pt_start_1[1] - point.y)
            pt_end_1 = [X, Y]

        if not pt_start_1 or not pt_end_1:
            print("Anchors not found")

        # if points are outside of cost footprint, set to None
        points = [pt_start_1, pt_end_1, pt_start_2, pt_end_2]
        for index, pt in enumerate(points):
            if pt:
                if not self.cost_footprint.contains(sh_geom.Point(pt)):
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
            if bt_const.BT_DEBUGGING:
                print("No anchors retrieved")
            return None

        centerline_1 = None
        centerline_2 = None
        intersection = None

        if bt_const.CL_USE_SKIMAGE_GRAPH:
            find_lc_path = algo_dijkstra.find_least_cost_path_skimage
        else:
            find_lc_path = algo_dijkstra.find_least_cost_path

        try:
            if len(self.anchors) == 4:
                seed_line = sh_geom.LineString(self.anchors[0:2])

                raster_clip, out_meta = bt_common.clip_raster(
                    self.in_raster, seed_line, self.line_radius
                )
                # if not bt_const.HAS_COST_RASTER:
                raster_clip, _ = bt_common.cost_raster(raster_clip, out_meta)

                centerline_1 = find_lc_path(raster_clip, out_meta, seed_line)
                seed_line = sh_geom.LineString(self.anchors[2:4])

                raster_clip, out_meta = bt_common.clip_raster(
                    self.in_raster, seed_line, self.line_radius
                )
                # if not bt_const.HAS_COST_RASTER:
                raster_clip, _ = bt_common.cost_raster(raster_clip, out_meta)

                centerline_2 = find_lc_path(raster_clip, out_meta, seed_line)

                if centerline_1 and centerline_2:
                    intersection = algo_common.intersection_of_lines(centerline_1, centerline_2)
            elif len(self.anchors) == 2:
                seed_line = sh_geom.LineString(self.anchors)

                raster_clip, out_meta = bt_common.clip_raster(
                    self.in_raster, seed_line, self.line_radius
                )
                # if not bt_const.HAS_COST_RASTER:
                raster_clip, _ = bt_common.cost_raster(raster_clip, out_meta)

                centerline_1 = find_lc_path(raster_clip, out_meta, seed_line)

                if centerline_1:
                    intersection = algo_common.closest_point_to_line(self.get_point(), centerline_1)
        except Exception as e:
            print(e)

        # Update vertices according to intersection, new center lines are returned
        if type(intersection) is sh_geom.MultiPoint:
            intersection = intersection.centroid

        self.centerlines = [centerline_1, centerline_2]
        self.pt_optimized = intersection

    def get_lines(self):
        lines = [item.line for item in self.lines]
        return lines

    def get_point(self):
        return self.vertex


class VertexGrouping:
    def __init__(
        self, callback, in_line, in_raster, search_distance, line_radius, out_line
    ):
        self.in_line = in_line
        self.in_raster = in_raster
        self.line_radius = float(line_radius)
        self.search_distance = float(search_distance)
        self.out_line = out_line
        self.segment_all = []
        self.in_schema = None  # input shapefile schema
        self.crs = None
        self.vertex_grp = []
        self.sindex = None

        self.line_list = []
        self.line_visited = None

        # calculate cost raster footprint
        self.cost_footprint = bt_common.generate_raster_footprint(
            self.in_raster, latlon=False
        )

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
            line = sh_geom.shape({"type": "LineString", "coordinates": line_coords})
            if not np.isclose(line.length, 0.0):
                return [line]
        elif len(line_coords) > 2:
            seg_list = zip(line_coords[:-1], line_coords[1:])
            line_list = [
                sh_geom.shape({"type": "LineString", "coordinates": coords})
                for coords in seg_list
            ]
            return [line for line in line_list if not np.isclose(line.length, 0.0)]

        return None

    def split_lines(self):
        with fiona.open(self.in_line) as open_line_file:
            # get input shapefile fields
            self.in_schema = open_line_file.meta["schema"]
            self.in_schema["properties"]["BT_UID"] = "int:10"  # add field

            i = 0
            self.crs = open_line_file.crs
            for line in open_line_file:
                props = OrderedDict(line["properties"])
                if not line["geometry"]:
                    continue
                if line["geometry"]["type"] != "MultiLineString":
                    props[bt_const.BT_UID] = i
                    self.segment_all.append([sh_geom.shape(line["geometry"]), props])
                    i += 1
                else:
                    print("MultiLineString found.")
                    geoms = sh_geom.shape(line["geometry"]).geoms
                    for item in geoms:
                        props[bt_const.BT_UID] = i
                        self.segment_all.append([sh_geom.shape(item), props])
                        i += 1

        # split line segments at vertices
        input_lines_temp = []
        line_no = 0
        for line in self.segment_all:
            line_segs = self.segments(list(line[0].coords))
            if line_segs:
                for seg in line_segs:
                    input_lines_temp.append(
                        {
                            "line": sh_geom.shape(seg),
                            "line_no": line_no,
                            "prop": line[1],
                            "start_visited": False,
                            "end_visited": False,
                        }
                    )
                    line_no += 1

            bt_base.print_msg("Splitting lines", line_no, len(self.segment_all))

        self.segment_all = input_lines_temp

        # create spatial index for all line segments
        self.sindex = STRtree([item["line"] for item in self.segment_all])

    def create_vertex_group(self, line_obj):
        """

        Parameters
        ----------
        point :
        line_obj :

        Returns
        -------

        """
        # all end points not added will stay with this vertex
        vertex = line_obj.get_end_vertex()
        vertex_obj = Vertex(line_obj)
        search = self.sindex.query(vertex.buffer(bt_const.CL_POLYGON_BUFFER))

        # add more vertices to the new group
        for i in search:
            line = self.line_list[i]
            if i == line_obj.line_no:
                continue

            if not self.line_visited[i][0]:
                new_line = SingleLine(line, i, 0, self.search_distance)
                if new_line.touches_point(vertex):
                    vertex_obj.add_line(new_line)
                    self.line_visited[i][0] = True

            if not self.line_visited[i][-1]:
                new_line = SingleLine(line, i, -1, self.search_distance)
                if new_line.touches_point(vertex):
                    vertex_obj.add_line(new_line)
                    self.line_visited[i][-1] = True

        vertex_obj.in_raster = self.in_raster

        vertex_obj.line_radius = self.line_radius
        vertex_obj.cost_footprint = self.cost_footprint
        self.vertex_grp.append(vertex_obj)

    def create_all_vertex_groups(self):
        try:
            self.split_lines()
            print("split_lines done.")

            self.line_list = algo_common.prepare_lines_gdf(self.in_line, layer=None, proc_segments=True)
            self.line_visited = [{0: False, -1: False} for _ in range(len(self.line_list))]

            i = 0
            for line_no in range(len(self.line_list)):
                if not self.line_visited[line_no][0]:
                    line = SingleLine(self.line_list[line_no], line_no, 0, self.search_distance)
                    pt_list = line.line_coord_list()

                    if not line.is_valid:
                        print(f"Line {line['line_no']} is invalid")
                        continue

                    self.create_vertex_group(line)
                    self.line_visited[line_no][0] = True
                    i += 1

                if not self.line_visited[line_no][-1]:
                    line = SingleLine(self.line_list[line_no], line_no, -1, self.search_distance)
                    pt_list = line.line_coord_list()

                    if not line.is_valid:
                        print(f"Line {line['line_no']} is invalid")
                        continue

                    self.create_vertex_group(line)
                    self.line_visited[line_no][-1] = True
                    i += 1

                bt_base.print_msg("Grouping vertices", i, len(self.segment_all))

            print("group_intersections done.")

        except Exception as e:
            print(e)


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

    return sh_geom.LineString(coords)


def process_single_line(vertex):
    """
    It uses memory workspace instead of shapefiles.
    The refactoring is to accelerate the processing speed.
        vertex: intersection with all lines crossed at the intersection
        return: optimized vertex
    """
    vertex.optimize()
    return vertex


def vertex_optimization(
    callback,
    in_line,
    in_raster,
    search_distance,
    line_radius,
    out_line,
    processes,
    verbose,
):
    if not bt_common.compare_crs(
        bt_common.vector_crs(in_line), bt_common.raster_crs(in_raster)
    ):
        return

    vg = VertexGrouping(
        callback, in_line, in_raster, search_distance, line_radius, out_line
    )
    vg.create_all_vertex_groups()

    vertices = bt_base.execute_multiprocessing(
        process_single_line,
        vg.vertex_grp,
        "Vertex Optimization",
        processes,
        1,
        verbose=verbose,
        mode=bt_const.ParallelMode.SEQUENTIAL
    )

    # No line generated, exit
    if len(vertices) <= 0:
        print("No lines optimized.")
        return

    # Flatten vertices which is a list of list
    anchor_list = []
    leastcost_list = []
    inter_list = []

    # Dump all lines into point array for vertex updates
    feature_all = {}
    # for i in vg.segment_all:
    #     feature = [i["line"], i["prop"]]
    #     feature_all[i["line_no"]] = feature

    for vertex in vertices:
        if not vertex:
            continue

        if vertex.anchors:
            for pt in vertex.anchors:
                anchor_list.append(sh_geom.Point(pt))

        if vertex.centerlines:
            for line in vertex.centerlines:
                if line:
                    leastcost_list.append(line)

        if vertex.pt_optimized:
            inter_list.append(vertex.pt_optimized)

        for line in vertex.get_lines():
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
    file_line = line_path.as_posix()
    file_aux = line_path.with_stem(file_name + "_aux").with_suffix(".gpkg").as_posix()

    fields = []
    properties = []
    all_lines = [value[0] for key, value in feature_all.items()]
    all_props = [value[1] for key, value in feature_all.items()]
    bt_common.save_features_to_file(
        file_line, vg.crs, all_lines, all_props, vg.in_schema
    )

    bt_common.save_features_to_file(
        file_aux,
        vg.crs,
        leastcost_list,
        properties,
        fields,
        driver="GPKG",
        layer="leastcost",
    )
    bt_common.save_features_to_file(
        file_aux,
        vg.crs,
        anchor_list,
        properties,
        fields,
        driver="GPKG",
        layer="anchors",
    )
    bt_common.save_features_to_file(
        file_aux,
        vg.crs,
        inter_list,
        properties,
        fields,
        driver="GPKG",
        layer="intersections",
    )


if __name__ == "__main__":
    in_args, in_verbose = bt_common.check_arguments()
    start_time = time.time()
    vertex_optimization(
        print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose
    )
    print("Elapsed time: {}".format(time.time() - start_time))
