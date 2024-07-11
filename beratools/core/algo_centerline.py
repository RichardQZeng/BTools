import numpy as np
from rasterio import features
import shapely
from shapely.geometry import shape
from shapely.ops import unary_union, substring, linemerge, nearest_points, split
from shapely.geometry import Point, MultiPoint, Polygon, MultiPolygon, LineString, MultiLineString
from beratools.third_party.label_centerlines import get_centerline

from beratools.core.tool_base import *
from beratools.core.constants import *
from beratools.tools.common import generate_perpendicular_line_precise


def centerline_is_valid(centerline, input_line):
    """
    Check if centerline is valid
    Parameters
    ----------
    centerline :
    input_line : shapely LineString
        This can be input seed line or least cost path. Only two end points are used.

    Returns
    -------

    """
    if not centerline:
        return False

    # centerline length less the half of least cost path
    if (centerline.length < input_line.length / 2 or
            centerline.distance(Point(input_line.coords[0])) > BT_EPSILON or
            centerline.distance(Point(input_line.coords[-1])) > BT_EPSILON):
        return False

    return True


def snap_end_to_end(in_line, line_reference):
    if type(in_line) is MultiLineString:
        in_line = linemerge(in_line)
        if type(in_line) is MultiLineString:
            print(f'algo_centerline: MultiLineString found {in_line.centroid}, pass.')
            return None

    pts = list(in_line.coords)
    if len(pts) < 2:
        print('snap_end_to_end: input line invalid.')
        return in_line

    line_start = Point(pts[0])
    line_end = Point(pts[-1])
    ref_ends = MultiPoint([line_reference.coords[0], line_reference.coords[-1]])

    _, snap_start = nearest_points(line_start, ref_ends)
    _, snap_end = nearest_points(line_end, ref_ends)

    if in_line.has_z:
        snap_start = shapely.force_3d(snap_start)
        snap_end = shapely.force_3d(snap_end)
    else:
        snap_start = shapely.force_2d(snap_start)
        snap_end = shapely.force_2d(snap_end)

    pts[0] = snap_start.coords[0]
    pts[-1] = snap_end.coords[0]

    return LineString(pts)


def find_centerline(poly, input_line):
    """
    Parameters
    ----------
    poly : Polygon
    input_line : LineString
        Least cost path or seed line

    Returns
    -------

    """
    default_return = input_line, CenterlineStatus.FAILED
    if not poly:
        print('find_centerline: No polygon found')
        return default_return

    poly = shapely.segmentize(poly, max_segment_length=CL_SEGMENTIZE_LENGTH)

    poly = poly.buffer(CL_POLYGON_BUFFER)  # buffer polygon to reduce MultiPolygons
    if type(poly) is MultiPolygon:
        print('MultiPolygon encountered, skip.')
        return default_return

    exterior_pts = list(poly.exterior.coords)

    if CL_DELETE_HOLES:
        poly = Polygon(exterior_pts)
    if CL_SIMPLIFY_POLYGON:
        poly = poly.simplify(CL_SIMPLIFY_LENGTH)

    try:
        centerline = get_centerline(poly, segmentize_maxlen=1, max_points=3000,
                                    simplification=0.05, smooth_sigma=CL_SMOOTH_SIGMA, max_paths=1)
    except Exception as e:
        print('Exception in get_centerline.')
        return default_return

    if type(centerline) is MultiLineString:
        if len(centerline.geoms) > 1:
            print(" Multiple centerline segments detected, no further processing.")
            return centerline, CenterlineStatus.SUCCESS  # TODO: inspect
        elif len(centerline.geoms) == 1:
            centerline = centerline.geoms[0]
        else:
            return default_return

    cl_coords = list(centerline.coords)

    # trim centerline at two ends
    head_buffer = Point(cl_coords[0]).buffer(CL_BUFFER_CLIP)
    centerline = centerline.difference(head_buffer)

    end_buffer = Point(cl_coords[-1]).buffer(CL_BUFFER_CLIP)
    centerline = centerline.difference(end_buffer)

    if not centerline:
        print('No centerline detected, use input line instead.')
        return default_return
    try:
        if centerline.is_empty:
            print('Empty centerline detected, use input line instead.')
            return default_return
    except Exception as e:
        print(e)

    centerline = snap_end_to_end(centerline, input_line)

    # Check if centerline is valid. If not, regenerate by splitting polygon into two halves.
    if not centerline_is_valid(centerline, input_line):
        try:
            print(f'Regenerating line ...')
            centerline = regenerate_centerline(poly, input_line)
            return centerline, CenterlineStatus.REGENERATE_SUCCESS
        except Exception as e:
            print('find_centerline:  Exception occurred. \n {}'.format(e))
            return input_line, CenterlineStatus.REGENERATE_FAILED

    return centerline, CenterlineStatus.SUCCESS


# def find_route(array, start, end, fully_connected, geometric):
#     route_list, cost_list = route_through_array(array, start, end, fully_connected, geometric)
#     return route_list, cost_list


def find_corridor_polygon(corridor_thresh, in_transform, line_gpd):
    # Threshold corridor raster used for generating centerline
    corridor_thresh_cl = np.ma.where(corridor_thresh == 0.0, 1, 0).data
    corridor_mask = np.where(1 == corridor_thresh_cl, True, False)
    poly_generator = features.shapes(corridor_thresh_cl, mask=corridor_mask, transform=in_transform)
    corridor_polygon = []

    try:
        for poly, value in poly_generator:
            if shape(poly).area > 1:
                corridor_polygon.append(shape(poly))
    except Exception as e:
        print(e)

    if corridor_polygon:
        corridor_polygon = (unary_union(corridor_polygon))
        if type(corridor_polygon) is MultiPolygon:
            poly_list = shapely.get_parts(corridor_polygon)
            merge_poly = poly_list[0]
            for i in range(1, len(poly_list)):
                if shapely.intersects(merge_poly, poly_list[i]):
                    merge_poly = shapely.union(merge_poly, poly_list[i])
                else:
                    buffer_dist = poly_list[i].distance(merge_poly) + 0.1
                    buffer_poly = poly_list[i].buffer(buffer_dist)
                    merge_poly = shapely.union(merge_poly, buffer_poly)
            corridor_polygon = merge_poly
    else:
        corridor_polygon = None

    # create GeoDataFrame for centerline
    corridor_poly_gpd = gpd.GeoDataFrame.copy(line_gpd)
    corridor_poly_gpd.geometry = [corridor_polygon]

    return corridor_poly_gpd


def find_single_centerline(row_and_path):
    """

    Parameters
    ----------
    row_and_path:
        list of row (polygon and props) and least cost path
        first is geopandas row, second is input line, (least cost path)

    Returns
    -------

    """
    row = row_and_path[0]
    lc_path = row_and_path[1]

    poly = row.geometry.iloc[0]
    centerline, status = find_centerline(poly, lc_path)
    row['centerline'] = centerline

    return row


def find_centerlines(poly_gpd, line_seg, processes):
    centerline = None
    centerline_gpd = []
    rows_and_paths = []

    try:
        for i in poly_gpd.index:
            row = poly_gpd.loc[[i]]
            poly = row.geometry.iloc[0]
            if 'OLnSEG' in line_seg.columns:
                line_id, Seg_id = row['OLnFID'].iloc[0], row['OLnSEG'].iloc[0]
                lc_path = line_seg.loc[(line_seg.OLnFID == line_id) & (line_seg.OLnSEG == Seg_id)]['geometry'].iloc[0]
            else:
                line_id = row['OLnFID'].iloc[0]
                lc_path = line_seg.loc[(line_seg.OLnFID == line_id)]['geometry'].iloc[0]

            rows_and_paths.append((row, lc_path))
    except Exception as e:
        print(e)

    total_steps = len(rows_and_paths)
    step = 0

    # if PARALLEL_MODE == ParallelMode.MULTIPROCESSING:
    #     with Pool(processes=processes) as pool:
    #         # execute tasks in order, process results out of order
    #         for result in pool.imap_unordered(find_single_centerline, rows_and_paths):
    #             centerline_gpd.append(result)
    #             step += 1
    #             print(' "PROGRESS_LABEL Centerline {} of {}" '.format(step, total_steps), flush=True)
    #             print(' %{} '.format(step / total_steps * 100))
    #             print('Centerline No. {} done'.format(step))
    # elif PARALLEL_MODE == ParallelMode.SEQUENTIAL:
    #     for item in rows_and_paths:
    #         row_with_centerline = find_single_centerline(item)
    #         centerline_gpd.append(row_with_centerline)
    #         step += 1
    #         print(' "PROGRESS_LABEL Centerline {} of {}" '.format(step, total_steps), flush=True)
    #         print(' %{} '.format(step / total_steps * 100))
    #         print('Centerline No. {} done'.format(step))
    centerline_gpd = execute_multiprocessing(find_single_centerline, rows_and_paths,
                                             'find_centerlines', processes, 1)
    return pd.concat(centerline_gpd)


def regenerate_centerline(poly, input_line):
    """
    Regenerates centerline when initial
    ----------
    poly : line is not valid
    Parameters
    input_line : shapely LineString
        This can be input seed line or least cost path. Only two end points will be used

    Returns
    -------

    """
    line_1 = substring(input_line, start_dist=0.0, end_dist=input_line.length / 2)
    line_2 = substring(input_line, start_dist=input_line.length / 2, end_dist=input_line.length)

    pts = shapely.force_2d([Point(list(input_line.coords)[0]),
                            Point(list(line_1.coords)[-1]),
                            Point(list(input_line.coords)[-1])])
    perp = generate_perpendicular_line_precise(pts)

    # MultiPolygon is rare, but need to be dealt with
    # remove polygon of area less than CL_CLEANUP_POLYGON_BY_AREA
    poly = poly.buffer(CL_POLYGON_BUFFER)
    if type(poly) is MultiPolygon:
        poly_geoms = list(poly.geoms)
        poly_valid = [True] * len(poly_geoms)
        for i, item in enumerate(poly_geoms):
            if item.area < CL_CLEANUP_POLYGON_BY_AREA:
                poly_valid[i] = False

        poly_geoms = list(compress(poly_geoms, poly_valid))
        if len(poly_geoms) != 1:  # still multi polygon
            print('regenerate_centerline: Multi or none polygon found, pass.')

        poly = Polygon(poly_geoms[0])

    poly_exterior = Polygon(poly.buffer(CL_POLYGON_BUFFER).exterior)
    poly_split = split(poly_exterior, perp)

    if len(poly_split.geoms) < 2:
        print('regenerate_centerline: polygon split failed, pass.')
        return None

    poly_1 = poly_split.geoms[0]
    poly_2 = poly_split.geoms[1]

    # find polygon and line pairs
    pair_line_1 = line_1
    pair_line_2 = line_2
    if not poly_1.intersects(line_1):
        pair_line_1 = line_2
        pair_line_2 = line_1
    elif poly_1.intersection(line_1).length < line_1.length / 3:
        pair_line_1 = line_2
        pair_line_2 = line_1

    center_line_1 = find_centerline(poly_1, pair_line_1)
    center_line_2 = find_centerline(poly_2, pair_line_2)

    center_line_1 = center_line_1[0]
    center_line_2 = center_line_2[0]

    if not center_line_1 or not center_line_2:
        print('Regenerate line: centerline is None')
        return None

    try:
        if center_line_1.is_empty or center_line_2.is_empty:
            print('Regenerate line: centerline is empty')
            return None
    except Exception as e:
        print(e)

    print(f'Centerline is regenerated.')
    return linemerge(MultiLineString([center_line_1, center_line_2]))

