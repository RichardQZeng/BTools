import logging
import time

import sys
from pathlib import Path
from inspect import getsourcefile

if __name__ == "__main__":
    current_file = Path(getsourcefile(lambda: 0)).resolve()
    btool_dir = current_file.parents[2]
    sys.path.insert(0, btool_dir.as_posix())

from beratools.core.logger import Logger
from beratools.core.algo_centerline import *
from beratools.core.dijkstra_algorithm import *
from beratools.core.constants import *
from common import *

log = Logger('centerline', file_level=logging.INFO)
print = log.print


def process_single_line(line_args):
    line = line_args[0][0]
    prop = line_args[0][1]
    line_radius = line_args[1]
    in_raster = line_args[2]
    line_id = line_args[3]
    seed_line = shape(line)  # LineString
    line_radius = float(line_radius)

    cost_clip, out_meta = clip_raster(in_raster, seed_line, line_radius)

    if not HAS_COST_RASTER:
        cost_clip, _ = cost_raster(cost_clip, out_meta)

    try:
        if CL_USE_SKIMAGE_GRAPH:
            # skimage shortest path
            lc_path = find_least_cost_path_skimage(cost_clip, out_meta, seed_line)
        else:
            lc_path = find_least_cost_path(cost_clip, out_meta, seed_line)
    except Exception as e:
        print(e)
        return

    if lc_path:
        lc_path_coords = lc_path.coords
    else:
        lc_path_coords = []

    # search for centerline
    if len(lc_path_coords) < 2:
        print('No least cost path detected, use input line.')
        prop['status'] = CenterlineStatus.FAILED.value
        return seed_line, prop, seed_line, None

    # get corridor raster
    lc_path = LineString(lc_path_coords)
    cost_clip, out_meta = clip_raster(in_raster, lc_path, line_radius * 0.9)
    if not HAS_COST_RASTER:
        cost_clip, _ = cost_raster(cost_clip, out_meta)

    out_transform = out_meta['transform']
    transformer = rasterio.transform.AffineTransformer(out_transform)
    cell_size = (out_transform[0], -out_transform[4])

    x1, y1 = lc_path_coords[0]
    x2, y2 = lc_path_coords[-1]
    source = [transformer.rowcol(x1, y1)]
    destination = [transformer.rowcol(x2, y2)]
    corridor_thresh_cl = corridor_raster(cost_clip, out_meta, source, destination,
                                         cell_size, FP_CORRIDOR_THRESHOLD)

    # find contiguous corridor polygon and extract centerline
    df = gpd.GeoDataFrame(geometry=[seed_line], crs=out_meta['crs'])
    corridor_poly_gpd = find_corridor_polygon(corridor_thresh_cl, out_transform, df)
    center_line, status = find_centerline(corridor_poly_gpd.geometry.iloc[0], lc_path)
    prop['status'] = status.value

    print(" Searching centerline: line {} ".format(line_id), flush=True)
    return center_line, lc_path, prop, corridor_poly_gpd


def centerline(callback, in_line, in_raster, line_radius,
               proc_segments, out_line, processes, verbose):
    if not compare_crs(vector_crs(in_line), raster_crs(in_raster)):
        print("Line and CHM have different spatial references, please check.")
        return

    # Read input line features
    layer_crs = None
    schema = None
    input_lines = []

    with fiona.open(in_line) as open_line_file:
        layer_crs = open_line_file.crs
        schema = open_line_file.meta['schema']
        for line in open_line_file:
            if line.geometry:
                if line.geometry['type'] != 'MultiLineString':
                    input_lines.append([line.geometry, line.properties])
                else:
                    print('MultiLineString found.')
                    geoms = shape(line['geometry']).geoms
                    for item in geoms:
                        line_part = fiona.Geometry.from_dict(item)
                        if line_part:
                            input_lines.append([line_part, line.properties])
            else:
                print(f'Line {line.id} has empty geometry.')

    if proc_segments:
        # split line segments at vertices
        input_lines_temp = []
        for line in input_lines:
            line_seg = line[0]
            line_prop = line[1]
            line_segs = segments(line_seg.coordinates)
            line_feats = [(line, line_prop) for line in line_segs]
            if line_segs:
                input_lines_temp.extend(line_feats)

        input_lines = input_lines_temp

    # Process lines
    all_lines = []
    i = 0
    for line in input_lines:
        all_lines.append((line, line_radius, in_raster, i))
        i += 1

    print('{} lines to be processed.'.format(len(all_lines)))

    lc_path_geoms = []
    feat_props = []
    center_line_geoms = []
    corridor_poly_list = []
    result = execute_multiprocessing(process_single_line, all_lines, 'Centerline',
                                     processes, 1, verbose=verbose)

    for item in result:
        center_line = item[0]
        lc_path = item[1]
        prop = item[2]
        corridor_poly = item[3]

        if lc_path and prop:
            lc_path_geoms.append(lc_path)
            feat_props.append(prop)
            center_line_geoms.append(center_line)
            corridor_poly_list.append(corridor_poly)

    out_centerline_path = Path(out_line)
    schema['properties']['status'] = 'int'
    save_features_to_shapefile(out_centerline_path.as_posix(), layer_crs, center_line_geoms, feat_props, schema)

    out_least_cost_path = out_centerline_path.with_stem(out_centerline_path.stem + '_least_cost_path')
    save_features_to_shapefile(out_least_cost_path.as_posix(), layer_crs, lc_path_geoms, feat_props, schema)

    # save corridor polygons
    corridor_polys = pd.concat(corridor_poly_list)
    out_corridor_poly_path = Path(out_line)
    out_corridor_poly_path = out_corridor_poly_path.with_stem(out_corridor_poly_path.stem + '_corridor_poly')
    corridor_polys.to_file(out_corridor_poly_path.as_posix())


if __name__ == '__main__':
    in_args, in_verbose = check_arguments()
    start_time = time.time()
    centerline(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Elapsed time: {}'.format(time.time() - start_time))
