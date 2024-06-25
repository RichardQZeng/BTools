from collections import OrderedDict
from multiprocessing.pool import Pool

import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd

import rasterio
import fiona
from shapely.geometry import shape, LineString, MultiLineString

from dijkstra_algorithm import *
from common import *


class OperationCancelledException(Exception):
    pass


def LCP_centerline(callback, in_line, in_chm, line_radius,
               proc_segments, out_line, processes, verbose):
    if not compare_crs(vector_crs(in_line), raster_crs(in_chm)):
        print("Line and CHM have different spatial references, please check.")
        return

    # Read input line features
    layer_crs = None
    schema = None
    input_lines = []

    df,found=chk_df_multipart(gpd.GeoDataFrame.from_file(in_line),'MultiLineString')
    if found:
        df.to_file(in_line)
    else:
        del df, found

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
        all_lines.append((line, line_radius, in_chm, i))
        i += 1

    print('{} lines to be processed.'.format(len(all_lines)))

    feat_geoms = []
    feat_props = []
    center_line_geoms = []
    corridor_poly_list = []
    result = execute_multiprocessing(process_single_line, 'Centerline',
                                     all_lines, processes, 1, verbose)

    for item in result:
        geom = item[0]
        prop = item[1]
        center_line = item[2]
        corridor_poly = item[3]

        if geom and prop:
            feat_geoms.append(geom)
            feat_props.append(prop)
            center_line_geoms.append(center_line)
            corridor_poly_list.append(corridor_poly)

    out_least_cost_path = Path(out_line)
    out_least_cost_path = out_least_cost_path.with_stem(out_least_cost_path.stem+'_least_cost_path')
    schema['properties']['status'] = 'int'

    save_features_to_shapefile(out_least_cost_path.as_posix(), layer_crs, feat_geoms, schema, feat_props)

    # save_features_to_shapefile(out_line, layer_crs, center_line_geoms, schema, feat_props)

    # save corridor polygons
    # corridor_polys = pd.concat(corridor_poly_list)
    # out_corridor_poly_path = Path(out_line)
    # out_corridor_poly_path = out_corridor_poly_path.with_stem(out_corridor_poly_path.stem + '_corridor_poly')
    # corridor_polys.to_file(out_corridor_poly_path.as_posix())


def process_single_line(line_args):
    line = line_args[0][0]
    prop = line_args[0][1]
    line_radius = line_args[1]
    in_chm_raster = line_args[2]
    line_id = line_args[3]
    seed_line = shape(line)  # LineString
    line_radius = float(line_radius)



    chm_clip, out_meta = clip_raster(in_chm_raster, seed_line, line_radius)
    in_chm = np.squeeze(chm_clip, axis=0)
    cell_x, cell_y=out_meta['transform'][0],-out_meta['transform'][4]
    kernel = convolution.circle_kernel(cell_x, cell_y, 2.5)
    dyn_canopy_ndarray = dyn_np_cc_map(in_chm, FP_CORRIDOR_THRESHOLD, BT_NODATA)
    cc_std, cc_mean = dyn_fs_raster_stdmean(dyn_canopy_ndarray, kernel,BT_NODATA)
    cc_smooth = dyn_smooth_cost(dyn_canopy_ndarray, 2.5, [cell_x, cell_y])
    avoidance = max(min(float(0.4), 1), 0)
    cost_clip = dyn_np_cost_raster(dyn_canopy_ndarray, cc_mean, cc_std,
                                          cc_smooth, 0.4, 1.5)


    # if CL_USE_SKIMAGE_GRAPH:
        # skimage shortest path (Cost Array elements with infinite or negative costs will simply be ignored.)
    negative_cost_clip=np.where(np.isnan(cost_clip),-9999,cost_clip)
    # lc_path = find_least_cost_path_skimage(negative_cost_clip, out_meta, seed_line)
    lc_path = LCP_skimage_mcp_connect(negative_cost_clip, out_meta, seed_line)
    # else:
    #     lc_path = find_least_cost_path(cost_clip, out_meta, seed_line)

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
    # lc_path = LineString(lc_path_coords)
    # if seed_line.has_z:
    #     lc_path_coords = np.delete(np.array(seed_line.coords),np.s_[2],1)
    #     lc_path_coords=[tuple(x) for x in lc_path_coords.tolist()]
    # else:
    #     lc_path_coords=seed_line.coords
    # lc_path = seed_line
    # cost_clip, out_meta = clip_raster(in_cost_raster, lc_path, line_radius*0.9)
    out_transform = out_meta['transform']
    transformer = rasterio.transform.AffineTransformer(out_transform)
    cell_size = (out_transform[0], -out_transform[4])

    x1, y1 = lc_path_coords[0]
    x2, y2 = lc_path_coords[-1]
    source = [transformer.rowcol(x1, y1)]
    destination = [transformer.rowcol(x2, y2)]
    # skimage graph.MCP (Cost Array elements with infinite or negative costs will simply be ignored.)
    corridor_thresh_cl = corridor_raster(negative_cost_clip, out_meta, source, destination,
                                         cell_size, FP_CORRIDOR_THRESHOLD)

    # find contiguous corridor polygon and extract centerline
    df = gpd.GeoDataFrame(geometry=[seed_line], crs=out_meta['crs'])
    corridor_poly_gpd = find_corridor_polygon(corridor_thresh_cl, out_transform, df)
    center_line, status = find_centerline(corridor_poly_gpd.geometry.iloc[0], lc_path)
    prop['status'] = status.value

    print(" Searching centerline: line {} ".format(line_id), flush=True)
    return lc_path, prop, center_line, corridor_poly_gpd


if __name__ == '__main__':
    in_args, in_verbose = check_arguments()
    start_time = time.time()
    LCP_centerline(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Elapsed time: {}'.format(time.time() - start_time))
