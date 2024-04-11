from collections import OrderedDict
from multiprocessing.pool import Pool

import time
import uuid
from pathlib import Path
import numpy as np
import pandas as pd

import rasterio
import rasterio.mask
import fiona
from fiona import Geometry
from osgeo import gdal, ogr
from shapely.geometry import shape, mapping, LineString, MultiLineString, Point
from skimage.graph import MCP_Geometric

from dijkstra_algorithm import *
from common import *

# import line_profiler
# profile = line_profiler.LineProfiler()

# from memory_profiler import profile


class OperationCancelledException(Exception):
    pass


def centerline(callback, in_line, in_cost, line_radius,
               proc_segments, out_line, processes, verbose):
    if not compare_crs(vector_crs(in_line), raster_crs(in_cost)):
        print("Line and CHM spatial references are not same, please check.")
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
                        line_part = Geometry.from_dict(item)
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
    step = 0
    total_steps = len(all_lines)

    feat_geoms = []
    feat_props = []
    center_line_geoms = []
    corridor_poly_list = []
    if PARALLEL_MODE == MODE_MULTIPROCESSING:
        feat_geoms, feat_props, center_line_geoms, corridor_poly_list = execute_multiprocessing(all_lines,
                                                                                                processes, verbose)
    elif PARALLEL_MODE == MODE_SEQUENTIAL:
        for line in all_lines:
            geom, prop, center_line, corridor_poly_gpd = process_single_line(line)
            if geom and prop:
                feat_geoms.append(geom)
                feat_props.append(prop)
                center_line_geoms.append(center_line)
                corridor_poly_list.append(corridor_poly_gpd)
            step += 1
            if verbose:
                print(' "PROGRESS_LABEL Ceterline {} of {}" '.format(step, total_steps), flush=True)
                print(' %{} '.format(step / total_steps * 100), flush=True)

    i = 0

    driver = 'ESRI Shapefile'
    print('Writing lines to shapefile')

    out_least_cost_path = Path(out_line)
    out_least_cost_path = out_least_cost_path.with_stem(out_least_cost_path.stem+'_least_cost_path')
    if not BT_DEBUGGING:
        save_features_to_shapefile(out_least_cost_path.as_posix(), layer_crs, feat_geoms, schema, feat_props)

    save_features_to_shapefile(out_line, layer_crs, center_line_geoms, schema, feat_props)

    # save corridor polygons
    corridor_polys = pd.concat(corridor_poly_list)
    out_corridor_poly_path = Path(out_line)
    out_corridor_poly_path = out_corridor_poly_path.with_stem(out_corridor_poly_path.stem + '_corridor_poly')
    corridor_polys.to_file(out_corridor_poly_path.as_posix())


# @profile
def process_single_line(line_args, find_nearest=True, output_linear_reference=False):
    line = line_args[0][0]
    prop = line_args[0][1]
    line_radius = line_args[1]
    in_cost_raster = line_args[2]

    line_buffer = shape(line).buffer(float(line_radius))

    # buffer clip
    with rasterio.open(in_cost_raster) as raster_file:
        out_image, out_transform = rasterio.mask.mask(raster_file, [line_buffer],
                                                      crop=True, nodata=BT_NODATA, filled=True)

    line_id = line_args[3]

    ras_nodata = raster_file.meta['nodata']
    if not ras_nodata:
        ras_nodata = BT_NODATA

    least_cost_path = find_least_cost_path(ras_nodata, out_image, out_transform, line_id, shape(line))
    center_line = None

    # search for centerline
    if len(least_cost_path[0]) < 2:
        print('Lest cost path {} is too short.'.format(least_cost_path[0]))
        return None

    least_cost_line = LineString(least_cost_path[0])
    cost_clip, out_meta = clip_raster(in_cost_raster, least_cost_line, float(line_radius))
    out_transform = out_meta['transform']

    if type(least_cost_path) is MultiLineString:
        print('MultiLineString found.')
        return
    else:
        x1, y1 = least_cost_path[0][0]
        x2, y2 = least_cost_path[0][-1]

    # Work out the corridor from both end of the centerline
    try:
        # change all nan to BT_NODATA_COST for workaround
        cost_clip = np.squeeze(cost_clip, axis=0)
        remove_nan_from_array(cost_clip)

        # generate the cost raster to source point
        transformer = rasterio.transform.AffineTransformer(out_transform)
        source = [transformer.rowcol(x1, y1)]

        # generate the cost raster to source point
        mcp_source = MCP_Geometric(cost_clip)
        source_cost_acc = mcp_source.find_costs(source)[0]
        del mcp_source

        # generate the cost raster to destination point
        destination = [transformer.rowcol(x2, y2)]

        # # # generate the cost raster to destination point
        mcp_dest = MCP_Geometric(cost_clip)
        dest_cost_acc = mcp_dest.find_costs(destination)[0]

        # Generate corridor
        corridor = source_cost_acc + dest_cost_acc
        corridor = np.ma.masked_invalid(corridor)

        # Calculate minimum value of corridor raster
        if not np.ma.min(corridor) is None:
            corr_min = float(np.ma.min(corridor))
        else:
            corr_min = 0.5

        # normalize corridor raster by deducting corr_min
        corridor_norm = corridor - corr_min

        cell_size_x = out_transform[0]
        # cell_size_y = -out_transform[4]
        corridor_th_value = FP_CORRIDOR_THRESHOLD/cell_size_x
        corridor_thresh_cl = np.ma.where(corridor_norm >= corridor_th_value, 1.0, 0.0)

        # export intermediate raster for debugging
        if BT_DEBUGGING:
            suffix = str(uuid.uuid4())[:8]
            path_temp = Path(r'C:\BERATools\Surmont_New_AOI\test_selected_lines\temp_files')
            if path_temp.exists():
                path_cost = path_temp.joinpath(suffix + '_cost.tif')
                path_corridor = path_temp.joinpath(suffix + '_corridor.tif')
                path_corridor_norm = path_temp.joinpath(suffix + '_corridor_norm.tif')
                path_corridor_cl = path_temp.joinpath(suffix + '_corridor_cl_poly.tif')
                out_cost = np.ma.masked_equal(cost_clip, np.inf)
                save_raster_to_file(out_cost, out_meta, path_cost)
                save_raster_to_file(corridor, out_meta, path_corridor)
                save_raster_to_file(corridor_norm, out_meta, path_corridor_norm)
                save_raster_to_file(corridor_thresh_cl, out_meta, path_corridor_cl)
            else:
                print('Debugging: raster folder not exists.')

        # find contiguous corridor polygon and extract centerline
        df = gpd.GeoDataFrame(geometry=[shape(line)], crs=out_meta['crs'])
        corridor_poly_gpd = find_corridor_polygon(corridor_thresh_cl, out_transform, df)
        center_line = find_centerline(corridor_poly_gpd.geometry.iloc[0], LineString(least_cost_path[0]))

        # Check if centerline is valid. If not, regenerate by splitting polygon into two halves.
        if not centerline_is_valid(center_line, LineString(least_cost_path[0])):
            center_line = regenerate_centerline(corridor_poly_gpd.geometry.iloc[0], LineString(least_cost_path[0]))
    except Exception as e:
        print(e)

    return least_cost_path[0], prop, center_line, corridor_poly_gpd


def execute_multiprocessing(line_args, processes, verbose):
    try:
        total_steps = len(line_args)
        feat_geoms = []
        feat_props = []
        center_line_geoms = []
        corridor_poly_list = []
        with Pool(processes) as pool:
            step = 0
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(process_single_line, line_args):
                if BT_DEBUGGING:
                    print('Got result: {}'.format(result), flush=True)

                if not result:
                    print('No line detected.')
                    continue

                geom = result[0]
                prop = result[1]
                center_line = result[2]
                corridor_poly = result[3]
                if geom and prop:
                    if len(geom) <= 1:
                        print("No least cost path detected")
                        continue
                    try:
                        feat_geoms.append(LineString(geom))
                        feat_props.append(prop)
                        center_line_geoms.append(center_line)
                        corridor_poly_list.append(corridor_poly)
                    except Exception as e:
                        print(e)

                step += 1
                if verbose:
                    print(' "PROGRESS_LABEL Ceterline {} of {}" '.format(step, total_steps), flush=True)
                    print(' %{} '.format(step/total_steps*100), flush=True)

        return feat_geoms, feat_props, center_line_geoms, corridor_poly_list
    except OperationCancelledException:
        print("Operation cancelled")
        return None


if __name__ == '__main__':
    in_args, in_verbose = check_arguments()
    start_time = time.time()
    centerline(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Elapsed time: {}'.format(time.time() - start_time))
