import time
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd

import shapely
import rasterio

import beratools.tools.common as bt_common
import beratools.core.constants as bt_const
import beratools.core.tool_base as bt_base
import beratools.core.algo_centerline as algo_cl
import beratools.core.algo_common as algo_common

class FootprintAbsolute:
    def __init__(
        self,
        line_seg,
        in_chm,
        max_ln_width,
        exp_shk_cell,
    ):
        self.line_seg = line_seg
        self.in_chm = in_chm
        self.corridor_th_value = 3.0
        self.max_ln_width = max_ln_width
        self.exp_shk_cell = exp_shk_cell

        self.footprint = footprint = None
        self.corridor_poly_gpd = None
        self.centerline = None

    def compute(self):
        """
            this function takes single line to work the line footprint
            (regardless it process the whole line or individual segment)
            """
        in_chm = self.in_chm
        corridor_th_value = self.corridor_th_value
        line_gpd = self.line_seg
        max_ln_width = self.max_ln_width
        exp_shk_cell = self.exp_shk_cell

        try:
            corridor_th_value = float(corridor_th_value)
            if corridor_th_value < 0.0:
                corridor_th_value = 3.0
        except ValueError as e:
            print(f"process_single_line_segment: {e}")
            corridor_th_value = 3.0

        segment_list = []
        feat = self.line_seg.geometry[0]
        for coord in feat.coords:
            segment_list.append(coord)

        # Find origin and destination coordinates
        x1, y1 = segment_list[0][0], segment_list[0][1]
        x2, y2 = segment_list[-1][0], segment_list[-1][1]

        # Buffer around line and clip cost raster and canopy raster
        # TODO: deal with NODATA
        clip_cost, out_meta = bt_common.clip_raster(in_chm, feat, max_ln_width)
        out_transform = out_meta["transform"]
        cell_size_x = out_transform[0]
        cell_size_y = -out_transform[4]

        clip_cost, clip_canopy = bt_common.cost_raster(clip_cost, out_meta)

        # Work out the corridor from both end of the centerline
        if len(clip_canopy.shape) > 2:
            clip_canopy = np.squeeze(clip_canopy, axis=0)

        transformer = rasterio.transform.AffineTransformer(out_transform)
        source = [transformer.rowcol(x1, y1)]
        destination = [transformer.rowcol(x2, y2)]

        corridor_thresh = bt_common.corridor_raster(
            clip_cost,
            out_meta,
            source,
            destination,
            (cell_size_x, cell_size_y),
            corridor_th_value,
        )

        clean_raster = algo_common.morph_raster(
            corridor_thresh, clip_canopy, exp_shk_cell, cell_size_x
        )

        # create mask for non-polygon area
        msk = np.where(clean_raster == 1, True, False)
        if clean_raster.dtype == np.int64:
            clean_raster = clean_raster.astype(np.int32)

        # Process: ndarray to shapely Polygon
        out_polygon = rasterio.features.shapes(
            clean_raster, mask=msk, transform=out_transform
        )

        # create a shapely multipolygon
        multi_polygon = []
        for shp, value in out_polygon:
            multi_polygon.append(shapely.geometry.shape(shp))
        poly = shapely.geometry.MultiPolygon(multi_polygon)

        # create a pandas dataframe for the footprint
        footprint = gpd.GeoDataFrame(geometry=[poly], crs=self.line_seg.crs)

        # find contiguous corridor polygon for centerline
        corridor_poly_gpd = algo_cl.find_corridor_polygon(
            corridor_thresh, out_transform, line_gpd
        )
        centerline, status = algo_cl.find_centerline(
            corridor_poly_gpd.geometry.iloc[0], feat
        )

        self.footprint = footprint
        self.corridor_poly_gpd = corridor_poly_gpd
        self.centerline = centerline

def process_single_line_class(line_footprint):
    line_footprint.compute()
    return line_footprint


def prepare_lines_gdf(
    in_line, in_chm, corridor_th_value, max_ln_width, exp_shk_cell, in_layer=None
):
    line_seg = gpd.GeoDataFrame.from_file(in_line, layer=in_layer)

    if not bt_common.compare_crs(
        bt_common.vector_crs(in_line), bt_common.raster_crs(in_chm)
    ):
        print("Line and cost have different spatial references, please check.")
        return

    if "OLnFID" not in line_seg.columns.array:
        line_seg["OLnFID"] = line_seg.index

    if "CorridorTh" not in line_seg.columns.array:
        line_seg["CorridorTh"] = corridor_th_value
    else:
        corridor_th_value = float(9999999)

    if "OLnSEG" not in line_seg.columns.array:
        line_seg["OLnSEG"] = 0

    # get the list of original columns names
    field_list_col = field_name_list(line_seg)
    keep_field_name = []
    for col_name in line_seg.columns:
        if col_name != "geometry":
            keep_field_name.append(col_name)

    list_of_segment = []

    i = 0
    if len(line_seg) > 0:
        for row in range(0, len(line_seg)):
            # creates a geometry object
            line_gpd = line_seg.loc[[row]]
            feat = line_gpd.geometry.iloc[0]
            if feat:
                feature_attributes = {
                    "seg_length": feat.length,
                    "geometry": feat,
                    "Proj_crs": line_seg.crs,
                    "line_gpd": line_gpd,
                }

                for col_name in keep_field_name:
                    feature_attributes[col_name] = line_seg.loc[row, col_name]
                list_of_segment.append(feature_attributes)
                i += 1

        # print(f"There are {ori_total_feat} lines to be processed.")
    else:
        print("Input line feature is corrupted, exit!")
        exit()

    # Add tools arguments into GeoDataFrame record
    for record in list_of_segment:
        record["corridor_th_value"] = record["CorridorTh"]
        record["max_ln_width"] = max_ln_width
        record["exp_shk_cell"] = exp_shk_cell
        record["org_col"] = field_list_col

    # TODO: data type changed - return list of GeoDataFrame represents each line or segment
    # returns list of list of line attributes, arguments and line gpd
    if bt_const.GROUPING_SEGMENT:
        # group line segments by line id
        def key_func(x):
            return x["OLnFID"]

        lines = []

        for key, group in itertools.groupby(list_of_segment, key_func):
            lines.append(list(group))

        return lines
    else:
        return list_of_segment


def generate_line_class_list(
    in_line,
    in_chm,
    max_ln_width,
    exp_shk_cell,
    in_layer=None,
):
    line_classes = []
    line_list = algo_common.prepare_lines_gdf(in_line, in_layer, proc_segments=False)

    for line in line_list:
        line_classes.append(FootprintAbsolute(line, in_chm, max_ln_width, exp_shk_cell))

    return line_classes


def line_footprint_abs(
    in_line,
    in_chm,
    corridor_th_value,
    max_ln_width,
    exp_shk_cell,
    out_footprint,
    out_centerline,
    processes,
    verbose,
    in_layer=None,
    out_layer_fp=None,
    out_layer_cl=None,
):
    # corridor_th_field = "CorridorTh"
    # line_seg = gpd.GeoDataFrame.from_file(in_line, layer=in_layer)
    max_ln_width = float(max_ln_width)
    exp_shk_cell = int(exp_shk_cell)
    #
    # if not bt_common.compare_crs(
    #     bt_common.vector_crs(in_line), bt_common.raster_crs(in_chm)
    # ):
    #     print("Line and cost have different spatial references, please check.")
    #     return
    #
    # if "OLnFID" not in line_seg.columns.array:
    #     line_seg["OLnFID"] = line_seg.index
    #
    # if "CorridorTh" not in line_seg.columns.array:
    #     line_seg["CorridorTh"] = corridor_th_value
    # else:
    #     corridor_th_value = float(9999999)
    #
    # if "OLnSEG" not in line_seg.columns.array:
    #     line_seg["OLnSEG"] = 0
    #
    # line_args = line_prepare(
    #     line_seg,
    #     in_chm,
    #     corridor_th_field,
    #     corridor_th_value,
    #     max_ln_width,
    #     exp_shk_cell,
    #     out_footprint,
    #     out_centerline,
    # )
    #
    # # pass single line one at a time for footprint
    feat_list = []
    footprint_list = []
    poly_list = []
    centerline_list = []
    #
    # process_single_line = process_single_line_segment
    # if bt_const.GROUPING_SEGMENT:
    #     process_single_line = process_single_line_whole

    # feat_list = bt_base.execute_multiprocessing(
    #     process_single_line, line_args, "Line footprint", processes, 1, verbose=verbose
    # )

    line_class_list = generate_line_class_list(in_line, in_chm, max_ln_width, exp_shk_cell, in_layer)

    feat_list = bt_base.execute_multiprocessing(
        process_single_line_class,
        line_class_list,
        "Line footprint",
        processes,
        1,
        verbose=verbose,
        mode=bt_const.ParallelMode.SEQUENTIAL,
    )

    if feat_list:
        for i in feat_list:
            footprint_list.append(i.footprint)
            poly_list.append(i.corridor_poly_gpd)

    results = gpd.GeoDataFrame(pd.concat(footprint_list))
    results = results.reset_index(drop=True)
    results.to_file(out_footprint, layer=out_layer_fp)


def field_name_list(fc):
    # return a list of column name from shapefile
    if isinstance(fc, gpd.GeoDataFrame):
        field_list = fc.columns.array
    else:
        field_list = gpd.read_file(fc).columns.array
    return field_list


def has_field(fc, fi):
    # Check column name
    field_list = field_name_list(fc)
    if fi in field_list:
        print("column: {fi} is found")
        return True
    elif fi == "CorridorTh":
        shapefile = gpd.GeoDataFrame.from_file(fc)
        for row in range(0, len(shapefile)):
            shapefile.loc[row, fi] = 3.0

        shapefile.to_file(fc)
        print(
            "Warning: There is no field named {} in the input data".format("CorridorTh")
        )
        print("Field: 'CorridorTh' is added and default threshold (i.e.3) is adopted")
        return True
    else:
        print(f"Warning: There is no field named {fi} in the input data")
        return False


def process_single_line_whole(line):
    footprints = []
    line_polys = []
    centerline_list = []
    for line_seg in line:
        footprint = process_single_line_segment(line_seg)
        if footprint:
            footprints.append(footprint[0])
            line_polys.append(footprint[1])
            centerline_list.append(footprint[2])
        else:
            print("No footprint or centerline found.")

    polys = None
    if line_polys:
        polys = pd.concat(line_polys)
        polys = polys.dissolve()

    footprint_merge = None
    if footprints:
        if not all(item is None for item in footprints):
            footprint_merge = pd.concat(footprints)
            footprint_merge.dissolve()
            footprint_merge.drop(columns=["OLnSEG"])
        else:
            print("Empty footprint returned.")

    # if len(line) > 0:
    #     print(f"Processing line: {line[0]['OLnFID']}, done.", flush=True)

    return footprint_merge, polys, centerline_list


def process_single_line_segment(dict_segment):
    """
    this function takes single line to work the line footprint
    (regardless it process the whole line or individual segment)
    """
    in_chm = dict_segment["in_chm"]
    corridor_th_value = dict_segment["corridor_th_value"]
    line_gpd = dict_segment["line_gpd"]

    try:
        corridor_th_value = float(corridor_th_value)
        if corridor_th_value < 0.0:
            corridor_th_value = 3.0
    except ValueError as e:
        print(f"process_single_line_segment: {e}")
        corridor_th_value = 3.0

    max_ln_width = dict_segment["max_ln_width"]
    exp_shk_cell = dict_segment["exp_shk_cell"]
    shapefile_proj = dict_segment["Proj_crs"]
    # original_col_name_list = dict_segment['org_col']

    FID = dict_segment["OLnSEG"]  # segment line feature ID
    OID = dict_segment["OLnFID"]  # original line ID for segment line

    segment_list = []
    feat = dict_segment["geometry"]
    for coord in feat.coords:
        segment_list.append(coord)

    # Find origin and destination coordinates
    x1, y1 = segment_list[0][0], segment_list[0][1]
    x2, y2 = segment_list[-1][0], segment_list[-1][1]

    # # Create Point "origin"
    # origin_point = shapely.Point([x1, y1])
    # origin = [shapes for shapes in gpd.GeoDataFrame(geometry=[origin_point],
    #                                                 crs=shapefile_proj).geometry]

    # # Create Point "destination"
    # destination_point = shapely.Point([x2, y2])
    # destination = [shapes for shapes in gpd.GeoDataFrame(geometry=[destination_point],
    #                                                      crs=shapefile_proj).geometry]

    # Buffer around line and clip cost raster and canopy raster
    # TODO: deal with NODATA
    clip_cost, out_meta = bt_common.clip_raster(in_chm, feat, max_ln_width)
    out_transform = out_meta["transform"]
    cell_size_x = out_transform[0]
    cell_size_y = -out_transform[4]

    clip_cost, clip_canopy = bt_common.cost_raster(clip_cost, out_meta)

    # Work out the corridor from both end of the centerline
    try:
        if len(clip_canopy.shape) > 2:
            clip_canopy = np.squeeze(clip_canopy, axis=0)

        transformer = rasterio.transform.AffineTransformer(out_transform)
        source = [transformer.rowcol(x1, y1)]
        destination = [transformer.rowcol(x2, y2)]

        corridor_thresh = bt_common.corridor_raster(
            clip_cost,
            out_meta,
            source,
            destination,
            (cell_size_x, cell_size_y),
            corridor_th_value,
        )

        clean_raster = algo_common.morph_raster(
            corridor_thresh, clip_canopy, exp_shk_cell, cell_size_x
        )

        # create mask for non-polygon area
        msk = np.where(clean_raster == 1, True, False)
        if clean_raster.dtype == np.int64:
            clean_raster = clean_raster.astype(np.int32)

        # Process: ndarray to shapely Polygon
        out_polygon = rasterio.features.shapes(
            clean_raster, mask=msk, transform=out_transform
        )

        # create a shapely multipolygon
        multi_polygon = []
        for shp, value in out_polygon:
            multi_polygon.append(shapely.geometry.shape(shp))
        poly = shapely.geometry.MultiPolygon(multi_polygon)

        # create a pandas dataframe for the footprint
        out_data = pd.DataFrame({"OLnFID": [OID], "OLnSEG": [FID], "geometry": poly})
        out_gdf = gpd.GeoDataFrame(out_data, geometry="geometry", crs=shapefile_proj)

        # if not bt_const.GROUPING_SEGMENT:
        #     print(
        #         f"LP: Processing line ID: {dict_segment['OLnSEG']}, done.", flush=True
        #     )

        # find contiguous corridor polygon for centerline
        corridor_poly_gpd = algo_cl.find_corridor_polygon(
            corridor_thresh, out_transform, line_gpd
        )
        centerline, status = algo_cl.find_centerline(
            corridor_poly_gpd.geometry.iloc[0], feat
        )

        return out_gdf, corridor_poly_gpd, centerline

    except Exception as e:
        print(f"process_single_line_segment: {e}")
        return None


def line_prepare(
    line_seg,
    in_chm,
    corridor_th_field,
    corridor_th_value,
    max_ln_width,
    exp_shk_cell,
    out_footprint,
    out_centerline,
):
    # get the list of original columns names
    field_list_col = field_name_list(line_seg)
    keep_field_name = []
    for col_name in line_seg.columns:
        if col_name != "geometry":
            keep_field_name.append(col_name)

    list_of_segment = []

    i = 0
    # process when shapefile is not an empty feature class
    if len(line_seg) > 0:
        for row in range(0, len(line_seg)):
            # creates a geometry object
            line_gpd = line_seg.loc[[row]]
            feat = line_gpd.geometry.iloc[0]
            if feat:
                feature_attributes = {
                    "seg_length": feat.length,
                    "geometry": feat,
                    "Proj_crs": line_seg.crs,
                    "line_gpd": line_gpd,
                }

                for col_name in keep_field_name:
                    feature_attributes[col_name] = line_seg.loc[row, col_name]
                list_of_segment.append(feature_attributes)
                i += 1

        # print(f"There are {ori_total_feat} lines to be processed.")
    else:
        print("Input line feature is corrupted, exit!")
        exit()

    # Add tools arguments into GeoDataFrame record
    for record in list_of_segment:
        record["in_chm"] = in_chm
        record["corridor_th_field"] = corridor_th_field
        record["corridor_th_value"] = record["CorridorTh"]
        record["max_ln_width"] = max_ln_width
        record["exp_shk_cell"] = exp_shk_cell
        # record["proc_seg"] = proc_seg
        record["out_footprint"] = out_footprint
        record["out_centerline"] = out_centerline
        record["org_col"] = field_list_col

    # TODO: data type changed - return list of GeoDataFrame represents each line or segment
    # returns list of list of line attributes, arguments and line gpd
    if bt_const.GROUPING_SEGMENT:
        # group line segments by line id
        def key_func(x):
            return x["OLnFID"]

        lines = []

        for key, group in itertools.groupby(list_of_segment, key_func):
            lines.append(list(group))

        return lines
    else:
        return list_of_segment


if __name__ == "__main__":
    start_time = time.time()
    print("Footprint processing started")
    print(f"Current time: {time.strftime('%b %Y %H:%M:%S', time.localtime())}")

    in_args, in_verbose = bt_common.check_arguments()
    line_footprint_abs(
        **in_args.input, processes=int(in_args.processes), verbose=in_verbose
    )
    print(f"Current time: {time.strftime('%b %Y %H:%M:%S', time.localtime())}")
