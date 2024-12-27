import os.path
from multiprocessing.pool import Pool
import geopandas as gpd
import json
import argparse
import time
import pandas as pd
import numpy as np
import shapely
from shapely import ops
from beratools.tools.common import (
    compare_crs,
    vector_crs,
    raster_crs,
    chk_df_multipart,
    check_arguments,
    clip_raster,
)

from beratools.core.constants import BT_NODATA
from beratools.core.tool_base import execute_multiprocessing
import sys
import math


class OperationCancelledException(Exception):
    pass


def split_line_fc(line):
    if line:
        return list(map(shapely.LineString, zip(line.coords[:-1], line.coords[1:])))
    else:
        return None


def split_into_segments(df):
    odf = df
    crs = odf.crs
    if "OLnSEG" not in odf.columns.array:
        df["OLnSEG"] = np.nan
    else:
        pass
    df = odf.assign(geometry=odf.apply(lambda x: split_line_fc(x.geometry), axis=1))
    df = df.explode()

    df["OLnSEG"] = df.groupby("OLnFID").cumcount()
    gdf = gpd.GeoDataFrame(df, geometry=df.geometry, crs=crs)
    gdf = gdf.sort_values(by=["OLnFID", "OLnSEG"])
    gdf = gdf.reset_index(drop=True)
    return gdf


def multiringbuffer(df, nrings, ringdist):
    """
    Buffers an input dataframes geometry nring (number of rings) times, with a distance between
    rings of ringdist and returns a list of non overlapping buffers
    """

    rings = []  # A list to hold the individual buffers
    for ring in np.arange(0, ringdist, nrings):  # For each ring (1, 2, 3, ..., nrings)
        big_ring = df["geometry"].buffer(
            nrings + ring, single_sided=True, cap_style="flat"
        )  # Create one big buffer
        small_ring = df["geometry"].buffer(
            ring, single_sided=True, cap_style="flat"
        )  # Create one smaller one
        the_ring = big_ring.difference(
            small_ring
        )  # Difference the big with the small to create a ring
        if (
            ~shapely.is_empty(the_ring)
            or ~shapely.is_missing(the_ring)
            or not None
            or ~the_ring.area == 0
        ):
            if isinstance(the_ring, shapely.MultiPolygon) or isinstance(
                the_ring, shapely.Polygon
            ):
                rings.append(the_ring)  # Append the ring to the rings list
            else:
                if isinstance(the_ring, shapely.GeometryCollection):
                    for i in range(0, len(the_ring.geoms)):
                        if not isinstance(the_ring.geoms[i], shapely.LineString):
                            rings.append(the_ring.geoms[i])
        # print(" %{} ".format((ring / ringdist) * 100))

    return rings  # return the list


def rate_of_change(in_arg):
    x = in_arg[0]
    Olnfid = in_arg[1]
    Olnseg = in_arg[2]
    side = in_arg[3]
    df = in_arg[4]
    index = in_arg[5]

    # Since the x interval is 1 unit, the array 'diff' is the rate of change (slope)
    diff = np.ediff1d(x)
    cut_dist = len(x) / 5

    median_percentile = np.nanmedian(x)
    if not np.isnan(median_percentile):
        cut_percentile = math.floor(median_percentile)
    else:
        cut_percentile = 0.5
    found = False
    changes = 1.50
    Change = np.insert(diff, 0, 0)
    scale_down = 1

    # test the rate of change is > than 150% (1.5), if it is
    # no result found then lower to 140% (1.4) until 110% (1.1)
    try:
        while not found and changes >= 1.1:
            for ii in range(0, len(Change) - 1):
                if x[ii] >= 0.5:
                    if (Change[ii]) >= changes:
                        cut_dist = (ii + 1) * scale_down
                        cut_percentile = math.floor(x[ii])

                        if 0.5 >= cut_percentile:
                            if cut_dist > 5:
                                cut_percentile = 2
                                cut_dist = cut_dist * scale_down**3
                                # print(
                                #     f"{side}: OLnFID:{Olnfid}, OLnSEG: {Olnseg} @<0.5  found and modified",
                                #     flush=True,
                                # )
                        elif 0.5 < cut_percentile <= 5.0:
                            if cut_dist > 6:
                                cut_dist = cut_dist * scale_down**3  # 4.0
                                # print(
                                #     f"{side}: OLnFID:{Olnfid}, OLnSEG: {Olnseg} @0.5-5.0  found and modified",
                                #     flush=True,
                                # )
                        elif 5.0 < cut_percentile <= 10.0:
                            if cut_dist > 8:  # 5
                                cut_dist = cut_dist * scale_down**3
                                # print(
                                #     f"{side}: OLnFID:{Olnfid}, OLnSEG: {Olnseg} @5-10  found and modified", 
                                #     flush=True,
                                # )
                        elif 10.0 < cut_percentile <= 15:
                            if cut_dist > 5:
                                cut_dist = cut_dist * scale_down**3  # 5.5
                                # print(
                                #     f"{side}: OLnFID:{Olnfid}, OLnSEG: {Olnseg} @10-15  found and modified", 
                                #     flush=True,
                                # )
                        elif 15 < cut_percentile:
                            if cut_dist > 4:
                                cut_dist = cut_dist * scale_down**2
                                cut_percentile = 15.5
                                # print(
                                #     f"{side}: OLnFID:{Olnfid}, OLnSEG: {Olnseg} @>15  found and modified", 
                                #     flush=True,
                                # )
                        found = True
                        # print(
                        #     f"{side}: OLnFID:{Olnfid}, OLnSEG: {Olnseg} rate of change found",
                        #     flush=True,
                        # )
                        break
            changes = changes - 0.1

    except IndexError:
        pass

    # if still is no result found, lower to 10% (1.1), if no result found then default is used
    if not found:
        if 0.5 >= median_percentile:
            cut_dist = 4 * scale_down  # 3
            cut_percentile = 0.5
        elif 0.5 < median_percentile <= 5.0:
            cut_dist = 4.5 * scale_down  # 4.0
            cut_percentile = math.floor(median_percentile)
        elif 5.0 < median_percentile <= 10.0:
            cut_dist = 5.5 * scale_down  # 5
            cut_percentile = math.floor(median_percentile)
        elif 10.0 < median_percentile <= 15:
            cut_dist = 6 * scale_down  # 5.5
            cut_percentile = math.floor(median_percentile)
        elif 15 < median_percentile:
            cut_dist = 5 * scale_down  # 5
            cut_percentile = 15.5
        # print(
        #     "{}: OLnFID:{}, OLnSEG: {} Estimated".format(side, Olnfid, Olnseg),
        #     flush=True,
        # )
    if side == "Right":
        df["RDist_Cut"] = cut_dist
        df["CR_CutHt"] = cut_percentile
    elif side == "Left":
        df["LDist_Cut"] = cut_dist
        df["CL_CutHt"] = cut_percentile

    return df


def cal_percentileRing(line_arg):
    try:
        df = line_arg[0]
        # CanPercentile = line_arg[1]
        # CanThrPercentage = line_arg[2]
        in_CHM = line_arg[3]
        row_index = line_arg[4]
        PerCol = line_arg[5]

        line_buffer = df.loc[row_index, "geometry"]
        if line_buffer.is_empty or shapely.is_missing(line_buffer):
            return None
        if line_buffer.has_z:
            line_buffer = ops.transform(lambda x, y, z=None: (x, y), line_buffer)

    except Exception as e:
        print(e)
        print(
            "Assigning variable on index:{} Error: ".format(line_arg) + sys.exc_info()
        )
        exit()

    # TODO: temporary workaround for exception causing not percentile defined
    percentile = 0.5
    Dyn_Canopy_Threshold = 0.05
    try:
        clipped_raster, _ = clip_raster(in_CHM, line_buffer, 0)
        clipped_raster = np.squeeze(clipped_raster, axis=0)

        # mask all -9999 (nodata) value cells
        masked_raster = np.ma.masked_where(clipped_raster == BT_NODATA, clipped_raster)
        filled_raster = np.ma.filled(masked_raster, np.nan)

        # Calculate the percentile
        percentile = np.nanpercentile(filled_raster, 50)

        if percentile > 1:
            Dyn_Canopy_Threshold = percentile * (0.3)
        else:
            Dyn_Canopy_Threshold = 1
    # return the generated value
    except Exception as e:
        print(e)
        print("Default values are used.")

    finally:
        df.loc[row_index, PerCol] = percentile
        df.loc[row_index, "DynCanTh"] = Dyn_Canopy_Threshold
        return df


def prepare_multiprocessing_rate_of_change(line_seg, worklnbuffer_dfLRing, worklnbuffer_dfRRing):
    in_argsL = []
    in_argsR = []

    for index in line_seg.index:
        Olnfid = int(line_seg.OLnFID.iloc[index])
        Olnseg = int(line_seg.OLnSEG.iloc[index])
        sql_dfL = worklnbuffer_dfLRing.loc[
            (worklnbuffer_dfLRing["OLnFID"] == Olnfid)
            & (worklnbuffer_dfLRing["OLnSEG"] == Olnseg)
        ].sort_values(by=["iRing"])
        PLRing = list(sql_dfL["Percentile_LRing"])

        sql_dfR = worklnbuffer_dfRRing.loc[
            (worklnbuffer_dfRRing["OLnFID"] == Olnfid)
            & (worklnbuffer_dfRRing["OLnSEG"] == Olnseg)
        ].sort_values(by=["iRing"])
        PRRing = list(sql_dfR["Percentile_RRing"])

        in_argsL.append([PLRing, Olnfid, Olnseg, "Left", line_seg.loc[index], index])
        in_argsR.append([PRRing, Olnfid, Olnseg, "Right", line_seg.loc[index], index])

    total_steps = len(in_argsL) + len(in_argsR)
    return in_argsL, in_argsR, total_steps


def multiprocessing_rate_of_change(
    line_seg, worklnbuffer_dfLRing, worklnbuffer_dfRRing, processes
):
    line_seg["CL_CutHt"] = np.nan
    line_seg["CR_CutHt"] = np.nan
    line_seg["RDist_Cut"] = np.nan
    line_seg["LDist_Cut"] = np.nan

    in_argsL, in_argsR, total_steps = prepare_multiprocessing_rate_of_change(
        line_seg, worklnbuffer_dfLRing, worklnbuffer_dfRRing
    )

    featuresL = []
    featuresR = []
    featuresL = execute_multiprocessing(
        rate_of_change, in_argsL, "Change In Buffer Area", processes, 1, verbose=False
    )
    gpdL = gpd.GeoDataFrame(pd.concat(featuresL, axis=1).T)

    featuresR = execute_multiprocessing(
        rate_of_change, in_argsR, "Change In Buffer Area", processes, 1, verbose=False
    )
    gpdR = gpd.GeoDataFrame(pd.concat(featuresR, axis=1).T)

    for index in line_seg.index:
        lnfid = line_seg.OLnFID.iloc[index]
        Olnseg = line_seg.OLnSEG.iloc[index]
        line_seg.loc[index, "RDist_Cut"] = float(
            gpdR.loc[(gpdR.OLnFID == lnfid) & (gpdR.OLnSEG == Olnseg)]["RDist_Cut"]
        )
        line_seg.loc[index, "LDist_Cut"] = float(
            gpdL.loc[(gpdL.OLnFID == lnfid) & (gpdL.OLnSEG == Olnseg)]["LDist_Cut"]
        )
        line_seg.loc[index, "CL_CutHt"] = float(
            gpdL.loc[(gpdL.OLnFID == lnfid) & (gpdL.OLnSEG == Olnseg)]["CL_CutHt"]
        )
        line_seg.loc[index, "CR_CutHt"] = float(
            gpdR.loc[(gpdR.OLnFID == lnfid) & (gpdR.OLnSEG == Olnseg)]["CR_CutHt"]
        )
        line_seg.loc[index, "DynCanTh"] = (
            line_seg.loc[index, "CL_CutHt"] + line_seg.loc[index, "CR_CutHt"]
        ) / 2

    return line_seg


def prepare_multiprocessing_percentile(
    df, CanPercentile, CanThrPercentage, in_CHM, side
):
    line_arg = []
    total_steps = len(df)
    cal_percentile = cal_percentileRing

    if side == "LRing":
        PerCol = "Percentile_LRing"
        cal_percentile = cal_percentileRing
        which_side = "left"
    elif side == "RRing":
        PerCol = "Percentile_RRing"
        which_side = "right"
        cal_percentile = cal_percentileRing

    print(
        f"Calculating surrounding {which_side} forest population for buffer area ...")

    for item in df.index:
        item_list = [
            df.iloc[[item]],
            CanPercentile,
            CanThrPercentage,
            in_CHM,
            item,
            PerCol,
        ]
        line_arg.append(item_list)

    return line_arg, total_steps, cal_percentile


def multiprocessing_percentile(
    df, CanPercentile, CanThrPercentage, in_CHM, processes, side
):
    line_arg, total_steps, cal_percentile = prepare_multiprocessing_percentile(
        df, CanPercentile, CanThrPercentage, in_CHM, side
    )

    features = []
    features = execute_multiprocessing(
        cal_percentile, line_arg, "Calculate Percentile", processes, workers=1
    )
    gdf_percentile = gpd.GeoDataFrame(pd.concat(features))

    gdf_percentile = gdf_percentile.sort_values(by=["OLnFID", "OLnSEG", "iRing"])
    gdf_percentile = gdf_percentile.reset_index(drop=True)

    return gdf_percentile


def prepare_line_seg(in_line, canopy_percentile):
    file_path, in_file_name = os.path.split(in_line)
    out_file = os.path.join(file_path, "DynCanTh_" + in_file_name)
    line_seg = gpd.GeoDataFrame.from_file(in_line)

    # Check the canopy threshold percent in 0-100 range.  If it is not, 50% will be applied
    if not 100 >= int(canopy_percentile) > 0:
        canopy_percentile = 50

    # Check the Dynamic Canopy threshold column in data. If it is not, new column will be created
    if "DynCanTh" not in line_seg.columns.array:
        print("New column created: {}".format("DynCanTh"))
        line_seg["DynCanTh"] = np.nan

    # Check the OLnFID column in data. If it is not, column will be created
    if "OLnFID" not in line_seg.columns.array:
        print("New column created: {}".format("OLnFID"))
        line_seg["OLnFID"] = line_seg.index

    # Check the OLnSEG column in data. If it is not, column will be created
    if "OLnSEG" not in line_seg.columns.array:
        print("New column created: {}".format("OLnSEG"))
        line_seg["OLnSEG"] = 0

    line_seg = chk_df_multipart(line_seg, "LineString")[0]

    proc_segments = False
    if proc_segments:
        line_seg = split_into_segments(line_seg)
    else:
        pass

    return canopy_percentile, out_file, line_seg


def prepar_ring_buffer(workln_dfC, nrings, ringdist):
    gdf_buffer_ring = gpd.GeoDataFrame.copy((workln_dfC))

    print("Create ring buffer for input line to find the forest edge....")

    # Create a column with the rings as a list
    gdf_buffer_ring["mgeometry"] = gdf_buffer_ring.apply(
        lambda x: multiringbuffer(df=x, nrings=nrings, ringdist=ringdist), axis=1
    )

    # Explode to create a row for each ring
    gdf_buffer_ring = gdf_buffer_ring.explode("mgeometry")
    gdf_buffer_ring = gdf_buffer_ring.set_geometry("mgeometry")
    gdf_buffer_ring = (
        gdf_buffer_ring.drop(columns=["geometry"])
        .rename_geometry("geometry")
        .set_crs(workln_dfC.crs)
    )
    gdf_buffer_ring["iRing"] = gdf_buffer_ring.groupby(["OLnFID", "OLnSEG"]).cumcount()
    gdf_buffer_ring = gdf_buffer_ring.sort_values(by=["OLnFID", "OLnSEG", "iRing"])
    gdf_buffer_ring = gdf_buffer_ring.reset_index(drop=True)
    return gdf_buffer_ring


def main_canopy_threshold_relative(
    callback,
    in_line,
    in_chm,
    canopy_percentile,
    canopy_thresh_percentage,
    processes,
    verbose,
):
    # check coordinate systems between line and raster features
    # with rasterio.open(in_chm) as in_raster:
    if compare_crs(vector_crs(in_line), raster_crs(in_chm)):
        pass
    else:
        print("Line and raster spatial references are not same, please check.")
        exit()

    canopy_percentile, out_file, line_seg = prepare_line_seg(in_line, canopy_percentile)

    # copy original line input to another GeoDataframe
    workln_dfC = gpd.GeoDataFrame.copy((line_seg))
    workln_dfC.geometry = workln_dfC.geometry.simplify(
        tolerance=0.5, preserve_topology=True
    )

    worklnbuffer_dfLRing = prepar_ring_buffer(workln_dfC, 1, 15)
    worklnbuffer_dfRRing = prepar_ring_buffer(workln_dfC, -1, -15)
    print("Ring buffers are created.")

    worklnbuffer_dfRRing["Percentile_RRing"] = np.nan
    worklnbuffer_dfLRing["Percentile_LRing"] = np.nan

    # calculate the Height percentile for each parallel area using CHM
    worklnbuffer_dfLRing = multiprocessing_percentile(
        worklnbuffer_dfLRing,
        int(canopy_percentile),
        float(canopy_thresh_percentage),
        in_chm,
        processes,
        side="LRing",
    )

    worklnbuffer_dfRRing = multiprocessing_percentile(
        worklnbuffer_dfRRing,
        int(canopy_percentile),
        float(canopy_thresh_percentage),
        in_chm,
        processes,
        side="RRing",
    )

    result = multiprocessing_rate_of_change(
        line_seg, worklnbuffer_dfLRing, worklnbuffer_dfRRing, processes
    )

    print("Saving percentile information to input line ...")
    gpd.GeoDataFrame.to_file(result, out_file)
    print("Task done.")

    return out_file


if __name__ == "__main__":
    start_time = time.time()
    print("Dynamic Canopy Threshold Started")

    in_args, in_verbose = check_arguments()
    main_canopy_threshold_relative(
        print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose
    )

    print("Dynamic Canopy Threshold finished")
    print('Elapsed time: {}'.format(time.time() - start_time))
