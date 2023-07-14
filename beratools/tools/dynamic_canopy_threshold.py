import os.path
from multiprocessing.pool import Pool
import geopandas
import json
import argparse
import time
import pandas
import numpy as np
import shapely
from common import *
import sys
import math


class OperationCancelledException(Exception):
    pass


def dynamic_canopy_threshold(callback, in_line, in_chm, proc_segments, off_ln_dist, canopy_percentile,
                             canopy_thresh_percentage, tree_radius, max_line_dist, canopy_avoidance,
                             exponent, full_step, processes, verbose):

    file_path, in_file_name = os.path.split(in_line)
    out_file = os.path.join(file_path, 'DynCanTh_'+in_file_name)
    line_seg = geopandas.GeoDataFrame.from_file(in_line)

    # check coordinate systems between line and raster features
    with rasterio.open(in_chm) as in_raster:
        if line_seg.crs.to_epsg() != in_raster.crs.to_epsg():
            print("Line and raster spatial references are not same, please check.")
            exit()
    del in_raster

    # Check the canopy threshold percent in 0-100 range.  If it is not, 50% will be applied
    if not 100 >= int(canopy_percentile) > 0:
        canopy_percentile = 50

    # Check the Dynamic Canopy threshold column in data. If it is not, new column will be created
    if 'DynCanTh' not in line_seg.columns.array:
        print("{} column not found in input line.\n '{}' column is create".format('DynCanTh', 'DynCanTh'))
        line_seg['DynCanTh'] = np.nan

    # Check the OLnFID column in data. If it is not, column will be created
    if 'OLnFID' not in line_seg.columns.array:
        print("{} column not found in input line.\n '{}' column is create".format('OLnFID', 'OLnFID'))
        line_seg['OLnFID'] = line_seg.index
    # else:
    #     for row in line_seg.index:
    #         if row != line_seg.loc[row,'OLnFID']:
    #             print("Warning: index and OLnFID are not consistency at index: {}.".format(row))
    #             print("Please check data")
    #             exit()
    if proc_segments:
        line_seg = split_into_segments(line_seg)
    else:
        pass

    # copy original line input to another GeoDataframe
    workln_dfL = geopandas.GeoDataFrame.copy((line_seg))
    workln_dfR = geopandas.GeoDataFrame.copy((line_seg))

    # copy parallel lines for both side of the input lines
    print("Creating offset area for surrounding forest....")
    workln_dfL, workln_dfR = multiprocessing_copyparallel_lineLR(workln_dfL, workln_dfR, processes,
                                                                 left_dis=float(off_ln_dist),
                                                                 right_dist=-float(off_ln_dist))
    workln_dfR = workln_dfR.sort_values(by=['OLnFID'])
    workln_dfL = workln_dfL.sort_values(by=['OLnFID'])
    workln_dfL = workln_dfL.reset_index(drop=True)
    workln_dfR = workln_dfR.reset_index(drop=True)

    print('%{}'.format(30))

    worklnbuffer_dfL = geopandas.GeoDataFrame.copy((workln_dfL))
    worklnbuffer_dfR = geopandas.GeoDataFrame.copy((workln_dfR))

    # buffer the parallel line in one side (extend the area into forest)
    worklnbuffer_dfL['geometry'] = shapely.buffer(workln_dfL['geometry'], distance=float(tree_radius),
                                                  cap_style=2, join_style=2, single_sided=True)
    worklnbuffer_dfR['geometry'] = shapely.buffer(workln_dfR['geometry'], distance=-float(tree_radius),
                                                  cap_style=2, join_style=2, single_sided=True)
    print("Creating offset area for surrounding forest....Done")
    print('%{}'.format(50))

    # create a New column for surrounding forest statistics:
    # 1) Height Percentile (add more in the future)
    worklnbuffer_dfL['Percentile_L'] = np.nan
    worklnbuffer_dfR['Percentile_R'] = np.nan
    line_seg['L_Pertiels'] = np.nan
    line_seg['R_Pertiels'] = np.nan
    print('%{}'.format(80))

    # calculate the Height percentile for each parallel area using CHM
    print("Calculating surrounding forest percentile LEFT..")
    worklnbuffer_dfL = multiprocessing_Percentile(worklnbuffer_dfL, int(canopy_percentile),
                                                  float(canopy_thresh_percentage), in_chm,
                                                  processes, side='left')
    worklnbuffer_dfL = worklnbuffer_dfL.sort_values(by=['OLnFID'])
    worklnbuffer_dfL = worklnbuffer_dfL.reset_index(drop=True)

    print("Calculating surrounding forest percentile RIGHT....")
    worklnbuffer_dfR = multiprocessing_Percentile(worklnbuffer_dfR, int(canopy_percentile),
                                                  float(canopy_thresh_percentage), in_chm,
                                                  processes, side='right')
    worklnbuffer_dfR = worklnbuffer_dfR.sort_values(by=['OLnFID'])
    worklnbuffer_dfR = worklnbuffer_dfR.reset_index(drop=True)

    print('%{}'.format(90))
    print("Forest percentile calculation, Done.")

    for index in (line_seg.index):
        line_seg.loc[index,'L_Pertiels'] = worklnbuffer_dfL.Percentile_L.iloc[index]
        line_seg.loc[index,'R_Pertiels'] = worklnbuffer_dfR.Percentile_R.iloc[index]
        line_seg.loc[index,'DynCanTh'] = ((worklnbuffer_dfL.DynCanTh.iloc[index] +
                                           worklnbuffer_dfR.DynCanTh.iloc[index])/2.0)

    print("Saving dynamic canopy threshold output.....")
    geopandas.GeoDataFrame.to_file(line_seg, out_file)
    print("Saving dynamic canopy threshold output.....Done")

    del line_seg, worklnbuffer_dfL, worklnbuffer_dfR, workln_dfL, workln_dfR
    if full_step:
        return out_file

    print('%{}'.format(100))


def split_line_fc(line):
    return list(map(shapely.LineString, zip(line.coords[:-1], line.coords[1:])))


def split_into_segments(df):
    odf = df
    crs = odf.crs
    if 'OLnSEG' not in odf.columns.array:
        df['OLnSEG'] = np.nan
    else:
        pass
    df = odf.assign(geometry=odf.apply(lambda x: split_line_fc(x.geometry), axis=1))
    df = df.explode()

    df['OLnSEG'] = df.groupby('OLnFID').cumcount()
    gdf = geopandas.GeoDataFrame(df, geometry=df.geometry, crs=crs)
    gdf = gdf.sort_values(by=['OLnFID', 'OLnSEG'])
    gdf = gdf.reset_index(drop=True)
    return gdf


def multiprocessing_copyparallel_lineLR(dfL, dfR, processes, left_dis, right_dist):
    try:
        line_arg = []
        total_steps = len(dfL)

        for item in dfL.index:
            item_list = [dfL,dfR, left_dis, right_dist, item]
            line_arg.append(item_list)

        featuresL = []
        featuresR = []
        chunksize = math.ceil(total_steps / processes)
        with Pool(processes=int(processes)) as pool:
            step = 0
            # execute tasks in order, process results out of order
            for resultL, resultR in pool.imap_unordered(copyparallel_lineLR, line_arg, chunksize=chunksize):
                if BT_DEBUGGING:
                    print('Got result: {}{}'.format(resultL,resultR), flush=True)
                featuresL.append(resultL)
                featuresR.append(resultR)
                step += 1
                print('%{}'.format(step / total_steps * 100))
            return geopandas.GeoDataFrame(pandas.concat(featuresL)), geopandas.GeoDataFrame(pandas.concat(featuresR))
    except OperationCancelledException:
        print("Operation cancelled")


def multiprocessing_Percentile(df, CanPercentile, CanThrPercentage, in_CHM, processes, side):
    try:
        line_arg = []
        total_steps = len(df)
        if side == 'left':
            PerCol = 'Percentile_L'
        else:
            PerCol = 'Percentile_R'

        for item in df.index:
            item_list = [df.iloc[[item]], CanPercentile, CanThrPercentage, in_CHM, item, PerCol]
            line_arg.append(item_list)
        features = []
        chunksize = math.ceil(total_steps / processes)
        with Pool(processes=int(processes)) as pool:

            step = 0
            # execute tasks in order, process results out of order
            try:
                for result in pool.imap_unordered(cal_percentile, line_arg, chunksize=chunksize):
                    if BT_DEBUGGING:
                        print('Got result: {}'.format(result), flush=True)
                    features.append(result)
                    step += 1
                    print('%{}'.format(step / total_steps * 100))
            except Exception:
                print(Exception)
                raise
            del line_arg
            return geopandas.GeoDataFrame(pandas.concat(features))

    except OperationCancelledException:
        print("Operation cancelled")


def cal_percentile(line_arg):
    try:
        df = line_arg[0]
        CanPercentile = line_arg[1]
        CanThrPercentage = line_arg[2]
        in_CHM = line_arg[3]
        row_index = line_arg[4]
        PerCol = line_arg[5]
        line_buffer = df.loc[row_index, 'geometry']
    except Exception as e:
        print(e)
        print("Assigning variable on index:{} Error: ".format(line_arg) + sys.exc_info())
        exit()

    # TODO: temporary workaround for exception causing not percentile defined
    percentile = 0
    Dyn_Canopy_Threshold = 0.05
    try:
        with rasterio.open(in_CHM) as raster:
            clipped_raster, out_transform = rasterio.mask.mask(raster, [line_buffer], crop=True,
                                                               nodata=-9999, filled=True)
            clipped_raster = np.squeeze(clipped_raster, axis=0)

            # mask all -9999 (nodata) value cells
            masked_raster = np.ma.masked_where(clipped_raster == -9999, clipped_raster)
            filled_raster = np.ma.filled(masked_raster, np.nan)

            # Calculate the percentile
            # masked_mean = np.ma.mean(masked_raster)
            percentile = np.nanpercentile(filled_raster, CanPercentile,method='hazen')
            median = np.nanmedian(filled_raster)
            if percentile>0.05:  # (percentile+median)>0.0:
                # ((50 Percentile + user defined percentile)/2)x(User defined Canopy Threshold Percentage)
                # Dyn_Canopy_Threshold = ((percentile+median)/2.0) * (CanThrPercentage / 100.0)
                # (user defined percentile)x(User defined Canopy Threshold Percentage)
                Dyn_Canopy_Threshold = percentile * (CanThrPercentage / 100.0)
            else:
                # print("(percentile)<0.05 @ {}".format(row_index))
                Dyn_Canopy_Threshold=0.05

            del clipped_raster, out_transform
        del raster
    # return the generated value
    except Exception as e:
        print(e)
        print(sys.exc_info())

    try:
        df.loc[row_index, PerCol] = percentile
        df.loc[row_index, 'DynCanTh'] = Dyn_Canopy_Threshold
        return df
    except Exception as e:
        print("Error writing Percentile and Dynamic Canopy into table: "+sys.exc_info())


def copyparallel_lineLR(line_arg):
    dfL = line_arg[0]
    dfR = line_arg[1]

    # Simplify input center lines
    lineL = dfL.loc[line_arg[4], 'geometry'].simplify(tolerance=0.05, preserve_topology=True)
    lineR = dfL.loc[line_arg[4], 'geometry'].simplify(tolerance=0.05, preserve_topology=True)
    offset_distL = float(line_arg[2])
    offset_distR= float(line_arg[3])

    # Older alternative method to the offset_curve() method,
    # but uses resolution instead of quad_segs and a side keyword (‘left’ or ‘right’) instead
    # of sign of the distance. This method is kept for backwards compatibility for now,
    # but it is recommended to use offset_curve() instead.
    # (ref: https://shapely.readthedocs.io/en/stable/manual.html#object.offset_curve)
    parallel_lineL = lineL.parallel_offset(distance=offset_distL, side='left',
                                          join_style=shapely.BufferJoinStyle.mitre)
    parallel_lineR = lineR.parallel_offset(distance=-offset_distR, side='right',
                                         join_style=shapely.BufferJoinStyle.mitre)

    if not parallel_lineL.is_empty:
        dfL.loc[line_arg[4], 'geometry'] = parallel_lineL
    if not parallel_lineR.is_empty:
        dfR.loc[line_arg[4], 'geometry'] = parallel_lineR

    return dfL.iloc[[line_arg[4]]],dfR.iloc[[line_arg[4]]]


if __name__ == '__main__':
    start_time = time.time()
    print('Starting Dynamic Canopy Threshold calculation processing\n @ {}'.format(
        time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=json.loads)
    parser.add_argument('-p', '--processes')
    parser.add_argument('-v', '--verbose')
    args = parser.parse_args()
    args.input['full_step'] = False

    verbose = True if args.verbose == 'True' else False
    dynamic_canopy_threshold(print, **args.input, processes=int(args.processes), verbose=verbose)

    print('%{}'.format(100))
    print('Finishing Dynamic Canopy Threshold calculation @ {}\n(or in {} second)'.format(
        time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), round(time.time() - start_time, 5)))
