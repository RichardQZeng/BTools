import os.path
from multiprocessing.pool import Pool
import geopandas as gpd
import json
import argparse
import time
import pandas as pd
import numpy as np
import shapely
from common import *
import sys
import math


class OperationCancelledException(Exception):
    pass


def main_canopy_threshold_relative(callback, in_line, in_chm, off_ln_dist, canopy_percentile,
                                   canopy_thresh_percentage, tree_radius, max_line_dist, canopy_avoidance,
                                   exponent, full_step, processes, verbose):

    file_path, in_file_name = os.path.split(in_line)
    out_file = os.path.join(file_path, 'DynCanTh_'+in_file_name)
    line_seg = gpd.GeoDataFrame.from_file(in_line)

    # check coordinate systems between line and raster features
    with rasterio.open(in_chm) as in_raster:
        if compare_crs(vector_crs(in_line), raster_crs(in_chm)):
            #Do nothing
            pass
        else:
            print("Line and raster spatial references are not same, please check.")
            exit()
    del in_raster

    # Check the canopy threshold percent in 0-100 range.  If it is not, 50% will be applied
    if not 100 >= int(canopy_percentile) > 0:
        canopy_percentile = 50

    # Check the Dynamic Canopy threshold column in data. If it is not, new column will be created
    if 'DynCanTh' not in line_seg.columns.array:
        if BT_DEBUGGING:
            print("{} column not found in input line".format('DynCanTh'))
        print("New column created: {}".format('DynCanTh'))
        line_seg['DynCanTh'] = np.nan

    # Check the OLnFID column in data. If it is not, column will be created
    if 'OLnFID' not in line_seg.columns.array:
        if BT_DEBUGGING:
            print("{} column not found in input line".format('OLnFID'))

        print("New column created: {}".format('OLnFID'))
        line_seg['OLnFID'] = line_seg.index

    # Check the OLnSEG column in data. If it is not, column will be created
    if 'OLnSEG' not in line_seg.columns.array:
        if BT_DEBUGGING:
            print("{} column not found in input line".format('OLnSEG'))

        print("New column created: {}".format('OLnSEG'))
        line_seg['OLnSEG'] = 0

    proc_segments = False
    if proc_segments:
        line_seg = split_into_segments(line_seg)
    else:
        # line_seg=split_into_Equal_Nth_segments(line_seg,100)
        pass

    # copy original line input to another GeoDataframe
    # workln_dfL = gpd.GeoDataFrame.copy((line_seg))
    # workln_dfR = gpd.GeoDataFrame.copy((line_seg))
    workln_dfC = gpd.GeoDataFrame.copy((line_seg))
    workln_dfC.geometry = workln_dfC.geometry.simplify(tolerance=0.05, preserve_topology=True)

    print('%{}'.format(5))

    worklnbuffer_dfLRing = gpd.GeoDataFrame.copy((workln_dfC))
    worklnbuffer_dfRRing = gpd.GeoDataFrame.copy((workln_dfC))

    print('Create ring buffer from centerline to find the edge....')
    def multiringbuffer(df, nrings, ringdist):
        """Buffers an input dataframes geometry nring (number of rings) times, with a distance between rings of ringdist and
    returns a list of non overlapping buffers"""

        rings = []  # A list to hold the individual buffers
        for ring in np.arange(0, ringdist, nrings):  # For each ring (1, 2, 3, ..., nrings)
            big_ring = df["geometry"].buffer(nrings + ring,single_sided=True, cap_style='flat')  # Create one big buffer
            small_ring = df["geometry"].buffer(ring,single_sided=True,cap_style='flat')  # Create one smaller one
            the_ring = big_ring.difference(small_ring)  # Difference the big with the small to create a ring
            # if ~shapely.is_empty(the_ring) or ~shapely.is_missing(the_ring) or ~None or ~the_ring.area==0:
            rings.append(the_ring)  # Append the ring to the rings list

        return rings  # return the list

    # Create a column with the rings as a list
    worklnbuffer_dfLRing['mgeometry']= worklnbuffer_dfLRing.apply(lambda x: multiringbuffer(df=x, nrings=float(1),
                                                                            ringdist=float(off_ln_dist)), axis=1)

    worklnbuffer_dfLRing = worklnbuffer_dfLRing.explode("mgeometry")  # Explode to create a row for each ring

    worklnbuffer_dfLRing = worklnbuffer_dfLRing.set_geometry("mgeometry")
    worklnbuffer_dfLRing = worklnbuffer_dfLRing.drop(columns=["geometry"]).rename_geometry("geometry").set_crs(workln_dfC.crs)
    worklnbuffer_dfLRing['iRing'] = worklnbuffer_dfLRing.groupby(['OLnFID', 'OLnSEG']).cumcount()
    worklnbuffer_dfLRing = worklnbuffer_dfLRing.sort_values(by=['OLnFID', 'OLnSEG','iRing'])
    worklnbuffer_dfLRing = worklnbuffer_dfLRing.reset_index(drop=True)

    worklnbuffer_dfRRing['mgeometry'] = worklnbuffer_dfRRing.apply(
        lambda x: multiringbuffer(df=x, nrings=-float(1),
                                  ringdist=-float(off_ln_dist)), axis=1)

    worklnbuffer_dfRRing = worklnbuffer_dfRRing.explode("mgeometry")  # Explode to create a row for each ring

    worklnbuffer_dfRRing = worklnbuffer_dfRRing.set_geometry("mgeometry")
    worklnbuffer_dfRRing = worklnbuffer_dfRRing.drop(columns=["geometry"]).rename_geometry("geometry").set_crs(
        workln_dfC.crs)
    worklnbuffer_dfRRing['iRing'] = worklnbuffer_dfRRing.groupby(['OLnFID', 'OLnSEG']).cumcount()
    worklnbuffer_dfRRing = worklnbuffer_dfRRing.sort_values(by=['OLnFID', 'OLnSEG','iRing'])
    worklnbuffer_dfRRing = worklnbuffer_dfRRing.reset_index(drop=True)


    print("Task done.")
    print('%{}'.format(20))

    worklnbuffer_dfRRing['Percentile_RRing'] = np.nan
    worklnbuffer_dfLRing['Percentile_LRing'] = np.nan
    line_seg['L_Pertile'] = np.nan
    line_seg['R_Pertile'] = np.nan
    line_seg['CL_CutHt'] = np.nan
    line_seg['CR_CutHt'] = np.nan
    line_seg['RDist_Cut'] = np.nan
    line_seg['LDist_Cut'] = np.nan
    print('%{}'.format(80))

   # calculate the Height percentile for each parallel area using CHM
    print("Calculating surrounding forest percentile from centerline buffer area ...")
    worklnbuffer_dfLRing = multiprocessing_Percentile(worklnbuffer_dfLRing, int(canopy_percentile),
                                                   float(canopy_thresh_percentage), in_chm,
                                                   processes, side='LRing')

    worklnbuffer_dfLRing = worklnbuffer_dfLRing.sort_values(by=['OLnFID','OLnSEG','iRing'])
    worklnbuffer_dfLRing = worklnbuffer_dfLRing.reset_index(drop=True)

    print("Calculating ...")
    worklnbuffer_dfRRing = multiprocessing_Percentile(worklnbuffer_dfRRing, int(canopy_percentile),
                                                      float(canopy_thresh_percentage), in_chm,
                                                      processes, side='RRing')

    worklnbuffer_dfRRing = worklnbuffer_dfRRing.sort_values(by=['OLnFID','OLnSEG','iRing'])
    worklnbuffer_dfRRing = worklnbuffer_dfRRing.reset_index(drop=True)

    gpd.GeoDataFrame.to_file(worklnbuffer_dfLRing, os.path.join(file_path,"worklnbuffer_dfLRing_Percentile.shp"))
    gpd.GeoDataFrame.to_file(worklnbuffer_dfRRing, os.path.join(file_path,"worklnbuffer_dfRRing_percentile.shp"))

    def rate_of_change(x):
        #Since the x interval is 1 unit, the array 'diff' is the rate of change (slope)
        diff = np.ediff1d(x)
        cut_dist = len(x)/5
        cut_percentile = np.nanmedian(x)
        found = False
        changes =1.50
        Change = np.insert(diff, 0, 0)

        # test the rate of change is > than the 50% (1.5), if it is
        # no result found then lower to 30% (1.3) until 10% (1.1)
        while not found and changes >1.0:
            for ii in range(0, len(Change)-1):
                try:
                    if x[ii]>=0.5:
                        if (Change[ii]) >= changes:
                            cut_dist = ii+1
                            cut_percentile = math.floor(x[ii])
                            found = True
                            break
                    changes = changes - 0.1
                except IndexError:
                    pass

        # if still is no result found, lower to 10% (1.1), if no result found then default is used
        if not found:
            if cut_percentile<=0.5:
                cut_dist = len(x) / 3
                cut_percentile=0.5
            elif cut_percentile>=3.0 and cut_percentile<=10.0:
                cut_dist = 6
                cut_percentile = np.nanmedian(x)
            elif cut_percentile >10:
                cut_dist = 3
                cut_percentile = np.nanmedian(x)

        return cut_dist,cut_percentile

    print("Finding edge............")
    for index in (line_seg.index):
        Olnfid=line_seg.OLnFID.iloc[index]
        Olnseg = line_seg.OLnSEG.iloc[index]
        # worklnbuffer_dfRRing['Percentile_RRing'] = np.nan
        # worklnbuffer_dfLRing['Percentile_LRing'] = np.nan
        sql_dfL=worklnbuffer_dfLRing.loc[(worklnbuffer_dfLRing['OLnFID']==Olnfid) & (worklnbuffer_dfLRing['OLnSEG']==Olnseg)].sort_values(by=['iRing'])
        PLRing= list(sql_dfL['Percentile_LRing'])


        #Testing where the rate of chenage is more than 30% or  more
        LStd,RL_Percentile = rate_of_change(PLRing)



        sql_dfR = worklnbuffer_dfRRing.loc[(worklnbuffer_dfRRing['OLnFID']==Olnfid) & (worklnbuffer_dfRRing['OLnSEG']==Olnseg)].sort_values(by=['iRing'])
        PRRing=list(sql_dfR['Percentile_RRing'])

        #Testing where the rate of chenage is more than 30% or  more
        RStd,RR_Percentile = rate_of_change(PRRing)

        line_seg.loc[index,'RDist_Cut'] = RStd
        line_seg.loc[index,'LDist_Cut'] = LStd
        line_seg.loc[index, 'CL_CutHt'] = RL_Percentile
        line_seg.loc[index, 'CR_CutHt'] = RR_Percentile

    print('%{}'.format(40))
    print("Task done.")

    # copy parallel lines for both side of the input lines
    print("Creating offset area for surrounding forest ...")
    workln_dfL, workln_dfR  = multiprocessing_copyparallel_lineLRC(line_seg, line_seg, line_seg,
                                                                              processes,
                                                                              left_dis=float(off_ln_dist),
                                                                              right_dist=-float(off_ln_dist),
                                                                              center_dist=float(off_ln_dist))

    workln_dfL = workln_dfL.sort_values(by=['OLnFID','OLnSEG'])
    workln_dfL = workln_dfL.reset_index(drop=True)
    workln_dfR = workln_dfR.sort_values(by=['OLnFID','OLnSEG'])
    workln_dfR = workln_dfR.reset_index(drop=True)

    worklnbuffer_dfL = gpd.GeoDataFrame.copy((workln_dfL))
    worklnbuffer_dfR = gpd.GeoDataFrame.copy((workln_dfR))


    # create a New column for surrounding forest statistics:
    # 1) Height Percentile (add more in the future)
    worklnbuffer_dfL['Percentile_L'] = np.nan
    worklnbuffer_dfR['Percentile_R'] = np.nan


    worklnbuffer_dfL['geometry'] = shapely.buffer(workln_dfL['geometry'], distance=float(tree_radius),
                                                  cap_style=2, join_style=2, single_sided=True)
    worklnbuffer_dfR['geometry'] = shapely.buffer(workln_dfR['geometry'], distance=-float(tree_radius),
                                                  cap_style=2, join_style=2, single_sided=True)

    print("Calculating surrounding forest percentile from LEFT of centerline...")
    worklnbuffer_dfL = multiprocessing_Percentile(worklnbuffer_dfL, int(canopy_percentile),
                                                  float(canopy_thresh_percentage), in_chm,
                                                  processes, side='left')
    worklnbuffer_dfL = worklnbuffer_dfL.sort_values(by=['OLnFID'])
    worklnbuffer_dfL = worklnbuffer_dfL.reset_index(drop=True)
    print("Task done.")
    #
    print("Calculating surrounding forest percentile from RIGHT of centerline ...")
    worklnbuffer_dfR = multiprocessing_Percentile(worklnbuffer_dfR, int(canopy_percentile),
                                                  float(canopy_thresh_percentage), in_chm,
                                                  processes, side='right')
    worklnbuffer_dfR = worklnbuffer_dfR.sort_values(by=['OLnFID'])
    worklnbuffer_dfR = worklnbuffer_dfR.reset_index(drop=True)
    print("Task done.")


    for index in (line_seg.index):
        line_seg.loc[index, 'L_Pertile'] = worklnbuffer_dfL.Percentile_L.iloc[index]
        line_seg.loc[index, 'R_Pertile'] = worklnbuffer_dfR.Percentile_R.iloc[index]

        line_seg.loc[index, 'DynCanTh'] = ( line_seg.loc[index, 'CL_CutHt'] + line_seg.loc[index, 'CR_CutHt'])/2

    print("Saving dynamic canopy threshold output ...")
    gpd.GeoDataFrame.to_file(line_seg, out_file)
    print("Task done.")

    del line_seg, worklnbuffer_dfL, worklnbuffer_dfR, workln_dfL, workln_dfR,  worklnbuffer_dfRRing, worklnbuffer_dfLRing, workln_dfC
    if full_step:
        return out_file

    print('%{}'.format(100))


def split_line_fc(line):
    if line:
        return list(map(shapely.LineString, zip(line.coords[:-1], line.coords[1:])))
    else:
        return None


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
    gdf = gpd.GeoDataFrame(df, geometry=df.geometry, crs=crs)
    gdf = gdf.sort_values(by=['OLnFID', 'OLnSEG'])
    gdf = gdf.reset_index(drop=True)
    return gdf


def multiprocessing_copyparallel_lineLRC(dfL, dfR,dfc, processes, left_dis, right_dist,center_dist):
    try:
        line_arg = []
        total_steps = len(dfL)

        for item in dfL.index:
            item_list = [dfL, dfR,dfc, left_dis, right_dist, center_dist,item]
            line_arg.append(item_list)

        featuresL = []
        featuresR = []
        #featuresC = []
        result = None
        step = 0
        # chunksize = math.ceil(total_steps / processes)

        if PARALLEL_MODE == MODE_MULTIPROCESSING:
            with Pool(processes=int(processes)) as pool:
                # execute tasks in order, process results out of order
                for result in pool.imap_unordered(copyparallel_lineLRC, line_arg):
                    if BT_DEBUGGING:
                        print(f'Got result: {result}', flush=True)
                    if result:
                        featuresL.append(result[0])  # resultL
                        featuresR.append(result[1])  # resultR
                        # featuresC.append(result[2])  # resultC
                    step += 1
                    print(f' %{step/total_steps*100} ')

                return gpd.GeoDataFrame(pd.concat(featuresL)), \
                       gpd.GeoDataFrame(pd.concat(featuresR))#,  gpd.GeoDataFrame(pd.concat(featuresC))
        elif PARALLEL_MODE == MODE_SEQUENTIAL:
            for line in line_arg:
                result = copyparallel_lineLRC(line)
                if BT_DEBUGGING:
                    print(f'Got result: {result}', flush=True)
                if result:
                    featuresL.append(result[0])  # resultL
                    featuresR.append(result[1])  # resultR
                step += 1
                print(f' %{step / total_steps * 100} ')

            return gpd.GeoDataFrame(pd.concat(featuresL)), \
                gpd.GeoDataFrame(pd.concat(featuresR))#, gpd.GeoDataFrame(pd.concat(featuresC))

    except OperationCancelledException:
        print("Operation cancelled")


def multiprocessing_Percentile(df, CanPercentile, CanThrPercentage, in_CHM, processes, side):

    try:
        line_arg = []
        total_steps = len(df)
        if side == 'left':
            PerCol = 'Percentile_L'
            cal_percentile=cal_percentileLR
        elif side=='right':
            PerCol = 'Percentile_R'
            # for item in df.index:
            #     item_list = [df.iloc[[item]], CanPercentile, CanThrPercentage, in_CHM, item, PerCol]
            #     line_arg.append(item_list)
            cal_percentile=cal_percentileLR

        elif side=='CL':
            PerCol = 'Percentile_CL'

        elif side == 'CR':
            PerCol = 'Percentile_CR'

        elif side == 'LRing':
            PerCol = 'Percentile_LRing'
            cal_percentile = cal_percentileRing
        elif side == 'RRing':
            PerCol = 'Percentile_RRing'
            cal_percentile = cal_percentileRing

        for item in df.index:
            item_list = [df.iloc[[item]], CanPercentile, CanThrPercentage, in_CHM, item, PerCol]
            line_arg.append(item_list)
        features = []
        # chunksize = math.ceil(total_steps / processes)
        with Pool(processes=int(processes)) as pool:

            step = 0
            # execute tasks in order, process results out of order
            try:
                for result in pool.imap_unordered(cal_percentile, line_arg):
                    if BT_DEBUGGING:
                        print('Got result: {}'.format(result), flush=True)
                    features.append(result)
                    step += 1
                    print('%{}'.format(step / total_steps * 100))
            except Exception:
                print(Exception)
                raise
            del line_arg
            return gpd.GeoDataFrame(pd.concat(features))

    except OperationCancelledException:
        print("Operation cancelled")


def cal_percentileLR(line_arg):
    try:
        df = line_arg[0]
        CanPercentile = line_arg[1]
        CanThrPercentage = line_arg[2]
        in_CHM = line_arg[3]
        row_index = line_arg[4]
        PerCol = line_arg[5]
        line_buffer = df.loc[row_index, 'geometry']

        if line_buffer.is_empty or shapely.is_missing(line_buffer) :
            return None

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
                                                               nodata=BT_NODATA, filled=True)
            clipped_raster = np.squeeze(clipped_raster, axis=0)

            # mask all -9999 (nodata) value cells
            masked_raster = np.ma.masked_where(clipped_raster == BT_NODATA, clipped_raster)
            filled_raster = np.ma.filled(masked_raster, np.nan)

            # Calculate the percentile
            # masked_mean = np.ma.mean(masked_raster)
            percentile = np.nanpercentile(filled_raster, CanPercentile)#,method='hazen')
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

def cal_percentileRing(line_arg):
    try:
        df = line_arg[0]
        CanPercentile = line_arg[1]
        CanThrPercentage = line_arg[2]
        in_CHM = line_arg[3]
        row_index = line_arg[4]
        PerCol = line_arg[5]


        line_buffer = df.loc[row_index, 'geometry']
        if line_buffer.is_empty or shapely.is_missing(line_buffer) :
            return None

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
                                                               nodata=BT_NODATA, filled=True)
            clipped_raster = np.squeeze(clipped_raster, axis=0)

            # mask all -9999 (nodata) value cells
            masked_raster = np.ma.masked_where(clipped_raster == BT_NODATA, clipped_raster)
            filled_raster = np.ma.filled(masked_raster, np.nan)

            # Calculate the percentile
            # masked_mean = np.ma.mean(masked_raster)
            percentile = np.nanpercentile(filled_raster,50)# CanPercentile)#,method='hazen')

            if percentile>1:  # (percentile+median)>0.0:
                # ((50 Percentile + user defined percentile)/2)x(User defined Canopy Threshold Percentage)
                # Dyn_Canopy_Threshold = ((percentile+median)/2.0) * (CanThrPercentage / 100.0)
                # (user defined percentile)x(User defined Canopy Threshold Percentage)
                Dyn_Canopy_Threshold = percentile * (0.3)
            else:
                # print("(percentile)<0.05 @ {}".format(row_index))
                Dyn_Canopy_Threshold=1

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

def copyparallel_lineLRC(line_arg):
    #line_arg = [dfL, dfR,dfc, left_dis, right_dist, center_dist,item]
    dfL = line_arg[0]
    dfR = line_arg[1]
#    dfC= line_arg[2]

    # Simplify input center lines
    geom = dfL.loc[line_arg[6], 'geometry']
    if not geom:
        return None

    lineL = dfL.loc[line_arg[6], 'geometry'].simplify(tolerance=0.05, preserve_topology=True)
    lineR = dfR.loc[line_arg[6], 'geometry'].simplify(tolerance=0.05, preserve_topology=True)
    # lineC = dfC.loc[line_arg[6], 'geometry'].simplify(tolerance=0.05, preserve_topology=True)
    offset_distL = float(line_arg[3])
    offset_distR= float(line_arg[4])


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
        dfL.loc[line_arg[6], 'geometry'] = parallel_lineL
    if not parallel_lineR.is_empty:
        dfR.loc[line_arg[6], 'geometry'] = parallel_lineR
    # if not lineC.is_empty:
    #     dfC.loc[line_arg[6], 'geometry'] = lineC



    return dfL.iloc[[line_arg[6]]], dfR.iloc[[line_arg[6]]] #,dfC.iloc[[line_arg[6]]]


if __name__ == '__main__':
    start_time = time.time()
    print('Starting Dynamic Canopy Threshold calculation processing\n @ {}'.format(
        time.strftime("%d %b %Y %H:%M:%S", time.localtime())))

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=json.loads)
    parser.add_argument('-p', '--processes')
    parser.add_argument('-v', '--verbose')
    args = parser.parse_args()
    args.input['full_step'] = False

    verbose = True if args.verbose == 'True' else False
    main_canopy_threshold_relative(print, **args.input, processes=int(args.processes), verbose=verbose)

    print('%{}'.format(100))
    print('Finishing Dynamic Canopy Threshold calculation @ {}\n(or in {} second)'.format(
        time.strftime("%d %b %Y %H:%M:%S", time.localtime()), round(time.time() - start_time, 5)))
