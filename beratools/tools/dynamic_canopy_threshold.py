import os.path
from multiprocessing.pool import Pool
import geopandas
import json
import argparse
import time
import pandas
import numpy
import rasterio
import shapely
from shapely.ops import split, snap
from common import *
import sys


class OperationCancelledException(Exception):
    pass


def dynamic_canopy_threshold(callback, in_line, in_CHM, Proc_Seg, Off_ln_dist, CanPercentile, CanThrPercentage,
                             Tree_radius, Max_ln_dist, canopy_avoid, exponent, processes, verbose):

    file_path,in_file_name=os.path.split(in_line)
    out_file= os.path.join(file_path,'DynCanTh_'+in_file_name)

    line_seg = geopandas.GeoDataFrame.from_file(in_line)

    # check coordinate systems between line and raster features
    with rasterio.open(in_CHM) as in_raster:
        if line_seg.crs.to_epsg() != in_raster.crs.to_epsg():
            print("Line and raster spatial references are not same, please check.")
            exit()
    del in_raster

    # Check the canopy threshold percent in 0-100 range.  If it is not, 50% will be applied
    if not 100 >= int(CanPercentile) > 0:
        CanPercentile = 50

    # Check the Dynamic Canopy threshold column in data. If it is not, new column will be created
    if not 'DynCanTh' in line_seg.columns.array:
        print("Cannot find {} column in input line data.\n '{}' column will be create".format('DynCanTh', 'DynCanTh'))
        line_seg['DynCanTh'] = numpy.nan

    # Check the OLnFID column in data. If it is not, column will be created
    if not 'OLnFID' in line_seg.columns.array:
        print(
            "Cannot find {} column in input line data.\n '{}' column will be create".format('OLnFID', 'OLnFID'))
        line_seg['OLnFID'] = line_seg.index
    else:
        for row in line_seg.index:
            if row != line_seg.loc[row,'OLnFID']:
                print("Wraning: index and OLnFID are not consistency at index: {}.".format(row))
                print("Please check data")
                exit()
    if Proc_Seg.lower() =='true':
        line_seg=split_into_segments(line_seg)
    else:
        pass


    # copy original line input to another Geodataframe
    workln_dfL = geopandas.GeoDataFrame.copy((line_seg))
    workln_dfR = geopandas.GeoDataFrame.copy((line_seg))


    # copy parallel lines for both side of the input lines
    print("Creating offset area for surrounding forest....")
    workln_dfL, workln_dfR = multiprocessing_copyparallel_lineLR(workln_dfL,workln_dfR, left_dis=float(Off_ln_dist),right_dist=-float(Off_ln_dist))
    workln_dfL.reset_index(drop=True)
    workln_dfR.reset_index(drop=True)
    workln_dfL=workln_dfL.sort_index()
    workln_dfR = workln_dfR.sort_index()
    print('%{}'.format(30))

    worklnbuffer_dfL = geopandas.GeoDataFrame.copy((workln_dfL))
    worklnbuffer_dfR = geopandas.GeoDataFrame.copy((workln_dfR))

    # buffer the parallel line in one side (extend the area into forest)
    worklnbuffer_dfL['geometry'] = shapely.buffer(workln_dfL['geometry'], distance=float(Tree_radius),
                                                  cap_style=2, join_style=2, single_sided=True)
    worklnbuffer_dfR['geometry'] = shapely.buffer(workln_dfR['geometry'], distance=-float(Tree_radius),
                                                  cap_style=2, join_style=2,  single_sided=True)
    print("Creating offset area for surrounding forest....Done")
    print('%{}'.format(50))
    # create a New column for surrounding forest statistics:
    # 1) Height Percentile (add more in the future)
    worklnbuffer_dfL['Percentile_L'] = numpy.nan
    worklnbuffer_dfR['Percentile_R'] = numpy.nan
    worklnbuffer_dfL = worklnbuffer_dfL.reset_index(drop=True)
    worklnbuffer_dfR=worklnbuffer_dfR.reset_index(drop=True)
    line_seg['L_Pertiels'] = numpy.nan
    line_seg['R_Pertiels'] = numpy.nan
    print('%{}'.format(80))

    print("Calculating surrounding forest percentile....")
    # calculate the Height percentile for each parallel area using CHM
    worklnbuffer_dfL = multiprocessing_Percentile(worklnbuffer_dfL, CanPercentile, CanThrPercentage, in_CHM,side='left')

    worklnbuffer_dfR = multiprocessing_Percentile(worklnbuffer_dfR, CanPercentile, CanThrPercentage, in_CHM,
                                                  side='right')

    worklnbuffer_dfL = worklnbuffer_dfL.reset_index(drop=True)
    worklnbuffer_dfR = worklnbuffer_dfR.reset_index(drop=True)

    print('%{}'.format(90))

    for index in (line_seg.index):
        line_seg.loc[index,'L_Pertiels'] =worklnbuffer_dfL.Percentile_L.iloc[index]
        line_seg.loc[index,'R_Pertiels'] = worklnbuffer_dfR.Percentile_R.iloc[index]
        if ((worklnbuffer_dfL.DynCanTh.iloc[index]+worklnbuffer_dfR.DynCanTh.iloc[index])/2)>0.05:
            line_seg.loc[index,'DynCanTh'] = ((worklnbuffer_dfL.DynCanTh.iloc[index]+worklnbuffer_dfR.DynCanTh.iloc[index])/2)
        else:
            line_seg.loc[index, 'DynCanTh'] = 0.05
    print("Saving output.....")

    geopandas.GeoDataFrame.to_file(line_seg,out_file )

    print('%{}'.format(100))
def split_line_by_point(line, point, tolerance: float = 1.0e-12):
    return split(snap(line, point, tolerance), point)

def split_line_fc(line):
    return list(map(shapely.LineString, zip(line.coords[:-1], line.coords[1:])))

def split_into_segments(df):
    odf=df
    crs=odf.crs
    if not 'OLnSEG' in odf.columns.array:
        df['OLnSEG'] = numpy.nan
    else:
        pass
    df=odf.assign(geometry=odf.apply(lambda x: split_line_fc(x.geometry), axis=1))
    df=df.explode()

    df['OLnSEG'] = df.groupby('OLnFID').cumcount()
    gdf=geopandas.GeoDataFrame(df,geometry=df.geometry,crs=crs)
    gdf=gdf.reset_index(drop=True)
    return  gdf
def multiprocessing_copyparallel_lineLR(dfL,dfR, left_dis,right_dist):
    try:

        line_arg = []
        total_steps = len(dfL)
        # df['Offset']=None
        for item in dfL.index:
            # df.loc[item,'Offset']='Left'

            item_list = [dfL,dfR, left_dis,right_dist, item]
            line_arg.append(item_list)
        # for item in df.index:
        #     df.loc[item, 'Offset'] = 'Right'
        #     item_list = [df, right_dist, item]
        #     line_arg.append(item_list)
        featuresL = []
        featuresR = []
        with Pool(processes=int(args.processes)) as pool:
            step = 0
            # execute tasks in order, process results out of order
            for resultL,resultR in pool.imap_unordered(copyparallel_lineLR, line_arg):
                if BT_DEBUGGING:
                    print('Got result: {}{}'.format(resultL,resultR), flush=True)
                featuresL.append(resultL)
                featuresR.append(resultR)
                step += 1
                print('%{}'.format(step / total_steps * 100))
            return geopandas.GeoDataFrame(pandas.concat(featuresL)),geopandas.GeoDataFrame(pandas.concat(featuresR))
    except OperationCancelledException:
        print("Operation cancelled")
# def multiprocessing_copyparallel_line(df, Off_ln_dist):
#     try:
#
#         line_arg = []
#         total_steps = len(df)
#         for item in df.index:
#             item_list = [df, Off_ln_dist, item]
#             line_arg.append(item_list)
#         features = []
#         with Pool(processes=int(args.processes)) as pool:
#             step = 0
#             # execute tasks in order, process results out of order
#             for result in pool.imap_unordered(copyparallel_line, line_arg):
#                 if BT_DEBUGGING:
#                     print('Got result: {}'.format(result), flush=True)
#                 features.append(result)
#                 step += 1
#                 print('%{}'.format(step / total_steps * 100))
#             return geopandas.GeoDataFrame(pandas.concat(features))
#     except OperationCancelledException:
#         print("Operation cancelled")


def multiprocessing_Percentile(df, CanPercentile, CanThrPercentage, in_CHM, side):

    try:
        line_arg = []
        total_steps = len(df)
        if side == 'left':
            PerCol = 'Percentile_L'
        else:
            PerCol = 'Percentile_R'

        for item in df.index:
            item_list = [df.iloc[[item]], int(CanPercentile), float(CanThrPercentage), in_CHM, item, PerCol]
            line_arg.append(item_list)
        features = []
        with Pool(processes=int(args.processes)) as pool:
            step = 0
            # execute tasks in order, process results out of order
            try:
                # features=pool.map(cal_percentile, line_arg)
                for result in pool.imap(cal_percentile, line_arg):
                    if BT_DEBUGGING:
                        print('Got result: {}'.format(result), flush=True)
                    features.append(result)
                    step += 1
                    print('%{}'.format(step / total_steps * 100))
            except Exception:
                print(Exception)
                raise
            return geopandas.GeoDataFrame(pandas.concat(features))


    except OperationCancelledException:
        print("Operation cancelled")

def multiprocessing_PercentileLR(df, CanPercentile, CanThrPercentage, in_CHM):

    try:
        line_arg = []
        total_steps = len(df)
        for item in df.index:
            if df.loc[item,'Offset'].lower() == 'left':
                PerCol = 'Percentile_L'
            else:
                PerCol = 'Percentile_R'

            item_list = [df.iloc[[item]], int(CanPercentile), float(CanThrPercentage), in_CHM, item, PerCol]
            line_arg.append(item_list)

        features = []
        with Pool(processes=int(args.processes)) as pool:
            step = 0
            # execute tasks in order, process results out of order
            try:
                # features=pool.map(cal_percentile, line_arg)
                for result in pool.imap(cal_percentile, line_arg):
                    if BT_DEBUGGING:
                        print('Got result: {}'.format(result), flush=True)
                    features.append(result)
                    step += 1
                    print('%{}'.format(step / total_steps * 100))
            except Exception:
                print(Exception)
                raise
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
    except:
        print("Assigning variable on index:{} Error: ".format(line_arg) + sys.exc_info())
        exit()
    try:
        with rasterio.open(in_CHM) as raster:

            clipped_raster, out_transform = rasterio.mask.mask(raster, [line_buffer], crop=True, nodata=-9999, filled=True)
            clipped_raster = numpy.squeeze(clipped_raster, axis=0)

            # mask all -9999 (nodata) value cells
            masked_raster = numpy.ma.masked_where(clipped_raster == -9999, clipped_raster)
            filled_raster=numpy.ma.filled(masked_raster, numpy.nan)

            # Calculate the percentile
            # masked_mean = numpy.ma.mean(masked_raster)
            percentile = numpy.nanpercentile(filled_raster, CanPercentile)
            median = numpy.nanmedian(filled_raster)
            Dyn_Canopy_Threshold = ((percentile+median)/2) * (CanThrPercentage / 100.0)
        del raster
    # return the generated value
    except:
        print(sys.exc_info())
    try:
        df.loc[row_index,PerCol] = percentile
        df.loc[row_index,'DynCanTh'] = Dyn_Canopy_Threshold
    except:
        print("Writing Percentile and Dynamic Canopy into table Error: "+sys.exc_info())

    return df

def copyparallel_lineLR(line_arg):

    dfL = line_arg[0]
    dfR = line_arg[1]
    line = dfL.loc[line_arg[4], 'geometry'].simplify(tolerance=0.05, preserve_topology=True)

    offset_distL = float(line_arg[2])
    offset_distR= float(line_arg[3])
    parallel_lineL = shapely.offset_curve(geometry=line, distance=offset_distL, join_style=shapely.BufferJoinStyle.mitre)
    parallel_lineR = shapely.offset_curve(geometry=line, distance=offset_distR,
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

    verbose = True if args.verbose == 'True' else False
    dynamic_canopy_threshold(print, **args.input, processes=int(args.processes), verbose=verbose)

    print('%{}'.format(100))
    print('Finishing Dynamic Canopy Threshold calculation @ {}\n(or in {} second)'.format(
        time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), round(time.time() - start_time, 5)))
