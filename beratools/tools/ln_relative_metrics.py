import os.path
from multiprocessing.pool import Pool
from numpy.lib.stride_tricks import as_strided
import geopandas
from xrspatial import convolution
import warnings
# to suppress panadas UserWarning: SettingWithCopyWarning:
# A value is trying to be set on a copy of a slice from a DataFrame.
# Try using .loc[row_indexer,col_indexer] = value instead
warnings.simplefilter(action='ignore', category=UserWarning)
import json
import argparse
import time
import pandas
import numpy
import shapely
from common import *
import sys
import math

USE_MULTI_PROCESSING=True
class OperationCancelledException(Exception):
    pass

# by  Dan Patterson

def _check(a, r_c, subok=False):
    """Performs the array checks necessary for stride and block.
    : in_array   - Array or list.
    : r_c - tuple/list/array of rows x cols.
    : subok - from numpy 1.12 added, keep for now
    :Returns:
    :------
    :Attempts will be made to ...
    :  produce a shape at least (1*c).  For a scalar, the
    :  minimum shape will be (1*r) for 1D array or (1*c) for 2D
    :  array if r<c.  Be aware
    """
    if isinstance(r_c, (int, float)):
        r_c = (1, int(r_c))
    r, c = r_c
    if a.ndim == 1:
        a = numpy.atleast_2d(a)
    r, c = r_c = (min(r, a.shape[0]), min(c, a.shape[1]))
    a = numpy.array(a, copy=False, subok=subok)
    return a, r, c, tuple(r_c)


def _pad(in_array, kernel):
    """Pad a sliding array to allow for stats"""
    pad_x = int(kernel.shape[0] / 2)
    pad_y = int(kernel.shape[0] / 2)
    result = numpy.pad(in_array, pad_width=(pad_x, pad_y), mode="constant", constant_values=(numpy.NaN, numpy.NaN))

    return result


def stride(a, r_c):
    """Provide a 2D sliding/moving view of an array.
    :  There is no edge correction for outputs.
    :
    :Requires:
    :--------
    : _check(a, r_c) ... Runs the checks on the inputs.
    : a - array or list, usually a 2D array.  Assumes rows is >=1,
    :     it is corrected as is the number of columns.
    : r_c - tuple/list/array of rows x cols.  Attempts  to
    :     produce a shape at least (1*c).  For a scalar, the
    :     minimum shape will be (1*r) for 1D array or 2D
    :     array if r<c.  Be aware
    """

    a, r, c, r_c = _check(a, r_c)
    shape = (a.shape[0] - r + 1, a.shape[1] - c + 1) + r_c
    strides = a.strides * 2
    a_s = (as_strided(a, shape=shape, strides=strides)).squeeze()
    return a_s

def cal_sar(a_s,cell_x,cell_y,diag):
    # Jenness, J. S. 2004.  Calculating landscape surface area from digital elevation models.
    # Wildlife Society Bulletin. 32(3):829-839
    # For SAR
    # kernel index				cell values (example)
    # 0,0	0,1	 0,2		4.326477	9.00671	   10.430054
    # 1,0	1,1	 1,2		7.472778	7.408875	4.323486
    # 2,0	2,1	 2,2		8.534485	8.106201	7.350098
    #
    # Direction
    # 8	  1	   2
    # 7	Center 3
    # 6   5	   4
    #
    center = a_s[1, 1]
    # Pythagorean Theorem
    # 8 Directions
    if not numpy.isnan(center):
        dir1 = math.sqrt(abs(center - a_s[0, 1]) ** 2 + (cell_y ** 2))
        dir2 = math.sqrt(abs(center - a_s[0, 2]) ** 2 + diag ** 2)
        dir3 = math.sqrt(abs(center - a_s[1, 2]) ** 2 + cell_x ** 2)
        dir4 = math.sqrt(abs(center - a_s[2, 2]) ** 2 + diag ** 2)
        dir5 = math.sqrt(abs(center - a_s[2, 1]) ** 2 + cell_y ** 2)
        dir6 = math.sqrt(abs(center - a_s[2, 0]) ** 2 + diag ** 2)
        dir7 = math.sqrt(abs(center - a_s[1, 0]) ** 2 + cell_x ** 2)
        dir8 = math.sqrt(abs(center - a_s[0, 0]) ** 2 + diag ** 2)
        # 8 Outer sides
        dir1_2 = math.sqrt(abs(a_s[0, 1] - a_s[0, 2]) ** 2 + cell_x ** 2)
        dir2_3 = math.sqrt(abs(a_s[0, 2] - a_s[1, 2]) ** 2 + cell_y ** 2)
        dir3_4 = math.sqrt(abs(a_s[1, 2] - a_s[2, 2]) ** 2 + cell_y ** 2)
        dir4_5 = math.sqrt(abs(a_s[2, 2] - a_s[2, 1]) ** 2 + cell_x ** 2)
        dir5_6 = math.sqrt(abs(a_s[2, 1] - a_s[2, 0]) ** 2 + cell_x ** 2)
        dir6_7 = math.sqrt(abs(a_s[2, 0] - a_s[1, 0]) ** 2 + cell_y ** 2)
        dir7_8 = math.sqrt(abs(a_s[1, 0] - a_s[0, 0]) ** 2 + cell_y ** 2)
        dir8_1 = math.sqrt(abs(a_s[0, 0] - a_s[0, 1]) ** 2 + cell_x ** 2)

        # Heron of Alexandria and Archimedes (see also Abramowitz and Stegun [1972, p. 79]):
        p1 = (dir1 + dir2 + dir1_2) / 2
        area1 = math.sqrt(p1 * (p1 - dir1) * (p1 - dir2) * (p1 - dir1_2))
        p2 = (dir2 + dir3 + dir2_3) / 2
        area2 = math.sqrt(p2 * (p2 - dir2) * (p2 - dir3) * (p2 - dir2_3))
        p3 = (dir3 + dir4 + dir3_4) / 2
        area3 = math.sqrt(p3 * (p3 - dir3) * (p3 - dir4) * (p3 - dir3_4))
        p4 = (dir4 + dir5 + dir4_5) / 2
        area4 = math.sqrt(p4 * (p4 - dir4) * (p4 - dir5) * (p4 - dir4_5))
        p5 = (dir5 + dir6 + dir5_6) / 2
        area5 = math.sqrt(p5 * (p5 - dir5) * (p5 - dir6) * (p5 - dir5_6))
        p6 = (dir6 + dir7 + dir6_7) / 2
        area6 = math.sqrt(p6 * (p6 - dir6) * (p6 - dir7) * (p6 - dir6_7))
        p7 = (dir7 + dir8 + dir7_8) / 2
        area7 = math.sqrt(p7 * (p7 - dir7) * (p7 - dir8) * (p7 - dir7_8))
        p8 = (dir8 + dir1 + dir8_1) / 2
        area8 = math.sqrt(p8 * (p8 - dir8) * (p8 - dir1) * (p8 - dir8_1))
        areas = (list([area1, area2, area3, area4, area5, area6, area7, area8]))
        surface_area = 0
        for area in areas:
            if not math.isnan(area):
                surface_area = surface_area + area
        return surface_area
    else:
        surface_area = math.nan
        return surface_area

def cal_tri(a_s):
    # For TRI
    # refer https://livingatlas-dcdev.opendata.arcgis.com/content/28360713391948af9303c0aeabb45afd/about
    # for example: TRI with your elevation data.The results are interpreted as follows:
    # 0-80m is considered to represent a level terrain surface
    # 81-116m represents a nearly level surface
    # 117-161m represents a slightly rugged surface
    # 162-239m represents an intermediately rugged surface
    # 240-497m represents a moderately rugged surface
    # 498-958m represents a highly rugged surface
    # 959-4367m represents an extremely rugged surface
    if not numpy.isnan(a_s[1,1]):
        result=math.sqrt(abs((numpy.nanmax(a_s))**2-(numpy.nanmin(a_s))**2))
    else:
        result=math.nan
    return result

def cal_index(in_ndarray, cell_x,cell_y,type):
    kernel=numpy.arange(3**2)
    kernel.shape=(3,3)
    kernel.fill(1)
    padded_array = _pad(in_ndarray, kernel)
    a_s = stride(padded_array, kernel.shape)
    rows,cols=a_s.shape[0],a_s.shape[1]
    result=numpy.arange(rows*cols)
    result.shape=(rows,cols)
    result=result.astype('float64')
    result.fill(numpy.nan)
    diag=math.sqrt(cell_x**2+cell_y**2)
    plannar_area = (cell_y * cell_x) / 2  # area of one cell

    if type == 'SAR':
        for y in range(rows):
            for x in range(cols):
                result[y,x]=cal_sar(a_s[y,x],cell_x,cell_y,diag)
        total_surface_area = numpy.nansum(result)
        with numpy.errstate(divide='ignore', invalid='ignore'):
            result_ratio= numpy.true_divide(plannar_area, result)
            # result_ratio[result_ratio == np.inf] = 0
            # result_ratio = np.nan_to_num(result_ratio)

        # result_ratio = np.divide(plannar_area,result, out=np.zeros_like(result), where=result!=0)
        return result_ratio, total_surface_area
    elif type=='TRI':
        for y in range(rows):
            for x in range(cols):
                result[y, x] = cal_tri(a_s[y, x])
        return result

def forest_metrics(callback, in_line, out_line, raster_type, in_raster, proc_segments, buffer_ln_dist,
                   cl_metrics_gap, forest_buffer_dist, processes, verbose, worklnbuffer_dfLR=None):

    # file_path,in_file_name=os.path.split(in_line)
    output_path,output_file=os.path.split(out_line)
    output_filename,file_ext=os.path.splitext((output_file))
    if not os.path.exists(output_path):
        print("No output file path found, pls check.")
        exit()
    else:
        if file_ext.lower()!=".shp":
            print("Output file type should be shapefile, pls check.")
            exit()

    try:
        line_seg = geopandas.GeoDataFrame.from_file(in_line)
    except:
        print(sys.exc_info())
        exit()

    # check coordinate systems between line and raster features
    with rasterio.open(in_raster) as in_image:
        if line_seg.crs.to_epsg() != in_image.crs.to_epsg():
            print("Line and raster spatial references are not same, please check.")
            exit()

    # # Check the ** column in data. If it is not, new column will be created
    # if not '**' in line_seg.columns.array:
    #     print("Cannot find {} column in input line data.\n '{}' column will be create".format("**","**"))
    #     line_seg['**'] = numpy.nan

    # Check the OLnFID column in data. If it is not, column will be created
    if not 'OLnFID' in line_seg.columns.array:
        print(
            "Cannot find {} column in input line data.\n '{}' column will be create".format('OLnFID', 'OLnFID'))
        line_seg['OLnFID'] = line_seg.index

    if proc_segments== True:
        line_seg=split_into_Equal_Nth_segments(line_seg)

    else:
        pass

    # copy original line input to another Geodataframe
    workln_dfC = geopandas.GeoDataFrame.copy((line_seg))
    workln_dfL = geopandas.GeoDataFrame.copy((line_seg))
    workln_dfR = geopandas.GeoDataFrame.copy((line_seg))

    # copy parallel lines for both side of the input lines
    print("Creating area for CL....")
    workln_dfC['geometry'] = workln_dfC['geometry'].simplify(tolerance=0.05, preserve_topology=True)
    worklnbuffer_dfC = geopandas.GeoDataFrame.copy((workln_dfC))
    worklnbuffer_dfC['geometry'] = shapely.buffer(workln_dfC['geometry'], distance=float(buffer_ln_dist),
                                                  cap_style=2, join_style=2, single_sided=False)



    print("Creating offset area for surrounding forest....")
    workln_dfL, workln_dfR = multiprocessing_copyparallel_lineLR(workln_dfL, workln_dfR, processes, left_dis=float(buffer_ln_dist + cl_metrics_gap), right_dist=-float(buffer_ln_dist + cl_metrics_gap))
    workln_dfR=workln_dfR.sort_values(by=['OLnFID'])
    workln_dfL=workln_dfL.sort_values(by=['OLnFID'])
    workln_dfL=workln_dfL.reset_index(drop=True)
    workln_dfR=workln_dfR.reset_index(drop=True)

    print('%{}'.format(30))

    worklnbuffer_dfL = geopandas.GeoDataFrame.copy((workln_dfL))
    worklnbuffer_dfR = geopandas.GeoDataFrame.copy((workln_dfR))

    # buffer the parallel line in one side (extend the area into forest)

    worklnbuffer_dfL['geometry'] = shapely.buffer(workln_dfL['geometry'], distance=float(forest_buffer_dist),
                                                  cap_style=2, join_style=2, single_sided=True)
    worklnbuffer_dfR['geometry'] = shapely.buffer(workln_dfR['geometry'], distance=-float(forest_buffer_dist),
                                                  cap_style=2, join_style=2, single_sided=True)
    print("Creating offset area for surrounding forest....Done")
    print('%{}'.format(50))
    # create a New column for surrounding forest statistics:
    # 1) Height Percentile (add more in the future)
    # worklnbuffer_dfL['**_L'] = numpy.nan
    # worklnbuffer_dfR['**_R'] = numpy.nan
    # worklnbuffer_dfL = worklnbuffer_dfL.reset_index(drop=True)
    # worklnbuffer_dfR=worklnbuffer_dfR.reset_index(drop=True)
    # line_seg['L_**'] = numpy.nan
    # line_seg['R_**'] = numpy.nan
    print('%{}'.format(80))

    print("Calculating CL metrics ..")
    worklnbuffer_dfC, new_col_c = multiprocessing_metrics(worklnbuffer_dfC, in_raster, raster_type, processes,
                                                          side='center')

    worklnbuffer_dfC = worklnbuffer_dfC.sort_values(by=['OLnFID'])
    worklnbuffer_dfC = worklnbuffer_dfC.reset_index(drop=True)
    print("Calculating surrounding forest metrics....")
    # calculate the Height percentile for each parallel area using CHM
    worklnbuffer_dfL,new_col_l = multiprocessing_metrics(worklnbuffer_dfL, in_raster, raster_type, processes, side='left')

    worklnbuffer_dfL = worklnbuffer_dfL.sort_values(by=['OLnFID'])
    worklnbuffer_dfL = worklnbuffer_dfL.reset_index(drop=True)

    worklnbuffer_dfR,new_col_r = multiprocessing_metrics(worklnbuffer_dfR, in_raster, raster_type, processes, side='right')

    worklnbuffer_dfR = worklnbuffer_dfR.sort_values(by=['OLnFID'])
    worklnbuffer_dfR = worklnbuffer_dfR.reset_index(drop=True)

    print('%{}'.format(90))

    all_new_col= numpy.append(numpy.array(new_col_c),numpy.array(new_col_l))
    all_new_col=numpy.append(all_new_col,numpy.array(new_col_r))

    for index in (line_seg.index):
        if  raster_type=='DEM':
            for col in all_new_col:
                if "C_" in col:
                    line_seg.loc[index, col] = worklnbuffer_dfC.loc[index, col]
                elif "L_" in col:
                    line_seg.loc[index,col] =worklnbuffer_dfL.loc[index,col]
                elif "R_" in col:
                    line_seg.loc[index, col] = worklnbuffer_dfR.loc[index, col]
    print("Calculating forest metrics....Done")



    print("Saving forest metrics output.....")
    geopandas.GeoDataFrame.to_file(line_seg, out_line)
    print("Saving forest metrics output.....Done")
    del line_seg, worklnbuffer_dfL, worklnbuffer_dfR, workln_dfL, workln_dfR

    print('%{}'.format(100))

def split_line_nPart(line):
    from shapely.ops import split,snap
    # Work out n parts for each line (divided by 10m)
    n=math.ceil(line.length/10)
    if n>1:
        # divided line into n-1 equal parts;
        distances=numpy.linspace(0,line.length,n)
        points=[line.interpolate(distance) for distance in distances]

        split_points=shapely.multipoints(points)
        # mline=cut_line_at_points(line,points)
        mline = split(line, split_points)
        # mline=split_line_fc(line)
    else:
        mline=line
    return mline
def split_into_Equal_Nth_segments(df):
    odf=df
    crs=odf.crs
    if not 'OLnSEG' in odf.columns.array:
        df['OLnSEG'] = numpy.nan
    df=odf.assign(geometry=odf.apply(lambda x: split_line_nPart(x.geometry), axis=1))
    df=df.explode()

    df['OLnSEG'] = df.groupby('OLnFID').cumcount()
    gdf=geopandas.GeoDataFrame(df,geometry=df.geometry,crs=crs)
    gdf = gdf.sort_values(by=['OLnFID', 'OLnSEG'])
    gdf=gdf.reset_index(drop=True)
    return  gdf

def split_line_fc(line):
    return list(map(shapely.LineString, zip(line.coords[:-1], line.coords[1:])))

def split_into_every_segments(df):
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
    gdf=gdf.sort_values(by=['OLnFID','OLnSEG'])
    gdf=gdf.reset_index(drop=True)
    return  gdf
def multiprocessing_copyparallel_lineLR(dfL,dfR,processes, left_dis,right_dist):
    try:

        line_arg = []
        total_steps = len(dfL)

        for item in dfL.index:
            item_list = [dfL,dfR, left_dis,right_dist, item]
            line_arg.append(item_list)

        featuresL = []
        featuresR = []
        chunksize = math.ceil(total_steps / processes)
        with Pool(processes=int(processes)) as pool:
            step = 0
            # execute tasks in order, process results out of order
            for resultL,resultR in pool.imap_unordered(copyparallel_lineLR, line_arg,chunksize=chunksize):
                if BT_DEBUGGING:
                    print('Got result: {}{}'.format(resultL,resultR), flush=True)
                featuresL.append(resultL)
                featuresR.append(resultR)
                step += 1
                print('%{}'.format(step / total_steps * 100))
            return geopandas.GeoDataFrame(pandas.concat(featuresL)),geopandas.GeoDataFrame(pandas.concat(featuresR))
    except OperationCancelledException:
        print("Operation cancelled")

def multiprocessing_metrics(df, in_raster, raster_type, processes, side):

    try:
        line_arg = []
        total_steps = len(df)
        if side == 'left':
            PerCol = 'L'
        elif side=='right':
            PerCol = 'R'
        else:
            PerCol = 'C'

        for item in df.index:
            item_list = [df.iloc[[item]], in_raster, item, PerCol,raster_type]
            line_arg.append(item_list)
        features = []
        chunksize = math.ceil(total_steps / processes)

        if USE_MULTI_PROCESSING:
            with Pool(processes=int(processes)) as pool:

                step = 0
                # execute tasks in order, process results out of order
                try:
                    for result in pool.imap_unordered(cal_metrics, line_arg, chunksize=chunksize):
                        if BT_DEBUGGING:
                            print('Got result: {}'.format(result), flush=True)
                        features.append(result)
                        step += 1
                        print('%{}'.format(step / total_steps * 100))
                except Exception:
                    print(Exception)
                    raise
                del line_arg
                df=geopandas.GeoDataFrame(pandas.concat(features))
                new_col = []
                for col in df.columns.array:
                    if "C_" in col:
                        new_col.append(col)
                    elif "L_" in col:
                        new_col.append(col)
                    elif "R_" in col:
                        new_col.append(col)

                return df,new_col
        else:
            for row in line_arg:
                features.append(cal_metrics(row))
            df = geopandas.GeoDataFrame(pandas.concat(features))
            new_col = []
            for col in df.columns.array:
                if "C_" in col:
                    new_col.append(col)
                elif "L_" in col:
                    new_col.append(col)
                elif "R_" in col:
                    new_col.append(col)

            return df, new_col

    except OperationCancelledException:
        print("Operation cancelled")

def cal_metrics(line_arg):
    try:
        df = line_arg[0]
        raster = line_arg[1]
        row_index = line_arg[2]
        PerCol = line_arg[3]
        raster_type=line_arg[4]
        line_buffer = df.loc[row_index, 'geometry']
    except:
        print("Assigning variable on index:{} Error: ".format(line_arg) + sys.exc_info())
        exit()
    try:
        with rasterio.open(raster) as image:

            clipped_raster, out_transform = rasterio.mask.mask(image, [line_buffer], crop=True, nodata=-9999, filled=True)
            clipped_raster = numpy.squeeze(clipped_raster, axis=0)
            cell_x, cell_y = image.res
            cell_area=cell_x*cell_y
            # mask all -9999 (nodata) value cells
            masked_raster = numpy.ma.masked_where(clipped_raster == -9999, clipped_raster)
            filled_raster=numpy.ma.filled(masked_raster, numpy.nan)

            # Calculate the metrics
            if raster_type=="DEM":
                # Surface area ratio (SAR)
                SAR,total_surface_area=cal_index(filled_raster,cell_x,cell_y,'SAR')
                SAR_mean = numpy.nanmean(SAR)
                SAR_percentile90 = numpy.nanpercentile(SAR, 90, method='hazen')
                SAR_median = numpy.nanmedian(SAR)
                SAR_std = numpy.nanstd(SAR)
                SAR_max = numpy.nanmax(SAR)
                SAR_min = numpy.nanmin(SAR)

                # Terrain Ruggedness Index (TRI)
                # TRI = cal_index(filled_raster, cell_x, cell_y, 'TRI')
                # TRI_mean = numpy.nanmean(TRI)
                # TRI_percentile90 = numpy.nanpercentile(TRI, 90, method='hazen')
                # TRI_median = numpy.nanmedian(TRI)
                # TRI_std = numpy.nanstd(TRI)
                # TRI_max = numpy.nanmax(TRI)
                # TRI_min = numpy.nanmin(TRI)

                # General Statistics
                total_planar_area= numpy.ma.count(masked_raster) * cell_area
                cell_volume=masked_raster*cell_area
                total_volume=numpy.ma.sum(cell_volume)
                mean = numpy.nanmean(filled_raster)
                percentile90 = numpy.nanpercentile(filled_raster, 90,method='hazen')
                median = numpy.nanmedian(filled_raster)
                std=numpy.nanstd(filled_raster)
                max=numpy.nanmax(filled_raster)
                min=numpy.nanmin(filled_raster)

            del clipped_raster, out_transform

    # return the generated value
    except:
        print(sys.exc_info())
    try:
        # Writing SAR statisic
        df.at[row_index, PerCol + '_SurArea'] = total_surface_area
        df.at[row_index, PerCol + '_SARmean'] = SAR_mean
        # df.at[row_index, PerCol + '_SARP90'] = SAR_percentile90
        # df.at[row_index, PerCol + '_SARmed'] = SAR_median
        df.at[row_index, PerCol + '_SARStd'] = SAR_std
        df.at[row_index, PerCol + '_SARmax'] = SAR_max
        df.at[row_index, PerCol + '_SARmin'] = SAR_min

        # Writing TRI statisic
        # df.at[row_index, PerCol + '_TRImean'] = TRI_mean
        # # df.at[row_index, PerCol + '_TRIP90'] = TRI_percentile90
        # # df.at[row_index, PerCol + '_TRImed'] = TRI_median
        # df.at[row_index, PerCol + '_TRIStd'] = TRI_std
        # df.at[row_index, PerCol + '_TRImax'] = TRI_max
        # df.at[row_index, PerCol + '_TRImin'] = TRI_min

        # Writing General statisic
        df.at[row_index, PerCol + '_PlArea'] = total_planar_area
        df.at[row_index, PerCol + '_mean'] = mean
        # df.at[row_index, PerCol + '_P90'] = percentile90
        # df.at[row_index, PerCol + '_median'] = median
        df.at[row_index, PerCol + '_StdDev'] = std
        df.at[row_index, PerCol + '_max'] = max
        df.at[row_index, PerCol + '_min'] = min
        # df.at[row_index, PerCol + '_Vol'] = total_volume

        return df
    except:
        print("Error writing forest metrics into table: "+sys.exc_info())


def copyparallel_lineLR(line_arg):

    dfL = line_arg[0]
    dfR = line_arg[1]
    #Simplify input center lines
    lineL = dfL.loc[line_arg[4], 'geometry'].simplify(tolerance=0.05, preserve_topology=True)
    lineR = dfL.loc[line_arg[4], 'geometry'].simplify(tolerance=0.05, preserve_topology=True)
    offset_distL = float(line_arg[2])
    offset_distR= float(line_arg[3])

    # Older alternative method to the offset_curve() method,
    # but uses resolution instead of quad_segs and a side keyword (‘left’ or ‘right’) instead
    # of sign of the distance. This method is kept for backwards compatibility for now,
    # but it is recommended to use offset_curve() instead.
    # (ref: https://shapely.readthedocs.io/en/stable/manual.html#object.offset_curve)

    # parallel_lineL = shapely.offset_curve(geometry=lineL, distance=offset_distL,
    #                                       join_style=shapely.BufferJoinStyle.mitre)
    # parallel_lineR = shapely.offset_curve(geometry=lineR, distance=offset_distR,
    #                                       join_style=shapely.BufferJoinStyle.mitre)

    parallel_lineL = lineL.parallel_offset(distance=offset_distL,side='left',
                                          join_style=shapely.BufferJoinStyle.mitre)
    parallel_lineR = lineR.parallel_offset(distance=-offset_distR,side='right',
                                         join_style=shapely.BufferJoinStyle.mitre)

    if not parallel_lineL.is_empty:
        dfL.loc[line_arg[4], 'geometry'] = parallel_lineL
    if not parallel_lineR.is_empty:
        dfR.loc[line_arg[4], 'geometry'] = parallel_lineR
    return dfL.iloc[[line_arg[4]]],dfR.iloc[[line_arg[4]]]


if __name__ == '__main__':
    start_time = time.time()
    print('Starting forest metrics calculation processing\n @ {}'.format(
        time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    print("Checking input parameters....")
    in_args, in_verbose = check_arguments()

    verbose = True if in_args.verbose == 'True' else False
    forest_metrics(print, **in_args.input, processes=int(in_args.processes), verbose=verbose)

    print('%{}'.format(100))
    print('Finishing Dynamic Canopy Threshold calculation @ {}\n(or in {} second)'.format(
        time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), round(time.time() - start_time, 5)))
