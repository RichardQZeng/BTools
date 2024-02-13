from multiprocessing.pool import Pool
import geopandas
import json
import argparse
import time
import pandas
import numpy
import shapely


from common import *

corridor_th_field = 'CorridorTh'

class OperationCancelledException(Exception):
    pass


def zonal_threshold(callback, in_line, in_canopy_raster, canopy_search_r, min_canopy_th, max_canopy_th,
                    out_line, processes, verbose):
    line_seg = geopandas.GeoDataFrame.from_file(in_line)
    # check coordinate systems between line and raster features
    with rasterio.open(in_canopy_raster) as in_canopy:
        if line_seg.crs.to_epsg()!= in_canopy.crs.to_epsg():
            print("Line and raster spatial references are not same, please check.")
            exit()

    del in_canopy

    if not corridor_th_field in  line_seg.columns.array:
        print("Cannot find {} column in input data.\n '{}' column will be create".format(corridor_th_field,corridor_th_field))
        line_seg[corridor_th_field]=numpy.nan

    # copy original line input to another Geodataframe
    line_buffer=geopandas.GeoDataFrame.copy((line_seg))
    #buffer the input lines
    buffer=shapely.buffer(line_buffer['geometry'], float(canopy_search_r), cap_style=1, quad_segs=16)
    #replace line geometry by polygon geometry
    line_buffer['geometry']=buffer
    # create a New column for Zonal Mean
    line_buffer['ZonMean'] = numpy.nan
    print('%{}'.format(10))
    print("Line preparing... ")
    #
    line_arg=[]


    for row in line_buffer.index:
        list_items=[row]  #0
        list_items.append(line_buffer.iloc[[row]]) #1
        # list_items.append(line_buffer) #1
        list_items.append(in_canopy_raster) #2
        list_items.append(canopy_search_r) #3
        list_items.append(min_canopy_th) #4
        list_items.append(max_canopy_th) #5
        list_items.append(corridor_th_field)  # 6

        line_arg.append(list_items)
    print('%{}'.format(60))
    print("Calculating zonal statistic ...")
    features=[]
    # for row in range(0,len(line_arg)):
    #     features.append(zonal_prepare(line_arg[row]))
    features=execute_multiprocessing(line_arg)
    print("Merging results ...")
    results = geopandas.GeoDataFrame(pandas.concat(features))
    results['geometry']=line_seg['geometry']

    print("Saving output ...")
    print('%{}'.format(100))
    #Debug save
    geopandas.GeoDataFrame.to_file(results,out_line )
    callback('Zonal Threshold tool done.')


# task executed in a worker process
def zonal_prepare(task_data):
    # report a message
    row_index=task_data[0]
    df = task_data[1]
    in_canopy_raster=task_data[2]
    corridor_th_field=task_data[6]
    MinValue=float(task_data[4])
    MaxValue=float(task_data[5])

    line_buffer = df['geometry']

    with rasterio.open(in_canopy_raster) as in_canopy:
        # clipped the chm base on polygon of line buffer or footprint
        clipped_canopy, out_transform = rasterio.mask.mask(in_canopy, line_buffer, crop=True,
                                                           nodata=BT_NODATA, filled=True)
        clipped_canopy = numpy.squeeze(clipped_canopy, axis=0)

        # mask out all -9999 value cells
        zonal_canopy = numpy.ma.masked_where(clipped_canopy == BT_NODATA, clipped_canopy)

        # Calculate the zonal mean and threshold
        zonal_mean = numpy.ma.mean(zonal_canopy)
        threshold=MinValue + (zonal_mean * zonal_mean) * (MaxValue - MinValue)
        # return the generated value
        df.loc[df.index==row_index,'ZonMean']=zonal_mean
        df.loc[df.index == row_index, corridor_th_field] =threshold

    return df


# protect the entry point
def execute_multiprocessing(line_args):
    # create and configure the process pool
    # data = [[random() for n in range(100)] for i in range(300)]
    try:
        total_steps = len(line_args)
        features = []
        with Pool(processes=int(args.processes)) as pool:
            step = 0
            # execute tasks in order, process results out of order
            for result in pool.imap_unordered(zonal_prepare, line_args):
                if BT_DEBUGGING:
                    print('Got result: {}'.format(result), flush=True)
                features.append(result)
                step += 1
                print('%{}'.format(step / total_steps * 100))
        return features
    except OperationCancelledException:
        print("Operation cancelled")


if __name__ == '__main__':
    start_time = time.time()
    print('Starting zonal threshold calculation processing @ {}'.format(
        time.strftime("%d %b %Y %H:%M:%S", time.localtime())))

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=json.loads)
    parser.add_argument('-p', '--processes')
    parser.add_argument('-v', '--verbose')
    args = parser.parse_args()

    verbose = True if args.verbose == 'True' else False
    zonal_threshold(print, **args.input, processes=int(args.processes), verbose=verbose)


    print('%{}'.format(100))
    print('Finishing zonal threshold calculation processing @ {} (or in {} second)'.format(
        time.strftime("%d %b %Y %H:%M:%S", time.localtime()), round(time.time() - start_time, 5)))
