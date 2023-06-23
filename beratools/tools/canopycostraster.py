import sys
import numpy
import rasterio
import xarray as xr
import xrspatial.focal
from xrspatial import convolution, allocation
from numpy.lib.stride_tricks import as_strided
import os, json
import time
import argparse
import warnings
from scipy import ndimage

from common import *


# TODO: Rolling Statistics for grid data... an alternative
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


def normalize_chm(raster):
    n_raster = numpy.where(raster >= 0, raster, 0)
    return n_raster


def np_cc_map(out_canopy_r, chm, in_array, min_ht):
    print('Generating Canopy Closure Raster.......')

    # canopy_ndarray = numpy.where(in_array >= min_ht, 1., 0.).astype(float)
    canopy_ndarray=numpy.ma.where(in_array>min_ht,1.,0.).astype(float)
    canopy_ndarray = numpy.ma.filled(canopy_ndarray, chm.nodata)
    try:
        write_canopy = rasterio.open(out_canopy_r, 'w', **chm.profile)
        write_canopy.write(canopy_ndarray, 1)
        write_canopy.close()
        print('Generating Canopy Closure (CC) Raster.......Done')
    except Exception as e:
        print(sys.exc_info())
    del in_array
    return canopy_ndarray


def fs_raster(in_ndarray, kernel):
    print('Generating Canopy Closure Focal Statistic .......')
    padded_array = _pad(in_ndarray, kernel)
    a_s = stride(padded_array, kernel.shape)

    # TODO: numpy.where on large ndarray fail (allocate memory error)
    a_s_masked = numpy.where(kernel == 1, a_s, numpy.NaN)
    print("Calculating Canopy Closure's Focal Statistic-Mean .......")
    mean_result = numpy.nanmean(a_s_masked, axis=(2, 3))
    print("Calculating Canopy Closure's Focal Statistic-Stand Deviation Raster .......")
    stdev_result = numpy.nanstd(a_s_masked, axis=(2, 3))
    del a_s
    return mean_result, stdev_result


def fs_raster_stdmean(in_ndarray, kernel,nodata ):
    # This function uses xrspatial whcih can handle large data but slow
    in_ndarray[in_ndarray==nodata] = numpy.nan
    result_ndarray= xrspatial.focal._focal_stats_cpu(xr.DataArray(in_ndarray), kernel, stats_funcs=['std', 'mean'])

    # Flattening the array
    flatten_std_result_ndarray = result_ndarray[0].data.reshape(-1)
    flatten_mean_result_ndarray = result_ndarray[1].data.reshape(-1)

    # Re-shaping the array
    reshape_std_ndarray = flatten_std_result_ndarray.reshape(in_ndarray.shape[0], in_ndarray.shape[1])
    reshape_mean_ndarray = flatten_mean_result_ndarray.reshape(in_ndarray.shape[0], in_ndarray.shape[1])
    return reshape_std_ndarray, reshape_mean_ndarray


def smooth_cost(in_raster, search_dist,sampling):
    print('Generating Cost Raster .......')
    from tempfile import mkdtemp
    import os.path as path
    import shutil

    euc_dist_array = None
    row, col = in_raster.shape
    if row * col >= 30000 * 30000:
        filename = path.join(mkdtemp(), 'tempmmemap.dat')
        a_in_mat = numpy.memmap(filename, in_raster.dtype, 'w+', shape=in_raster.shape)
        a_in_mat[:] = in_raster[:]
        a_in_mat.flush()
        euc_dist_array = ndimage.distance_transform_edt(numpy.logical_not(a_in_mat),sampling=sampling)
        del a_in_mat, in_raster
        shutil.rmtree(path.dirname(filename))
    else:
        euc_dist_array = ndimage.distance_transform_edt(numpy.logical_not(in_raster), sampling=sampling)

    smooth1 = float(search_dist) - euc_dist_array
    # cond_smooth1 = numpy.where(smooth1 > 0, smooth1, 0.0)
    smooth1[smooth1 <= 0.0] = 0.0
    smooth_cost_array = smooth1 / float(search_dist)

    return smooth_cost_array


def np_cost_raster(canopy_ndarray, cc_mean, cc_std, cc_smooth, chm, avoidance, cost_raster_exponent, out_cost_r):
    print('Generating Smoothed Cost Raster.......')
    aM1a = (cc_mean - cc_std)
    aM1b = (cc_mean + cc_std)
    aM1 = numpy.divide(aM1a, aM1b, where=aM1b != 0, out=numpy.zeros(aM1a.shape, dtype=float))
    aM = (1 + aM1) / 2
    aaM = (cc_mean + cc_std)
    bM = numpy.where(aaM <= 0, 0, aM)
    cM = bM * (1 - avoidance) + (cc_smooth * avoidance)
    dM = numpy.where(canopy_ndarray == 1, 1, cM)
    eM = numpy.exp(dM)
    result = numpy.power(eM, float(cost_raster_exponent))
    write_cost = rasterio.open(out_cost_r, 'w+', driver='GTiff', height=chm.shape[0], width=chm.shape[1],
                               count=1, dtype=chm.read(1).dtype, crs=chm.crs, transform=chm.transform)
    write_cost.write(result, 1)
    write_cost.close()
    print('Generating Smoothed Cost Raster.......Done')
    return


# TODO: deal with NODATA
def canopy_cost_raster(callback, in_chm, canopy_ht_threshold, tree_radius, max_line_dist,
                       canopy_avoidance, exponent, out_canopy, out_cost, processes, verbose):

    canopy_ht_threshold = float(canopy_ht_threshold)
    tree_radius = float(tree_radius)
    max_line_dist = float(max_line_dist)
    canopy_avoidance = float(canopy_avoidance)
    cost_raster_exponent = float(exponent)
    # out_canopy_r = args.get('out_canopy_r')
    # out_cost_r = args.get('out_cost_r')

    print('In CHM: ' + in_chm)
    chm = rasterio.open(in_chm)

    # chm_info = chm.transform
    # chm_crs = chm.crs
    # nodata=chm.nodata
    (cell_x, cell_y) = chm.res
    # row, col = chm.shape

    print('Loading CHM............')
    band1_ndarray = chm.read(1, masked=True)
    print('%{}'.format(10))

    print('Preparing Kernel window............')
    kernel = convolution.circle_kernel(cell_x, cell_y, tree_radius)
    print('%{}'.format(20))
    # Generate Canopy Raster and return the Canopy array
    canopy_ndarray = np_cc_map(out_canopy, chm, band1_ndarray, canopy_ht_threshold)

    print('%{}'.format(40))

    print('Apply focal statistic on raster.....')
    # Calculating focal statistic from canopy raster
    #
    # Alternative: (only work on large cell size
    if cell_y >1 and cell_x > 1:
        cc_mean, cc_std = fs_raster(canopy_ndarray, kernel)
    else:
        cc_std, cc_mean = fs_raster_stdmean(canopy_ndarray, kernel, chm.nodata)
    print('%{}'.format(60))
    print('Apply focal statistic on raster.....Done')

    # Smoothing raster
    cc_smooth = smooth_cost(canopy_ndarray, max_line_dist, [cell_x, cell_y])
    avoidance = max(min(float(canopy_avoidance), 1), 0)
    np_cost_raster(canopy_ndarray, cc_mean, cc_std, cc_smooth, chm, avoidance, cost_raster_exponent, out_cost)
    print('%{}'.format(100))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=json.loads)
    parser.add_argument('-p', '--processes')
    parser.add_argument('-v', '--verbose')
    args = parser.parse_args()

    verbose = True if args.verbose == 'True' else False

    start_time = time.time()
    print('Starting Canopy and Cost Raster processing\n @ {}'.format(
        time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    canopy_cost_raster(print, **args.input, processes=int(args.processes), verbose=verbose)
    print('Finishing Dynamic Canopy Threshold calculation in {} seconds)'.format(round(time.time() - start_time, 5)))


