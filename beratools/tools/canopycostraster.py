import numpy as np
import rasterio
import xarray as xr
import xrspatial.focal
from xrspatial import convolution
from numpy.lib.stride_tricks import as_strided

import time
import argparse
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
        a = np.atleast_2d(a)
    r, c = r_c = (min(r, a.shape[0]), min(c, a.shape[1]))
    a = np.array(a, copy=False, subok=subok)
    return a, r, c, tuple(r_c)


def _pad(in_array, kernel):
    """Pad a sliding array to allow for stats"""
    pad_x = int(kernel.shape[0] / 2)
    pad_y = int(kernel.shape[0] / 2)
    result = np.pad(in_array, pad_width=(pad_x, pad_y), mode="constant", constant_values=(np.NaN, np.NaN))

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
    n_raster = np.where(raster >= 0, raster, 0)
    return n_raster


def np_cc_map(out_canopy_r, chm, in_array, min_ht):
    print('Generating Canopy Closure Raster.......')

    canopy_ndarray = np.where(in_array >= min_ht, 1., 0.).astype(float)

    write_canopy = rasterio.open(out_canopy_r, 'w', driver='GTiff', height=chm.shape[0], width=chm.shape[1], count=1,
                                 dtype=chm.read(1).dtype, crs=chm.crs, transform=chm.transform)
    write_canopy.write(canopy_ndarray, 1)
    write_canopy.close()
    print('Generating Canopy Closure (CC) Raster.......Done')

    return canopy_ndarray
    # print(band1.dtype, band1.shape, band1.size, band1.data)
    # print(cc_raster.dtype, cc_raster.shape, cc_raster.size, cc_raster.data)


def fs_raster(in_ndarray, kernel):
    print('Generating Canopy Closure Focal Statistic .......')
    padded_array = _pad(in_ndarray, kernel)
    a_s = stride(padded_array, kernel.shape)

    # TODO: np.where on large ndarray fail (allocate memory error)
    a_s_masked = np.where(kernel == 1, a_s, np.NaN)
    print("Calculating Canopy Closure's Focal Statistic-Mean .......")
    mean_result = np.nanmean(a_s_masked, axis=(2, 3))
    print("Calculating Canopy Closure's Focal Statistic-Stand Deviation Raster .......")
    stdev_result = np.nanstd(a_s_masked, axis=(2, 3))
    del a_s
    return mean_result, stdev_result


def fs_raster_mean(chm, in_ndarray, kernel):
    # This function uses xrspatial which can handle large data but slow
    print("Calculating Canopy Closure's Focal Statistic-Mean .......")
    result_ndarray = xrspatial.focal.focal_stats(xr.DataArray(in_ndarray), kernel, stats_funcs=['mean'])

    # Flattening the array
    flatten_result_ndarray = result_ndarray.data.reshape(-1)
    # Re-shaping the array

    reshape_ndarray = flatten_result_ndarray.reshape(chm.shape[0], chm.shape[1])
    return reshape_ndarray


def fs_raster_std(chm, in_ndarray, kernel):
    # This function uses xrspatial whcih can handle large data but slow
    print("Calculating Canopy Closure's Focal Statistic-Stand Deviation Raster .......")
    result_ndarray = xrspatial.focal.focal_stats(xr.DataArray(in_ndarray), kernel, stats_funcs=['std'])

    # Flattening the array
    flatten_result_ndarray = result_ndarray.data.reshape(-1)

    # Re-shaping the array
    reshape_ndarray = flatten_result_ndarray.reshape(chm.shape[0], chm.shape[1])
    return reshape_ndarray


def smooth_cost(in_raster, search_dist, out_path):
    print('Generating Cost Raster .......')

    euc_dist_array = None
    # scipy way to do Euclidean distance transform
    with rasterio.open(in_raster) as in_image:
        in_mat = in_image.read(1)
        euc_dist_array = ndimage.distance_transform_edt(np.logical_not(in_mat))

    smooth1 = float(search_dist) - euc_dist_array
    cond_smooth1 = np.where(smooth1 > 0, smooth1, 0.0)
    smooth_cost_array = cond_smooth1 / float(search_dist)

    return smooth_cost_array


def np_cost_raster(canopy_ndarray, cc_mean, cc_std, cc_smooth, chm, avoidance, cost_raster_exponent, out_cost_r):
    print('Generating Smoothed Cost Raster.......')
    aM1a = (cc_mean - cc_std)
    aM1b = (cc_mean + cc_std)
    aM1 = np.divide(aM1a, aM1b, where=aM1b != 0)
    aM = (1 + aM1) / 2
    aaM = (cc_mean + cc_std)
    bM = np.where(aaM <= 0, 0, aM)
    cM = bM * (1 - avoidance) + (cc_smooth * avoidance)
    dM = np.where(canopy_ndarray == 1, 1, cM)
    eM = np.exp(dM)
    result = np.power(eM, float(cost_raster_exponent))
    write_cost = rasterio.open(out_cost_r, 'w+', driver='GTiff', height=chm.shape[0], width=chm.shape[1],
                               count=1,
                               dtype=chm.read(1).dtype, crs=chm.crs, transform=chm.transform)
    write_cost.write(result, 1)
    write_cost.close()
    print('Generating Smoothed Cost Raster.......Done')
    return


# TODO: deal with NODATA
def canopy_cost_raster(callback, in_chm, canopy_ht_threshold, tree_radius, max_line_dist,
                       canopy_avoid, exponent, out_canopy_r, out_cost_r, processes, verbose):
    start_time = time.time()
    #
    # in_chm = args.get('in_chm')
    canopy_ht_threshold = float(canopy_ht_threshold)
    tree_radius = float(tree_radius) # TODO: integer expected
    max_line_dist = float(max_line_dist)
    canopy_avoid = float(canopy_avoid)
    cost_raster_exponent = float(exponent)
    # out_canopy_r = args.get('out_canopy_r')
    # out_cost_r = args.get('out_cost_r')

    print('In CHM: ' + in_chm)
    chm = rasterio.open(in_chm)
    # chm_info = chm.transform
    # chm_crs = chm.crs
    (cell_x, cell_y) = chm.res
    # (row, col) = chm.shape

    print('Loading CHM............')
    band1_ndarray = chm.read(1)

    print('Preparing Kernel window............')
    kernel = convolution.circle_kernel(cell_x, cell_y, tree_radius)

    # Generate Canopy Raster and return the Canopy array
    canopy_ndarray = np_cc_map(out_canopy_r, chm, band1_ndarray, canopy_ht_threshold)
    print(canopy_ndarray.shape[0] * cell_y)
    print(canopy_ndarray.shape[1] * cell_x)

    # Calculating focal statistic from canopy raster
    # TODO: the focal statistic function only can handle small raster, find alternative for handle large raster
    if canopy_ndarray.shape[0] * cell_y <= 100 and canopy_ndarray.shape[1] * cell_x <= 100:
        cc_mean, cc_std = fs_raster(canopy_ndarray, kernel)
    else:
        cc_mean = fs_raster_mean(chm, canopy_ndarray, kernel)
        cc_std = fs_raster_std(chm, canopy_ndarray, kernel)

    # Smoothing raster
    cc_smooth = smooth_cost(out_canopy_r, max_line_dist, os.path.dirname(out_canopy_r))
    avoidance = max(min(float(canopy_avoid), 1), 0)
    np_cost_raster(canopy_ndarray, cc_mean, cc_std, cc_smooth, chm, avoidance, cost_raster_exponent, out_cost_r)
    print('Canopy Cost Raster Process finish in {} sec'.format(time.time() - start_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=json.loads)
    parser.add_argument('-p', '--processes')
    parser.add_argument('-v', '--verbose')
    args = parser.parse_args()

    verbose = True if args.verbose == 'True' else False

    canopy_cost_raster(print, **args.input, processes=int(args.processes), verbose=verbose)
