import numpy as np
import xrspatial
import xarray as xr
from scipy import ndimage
import beratools.core.constants as bt_const

def remove_nan_from_array(matrix):
    with np.nditer(matrix, op_flags=["readwrite"]) as it:
        for x in it:
            if np.isnan(x[...]):
                x[...] = bt_const.BT_NODATA_COST

def dyn_np_cc_map(in_chm, canopy_ht_threshold, nodata):
    canopy_ndarray = np.ma.where(in_chm >= canopy_ht_threshold, 1.0, 0.0).astype(float)
    canopy_ndarray.fill_value = nodata

    return canopy_ndarray

def dyn_fs_raster_stdmean(canopy_ndarray, kernel, nodata):
    # This function uses xrspatial which can handle large data but slow
    mask = canopy_ndarray.mask
    in_ndarray = np.ma.where(mask == True, np.NaN, canopy_ndarray)
    result_ndarray = xrspatial.focal.focal_stats(
        xr.DataArray(in_ndarray.data), kernel, stats_funcs=["std", "mean"]
    )

    # Assign std and mean ndarray (return array contain NaN value)
    reshape_std_ndarray = result_ndarray[0].data
    reshape_mean_ndarray = result_ndarray[1].data

    return reshape_std_ndarray, reshape_mean_ndarray


def dyn_smooth_cost(canopy_ndarray, max_line_dist, sampling):
    mask = canopy_ndarray.mask
    in_ndarray = np.ma.where(mask == True, np.NaN, canopy_ndarray)
    # scipy way to do Euclidean distance transform
    euc_dist_array = ndimage.distance_transform_edt(
        np.logical_not(np.isnan(in_ndarray.data)), sampling=sampling
    )
    euc_dist_array[mask == True] = np.NaN
    smooth1 = float(max_line_dist) - euc_dist_array
    smooth1[smooth1 <= 0.0] = 0.0
    smooth_cost_array = smooth1 / float(max_line_dist)

    return smooth_cost_array


def dyn_np_cost_raster(
    canopy_ndarray, cc_mean, cc_std, cc_smooth, avoidance, cost_raster_exponent
):
    aM1a = cc_mean - cc_std
    aM1b = cc_mean + cc_std
    aM1 = np.divide(aM1a, aM1b, where=aM1b != 0, out=np.zeros(aM1a.shape, dtype=float))
    aM = (1 + aM1) / 2
    aaM = cc_mean + cc_std
    bM = np.where(aaM <= 0, 0, aM)
    cM = bM * (1 - avoidance) + (cc_smooth * avoidance)
    dM = np.where(canopy_ndarray.data == 1, 1, cM)
    eM = np.exp(dM)
    result = np.power(eM, float(cost_raster_exponent))

    return result

# Function using scipy.ndimage.generic_filter (new version)
def dyn_fs_raster_stdmean_scipy(canopy_ndarray, kernel, nodata):
    # Ensure the input array is a float type to accommodate NaN values
    in_ndarray = canopy_ndarray.astype(float)

    # Mask the array where the nodata values are present
    mask = in_ndarray == nodata
    in_ndarray[mask] = np.NaN  # Replace nodata values with NaN

    # Function to compute mean and standard deviation
    def calc_mean(arr):
        return np.nanmean(arr)

    def calc_std(arr):
        return np.nanstd(arr)

    # Apply the generic_filter function to compute mean and std
    mean_array = ndimage.generic_filter(in_ndarray, calc_mean, footprint=kernel, mode='nearest')
    std_array = ndimage.generic_filter(in_ndarray, calc_std, footprint=kernel, mode='nearest')

    return std_array, mean_array

from scipy.spatial.distance import cdist

# Function to create a circular kernel using Scipy
def circle_kernel_scipy(size, radius):
    """

    Parameters
    ----------
    size :
    radius :

    Returns
    -------

    Examples:
    kernel_scipy = create_circle_kernel_scipy(17, 8)
    will replicate xrspatial kernel
    cell_x = 0.3
    cell_y = 0.3
    tree_radius = 2.5
    xrspatial.convolution.circle_kernel(cell_x, cell_y, tree_radius)

    """
    # Create grid points (mesh)
    y, x = np.ogrid[:size, :size]

    # Center of the kernel
    center_x, center_y = (size - 1) / 2, (size - 1) / 2

    # Calculate the distance from the center
    distance = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)

    # Create a circular kernel
    kernel = distance <= radius
    return kernel.astype(float)