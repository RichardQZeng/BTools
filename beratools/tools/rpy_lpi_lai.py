import math
import os
import sys
import time
import numpy
import rasterio
from xrspatial import convolution
from xrspatial.focal import _focal_stats_cpu
from multiprocessing.pool import Pool
import geopandas
import winreg
from common import *
class OperationCancelledException(Exception):
    pass

import xarray
from xrspatial import convolution
from xrspatial.focal import focal_stats
# import platform
#
# bitness = platform.architecture()[0]
# if bitness == '32bit':
#     other_view_flag = winreg.KEY_WOW64_64KEY
# elif bitness == '64bit':
#     other_view_flag = winreg.KEY_WOW64_32KEY
r_library=os.path.join(os.path.expanduser('~'), "AppData\\Local\\R\\win-library")
try:
    aKey = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, "SOFTWARE\\R-core\\R")
    for i in range((winreg.QueryInfoKey(aKey))[0]):
        aValue_name = winreg.EnumKey(aKey, i)
        oKey = winreg.OpenKey(aKey, aValue_name)
        r_install_path = winreg.QueryValueEx(oKey, "installPath")[0]
    os.environ['R_HOME'] = r_install_path
    os.environ['R_USER'] = os.path.expanduser('~')
    if os.path.isdir(r_library):
        os.environ['R_LIBS_USER'] = r_library
    else:
        os.makedirs(r_library)
        os.environ['R_LIBS_USER'] = r_library

except FileNotFoundError:
    print("Warning: Please install R for this process!!")
    exit()


def rfocal(inarray, fw, fun, na_rm,type):
    r=robjects.r
    out_focal = r.focal(inarray, w=fw, fun=fun, na_rm=na_rm)
    return out_focal, type
def fs_raster_sum(in_ndarray, kernel, nodata):
    # This function uses xrspatial whcih can handle large data but slow
    import xarray as xr
    from xrspatial import convolution, focal
    in_ndarray[in_ndarray == nodata] = numpy.nan
    result_ndarray = focal.focal_stats(xr.DataArray(in_ndarray), kernel, stats_funcs=['sum'])

    # Flattening the array
    flatten_sum_result_ndarray = result_ndarray[0].data.reshape(-1)
    # flatten_mean_result_ndarray = result_ndarray[1].data.reshape(-1)

    # Re-shaping the array
    reshape_sum_ndarray = flatten_sum_result_ndarray.reshape(in_ndarray.shape[0], in_ndarray.shape[1])
    # reshape_mean_ndarray = flatten_mean_result_ndarray.reshape(in_ndarray.shape[0], in_ndarray.shape[1])
    return reshape_sum_ndarray


if __name__ == '__main__':
    import rpy2.robjects as robjects
    from rpy2.robjects.packages import importr, data
    from rpy2.robjects.vectors import StrVector
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=json.loads)
    parser.add_argument('-p', '--processes')
    parser.add_argument('-v', '--verbose')
    args = parser.parse_args()
    processes = int(args.processes)
    start_time = time.time()
    print('Starting Generate LPI and eLAI raster processing\n @ {}'
          .format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    verbose = True if args.verbose == 'True' else False
    # tool_name(print, **args.input, processes=int(args.processes), verbose=verbose)
    in_polygon_file = args.input['in_polygon_file']
    pd_total = args.input['pd_total']
    pd_ground = args.input['pd_ground']
    radius_fr_CHM=args.input['radius_fr_CHM']

    # in_chm = args.input['in_chm']
    pd = int(args.input['point_density'])
    ps = float(args.input['cell_size'])

    scan_angle = float(args.input['mean_scanning_angle'])
    out_lpi = args.input['out_lpi']
    out_elai = args.input['out_elai']
    print("Checking input......")

    # if not isinstance(geopandas.GeoDataFrame.from_file(in_polygon_file), geopandas.GeoDataFrame):
    #     print("Error input file: Please the effective LiDAR data extend shapefile")
    #     exit()

    if not os.path.isfile(pd_total):
        print("Error input file: Please check input Pulse density total (Rt) Raster.")
        exit()
    else:
        in_file = os.path.split(pd_total)[1]
        in_filename, infile_ext = os.path.splitext(in_file)

    if not os.path.isfile(pd_ground):
        print("Error input file: Please check input Pulse density ground < x meter (Rt) Raster.")
        exit()

    if not os.path.exists(os.path.dirname(out_lpi)):
        print("Error output file: Could not locate the output folder/path.")
        exit()
    else:
        out_folder = os.path.normpath(os.path.dirname(out_lpi))

    if not os.path.exists(os.path.dirname(out_elai)):
        print("Error output file: Could not locate the output folder/path.")
        exit()
    else:
        pass
    tfocal_filename = in_filename + "_tfocal.tif"
    gfocal_filename = in_filename + "_gfocal.tif"
    out_tfocal = os.path.join(out_folder, tfocal_filename)
    out_gfocal = os.path.join(out_folder, gfocal_filename)
    print('%{}'.format(10))
    print("Checking input......Done")

    print("Check and install necessary packages......")
    r = robjects.r
    utils = importr('utils')
    base = importr('base')
    utils.chooseCRANmirror(ind=12)  # select the 12th mirror in the list: Canada
    CRANpacknames = ['comprehenr', 'na.tools', 'terra']
    CRANnames_to_install = [x for x in CRANpacknames if not robjects.packages.isinstalled(x)]

    if len(CRANnames_to_install) > 0:
        utils.install_packages(StrVector(CRANnames_to_install))

    comprehenr = importr('comprehenr')
    na = importr('na.tools')
    terra = importr('terra')
    del CRANpacknames, CRANnames_to_install
    print("Check and install necessary packages......Done")
    print('%{}'.format(30))
    #
    # r('''
    # #create a 'substRight' function
    # substrRight <- function(x, n){
    #   substr(x, nchar(x)-n+1, nchar(x))
    # }''')
    # substrRight = r['substrRight']
    if isinstance(float(args.input['search_radius']), float) and float(args.input['search_radius'])>0.0:
        radius=float(args.input['search_radius'])
    else:
        if radius_fr_CHM == False:
            print("Invalid focal radius. Default value is adopted")
            radius=10.0
        else:
            print("Default value is adopted")
            radius = 10.0
        # elif os.path.isfile(in_chm):
        #     with rasterio.open(in_chm) as image:
        #         chm = image.read(1)
        #         radius=math.ceil(numpy.nanmean(chm)*2)
        # else:
        #     exit()

    # area = r.substr(area, 1, 16)
    # print("\n".join([str(i),area]))
    shp = r.vect(in_polygon_file)  # will drop z values
    # shp = r.st_zm(shp)
    shp_small = r.buffer(shp, -1 * radius)
    # loading raster using r.terra.rast
    total = r.rast(pd_total)
    ground = r.rast(pd_ground)
    ground = r.extend(ground, r.ext(total))

    # create a kernel in R matix
    # fw = terra.focalMat(total, radius, "circle", fillNA=True)
    # # turn to numpy
    # np_fw = numpy.array(fw)
    # np_fw[np_fw > 0] = 1
    # npr, npc = np_fw.shape
    # t_np_fw = robjects.FloatVector(np_fw.transpose().reshape((np_fw.size)))
    # fw = r.matrix(t_np_fw, nrow=npr, ncol=npc)
    # # R: focal_area = sum(fw, na.rm=T) * (ps) * (ps)
    # focal_area = numpy.nansum(np_fw) * (ps) * (ps )

    total_fw = terra.focalMat(total, radius, "circle", fillNA=True)
    np_total_fw = numpy.array(total_fw)
    np_total_fw[np_total_fw > 0] = 1
    npr, npc = np_total_fw.shape
    t_np_total_fw = robjects.FloatVector(np_total_fw.transpose().reshape((np_total_fw.size)))
    total_fw = r.matrix(t_np_total_fw, nrow=npr, ncol=npc)
    print('%{}'.format(40))
    ground_fw = terra.focalMat(ground, radius, "circle", fillNA=True)
    np_ground_fw = numpy.array(ground_fw)
    np_ground_fw[np_ground_fw > 0] = 1
    npr, npc = np_ground_fw.shape
    t_np_ground_fw = robjects.FloatVector(np_ground_fw.transpose().reshape((np_ground_fw.size)))
    ground_fw = r.matrix(t_np_ground_fw, nrow=npr, ncol=npc)
    print('%{}'.format(50))

    print("Calculating focal statistic on pulse density rasters....")
    total_focal = r.focal(total, w=total_fw, fun="sum", na_rm=True)
    # total_focal = r.rast(r"D:\Maverick\BERATool_Test_Data\Kirby_Test\DJI_Drone\Test_v10\KirbySouth2022_B_PD147_PD_Total_focalSum_tfocal.tif")
    print('%{}'.format(60))
    print("....")
    ground_focal = r.focal(ground, w=ground_fw, fun="sum", na_rm=True)
    # ground_focal = r.rast(r"D:\Maverick\BERATool_Test_Data\Kirby_Test\DJI_Drone\Test_v10\KirbySouth2022_B_PD147_PD_Total_focalSum_gfocal.tif")
    print('%{}'.format(70))
    # total_focal = r.mask(total_focal, shp_small)
    r.writeRaster(total_focal, out_tfocal, overwrite=True)
    # ground_focal = r.mask(ground_focal, shp_small)
    r.writeRaster(ground_focal, out_gfocal, overwrite=True)
    print("Calculating focal statistic on pulse density rasters....Done")


    with rasterio.open(out_tfocal) as tfocal:
        tfocal_array = tfocal.read(1)
        raster_profile = tfocal.profile
    del tfocal
    with rasterio.open(out_gfocal) as gfocal:
        gfocal_array = gfocal.read(1)
    del gfocal
    lpi_array = numpy.divide(gfocal_array, tfocal_array, out=numpy.zeros_like(gfocal_array), where=tfocal_array != 0)
    print("Calculating LPI....")
    write_lpi = rasterio.open(out_lpi, 'w', **raster_profile)
    write_lpi.write(lpi_array, 1)
    write_lpi.close()

    del write_lpi
    print('%{}'.format(80))
    print("Calculating LPI....Done")

    print("Calculating eLAI....")
    elai_array= ((math.cos(((scan_angle/2.0) / 180.0) * math.pi)) / 0.5) * (numpy.log(lpi_array)) * -1

    write_elai = rasterio.open(out_elai, 'w', **raster_profile)
    write_elai.write(elai_array, 1)
    write_elai.close()
    del write_elai
    print("Calculating eLAI....Done")
    print('%{}'.format(100))

    print('Generate LPI and eLAI rasters are done in {} seconds)'
          .format(round(time.time() - start_time, 5)))
