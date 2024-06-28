import math
import os
import winreg
import time
import geopandas
import sys
import numpy
import xrspatial.focal as focal
from xrspatial import convolution
import xarray as xr

from multiprocessing.pool import Pool
from beratools.tools.common import *


class OperationCancelledException(Exception):
    pass


try:  # integrated R env
    # check R language within env
    current_env_path = os.environ['CONDA_PREFIX']
    # if os.path.isdir(current_env_path):
    os.environ['R_HOME'] = os.path.join(current_env_path, r"Lib\R")
    os.environ['R_USER'] = os.path.expanduser('~')
    os.environ['R_LIBS_USER'] = os.path.join(current_env_path, r"Lib\R\library")

except FileNotFoundError:
    print("Warning: Please install R for this process!!")
    exit()

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, data
from rpy2.robjects.vectors import StrVector


def lpi_lai(arg):
    pdTotal = arg[0]
    pdGround = arg[1]
    out_folder = arg[2]
    filename = arg[3]
    scan_angle = float(arg[4])

    ##variable for not calling R
    cell_size = float(arg[5])
    radius = float(arg[6])
    tfocal_filename = filename + "_tfocal_py.tif"
    gfocal_filename = filename + "_gfocal_py.tif"
    # out_tfocal = os.path.join(out_folder, tfocal_filename)
    # out_gfocal = os.path.join(out_folder, gfocal_filename)

    ## output files variables
    out_lpi_fielname = filename + "_LPI_py.tif"
    out_elai_fielname = filename + "_eLAI_py.tif"
    LPI_folder = os.path.join(out_folder, "LPI")
    eLAI_folder = os.path.join(out_folder, "eLAI")
    out_lpi = os.path.join(LPI_folder, out_lpi_fielname)
    out_elai = os.path.join(eLAI_folder, out_elai_fielname)

    # Working out the searching radius
    # with rasterio.open(chm) as image:
    #     ndarray = image.read(1)
    #     ndarray[ndarray==image.nodata]=numpy.NaN
    #     ndarray[ndarray <0.0] = numpy.NaN
    #     radius = math.ceil(numpy.nanmean(ndarray) * 2)

    print("Calculating LPI and eLAI for {} ...".format(filename))
    with rasterio.open(pdTotal) as pd_total:
        with rasterio.open(pdGround) as pd_Ground:
            raster_profile = pd_total.profile
            pd_total_ndarray = pd_total.read(1, boundless=True)
            nodata = pd_total.nodata
            pd_total_ndarray[pd_total_ndarray == nodata] = numpy.nan
            kernel = convolution.circle_kernel(cell_size, cell_size, radius)
            total_focalsum = fs_raster_stdmean(pd_total_ndarray, kernel, nodata)
            write_total_focalsum = rasterio.open(tfocal_filename, 'w', **raster_profile)
            write_total_focalsum.write(total_focalsum, 1)
            write_total_focalsum.close()
            del write_total_focalsum

            pd_ground_ndarray = pd_Ground.read(1, boundless=True)
            nodata = pd_Ground.nodata
            pd_ground_ndarray[pd_ground_ndarray == nodata] = numpy.nan
            ground_focalsum = fs_raster_stdmean(pd_ground_ndarray, kernel, nodata)
            write_ground_focalsum = rasterio.open(gfocal_filename, 'w', **raster_profile)
            write_ground_focalsum.write(ground_focalsum, 1)
            write_ground_focalsum.close()
            del write_ground_focalsum
    del pd_total

    del pd_Ground
    lpi_array = numpy.divide(pd_ground_ndarray, pd_total_ndarray, out=numpy.zeros_like(pd_ground_ndarray),
                             where=pd_total_ndarray != 0)

    print("Calculating LPI: {} ...".format(filename))
    write_lpi = rasterio.open(out_lpi, 'w', **raster_profile)
    write_lpi.write(lpi_array, 1)
    write_lpi.close()
    del write_lpi
    print('%{}'.format(80))
    print("Calculating LPI: {} ...Done".format(filename))

    print("Calculating eLAI: {} ...".format(filename))
    elai_array = ((math.cos(((scan_angle / 2.0) / 180.0) * math.pi)) / 0.5) * (numpy.log(lpi_array)) * -1

    write_elai = rasterio.open(out_elai, 'w', **raster_profile)
    write_elai.write(elai_array, 1)
    write_elai.close()
    del write_elai
    print("Calculating eLAI: {} ... Done".format(filename))


def fs_raster_stdmean(in_ndarray, kernel, nodata):
    # This function uses xrspatial whcih can handle large data but slow
    in_ndarray[in_ndarray == nodata] = numpy.nan
    result_ndarray = focal.focal_stats(xr.DataArray(in_ndarray), kernel, stats_funcs=['sum'])

    # Flattening the array
    flatten_sum_result_ndarray = result_ndarray.data.reshape(-1)

    # Re-shaping the array
    reshape_sum_ndarray = flatten_sum_result_ndarray.reshape(in_ndarray.shape[0], in_ndarray.shape[1])
    return reshape_sum_ndarray


def r_lpi_lai_with_focalR(arg):
    r = robjects.r
    pdTotal = arg[0]
    pdGround = arg[1]
    out_folder = arg[2]
    filename = arg[3]
    scan_angle = float(arg[4])

    ## output files variables
    out_lpi_fielname = filename + "_LPI_r.tif"
    out_elai_fielname = filename + "_eLAI_r.tif"
    LPI_folder = os.path.join(out_folder, "LPI")
    eLAI_folder = os.path.join(out_folder, "eLAI")
    out_lpi = os.path.join(LPI_folder, out_lpi_fielname)
    out_elai = os.path.join(eLAI_folder, out_elai_fielname)

    radius = float(arg[6])
    print("Calculating LPI and eLAI for {} ...".format(filename))

    # assign R script file to local variable
    rlpi_elai_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'r_cal_lpi_elai.r')
    # Defining the R script and loading the instance in Python
    r['source'](rlpi_elai_script)
    # Loading the function defined in R script.
    rlpi_elai = robjects.globalenv['rlpi_elai']
    # Invoking the R function
    rlpi_elai(pdTotal, pdGround, radius, scan_angle, out_lpi, out_elai)

    # At this stage no process for CHM

    print("Calculating LPI adn eLAI: {} ... Done".format(filename))


def f_pulse_density(ctg, out_folder, rprocesses, verbose):
    r = robjects.r
    print('Calculate cell size from average point cloud density...')
    cache_folder = os.path.join(out_folder, "Cache")
    # assign R script file to local variable
    beratools_r_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Beratools_r_script.r')
    # Defining the R script and loading the instance in Python
    r['source'](beratools_r_script)
    # Loading the function defined in R script.
    pd2cellsize = robjects.globalenv['pd2cellsize']
    # Invoking the R function
    cell_size = pd2cellsize(ctg, rprocesses)

    return (cell_size)


def pd_raster(callback, in_polygon_file, in_las_folder, cut_ht, radius_fr_CHM, focal_radius, pulse_density,
              cell_size, mean_scanning_angle, out_folder, processes, verbose):
    r = robjects.r
    import psutil
    stats = psutil.virtual_memory()
    available = getattr(stats, 'available') / 1024000000
    if 2 < processes <= 8:
        if available <= 50:
            rprocesses = 2
        elif 50 < available <= 150:
            rprocesses = 4
        elif 150 < available <= 250:
            rprocesses = 8
    else:
        rprocesses = 8

    cache_folder = os.path.join(out_folder, "Cache")
    # dtm_folder=os.path.join(out_folder,"DTM")
    # dsm_folder=os.path.join(out_folder,"DSM")
    # chm_folder=os.path.join(out_folder,"CHM")
    PD_folder = os.path.join(out_folder, "PD")
    PD_Total_folder = os.path.join(PD_folder, "Total")
    PD_Ground_folder = os.path.join(PD_folder, "Ground")
    LPI_folder = os.path.join(out_folder, "LPI")
    eLAI_folder = os.path.join(out_folder, "eLAI")

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    if not os.path.exists(PD_folder):
        os.makedirs(PD_folder)
    if not os.path.exists(PD_Total_folder):
        os.makedirs(PD_Total_folder)
    if not os.path.exists(PD_Ground_folder):
        os.makedirs(PD_Ground_folder)
    if not os.path.exists(LPI_folder):
        os.makedirs(LPI_folder)
    if not os.path.exists(eLAI_folder):
        os.makedirs(eLAI_folder)

    lascat = lidR.readLAScatalog(in_las_folder, filter="-drop_class 7")
    cache_folder = cache_folder.replace("\\", "/")
    # dtm_folder = dtm_folder.replace("\\", "/") + "/{*}_dtm"
    # chm_folder = chm_folder.replace("\\","/") + "/{*}_chm"
    PD_Total_folder = PD_folder.replace("\\", "/") + "/Total"
    PD_Ground_folder = PD_folder.replace("\\", "/") + "/Ground"
    LPI_folder = LPI_folder.replace("\\", "/")
    eLAI_folder = eLAI_folder.replace("\\", "/")

    if not in_polygon_file == "":
        try:
            r.vect(in_polygon_file)
        except FileNotFoundError:
            print("Could not locate shapefile, all area will be process")

    if cell_size <= 0:
        if pulse_density <= 0:
            cell_size = f_pulse_density(lascat, out_folder, rprocesses, verbose)

    # assign R script file to local variable
    Beratools_R_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Beratools_r_script.r')
    # Defining the R script and loading the instance in Python
    r['source'](Beratools_R_script)
    # Loading the function defined in R script.
    generate_pd = robjects.globalenv['generate_pd']
    # Invoking the R function
    generate_pd(lascat, radius_fr_CHM, focal_radius, cell_size, cache_folder, cut_ht, PD_Ground_folder,
                PD_Total_folder, rprocesses)

    # At this stage no process for CHM
    #  locate the point density raster for generating eLAI and LPI
    pd_total_filelist = []
    pd_ground_filelist = []

    # Get raster files lists
    for root, dirs, files in sorted(os.walk(os.path.join(out_folder, "PD\\Total"))):
        for file in files:
            if file.endswith("_PD_Tfocalsum.tif"):
                pd_total_filelist.append(os.path.join(root, file))
    del root, dirs, files
    for root, dirs, files in sorted(os.walk(os.path.join(out_folder, "PD\\Ground"))):
        for file in files:
            if file.endswith("_PD_Gfocalsum.tif"):
                pd_ground_filelist.append(os.path.join(root, file))
    del root, dirs, files
    args_list = []

    # At this stage no process for finding average cell size from all CHM
    radius_fr_CHM = False
    if radius_fr_CHM:
        pass
        # chm_filelist = []
        # for root, dirs, files in sorted(os.walk(os.path.join(out_folder, "CHM"))):
        #     for file in files:
        #         if file.endswith("_chm.tif"):
        #             chm_filelist.append(os.path.join(root, file))
        # del root, dirs, files
        #
        # if len(pd_total_filelist) == len(pd_ground_filelist) == len(chm_filelist):
        #     for i in range(0, len(pd_total_filelist)):
        #         chm_filename = os.path.splitext(os.path.split(chm_filelist[i])[1])[0]
        #         pdtotal_filename = os.path.splitext(os.path.split(pd_total_filelist[i])[1])[0]
        #         pdGround_filename = os.path.splitext(os.path.split(pd_ground_filelist[i])[1])[0]
        #         result_list = []
        #         if chm_filename[0:-4] == pdtotal_filename[0:-13] == pdGround_filename[2:-13]:
        #             result_list.append(chm_filelist[i])
        #             result_list.append(pd_total_filelist[i])
        #             result_list.append(pd_ground_filelist[i])
        #             result_list.append(out_folder)
        #             result_list.append(chm_filename[0:-4])
        #             result_list.append(mean_scanning_angle)
        #             result_list.append(cell_size)
        #             result_list.append(focal_radius)
        #             args_list.append(result_list)
        #
        # try:
        #     total_steps = len(args_list)
        #     features = []
        #     with Pool(processes=int(processes)) as pool:
        #         step = 0
        #         # execute tasks in order, process results out of order
        #         for result in pool.imap_unordered(lpi_lai, args_list):
        #             if BT_DEBUGGING:
        #                 print('Got result: {}'.format(result), flush=True)
        #             features.append(result)
        #             step += 1
        #             print('%{}'.format(step / total_steps * 100))
        #
        # except OperationCancelledException:
        #     print("Operation cancelled")
        #     exit()
    else:
        # processing LPI_eLAI
        # prepare arguments
        if len(pd_total_filelist) == len(pd_ground_filelist):
            for i in range(0, len(pd_total_filelist)):

                pdtotal_filename = os.path.splitext(os.path.split(pd_total_filelist[i])[1])[0]
                pdGround_filename = os.path.splitext(os.path.split(pd_ground_filelist[i])[1])[0]
                result_list = []
                if pdtotal_filename[0:-13] == pdGround_filename[2:-13]:
                    result_list.append(pd_total_filelist[i])
                    result_list.append(pd_ground_filelist[i])
                    result_list.append(out_folder)
                    result_list.append(pdtotal_filename[0:-13])
                    result_list.append(mean_scanning_angle)
                    result_list.append(cell_size)
                    result_list.append(focal_radius)
                    args_list.append(result_list)

        # Multiprocessing eLAI and LPI raster using R package.
        try:
            total_steps = len(args_list)
            if processes >= total_steps:
                processes = total_steps

            features = []
            with Pool(processes=int(processes)) as pool:
                step = 0
                # execute tasks in order, process results out of order
                for result in pool.imap_unordered(r_lpi_lai_with_focalR, args_list):
                    if BT_DEBUGGING:
                        print('Got result: {}'.format(result), flush=True)
                    features.append(result)
                    step += 1
                    print('%{}'.format(step / total_steps * 100))

        except OperationCancelledException:
            print("Operation cancelled")
            exit()


if __name__ == '__main__':
    start_time = time.time()
    print('Starting generating LPA and eLAI raster processing\n @ {}'
          .format(time.strftime("%d %b %Y %H:%M:%S", time.localtime())))

    r = robjects.r
    utils = importr('utils')
    base = importr('base')
    utils.chooseCRANmirror(ind=12)  # select the 12th mirror in the list: Canada
    print("Checking R packages ...")
    CRANpacknames = ['lidR', 'rgrass', 'rlas', 'future', 'terra', 'na.tools', 'sf', 'sp']  # ,'fasterRaster']
    CRANnames_to_install = [x for x in CRANpacknames if not robjects.packages.isinstalled(x)]
    need_fasterRaster = False
    if len(CRANnames_to_install) > 0:
        if not 'fasterRaster' in CRANnames_to_install:
            utils.install_packages(StrVector(CRANnames_to_install))
            need_fasterRaster = False
        else:
            CRANnames_to_install.remove('fasterRaster')
            need_fasterRaster = True
            if len(CRANnames_to_install) > 0:
                utils.install_packages(StrVector(CRANnames_to_install))

    # if need_fasterRaster:
    #     devtools=importr('devtools')
    #     devtools.install_github("adamlilith/fasterRaster", dependencies=True)

    del CRANpacknames, CRANnames_to_install

    in_args, in_verbose = check_arguments()
    # loading R packages
    # utils = importr('utils')
    # base = importr('base')
    print("Loading R packages ...")
    na = importr('na.tools')
    terra = importr('terra')
    lidR = importr('lidR')
    sf = importr('sf')
    sp = importr('sp')
    future = importr('future')

    print("Checking input parameters ...")

    aoi_shapefile = in_args.input['in_polygon_file']
    in_las_folder = in_args.input['in_las_folder']
    cut_ht = float(in_args.input['cut_ht'])
    radius_fr_CHM = in_args.input['radius_fr_CHM']
    focal_radius = float(in_args.input['focal_radius'])
    pulse_density = int(in_args.input['pulse_density'])
    cell_size = float(in_args.input['cell_size'])
    mean_scanning_angle = float(in_args.input['mean_scanning_angle'])
    out_folder = in_args.input['out_folder']

    # if optional shapefile is empty, then do nothing, else verify shapefile
    if not aoi_shapefile == "":
        if not os.path.exists(os.path.dirname(aoi_shapefile)):
            print("Can't locate the input polygon folder.  Please check.")
            exit()
        else:
            if not isinstance(geopandas.GeoDataFrame.from_file(aoi_shapefile), geopandas.GeoDataFrame):
                print("Error input file: Please check effective LiDAR data extend shapefile")
                exit()
    # check existence of input las/laz folder
    if not os.path.exists(in_las_folder):
        print("Error! Cannot locate LAS/LAZ folder, please check.")
        exit()
    else:
        found = False
        for files in os.listdir(in_las_folder):
            if files.endswith(".las") or files.endswith(".laz"):
                found = True
                break
        if not found:
            print("Error! Cannot locate input LAS file(s), please check!")
            exit()

    # if doing focal radius divided from point cloud CHM
    if radius_fr_CHM == True:
        pass
        # do nothing for now
    else:
        # check manual input for radius, check input
        if not isinstance(focal_radius, float) or focal_radius <= 0.0:
            print("Invalid search radius!!Default radius will be adopted (10m).")
            in_args.input['focal_radius'] = 10.0
        else:
            in_args.input['focal_radius'] = focal_radius
        # check manual input for cell size and pulse density
        if not isinstance(cell_size, float) or cell_size <= 0.00:
            if not isinstance(pulse_density, int) or pulse_density <= 0.00:
                print("Invalid cell size and average pulse density provided.\n"
                      "Cell size will be calulated based on aveage point density.")
                in_args.input['cell_size'] = 0.0
                in_args.input['pulse_density'] = 0
            else:
                # mean_pd = (((math.pow(3 / pulse_density, 1 / 2)) + (math.pow(5 / pulse_density, 1 / 2))) / 2)
                mean_pd = math.pow(3 / pulse_density, 1 / 2)
                # mean_pd =   math.pow(5 / pulse_density, 1 / 2)
                in_args.input['cell_size'] = round(0.05 * round(mean_pd / 0.05), 2)
                in_args.input['pulse_density'] = pulse_density
        else:
            in_args.input['cell_size'] = (cell_size)
            in_args.input['pulse_density'] = pulse_density

    # Check manual input for cutt off height
    if not isinstance(cut_ht, float) and cut_ht > 0.0:
        print("Invalid cut off height!! Default cut off height will be adopted (1m).")
        in_args.input["cut_ht"] = 1.0

    if not isinstance(mean_scanning_angle, float) and mean_scanning_angle > 0.00:
        print("Invalid sensor scanning angle.\n Default sensor scanning angle will size be adopted (30 degree).")
        in_args.input['mean_scanning_angle'] = 30.0
    else:
        in_args.input['mean_scanning_angle'] = mean_scanning_angle

    print("Checking input parameters ... Done")

    pd_raster(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Task is done in {} seconds)'.format(round(time.time() - start_time, 5)))
