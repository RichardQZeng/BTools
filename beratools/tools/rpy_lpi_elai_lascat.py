import math
import os
import winreg
import time
import geopandas
import sys
import numpy
import rasterio
import xarray
from xrspatial import convolution
from xrspatial.focal import focal_stats
from multiprocessing.pool import Pool

from common import *
class OperationCancelledException(Exception):
    pass
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

import rpy2
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, data
from rpy2.robjects.vectors import StrVector


def lpi_lai(arg):
    r = robjects.r
    terra = importr('terra')
    chm=arg[0]
    pdTotal=arg[1]
    pdGround=arg[2]
    out_folder=arg[3]
    filename=arg[4]
    scan_angle=float(arg[5])

    ##variable for not calling R
    cell_size =float(arg[6])
    tfocal_filename = filename + "_tfocal.tif"
    gfocal_filename = filename + "_gfocal.tif"
    out_tfocal = os.path.join(out_folder, tfocal_filename)
    out_gfocal = os.path.join(out_folder, gfocal_filename)

    ## output files variables
    out_lpi_fielname = filename + "_LPI.tif"
    out_elai_fielname = filename + "_eLAI.tif"
    LPI_folder = os.path.join(out_folder, "LPI")
    eLAI_folder = os.path.join(out_folder, "eLAI")
    out_lpi= os.path.join(LPI_folder, out_lpi_fielname)
    out_elai= os.path.join(eLAI_folder, out_elai_fielname)

    # Working out the searching radius
    with rasterio.open(chm) as image:
        ndarray = image.read(1)
        ndarray[ndarray==image.nodata]=numpy.NaN
        ndarray[ndarray <0.0] = numpy.NaN
        radius = math.ceil(numpy.nanmean(ndarray) * 2)

    print("Calculating LPI and eLAI for {}....".format(filename))
    r(''' #create a 'rlpi_elai' function
    library(terra)
    rlpi_elai <- function(pdTotal,pdGround,radius,scan_angle,out_lpi,out_elai){

    pdTotal_SpatRaster <- rast(pdTotal)
    pdGround_SpatRaster <- rast(pdGround)
    fw <- focalMat(pdTotal_SpatRaster, radius, "circle", fillNA=TRUE)
    fw[fw>0] = 1
    fw[fw==0] = NA

    total_focal <- focal(pdTotal_SpatRaster,na.policy="omit", w=fw, fun="sum", na.rm=TRUE)
    ground_focal = focal(pdGround_SpatRaster,na.policy="omit", w=fw, fun="sum", na.rm=TRUE)

    # lpi
    lpi = ground_focal / total_focal
    #lpi
    lpi[is.infinite(lpi)] = NA

    elai = -cos(((scan_angle / 2.0)/180)*pi)/0.5 * log(lpi)

    writeRaster(lpi,out_lpi,overwrite=TRUE)


    writeRaster(lpi,out_elai,overwrite=TRUE)

    }''')

    rlpi_elai = r['rlpi_elai']
    rlpi_elai(pdTotal,pdGround,radius,scan_angle,out_lpi,out_elai)

    # calling R through rpy2 and in/out python package
    # create a kernel in R matix
    # pdTotal_SpatRaster=terra.rast(pdTotal)
    # pdGround_SpatRaster=terra.rast(pdGround)
    # fw = terra.focalMat(pdTotal_SpatRaster, radius, "circle", fillNA=True)
    # # turn to numpy
    # np_fw = numpy.array(fw)
    # np_fw[np_fw > 0] = 1
    # npr, npc = np_fw.shape
    # t_np_fw = robjects.FloatVector(np_fw.transpose().reshape((np_fw.size)))
    # fw = r.matrix(t_np_fw, nrow=npr, ncol=npc)
    # # R: focal_area = sum(fw, na.rm=T) * (ps) * (ps)
    # focal_area = numpy.nansum(np_fw) * (cell_size) * (cell_size)
    #
    # total_fw = terra.focalMat(pdTotal_SpatRaster, radius, "circle", fillNA=True)
    # np_total_fw = numpy.array(total_fw)
    # np_total_fw[np_total_fw > 0] = 1
    # npr, npc = np_total_fw.shape
    # t_np_total_fw = robjects.FloatVector(np_total_fw.transpose().reshape((np_total_fw.size)))
    # total_fw = r.matrix(t_np_total_fw, nrow=npr, ncol=npc)
    # print('%{}'.format(40))
    # ground_fw = terra.focalMat(pdGround_SpatRaster, radius, "circle", fillNA=True)
    # np_ground_fw = numpy.array(ground_fw)
    # np_ground_fw[np_ground_fw > 0] = 1
    # npr, npc = np_ground_fw.shape
    # t_np_ground_fw = robjects.FloatVector(np_ground_fw.transpose().reshape((np_ground_fw.size)))
    # ground_fw = r.matrix(t_np_ground_fw, nrow=npr, ncol=npc)
    # print('%{}'.format(50))
    #
    # print("Calculating focal statistic on pulse density rasters: {}....".format(filename))
    # total_focal = terra.focal(pdTotal_SpatRaster,na_policy="omit", w=total_fw, fun="sum", na_rm=True)
    # print('%{}'.format(60))
    # ground_focal = terra.focal(pdGround_SpatRaster,na_policy="omit", w=ground_fw, fun="sum", na_rm=True)
    # print('%{}'.format(70))
    # r.writeRaster(total_focal, out_tfocal, overwrite=True)
    # r.writeRaster(ground_focal, out_gfocal, overwrite=True)
    # print("Calculating focal statistic on pulse density rasters: {}....Done".format(filename))
    #
    # with rasterio.open(out_tfocal) as tfocal:
    #     tfocal_array = tfocal.read(1)
    #     raster_profile = tfocal.profile
    # del tfocal
    # with rasterio.open(out_gfocal) as gfocal:
    #     gfocal_array = gfocal.read(1)
    # del gfocal
    # lpi_array = numpy.divide(gfocal_array, tfocal_array, out=numpy.zeros_like(gfocal_array),
    #                          where=tfocal_array != 0)
    #
    # print("Calculating LPI: {}....".format(filename))
    # write_lpi = rasterio.open(out_lpi, 'w', **raster_profile)
    # write_lpi.write(lpi_array, 1)
    # write_lpi.close()
    #
    # del write_lpi
    # print('%{}'.format(80))
    print("Calculating LPI: {}....Done".format(filename))
    #
    # print("Calculating eLAI: {}....".format(filename))
    # elai_array = ((math.cos(((scan_angle / 2.0) / 180.0) * math.pi)) / 0.5) * \
    #              (numpy.log(lpi_array,out=numpy.zeros_like(lpi_array),where=(lpi_array!=0))) * -1
    #
    # write_elai = rasterio.open(out_elai, 'w', **raster_profile)
    # write_elai.write(elai_array, 1)
    # write_elai.close()
    # del write_elai
    print("Calculating eLAI: {}....Done".format(filename))
    # print('%{}'.format(100))

def lpi_lai_with_focalR(arg):
    r = robjects.r
    terra = importr('terra')
    # chm=arg[0]
    pdTotal=arg[0]
    pdGround=arg[1]
    out_folder=arg[2]
    filename=arg[3]
    scan_angle=float(arg[4])


    ##variable for not calling R
    cell_size =float(arg[5])
    tfocal_filename = filename + "_tfocal.tif"
    gfocal_filename = filename + "_gfocal.tif"
    out_tfocal = os.path.join(out_folder, tfocal_filename)
    out_gfocal = os.path.join(out_folder, gfocal_filename)

    ## output files variables
    out_lpi_fielname = filename + "_LPI.tif"
    out_elai_fielname = filename + "_eLAI.tif"
    LPI_folder = os.path.join(out_folder, "LPI")
    eLAI_folder = os.path.join(out_folder, "eLAI")
    out_lpi = os.path.join(LPI_folder, out_lpi_fielname)
    out_elai = os.path.join(eLAI_folder, out_elai_fielname)




    radius = float(arg[6])
    print("Calculating LPI and eLAI for {}....".format(filename))
    r(''' #create a 'rlpi_elai' function
    library(terra)
    rlpi_elai <- function(pdTotal,pdGround,radius,scan_angle,out_lpi,out_elai){

    pdTotal_SpatRaster <- rast(pdTotal)
    pdGround_SpatRaster <- rast(pdGround)
    fw <- focalMat(pdTotal_SpatRaster, radius, "circle", fillNA=TRUE)
    fw[fw>0] = 1
    fw[fw==0] = NA

    total_focal <- focal(pdTotal_SpatRaster,na.policy="omit", w=fw, fun="sum", na.rm=TRUE)
    ground_focal = focal(pdGround_SpatRaster,na.policy="omit", w=fw, fun="sum", na.rm=TRUE)

    # lpi
    lpi = ground_focal / total_focal
    #lpi
    lpi[is.infinite(lpi)] = NA

    elai = -cos(((scan_angle / 2.0)/180)*pi)/0.5 * log(lpi)

    writeRaster(lpi,out_lpi,overwrite=TRUE)


    writeRaster(lpi,out_elai,overwrite=TRUE)

    }''')

    rlpi_elai = r['rlpi_elai']
    rlpi_elai(pdTotal,pdGround,radius,scan_angle,out_lpi,out_elai)

    # calling R through rpy2 and in/out python package
    # create a kernel in R matix
    # pdTotal_SpatRaster=terra.rast(pdTotal)
    # pdGround_SpatRaster=terra.rast(pdGround)
    # fw = terra.focalMat(pdTotal_SpatRaster, radius, "circle", fillNA=True)
    # # turn to numpy
    # np_fw = numpy.array(fw)
    # np_fw[np_fw > 0] = 1
    # npr, npc = np_fw.shape
    # t_np_fw = robjects.FloatVector(np_fw.transpose().reshape((np_fw.size)))
    # fw = r.matrix(t_np_fw, nrow=npr, ncol=npc)
    # # R: focal_area = sum(fw, na.rm=T) * (ps) * (ps)
    # focal_area = numpy.nansum(np_fw) * (cell_size) * (cell_size)
    #
    # total_fw = terra.focalMat(pdTotal_SpatRaster, radius, "circle", fillNA=True)
    # np_total_fw = numpy.array(total_fw)
    # np_total_fw[np_total_fw > 0] = 1
    # npr, npc = np_total_fw.shape
    # t_np_total_fw = robjects.FloatVector(np_total_fw.transpose().reshape((np_total_fw.size)))
    # total_fw = r.matrix(t_np_total_fw, nrow=npr, ncol=npc)
    # print('%{}'.format(40))
    # ground_fw = terra.focalMat(pdGround_SpatRaster, radius, "circle", fillNA=True)
    # np_ground_fw = numpy.array(ground_fw)
    # np_ground_fw[np_ground_fw > 0] = 1
    # npr, npc = np_ground_fw.shape
    # t_np_ground_fw = robjects.FloatVector(np_ground_fw.transpose().reshape((np_ground_fw.size)))
    # ground_fw = r.matrix(t_np_ground_fw, nrow=npr, ncol=npc)
    # print('%{}'.format(50))
    #
    # print("Calculating focal statistic on pulse density rasters: {}....".format(filename))
    # total_focal = terra.focal(pdTotal_SpatRaster,na_policy="omit", w=total_fw, fun="sum", na_rm=True)
    # print('%{}'.format(60))
    # ground_focal = terra.focal(pdGround_SpatRaster,na_policy="omit", w=ground_fw, fun="sum", na_rm=True)
    # print('%{}'.format(70))
    # r.writeRaster(total_focal, out_tfocal, overwrite=True)
    # r.writeRaster(ground_focal, out_gfocal, overwrite=True)
    # print("Calculating focal statistic on pulse density rasters: {}....Done".format(filename))
    #
    # with rasterio.open(out_tfocal) as tfocal:
    #     tfocal_array = tfocal.read(1)
    #     raster_profile = tfocal.profile
    # del tfocal
    # with rasterio.open(out_gfocal) as gfocal:
    #     gfocal_array = gfocal.read(1)
    # del gfocal
    # lpi_array = numpy.divide(gfocal_array, tfocal_array, out=numpy.zeros_like(gfocal_array),
    #                          where=tfocal_array != 0)
    #
    # print("Calculating LPI: {}....".format(filename))
    # write_lpi = rasterio.open(out_lpi, 'w', **raster_profile)
    # write_lpi.write(lpi_array, 1)
    # write_lpi.close()
    #
    # del write_lpi
    # print('%{}'.format(80))
    print("Calculating LPI: {}....Done".format(filename))
    #
    # print("Calculating eLAI: {}....".format(filename))
    # elai_array = ((math.cos(((scan_angle / 2.0) / 180.0) * math.pi)) / 0.5) * \
    #              (numpy.log(lpi_array,out=numpy.zeros_like(lpi_array),where=(lpi_array!=0))) * -1
    #
    # write_elai = rasterio.open(out_elai, 'w', **raster_profile)
    # write_elai.write(elai_array, 1)
    # write_elai.close()
    # del write_elai
    print("Calculating eLAI: {}....Done".format(filename))
    # print('%{}'.format(100))

def pulse_density(ctg, out_folder, processes, verbose):
    r = robjects.r

    cache_folder = os.path.join(out_folder, "Cache")

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    r('''
       routine <- function(chunk){ 
      las <- readLAS(chunk)

      if (is.empty(las)) return(NULL)
      las <- retrieve_pulses(las)
      output <- density(las)[1]
      # output is a list
      return(output)
       }
       ''')
    routine = r['routine']
    list_pd = r.catalog_apply(ctg, routine)
    pd=numpy.nanmean(numpy.array(list_pd))

    # mean_pd = (((math.pow(3 / pd, 1 / 2)) + (math.pow(5 / pd, 1 / 2))) / 2)
    mean_pd = (math.pow(3 / pd, 1 / 2))
    result = round(0.05 * round(mean_pd / 0.05), 2)
    return (result)


def pd_raster(callback, in_polygon_file, in_las_folder, cut_ht, radius_fr_CHM, focal_radius, pulse_density,
              cell_size, mean_scanning_angle, out_folder, processes, verbose):

    r = robjects.r
    if 8<=processes:
        processes=8

    r.plan(r.multisession,workers=processes)
    r.set_lidr_threads(math.ceil(processes))


    cache_folder=os.path.join(out_folder,"Cache")
    dtm_folder=os.path.join(out_folder,"DTM")
    # dsm_folder=os.path.join(out_folder,"DSM")
    chm_folder=os.path.join(out_folder,"CHM")
    PD_folder=os.path.join(out_folder,"PD")
    PD_Total_folder=os.path.join(PD_folder,"Total")
    PD_Ground_folder = os.path.join(PD_folder, "Ground")
    LPI_folder = os.path.join(out_folder, "LPI")
    eLAI_folder = os.path.join(out_folder, "eLAI")



    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    if not os.path.exists(dtm_folder):
        os.makedirs(dtm_folder)

    if not os.path.exists(chm_folder):
        os.makedirs(chm_folder)
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

    lascat = r.readLAScatalog(in_las_folder)
    cache_folder = cache_folder.replace("\\","/")
    dtm_folder = dtm_folder.replace("\\", "/") + "/{*}_dtm"

    chm_folder = chm_folder.replace("\\","/")+"/{*}_chm"
    PD_Total_folder = PD_folder.replace("\\","/")+"/Total/{*}_PD_Total"
    PD_Ground_folder = PD_folder.replace("\\", "/") + "/Ground/{*}_PD_Ground"

    if not in_polygon_file=="":
        r.vect(in_polygon_file)

    if cell_size<=0:
        if pulse_density<=0:
            cell_size=pulse_density(lascat,out_folder,processes, verbose)


    print("Retrieving pulses information....")
    r('''
    retrieve_pd <- function(ctg,cache_folder){
    routine <- function(chunk,folder)
    {
    las <- readLAS(chunk)
    if (is.empty(las)) return(NULL)
    las <- retrieve_pulses(las)
    las <- filter_poi(las,buffer==0)
    }
    opt <- list(need_output_file =TRUE, autocrop = TRUE)
    # opt_laz_compression(ctg) <- TRUE
    opt_output_files(ctg) <- paste0(cache_folder,"/{XLEFT}_{YBOTTOM}")
    catalog_apply(ctg, routine,folder=cache_folder,.options=opt)
    }''')
    retrieve_pd=r['retrieve_pd']
    retrieve_pd(lascat, cache_folder)

    lascat = r.readLAScatalog(cache_folder, filter="-drop_class 7 ")

    r('''#create a 'generate_pd' function
    generate_pd <- function(ctg,radius_fr_CHM,cell_size,cache_folder,dtm_folder,chm_folder,cut_ht,PD_Ground_folder,PD_Total_folder){
    
    print("Generate DTM....")
    opt_output_files(ctg) <- opt_output_files(ctg) <- dtm_folder
    ctg@output_options$drivers$SpatRaster$param$overwrite <- TRUE
    opt_stop_early(ctg) <- FALSE
    dtm <- rasterize_terrain(ctg,cell_size,knnidw(),pkg="terra")

    print("Normalize point cloud....")
    folder <- paste0(cache_folder,"/n_{*}" )
    opt_output_files(ctg) <- opt_output_files(ctg) <- folder
    opt_laz_compression(ctg) <- TRUE
    ctg2 <- normalize_height(ctg, algorithm=dtm)

    if (radius_fr_CHM==TRUE) {
    print("Generate CHM raster....")
    ctg2@output_options$drivers$SpatRaster$param$overwrite <- TRUE
    opt_output_files(ctg2) <- chm_folder
    chm <- rasterize_canopy(ctg2, res = cell_size, algorithm = dsmtin(max_edge = 5))}

    print("Generate pulse density (total) raster....")
    opt_output_files(ctg2) <- PD_Total_folder
    ctg2@output_options$drivers$SpatRaster$param$overwrite <- TRUE
    opt_merge(ctg2) <- TRUE
    density_raster_total <- rasterize_density(ctg2, res=cell_size,pkg="terra")[1]

    print("Generate pulse density (ground) raster....")
    ht<- paste0("-keep_first -first_only -drop_class 7 -drop_z_above ",cut_ht)
    opt_filter(ctg2) <- ht
    opt_output_files(ctg2) <- PD_Ground_folder
    opt_merge(ctg2) <- TRUE
    ctg2@output_options$drivers$SpatRaster$param$overwrite <- TRUE
    density_raster_ground <- rasterize_density(ctg2, res=cell_size,pkg="terra")[1] }''')

    generate_pd = r['generate_pd']

    generate_pd(lascat, radius_fr_CHM, cell_size, cache_folder, dtm_folder, chm_folder, cut_ht, PD_Ground_folder,
                    PD_Total_folder)



    pd_total_filelist=[]
    pd_ground_filelist = []
    if radius_fr_CHM:
        chm_filelist = []
        for root,dirs ,files in sorted(os.walk(os.path.join(out_folder,"CHM"))):
            for file in files:
                if file.endswith("_chm.tif"):
                    chm_filelist.append(os.path.join(root, file))
        del root,dirs,files
    for root,dirs ,files in sorted(os.walk(os.path.join(out_folder,"PD\\Total"))):
        for file in files:
            if file.endswith("PD_Total.tif"):
                pd_total_filelist.append(os.path.join(root, file))
    del root, dirs, files
    for root,dirs ,files in sorted(os.walk(os.path.join(out_folder,"PD\\Ground"))):
        for file in files:
            if file.endswith("PD_Ground.tif"):
                pd_ground_filelist.append(os.path.join(root, file))
    del root, dirs, files
    args_list=[]


    if radius_fr_CHM:
        # Multiprocessing LPI_eLAI polygon
        # for i in range(0,len(args_list)):
        #     lpi_lai(args_list[i])
        if len(pd_total_filelist) == len(pd_ground_filelist) == len(chm_filelist):
            for i in range(0, len(pd_total_filelist)):
                chm_filename = os.path.splitext(os.path.split(chm_filelist[i])[1])[0]
                pdtotal_filename = os.path.splitext(os.path.split(pd_total_filelist[i])[1])[0]
                pdGround_filename = os.path.splitext(os.path.split(pd_ground_filelist[i])[1])[0]
                result_list = []
                if chm_filename[0:-4] == pdtotal_filename[0:-9] == pdGround_filename[0:-10]:
                    result_list.append(chm_filelist[i])
                    result_list.append(pd_total_filelist[i])
                    result_list.append(pd_ground_filelist[i])
                    result_list.append(out_folder)
                    result_list.append(chm_filename[0:-4])
                    result_list.append(mean_scanning_angle)
                    result_list.append(cell_size)
                    args_list.append(result_list)

        try:
            total_steps = len(args_list)
            features = []
            with Pool(processes=int(processes)) as pool:
                step = 0
                # execute tasks in order, process results out of order
                for result in pool.imap_unordered(lpi_lai, args_list):
                    if BT_DEBUGGING:
                        print('Got result: {}'.format(result), flush=True)
                    features.append(result)
                    step += 1
                    print('%{}'.format(step / total_steps * 100))

        except OperationCancelledException:
            print("Operation cancelled")
            exit()
    else:
        # Multiprocessing LPI_eLAI polygon
        # for i in range(0,len(args_list)):
        #     lpi_lai_with_focalR(args_list[i])
        if len(pd_total_filelist) == len(pd_ground_filelist):
            for i in range(0, len(pd_total_filelist)):

                pdtotal_filename = os.path.splitext(os.path.split(pd_total_filelist[i])[1])[0]
                pdGround_filename = os.path.splitext(os.path.split(pd_ground_filelist[i])[1])[0]
                result_list = []
                if pdtotal_filename[0:-9] == pdGround_filename[0:-10]:
                    result_list.append(pd_total_filelist[i])
                    result_list.append(pd_ground_filelist[i])
                    result_list.append(out_folder)
                    result_list.append(pdtotal_filename[0:-9])
                    result_list.append(mean_scanning_angle)
                    result_list.append(cell_size)
                    result_list.append(focal_radius)
                    args_list.append(result_list)
        try:
            total_steps = len(args_list)
            features = []
            with Pool(processes=int(processes)) as pool:
                step = 0
                # execute tasks in order, process results out of order
                for result in pool.imap_unordered(lpi_lai_with_focalR, args_list):
                    if BT_DEBUGGING:
                        print('Got result: {}'.format(result), flush=True)
                    features.append(result)
                    step += 1
                    print('%{}'.format(step / total_steps * 100))

        except OperationCancelledException:
            print("Operation cancelled")
            exit()
    import shutil
    shutil.rmtree(cache_folder,ignore_errors=True)
    if not radius_fr_CHM:
      os.remove(os.path.split(chm_folder)[0])

if __name__ == '__main__':
    start_time = time.time()
    print('Starting generate LPA and eLAI raster processing\n @ {}'
          .format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    r=robjects.r
    utils = importr('utils')
    base = importr('base')
    utils.chooseCRANmirror(ind=12) # select the 12th mirror in the list: Canada

    CRANpacknames = ['lidR','rlas','future','terra','comprehenr','na.tools','sf','sp']#,'devtools','fasterRaster','na.tools','snow','terra']
    CRANnames_to_install = [x for x in CRANpacknames if not robjects.packages.isinstalled(x)]
    if len(CRANnames_to_install) > 0:
        utils.install_packages(StrVector(CRANnames_to_install))


    del CRANpacknames,CRANnames_to_install

    in_args, in_verbose = check_arguments()
    print("loading R packages....")
    # utils = importr('utils')
    # base = importr('base')
    na = importr('na.tools')
    terra = importr('terra')
    lidR = importr('lidR')
    sf = importr('sf')
    sp = importr('sp')
    future=importr('future')

    print("Checking input parameters....")
    aoi_shapefile=in_args.input['in_polygon_file']
    in_las_folder=in_args.input['in_las_folder']
    cut_ht=float(in_args.input['cut_ht'])
    radius_fr_CHM=in_args.input['radius_fr_CHM']
    focal_radius=float(in_args.input['focal_radius'])
    pulse_density=int(in_args.input['pulse_density'])
    cell_size=float(in_args.input['cell_size'])
    mean_scanning_angle=float(in_args.input['mean_scanning_angle'])
    out_folder=in_args.input['out_folder']

    # if optional shapefile is empty, then do nothing, else verify shapefile
    if not aoi_shapefile=="":
        if not os.path.exists(os.path.dirname(aoi_shapefile)):
            print("Can't locate the input polygon folder.  Please check.")
            exit()
        else:
            if not isinstance(geopandas.GeoDataFrame.from_file(aoi_shapefile), geopandas.GeoDataFrame):
                print("Error input file: Please check effective LiDAR data extend shapefile")
                exit()
    # check existence of input las/laz folder
    if not os.path.exists(os.path.dirname(in_las_folder)):
        print("Can't locate the input las/laz folder.  Please check.")
        exit()

    #if doing focal radius divided from point cloud CHM
    if radius_fr_CHM==True:
        pass
        #do nothing for now
    else:
        # check manual input for radius, check input
        if not isinstance(focal_radius, float) and focal_radius>0.0:
            print( "Invalid search radius!!Default radius will be adopted (10m).")
            in_args.input['focal_radius']=10.0
        else:
            in_args.input['focal_radius'] = focal_radius
        # check manual input for cell size and pulse density
        if not isinstance(cell_size, float) and cell_size > 0.00:
            if not isinstance(pulse_density, int) and pulse_density <= 0.00:
                print("Invalid cell size and average pulse density provided.\n Default cell will size be adopted (1m).")
                in_args.input['cell_size'] = 1.0
                in_args.input['pulse_density'] = 0
            else:
                # mean_pd = (((math.pow(3 / pulse_density, 1 / 2)) + (math.pow(5 / pulse_density, 1 / 2))) / 2)
                # mean_pd = math.pow(3 / pulse_density, 1 / 2)
                mean_pd =   math.pow(5 / pulse_density, 1 / 2)
                in_args.input['cell_size'] = round(0.05 * round(mean_pd / 0.05), 2)
                in_args.input['pulse_density'] = pulse_density
        else:
            in_args.input['cell_size'] = (cell_size)
            in_args.input['pulse_density'] = pulse_density

    # Check manual input for cutt off height
    if not isinstance(cut_ht, float) and cut_ht > 0.0:
        print("Invalid cut off height!! Default cut off height will be adopted (1m).")
        in_args.input["cut_ht"] = 1.0
    else:
        in_args.input["cut_ht"] = cut_ht


    if not isinstance(cell_size, float) and cell_size > 0.00:
        if not isinstance(pulse_density, int) and pulse_density <= 0.00:
            print("Invalid cell size and average pulse density provided.\n Default cell will size be adopted (1m).")
            in_args.input['cell_size']= 1.0
            in_args.input['pulse_density']=0
        else:
            # mean_pd = (((math.pow(3 / pulse_density, 1 / 2)) + (math.pow(5 / pulse_density, 1 / 2))) / 2)
            # mean_pd = math.pow(3 / pulse_density, 1 / 2)
            mean_pd = math.pow(5 / pulse_density, 1 / 2)
            in_args.input['cell_size']= round(0.05 * round(mean_pd / 0.05), 2)
            in_args.input['pulse_density'] = pulse_density
    else:
        if not isinstance(pulse_density, int) and pulse_density <= 0.00:
            print("Invalid cell size and average pulse density provided.\n Default cell will size be adopted (1m).")
            in_args.input['cell_size'] = 1.0
            in_args.input['pulse_density'] = 0
        else:
            # mean_pd = (((math.pow(3 / pulse_density, 1 / 2)) + (math.pow(5 / pulse_density, 1 / 2))) / 2)
            # mean_pd = math.pow(3 / pulse_density, 1 / 2)
            mean_pd = math.pow(5 / pulse_density, 1 / 2)
            in_args.input['cell_size'] = round(0.05 * round(mean_pd / 0.05), 2)
            in_args.input['pulse_density'] = pulse_density


    if not isinstance(mean_scanning_angle, float) and mean_scanning_angle > 0.00:
        print("Invalid sensor scanning angle.\n Default sensor scanning angle will size be adopted (30 degree).")
        in_args.input['mean_scanning_angle'] =30.0
    else:
        in_args.input['mean_scanning_angle']=mean_scanning_angle

    print("Checking input parameters....Done")

    pd_raster(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)



    print('Generate LPA and eLAI rasters are done in {} seconds)'
          .format(round(time.time() - start_time, 5)))