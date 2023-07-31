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

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, data
from rpy2.robjects.vectors import StrVector

def lpi_lai(arg):
    pdTotal=arg[0]
    pdGround=arg[1]
    out_folder=arg[2]
    filename=arg[3]
    scan_angle=float(arg[4])

    ##variable for not calling R
    cell_size =float(arg[5])
    radius = float(arg[6])
    tfocal_filename = filename + "_tfocal_py.tif"
    gfocal_filename = filename + "_gfocal_py.tif"
    out_tfocal = os.path.join(out_folder, tfocal_filename)
    out_gfocal = os.path.join(out_folder, gfocal_filename)

    ## output files variables
    out_lpi_fielname = filename + "_LPI_py.tif"
    out_elai_fielname = filename + "_eLAI_py.tif"
    LPI_folder = os.path.join(out_folder, "LPI")
    eLAI_folder = os.path.join(out_folder, "eLAI")
    out_lpi= os.path.join(LPI_folder, out_lpi_fielname)
    out_elai= os.path.join(eLAI_folder, out_elai_fielname)

    # Working out the searching radius
    # with rasterio.open(chm) as image:
    #     ndarray = image.read(1)
    #     ndarray[ndarray==image.nodata]=numpy.NaN
    #     ndarray[ndarray <0.0] = numpy.NaN
    #     radius = math.ceil(numpy.nanmean(ndarray) * 2)

    print("Calculating LPI and eLAI for {}....".format(filename))
    with rasterio.open(pdTotal) as pd_total:
        with rasterio.open(pdGround) as pd_Ground:
            raster_profile=pd_total.profile
            pd_total_ndarray = pd_total.read(1,boundless=True )
            nodata= pd_total.nodata
            pd_total_ndarray[pd_total_ndarray==nodata]=numpy.nan
            kernel = convolution.circle_kernel(cell_size, cell_size,radius)
            total_focalsum=fs_raster_stdmean(pd_total_ndarray, kernel, nodata)
            write_total_focalsum = rasterio.open(tfocal_filename, 'w', **raster_profile)
            write_total_focalsum.write(total_focalsum, 1)
            write_total_focalsum.close()
            del write_total_focalsum

            pd_ground_ndarray = pd_Ground.read(1,boundless=True)
            nodata = pd_Ground.nodata
            pd_ground_ndarray[pd_ground_ndarray == nodata] = numpy.nan
            ground_focalsum = fs_raster_stdmean(pd_ground_ndarray, kernel, nodata)
            write_ground_focalsum = rasterio.open(gfocal_filename, 'w', **raster_profile)
            write_ground_focalsum.write(ground_focalsum, 1)
            write_ground_focalsum.close()
            del write_ground_focalsum
    del pd_total


    del pd_Ground
    lpi_array = numpy.divide(pd_ground_ndarray, pd_total_ndarray, out=numpy.zeros_like(pd_ground_ndarray), where=pd_total_ndarray != 0)

    print("Calculating LPI: {}....".format(filename))
    write_lpi = rasterio.open(out_lpi, 'w', **raster_profile)
    write_lpi.write(lpi_array, 1)
    write_lpi.close()
    del write_lpi
    print('%{}'.format(80))
    print("Calculating LPI: {}....Done".format(filename))

    print("Calculating eLAI: {}....".format(filename))
    elai_array = ((math.cos(((scan_angle / 2.0) / 180.0) * math.pi)) / 0.5) * (numpy.log(lpi_array)) * -1

    write_elai = rasterio.open(out_elai, 'w', **raster_profile)
    write_elai.write(elai_array, 1)
    write_elai.close()
    del write_elai
    print("Calculating eLAI: {}....Done".format(filename))

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
    terra = importr('terra')
    # chm=arg[0]
    pdTotal=arg[0]
    pdGround=arg[1]
    out_folder=arg[2]
    filename=arg[3]
    scan_angle=float(arg[4])


    ##variable for not calling R
    # cell_size =float(arg[5])
    tfocal_filename = filename + "_tfocal_r.tif"
    gfocal_filename = filename + "_gfocal_R.tif"
    # out_tfocal = os.path.join(out_folder, tfocal_filename)
    # out_gfocal = os.path.join(out_folder, gfocal_filename)

    ## output files variables
    out_lpi_fielname = filename + "_LPI_r.tif"
    out_elai_fielname = filename + "_eLAI_r.tif"
    LPI_folder = os.path.join(out_folder, "LPI")
    eLAI_folder = os.path.join(out_folder, "eLAI")
    out_lpi = os.path.join(LPI_folder, out_lpi_fielname)
    out_elai = os.path.join(eLAI_folder, out_elai_fielname)

    radius = float(arg[6])
    print("Calculating LPI and eLAI for {}....".format(filename))
    r(''' #create a 'rlpi_elai' function
    library(terra)
    library(sf)
    library(sp)
    rlpi_elai <- function(pdTotal,pdGround,radius,scan_angle,out_lpi,out_elai){
    
    total_focal <- rast(pdTotal)
    ground_focal <- rast(pdGround)

    # pdTotal_SpatRaster <- rast(pdTotal)
    # pdGround_SpatRaster <- rast(pdGround)
    ground_focal <- extend(ground_focal, ext(total_focal))
    
    # lpi
    lpi = ground_focal / total_focal
    #lpi
    lpi[is.infinite(lpi)] = NA

    elai = -cos(((scan_angle / 2.0)/180)*pi)/0.5 * log(lpi)
    elai[is.infinite(elai)] = NA
    elai[elai==0 | elai==-0 ] = 0
    
    writeRaster(lpi,out_lpi,overwrite=TRUE)
    writeRaster(elai,out_elai,overwrite=TRUE)

    }''')

    rlpi_elai = r['rlpi_elai']
    rlpi_elai(pdTotal,pdGround,radius,scan_angle,out_lpi,out_elai)

    print("Calculating LPI adn eLAI: {}....Done".format(filename))

def find_pulse_density(ctg,out_folder ):
    r = robjects.r

    cache_folder = os.path.join(out_folder, "Cache")

    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    r('''
       routine <- function(chunk){ 
      las <- readLAS(chunk)
      if (is.empty(las)) return(NULL)
      las <- retrieve_pulses(las)
      pulse<-rasterize_density(las, res=1,pkg="terra")
      pulse_density<-global(pulse,"mean",na.rm=TRUE)$mean
      # output is a list
      return(pulse_density)
       }
       ''')
    routine = r['routine']
    list_pd = r.catalog_apply(ctg, routine)
    point_density,pulse_density=numpy.nanmean(numpy.array(list_pd),axis=0)

    # mean_pd = (((math.pow(3 / pulse_density, 1 / 2)) + (math.pow(5 / pulse_density, 1 / 2))) / 2)
    mean_pd = math.pow(3 / pulse_density, 1 / 2)
    # mean_pd = math.pow(5 / pulse_density, 1 / 2)
    cell_size = round(0.05 * round(mean_pd / 0.05), 2)

    return (cell_size)


def pd_raster(callback, in_polygon_file, in_las_folder, cut_ht, radius_fr_CHM, focal_radius, pulse_density,
              cell_size, mean_scanning_angle, out_folder, processes, verbose):

    r = robjects.r

    import psutil
    stats = psutil.virtual_memory()  # returns a named tuple
    available = getattr(stats, 'available')/1024000000
    if available<= 50:
        rprocesses=4
    elif 50< available<= 150:
        rprocesses=8
    elif 150< available<= 250:
        rprocesses=12
    else:
        rprocesses = 4


    r.plan(r.multisession,workers=rprocesses)
    r.set_lidr_threads(math.ceil(rprocesses))


    cache_folder=os.path.join(out_folder,"Cache")
    # dtm_folder=os.path.join(out_folder,"DTM")
    # dsm_folder=os.path.join(out_folder,"DSM")
    # chm_folder=os.path.join(out_folder,"CHM")
    PD_folder=os.path.join(out_folder,"PD")
    PD_Total_folder=os.path.join(PD_folder,"Total")
    PD_Ground_folder = os.path.join(PD_folder, "Ground")
    LPI_folder = os.path.join(out_folder, "LPI")
    eLAI_folder = os.path.join(out_folder, "eLAI")



    if not os.path.exists(cache_folder):
        os.makedirs(cache_folder)

    # if not os.path.exists(dtm_folder):
    #     os.makedirs(dtm_folder)

    # if not os.path.exists(chm_folder):
    #     os.makedirs(chm_folder)
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

    lascat = r.readLAScatalog(in_las_folder,filter= "-drop_class 7")
    pulse_density = find_pulse_density(lascat,out_folder)
    cache_folder = cache_folder.replace("\\","/")
    # dtm_folder = dtm_folder.replace("\\", "/") + "/{*}_dtm"

    # chm_folder = chm_folder.replace("\\","/")
    PD_Total_folder = PD_folder.replace("\\","/")+"/Total"
    PD_Ground_folder = PD_folder.replace("\\", "/") + "/Ground"
    LPI_folder = LPI_folder.replace("\\", "/")
    eLAI_folder = eLAI_folder.replace("\\", "/")

    if not in_polygon_file=="":
        r.vect(in_polygon_file)

    if cell_size<=0:
        if pulse_density<=0:
            cell_size=pulse_density(lascat,out_folder,processes, verbose)

##############################################################
    r('''#create a 'generate_pd' function
    generate_pd <- function(ctg,radius_fr_CHM,focal_radius,cell_size,cache_folder,
    cut_ht,PD_Ground_folder,PD_Total_folder){
    library(terra)
    library(lidR)

    opts <- paste0("-drop_class 7")

    print("Processing using R packages.")
   

    # routine <- function(chunk)
    # {
    # las <- readLAS(chunk)
    # if (is.empty(las)) return(NULL)
    # las <- retrieve_pulses(las)
    # las <- filter_poi(las,buffer==0)
    # }
    # opt <- list(need_output_file =TRUE, autocrop = TRUE)
    # opt_laz_compression(ctg) <- TRUE
    # opt_chunk_alignment(ctg) <- c(0,0)
    # opt_output_files(ctg) <- paste0(cache_folder,"/{*}")
    # opt_stop_early(ctg) <- FALSE
    # catalog_apply(ctg, routine,.options=opt)

    #load normalized LAS with pulse info (debug only)
    # ctg2<- readLAScatalog( paste0(cache_folder,"/nlidar"))

    # print("Normalize point cloud using K-nearest neighbour IDW....")
    folder <- paste0(cache_folder,"/nlidar/n_{*}" )
    opt_output_files(ctg) <- opt_output_files(ctg) <- folder
    opt_laz_compression(ctg) <- TRUE
    opt_chunk_alignment(ctg) <- c(0,0)

    #normalized LAS with pulse info
    normalize_height(ctg, algorithm=knnidw())
    
    print("Generate pulse density (total focal sum) raster....")

    pd_total <- function(chunk,radius,cell_size)
    {
    las <- readLAS(chunk)
    if (is.empty(las)) return(NULL)

    las_1 <- filter_poi(readLAS(chunk), buffer==0)
    # hull <- st_convex_hull(las_1)
    bbox <- ext(las_1)

    # convert to SpatialPolygons
    # bbox <- vect(hull)

    las <- filter_poi(las, Classification != 7L)
    las <- retrieve_pulses(las)
    density_raster_total <- rasterize_density(las, res=cell_size,pkg="terra")[[2]]
    
    tfw <- focalMat(density_raster_total, radius, "circle")

    tfw[tfw>0] = 1
    tfw[tfw==0] = NA

    Total_focal <- focal(density_raster_total, w=tfw, fun="sum", na.rm=TRUE,na.policy="omit",fillvalue=NA,expand=FALSE)
    Total_focal <- crop(Total_focal,bbox)
    }
    opt <- list(need_output_file =TRUE, autocrop = TRUE)
    opt_chunk_alignment(ctg) <- c(0,0)
    ctg@output_options$drivers$SpatRaster$param$overwrite <- TRUE
    # opt_output_files(ctg) <- paste0(PD_Total_folder,"/{*}_PD_Total")
    opt_output_files(ctg) <- paste0(PD_Total_folder,"/{*}_PD_Tfocalsum")
    opt_stop_early(ctg) <- FALSE
    catalog_apply(ctg, pd_total,radius=focal_radius,cell_size=cell_size,.options=opt)

    #load normalized LAS
    ht<- paste0("-drop_class 7 -drop_z_above ",cut_ht)
    ctg2<- readLAScatalog( paste0(cache_folder,"/nlidar"), filter = ht)


    print("Generate pulse density (ground focal sum) raster....")
    pd_ground <- function(chunk,radius,cell_size,cut_ht)
    {
    las <- readLAS(chunk)
    if (is.empty(las)) return(NULL)

    las_1 <- filter_poi(readLAS(chunk), buffer==0)
    # hull <- st_convex_hull(las_1)

    # convert to SpatialPolygons
    # bbox <- vect(hull)
    bbox <- ext(las_1)

    las <- retrieve_pulses(las)
    density_raster_ground <- rasterize_density(las, res=cell_size,pkg="terra")[[2]]
    

    gfw <- focalMat(density_raster_ground, radius, "circle")
    gfw[gfw>0] = 1
    gfw[gfw==0] = NA

    Ground_focal <- focal(density_raster_ground, w=gfw, fun="sum",na.policy="omit",na.rm=TRUE,fillvalue=NA,expand=FALSE)
    ground_focal <- crop(Ground_focal,bbox)

    }
    opt <- list(need_output_file =TRUE, autocrop = TRUE)
    opt_chunk_alignment(ctg2) <- c(0,0)
    ctg2@output_options$drivers$SpatRaster$param$overwrite <- TRUE
    # opt_output_files(ctg2) <- paste0(PD_Ground_folder,"/{*}_PD_Ground")
    opt_output_files(ctg2) <- paste0(PD_Ground_folder,"/{*}_PD_Gfocalsum")
    opt_stop_early(ctg2) <- FALSE
    catalog_apply(ctg2, pd_ground,radius=focal_radius,cell_size=cell_size,cut_ht=cut_ht,.options=opt)

     }''')

    generate_pd = r['generate_pd']

    generate_pd(lascat, radius_fr_CHM, focal_radius, cell_size, cache_folder, cut_ht, PD_Ground_folder,
                    PD_Total_folder)#dtm_folder, chm_folder,

###########################################################
#fasterRaster
    #
    # r('''#create a 'fasterRaster_pd' function
    # fasterRaster_pd <- function(ctg,radius_fr_CHM,focal_radius,cell_size,cache_folder,
    # dtm_folder,chm_folder,cut_ht,PD_Ground_folder,PD_Total_folder,rprocesses){
    # library(terra)
    # library(rgrass)
    # library(fasterRaster)
    # library(lidR)
    #
    #
    # opts <- paste0("-drop_class 7")
    #
    # print("Processing using R packages.")
    # print("Retrieving pulses information....")
    #
    # routine <- function(chunk)
    # {
    # las <- readLAS(chunk)
    # if (is.empty(las)) return(NULL)
    # las <- retrieve_pulses(las)
    # las <- filter_poi(las,buffer==0)
    # }
    # opt <- list(need_output_file =TRUE, autocrop = TRUE)
    # opt_laz_compression(ctg) <- TRUE
    # opt_chunk_alignment(ctg) <- c(0,0)
    # opt_output_files(ctg) <- paste0(cache_folder,"/{*}")
    # opt_stop_early(ctg) <- FALSE
    # catalog_apply(ctg, routine,.options=opt)
    #
    #
    # #load normalized LAS with pulse info (debug only)
    # # ctg2<- readLAScatalog( paste0(cache_folder,"/nlidar"))
    #
    # print("Normalize point cloud using K-nearest neighbour IDW....")
    # folder <- paste0(cache_folder,"/nlidar/n_{*}" )
    # opt_output_files(ctg) <- opt_output_files(ctg) <- folder
    # opt_laz_compression(ctg) <- TRUE
    # # opt_filter(ctg) <- opts
    # opt_chunk_alignment(ctg) <- c(0,0)
    #
    # #normalized LAS with pulse info
    # normalize_height(ctg, algorithm=knnidw())
    #
    # #load LAS with pulse info
    # ctg <- readLAScatalog(cache_folder,filter=opts)
    # print("Generate pulse density (total focal sum) raster....")
    #
    # pd_total <- function(chunk,radius,cell_size)
    # {
    # las <- readLAS(chunk)
    # if (is.empty(las)) return(NULL)
    #
    # las_1 <- filter_poi(readLAS(chunk), buffer==0)
    # hull <- st_convex_hull(las_1)
    #
    #
    # # convert to SpatialPolygons
    # bbox <- vect(hull)
    #
    # las <- filter_poi(las, Classification != 7L)
    # density_raster_total <- rasterize_density(las, res=cell_size)#,pkg="terra")
    # # density_raster_total <- mask(density_raster_total,bbox,touches=FALSE)
    #
    # # tfw <- focalMat(density_raster_total, radius, "circle")
    # #
    # # tfw[tfw>0] = 1
    # # tfw[tfw==0] = NA
    # #
    # # Total_focal <- fasterFocal(density_raster_total, w=tfw, fun=sum, na.rm=TRUE,cores=rprocesses, forceMulti=TRUE)
    # # total_focal <- mask(as(Total_focal,"SpatRaster"),bbox,touches=FALSE)
    # }
    # opt <- list(need_output_file =TRUE, autocrop = TRUE)
    # opt_chunk_alignment(ctg) <- c(0,0)
    # ctg@output_options$drivers$SpatRaster$param$overwrite <- TRUE
    # opt_output_files(ctg) <- paste0(PD_Total_folder,"/{*}_PD_Total")
    # # opt_output_files(ctg) <- paste0(PD_Total_folder,"/{*}_PD_Tfocalsum")
    # opt_stop_early(ctg) <- FALSE
    # catalog_apply(ctg, pd_total,radius=focal_radius,cell_size=cell_size,.options=opt)
    #
    #
    # ht<- paste0("-drop_class 7 -drop_z_above ",cut_ht)
    # ctg2<- readLAScatalog( paste0(cache_folder,"/nlidar"), filter = ht)
    #
    #
    # print("Generate pulse density (ground focal sum) raster....")
    # pd_ground <- function(chunk,radius,cell_size,cut_ht)
    # {
    # las <- readLAS(chunk)
    # if (is.empty(las)) return(NULL)
    #
    # las_1 <- filter_poi(readLAS(chunk), buffer==0)
    # hull <- st_convex_hull(las_1)
    #
    # # convert to SpatialPolygons
    # bbox <- vect(hull)
    #
    #
    # density_raster_ground <- rasterize_density(las, res=cell_size)#,pkg="terra")
    # density_raster_ground <- mask(density_raster_ground,bbox,touches=FALSE)
    #
    # # gfw <- focalMat(density_raster_ground, radius, "circle")
    # # gfw[gfw>0] = 1
    # # gfw[gfw==0] = NA
    # #
    # # Ground_focal <- fasterFocal(density_raster_ground, w=gfw, fun=sum,na.rm=TRUE,cores=rprocesses, forceMulti=TRUE)
    # # ground_focal <- mask(as(Ground_focal,"SpatRaster"),bbox,touches=FALSE)
    #
    # }
    # opt <- list(need_output_file =TRUE, autocrop = TRUE)
    # opt_chunk_alignment(ctg2) <- c(0,0)
    # ctg2@output_options$drivers$SpatRaster$param$overwrite <- TRUE
    # opt_output_files(ctg2) <- paste0(PD_Ground_folder,"/{*}_PD_Ground")
    # # opt_output_files(ctg2) <- paste0(PD_Ground_folder,"/{*}_PD_Gfocalsum")
    # opt_stop_early(ctg2) <- FALSE
    # catalog_apply(ctg2, pd_ground,radius=focal_radius,cell_size=cell_size,cut_ht=cut_ht,.options=opt)
    #
    #  }''')
    #
    # fasterRaster_pd = r['fasterRaster_pd']
    #
    # fasterRaster_pd(lascat, radius_fr_CHM, focal_radius, cell_size, cache_folder, dtm_folder, chm_folder, cut_ht,
    #             PD_Ground_folder,
    #             PD_Total_folder,rprocesses)
##############################################
#################################################################

    # generate_LPI(lascat, radius_fr_CHM, focal_radius, cell_size, cache_folder, dtm_folder, chm_folder, cut_ht, LPI_folder)
    # At this stage no process for CHM

    pd_total_filelist=[]
    pd_ground_filelist = []

    # Get raster files lists
    for root,dirs ,files in sorted(os.walk(os.path.join(out_folder,"PD\\Total"))):
        for file in files:
            if file.endswith("_PD_Tfocalsum.tif"):
                pd_total_filelist.append(os.path.join(root, file))
    del root, dirs, files
    for root,dirs ,files in sorted(os.walk(os.path.join(out_folder,"PD\\Ground"))):
        for file in files:
            if file.endswith("_PD_Gfocalsum.tif"):
                pd_ground_filelist.append(os.path.join(root, file))
    del root, dirs, files
    args_list = []

    # At this stage no process for finding average cell size from all CHM
    radius_fr_CHM = False
    if radius_fr_CHM:
        chm_filelist = []
        for root, dirs, files in sorted(os.walk(os.path.join(out_folder, "CHM"))):
            for file in files:
                if file.endswith("_chm.tif"):
                    chm_filelist.append(os.path.join(root, file))
        del root, dirs, files

        if len(pd_total_filelist) == len(pd_ground_filelist) == len(chm_filelist):
            for i in range(0, len(pd_total_filelist)):
                chm_filename = os.path.splitext(os.path.split(chm_filelist[i])[1])[0]
                pdtotal_filename = os.path.splitext(os.path.split(pd_total_filelist[i])[1])[0]
                pdGround_filename = os.path.splitext(os.path.split(pd_ground_filelist[i])[1])[0]
                result_list = []
                if chm_filename[0:-4] == pdtotal_filename[0:-13] == pdGround_filename[2:-13]:
                    result_list.append(chm_filelist[i])
                    result_list.append(pd_total_filelist[i])
                    result_list.append(pd_ground_filelist[i])
                    result_list.append(out_folder)
                    result_list.append(chm_filename[0:-4])
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

        for i in range(0,len(args_list)):
            r_lpi_lai_with_focalR(args_list[i])

    # import shutil
    # shutil.rmtree(cache_folder,ignore_errors=True)
    # if not radius_fr_CHM:
    #   os.removedirs(os.path.split(chm_folder)[0])

if __name__ == '__main__':
    start_time = time.time()
    print('Starting generate LPA and eLAI raster processing\n @ {}'
          .format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    r=robjects.r
    utils = importr('utils')
    base = importr('base')
    utils.chooseCRANmirror(ind=12) # select the 12th mirror in the list: Canada
    print("Checking and loading R packages....")
    CRANpacknames = ['lidR','rgrass','rlas','future','terra','comprehenr','na.tools','sf','sp','devtools','fasterRaster']
    CRANnames_to_install = [x for x in CRANpacknames if not robjects.packages.isinstalled(x)]
    need_fasterRaster = False
    if len(CRANnames_to_install) > 0:
        if not 'fasterRaster' in CRANnames_to_install:
            utils.install_packages(StrVector(CRANnames_to_install))
            need_fasterRaster = False
        else:
            CRANnames_to_install.remove('fasterRaster')
            need_fasterRaster=True
            if len(CRANnames_to_install) > 0:
                utils.install_packages(StrVector(CRANnames_to_install))

    if need_fasterRaster:
        devtools=importr('devtools')
        devtools.install_github("adamlilith/fasterRaster", dependencies=True)


    del CRANpacknames,CRANnames_to_install

    in_args, in_verbose = check_arguments()
    # loading R packages
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
        if not isinstance(focal_radius, float) or focal_radius<=0.0:
            print( "Invalid search radius!!Default radius will be adopted (10m).")
            in_args.input['focal_radius']=10.0
        else:
            in_args.input['focal_radius'] = focal_radius
        # check manual input for cell size and pulse density
        if not isinstance(cell_size, float) or cell_size <= 0.00:
            if not isinstance(pulse_density, int) or pulse_density <= 0.00:
                print("Invalid cell size and average pulse density provided.\n Default cell will size be adopted (1m).")
                in_args.input['cell_size'] = 1.0
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
        in_args.input['mean_scanning_angle'] =30.0
    else:
        in_args.input['mean_scanning_angle']=mean_scanning_angle

    print("Checking input parameters....Done")

    pd_raster(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)



    print('Generate LPA and eLAI rasters are done in {} seconds)'
          .format(round(time.time() - start_time, 5)))