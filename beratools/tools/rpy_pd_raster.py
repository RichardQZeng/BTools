import math
import os
import winreg
import time
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

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, data
from rpy2.robjects.vectors import StrVector
def pd_raster(callback, in_polygon_file, in_las_file, point_density,radius_fr_CHM, cell_size, cut_ht, out_pd, out_pd_ground, processes, verbose):
    r = robjects.r


    file=os.path.split(in_las_file)[1]
    filename=os.path.splitext(file)[0]
    shp = r.vect(in_polygon_file)  # will drop z values
    print("Reading las/laz...")
    clean_las = r.readLAS(in_las_file,filter= "-drop_class 7")
    if isinstance(float(cell_size), float) and float(cell_size) >=0.01:
       pass
    else:
        #if input cell size < 0.01 then define cell size from 5 time of average point spacing
        # _clean_las = r.retrieve_pulses(clean_las)
        # result=r.rasterize_density(_clean_las,res=1)
        r('''
            #create a 'pd_fun' function
            pd_fun <- function(las){
            las <- retrieve_pulses(las)
            density <- rasterize_density(las,res=1)
            
            pud <- density[[2]] # pulse density
            pulse_data <- global(pud,"mean",na.rm=TRUE)
            pulse <- pulse_data$mean
           }''')
        pd_fun = r['pd_fun']
        pulse_density = pd_fun(clean_las)[0]

        #cell size = round to the nearest cm from 2 times point spacing (PD=1/PS^2) => PS= (1/PD)^(1/2)
        mean_pd=(((math.pow(3/pulse_density,1/2))+(math.pow(5/pulse_density,1/2)))/2)
        cell_size=round(0.05*round(mean_pd/0.05),2)

    print("Rasterize pulse density for all points...")
    clean_las=r.retrieve_pulses(clean_las)
    dens_raster_clean_las = r.rasterize_density(clean_las, res=float(cell_size))
    # if (r.relate(r.ext(shp),r.ext(clean_las),'overlaps')):
    #     ras_all=r.mask(dens_raster_clean_las,shp)
    #     r.writeRaster(ras_all,out_pd,overwrite=True)
    #     del ras_all
    # else:
    #     r.writeRaster(dens_raster_clean_las, out_pd, overwrite=True)
    r.writeRaster(dens_raster_clean_las, out_pd, overwrite=True)
    del dens_raster_clean_las

    # First return points, no noise, x meter ht above ground
    # normalize points on the fly using DTM from default idw
    print("Normalize point cloud...")
    nlas = r.normalize_height(clean_las, r.knnidw(k = 10, p = 2, rmax = 50))
    temp_nlas=os.path.join(os.path.dirname(out_pd),filename+"_nlas.las")
    r.writeLAS(nlas,temp_nlas)
    del nlas
    print("Rasterize pulse density from gournd to {}m ...".format(cut_ht))
    _nlas = r.readLAS(temp_nlas, filter="-drop_class 7 -drop_z_above " + cut_ht)
    nclean_las_1st=r.filter_first(_nlas)

    dens_raster_n1stlas = r.rasterize_density(nclean_las_1st, res=float(cell_size))
    # if (r.relate(r.ext(shp),r.ext(clean_las),'overlaps')):
    #     ras_n1stlas = r.mask(dens_raster_n1stlas, shp)
    #     r.writeRaster(ras_n1stlas, out_pd_ground, overwrite=True)
    # else:
    #     r.writeRaster(dens_raster_n1stlas, out_pd_ground, overwrite=True)
    r.writeRaster(dens_raster_n1stlas, out_pd_ground, overwrite=True)
    os.remove(temp_nlas)

if __name__ == '__main__':
    start_time = time.time()
    print('Starting generate pulse density raster processing\n @ {}'
          .format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    r=robjects.r
    utils = importr('utils')
    base = importr('base')
    utils.chooseCRANmirror(ind=12) # select the 12th mirror in the list: Canada

    CRANpacknames = ['lidR','terra','comprehenr','na.tools','sf','sp']#,'devtools','fasterRaster','na.tools','snow','terra']
    CRANnames_to_install = [x for x in CRANpacknames if not robjects.packages.isinstalled(x)]
    if len(CRANnames_to_install) > 0:
        utils.install_packages(StrVector(CRANnames_to_install))


    del CRANpacknames,CRANnames_to_install

    in_args, in_verbose = check_arguments()
    print("loading R packages....")
    utils = importr('utils')
    base = importr('base')
    # comprehenr = importr('comprehenr')
    na = importr('na.tools')
    terra = importr('terra')
    lidR = importr('lidR')
    # sf = importr('sf')
    # sp = importr('sp')


    import geopandas
    if not isinstance(geopandas.GeoDataFrame.from_file(in_args.input["in_polygon_file"]), geopandas.GeoDataFrame):
        print("Error input file: Please the effective LiDAR data extend shapefile")
        exit()

    if not os.path.isfile(in_args.input["in_las_file"]):
        print("Error input file: Please check input las/laz file.")
        exit()

    if not os.path.exists(os.path.dirname(in_args.input["out_pd"])):
        print("Error output file: Could not locate the output raster folder/path.")
        exit()

    if not os.path.exists(os.path.dirname(in_args.input["out_pd_ground"])):
        print("Error output file: Could not locate the output raster folder/path.")
        exit()

    pd_raster(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)



    print('Generate pulse density rasters are done in {} seconds)'
          .format(round(time.time() - start_time, 5)))