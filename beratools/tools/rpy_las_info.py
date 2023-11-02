import os
import time
from common import *
class OperationCancelledException(Exception):
    pass

try: # integrated R env
    #check R language within env
    current_env_path= os.environ['CONDA_PREFIX']
    # if os.path.isdir(current_env_path):
    os.environ['R_HOME'] =os.path.join(current_env_path,r"Lib\R")
    os.environ['R_USER'] = os.path.expanduser('~')
    os.environ['R_LIBS_USER'] = os.path.join(current_env_path,r"Lib\R\library")

except FileNotFoundError:
    print("Warning: Please install R for this process!!")
    exit()

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, data
from rpy2.robjects.vectors import StrVector

def las_info(callback,in_las_folder,processes,verbose):
    r = robjects.r
    base = importr('base')
    import psutil
    stats = psutil.virtual_memory()  # returns a named tuple
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

    in_las_folder = in_las_folder.replace("\\", "/")
    # assign R script file to local variable
    Beratools_R_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Beratools_r_script.r')
    # Defining the R script and loading the instance in Python
    r['source'](Beratools_R_script)
    # Loading the function defined in R script.
    r_normalized_lidar = robjects.globalenv['las_info']

    # Invoking the R function
    r_normalized_lidar(in_las_folder,rprocesses)



if __name__ == '__main__':
    start_time = time.time()
    print('LiDAR data information..\n @ {}'
          .format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    r=robjects.r
    utils = importr('utils')
    base = importr('base')
    utils.chooseCRANmirror(ind=12) # select the 12th mirror in the list: Canada
    print("Checking R packages....")
    CRANpacknames = ['lidR','future'] # ,'comprehenr','na.tools','sf','sp']#,'devtools','gdal']#,'fasterRaster']
    CRANnames_to_install = [x for x in CRANpacknames if not robjects.packages.isinstalled(x)]

    if len(CRANnames_to_install) > 0:
       utils.install_packages(StrVector(CRANnames_to_install))
       packages_found=True
    else:
        packages_found= True

    # if packages_found:
    #    utils.update_packages(checkBuilt = True, ask=False)

    del CRANpacknames,CRANnames_to_install

    print("Checking input parameters....")
    in_args, in_verbose = check_arguments()

    in_las_folder=in_args.input["in_las_folder"]

    if not os.path.exists(in_las_folder):
        print("Error! Cannot locate Las folder, please check.")
        exit()
    else:
        found = False
        for files in os.listdir(in_las_folder):
            if files.endswith(".las") or files.endswith(".laz"):
                    found=True
                    break
        if not found:
            print("Error! Cannot locate input LAS file(s), please check!")
            exit()


    print("Checking input parameters....Done")

    las_info(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)

    print('Display LiDAR data information is done in {} seconds)'
          .format(round(time.time() - start_time, 5)))

