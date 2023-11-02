import os
import time
from multiprocessing.pool import Pool

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

def chm2trees(callback,in_chm_folder,Min_ws,hmin,out_folder, processes, verbose):
    r = robjects.r
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

    # in_folder = in_chm_folder
    # in_folder=in_folder.replace("\\","/")
    out_folder=out_folder.replace("\\","/")

    # assign R script file to local variable
    Beratools_R_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Beratools_r_script.r')
    # Defining the R script and loading the instance in Python
    r['source'](Beratools_R_script)
    # Loading the function defined in R script.
    r_chm2trees = robjects.globalenv['chm2trees']

    args_list = []
    for root, dirs, files in sorted(os.walk(in_chm_folder)):
        for file in files:
            if file.endswith(".tif"):
                result = []
                chm_file=os.path.join(root, file)
                chm_file=chm_file.replace("\\", "/")
                result.append(chm_file)
                result.append(Min_ws)
                result.append(hmin)
                result.append(out_folder)
                result.append(rprocesses)
                args_list.append(result)
                # Invoking the R function
                r_chm2trees(chm_file, Min_ws, hmin, out_folder, rprocesses)

    del root, dirs, files


if __name__ == '__main__':
    start_time = time.time()
    print('Starting tree detection from CHM processing\n @ {}'
          .format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    r = robjects.r
    utils = importr('utils')
    base = importr('base')
    utils.chooseCRANmirror(ind=12)  # select the 12th mirror in the list: Canada
    print("Checking R packages....")
    CRANpacknames = ['lidR', 'rgrass', 'rlas', 'future',
                     'terra']  # ,'comprehenr','na.tools','sf','sp']#,'devtools','gdal']#,'fasterRaster']
    CRANnames_to_install = [x for x in CRANpacknames if not robjects.packages.isinstalled(x)]

    if len(CRANnames_to_install) > 0:
        utils.install_packages(StrVector(CRANnames_to_install))
        packages_found = True
    else:
        packages_found = True

    del CRANpacknames, CRANnames_to_install

    print("Checking input parameters....")
    in_args, in_verbose = check_arguments()
    in_chm_folder=in_args.input["in_chm_folder"]
    try:
        ws=float(in_args.input["Min_ws"])
        in_args.input["Min_ws"] = ws
    except ValueError:
        print("Invalid input of circular diameter, default is used")
        in_args.input["Min_ws"]=2.5
    try:
        hmin=float(in_args.input["hmin"])
        in_args.input["hmin"]=hmin
    except ValueError:
        print("Invalid input of minimum height of a tree, default is used")
        in_args.input["hmin"] =3.0

    out_folder=in_args.input["out_folder"]

    if not os.path.exists(in_chm_folder):
        print("Error! Cannot locate CHM raster folder, please check.")
        exit()
    else:
        found = False
        for files in os.listdir(in_chm_folder):
            if files.endswith(".tif") :
                    found=True
                    break
        if not found:
            print("Error! Cannot locate input CHM raster file(s), please check!")
            exit()

    if not os.path.exists(out_folder):
       print("Warning! Cannot locate output folder, It will be created.")
       os.makedirs(out_folder)

    print("Checking input parameters....Done")

    chm2trees(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)

    print('Tree detection from CHM raster data processing is done in {} seconds)'
          .format(round(time.time() - start_time, 5)))

