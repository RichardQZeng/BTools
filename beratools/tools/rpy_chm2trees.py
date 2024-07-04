import os
import time
from multiprocessing.pool import Pool

from beratools.tools.common import *
from beratools.tools.r_interface import *


def chm2trees(callback, in_chm_folder, Min_ws, hmin, out_folder, processes, verbose):
    rprocesses = r_processes(processes)

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
                chm_file = os.path.join(root, file)
                chm_file = chm_file.replace("\\", "/")
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

    packages = ['lidR', 'rgrass', 'rlas', 'future', 'terra']
    check_r_packages_installation(packages)

    print("Checking input parameters....")
    in_args, in_verbose = check_arguments()
    in_chm_folder = in_args.input["in_chm_folder"]
    try:
        ws = float(in_args.input["Min_ws"])
        in_args.input["Min_ws"] = ws
    except ValueError:
        print("Invalid input of circular diameter, default is used")
        in_args.input["Min_ws"] = 2.5
    try:
        hmin = float(in_args.input["hmin"])
        in_args.input["hmin"] = hmin
    except ValueError:
        print("Invalid input of minimum height of a tree, default is used")
        in_args.input["hmin"] = 3.0

    out_folder = in_args.input["out_folder"]

    if not os.path.exists(in_chm_folder):
        print("Error! Cannot locate CHM raster folder, please check.")
        exit()
    else:
        found = False
        for files in os.listdir(in_chm_folder):
            if files.endswith(".tif"):
                found = True
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
