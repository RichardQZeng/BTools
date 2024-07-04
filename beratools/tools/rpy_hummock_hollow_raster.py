import math
import os
import time

from beratools.tools.common import *
from beratools.tools.r_interface import *

def hh_raster(callback, in_raster, Min_ws, lawn_range, cell_size, out_folder, processes, verbose):
    rprocesses = r_processes(processes)

    in_raster = in_raster.replace("\\", "/")
    out_folder = out_folder.replace("\\", "/")

    # assign R script file to local variable
    Beratools_R_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Beratools_r_script.r')
    # Defining the R script and loading the instance in Python
    r['source'](Beratools_R_script)
    # Loading the function defined in R script.
    r_hh_function = robjects.globalenv['hh_function_byraster']

    # Invoking the R function
    r_hh_function(in_raster, cell_size, Min_ws, lawn_range, out_folder, rprocesses)


if __name__ == '__main__':
    start_time = time.time()
    print('Hummock and Hollow detection from DTM raster process.\n'
          '@ {}'.format(time.strftime("%d %b %Y %H:%M:%S", time.localtime())))

    packages = ['terra']  # ,'comprehenr','na.tools','sf','sp']#,'devtools','gdal']#,'fasterRaster']
    check_r_packages_installation(packages)

    print("Checking input parameters ...")
    in_args, in_verbose = check_arguments()
    in_raster = in_args.input["in_raster"]
    try:
        cell_size = float(in_args.input["cell_size"])
        in_args.input["cell_size"] = cell_size
    except ValueError:
        print("Invalid input of cell_size, default value will be used")
        in_args.input["cell_size"] = 1.0

    try:
        ws = float(in_args.input["Min_ws"])
        in_args.input["Min_ws"] = ws
    except ValueError:
        print("Invalid input of circular diameter, default value will be used")
        in_args.input["Min_ws"] = 3.0
    try:
        lawn_range = float(in_args.input["lawn_range"])
        in_args.input["lawn_range"] = lawn_range
    except ValueError:
        print("Invalid input of range height of lawns, default value will be used")
        in_args.input["lawn_range"] = 0.1

    out_folder = in_args.input["out_folder"]

    if not os.path.isfile(in_raster):
        print("Error! Cannot locate raster folder, please check.")
        exit()

    if not os.path.exists(out_folder):
        print("Warning! Cannot locate output folder, It will be created.")
        os.makedirs(out_folder)

    print("Checking input parameters ... Done")

    hh_raster(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)

    print('Hummock and Hollow detection from DTM raster process is done in {} seconds)'
          .format(round(time.time() - start_time, 5)))
