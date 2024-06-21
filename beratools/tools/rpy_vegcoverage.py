import os
import time
from common import *
from r_interface import *

check_r_env()


def veg_cover_percentage(callback, in_las_folder, is_normalized, out_folder, hmin, hmax, cell_size, processes, verbose):
    rprocesses = r_processes(processes)

    # assign R script file to local variable
    Beratools_R_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Beratools_r_script.r')
    # Defining the R script and loading the instance in Python
    r['source'](Beratools_R_script)
    # Loading the function defined in R script.
    r_veg_metrics = robjects.globalenv['veg_cover_percentage']
    # Invoking the R function
    r_veg_metrics(in_las_folder, is_normalized, out_folder, hmin, hmax, cell_size, rprocesses)


if __name__ == '__main__':
    start_time = time.time()
    print('Finding Vegetation Coverage from Lidar data process\n @ {}'
          .format(time.strftime("%d %b %Y %H:%M:%S", time.localtime())))

    print("Checking input parameters ...")
    in_args, in_verbose = check_arguments()
    in_las_folder = in_args.input["in_las_folder"]

    try:
        hmin = float(in_args.input["hmin"])
        if hmin < 0:
            raise ValueError
        else:
            in_args.input["hmin"] = hmin
    except ValueError:
        print("Invalid input of minimum height of a tree, default value is used")
        in_args.input["hmin"] = 3.0

    try:
        hmax = float(in_args.input["hmax"])
        if hmax <= hmin:
            raise ValueError
        else:
            in_args.input["hmax"] = hmax
    except ValueError:
        print("Invalid input of maximum height of a tree, default value is used")
        if hmin < 10:
            in_args.input["hmax"] = 10.0
        else:
            in_args.input["hmin"] = 3.0
            in_args.input["hmax"] = 10.0

    try:
        is_normalized = bool(in_args.input["is_normalized"])

    except ValueError:
        print("Invalid input of checking normalized data box, normalize data will be carried")
        in_args.input["is_normalized"] = False

    try:
        cell_size = float(in_args.input["cell_size"])
        in_args.input["cell_size"] = cell_size
    except ValueError:
        print("Invalid input of cell size, default value is used")
        in_args.input["cell_size"] = 5.0

    out_folder = in_args.input["out_folder"]

    if not os.path.exists(in_las_folder):
        print("Error! Cannot locate Las folder, please check.")
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

    if not os.path.exists(out_folder):
        print("Warning! Cannot locate output folder, It will be created.")
        os.makedirs(out_folder)

    print("Checking input parameters ... Done")

    veg_cover_percentage(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)

    print('Finding Vegetation Coverage from Lidar data process is done in {} seconds)'
          .format(round(time.time() - start_time, 5)))
