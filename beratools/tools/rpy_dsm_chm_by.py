import os
import time
from beratools.tools.common import *
from beratools.tools.r_interface import *

check_r_env()


def chm_by(callback, in_las_folder, is_normalized, out_folder, cell_size, style, processes, verbose):
    rprocesses = r_processes(processes)

    # assign R script file to local variable
    Beratools_R_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Beratools_r_script.r')
    # Defining the R script and loading the instance in Python
    r['source'](Beratools_R_script)
    # Loading the function defined in R script.
    if style == "tin":
        r_chm_by_algorithm = robjects.globalenv['chm_by_dsmtin']
    else:
        r_chm_by_algorithm = robjects.globalenv['chm_by_pitfree']

    # Invoking the R function
    r_chm_by_algorithm(in_las_folder, out_folder, cell_size, is_normalized, rprocesses)


if __name__ == '__main__':
    start_time = time.time()
    print('Normalize Lidar data processing\n @ {}'
          .format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    packages = ['lidR', 'future', 'terra']
    check_r_packages_installation(packages)

    print("Checking input parameters....")
    in_args, in_verbose = check_arguments()
    try:
        cell_size = float(in_args.input["cell_size"])
        in_args.input["cell_size"] = cell_size
    except ValueError:
        print("Invalid input of cell size, default value is used")
        in_args.input["DSMcell_size"] = 1.0

    try:
        is_normalized = bool(in_args.input["is_normalized"])

    except ValueError:
        print("Invalid input of checking normalized data box, DSM will be created")
        in_args.input["is_normalized"] = False

    in_las_folder = in_args.input["in_las_folder"]

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

    if in_args.input["style"] in ["tin", "pitfree"]:
        style = in_args.input["style"]
    else:
        print("Warning! invalid alogrithm, default algorthim will be used.")
        in_args.input["style"] = "tin"

    if not os.path.exists(out_folder):
        print("Warning! Cannot locate output folder, It will be created.")
        os.makedirs(out_folder)

    print("Checking input parameters....Done")

    chm_by(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)

    print('Normalize Lidar data processing is done in {} seconds)'
          .format(round(time.time() - start_time, 5)))
