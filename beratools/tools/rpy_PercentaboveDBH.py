import os
import time
from common import *
from r_interface import *

check_r_env()


def percentage_aboveDBH(callback, in_las_folder, is_normalized, out_folder, DBH, cell_size, processes, verbose):
    rprocesses = r_processes(processes)

    # assign R script file to local variable
    Beratools_R_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Beratools_r_script.r')
    # Defining the R script and loading the instance in Python
    r['source'](Beratools_R_script)
    # Loading the function defined in R script.
    r_percentage_aboveDBH = robjects.globalenv['percentage_aboveDBH']
    # Invoking the R function
    r_percentage_aboveDBH(in_las_folder, is_normalized, out_folder, DBH, cell_size, rprocesses)


if __name__ == '__main__':
    start_time = time.time()
    print('Finding percentage returns above DBH height process\n @ {}'
          .format(time.strftime("%d %b %Y %H:%M:%S", time.localtime())))

    print("Checking input parameters ...")
    in_args, in_verbose = check_arguments()
    in_las_folder = in_args.input["in_las_folder"]

    try:
        DBH = float(in_args.input["DBH"])
        if DBH < 0:
            raise ValueError
        else:
            in_args.input["DBH"] = DBH
    except ValueError:
        print("Invalid input of DBH, default value is used")
        in_args.input["DBH"] = 1.3

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

    percentage_aboveDBH(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)

    print('Finding percentage returns above DBH height process is done in {} seconds)'
          .format(round(time.time() - start_time, 5)))
