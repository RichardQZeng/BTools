import os
import time

from beratools.tools.common import *
from beratools.tools.r_interface import *

def las_info(callback, in_las_folder, processes, verbose):
    rprocesses = r_processes(processes)
    # assign R script file to local variable
    Beratools_R_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Beratools_r_script.r')
    # Defining the R script and loading the instance in Python
    r['source'](Beratools_R_script)
    # Loading the function defined in R script.
    r_normalized_lidar = robjects.globalenv['las_info']

    # Invoking the R function
    r_normalized_lidar(in_las_folder, rprocesses)


if __name__ == '__main__':
    start_time = time.time()
    print('LiDAR data information..\n @ {}'
          .format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    packages = ['lidR', 'future']
    check_r_packages_installation(packages)

    print("Checking input parameters....")
    in_args, in_verbose = check_arguments()

    in_las_folder = in_args.input["in_las_folder"]

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

    print("Checking input parameters....Done")

    las_info(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)

    print('Display LiDAR data information is done in {} seconds)'
          .format(round(time.time() - start_time, 5)))
