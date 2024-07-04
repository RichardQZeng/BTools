import os
import time

from beratools.tools.common import *
from beratools.tools.r_interface import *

def laz2las(callback, in_las_folder, out_folder, processes, verbose):
    rprocesses = r_processes(processes)

    # assign R script file to local variable
    Beratools_R_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Beratools_r_script.r')
    # Defining the R script and loading the instance in Python
    r['source'](Beratools_R_script)
    # Loading the function defined in R script.
    r_laz2las = robjects.globalenv['laz2las']

    # Invoking the R function
    r_laz2las(in_las_folder, out_folder, rprocesses)


if __name__ == '__main__':
    start_time = time.time()
    print('Unzip liDAR data processing..\n @ {}'
          .format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    packages = ['lidR', 'future']
    check_r_packages_installation(packages)

    print("Checking input parameters....")
    in_args, in_verbose = check_arguments()

    in_las_folder = in_args.input["in_las_folder"]
    out_folder = in_args.input["out_folder"]

    check_las_files_existence(in_las_folder)

    print("Checking input parameters....Done")
    laz2las(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Unzip liDAR data processing is done in {} seconds)'
          .format(round(time.time() - start_time, 5)))
