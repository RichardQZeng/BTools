import os
import time
from beratools.tools.common import *
from beratools.tools.r_interface import *

check_r_env()


def normalized_lidar(callback, in_las_folder, out_folder, style, processes, verbose):
    rprocesses = r_processes(processes)

    # assign R script file to local variable
    Beratools_R_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Beratools_r_script.r')
    # Defining the R script and loading the instance in Python
    r['source'](Beratools_R_script)
    # Loading the function defined in R script.
    if style == 'tin':
        r_normalized_lidar = robjects.globalenv['normalized_lidar_knnidw']
    elif style == 'knnidw':
        r_normalized_lidar = robjects.globalenv['normalized_lidar_tin']
    else:
        r_normalized_lidar = robjects.globalenv['normalized_lidar_kriging']

    # Invoking the R function
    r_normalized_lidar(in_las_folder, out_folder, rprocesses)


if __name__ == '__main__':
    start_time = time.time()
    print('Normalize Lidar data processing\n @ {}'
          .format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    packages = ['lidR', 'rgrass', 'rlas', 'future', 'terra', 'sp']
    check_r_packages_installation(packages)

    print("Checking input parameters....")
    in_args, in_verbose = check_arguments()

    if in_args.input["style"] in ["tin", "knnidw", "kriging"]:
        style = in_args.input["style"]
    else:
        print("Warning! invalid alogrithm, default algorthim will be used.")
        in_args.input["style"] = "tin"

    in_las_folder = in_args.input["in_las_folder"]
    out_folder = in_args.input["out_folder"]
    check_las_files_existence(in_las_folder)

    if not os.path.exists(out_folder):
        print("Warning! Cannot locate output folder, It will be created.")
        os.makedirs(out_folder)

    print("Checking input parameters....Done")

    normalized_lidar(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)

    print('Normalize Lidar data processing is done in {} seconds)'
          .format(round(time.time() - start_time, 5)))
