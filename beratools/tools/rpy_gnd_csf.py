import os
import time
from common import *
from r_interface import *

check_r_env()


def gnd_csf(callback, in_las_folder, out_folder, slope, class_threshold, cloth_resolution, rigidness, processes,
            verbose):
    # assign R script file to local variable
    Beratools_R_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Beratools_r_script.r')
    # Defining the R script and loading the instance in Python
    r['source'](Beratools_R_script)
    # Loading the function defined in R script.
    r_classify_gnd = robjects.globalenv['classify_gnd']
    # Invoking the R function
    r_classify_gnd(in_las_folder, out_folder, slope, class_threshold, cloth_resolution, rigidness)


if __name__ == '__main__':
    start_time = time.time()
    print('Ground Classification (Cloth Simulation Filter) processing\n @ {}'
          .format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    print("Checking input parameters....")
    in_args, in_verbose = check_arguments()
    try:
        slope = bool(in_args.input["slope"])
        in_args.input["slope"] = slope
    except ValueError:
        print("Invalid input of indicate slope exists, default value is used")
        in_args.input["slope"] = False

    try:
        class_threshold = float(in_args.input["class_threshold"])
        in_args.input["class_threshold"] = class_threshold
    except ValueError:
        print("Invalid input of ground threshold, default value is used")
        in_args.input["class_threshold"] = 0.5

    try:
        cloth_resolution = float(in_args.input["cloth_resolution"])
        in_args.input["cloth_resolution"] = cloth_resolution
    except ValueError:
        print("Invalid input of distance between particles in the cloth, default value is used")
        in_args.input["cloth_resolution"] = 0.5

    try:
        rigidness = int(in_args.input["rigidness"])
        if rigidness in [1, 2, 3]:
            in_args.input["rigidness"] = rigidness
        else:
            raise ValueError
    except ValueError:
        print("Invalid input of rigidness of the cloth, default value is used")
        in_args.input["rigidness"] = 1

    in_las_folder = in_args.input["in_las_folder"]
    out_folder = in_args.input["out_folder"]

    check_las_files_existence(in_las_folder)

    if not os.path.exists(out_folder):
        print("Warning! Cannot locate output folder, It will be created.")
        os.makedirs(out_folder)

    print("Checking input parameters....Done")

    gnd_csf(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)

    print('Ground Classification (Cloth Simulation Filter) processing is done in {} seconds)'
          .format(round(time.time() - start_time, 5)))
