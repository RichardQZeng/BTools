import os
import time
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

def gnd_csf(callback,in_las_folder,out_folder,slope,class_threshold,cloth_resolution,rigidness, processes, verbose):

    r = robjects.r
    in_las_folder=in_las_folder.replace("\\","/")
    out_folder=out_folder.replace("\\","/")

    # assign R script file to local variable
    Beratools_R_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Beratools_r_script.r')
    # Defining the R script and loading the instance in Python
    r['source'](Beratools_R_script)
    # Loading the function defined in R script.
    r_classify_gnd = robjects.globalenv['classify_gnd']
    # Invoking the R function
    r_classify_gnd(in_las_folder,out_folder,slope,class_threshold,cloth_resolution,rigidness)


if __name__ == '__main__':
    start_time = time.time()
    print('Ground Classification (Cloth Simulation Filter) processing\n @ {}'
          .format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    r=robjects.r
    utils = importr('utils')
    base = importr('base')
    utils.chooseCRANmirror(ind=12) # select the 12th mirror in the list: Canada
    print("Checking R packages....")
    CRANpacknames = ['lidR','RCSF']
    CRANnames_to_install = [x for x in CRANpacknames if not robjects.packages.isinstalled(x)]

    if len(CRANnames_to_install) > 0:
       utils.install_packages(StrVector(CRANnames_to_install))
       packages_found=True
    else:
        packages_found= True
        utils.update_packages(StrVector(CRANpacknames))

    # if packages_found:
    #    utils.update_packages(checkBuilt = True, ask=False)

    del CRANpacknames,CRANnames_to_install

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
        if rigidness in [1,2,3]:
            in_args.input["rigidness"] = rigidness
        else:
            raise ValueError
    except ValueError:
        print("Invalid input of rigidness of the cloth, default value is used")
        in_args.input["rigidness"] = 1




    in_las_folder=in_args.input["in_las_folder"]

    out_folder=in_args.input["out_folder"]

    if not os.path.exists(in_las_folder):
        print("Error! Cannot locate Las folder, please check.")
        exit()
    else:
        found = False
        for files in os.listdir(in_las_folder):
            if files.endswith(".las") or files.endswith(".laz"):
                    found=True
                    break
        if not found:
            print("Error! Cannot locate input LAS file(s), please check!")
            exit()


    if not os.path.exists(out_folder):
       print("Warning! Cannot locate output folder, It will be created.")
       os.makedirs(out_folder)

    print("Checking input parameters....Done")

    gnd_csf(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)

    print('Ground Classification (Cloth Simulation Filter) processing is done in {} seconds)'
          .format(round(time.time() - start_time, 5)))

