import os
import time
from common import *
from r_interface import *

check_r_env()


def dtm_by(callback, in_las_folder, out_folder, cell_size, style, processes, verbose):
    processes = r_processes(processes)

    # in_las_folder=in_las_folder.replace("\\","/")
    # out_folder=out_folder.replace("\\","/")

    # assign R script file to local variable
    Beratools_R_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Beratools_r_script.r')
    # Defining the R script and loading the instance in Python
    r['source'](Beratools_R_script)

    # Loading the function defined in R script.
    if style == "tin":
        r_chm_by_algorithm = robjects.globalenv['dtm_by_tin']
    elif style == "idw":
        r_chm_by_algorithm = robjects.globalenv['dtm_by_knnidw ']
    else:
        r_chm_by_algorithm = robjects.globalenv['dtm_by_kriging']
    # Invoking the R function
    r_chm_by_algorithm(in_las_folder, out_folder, cell_size, rprocesses)


if __name__ == '__main__':
    start_time = time.time()
    print('Normalize Lidar data processing\n @ {}'
          .format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    # r = robjects.r
    # utils = importr('utils')
    # base = importr('base')
    # utils.chooseCRANmirror(ind=12)  # select the 12th mirror in the list: Canada
    # print("Checking R packages....")
    # CRANpacknames = ['lidR', 'rgrass', 'rlas', 'future', 'terra',
    #                  'sp']  # ,'comprehenr','na.tools','sf','sp']#,'devtools','gdal']#,'fasterRaster']
    # CRANnames_to_install = [x for x in CRANpacknames if not robjects.packages.isinstalled(x)]
    #
    # if len(CRANnames_to_install) > 0:
    #     utils.install_packages(StrVector(CRANnames_to_install))
    #     packages_found = True
    # else:
    #     packages_found = True
    #
    # # if packages_found:
    # #    utils.update_packages(checkBuilt = True, ask=False)
    #
    # del CRANpacknames, CRANnames_to_install

    print("Checking input parameters....")
    in_args, in_verbose = check_arguments()
    try:
        cell_size = float(in_args.input["cell_size"])
        in_args.input["cell_size"] = cell_size
    except ValueError:
        print("Invalid input of cell size, default value is used")
        in_args.input["DSMcell_size"] = 1.0

    in_las_folder = in_args.input["in_las_folder"]
    out_folder = in_args.input["out_folder"]
    check_las_files_existence(in_las_folder)

    # if not os.path.exists(in_las_folder):
    #     print("Error! Cannot locate Las folder, please check.")
    #     exit()
    # else:
    #     found = False
    #     for files in os.listdir(in_las_folder):
    #         if files.endswith(".las") or files.endswith(".laz"):
    #             found = True
    #             break
    #     if not found:
    #         print("Error! Cannot locate input LAS file(s), please check!")
    #         exit()

    if in_args.input["style"] in ["tin", "idw", "kriging"]:
        style = in_args.input["style"]
    else:
        print("Warning! invalid alogrithm, default algorthim will be used.")
        in_args.input["style"] = "tin"

    if not os.path.exists(out_folder):
        print("Warning! Cannot locate output folder, It will be created.")
        os.makedirs(out_folder)

    print("Checking input parameters....Done")

    dtm_by(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)

    print('Normalize Lidar data processing is done in {} seconds)'
          .format(round(time.time() - start_time, 5)))
