import os
import sys
import time
from common import *
from r_interface import *

check_r_env()


def find_cell_size(callback, in_las_folder, processes, verbose):
    rprocesses = r_processes(processes)

    # assign R script file to local variable
    Beratools_R_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'Beratools_r_script.r')
    # Defining the R script and loading the instance in Python
    r['source'](Beratools_R_script)
    # Loading the function defined in R script.
    r_pd2cellsize = robjects.globalenv['pd2cellsize']

    # Invoking the R function
    cell_size = r_pd2cellsize(in_las_folder, rprocesses)
    print("The recommended cell size for the input liDAR data(set) is {}.".format(cell_size))


if __name__ == '__main__':
    start_time = time.time()
    print('Calculate cell size from liDAR data point density processing..\n @ {}'
          .format(time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    # utils = importr('utils')
    # base = importr('base')
    # utils.chooseCRANmirror(ind=12)  # select the 12th mirror in the list: Canada
    # print("Checking R packages....")
    # CRANpacknames = ['lidR', 'future']
    # CRANnames_to_install = [x for x in CRANpacknames if not robjects.packages.isinstalled(x)]
    #
    # if len(CRANnames_to_install) > 0:
    #     utils.install_packages(StrVector(CRANnames_to_install))
    #     packages_found = True
    # else:
    #     packages_found = True
    #
    # del CRANpacknames, CRANnames_to_install

    print("Checking input parameters....")
    in_args, in_verbose = check_arguments()

    in_las_folder = in_args.input["in_las_folder"]
    check_las_files_existence(in_las_folder)
    # if not os.path.exists(in_las_folder):
    #     print("Error! Cannot locate Las folder, please check.")
    #     exit()
    # else:
    #     found = False
    #     for files in os.listdir(in_las_folder):
    #         if files.endswith(".laz"):
    #             found = True
    #             break
    #     if not found:
    #         print("Error! Cannot locate input LAZ file(s), please check!")
    #         exit()

    print("Checking input parameters....Done")

    find_cell_size(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)

    print('Calculate cell size from liDAR data point density processing is done in {} seconds)'
          .format(round(time.time() - start_time, 5)))
