import os
import psutil

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr, data
from rpy2.robjects.vectors import StrVector

r = robjects.r


def check_r_env():
    try:  # integrated R env
        # check R language within env
        current_env_path = os.environ['CONDA_PREFIX']
        # if os.path.isdir(current_env_path):
        os.environ['R_HOME'] = os.path.join(current_env_path, r"Lib\R")
        os.environ['R_USER'] = os.path.expanduser('~')
        os.environ['R_LIBS_USER'] = os.path.join(current_env_path, r"Lib\R\library")

    except FileNotFoundError:
        print("Warning: Please install R for this process!!")
        exit()


def r_processes(processes):
    stats = psutil.virtual_memory()  # returns a named tuple
    available = getattr(stats, 'available') / 1024000000
    rprocesses = processes

    if 2 < processes <= 8:
        if available <= 50:
            rprocesses = 2
        elif 50 < available <= 150:
            rprocesses = 4
        elif 150 < available <= 250:
            rprocesses = 8
    else:
        rprocesses = 8

    return rprocesses


def check_las_files_existence(folder):
    if not os.path.exists(folder):
        print("Error! Cannot locate Las folder, please check.")
        exit()
    else:
        found = False
        for files in os.listdir(folder):
            if files.endswith(".las") or files.endswith(".laz"):
                found = True
                break
        if not found:
            print("Error! Cannot locate input LAS file(s), please check!")
            exit()