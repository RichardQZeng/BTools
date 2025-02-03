"""
Copyright (C) 2025 Applied Geospatial Research Group.

This script is licensed under the GNU General Public License v3.0.
See <https://gnu.org/licenses/gpl-3.0> for full license details.

Author: AUTHOR NAME

Description:
    This script is part of the BERA Tools.
    Webpage: https://github.com/appliedgrg/beratools

    The purpose of this script is to provide template for tool.
"""
import time
from multiprocessing.pool import Pool
from random import random

import numpy as np

import beratools.tools.common as bt_common
from beratools.core.tool_base import execute_multiprocessing

# Example task_data as a list of numpy ndarrays
task_data_list = [
    np.array([1, 2, 3]),
    np.array([4, 5, 6]),
    np.array([7, 8, 9])
]

def tool_name(
    in_line, in_cost_raster, line_radius, out_line, processes, verbose
):
    """
    Define tool entry point.
    
    These arguments are defined in beratools.json file. 
    execute_multiprocessing is common function to run tasks in parallel.
    """
    result = execute_multiprocessing(
        worker,
        task_data_list,
        "tool_template",
        processes,
        verbose=verbose
    )
    print(len(result))


# task executed in a worker process
def worker(task_data):
    # report a message
    value = np.mean(task_data)
    print(f'Task with {value} executed', flush=True)

    # block for a moment
    time.sleep(value * 10)

    # return the generated value
    return value


if __name__ == '__main__':
    in_args, in_verbose = bt_common.check_arguments()
    start_time = time.time()
    tool_name(
        print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose
    )
    print("Elapsed time: {}".format(time.time() - start_time))
