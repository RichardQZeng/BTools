"""
Provide a full workflow that runs the centerline, canopy and ground footprint tools.

usage: 
    python full_workflow.py [-f some.yml] [-m   PARALLEL_MODE]
"""

import argparse
import os
import platform
import sys
from pathlib import Path
from pprint import pprint

sys.path.append(Path(__file__).resolve().parents[1].as_posix())

import yaml

from beratools.core.algo_footprint_rel import line_footprint_rel
from beratools.core.constants import PARALLEL_MODE, ParallelMode
from beratools.tools.centerline import centerline
from beratools.tools.line_footprint_absolute import line_footprint_abs
from beratools.tools.line_footprint_fixed import line_footprint_fixed

print = pprint

gdal_env = os.environ.get("GDAL_DATA")

processes = 2
verbose = False

def check_arguments():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Run the full workflow parameters.")
    
    # Make the YAML file optional by setting required=False
    parser.add_argument(
        '-f', '--file', 
        type=str, 
        required=False,  # This makes the file parameter optional
        help="Path to the YAML configuration file (optional)"
    )

    parser.add_argument(
        '-m', '--multi', 
        type=int, 
        required=False,  # This makes the file parameter optional
        default=ParallelMode.MULTIPROCESSING,
        help="Parallel computing mode (optional)"
    )
    
    # Parse the arguments
    args = parser.parse_args()

    # Print out the arguments to debug
    print(f"Parsed arguments: {args}")
    return args

def print_message(message):
    print('-'*50)
    print(message)
    print('-'*50)

if __name__ == '__main__':
    script_dir = Path(__file__).parent

    args = check_arguments()
    platform = platform.system()

    # Access the parameters
    yml_file = args.file

    # Set default file based on platform if not provided
    if not yml_file:
        if platform == 'Linux':
            yml_file = script_dir.joinpath('params_hpc.yml')
        elif platform == 'Windows':
            yml_file = script_dir.joinpath('params_win.yml')

    # Get available CPU cores
    processes = os.cpu_count()

    # Print the received arguments (you can replace this with actual processing code)
    print(f"Platform: {platform}")
    print(f"Cores: {processes}")
    print(f'Parallel mode: {PARALLEL_MODE.name}')
    print(f"Configuration file: {yml_file}")

    with open(yml_file) as in_params:
        params = yaml.safe_load(in_params)

    # centerline
    print_message("Starting centerline")
    args_centerline = params['args_centerline']
    args_centerline['processes'] = processes
    print(args_centerline)
    centerline(**args_centerline)
    
    # canopy footprint abs
    print_message("Starting canopy footprint abs")
    args_footprint_abs = params["args_footprint_abs"]
    args_footprint_abs['processes'] = processes
    print(args_footprint_abs)
    line_footprint_abs(**args_footprint_abs)
    
    # canopy footprint relative
    print_message("Starting canopy footprint rel")
    args_footprint_rel = params["args_footprint_rel"]
    args_footprint_rel['processes'] = processes
    print(args_footprint_rel)
    line_footprint_rel(**args_footprint_rel)

    # ground footprint
    print_message("Starting ground footprint")
    args_footprint_fixed = params["args_footprint_fixed"]
    args_footprint_fixed['processes'] = processes
    args_footprint_fixed['parallel_mode'] = ParallelMode.SEQUENTIAL
    print(args_footprint_fixed)
    line_footprint_fixed(**args_footprint_fixed)
