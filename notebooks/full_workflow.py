"""
Provide a full workflow that runs the centerline, canopy and ground footprint tools.

usage: 
    python full_workflow.py -c 10 -p win/hpc -f some.yml
"""

import argparse
import os
from pathlib import Path
from pprint import pprint

import sys
sys.path.append(Path(__file__).resolve().parents[1].as_posix())

import yaml

from beratools.core.algo_footprint_rel import line_footprint_rel
from beratools.core.constants import PARALLEL_MODE, ParallelMode
from beratools.tools.centerline import centerline
from beratools.tools.line_footprint_absolute import line_footprint_abs
from beratools.tools.line_footprint_fixed import line_footprint_fixed

print = pprint

gdal_env = os.environ.get("GDAL_DATA")
gdal_env

processes = 12
verbose = False

def check_arguments():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Run the full workflow parameters.")
    
    # Add arguments for each parameter
    parser.add_argument(
        '-p', '--platform', 
        choices=['win', 'hpc'], 
        required=True, 
        default="hpc",
        help="Specify the platform: 'win' or 'hpc'"
    )
    
    parser.add_argument(
        '-c', '--cores', 
        type=int, 
        required=True, 
        help="Number of cores to allocate (integer)"
    )
    
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

if __name__ == '__main__':
    script_dir = Path(__file__).parent

    args = check_arguments()

    # Access the parameters
    platform = args.platform
    cores = args.cores
    file = args.file

    # Set default file based on platform if not provided
    if not file:
        if platform == 'hpc':
            file = script_dir / 'params_hpc.yml'  # Use pathlib to join paths
        elif platform == 'win':
            file = script_dir / 'params_win.yml'

    # Print the received arguments (you can replace this with actual processing code)
    print(f"Platform: {platform}")
    print(f"Cores: {cores}")
    if file:
        print(f"Configuration file: {file}")
    else:
        print("No configuration file provided.")

    if args.cores:
        processes = args.cores
        print(f'CPU cores: {processes}')

    print(f'Parallel mode: {PARALLEL_MODE.name}')

    yml_file = Path(__file__).parent.joinpath("params_" + platform).with_suffix(".yml")
    print(f"Config file: {yml_file}")

    with open(yml_file) as in_params:
        params = yaml.safe_load(in_params)

    # centerline
    args_centerline = params['args_centerline']
    args_centerline['processes'] = processes
    print(args_centerline)
    centerline(**args_centerline)

    # canopy footprint abs
    args_footprint_abs = params["args_footprint_abs"]
    args_footprint_abs['processes'] = processes
    print(args_footprint_abs)
    line_footprint_abs(**args_footprint_abs)

    # canopy footprint relative
    args_footprint_rel = params["args_footprint_rel"]
    args_footprint_rel['processes'] = processes
    print(args_footprint_rel)
    line_footprint_rel(**args_footprint_rel)

    # ground footprint
    args_line_footprint_fixed = params["args_line_footprint_fixed"]
    args_line_footprint_fixed['processes'] = processes
    print(args_line_footprint_fixed)
    line_footprint_fixed(**args_line_footprint_fixed)
