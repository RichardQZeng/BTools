# usage: python full_workflow.py -c 10 -p win/hpc -f some.yml

import os

import sys
import argparse
from pathlib import Path
from inspect import getsourcefile

if __name__ == "__main__":
    current_file = Path(getsourcefile(lambda: 0)).resolve()
    current_folder = current_file.parent
    btool_dir = current_file.parents[1]
    sys.path.insert(0, btool_dir.as_posix())

from beratools.core.constants import PARALLEL_MODE, ParallelMode
from beratools.tools.centerline import centerline
from beratools.tools.line_footprint_absolute import line_footprint_abs
from beratools.core.algo_footprint_canopy_rel import FootprintCanopy
from beratools.tools.line_footprint_fixed import line_footprint_fixed
import yaml
from pprint import pprint
print = pprint


gdal_env = os.environ.get("GDAL_DATA")
gdal_env

processes = 12
verbose = False

def check_arguments():
    # Create the argument parser
    parser = argparse.ArgumentParser(description="Run the full workflow tiler script with specific parameters.")
    
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

    yml_file = current_folder.joinpath("params_" + platform).with_suffix(".yml")
    print(f"Config file: {yml_file}")

    with open(yml_file) as in_params:
        params = yaml.safe_load(in_params)

    # centerline
    # args_centerline = params['args_centerline']
    # args_centerline["parallel_mode"] = PARALLEL_MODE
    # print(args_centerline)
    # centerline(**args_centerline, processes=processes, verbose=verbose)

    args_footprint_abs = params["args_footprint_abs"]
    args_footprint_abs['verbose'] = False
    args_footprint_abs['processes'] =12
    args_footprint_abs["callback"] = None
    print(args_footprint_abs)
    line_footprint_abs(**args_footprint_abs)

    # canopy footprint
    # fp_params = params['args_footprint_canopy']
    # in_file = fp_params['in_file']
    # in_chm = fp_params["in_chm"]
    # out_file_percentile = fp_params["out_file_percentile"]
    # out_file_fp = fp_params["out_file_fp"]

    # footprint = FootprintCanopy(in_file, in_chm)
    # footprint.compute(PARALLEL_MODE)
    # footprint.save_footprint(out_file_fp)

    # ground footprint
    # args_line_footprint_fixed = params["args_line_footprint_fixed"]
    # args_line_footprint_fixed["parallel_mode"] = PARALLEL_MODE
    # print(args_line_footprint_fixed)

    # line_footprint_fixed(
    #     callback=print, **args_line_footprint_fixed, processes=processes, verbose=verbose
    # )
