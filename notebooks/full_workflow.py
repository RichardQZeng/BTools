# usage: python full_workflow.py 10 win/hpc

import os

import sys
from pathlib import Path
from inspect import getsourcefile

if __name__ == "__main__":
    current_file = Path(getsourcefile(lambda: 0)).resolve()
    current_folder = current_file.parent
    btool_dir = current_file.parents[1]
    sys.path.insert(0, btool_dir.as_posix())

from beratools.tools.centerline import centerline
from notebooks.footprint_canopy import FootprintCanopy
from beratools.tools.line_footprint_fixed import line_footprint_fixed

import yaml

gdal_env = os.environ.get("GDAL_DATA")
gdal_env

processes = 18
verbose = False

if __name__ == '__main__':
    if len(sys.argv) > 2:
        processes = int(sys.argv[1])
        print(f'CPU cores: {processes}')
        platform_str = sys.argv[2]

    yml_file = current_folder.joinpath("params_" + platform_str).with_suffix(".yml")
    print(f"Config file: {yml_file}")

    with open(yml_file) as in_params:
        params = yaml.safe_load(in_params)

    # centerline
    args_centerline = params['args_centerline']
    print(args_centerline)
    centerline(**args_centerline, processes=processes, verbose=verbose)

    # alternative relative footprint
    fp_params = params['args_footprint_canopy']
    in_file = fp_params['in_file']
    in_chm = fp_params["in_chm"]
    out_file_percentile = fp_params["out_file_percentile"]
    out_file_fp = fp_params["out_file_fp"]

    footprint = FootprintCanopy(in_file, in_chm)
    footprint.compute()
    footprint.savve_line_percentile(out_file_percentile)
    footprint.save_footprint(out_file_fp)

    # ground footprint
    args_line_footprint_fixed = params["args_line_footprint_fixed"]
    print(args_line_footprint_fixed)

    line_footprint_fixed(
        callback=print, **args_line_footprint_fixed, processes=processes, verbose=verbose
    )
