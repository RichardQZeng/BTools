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
from beratools.core.algo_canopy_threshold_relative import main_canopy_threshold_relative
from beratools.core.algo_line_footprint_functions import main_line_footprint_relative
from beratools.tools.line_footprint_fixed import line_footprint_fixed

import yaml

gdal_env = os.environ.get("GDAL_DATA")
gdal_env

processes = 18
verbose = False

if __name__ == '__main__':
    if len(sys.argv) > 1:
        processes = int(sys.argv[1])
        print(f'CPU cores: {processes}')

    with open(current_folder.joinpath('params_win.yml')) as in_params:
        params = yaml.safe_load(in_params)

    # ### centerline
    args_centerline = params['args_centerline']
    print(args_centerline)

    centerline(**args_centerline, processes=processes, verbose=verbose)

    # canopy footprint
    args_canopy_threshold = params["args_canopy_threshold"]
    print(args_canopy_threshold)

    dy_cl_line = main_canopy_threshold_relative(
        callback=print, **args_canopy_threshold, processes=processes, verbose=verbose
    )

    args_line_footprint_relative = params["args_line_footprint_relative"]
    print(args_line_footprint_relative)

    main_line_footprint_relative(
        callback=print, **args_line_footprint_relative, processes=processes, verbose=verbose
    )
    
    # ground footprint
    args_line_footprint_fixed = params["args_line_footprint_fixed"]
    print(args_line_footprint_fixed)

    line_footprint_fixed(
        callback=print, **args_line_footprint_fixed, processes=processes, verbose=verbose
    )
