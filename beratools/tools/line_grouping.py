"""
Copyright (C) 2025 Applied Geospatial Research Group.

This script is licensed under the GNU General Public License v3.0.
See <https://gnu.org/licenses/gpl-3.0> for full license details.

---------------------------------------------------------------------------
Author: Richard Zeng

Description:
    This script is part of the BERA Tools.
    Webpage: https://github.com/appliedgrg/beratools

    The purpose of this script is to provide main interface for line grouping tool.
"""
import logging
import time

from beratools.core.logger import Logger
from beratools.core.algo_line_grouping import LineGrouping
import beratools.tools.common as bt_common

def line_grouping(callback, in_line, out_line, processes, verbose, in_layer=None, out_layer=None):
    print("line_grouping started")
    lg = LineGrouping(in_line, in_layer)
    lg.run_grouping()
    lg.lines.to_file(out_line, layer=out_layer)

log = Logger("line_grouping", file_level=logging.INFO)
logger = log.get_logger()
print = log.print

if __name__ == "__main__":
    in_args, in_verbose = bt_common.check_arguments()
    start_time = time.time()
    line_grouping(
        print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose
    )

    print("Elapsed time: {}".format(time.time() - start_time))