import logging
import time

import sys
from pathlib import Path
from inspect import getsourcefile

if __name__ == "__main__":
    current_file = Path(getsourcefile(lambda: 0)).resolve()
    btool_dir = current_file.parents[2]
    sys.path.insert(0, btool_dir.as_posix())

from beratools.core.logger import Logger
from beratools.core.linegrouping import LineGrouping
from beratools.core.constants import *
from common import *

def line_grouping(callback, in_line, out_line, processes, verbose):
    print("line_grouping started")
    lg = LineGrouping(in_line)
    lg.run_grouping()
    lg.lines.to_file(out_line)

log = Logger("line_grouping", file_level=logging.INFO)
logger = log.get_logger()
print = log.print

if __name__ == "__main__":
    in_args, in_verbose = check_arguments()
    start_time = time.time()
    line_grouping(
        print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose
    )

    print("Elapsed time: {}".format(time.time() - start_time))