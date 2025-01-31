"""
Copyright (C) 2025 Applied Geospatial Research Group.

This script is licensed under the GNU General Public License v3.0.
See <https://gnu.org/licenses/gpl-3.0> for full license details.

Author: Richard Zeng

Description:
    This script is part of the BERA Tools.
    Webpage: https://github.com/appliedgrg/beratools

    The purpose of this script is to split input vector and raster data into
    smaller tiles based on a specified tile size and buffer distance.
"""

import time

import beratools.tools.common as bt_common
import beratools.core.algo_tiler as bt_tiler

def tiler(
    in_line,
    in_raster,
    out_file,
    n_clusters,
    processes,
    verbose,
    tile_buffer=50,
    in_layer=None,
):
    clustering = bt_tiler.DensityBasedClustering(
        in_line=in_line,
        in_raster=in_raster,
        out_file=out_file,
        n_clusters=int(n_clusters),
        tile_buffer=tile_buffer,
        layer=in_layer,
    )

    clustering.run()


if __name__ == "__main__":
    in_args, in_verbose = bt_common.check_arguments()
    start_time = time.time()
    tiler(**in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print("Elapsed time: {}".format(time.time() - start_time))