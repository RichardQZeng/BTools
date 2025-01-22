#
#    Copyright (C) 2021  Applied Geospatial Research Group
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, version 3.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://gnu.org/licenses/gpl-3.0>.
#
# ---------------------------------------------------------------------------
#
# vertex_optimization.py
# Author: Richard Zeng
# Date: 2021-Oct-26
#
# This script is part of the Forest Line Mapper (FLM) toolset
# Webpage: https://github.com/appliedgrg/flm
#
# Move line vertices to right seismic line courses
#
# ---------------------------------------------------------------------------
# System imports
import time
import beratools.tools.common as bt_common
import beratools.core.algo_vertex_optimization as bt_vo

def vertex_optimization(
    in_line,
    in_raster,
    search_distance,
    line_radius,
    out_line,
    processes,
    verbose,
):
    if not bt_common.compare_crs(
        bt_common.vector_crs(in_line), bt_common.raster_crs(in_raster)
    ):
        return

    vg = bt_vo.VertexGrouping(
        in_line,
        in_raster,
        search_distance,
        line_radius,
        out_line,
        processes,
        verbose,
    )
    vg.create_all_vertex_groups()
    vg.compute()
    vg.update_all_lines()
    vg.save_all_layers(out_line)


if __name__ == "__main__":
    in_args, in_verbose = bt_common.check_arguments()
    start_time = time.time()
    vertex_optimization(
        **in_args.input, processes=int(in_args.processes), verbose=in_verbose
    )
    print("Elapsed time: {}".format(time.time() - start_time))
