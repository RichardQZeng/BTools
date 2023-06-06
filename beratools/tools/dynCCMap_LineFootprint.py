# ---------------------------------------------------------------------------
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
# dynCCMap_LineFootprint.py
# Script Author: Maverick Fong
# Date: 2023-May
# Use open sources python library for produce dynamic footprint from dynamic canopy and cost raster with lines and CHM input.
# Prerequisite:  Line feature class must have the attribute Fields: "OLnFID" adn CHM raster
# dynCCMap_LineFootprint.py
# This script is part of the BERA toolset
# Webpage: https://github.com/
#
# Purpose: Creates dynamic footprint polygons for each input line based on a least
# cost corridor method and individual line thresholds.
#
# ---------------------------------------------------------------------------
from dynamic_line_footprint import *
from dynamic_canopy_threshold import *

if __name__ == '__main__':
    start_time = time.time()
    print('Starting Dynamic CC and Footprint processing\n @ {}'.format(
        time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime())))

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=json.loads)
    parser.add_argument('-p', '--processes')
    parser.add_argument('-v', '--verbose')
    args = parser.parse_args()
    args.input['full_step']=True
    del args.input['out_footprint'], args.input['exp_shk_cell'], args.input['max_ln_width']

    verbose = True if args.verbose == 'True' else False
    dy_cl_line=dynamic_canopy_threshold(print, **args.input, processes=int(args.processes), verbose=verbose)
    args = parser.parse_args()
    args.input['full_step'] = True
    args.input["in_line"]=dy_cl_line
    del args.input['Off_ln_dist'],args.input['CanPercentile'],args.input['CanThrPercentage']
    verbose = True if args.verbose == 'True' else False
    dynamic_line_footprint(print, **args.input, processes=int(args.processes), verbose=verbose)

    print('%{}'.format(100))
    print('Finishing Dynamic Footprint processes @ {}\n(or in {} second)'.format(
        time.strftime("%a, %d %b %Y %H:%M:%S", time.localtime()), round(time.time() - start_time, 5)))








