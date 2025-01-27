import os
import sys
import pytest
from unittest.mock import patch
from pathlib import Path

sys.path.insert(0, Path(__file__).parents[1].as_posix())

# Fixture to get the path to the 'data' directory
@pytest.fixture
def testdata_dir():
    return Path(__file__).parent.joinpath("data")

# Shared arguments for all tools, now using the `testdata_dir` fixture
@pytest.fixture
def tool_arguments(testdata_dir):
    return {
        "args_centerline": {
            'in_line': testdata_dir.joinpath('seed_lines.gpkg').as_posix(),
            'in_layer': 'seed_lines',
            'in_raster': testdata_dir.joinpath('CHM.tif').as_posix(),
            'line_radius': 15,
            'proc_segments': True,
            'out_line': testdata_dir.joinpath('centerline.gpkg').as_posix(),
            'out_layer': 'centerline',
            'processes': 8,
            'verbose': True
        },
        "args_footprint_abs": {
            'in_line': testdata_dir.joinpath('centerline.gpkg').as_posix(),
            'in_chm': testdata_dir.joinpath('CHM.tif').as_posix(),
            'in_layer': 'centerline',
            'corridor_thresh': 3.0,
            'max_ln_width': 32.0,
            'exp_shk_cell': 0,
            'out_footprint': testdata_dir.joinpath('footprint_abs.shp').as_posix(),
            'out_layer': 'footprint_abs',
            'processes': 8,
            'verbose': True
        },
        "args_footprint_rel": {
            'in_line': testdata_dir.joinpath('centerline.gpkg').as_posix(),
            'in_chm': testdata_dir.joinpath('CHM.tif').as_posix(),
            'out_footprint': testdata_dir.joinpath('footprint_rel.gpkg').as_posix(),
            'in_layer': 'centerline',
            'out_layer': 'footprint_rel',
            'max_ln_width': 32,
            'tree_radius': 1.5,
            'max_line_dist': 1.5,
            'canopy_avoidance': 0.0,
            'exponent': 0,
            'canopy_thresh_percentage': 50,
            'processes': 8,
            'verbose': True
        },
        "args_line_footprint_fixed": {
            'in_line': testdata_dir.joinpath('centerline.gpkg').as_posix(),
            'in_footprint': testdata_dir.joinpath('footprint_rel.gpkg').as_posix(),
            'in_layer': 'centerline',
            'in_layer_ft': 'footprint_rel',
            'n_samples': 15,
            'offset': 30,
            'max_width': True,
            'out_footprint': testdata_dir.joinpath('footprint_final.gpkg').as_posix(),
            'out_layer': 'footprint_fixed',
            'processes': 8,
            'verbose': True
        }
    }
