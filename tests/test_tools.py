"""Test script for the tools in the beratools package."""
import time
import warnings
from pathlib import Path
from pprint import pprint

import pytest

from beratools.core.algo_footprint_rel import FootprintCanopy
from beratools.tools.centerline import centerline
from beratools.tools.line_footprint_absolute import line_footprint_abs
from beratools.tools.line_footprint_fixed import line_footprint_fixed

# Ignore the FutureWarning for the specific warning from osgeo.osr
warnings.filterwarnings("ignore", category=FutureWarning)

# E2E TESTS
# Define a helper function to check if the output file exists
def check_file_exists(file_path):
    """Check if the file exists and is not empty."""
    return Path(file_path).exists() and Path(file_path).stat().st_size > 0

def test_centerline_tool_e2e(tool_arguments):
    """E2E test for the centerline tool."""
    args_centerline = tool_arguments["args_centerline"]
    pprint(args_centerline)

    # Call the actual centerline tool (no mocks)
    centerline(**args_centerline)

    # Check if the output file is created
    assert check_file_exists(args_centerline["out_line"]), (
        "Centerline output file was not created!"
    )

def test_line_footprint_abs_tool_e2e(tool_arguments):
    """E2E test for the line_footprint_abs tool."""
    args_footprint_abs = tool_arguments["args_footprint_abs"]
    line_footprint_abs(**args_footprint_abs)
    pprint(args_footprint_abs)

    assert check_file_exists(args_footprint_abs["out_footprint"]), (
        "Footprint Abs output file was not created!"
    )

def test_footprint_canopy_tool_e2e(tool_arguments):
    """E2E test for the FootprintCanopy tool."""
    args_footprint_rel = tool_arguments["args_footprint_rel"]
    pprint(args_footprint_rel)

    footprint = FootprintCanopy(
        args_footprint_rel["in_line"],
        args_footprint_rel["in_chm"],
        in_layer=args_footprint_rel["in_layer"],
    )
    footprint.compute()
    footprint.save_footprint(args_footprint_rel['out_footprint'])

    assert check_file_exists(args_footprint_rel["out_footprint"]), (
        "Footprint Rel output file was not created!"
    )


def test_line_footprint_fixed_tool_e2e(tool_arguments):
    """E2E test for the line_footprint_fixed tool."""
    args_line_footprint_fixed = tool_arguments["args_line_footprint_fixed"]
    line_footprint_fixed(**args_line_footprint_fixed)
    pprint(args_line_footprint_fixed)

    assert check_file_exists(args_line_footprint_fixed["out_footprint"]), (
        "Line footprint fixed output file was not created!"
    )


# Optionally, a test for cleaning up test output files
@pytest.fixture
def test_output_files(testdata_dir):
    return [
        testdata_dir.joinpath('centerline.shp'),
        testdata_dir.joinpath('footprint_abs.shp'),
        testdata_dir.joinpath('footprint_rel.gpkg'),
        testdata_dir.joinpath('footprint_final.gpkg')
    ]

def test_cleanup_output_files(test_output_files):
    """Test to clean up generated output files after the test."""
    time.sleep(1)  # Wait a little to allow file system operations to complete
    for file_path in test_output_files:
        if file_path.exists():
            file_path.unlink()
            assert not file_path.exists(), f"Failed to remove {file_path}"

# Integration test for the entire workflow
def test_full_workflow(tool_arguments):
    """
    Full integration test (actually an E2E test) running the entire workflow.

    with real data, ensuring that each tool integrates properly with the next.
    """
    # 1. Test the centerline tool
    args_centerline = tool_arguments["args_centerline"]
    centerline(**args_centerline)
    assert check_file_exists(args_centerline["out_line"]), (
        "Centerline output file was not created!"
    )
    
    # 2. Test the line_footprint_abs tool
    args_footprint_abs = tool_arguments["args_footprint_abs"]
    line_footprint_abs(**args_footprint_abs)
    assert check_file_exists(args_footprint_abs["out_footprint"]), (
        "Footprint Abs output file was not created!"
    )
    
    # 3. Test the footprint canopy tool
    args_footprint_rel = tool_arguments["args_footprint_rel"]
    footprint = FootprintCanopy(
        args_footprint_rel["in_line"], args_footprint_rel["in_chm"]
    )
    footprint.compute(parallel_mode=False)
    footprint.save_footprint(args_footprint_rel['out_footprint'])
    assert check_file_exists(args_footprint_rel["out_footprint"]), (
        "Footprint Rel output file was not created!"
    )
    
    # 4. Test the line_footprint_fixed tool
    args_line_footprint_fixed = tool_arguments["args_line_footprint_fixed"]
    line_footprint_fixed(**args_line_footprint_fixed)
    assert check_file_exists(args_line_footprint_fixed["out_footprint"]), (
        "Line footprint fixed output file was not created!"
    )
