"""Test functions and command lines."""
import os

import geopandas as gpd
import pytest
from click.testing import CliRunner
from label_centerlines import __version__, get_centerline
from label_centerlines.cli import main


# Fixture to load the 'alps.geojson' shape using geopandas
@pytest.fixture
def footprint_shape(testdata_dir):
    # Read the GeoJSON file using geopandas
    gdf = gpd.read_file(testdata_dir.joinpath("footprint.geojson"))

    # Return the first geometry from the GeoDataFrame
    return gdf.geometry.iloc[0]

# Test the CLI version command
def test_cli():
    runner = CliRunner()
    result = runner.invoke(main, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output

# Test the centerline functionality
def test_centerline(footprint_shape):
    cl = get_centerline(footprint_shape)
    assert cl.is_valid
    assert cl.geom_type == "MultiLineString"