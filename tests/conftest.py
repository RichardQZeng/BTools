"""
Test file.

Returns:
    _type_: _description_

"""
import geopandas as gpd
import os
import pytest

TESTDATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "testdata")


@pytest.fixture
def alps_shape():
    gdf = gpd.read_file(os.path.join(TESTDATA_DIR, "alps.shp"))
    return gdf
