import numpy as np
import geopandas as gpd
from scipy import ndimage
import shapely.geometry as sh_geom

import beratools.core.constants as bt_const


def read_geospatial_file(file_path, layer=None):
    """
    Reads a geospatial file and returns a cleaned GeoDataFrame.

    Args:
        file_path (str): The path to the geospatial file (e.g., .shp, .gpkg).
        layer (str, optional): The specific layer to read if the file is multi-layered (e.g., GeoPackage).

    Returns:
        GeoDataFrame: The cleaned GeoDataFrame containing the data from the file with valid geometries only.
        None: If there is an error reading the file or layer.
    """
    try:
        if layer is None:
            gdf = gpd.read_file(file_path)  # Read the file without specifying a layer
        else:
            gdf = gpd.read_file(
                file_path, layer=layer
            )  # Read the file with the specified layer

        # Clean the geometries in the GeoDataFrame
        gdf = clean_geometries(gdf)
        gdf["BT_UID"] = range(len(gdf))  # assign temporary UID
        return gdf

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None

def has_multilinestring(gdf):
    """Check if any geometry is a MultiLineString."""
    # Filter out None values (invalid geometries) from the GeoDataFrame
    valid_geometries = gdf.geometry
    return any(isinstance(geom, sh_geom.MultiLineString) for geom in valid_geometries)

def clean_geometries(gdf):
    """
    Removes rows with invalid, None, or empty geometries from the GeoDataFrame.

    Args:
        gdf (GeoDataFrame): The GeoDataFrame to clean.

    Returns:
        GeoDataFrame: The cleaned GeoDataFrame with valid, non-null, and non-empty geometries.
    """
    # Remove rows where the geometry is invalid, None, or empty
    gdf = gdf[gdf.geometry.is_valid]  # Only keep valid geometries
    gdf = gdf[~gdf.geometry.isna()]  # Remove rows with None geometries
    gdf = gdf[
        gdf.geometry.apply(lambda geom: not geom.is_empty)
    ]  # Remove empty geometries
    return gdf

def prepare_lines_gdf(file_path, layer=None, proc_segments=True):
    """Split lines at vertices or return original rows, with handling for MultiLineString."""
    # Check if there are any MultiLineString geometries
    gdf = read_geospatial_file(file_path, layer=layer)

    if has_multilinestring(gdf):
        gdf = gdf.explode(index_parts=False)  # Explode MultiLineStrings into individual LineStrings

    split_gdf_list = []

    for row in gdf.itertuples(index=False):  # Use itertuples to iterate
        line = row.geometry  # Access geometry directly via the named tuple

        # If proc_segment is True, split the line at vertices
        if proc_segments:
            coords = list(line.coords)  # Extract the list of coordinates (vertices)

            # For each LineString, split the line into segments by the vertices
            for i in range(len(coords) - 1):
                segment = sh_geom.LineString([coords[i], coords[i + 1]])

                # Copy over all non-geometry columns from the parent row (excluding 'geometry')
                attributes = {col: getattr(row, col) for col in gdf.columns if col != 'geometry'}
                single_row_gdf = gpd.GeoDataFrame([attributes], geometry=[segment], crs=gdf.crs)
                split_gdf_list.append(single_row_gdf)

        else:
            # If proc_segment is False, just add the original row as a single-row GeoDataFrame
            attributes = {col: getattr(row, col) for col in gdf.columns if col != 'geometry'}
            single_row_gdf = gpd.GeoDataFrame([attributes], geometry=[line], crs=gdf.crs)
            split_gdf_list.append(single_row_gdf)

    return split_gdf_list


# TODO use function from common
def morph_raster(corridor_thresh, canopy_raster, exp_shk_cell, cell_size_x):
    # Process: Stamp CC and Max Line Width
    temp1 = corridor_thresh + canopy_raster
    raster_class = np.ma.where(temp1 == 0, 1, 0).data

    if exp_shk_cell > 0 and cell_size_x < 1:
        # Process: Expand
        # FLM original Expand equivalent
        cell_size = int(exp_shk_cell * 2 + 1)
        expanded = ndimage.grey_dilation(
            raster_class, size=(cell_size, cell_size)
        )

        # Process: Shrink
        # FLM original Shrink equivalent
        file_shrink = ndimage.grey_erosion(
            expanded, size=(cell_size, cell_size)
        )

    else:
        if bt_const.BT_DEBUGGING:
            print("No Expand And Shrink cell performed.")
        file_shrink = raster_class

    # Process: Boundary Clean
    clean_raster = ndimage.gaussian_filter(file_shrink, sigma=0, mode="nearest")

    return clean_raster


def closest_point_to_line(point, line):
    if not line:
        return None

    pt = line.interpolate(line.project(sh_geom.Point(point)))
    return pt


def points_in_line(line):
    point_list = []
    try:
        for point in list(line.coords):  # loops through every point in a line
            # loops through every vertex of every segment
            if point:  # adds all the vertices to segment_list, which creates an array
                point_list.append(sh_geom.Point(point[0], point[1]))
    except Exception as e:
        print(e)

    return point_list


def intersection_of_lines(line_1, line_2):
    """
     only LINESTRING is dealt with for now
    Parameters
    ----------
    line_1 :
    line_2 :

    Returns
    -------

    """
    # intersection collection, may contain points and lines
    inter = None
    if line_1 and line_2:
        inter = line_1.intersection(line_2)

    # TODO: intersection may return GeometryCollection, LineString or MultiLineString
    if inter:
        if (
            type(inter) is sh_geom.GeometryCollection
            or type(inter) is sh_geom.LineString
            or type(inter) is sh_geom.MultiLineString
        ):
            return inter.centroid

    return inter
