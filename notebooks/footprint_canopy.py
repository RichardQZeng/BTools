import math
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
from shapely import ops
from shapely.geometry import Point, MultiPolygon, shape
from rasterio.features import shapes, rasterize
from skimage.graph import MCP_Flexible
from enum import StrEnum

import sys
from pathlib import Path
from inspect import getsourcefile

if __name__ == "__main__":
    current_file = Path(getsourcefile(lambda: 0)).resolve()
    current_folder = current_file.parent
    btool_dir = current_file.parents[1]
    sys.path.insert(0, btool_dir.as_posix())

from beratools.tools.common import (
    clip_raster,
    cost_raster_2nd_version,
    remove_nan_from_array,
    morph_raster,
)

from beratools.core.constants import BT_NODATA, FP_CORRIDOR_THRESHOLD


class Side(StrEnum):
    left = "left"
    right = "right"


class FootprintCanopy:
    def __init__(self, in_geom, in_chm):
        data = gpd.read_file(in_file)
        self.lines = []

        for idx in data.index:
            line = LineInfo(data.iloc[[idx]], in_chm)
            self.lines.append(line)
    def compute(self):
        for item in self.lines:
            item.compute()
            print("line computation done")

        fp = [item.footprint for item in self.lines]
        self.footprints = pd.concat(fp)

        percentile = [item.line for item in self.lines]
        self.lines = pd.concat(percentile)

class BufferRing:
    def __init__(self, ring_poly, side):
        self.geometry = ring_poly
        self.side = side
        self.percentile = 0.5
        self.Dyn_Canopy_Threshold = 0.05

class LineInfo:
    def __init__(self, line_gdf, in_chm):
        self.line = line_gdf
        self.in_chm = in_chm
        self.line_simp = self.line.geometry.simplify(
            tolerance=0.5, preserve_topology=True
        )

        self.canopy_percentile = 50
        self.DynCanTh = np.nan
        # chk_df_multipart
        # if proc_segments:
        # line_seg = split_into_segments(line_seg)

        self.buffer_rings = []

        self.CL_CutHt = np.nan
        self.CR_CutHt = np.nan
        self.RDist_Cut = np.nan
        self.LDist_Cut = np.nan

        self.canopy_thresh_percentage = 50
        self.canopy_avoidance = 0.0
        self.exponent = 1.0
        self.max_ln_width = 32
        self.max_line_dist = 1.5
        self.tree_radius = 1.5

        self.nodata = -9999
        self.dyn_canopy_ndarray = None
        self.negative_cost_clip = None
        self.out_meta = None

        self.buffer_left = None
        self.buffer_right = None
        self.footprint = None

    def compute(self):
        self.prepar_ring_buffer()

        ring_list = []
        for item in self.buffer_rings:
            ring = self.cal_percentileRing(item)
            ring_list.append(ring)

        self.buffer_rings = ring_list

        self.rate_of_change(self.get_percentile_array(Side.left), Side.left)
        self.rate_of_change(self.get_percentile_array(Side.right), Side.right)

        self.line["CL_CutHt"] = self.CL_CutHt
        self.line["CR_CutHt"] = self.CR_CutHt
        self.line["RDist_Cut"] = self.RDist_Cut
        self.line["LDist_Cut"] = self.LDist_Cut

        self.DynCanTh = (self.CL_CutHt + self.CR_CutHt) / 2
        self.line["DynCanTh"] = self.DynCanTh

        self.prepare_line_buffer()

        fp_left = self.process_single_footprint(Side.left)
        fp_right = self.process_single_footprint(Side.right)
        fp_left.geometry = fp_left.buffer(0.005)
        fp_right.geometry = fp_right.buffer(0.005)
        self.footprint = pd.concat([fp_left, fp_right])
        self.footprint = self.footprint.dissolve()
        self.footprint.geometry = self.footprint.buffer(-0.005)

    def prepar_ring_buffer(self):
        nrings = 1
        ringdist = 15
        ring_list = self.multiringbuffer(self.line_simp, nrings, ringdist)
        for i in ring_list:
            self.buffer_rings.append(BufferRing(i, Side.left))

        nrings = -1
        ringdist = -15
        ring_list = self.multiringbuffer(self.line_simp, nrings, ringdist)
        for i in ring_list:
            self.buffer_rings.append(BufferRing(i, Side.right))

    def cal_percentileRing(self, ring):
        try:
            line_buffer = ring.geometry
            if line_buffer.is_empty or shapely.is_missing(line_buffer):
                return None
            if line_buffer.has_z:
                line_buffer = ops.transform(lambda x, y, z=None: (x, y), line_buffer)

        except Exception as e:
            print(f"cal_percentileRing: {e}")

        # TODO: temporary workaround for exception causing not percentile defined
        try:
            clipped_raster, _ = clip_raster(self.in_chm, line_buffer, 0)
            clipped_raster = np.squeeze(clipped_raster, axis=0)

            # mask all -9999 (nodata) value cells
            masked_raster = np.ma.masked_where(clipped_raster == BT_NODATA, clipped_raster)
            filled_raster = np.ma.filled(masked_raster, np.nan)

            # Calculate the percentile
            percentile = np.nanpercentile(filled_raster, 50)

            if percentile > 1:
                ring.Dyn_Canopy_Threshold = percentile * (0.3)
            else:
                ring.Dyn_Canopy_Threshold = 1

            ring.percentile = percentile
        except Exception as e:
            print(e)
            print("Default values are used.")

        return ring

    def get_percentile_array(self, side):
        per_array = []
        for item in self.buffer_rings:
            if item.side == side:
                per_array.append(item.percentile)

        return per_array

    def rate_of_change(self, percentile_array, side):
        x = percentile_array

        # Since the x interval is 1 unit, the array 'diff' is the rate of change (slope)
        diff = np.ediff1d(x)
        cut_dist = len(x) / 5

        median_percentile = np.nanmedian(x)
        if not np.isnan(median_percentile):
            cut_percentile = float(math.floor(median_percentile))
        else:
            cut_percentile = 0.5
        found = False
        changes = 1.50
        Change = np.insert(diff, 0, 0)
        scale_down = 1.0

        # test the rate of change is > than 150% (1.5), if it is
        # no result found then lower to 140% (1.4) until 110% (1.1)
        try:
            while not found and changes >= 1.1:
                for ii in range(0, len(Change) - 1):
                    if x[ii] >= 0.5:
                        if (Change[ii]) >= changes:
                            cut_dist = (ii + 1) * scale_down
                            cut_percentile = math.floor(x[ii])

                            if 0.5 >= cut_percentile:
                                if cut_dist > 5:
                                    cut_percentile = 2
                                    cut_dist = cut_dist * scale_down**3
                                    # @<0.5  found and modified
                            elif 0.5 < cut_percentile <= 5.0:
                                if cut_dist > 6:
                                    cut_dist = cut_dist * scale_down**3  # 4.0
                                    # @0.5-5.0  found and modified
                            elif 5.0 < cut_percentile <= 10.0:
                                if cut_dist > 8:  # 5
                                    cut_dist = cut_dist * scale_down**3
                                    # @5-10  found and modified
                            elif 10.0 < cut_percentile <= 15:
                                if cut_dist > 5:
                                    cut_dist = cut_dist * scale_down**3  # 5.5
                                    #  @10-15  found and modified
                            elif 15 < cut_percentile:
                                if cut_dist > 4:
                                    cut_dist = cut_dist * scale_down**2
                                    cut_percentile = 15.5
                                    #  @>15  found and modified
                            found = True
                            # rate of change found
                            break
                changes = changes - 0.1

        except IndexError:
            pass

        # if still is no result found, lower to 10% (1.1), if no result found then default is used
        if not found:
            if 0.5 >= median_percentile:
                cut_dist = 4 * scale_down  # 3
                cut_percentile = 0.5
            elif 0.5 < median_percentile <= 5.0:
                cut_dist = 4.5 * scale_down  # 4.0
                cut_percentile = math.floor(median_percentile)
            elif 5.0 < median_percentile <= 10.0:
                cut_dist = 5.5 * scale_down  # 5
                cut_percentile = math.floor(median_percentile)
            elif 10.0 < median_percentile <= 15:
                cut_dist = 6 * scale_down  # 5.5
                cut_percentile = math.floor(median_percentile)
            elif 15 < median_percentile:
                cut_dist = 5 * scale_down  # 5
                cut_percentile = 15.5

        if side == Side.right:
            self.RDist_Cut = cut_dist
            self.CR_CutHt = float(cut_percentile)
        elif side == Side.left:
            self.LDist_Cut = cut_dist
            self.CL_CutHt = float(cut_percentile)

    def multiringbuffer(self, df, nrings, ringdist):
        """
        Buffers an input DataFrames geometry nring (number of rings) times, with a distance between
        rings of ringdist and returns a list of non overlapping buffers
        """

        rings = []  # A list to hold the individual buffers
        line = df.geometry.iloc[0]
        for ring in np.arange(0, ringdist, nrings):  # For each ring (1, 2, 3, ..., nrings)
            big_ring = line.buffer(
                nrings + ring, single_sided=True, cap_style="flat"
            )  # Create one big buffer
            small_ring = line.buffer(
                ring, single_sided=True, cap_style="flat"
            )  # Create one smaller one
            the_ring = big_ring.difference(
                small_ring
            )  # Difference the big with the small to create a ring
            if (
                ~shapely.is_empty(the_ring)
                or ~shapely.is_missing(the_ring)
                or not None
                or ~the_ring.area == 0
            ):
                if isinstance(the_ring, shapely.MultiPolygon) or isinstance(
                    the_ring, shapely.Polygon
                ):
                    rings.append(the_ring)  # Append the ring to the rings list
                else:
                    if isinstance(the_ring, shapely.GeometryCollection):
                        for i in range(0, len(the_ring.geoms)):
                            if not isinstance(the_ring.geoms[i], shapely.LineString):
                                rings.append(the_ring.geoms[i])

        return rings  # return the list

    def prepare_line_buffer(self):
        line = self.line.geometry.iloc[0]
        buffer_left_1 = line.buffer(
            distance=self.max_ln_width + 1,
            cap_style=3,
            single_sided=True,
        )

        buffer_left_2 = line.buffer(
            distance=-1,
            cap_style=3,
            single_sided=True,
        )

        self.buffer_left = ops.unary_union([buffer_left_1, buffer_left_2])

        buffer_right_1 = line.buffer(
            distance=-self.max_ln_width - 1,
            cap_style=3,
            single_sided=True,
        )
        buffer_right_2 = line.buffer(distance=1, cap_style=3, single_sided=True)

        self.buffer_right = ops.unary_union([buffer_right_1, buffer_right_2])

    def dyn_canopy_cost_raster(self, side):
        in_chm_raster = self.in_chm
        tree_radius = self.tree_radius
        max_line_dist = self.max_line_dist
        canopy_avoid = self.canopy_avoidance
        exponent = self.exponent
        line_df = self.line
        out_meta = self.out_meta

        canopy_thresh_percentage = self.canopy_thresh_percentage / 100

        if side == Side.left:
            canopy_ht_threshold = line_df.CL_CutHt * canopy_thresh_percentage
            Cut_Dist = self.LDist_Cut
            line_buffer = self.buffer_left
        elif side == Side.right:
            canopy_ht_threshold = line_df.CR_CutHt * canopy_thresh_percentage
            Cut_Dist = self.RDist_Cut
            line_buffer = self.buffer_right
        else:
            canopy_ht_threshold = 0.5

        canopy_ht_threshold = float(canopy_ht_threshold)
        if canopy_ht_threshold <= 0:
            canopy_ht_threshold = 0.5

        # get the round up integer number for tree search radius
        tree_radius = float(tree_radius)
        max_line_dist = float(max_line_dist)
        canopy_avoid = float(canopy_avoid)
        cost_raster_exponent = float(exponent)

        try:
            clipped_rasterC, out_meta = clip_raster(in_chm_raster, line_buffer, 0)
            negative_cost_clip, dyn_canopy_ndarray = cost_raster_2nd_version(
                clipped_rasterC,
                out_meta,
                tree_radius,
                canopy_ht_threshold,
                max_line_dist,
                canopy_avoid,
                cost_raster_exponent,
            )

            return dyn_canopy_ndarray, negative_cost_clip, out_meta, Cut_Dist

        except Exception as e:
            print(f"dyn_canopy_cost_raster: {e}")
            return None

    def process_single_footprint(self, side):
        # this will change segment content, and parameters will be changed
        in_canopy_r, in_cost_r, in_meta, Cut_Dist = self.dyn_canopy_cost_raster(side)

        if np.isnan(in_canopy_r).all():
            print("Canopy raster empty")

        if np.isnan(in_cost_r).all():
            print("Cost raster empty")

        exp_shk_cell = self.exponent  # TODO: duplicate vars
        no_data = self.nodata

        shapefile_proj = self.line.crs
        in_transform = in_meta["transform"]

        segment_list = []

        feat = self.line.geometry.iloc[0]
        for coord in feat.coords:
            segment_list.append(coord)

        cell_size_x = in_transform[0]
        cell_size_y = -in_transform[4]

        # Work out the corridor from both end of the centerline
        try:
            if len(in_cost_r.shape) > 2:
                in_cost_r = np.squeeze(in_cost_r, axis=0)

            remove_nan_from_array(in_cost_r)
            in_cost_r[in_cost_r == no_data] = np.inf

            # generate 1m interval points along line
            distance_delta = 1
            distances = np.arange(0, feat.length, distance_delta)
            multipoint_along_line = [
                feat.interpolate(distance) for distance in distances
            ]
            multipoint_along_line.append(Point(segment_list[-1]))
            # Rasterize points along line
            rasterized_points_Alongln = rasterize(
                multipoint_along_line,
                out_shape=in_cost_r.shape,
                transform=in_transform,
                fill=0,
                all_touched=True,
                default_value=1,
            )
            points_Alongln = np.transpose(np.nonzero(rasterized_points_Alongln))

            # Find minimum cost paths through an N-d costs array.
            mcp_flexible1 = MCP_Flexible(
                in_cost_r, sampling=(cell_size_x, cell_size_y), fully_connected=True
            )
            flex_cost_alongLn, flex_back_alongLn = mcp_flexible1.find_costs(
                starts=points_Alongln
            )

            # Generate corridor
            corridor = flex_cost_alongLn
            corridor = np.ma.masked_invalid(corridor)

            # Calculate minimum value of corridor raster
            if np.ma.min(corridor) is not None:
                corr_min = float(np.ma.min(corridor))
            else:
                corr_min = 0.5

            # normalize corridor raster by deducting corr_min
            corridor_norm = corridor - corr_min

            # Set minimum as zero and save minimum file
            corridor_th_value = Cut_Dist / cell_size_x
            if corridor_th_value < 0:  # if no threshold found, use default value
                corridor_th_value = FP_CORRIDOR_THRESHOLD / cell_size_x

            corridor_thresh = np.ma.where(corridor_norm >= corridor_th_value, 1.0, 0.0)
            clean_raster = morph_raster(
                corridor_thresh, in_canopy_r, exp_shk_cell, cell_size_x
            )

            # create mask for non-polygon area
            mask = np.where(clean_raster == 1, True, False)
            if clean_raster.dtype == np.int64:
                clean_raster = clean_raster.astype(np.int32)

            # Process: ndarray to shapely Polygon
            out_polygon = shapes(clean_raster, mask=mask, transform=in_transform)

            # create a shapely MultiPolygon
            multi_polygon = []
            for poly, value in out_polygon:
                multi_polygon.append(shape(poly))
            poly = MultiPolygon(multi_polygon)

            # create a pandas DataFrame for the FP
            out_data = pd.DataFrame(
                {
                    "CorriThresh": [corridor_th_value],
                    "geometry": [poly]
                }
            )
            out_gdata = gpd.GeoDataFrame(out_data, geometry="geometry", crs=shapefile_proj)

            return out_gdata

        except Exception as e:
            print("Exception: {}".format(e))


if __name__ == "__main__":
    in_file = r"D:\BERATools\Surmont_New_AOI\Developement\centerline.shp"
    in_chm = r"D:\BERATools\Surmont_New_AOI\Developement\CHM_New_AOI_2022.tif"
    out_file_percentile = r"D:\BERATools\Surmont_New_AOI\Developement\centerline_percentile.shp"
    out_file_fp = r"D:\BERATools\Surmont_New_AOI\Developement\footprints.shp"

    footprint = FootprintCanopy(in_file, in_chm)
    footprint.compute()

    footprint.lines.to_file(out_file_percentile)
    footprint.footprints.to_file(out_file_fp)
