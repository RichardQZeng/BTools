"""
Copyright (C) 2025 Applied Geospatial Research Group.

This script is licensed under the GNU General Public License v3.0.
See <https://gnu.org/licenses/gpl-3.0> for full license details.

---------------------------------------------------------------------------

File: algo_tiler.py
Author: Richard Zeng

Description:
    This script is part of the BERA Tools.
    Webpage: https://github.com/appliedgrg/beratools

    The purpose of this script is to provide algorithm for 
    partitioning vector and raster.
"""

import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import shapely.geometry as sh_geom
import shapely.ops as sh_ops
from rasterio.mask import mask
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity


class DensityBasedClustering:
    """Density-based clustering of line features."""

    def __init__(
        self,
        in_line,
        in_raster,
        out_file,
        n_clusters=8,
        tile_buffer=50,
        bandwidth=1.5,
        layer=None,
    ):
        self.input_file = in_line
        self.input_raster = in_raster
        self.output_file = out_file
        self.n_clusters = n_clusters
        self.tile_buffer = tile_buffer
        self.bandwidth = bandwidth
        self.layer = layer

        self.gdf = None  # Initialize gdf attribute

    def read_points_from_geopackage(self):
        """Read points from GeoPackage and keep the 'group' field."""
        # Load the lines from GeoPackage
        self.gdf = gpd.read_file(self.input_file, layer=self.layer)

        # Merge lines by group
        grouped = (
            self.gdf.groupby("group")["geometry"]
            .apply(sh_ops.unary_union)
            .reset_index()
        )
        merged_gdf = gpd.GeoDataFrame(grouped, geometry=grouped["geometry"])

        # Generate centroids for the merged MultiLineStrings
        merged_gdf["centroid"] = merged_gdf.geometry.centroid

        # Calculate line lengths and assign to centroids as 'weight'
        merged_gdf["weight"] = merged_gdf.geometry.length
        merged_gdf = merged_gdf.drop(columns="geometry")

        # Create a new GeoDataFrame with centroids as the geometry
        centroid_gdf = gpd.GeoDataFrame(merged_gdf, geometry="centroid")

        # Ensure CRS is preserved from the original GeoDataFrame
        centroid_gdf.set_crs(self.gdf.crs, allow_override=True, inplace=True)

        # Filter for valid Point geometries
        centroid_gdf = centroid_gdf[
            centroid_gdf.geometry.apply(
                lambda geom: isinstance(geom, sh_geom.Point) and not geom.is_empty
            )
        ]
        return centroid_gdf, self.gdf

    def extract_coordinates_and_weights(self, gdf):
        """Extract coordinates and weights from GeoDataFrame."""
        points = np.array([point.coords[0] for point in gdf.geometry])
        weights = gdf[
            "weight"
        ].values  # Assuming 'weight' field exists in the GeoDataFrame
        return points, weights

    def estimate_density(self, points):
        """Estimate density using Kernel Density Estimation (KDE)."""
        kde = KernelDensity(kernel="gaussian", bandwidth=self.bandwidth)
        kde.fit(points)
        return kde

    def sample_points(self, kde, grid_points, n_samples=200):
        """Sample additional points based on density."""
        log_density = kde.score_samples(grid_points)
        density = np.exp(log_density)
        probabilities = density.ravel() / density.sum()
        sampled_indices = np.random.choice(
            grid_points.shape[0], size=n_samples, p=probabilities
        )
        return grid_points[sampled_indices]

    def initial_clustering(self, points):
        """Perform KMeans clustering."""
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        kmeans_labels = kmeans.fit_predict(points)
        return kmeans_labels, kmeans

    def rebalance_with_weight_sum_constraint(
        self, kmeans_labels, points, weights, kmeans, tolerance=0.5, max_iterations=20
    ):
        """Rebalance clusters with weight sum constraints."""
        if len(kmeans_labels) != len(weights):
            raise ValueError(
                f"""Length mismatch: kmeans_labels has {len(kmeans_labels)} entries,
                but weights has {len(weights)} entries."""
            )

        total_weight = np.sum(weights)
        target_weight = total_weight / self.n_clusters

        for iteration in range(max_iterations):
            cluster_weights = np.zeros(self.n_clusters)
            for cluster_id in range(self.n_clusters):
                cluster_weights[cluster_id] = np.sum(
                    weights[kmeans_labels == cluster_id]
                )

            weight_differences = cluster_weights - target_weight
            imbalance = np.abs(weight_differences) > tolerance * target_weight

            if not np.any(imbalance):
                print(f"Rebalancing completed after {iteration + 1} iterations.")
                break

            for cluster_id in range(self.n_clusters):
                if weight_differences[cluster_id] > tolerance * target_weight:
                    excess_points = np.where(kmeans_labels == cluster_id)[0]
                    for idx in excess_points:
                        distances_to_centroids = np.linalg.norm(
                            points[idx] - kmeans.cluster_centers_, axis=1
                        )
                        distances_to_centroids[cluster_id] = np.inf
                        closest_cluster = np.argmin(distances_to_centroids)

                        if (
                            cluster_weights[closest_cluster]
                            < target_weight - tolerance * target_weight
                        ):
                            kmeans_labels[idx] = closest_cluster
                elif weight_differences[cluster_id] < -tolerance * target_weight:
                    deficient_points = np.where(kmeans_labels == cluster_id)[0]
                    for idx in deficient_points:
                        distances_to_centroids = np.linalg.norm(
                            points[idx] - kmeans.cluster_centers_, axis=1
                        )
                        distances_to_centroids[cluster_id] = np.inf
                        closest_cluster = np.argmin(distances_to_centroids)

                        if (
                            cluster_weights[closest_cluster]
                            > target_weight + tolerance * target_weight
                        ):
                            kmeans_labels[idx] = cluster_id
        else:
            print("Maximum iterations reached without achieving perfect balance.")

        self.print_cluster_weight_sums(kmeans_labels, weights)
        return kmeans_labels

    def print_cluster_weight_sums(self, kmeans_labels, weights):
        """Print weight sums for clusters."""
        for cluster_id in range(self.n_clusters):
            cluster_weight_sum = np.sum(weights[kmeans_labels == cluster_id])
            print(f"Cluster {cluster_id} weight sum: {cluster_weight_sum}")

    def save_final_clustering_to_geopackage(
        self,
        all_points,
        all_weights,
        kmeans_labels_final,
        group_field,
        crs,
        gdf,
        centroid_gdf,
    ):
        """Save final clustering result to GeoPackage, keeping 'group' field."""
        # Only assign labels to centroids
        centroid_gdf["cluster"] = kmeans_labels_final[: len(centroid_gdf)]

        # Merge the centroids centroid_gdf with the original gdf based on 'group'
        gdf_merged = gdf.merge(
            centroid_gdf[["group", "cluster"]], on="group", how="left"
        )

        # Save the whole lines dataset with clustering information
        gdf_merged.to_file(self.output_file, layer="final_clusters", driver="GPKG")

        # Save separate layers for each cluster
        for cluster_id in range(self.n_clusters):
            cluster_gdf = gdf_merged[gdf_merged["cluster"] == cluster_id]
            cluster_gdf.to_file(
                self.output_file,
                layer=f"cluster_{cluster_id}",
                driver="GPKG",
            )

    def plot_clusters(self, points, labels, centroids, title):
        """Plot clusters."""
        fig, ax = plt.subplots(figsize=(10, 10))
        scatter = ax.scatter(
            points[:, 0], points[:, 1], c=labels, cmap="tab10", s=50, alpha=0.6
        )
        ax.scatter(
            centroids[:, 0],
            centroids[:, 1],
            c="red",
            marker="*",
            s=200,
            label="Centroids",
        )
        plt.colorbar(scatter, ax=ax, label="Cluster ID")
        ax.set_title(title)
        plt.show()

    def generate_and_clip_rasters(self, kmeans_labels_final):
        """Generate bounding box polygons based on line clusters and clip the raster."""
        parent_folder = os.path.dirname(self.input_file)
        output_folder = os.path.join(parent_folder, "rasters")
        os.makedirs(output_folder, exist_ok=True)

        with rasterio.open(self.input_raster) as src:
            for cluster_id in range(self.n_clusters):
                cluster_lines = self.get_lines_for_cluster(
                    kmeans_labels_final, cluster_id
                )

                if cluster_lines:
                    multi_line = sh_geom.MultiLineString(cluster_lines)

                    # Collect all coordinates from the lines in the multi_line object
                    all_coords = []
                    for line in multi_line.geoms:
                        # Make sure each line is of type LineString
                        if isinstance(line, sh_geom.LineString):
                            coords = list(line.coords)
                            all_coords.extend(coords)
                        else:
                            print(f"Warning: Found non-LineString geom in {cluster_id}")

                    if not all_coords:
                        print(
                            f"Warning: No coordinates found: {cluster_id}. Skipping..."
                        )
                        continue

                    # Create a bounding box from the coordinates
                    min_x = min(coord[0] for coord in all_coords)
                    max_x = max(coord[0] for coord in all_coords)
                    min_y = min(coord[1] for coord in all_coords)
                    max_y = max(coord[1] for coord in all_coords)

                    print(
                        f"""Cluster {cluster_id} BBox:
                        ({min_x}, {min_y}), ({max_x}, {max_y})"""
                    )

                    # Create a Polygon representing the bounding box
                    bounding_box = sh_geom.Polygon(
                        [
                            (min_x, min_y),
                            (max_x, min_y),
                            (max_x, max_y),
                            (min_x, max_y),
                            (min_x, min_y),
                        ]
                    )

                    # Clip the raster with the bounding box
                    out_image, out_transform = mask(src, [bounding_box], crop=True)

                    # Ensure the out_image shape is correct
                    out_image = out_image.squeeze()

                    cluster_raster_path = os.path.join(
                        output_folder, f"cluster_{cluster_id}.tif"
                    )

                    with rasterio.open(
                        cluster_raster_path,
                        "w",
                        driver="GTiff",
                        count=1,
                        dtype=out_image.dtype,
                        crs=src.crs,
                        transform=out_transform,
                        width=out_image.shape[1],
                        height=out_image.shape[0],
                    ) as dest:
                        dest.write(out_image, 1)

                    print(f"Cluster {cluster_id} raster saved to {cluster_raster_path}")
                else:
                    print(f"No lines found: {cluster_id}, skipping raster generation.")

    def get_lines_for_cluster(self, kmeans_labels_final, cluster_id):
        """Retrieve the lines corresponding to a specific cluster."""
        cluster_lines = []
        for idx, centroid in self.centroid_gdf.iterrows():
            if kmeans_labels_final[idx] == cluster_id:
                group_value = centroid["group"]
                # Find the lines in the original gdf that belong to this group
                lines_for_cluster = self.gdf[self.gdf["group"] == group_value]
                cluster_lines.extend(lines_for_cluster["geometry"])

        # flatten any MultiLineString objects into individual LineString objects
        flattened_lines = []
        for line in cluster_lines:
            if isinstance(line, sh_geom.MultiLineString):
                # Extract individual LineStrings from the MultiLineString
                flattened_lines.extend(
                    line.geoms
                )  # `line.geoms` is an iterable of LineString objects
            elif isinstance(line, sh_geom.LineString):
                flattened_lines.append(
                    line
                )  # Directly append the LineString if it's not a MultiLineString

        return flattened_lines

    def run(self):
        """Run the full clustering process."""
        # Step 1: Read points and original lines
        centroid_gdf, gdf = self.read_points_from_geopackage()

        # Assign centroid_gdf to the class attribute
        self.centroid_gdf = centroid_gdf  # Add this line

        # Step 2: Extract coordinates and weights
        points, weights = self.extract_coordinates_and_weights(centroid_gdf)

        # Step 3: Estimate density
        kde = self.estimate_density(points)

        # Step 4: Sample additional points based on density
        x_min, y_min = points.min(axis=0)
        x_max, y_max = points.max(axis=0)
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200)
        )
        grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
        sampled_points = self.sample_points(kde, grid_points, n_samples=200)

        # Combine original and sampled points
        all_points = np.vstack([points, sampled_points])
        all_weights = np.concatenate([weights, np.ones(sampled_points.shape[0])])

        # Preserve the 'group' field for the final output
        group_field = np.concatenate(
            [centroid_gdf["group"].values, np.full(sampled_points.shape[0], -1)]
        )  # Assign default value -1 for sampled points

        # Step 5: Initial clustering
        kmeans_labels_initial, kmeans_initial = self.initial_clustering(points)

        # Assign clusters to the new sampled points
        kmeans_labels_all = np.concatenate(
            [kmeans_labels_initial, kmeans_initial.predict(sampled_points)]
        )

        # Step 6: Rebalance clusters with weight sum constraints
        kmeans_labels_final = self.rebalance_with_weight_sum_constraint(
            kmeans_labels_all, all_points, all_weights, kmeans_initial
        )

        # Step 7: Save final clustering to GeoPackage
        self.save_final_clustering_to_geopackage(
            all_points,
            all_weights,
            kmeans_labels_final,
            group_field,
            centroid_gdf.crs,
            gdf,
            centroid_gdf,
        )

        # Step 8: Generate and clip rasters for each cluster
        self.generate_and_clip_rasters(kmeans_labels_final)

