import os
import yaml
import time
import geopandas as gpd
import numpy as np
import rasterio
from rasterio.mask import mask
from shapely.geometry import Polygon
from shapely.validation import make_valid
from tqdm import tqdm
from datetime import datetime
import json
import matplotlib.pyplot as plt

def read_masked_data(src, geom):
    out_image, out_transform = mask(src, [geom], crop=True, nodata=np.nan)
    masked_data = out_image[0]  # First band of the masked data
    return masked_data

def calculate_coverage(data, height_threshold_cm, resolution):
    height_threshold_m = height_threshold_cm / 100.0
    valid_pixels = (data > -0.01) & (data <= 50)
    total_pixels = np.sum(valid_pixels)

    recovered_pixels = np.sum(data[valid_pixels] > height_threshold_m)
    coverage = (recovered_pixels / total_pixels * 100) if total_pixels > 0 else 0
    area_sq_meters = total_pixels * resolution**2

    return coverage, total_pixels, recovered_pixels, area_sq_meters

def calculate_adjacency_coverage(buffer_data, height_threshold_cm):
    height_threshold_m = height_threshold_cm / 100.0
    valid_pixels = (buffer_data > -0.01) & (buffer_data <= 50)
    total_pixels = np.sum(valid_pixels)

    adjacency_coverage = np.sum(buffer_data[valid_pixels] > height_threshold_m) / total_pixels * 100 if total_pixels > 0 else 0
    return adjacency_coverage, total_pixels

def determine_dominant_ecosite_vector(ecosite_gdf, ground_geom):
    # Ecosite mapping from class attribute values in the vector dataset
    ecosite_mapping = {
        255: 'Transitional',
        0: 'Exclusion',
        76: 'Mesic_upland',
        194: 'Treed_wetland',
        115: 'Dry_upland'
    }

    intersecting_ecosites = ecosite_gdf[ecosite_gdf.intersects(ground_geom)]
    if intersecting_ecosites.empty:
        return 'Low_density_treed_wetland'

    clipped_ecosites = gpd.clip(intersecting_ecosites, ground_geom)
    ecosite_counts = clipped_ecosites['Band 1'].value_counts()
    non_exclusion_ecosites = ecosite_counts[ecosite_counts.index != 0]

    if not non_exclusion_ecosites.empty:
        return ecosite_mapping.get(non_exclusion_ecosites.idxmax(), 'Low_density_treed_wetland')
    return 'Error'


def process_segments(config):
    canopy_footprint_path = config['datasets']['canopy_footprint']
    ground_footprint_path = config['datasets']['ground_footprint']
    chm_path = config['lidar_data']['chm']
    buffer_size = config['recovery_parameters']['buffer_size']

    canopy_gdf = gpd.read_file(canopy_footprint_path)
    ground_gdf = gpd.read_file(ground_footprint_path)
    segmented_gdf = gpd.read_file(config['datasets']['segmented_ground_footprint'])
    ecosite_gdf = gpd.read_file(config['datasets']['ecosite_raster']).to_crs(ground_gdf.crs)

    canopy_gdf = canopy_gdf.to_crs(ground_gdf.crs)
    segmented_gdf = segmented_gdf.to_crs(ground_gdf.crs)

    results = []
    for _, ground_row in tqdm(ground_gdf.iterrows(), total=len(ground_gdf), desc="Processing segments"):
        ground_geom = ground_row.geometry
        ground_attrs = ground_row.drop(['geometry', 'id']).to_dict()

        with rasterio.open(chm_path) as src:
            ground_data = read_masked_data(src, ground_geom)

            dominant_ecosite = determine_dominant_ecosite_vector(ecosite_gdf, ground_geom)
            height_threshold = config['height_thresholds'].get(dominant_ecosite, 60)
            resolution = src.res[0]

            coverage, total_pixels, recovered_pixels, area_sq_meters = calculate_coverage(ground_data, height_threshold, resolution)
            buffer_data = read_masked_data(src, ground_geom.buffer(buffer_size))
            adjacency_coverage, _ = calculate_adjacency_coverage(buffer_data, height_threshold)

            segment_data = {
                'OLnSEG': 0,
                'geometry': ground_geom,
                'dominant_ecosite': dominant_ecosite,
                'coverage': coverage,
                'adjacency_coverage': adjacency_coverage,
                'total_pixels': total_pixels,
                'recovered_pixels': recovered_pixels,
            }
            segment_data.update(ground_attrs)
            results.append(segment_data)
    return results

def classify_segments(results, config):
    classifications = []
    for result in results:
        coverage_info = result['coverage']
        adjacency_coverage = result['adjacency_coverage']
        total_pixels = result['total_pixels']
        recovered_pixels = result['recovered_pixels']

        segment_classification = {
            'geometry': result['geometry'],
            'dominant_ecosite': result['dominant_ecosite'],
            'coverage': coverage_info,
            'adjacency_coverage': adjacency_coverage,
            'recovered_pixels': recovered_pixels,
            'total_pixels': total_pixels
        }

        for key, value in result.items():
            if key not in segment_classification:
                segment_classification[key] = value

        for scenario_name, scenario in config['scenarios'].items():
            classification = None
            if scenario_name == 'D':
                coverage_ratio = coverage_info / adjacency_coverage if adjacency_coverage > 0 else 0
                if coverage_info > adjacency_coverage:
                    classification = "Advanced_regeneration"
                else:
                    for status, condition in scenario.items():
                        if condition['cover_range'][0] <= coverage_ratio < condition['cover_range'][1]:
                            classification = status
                            break
            else:
                for status, condition in scenario.items():
                    if condition['cover_range'][0] <= coverage_info < condition['cover_range'][1]:
                        classification = status
                        break

            segment_classification[f'Ass_Stat{scenario_name}'] = classification
        classifications.append(segment_classification)
    return classifications

def save_output(classification_results, output_path):
    gdf = gpd.GeoDataFrame(classification_results, crs="EPSG:2956")
    gdf['geometry'] = gdf['geometry'].apply(make_valid)
    try:
        print(f"Saving GeoPackage with full attributes to: {output_path}")
        gdf.to_file(output_path, driver='ESRI Shapefile')
    except Exception as e:
        print(f"Error saving output: {e}")

def save_metadata(config, metadata_path):
    metadata = {
        "process_time": str(datetime.now()),
        "configuration": config  # Include the entire YAML configuration
    }
    with open(metadata_path, 'w') as metadata_file:
        json.dump(metadata, metadata_file, indent=4)

def regenass_main(config_path):
    config_path = r"I:\BERATools\Workshop\Code\regen_assess\config.yaml"

    start_time = time.time()
    regenass(config_path)

    elapsed_time = time.time() - start_time
    print(f"Processing completed in {elapsed_time:.2f} seconds.")

def update_config(config, chm, canopy_footprint, ground_footprint, recovery_ass, ecosite_raster):
    config['datasets']['ecosite_raster'] = ecosite_raster
    config['datasets']['canopy_footprint'] = canopy_footprint
    config['datasets']['ground_footprint'] = ground_footprint
    config['datasets']['segmented_ground_footprint'] = ground_footprint
    config['datasets']['recovery_ass'] = recovery_ass
    config['lidar_data']['chm'] = chm

    return config


def regenass(config_path, chm, canopy_footprint, ground_footprint, recovery_ass, ecosite_raster):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)

    config = update_config(config, chm, canopy_footprint, ground_footprint, recovery_ass, ecosite_raster)

    results = process_segments(config)
    classification_results = classify_segments(results, config)
    output_gpkg_path = config['datasets']['recovery_ass']
    save_output(classification_results, output_gpkg_path)
    metadata_path = config['datasets']['recovery_ass'].replace(".shp", ".json")
    save_metadata(config, metadata_path)


if __name__ == '__main__':
    regenass_main(config_path)
