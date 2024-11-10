import os,sys
from pathlib import Path
from inspect import getsourcefile
if __name__ == '__main__':

    current_file = Path(getsourcefile(lambda: 0)).resolve()
    main_dir = current_file.parents[1]
    data = os.path.join(main_dir, "regen_assess")
    tool_dir = current_file.parents[0]
    sys.path.insert(0, main_dir.as_posix())
    sys.path.insert(0, tool_dir.as_posix())

from regen_assess.common_1 import *
import geopandas as gpd
import pandas
import rasterio
import numpy as np
import time, os
from pathlib import Path
from inspect import getsourcefile


# Function to get landcover class name
def get_landcover_class_name(in_df, dataset, transform):

    def assign_class(row):
        point=row['geometry']
        col, row = ~transform * (point.x, point.y)
        col, row = int(col), int(row)
        class_id = dataset[row, col]
        return class_names.get(class_id, 'Unknown')  # Return the class name or 'Unknown'

    in_df['landcover_class_name'] =in_df.apply(assign_class,axis=1)

    return  in_df

class_names = {
    0: 'Open water',
    1: 'Deciduous forest',
    2: 'Coniferous forest',
    3: 'Mixed forest',
    4: 'Pine forest',
    5: 'Shrub upland dry',
    6: 'Grass upland dry',
    7: 'Marsh Wetland',
    8: 'Fen Wetland',
    9: 'Fen Wetland Treed',
    10: 'Bog Shrub Wetland',
    11: 'Bog Wetland Treed',
    12: 'Shrub Swamp',
    13: 'Coniferous Swamp',
    14: 'Deciduous Swamp'
}


# Define species probabilities for each landcover class by name
species_mapping = {
    'Open water': [('None', 100)],  # Assuming no seedlings in open water
    'Deciduous forest': [('White Birch', 30), ('Balsam Poplar', 10),('Balsam Fir',10), ('Trembling Aspen', 40), ('Black Spruce', 5), ('White Spruce', 5)],
    'Coniferous forest': [('Black Spruce', 50),('White Spruce', 20), ('Tamarack', 20), ('Trembling Aspen', 5), ('Alder', 3),('Willow',2)],
    'Mixed forest': [('Black Spruce', 20),('White Spruce', 10), ('Trembling Aspen', 30), ('White Birch', 20), ('Alder', 15),('Willow',5)],
    'Pine forest':[('Jack Pine', 100)],
    'Shrub upland dry':[('Alder',50),('Willow',50)],
    'Grass upland dry':[('Grasses',75),('Sedges',25)],
    'Marsh Wetland':[('Tamarack', 30), ('Black Spruce', 40),('White Spruce', 20), ('Alder', 7),('Willow',3)],
    'Fen Wetland': [('Tamarack', 30), ('Black Spruce', 30),('White Spruce', 10), ('Alder', 10),('Willow',10), ('White Birch', 10)],
    'Fen Wetland Treed': [('Black Spruce', 30),('White Spruce', 20), ('Tamarack', 30), ('Trembling Aspen', 10), ('Alder', 5),('Willow',5)],
    'Bog Shrub Wetland':[('Bog Rosemary',50),('Bog Laurel',50)],
    'Bog Wetland Treed':[('Bog Birch',50),('Bog Tea',50)],
    'Shrub Swamp':[('Balsam Poplar', 10),('Balsam Fir',10), ('Tamarack', 10), ('Black Spruce', 10), ('Trembling Aspen', 40), ('Alder', 10),('Willow',10)],
    'Coniferous Swamp': [('Black Spruce', 40),('White Spruce', 20), ('Tamarack', 20), ('Shrubs', 20)],
    'Deciduous Swamp': [('Balsam Poplar', 10),('Balsam Fir',10), ('Tamarack', 10), ('Black Spruce', 7),('White Spruce', 3), ('Trembling Aspen', 40), ('Alder', 10),('Willow',10)],


}

def assign_species(class_name):
    if class_name in species_mapping:
        species, probabilities = zip(*species_mapping[class_name])
        probabilities = np.array(probabilities, dtype=float) / sum(probabilities)
        return np.random.choice(species, p=probabilities)
    else:
        return 'Unknown'


def assign_species_for_seedling(raster_path, seedlings_path, tree_eco):
    global transform
    # Open the landcover raster
    print("Loading raster.......")
    with rasterio.open(raster_path) as src:
        landcover_data = src.read(1)  # Read the first band
        transform = src.transform
    # Load the seedlings geopackage
    # pandas_seedlings = pandas.(seedlings_path,npartitions=5)
    print("Loading trees.......")
    seedlings = gpd.read_file(seedlings_path, engine="pyogrio", use_arrow=True)
    # Ensure CRS compatibility
    if seedlings.crs != src.crs:
        seedlings = seedlings.to_crs(src.crs)
    # Assign landcover class names
    print("Assigning landcover.......")
    seedlings = get_landcover_class_name(seedlings, landcover_data, transform)
    print('Number of seedlings: ', len(seedlings))
    # Assign species to each seedling based on class name
    print("Assigning species.......")
    seedlings['species'] = seedlings['landcover_class_name'].apply(assign_species)
    # Save the updated seedlings back to a new GeoPackage
    print("Saving Tree species.......")
    seedlings.to_file(tree_eco, driver='GPKG')
    print(f"Updated data saved to {tree_eco}")


if __name__ == '__main__':
    start_time = time.time()
    print('Assign Class for seedlings started at {}'.format(time.strftime("%b %Y %H:%M:%S", time.localtime())))
    current_file = Path(getsourcefile(lambda: 0)).resolve()
    main_dir = current_file.parents[1]
    data_dir = r"I:\BERATools\Workshop"
    output_dir = r"I:\BERATools\Workshop\Output"

    raster_path = os.path.join(data_dir,working_dir,"Helpers\\Landcover_fr_veg1.tif")
    seedlings_path = os.path.join(data_dir,working_dir,"Intermediate\\Seedlings_above30cm_Ecosite_onFP.gpkg")
    tree_eco =os.path.join(data_dir, working_dir, "Intermediate\\Scenario_Paper_seedlings_classes_above30cm.gpkg")
    # Path to the raster and vector data files

    assign_species_for_seedling(raster_path, seedlings_path, tree_eco)
    print('Elapsed time: {}'.format(time.time() - start_time, 5))

