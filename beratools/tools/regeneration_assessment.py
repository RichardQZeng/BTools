# 1. regenass_no_metaflow
# 2. spatial join
# 3. assign species for seedlings
# 4. in tree biomass
# 5. plot biomass
from regen_assess.regenass_no_metaflow import *
from regen_assess.Spatial_Join import *
from regen_assess.Assign_Species_for_Seedling import *
from regen_assess.In_Tree_BioMass import *
from regen_assess.Plot_BioMass import *

from pathlib import Path
from inspect import getsourcefile

if __name__ == "__main__":
    current_file = Path(getsourcefile(lambda: 0)).resolve()
    btool_dir = current_file.parents[2]
    sys.path.insert(0, btool_dir.as_posix())

from common import *

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def regeneration_assessment(callback, in_chm, canopy_fp, ground_fp,
               out_dir, processes, verbose):
    config_path = r"I:\BERATools\Workshop\Code\regen_assess\config.yaml"

    helper_dir = r"I:\BERATools\Workshop"
    output_dir = r"I:\BERATools\Workshop\Output"

    chm = r"I:/BERATools/Workshop/Input/Scenario_Paper_CHM2022.tif"
    canopy_footprint = r"I:/BERATools/Workshop/Input/footprint_rel.shp"
    ground_footprint = r"I:/BERATools/Workshop/Input/footprint_fixed.shp"

    ecosite_raster = os.path.join(helper_dir, r"Helpers/vegetation_types_Surmont.gpkg")
    in_tree = os.path.join(helper_dir, working_dir, "Helpers\\All_trees_above30cm_v4.gpkg")
    in_Ecosite =os.path.join(helper_dir, working_dir, "Helpers\\Ecosite_SmoothPolygon_Clipped.shp")
    in_landcover = os.path.join(helper_dir, working_dir, "Helpers\\Landcover_fr_veg1.tif")

    recovery_ass = os.path.join(output_dir, "Intermediate/FFP_frm_v17.shp")
    in_FFP = os.path.join(output_dir, working_dir, "Intermediate\\FFP_frm_v17.shp")
    seedlings_path = os.path.join(output_dir, working_dir, "Intermediate\\Seedlings_above30cm_Ecosite_onFP.gpkg")
    tree_eco = os.path.join(output_dir, working_dir, "Intermediate\\Scenario_Paper_seedlings_classes_above30cm.gpkg")
    tree_biomass = os.path.join(output_dir, working_dir, "Intermediate\\Scenario_Paper_seedlings_classes_above30cm_biomass.gpkg")

    regenass(config_path, in_chm, canopy_fp, ground_fp, recovery_ass, ecosite_raster)
    spatial_join(in_tree, in_Ecosite, in_FFP, seedlings_path)
    assign_species_for_seedling(in_landcover, seedlings_path, tree_eco)
    in_tree_biomass(tree_eco, tree_biomass)
    plot_biomass(tree_biomass, in_FFP, in_Ecosite, output_dir)

if __name__ == '__main__':
    in_args, in_verbose = check_arguments()
    start_time = time.time()
    regeneration_assessment(print, **in_args.input, processes=int(in_args.processes), verbose=in_verbose)
    print('Elapsed time: {}'.format(time.time() - start_time))