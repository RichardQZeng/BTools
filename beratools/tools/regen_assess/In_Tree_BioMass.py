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


def in_tree_biomass(in_tree, out_tree):
    df_tree = read_data2gpd(in_tree)
    df_tree = chk_columns(df_tree)
    df_tree["distribute"] = 1
    df_tree, out_tree = BioMassC_attributes(df_tree, out_tree)
    df_tree, out_tree = BioMass0_attributes(df_tree, out_tree)
    BioMassN_attributes(df_tree, out_tree, yearn=40)


if __name__ == '__main__':

    current_file = Path(getsourcefile(lambda: 0)).resolve()
    main_dir = current_file.parents[1]
    data = os.path.join(main_dir, "regen_assess")
    start_time = time.time()
    print('Tree Biomass Attributes started at {}'.format(time.strftime("%b %Y %H:%M:%S", time.localtime())))
    
    in_tree=os.path.join(data,working_dir,"Intermediate\\Scenario_Paper_seedlings_classes_above30cm.gpkg")
    out_tree = os.path.join(data, working_dir,"Intermediate\\Scenario_Paper_seedlings_classes_above30cm_biomass.gpkg")

    in_tree_biomass(in_tree, out_tree)

    print('Elapsed time: {}'.format(time.time() - start_time, 5))
