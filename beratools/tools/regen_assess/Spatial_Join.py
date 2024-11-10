import os,sys
from pathlib import Path
from inspect import getsourcefile
if __name__ == '__main__':

    current_file = Path(getsourcefile(lambda: 0)).resolve()
    main_dir = current_file.parents[1]
    data_dir = os.path.join(main_dir, "regen_assess")
    tool_dir = current_file.parents[0]
    sys.path.insert(0, main_dir.as_posix())
    sys.path.insert(0, tool_dir.as_posix())
from regen_assess.common_1 import *


def spatial_join(in_tree, in_Ecosite, in_FFP, seedlings_path):
    df_mask = gpd.read_file(in_FFP)
    print("Loading trees data.......")
    trees = gpd.read_file(in_tree, engine="pyogrio", use_arrow=True, mask=df_mask)
    # print("Saving on FP trees data.......")
    # selected_trees=os.path.join(data,"Trees\\On_FP_seedlings_classes_above30cm.gpkg")
    # trees.to_file(selected_trees, index=False, driver="GPKG")
    print("Loading ecosites data.......")
    ecosites = gpd.read_file(in_Ecosite, engine="pyogrio", use_arrow=True)
    if trees.crs != ecosites.crs:
        ecosites = ecosites.to_crs(trees.crs)

    # seedlings_path = os.path.join(data_dir, working_dir, "Trees\\Seedlings_above30cm_Ecosite_onFP.gpkg")
    print("Joining ecosites and trees data.......")
    Ecosite_trees = gpd.sjoin(trees, ecosites, predicate="intersects")
    Ecosite_trees = Ecosite_trees.convert_dtypes()
    # plot_trees.to_parquet(join_file, index=False)
    Ecosite_trees.to_file(seedlings_path, index=False, driver="GPKG")
    print(f"Updated data saved to {seedlings_path}")


if __name__ == '__main__':
    print('Spatial Join started at {}'.format(time.strftime("%b %Y %H:%M:%S", time.localtime())))
    current_file = Path(getsourcefile(lambda: 0)).resolve()
    main_dir = r"/"
    data_dir = r"I:\BERATools\Workshop"
    output_dir = r"I:\BERATools\Workshop\Output"

    in_tree = os.path.join(data_dir, working_dir, "Helpers\\All_trees_above30cm_v4.gpkg")
    in_Ecosite =os.path.join(data_dir, working_dir, "Helpers\\Ecosite_SmoothPolygon_Clipped.shp")
    in_FFP = os.path.join(output_dir, working_dir, "Intermediate\\FFP_frm_v17.shp")
    out_footprint = os.path.join(output_dir, working_dir, "Intermediate\\Seedlings_above30cm_Ecosite_onFP.gpkg")

    spatial_join(in_tree, in_Ecosite, in_FFP, out_footprint)
