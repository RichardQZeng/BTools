import functools
import os,sys,math
from pathlib import Path
from inspect import getsourcefile

import numpy as np

if __name__ == '__main__':

    current_file = Path(getsourcefile(lambda: 0)).resolve()
    main_dir = current_file.parents[1]
    output_dir = os.path.join(main_dir, "regen_assess")
    tool_dir = current_file.parents[0]
    sys.path.insert(0, main_dir.as_posix())
    sys.path.insert(0, tool_dir.as_posix())
from regen_assess.common_1 import *


def plot_biomass(in_tree, in_FFP, split_file, output_dir):
    ## Load AOI data
    print("Loading tree data.......")
    df_tree = gpd.read_file(in_tree)
    df_tree = chk_columns(df_tree)
    df_FFP = gpd.read_file(in_FFP).to_crs(df_tree.crs)
    df_FFP_, _ = chk_df_multipart(df_FFP, 'MultiPolygon')
    segmented_gdf, right_segements, left_segements = split_polygon_into_area(df_FFP_, 100, 30)
    save_gpkg(segmented_gdf, os.path.join(output_dir, working_dir, "FFP\\Plot_FP_into_about100m2.gpkg"))
    save_gpkg(left_segements, os.path.join(output_dir, working_dir, "FFP\\LPlot_FP_into_about100m2.gpkg"))
    save_gpkg(right_segements, os.path.join(output_dir, working_dir, "FFP\\RPlot_FP_into_about100m2.gpkg"))
    df_tree = del_joined_index(df_tree)
    print("Spatial joining FP to ecosite.......")
    # split_file = os.path.join(data_dir, working_dir, "Ecosite\\Ecosite_SmoothPolygon_Clipped.shp")
    df_split = gpd.read_file(split_file).to_crs(segmented_gdf.crs)
    df_split_, _ = chk_df_multipart(df_split, 'MultiPolygon')
    splitted_plot = []
    for in_df, side in [(segmented_gdf, ""), (left_segements, "L"), (right_segements, "R")]:
        if side == "":
            org_df = in_df.copy()

        splited_df = in_df.overlay(df_split, how='intersection')
        splited_df = splited_df.explode()
        splited_df['OLnSEG'] = splited_df.groupby(['OLnFID', 'OLnPLT']).cumcount()
        splited_df = splited_df.sort_values(by=['OLnFID', 'OLnPLT', 'OLnSEG'])
        splited_df = splited_df.reset_index(drop=True)
        splited_df['Plt_Area'] = splited_df['geometry'].area
        if side == "":
            org_df = org_df.set_index(['OLnFID', 'OLnPLT'])
            splited_df = splited_df.set_index(['OLnFID', 'OLnPLT'])
            splited_df = on_FP_Plot_Ecosite_Merge(splited_df, org_df)
        has_code = False
        if 'gridcode' in splited_df.columns:
            splited_df = splited_df.rename(columns={"gridcode": "Site_Type_Code"})
            has_code = True
        elif 'Ecosite' in splited_df.columns:
            splited_df = splited_df.rename(columns={"Ecosite": "Site_Type_Code"})
            has_code

        has_site_type = False
        if 'EcositeType' in splited_df.columns:
            splited_df = splited_df.rename(columns={"EcositeType": "Site_Type"})
            has_site_type = True
        elif 'EcositeTyp' in splited_df.columns:
            splited_df = splited_df.rename(columns={"EcositeTyp": "Site_Type"})
            has_site_type = True
        splitted_plot.append(splited_df)
        out_splited_file = os.path.join(output_dir, working_dir, "FFP\\", side + "FFP_frm_v17_ecosites.gpkg")
        splited_df.to_file(out_splited_file)
    print("Spatial joining FP to ecosite.......Done")
    print("Joining segmented FP to trees.......")
    plot_trees = gpd.sjoin(splitted_plot[0], df_tree, how='left', predicate="intersects").rename(columns=col_rename)
    plot_trees = plot_trees.loc[:, ~plot_trees.columns.duplicated()]
    Lplot_trees = gpd.sjoin(splitted_plot[1], df_tree, how='left', predicate="intersects").rename(columns=col_rename)
    Lplot_trees = Lplot_trees.loc[:, ~Lplot_trees.columns.duplicated()]
    Rplot_trees = gpd.sjoin(splitted_plot[2], df_tree, how='left', predicate="intersects").rename(columns=col_rename)
    Rplot_trees = Rplot_trees.loc[:, ~Rplot_trees.columns.duplicated()]
    plot_trees = plot_trees.convert_dtypes()
    Lplot_trees = Lplot_trees.convert_dtypes()
    Rplot_trees = Rplot_trees.convert_dtypes()
    plot_trees = plot_trees.sort_values(by=['OLnFID', 'OLnPLT'])
    plot_trees = plot_trees.reset_index(drop=True)
    Lplot_trees = Lplot_trees.sort_values(by=['OLnFID', 'OLnPLT', 'OLnSEG'])
    Lplot_trees = Lplot_trees.reset_index(drop=True)
    Rplot_trees = Rplot_trees.sort_values(by=['OLnFID', 'OLnPLT', 'OLnSEG'])
    Rplot_trees = Rplot_trees.reset_index(drop=True)
    # plot_trees.to_parquet(join_file, index=False)
    join_file = os.path.join(output_dir, working_dir, "Regen_Ass\\Plot100m2_with_trees.gpkg")
    Ljoin_file = os.path.join(output_dir, working_dir, "Regen_Ass\\LPlot100m2_with_trees.gpkg")
    Rjoin_file = os.path.join(output_dir, working_dir, "Regen_Ass\\RPlot100m2_with_trees.gpkg")
    plot_trees.to_file(join_file, index=True, driver="GPKG")
    Lplot_trees.to_file(Ljoin_file, index=True, driver="GPKG")
    Rplot_trees.to_file(Rjoin_file, index=True, driver="GPKG")
    print("Joining segmented FP to trees.......Done")
    # Calculate plot statistic by species
    StatBySpeices_file = os.path.join(output_dir, working_dir, "Regen_Ass\\Plot100_SumStat_bySpecies.gpkg")
    LStatBySpeices_file = os.path.join(output_dir, working_dir, "Regen_Ass\\LPlot100_SumStat_bySpecies.gpkg")
    RStatBySpeices_file = os.path.join(output_dir, working_dir, "Regen_Ass\\RPlot100_SumStat_bySpecies.gpkg")
    #  Summarize understory biomass and methane flux by plot.
    plot_trees_all = plot_statis(plot_trees, 'Ass_StatC', 'Ass_StatD')
    Lplot_trees_all = plot_statis(Lplot_trees, 'Ass_StatC', 'Ass_StatD')
    Rplot_trees_all = plot_statis(Rplot_trees, 'Ass_StatC', 'Ass_StatD')
    plot_trees_all = plot_trees_all.set_index(['OLnFID', 'OLnPLT'])
    Lplot_trees_all = Lplot_trees_all.set_index(['OLnFID', 'OLnPLT', 'OLnSEG'])
    Rplot_trees_all = Rplot_trees_all.set_index(['OLnFID', 'OLnPLT', 'OLnSEG'])
    ##Scenario 1A
    plot_Summary_all_1A = calculate_LenA_OnFP(plot_trees_all, plot_trees, 'On FP', 'Ass_StatC')
    Summaries_Statistic_1A = os.path.join(output_dir, working_dir, "Regen_Ass\\Plot100_Summery_Stat_1A.GPKG")
    plot_Summary_all_1A.to_file(Summaries_Statistic_1A, index=True, driver="GPKG")
    ##Scenario 1B
    Lplot_Summary_all = calculate_LenB_OffFP(Lplot_trees_all, Lplot_trees, 'Off FP', 'Ass_StatD')
    Lplot_Summary_all_file = os.path.join(output_dir, working_dir, "Regen_Ass\\Lplot_Summary_all.gpkg")
    Lplot_Summary_all.to_file(Lplot_Summary_all_file, index=True, driver="GPKG")
    #
    Rplot_Summary_all = calculate_LenB_OffFP(Rplot_trees_all, Rplot_trees, 'Off FP', 'Ass_StatD')
    Rplot_Summary_all_file = os.path.join(output_dir, working_dir, "Regen_Ass\\Rplot_Summary_all.gpkg")
    Rplot_Summary_all.to_file(Rplot_Summary_all_file, index=True, driver="GPKG")
    plot_Summary_all_1B = calculate_LenB_OnFP(plot_trees_all, Lplot_Summary_all, Rplot_Summary_all, plot_trees,
                                              'Ass_StatD')
    Summaries_Statistic_1B = os.path.join(output_dir, working_dir, "Regen_Ass\\Plot100_Summery_Stat_1B.GPKG")
    plot_Summary_all_1B.to_file(Summaries_Statistic_1B, index=True, driver="GPKG")


if __name__ == '__main__':
    start_time = time.time()
    print('Plot Biomass computation started at {}'.format(time.strftime("%b %Y %H:%M:%S", time.localtime())))

    current_file = Path(getsourcefile(lambda: 0)).resolve()
    main_dir = current_file.parents[1]
    data_dir = r"I:\BERATools\Workshop"
    output_dir = r"I:\BERATools\Workshop\Output"
    print("Loading data.......")

    # "ScenarioC\\FFP\\
    ##Load sample data
    in_tree = os.path.join(data_dir, working_dir, "Intermediate\\Scenario_Paper_seedlings_classes_above30cm_biomass.gpkg")
    in_FFP = os.path.join(data_dir, working_dir, "Intermediate\\FFP_frm_v17.shp")
    split_file = os.path.join(data_dir, working_dir, "Helpers\\Ecosite_SmoothPolygon_Clipped.shp")

    plot_biomass(in_tree, in_FFP, output_dir)

    print('Elapsed time: {}'.format(time.time() - start_time, 5))


