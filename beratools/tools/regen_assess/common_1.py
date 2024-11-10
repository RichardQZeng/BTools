import geopandas as gpd
import os,sys
import math
from multiprocessing.pool import Pool
from tqdm.auto import tqdm
#import concurrent.futures
from pathlib import Path
from inspect import getsourcefile
import pandas as pd
#from dask.distributed import Client, as_completed
import time

if __name__ == '__main__':

    current_file = Path(getsourcefile(lambda: 0)).resolve()
    main_dir = current_file.parents[1]
    data = os.path.join(main_dir, "Data")
    tool_dir = current_file.parents[0]
    sys.path.insert(0, main_dir.as_posix())
    sys.path.insert(0, tool_dir.as_posix())

from regen_assess.equation import *
from shapely.geometry import Polygon, MultiPolygon
from shapely import length,LineString
import shapely
def read_data2gpd(in_data_path):
    # print("Reading data: {}.......".format(in_data_path))
    out_gpd_obj = gpd.GeoDataFrame.from_file(in_data_path ,engine="pyogrio", use_arrow=True)
    return out_gpd_obj

def chk_df_multipart(df, chk_shp_in_string):
    try:
        found = False
        if str.upper(chk_shp_in_string) in [x.upper() for x in df.geom_type.values]:
            found = True
            df = df.explode()
            if type(df) is gpd.geodataframe.GeoDataFrame:
                if 'OLnFID' in df.columns:
                    df['OLnSEG'] = df.groupby('OLnFID').cumcount()
                    df = df.sort_values(by=['OLnFID', 'OLnSEG'])
                    df = df.reset_index(drop=True)
                elif 'FID' in df.columns:
                    df['SEG'] = df.groupby('FID').cumcount()
                    df = df.sort_values(by=['FID', 'SEG'])
                    df = df.reset_index(drop=True)
                elif 'Id' in df.columns:
                    df['SEG'] = df.groupby('Id').cumcount()
                    df = df.sort_values(by=['Id', 'SEG'])
                    df = df.reset_index(drop=True)

        else:
            found = False
        return df, found
    except Exception as e:
        print(e)
        return df, True
def find_intersections(gdf: gpd.GeoDataFrame):
    # Save geometries to another field

    gdf['geom'] = gdf.geometry

    # Self join
    sj = gpd.sjoin(gdf, gdf,
                   how="inner",
                   predicate="intersects",
                   lsuffix="left",
                   rsuffix="right")

    # Remove geometries that intersect themselves
    sj = sj[sj.index != sj.index_right]

    # Extract the intersecting geometry
    sj['intersection_geom'] = sj['geom_left'].intersection(sj['geom_right'])

    # Reset the geometry (remember to set the CRS correctly!)
    sj.set_geometry('intersection_geom', drop=True, inplace=True, crs=gdf.crs)

    # Drop duplicate geometries
    final_gdf = sj.drop_duplicates(subset=['geometry']).reset_index()

    # Drop intermediate fields
    drops = ['geom_left', 'geom_right', 'index_right', 'index']
    final_gdf = final_gdf.drop(drops, axis=1)

    return final_gdf


def execute_multiprocessing(in_func, in_data, app_name, processes, workers,
                            mode=PARALLEL_MODE, verbose=False):
    out_result = []
    step = 0
    print("Using {} CPU cores".format(processes), flush=True)
    total_steps = len(in_data)

    try:
        if mode == ParallelMode.MULTIPROCESSING:
            print("Multiprocessing started...", flush=True)

            with Pool(processes) as pool:
                # print(multiprocessing.active_children())
                with tqdm(total=total_steps, disable=verbose) as pbar:
                    for result in pool.imap_unordered(in_func, in_data):
                        if result_is_valid(result):
                            out_result.append(result)

                        step += 1
                        if verbose:
                            print_msg(app_name, step, total_steps)
                        else:
                            pbar.update()

            pool.close()
            pool.join()
        # elif mode == ParallelMode.SEQUENTIAL:
        #     with tqdm(total=total_steps, disable=verbose) as pbar:
        #         for line in in_data:
        #             result_item = in_func(line)
        #             if result_is_valid(result_item):
        #                 out_result.append(result_item)
        #
        #             step += 1
        #             if verbose:
        #                 print_msg(app_name, step, total_steps)
        #             else:
        #                 pbar.update()
        # elif mode == ParallelMode.CONCURRENT:
        #     with concurrent.futures.ProcessPoolExecutor(max_workers=processes) as executor:
        #         futures = [executor.submit(in_func, line) for line in in_data]
        #         with tqdm(total=total_steps, disable=verbose) as pbar:
        #             for future in concurrent.futures.as_completed(futures):
        #                 result_item = future.result()
        #                 if result_is_valid(result_item):
        #                     out_result.append(result_item)
        #
        #                 step += 1
        #                 if verbose:
        #                     print_msg(app_name, step, total_steps)
        #                 else:
        #                     pbar.update()
        # elif mode == ParallelMode.DASK:
        #     dask_client = Client(threads_per_worker=1, n_workers=processes)
        #     print(dask_client)
        #     try:
        #         print('start processing')
        #         result = dask_client.map(in_func, in_data)
        #         seq = as_completed(result)
        #
        #         with tqdm(total=total_steps, disable=verbose) as pbar:
        #             for i in seq:
        #                 if result_is_valid(result):
        #                     out_result.append(i.result())
        #
        #                 step += 1
        #                 if verbose:
        #                     print_msg(app_name, step, total_steps)
        #                 else:
        #                     pbar.update()
        #     except Exception as e:
        #         dask_client.close()
        #
        #     dask_client.close()

        # ! important !
        # comment temporarily, man enable later if need to use ray
        # elif mode == ParallelMode.RAY:
        #     ray.init(log_to_driver=False)
        #     process_single_line_ray = ray.remote(in_func)
        #     result_ids = [process_single_line_ray.remote(item) for item in in_data]
        #
        #     while len(result_ids):
        #         done_id, result_ids = ray.wait(result_ids)
        #         result_item = ray.get(done_id[0])
        #
        #         if result_is_valid(result_item):
        #             out_result.append(result_item)
        #
        #         step += 1
        #         print_msg(app_name, step, total_steps)

        #     ray.shutdown()
    except OperationCancelledException:
        print("Operation cancelled")
        return None

    return out_result

def result_is_valid(result):
    if type(result) is list or type(result) is tuple:
        if len(result) > 0:
            return True
    elif type(result) is pd.DataFrame or type(result) is gpd.GeoDataFrame:
        if not result.empty:
            return True
    elif result:
        return True

    return False
def print_msg(app_name, step, total_steps):
    print(f' "PROGRESS_LABEL {app_name} {step} of {total_steps}" ', flush=True)
    print(f' %{step / total_steps * 100} ', flush=True)


def calculate_angle(point1, point2):
    """Calculate the angle between two points in degrees."""
    angle=math.degrees(math.atan2(point2[1] - point1[1], point2[0] - point1[0]))
    # if angle<0:
    #     angle=180+angle
    return angle


def get_orientation(polygon):


    # min_rect = polygon.oriented_envelope#()
    coords = polygon.exterior.coords#get_coordinates(ignore_index=True)
    # Take the first two points to determine the regenass_main axis
    # angle = calculate_angle([max(coords.x),max(coords.y)], [min(coords.x),min(coords.y)])
    x_coord=(coords.xy[0])
    y_coord=(coords.xy[1])
    a=[x_coord[0],y_coord[0]]
    b=[x_coord[1],y_coord[1]]
    c=[x_coord[2],y_coord[2]]
    a_b=shapely.length(LineString((a, b)))
    b_c = shapely.length(
        LineString((b, c)))
    if a_b>=b_c:
        angle = calculate_angle(a, b)
        point1=a
        point2=b
        width=b_c
        long_side=a_b
    else:
        angle = calculate_angle(b, c)
        point1 = b
        point2 = c
        width = a_b
        long_side = b_c
    return angle,point1,point2,width,long_side

def split_polygon_into_area(in_df_polygon,target_area,offset_dist):
    orginal_columns=in_df_polygon.columns
    df_crs = in_df_polygon.crs
    def segment_polygon(row):
        # avg_width=row['MAX_WIDTH']*1.2
        polygon=row['geometry']

        try:
            ## Segment the polygon into approximately equal-area parts, perpendicular to the regenass_main direction of the polygon. """

            ##Get the regenass_main orientation of a polygon based on its minimum rotated rectangle
            ## Return the longerest side's orientation angle, start and end points of the longerest side
            ## the width of the rotated rectangle
            angle,point1,point2,width,long_side = get_orientation(shapely.minimum_rotated_rectangle(polygon))
            # Get the ratio of bounds box to FP
            area_ratio=shapely.minimum_rotated_rectangle(polygon).area/polygon.area
            # Compute the length required for the target area based on the bounds box width and area ratio
            avg_length=(area_ratio*target_area)/width
            # Compute how many segments along the whole FP
            n_segments = int(np.ceil((long_side / avg_length)))
            segments = []

            for i in range(n_segments):
                    ##Generate segment base on orientation angle and start point1 x,y
                    if long_side-((i+1) *avg_length)>=avg_length/4:
                        avg_length=avg_length
                        startx1 = point1[0] + i * math.cos(math.radians(angle)) * avg_length
                        starty1 = point1[1] + i * math.sin(math.radians(angle)) * avg_length
                        endx1 = startx1 + math.cos(math.radians(angle)) * avg_length
                        endy1 = starty1 + math.sin(math.radians(angle)) * avg_length
                    else:

                        startx1 = point1[0] + i * math.cos(math.radians(angle)) * avg_length
                        starty1 = point1[1] + i * math.sin(math.radians(angle)) * avg_length
                        endx1 = startx1 + math.cos(math.radians(angle)) * ((long_side-((i+1) *avg_length))+avg_length)
                        endy1 = starty1 + math.sin(math.radians(angle)) * ((long_side-((i+1) *avg_length))+avg_length)

                    ##Construct line geometry for the segment
                    edge = LineString([(startx1, starty1), (endx1, endy1)])
                    ##Construct parallel line geometry for the segment
                    parallel_edge = edge.parallel_offset(width)
                    ##Get the start and end points of parallel line geometry
                    coords = parallel_edge.coords
                    startx2 = coords[0][0]
                    starty2 = coords[0][-1]
                    endx2 = coords[-1][0]
                    endy2 = coords[-1][-1]
                    ##Construct a polygon for the Nth segment
                    segment = Polygon([(startx1, starty1), (endx1, endy1), (endx2, endy2), (startx2, starty2)])
                    ##Intersect the constructed polygon to original FP to create a approxiate 100m2 plot
                    plot = polygon.intersection(segment)
                    # if polygon.intersection(segment).geom_type == 'MultiPolygon':
                    if isinstance(plot, (MultiPolygon)):
                        # for a in list(plot.geoms):
                        # segments.append(a)
                        segments.extend(plot.geoms)
                    else:
                        segments.append(plot)

            MPolygons = shapely.MultiPolygon(segments)
            RPolygons=shapely.transform(MPolygons,lambda x:x+([offset_dist*math.cos(math.radians(angle+90)),offset_dist*math.sin(math.radians(angle+90))]))
            LPolygons = shapely.transform(MPolygons, lambda x:x+([offset_dist*math.cos(math.radians(angle-90)),offset_dist*math.sin(math.radians(angle-90))]))

            return pd.Series([MPolygons,RPolygons,LPolygons], index=['geometry', 'Rgeometry','Lgeometry'])

        except Exception as e:
            print(e)
            exit()

    print("Segmenting Polygon......")
    # if "GEOMETRY" in in_df_polygon.columns:
    #     in_df_polygon = in_df_polygon.rename(columns={'GEOMETRY':'geometry'})
    # else:
    #     in_df_polygon.set_geometry("geometry")

    in_df_polygon[['geometry','Rgeometry','Lgeometry']] = in_df_polygon.apply(segment_polygon, axis=1)
    left_df_polygon=gpd.GeoDataFrame.copy(in_df_polygon)
    right_df_polygon = gpd.GeoDataFrame.copy(in_df_polygon)

    in_df_polygon=in_df_polygon.set_geometry("geometry")
    in_df_polygon=in_df_polygon.drop(columns=['Rgeometry','Lgeometry'])
    in_df_polygon=in_df_polygon.explode()
    # in_df_polygon.columns=orginal_columns
    in_df_polygon['Plt_Area'] = in_df_polygon['geometry'].area
    in_df_polygon['OLnPLT'] = in_df_polygon.groupby('OLnFID').cumcount()
    in_df_polygon = in_df_polygon.reset_index(drop=True)
    in_df_polygon.crs=df_crs

    left_df_polygon=left_df_polygon.set_geometry("Lgeometry")
    left_df_polygon=left_df_polygon.drop(columns=['geometry','Rgeometry'])
    left_df_polygon = left_df_polygon.rename(columns={'Lgeometry':'geometry'})
    left_df_polygon = left_df_polygon.set_geometry("geometry")
    left_df_polygon=left_df_polygon.explode()
    # left_df_polygon.columns=orginal_columns
    left_df_polygon['Plt_Area'] = left_df_polygon['geometry'].area
    left_df_polygon['OLnPLT'] = left_df_polygon.groupby('OLnFID').cumcount()
    left_df_polygon = left_df_polygon.reset_index(drop=True)
    left_df_polygon.crs=df_crs

    right_df_polygon=right_df_polygon.set_geometry("Rgeometry")
    right_df_polygon=right_df_polygon.drop(columns=['geometry','Lgeometry'])
    right_df_polygon = right_df_polygon.rename(columns={'Rgeometry':'geometry'})
    right_df_polygon = right_df_polygon.set_geometry("geometry")
    right_df_polygon=right_df_polygon.explode()
    # right_df_polygon.columns=orginal_columns
    right_df_polygon['Plt_Area'] = right_df_polygon['geometry'].area
    right_df_polygon['OLnPLT'] = right_df_polygon.groupby('OLnFID').cumcount()
    right_df_polygon = right_df_polygon.reset_index(drop=True)
    right_df_polygon.crs=df_crs




    return in_df_polygon, right_df_polygon,left_df_polygon

def on_FP_Plot_Ecosite_Merge(in_df,org_df):
    org_df['EcositeTyp']=Ecosite_Type.WETLAND_TREED.value
    org_df['gridcode'] = 0

    for index, row in org_df.iterrows():
        selected_rows=in_df[in_df.index==index]
        if len(selected_rows)>1:
            major_ecosite=selected_rows[
                selected_rows['Plt_Area']==selected_rows['Plt_Area'].max()].EcositeTyp.item()
            ecositecode = selected_rows[
                selected_rows['Plt_Area'] == selected_rows['Plt_Area'].max()].gridcode.item()
        else:
            major_ecosite = selected_rows['EcositeTyp'].item()
            ecositecode= selected_rows['gridcode'].item()

        org_df._set_value(index,'EcositeTyp',major_ecosite)
        org_df._set_value(index, 'gridcode', ecositecode)
    org_df['OLnFID'] = (org_df.index.get_level_values(0)).astype(int)
    org_df['OLnPLT'] = (org_df.index.get_level_values(1)).astype(int)
    org_df = org_df.reset_index(drop=True)
    org_df = org_df.sort_values(by=['OLnFID', 'OLnPLT'])

    return org_df


def col_rename(x):
    if x.find('_right')!=-1:
        x=x.replace('_right','')
    elif x.find('_left')!=-1:
          x=  x.replace('_left', '')
    return x


def popluate_flux_underBio(in_df,target_area,scenario,UBioMass_Col,Methane_Col):
    in_df[UBioMass_Col]=np.nan
    in_df[Methane_Col] = np.nan
    print("Calculate Plot's Understory biomass and Methane flux.....")
    def find_flux_underBio(row):
        ecosite_list=','.join(row['Site_Type']).split(",")
        status=row[scenario+'_first']
        plot_area=row['geometry'].area
        ratio=plot_area/target_area
        # ratio = 1
        under_bio = 0
        flux = 0
        total_under_bio=0
        total_flux=0
        for ecosite in ecosite_list:
            if ecosite==Ecosite_Type.MESIC_UPLAND.value:
                if status==Line_Status.AdvReg.value:
                    under_bio=Under_BioTMass_PlotLevel.MESIC_UPLAND_AdvReg.value*ratio
                    flux=Methane_Flux_PlotLevel.MESIC_UPLAND_AdvReg.value*ratio
                elif status==Line_Status.Reg.value:
                    under_bio = Under_BioTMass_PlotLevel.MESIC_UPLAND_Reg.value*ratio
                    flux = Methane_Flux_PlotLevel.MESIC_UPLAND_Reg.value*ratio
                elif status==Line_Status.Arr.value:
                    under_bio = Under_BioTMass_PlotLevel.MESIC_UPLAND_Arr.value*ratio
                    flux = Methane_Flux_PlotLevel.MESIC_UPLAND_Arr.value*ratio
            elif ecosite == Ecosite_Type.DRY_UPLAND.value:
                if status == Line_Status.AdvReg.value:
                    under_bio = Under_BioTMass_PlotLevel.DRY_UPLAND_AdvReg.value*ratio
                    flux = Methane_Flux_PlotLevel.DRY_UPLAND_AdvReg.value*ratio
                elif status == Line_Status.Reg.value:
                    under_bio = Under_BioTMass_PlotLevel.DRY_UPLAND_Reg.value*ratio
                    flux = Methane_Flux_PlotLevel.DRY_UPLAND_Reg.value*ratio
                elif status == Line_Status.Arr.value:
                    under_bio = Under_BioTMass_PlotLevel.DRY_UPLAND_Arr.value*ratio
                    flux = Methane_Flux_PlotLevel.DRY_UPLAND_Arr.value*ratio
            elif ecosite == Ecosite_Type.TRAN_TREED.value:
                if status == Line_Status.AdvReg.value:
                    under_bio = Under_BioTMass_PlotLevel.TRAN_TREED_AdvReg.value*ratio
                    flux = Methane_Flux_PlotLevel.TRAN_TREED_AdvReg.value*ratio
                elif status == Line_Status.Reg.value:
                    under_bio = Under_BioTMass_PlotLevel.TRAN_TREED_Reg.value*ratio
                    flux = Methane_Flux_PlotLevel.TRAN_TREED_Reg.value*ratio
                elif status == Line_Status.Arr.value:
                    under_bio = Under_BioTMass_PlotLevel.TRAN_TREED_Arr.value*ratio
                    flux = Methane_Flux_PlotLevel.TRAN_TREED_Arr.value*ratio
            elif ecosite == Ecosite_Type.WETLAND_TREED.value:
                if status == Line_Status.AdvReg.value:
                    under_bio = Under_BioTMass_PlotLevel.WETLAND_TREED_AdvReg.value*ratio
                    flux = Methane_Flux_PlotLevel.WETLAND_TREED_AdvReg.value*ratio
                elif status == Line_Status.Reg.value:
                    under_bio = Under_BioTMass_PlotLevel.WETLAND_TREED_Reg.value*ratio
                    flux = Methane_Flux_PlotLevel.WETLAND_TREED_Reg.value*ratio
                elif status == Line_Status.Arr.value:
                    under_bio = Under_BioTMass_PlotLevel.WETLAND_TREED_Arr.value*ratio
                    flux = Methane_Flux_PlotLevel.WETLAND_TREED_Arr.value*ratio
            elif ecosite == Ecosite_Type.WETLAND_LOWDEN.value:
                if status == Line_Status.AdvReg.value:
                    under_bio = Under_BioTMass_PlotLevel.WETLAND_LOWDEN_AdvReg.value*ratio
                    flux = Methane_Flux_PlotLevel.WETLAND_LOWDEN_AdvReg.value*ratio
                elif status == Line_Status.Reg.value:
                    under_bio = Under_BioTMass_PlotLevel.WETLAND_LOWDEN_Reg.value*ratio
                    flux = Methane_Flux_PlotLevel.WETLAND_LOWDEN_Reg.value*ratio
                elif status == Line_Status.Arr.value:
                    under_bio = Under_BioTMass_PlotLevel.WETLAND_LOWDEN_Arr.value*ratio
                    flux = Methane_Flux_PlotLevel.WETLAND_LOWDEN_Arr.value*ratio
            total_under_bio=total_under_bio+under_bio
            total_flux=total_flux+flux

        return pd.Series([total_under_bio, total_flux], index=['Under_Bio_40', 'Methane_40'])

    print("Calculate Plot's Understory biomass and Methane flux.....Done")

    in_df[[UBioMass_Col,Methane_Col]] = in_df.apply(find_flux_underBio, axis=1)

    return in_df

def read_data(in_tree, in_FP, out_tree):


    df_tree = read_data2gpd(in_tree)
    df_FP = read_data2gpd(in_FP)
    df_CL, _ = chk_df_multipart(in_FP, 'MultiLineString')
    return df_tree,df_FP,out_tree

def calculate_LenA_OnFP(in_df, df_tree, where, scenario):
    in_df['Treated'] = "no"
    in_df['Treatment_Type'] ="None"
    in_df['Est_ht_threshold']=0.65
    in_df['Pop1_Pop2_Pop3_T0'] = 0
    in_df['Pop1_all_T0_count']=0
    in_df['Pop1_est_T0_count'] = 0
    in_df['Pop1_est_T0_density (stems/ha)'] = 0.0
    in_df['Ref_density']=800
    in_df['Ref_dominance'] = ""
    in_df['Ref_dominance'] = in_df['Ref_dominance'].astype('object')
    in_df['Ref_Ht'] = 2.0
    in_df['Pop1_est_T0_dominant'] = "None"
    in_df['Pop1_est_T0_dominant_Count'] = 0
    in_df['Pop1_est_T0_RHt']=0.0
    in_df['RS_CA_Time0']="No"
    in_df['RS_CB_Time0'] = "No"
    in_df['RS_CC_Time0'] = "No"
    in_df['RS_Status_Time0'] = ""
    in_df['Pop2_T0_count'] = 0
    in_df['Pop2_T0_density'] = 0.0
    in_df['Pop3_T0_count'] = 0
    in_df['BioMass_Under_T0']=0.0
    in_df['Methane_T0'] = 0.0
    in_df['Soil_Carbon_T0'] = 0.0
    in_df['Pop1_est_T40_dominant'] = "None"
    in_df['Pop1_est_T40_count'] = 0
    in_df['Pop1_est_T40_density (stems/ha)'] = 0.0
    in_df['Ass_Status_Time40'] = ""
    in_df['RS_CA_Time40']="No"
    in_df['RS_CB_Time40'] = "No"
    in_df['RS_CC_Time40'] = "No"
    in_df['RS_Status_Time40']=""
    in_df['BioMass_Under_T40'] = 0.0
    in_df['Methane_T40'] = 0.0
    in_df['Soil_Carbon_T40'] = 0.0
    if not 'BioMass_Time0_sum' in in_df.columns:
        in_df['BioMass_Time0_sum'] = 0.0
    if 'distribute_count' in in_df.columns:
        in_df=in_df.drop(columns=['distribute_count'])
    # in_df['P90_Age_Time0']=np.nan
    # in_df['P90_Age_Time40']=np.nan

    print("Scenario Using LenA:\n Calculating {} Plot's acceptable trees and shrubs density @ current year and 40 years later,"
          .format(where))
    print("calculating the dominant acceptable tree @ current year and 40 years later".format(
        where))
    print("and calculate understory biomass and methane flux by species.......")

    for OLnFID,OLnPLT in in_df.index:
        dominant_count_T0 = 0
        dominant_count_T40 = 0
        dominant_Species_T0 = "None"
        dominant_Species_T40 = "None"
        Pop1_est_T0_RHt = 0.0
        Pop1_est_T40_RHt = 0.0
        P90_appected_trees_at_Year0 = np.nan
        P90_appected_trees_at_YearN = np.nan
        other_T0 = 0
        stem_T40 = 0
        Pop1_all_T0_count = 0
        site_type = (in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Site_Type']).item()
        line_status = (in_df.loc[in_df.index == (OLnFID, OLnPLT), scenario]).item()
        plot_polygon = in_df.loc[in_df.index == (OLnFID, OLnPLT), 'geometry']
        Est_thr = Est_ht_thresholdA(site_type)
        ref_den = ref_density(site_type)
        ref_dom_list = ref_dominance(site_type)
        ref_ht = ref_ht_func(site_type)
        est_stems_count=0
        shrubs_count=0
        total_species=0
        BioMass_Time0_sum=0.0
        BioMass_Time40_sum=0.0
        est_stems_T40=0
        Pop1_all_T0_height_min=0.0
        Pop1_all_T0_height_max=0.0
        Pop1_all_T40_height_min = 0.0
        Pop1_all_T40_height_max = 0.0

        try:
            if not pd.isna((in_df.loc[in_df.index==(OLnFID,OLnPLT),'Tree_Species']).iat[0][0]):

                    selected_row = df_tree[
                        ((df_tree['OLnFID'] == OLnFID) & (df_tree['OLnPLT'] == OLnPLT))].copy()
                    total_species=len(selected_row)
                    appected_shrubs = selected_row[selected_row['Tree_Species'].apply(lambda x: x in shrubs_list)]
                    shrubs_count = len(appected_shrubs)
                    BioMass_Time0_sum=BioMass_Time0_sum+appected_shrubs['BioMass_Time0'].sum(skipna=True)
                    appected_trees_at_Year0=selected_row[selected_row['Tree_Species'].apply(lambda x: x in trees_list)]
                    Pop1_all_T0_count=len(appected_trees_at_Year0)
                    if Pop1_all_T0_count>0:
                        Pop1_all_T0_height_min = appected_trees_at_Year0['Height_T0'].min().item()
                        Pop1_all_T0_height_max = appected_trees_at_Year0['Height_T0'].max().item()

                    BioMass_Time0_sum = BioMass_Time0_sum + appected_trees_at_Year0['BioMass_Time0'].sum(
                                                                                                     skipna=True)
                    appected_trees_at_Year0 = appected_trees_at_Year0[appected_trees_at_Year0['Height_T0'] >=Est_thr ]
                    est_stems_count=len(appected_trees_at_Year0)

                    if est_stems_count>0:

                        # P90_appected_trees_at_Year0 = appected_trees_at_Year0['Height_T0'].describe(percentiles=[0.9]).iloc[-2].item()
                        dominant_count_T0=appected_trees_at_Year0['Tree_Species'].value_counts().iat[0].item()
                        dominant_Species_T0= appected_trees_at_Year0['Tree_Species'].value_counts().index[0]
                        Pop1_est_T0_RHt = len(appected_trees_at_Year0[appected_trees_at_Year0['Height_T0'] >= ref_ht])/est_stems_count

                    selected_row = df_tree[
                        ((df_tree['OLnFID'] == OLnFID) & (df_tree['OLnPLT'] == OLnPLT))].copy()
                    appected_trees_at_YearN = selected_row[selected_row['Tree_Species'].apply(lambda x: x in trees_list)]

                    appected_trees_at_YearN = appected_trees_at_YearN[appected_trees_at_YearN['Height_Time40'] >=Est_thr]
                    est_stems_T40=len(appected_trees_at_YearN)

                    if est_stems_T40>0:
                        Pop1_all_T40_height_min = appected_trees_at_YearN['Height_Time40'].min().item()
                        Pop1_all_T40_height_max = appected_trees_at_YearN['Height_Time40'].max().item()
                        # P90_appected_trees_at_YearN = appected_trees_at_YearN['Height_Time40'].describe(percentiles=[0.9]).iloc[
                        #     -2].item()
                        # dominant_count_T40 = appected_trees_at_YearN['Tree_Species'].value_counts().iat[0].item()
                        dominant_Species_T40 = appected_trees_at_YearN['Tree_Species'].value_counts().index[0]
                        # stem_T40=len(
                        #     appected_trees_at_YearN[appected_trees_at_YearN['Height_Time40'] >= ref_ht])
                        Pop1_est_T40_RHt = len(
                            appected_trees_at_YearN[appected_trees_at_YearN['Height_Time40'] >= ref_ht]) / est_stems_T40
                    BioMass_Time40_sum = BioMass_Time40_sum + appected_trees_at_YearN['BioMass_Time40'].sum(skipna=True)
                    other_T0=total_species-est_stems_count-shrubs_count

            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Est_ht_threshold'] = Est_thr
            in_df.loc[in_df.index == (OLnFID, OLnPLT),'Pop1_Pop2_Pop3_T0'] =total_species
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_all_T0_count'] = Pop1_all_T0_count
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_est_T0_count'] = est_stems_count
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_all_T0_height_min']=Pop1_all_T0_height_min
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_all_T0_height_max']=Pop1_all_T0_height_max
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_all_T40_height_min']=Pop1_all_T40_height_min
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_all_T40_height_max']=Pop1_all_T40_height_max

            in_df.loc[in_df.index==(OLnFID,OLnPLT),'Pop1_est_T0_density (stems/ha)']=est_stems_count*100
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Ref_density'] = ref_den
            in_df.at[(OLnFID, OLnPLT), 'Ref_dominance'] = ref_dom_list
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Ref_Ht'] = ref_ht
            in_df.loc[in_df.index == (OLnFID,OLnPLT), 'Pop1_est_T0_dominant'] = dominant_Species_T0
            in_df.loc[in_df.index == (OLnFID,OLnPLT), 'Pop1_est_T0_dominant_Count'] = dominant_count_T0
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_est_T0_RHt']=Pop1_est_T0_RHt*100
            if est_stems_count*100>=ref_den:
                RS_CA_Time0 = "Yes"
            else:
                RS_CA_Time0 = "No"

            if Pop1_est_T0_RHt*100>=50.0:
                RS_CB_Time0= "Yes"
            else:
                RS_CB_Time0= "No"
            if dominant_Species_T0 in ref_dom_list:
                RS_CC_Time0 = "Yes"
            else:
                RS_CC_Time0 = "No"
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'RS_CA_Time0']=RS_CA_Time0
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'RS_CB_Time0']=RS_CB_Time0
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'RS_CC_Time0']=RS_CC_Time0
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'RS_Status_Time0'] = assign_restortation_status(RS_CA_Time0,
                                                                                                       RS_CB_Time0,
                                                                                                       RS_CC_Time0)

            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop2_T0_count'] = shrubs_count
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop2_T0_density'] = shrubs_count * 100
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop3_T0_count'] = other_T0
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'BioMass_Time0_sum']=BioMass_Time0_sum
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'BioMass_Time40_sum'] = BioMass_Time40_sum
            ubio,flux=find_flux_underBio(site_type,line_status,100,plot_polygon)
            in_df.loc[in_df.index == (OLnFID, OLnPLT),'BioMass_Under_T0'] = ubio
            in_df.loc[in_df.index == (OLnFID, OLnPLT),'Methane_T0'] = flux

            # in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Soil_Carbon_T0'] = 0

            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_est_T40_count'] = est_stems_T40
            in_df.loc[in_df.index == (OLnFID,OLnPLT), 'Pop1_est_T40_density (stems/ha)'] =est_stems_T40*100
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_est_T40_dominant'] = dominant_Species_T40
            in_df.loc[in_df.index == (OLnFID,OLnPLT), 'Pop1_est_T40_RHt']=Pop1_est_T40_RHt*100

            status_T40= Ass_Status_LenA_B_Time40((Pop1_est_T40_RHt * 100), site_type)
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Ass_Status_Time40'] =status_T40


            if dominant_count_T40*100 >= ref_den:
                RS_CA_Time40 = "Yes"
            else:
                RS_CA_Time40 = "No"

            if Pop1_est_T40_RHt * 100 >= 50.0:
                RS_CB_Time40 = "Yes"
            else:
                RS_CB_Time40 = "No"
            if dominant_Species_T40 in ref_dom_list:
                RS_CC_Time40 = "Yes"
            else:
                RS_CC_Time40 = "No"

            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'RS_CA_Time40'] = RS_CA_Time40
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'RS_CB_Time40'] = RS_CB_Time40
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'RS_CC_Time40'] = RS_CC_Time40
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'RS_Status_Time40'] = assign_restortation_status(RS_CA_Time40,
                                                                                                       RS_CB_Time40,
                                                                                                       RS_CC_Time40)

            ubio40,flux40=find_flux_underBio(site_type,status_T40,100,plot_polygon)
            in_df.loc[in_df.index == (OLnFID, OLnPLT),'BioMass_Under_T40'] = ubio40
            in_df.loc[in_df.index == (OLnFID, OLnPLT),'Methane_T40'] = flux40
            # in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Soil_Carbon_T40'] = 0

            # in_df.loc[in_df.index == (OLnFID,OLnPLT),'P90_Age_Time0'] = P90_appected_trees_at_Year0
            # in_df.loc[in_df.index == (OLnFID,OLnPLT),'P90_Age_Time40'] = P90_appected_trees_at_YearN
        except Exception as e:
            print(e)
    in_df=in_df.rename(columns={scenario:'Ass_Status_Time0','Tree_Species':'Pop1_all_T0_species'})
    in_df= in_df.loc[:, S1a_OnFP_columnsTitles]
    print("Calculating {} Plot's trees and shrubs density @ current year and 40 years later......Done".format(where))
    return in_df
def calculate_LenB_OffFP(in_df, df_tree, where, scenario):
    in_df['Treated'] = "no"
    in_df['Treatment_Type'] = "None"
    in_df['Est_ht_threshold'] = 0.65
    in_df['Pop1_Pop2_Pop3_T0'] = 0
    in_df['Pop1_all_T0_count'] = 0
    in_df['Pop1_est_T0_count'] = 0
    in_df['Pop1_est_T0_density (stems/ha)'] = 0.0
    in_df['Ref_density'] = 800
    in_df['Ref_dominance'] = ""
    in_df['Ref_dominance'] = in_df['Ref_dominance'].astype('object')
    in_df['Ref_Ht'] = 2.0
    in_df['Pop1_est_T0_dominant'] = "None"
    in_df['Pop1_est_T0_dominant_Count'] = 0
    in_df['Pop1_est_T0_RHt'] = 0.0
    in_df['RS_CA_Time0'] = "No"
    in_df['RS_CB_Time0'] = "No"
    in_df['RS_CC_Time0'] = "No"
    in_df['RS_Status_Time0'] = ""
    in_df['Pop2_T0_count'] = 0
    in_df['Pop2_T0_density'] = 0.0
    in_df['Pop3_T0_count'] = 0
    in_df['BioMass_Under_T0'] = 0.0
    in_df['Methane_T0'] = 0.0
    in_df['Soil_Carbon_T0'] = 0.0
    in_df['Pop1_est_T40_dominant'] = "None"
    in_df['Pop1_est_T40_count'] = 0
    in_df['Pop1_est_T40_density (stems/ha)'] = 0.0
    in_df['Ass_Status_Time40'] = ""
    in_df['RS_CA_Time40'] = "No"
    in_df['RS_CB_Time40'] = "No"
    in_df['RS_CC_Time40'] = "No"
    in_df['RS_Status_Time40'] = ""
    in_df['BioMass_Under_T40'] = 0.0
    in_df['Methane_T40'] = 0.0
    in_df['Soil_Carbon_T40'] = 0.0
    if not 'BioMass_Time0_sum' in in_df.columns:
        in_df['BioMass_Time0_sum'] = 0.0
    if 'distribute_count' in in_df.columns:
        in_df = in_df.drop(columns=['distribute_count'])
    in_df['P90_Age_Time0']=0.0
    in_df['P90_Age_Time40']=0.0

    print("Scenario Using LenB:\n"
          "Calculating {} Plot's acceptable trees and shrubs density and finding the dominant acceptable tree .....".format(
            where))
    for OLnFID, OLnPLT,OLnSEG in in_df.index:
        dominant_count_T0 = 0
        dominant_count_T40 = 0
        dominant_Species_T0 = "None"
        dominant_Species_T40 = "None"
        Pop1_est_T0_RHt = 0.0
        Pop1_est_T40_RHt = 0.0
        P90_appected_trees_at_Year0 = 0.0
        P90_appected_trees_at_YearN = 0.0
        other_T0 = 0
        stem_T40 = 0
        Pop1_all_T0_count = 0
        site_type = (in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Site_Type']).item()
        line_status = (in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), scenario]).item()
        plot_polygon = in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'geometry']
        Est_thr = Est_ht_thresholdB(site_type)
        ref_den = ref_density(site_type)
        ref_dom_list = ref_dominance(site_type)
        ref_ht = ref_ht_func(site_type)
        est_stems_count = 0
        shrubs_count = 0
        total_species = 0
        BioMass_Time0_sum = 0.0
        BioMass_Time40_sum = 0.0
        est_stems_T40 = 0
        Pop1_all_T0_height_min = 0.0
        Pop1_all_T0_height_max = 0.0
        Pop1_all_T40_height_min = 0.0
        Pop1_all_T40_height_max = 0.0

        try:
            if not pd.isna((in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Tree_Species']).iat[0][0]):

                selected_row = df_tree[
                    (((df_tree['OLnFID'] == OLnFID) & (df_tree['OLnPLT'] == OLnPLT))&(df_tree['OLnSEG'] == OLnSEG))].copy()
                total_species = len(selected_row)
                appected_shrubs = selected_row[selected_row['Tree_Species'].apply(lambda x: x in shrubs_list)]
                shrubs_count = len(appected_shrubs)
                BioMass_Time0_sum = BioMass_Time0_sum + appected_shrubs['BioMass_Time0'].sum(skipna=True)
                appected_trees_at_Year0 = selected_row[selected_row['Tree_Species'].apply(lambda x: x in trees_list)]
                Pop1_all_T0_count = len(appected_trees_at_Year0)
                if Pop1_all_T0_count > 0:
                    Pop1_all_T0_height_min = appected_trees_at_Year0['Height_T0'].min().item()
                    Pop1_all_T0_height_max = appected_trees_at_Year0['Height_T0'].max().item()

                BioMass_Time0_sum = BioMass_Time0_sum + appected_trees_at_Year0['BioMass_Time0'].sum(
                    skipna=True)
                est_trees_at_Year0 = appected_trees_at_Year0[appected_trees_at_Year0['Height_T0'] >= Est_thr]
                est_stems_count = len(est_trees_at_Year0)

                if est_stems_count > 0:
                    # P90_appected_trees_at_Year0 = est_trees_at_Year0['Height_T0'].describe(percentiles=[0.9]).iloc[-2].item()
                    dominant_count_T0 = est_trees_at_Year0['Tree_Species'].value_counts().iat[0].item()
                    dominant_Species_T0 = est_trees_at_Year0['Tree_Species'].value_counts().index[0]
                    # est_domin_trees_T0=est_trees_at_Year0[est_trees_at_Year0['Tree_Species']==dominant_Species_T0]
                    # if len(est_domin_trees_T0)>1:
                    #     P90_appected_trees_at_Year0 = est_domin_trees_T0['Height_T0'].describe(percentiles=[0.9]).iloc[
                    #         -2].item()
                    # else:
                    #     P90_appected_trees_at_Year0 = est_domin_trees_T0['Height_T0'].item()
                    # est_domin_trees_T0=est_trees_at_Year0[est_trees_at_Year0['Tree_Species']==dominant_Species_T0]
                    P90_appected_trees_at_Year0 = est_trees_at_Year0['Height_T0'].describe(percentiles=[0.9]).iloc[
                            -2].item()

                    Pop1_est_T0_RHt = len(
                        est_trees_at_Year0[est_trees_at_Year0['Height_T0'] >= ref_ht]) / est_stems_count

                selected_row = df_tree[
                    (((df_tree['OLnFID'] == OLnFID) & (df_tree['OLnPLT'] == OLnPLT))&(df_tree['OLnSEG'] == OLnSEG))].copy()
                appected_trees_at_YearN = selected_row[selected_row['Tree_Species'].apply(lambda x: x in trees_list)]

                est_trees_at_YearN = appected_trees_at_YearN[appected_trees_at_YearN['Height_Time40'] >= Est_thr]
                est_stems_T40 = len(est_trees_at_YearN)
                if est_stems_T40 > 0:
                    Pop1_all_T40_height_min = est_trees_at_YearN['Height_Time40'].min().item()
                    Pop1_all_T40_height_max = est_trees_at_YearN['Height_Time40'].max().item()
                BioMass_Time40_sum = BioMass_Time40_sum + appected_trees_at_YearN['BioMass_Time40'].sum(skipna=True)

                if est_stems_T40 > 0:
                    # P90_appected_trees_at_YearN = appected_trees_at_YearN['Height_Time40'].describe(percentiles=[0.9]).iloc[
                    #     -2].item()
                    # dominant_count_T40 = est_trees_at_YearN['Tree_Species'].value_counts().iat[0].item()
                    dominant_Species_T40 = est_trees_at_YearN['Tree_Species'].value_counts().index[0]
                    # est_domin_trees_T40=est_trees_at_YearN[est_trees_at_YearN['Tree_Species']==dominant_Species_T40]
                    # if len(est_domin_trees_T40)>1:
                    #     P90_appected_trees_at_YearN = est_domin_trees_T40['Height_Time40'].describe(percentiles=[0.9]).iloc[
                    #         -2].item()
                    # else:
                    #     P90_appected_trees_at_YearN = est_domin_trees_T40['Height_Time40'].item()
                    # stem_T40=len(
                    #     appected_trees_at_YearN[appected_trees_at_YearN['Height_Time40'] >= ref_ht])

                    P90_appected_trees_at_YearN = est_trees_at_YearN['Height_Time40'].describe(percentiles=[0.9]).iloc[
                                -2].item()
                    Pop1_est_T40_RHt = len(
                        appected_trees_at_YearN[appected_trees_at_YearN['Height_Time40'] >= ref_ht]) / est_stems_T40

                other_T0 = total_species - est_stems_count - shrubs_count

            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Est_ht_threshold'] = Est_thr
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Pop1_Pop2_Pop3_T0'] = total_species
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Pop1_all_T0_count'] = Pop1_all_T0_count
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Pop1_est_T0_count'] = est_stems_count
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Pop1_all_T0_height_min'] = Pop1_all_T0_height_min
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Pop1_all_T0_height_max'] = Pop1_all_T0_height_max
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Pop1_all_T40_height_min'] = Pop1_all_T40_height_min
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Pop1_all_T40_height_max'] = Pop1_all_T40_height_max

            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Pop1_est_T0_density (stems/ha)'] = est_stems_count * 100
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Ref_density'] = ref_den
            in_df.at[(OLnFID, OLnPLT,OLnSEG), 'Ref_dominance'] = ref_dom_list
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Ref_Ht'] = ref_ht
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Pop1_est_T0_dominant'] = dominant_Species_T0
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Pop1_est_T0_dominant_Count'] = dominant_count_T0
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Pop1_est_T0_RHt'] = Pop1_est_T0_RHt * 100
            if est_stems_count * 100 >= ref_den:
                RS_CA_Time0 = "Yes"
            else:
                RS_CA_Time0 = "No"

            if Pop1_est_T0_RHt * 100 >= 50.0:
                RS_CB_Time0 = "Yes"
            else:
                RS_CB_Time0 = "No"
            if dominant_Species_T0 in ref_dom_list:
                RS_CC_Time0 = "Yes"
            else:
                RS_CC_Time0 = "No"
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'RS_CA_Time0'] = RS_CA_Time0
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'RS_CB_Time0'] = RS_CB_Time0
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'RS_CC_Time0'] = RS_CC_Time0
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'RS_Status_Time0'] = assign_restortation_status(RS_CA_Time0,
                                                                                                       RS_CB_Time0,
                                                                                                       RS_CC_Time0)

            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Pop2_T0_count'] = shrubs_count
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Pop2_T0_density'] = shrubs_count * 100
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Pop3_T0_count'] = other_T0
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'BioMass_Time0_sum'] = BioMass_Time0_sum
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'BioMass_Time40_sum'] = BioMass_Time40_sum
            ubio, flux = find_flux_underBio(site_type, line_status, 100, plot_polygon)
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'BioMass_Under_T0'] = ubio
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Methane_T0'] = flux

            # in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Soil_Carbon_T0'] = 0

            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Pop1_est_T40_count'] = est_stems_T40
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Pop1_est_T40_density (stems/ha)'] = est_stems_T40 * 100
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Pop1_est_T40_dominant'] = dominant_Species_T40
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Pop1_est_T40_RHt'] = Pop1_est_T40_RHt * 100

            status_T40 = Ass_Status_LenA_B_Time40((Pop1_est_T40_RHt * 100), site_type)
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Ass_Status_Time40'] = status_T40

            if dominant_count_T40 * 100 >= ref_den:
                RS_CA_Time40 = "Yes"
            else:
                RS_CA_Time40 = "No"

            if Pop1_est_T40_RHt * 100 >= 50.0:
                RS_CB_Time40 = "Yes"
            else:
                RS_CB_Time40 = "No"
            if dominant_Species_T40 in ref_dom_list:
                RS_CC_Time40 = "Yes"
            else:
                RS_CC_Time40 = "No"

            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'RS_CA_Time40'] = RS_CA_Time40
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'RS_CB_Time40'] = RS_CB_Time40
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'RS_CC_Time40'] = RS_CC_Time40
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'RS_Status_Time40'] = assign_restortation_status(RS_CA_Time40,
                                                                                                        RS_CB_Time40,
                                                                                                        RS_CC_Time40)

            ubio40, flux40 = find_flux_underBio(site_type, status_T40, 100, plot_polygon)
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'BioMass_Under_T40'] = ubio40
            in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Methane_T40'] = flux40
            # in_df.loc[in_df.index == (OLnFID, OLnPLT,OLnSEG), 'Soil_Carbon_T40'] = 0

            in_df.loc[in_df.index == (OLnFID,OLnPLT,OLnSEG),'P90_Age_Time0'] = P90_appected_trees_at_Year0
            in_df.loc[in_df.index == (OLnFID,OLnPLT,OLnSEG),'P90_Age_Time40'] = P90_appected_trees_at_YearN
        except Exception as e:
            print(e)
    in_df = in_df.rename(columns={scenario: 'Ass_Status_Time0', 'Tree_Species': 'Pop1_all_T0_species'})
    # in_df['Ass_Status_Time0']="Reference"
    in_df= in_df.loc[:, S1b_OffFP_columnsTitles]
    print("Calculating {} Plot's trees and shrubs density ......Done".format(where))
    return in_df


def calculate_LenB_OnFP(in_df, in_Ldf, in_Rdf, df_tree, scenario):
    in_df = in_df.rename(columns={scenario: 'Ass_Status_Time0', 'Tree_Species': 'Pop1_all_T0_species'})
    scenario='Ass_Status_Time0'
    in_df['Treated'] = "no"
    in_df['Treatment_Type'] ="None"
    in_df['Est_ht_threshold']=0.65
    in_df['Pop1_Pop2_Pop3_T0'] = 0
    in_df['Pop1_all_T0_count']=0
    in_df['Pop1_est_T0_count'] = 0
    in_df['Pop1_est_T0_density (stems/ha)'] = 0.0
    in_df['Ref_density']=800
    in_df['Ref_dominance'] = ""
    in_df['Ref_dominance'] = in_df['Ref_dominance'].astype('object')
    in_df['Ref_Ht'] = 2.0
    in_df['Pop1_est_T0_dominant'] = "None"
    in_df['Pop1_est_T0_dominant_Count'] = 0
    in_df['Pop1_est_T0_RHt']=0.0
    in_df['RS_CA_Time0']="No"
    in_df['RS_CB_Time0'] = "No"
    in_df['RS_CC_Time0'] = "No"
    in_df['RS_Status_Time0'] = ""
    in_df['Pop2_T0_count'] = 0
    in_df['Pop2_T0_density'] = 0.0
    in_df['Pop3_T0_count'] = 0
    in_df['BioMass_Under_T0']=0.0
    in_df['Methane_T0'] = 0.0
    in_df['Soil_Carbon_T0'] = 0.0
    in_df['Pop1_est_T40_dominant'] = "None"
    in_df['Pop1_est_T40_count'] = 0
    in_df['Pop1_est_T40_density (stems/ha)'] = 0.0
    in_df['Ass_Status_Time40'] = ""
    in_df['RS_CA_Time40']="No"
    in_df['RS_CB_Time40'] = "No"
    in_df['RS_CC_Time40'] = "No"
    in_df['RS_Status_Time40']=""
    in_df['BioMass_Under_T40'] = 0.0
    in_df['Methane_T40'] = 0.0
    in_df['Soil_Carbon_T40'] = 0.0
    if not 'BioMass_Time0_sum' in in_df.columns:
        in_df['BioMass_Time0_sum'] = 0.0
    if 'distribute_count' in in_df.columns:
        in_df=in_df.drop(columns=['distribute_count'])
    # in_df['P90_Age_Time0']=np.nan
    # in_df['P90_Age_Time40']=np.nan

    print("Scenario Using LenB:\n"
          "Calculating On FP Plot's acceptable trees and shrubs density and finding the dominant acceptable tree ......")
    print("calculating the dominant acceptable tree @ current year and 40 years later")
    print("and calculate understory biomass and methane flux by species.......")
    for OLnFID,OLnPLT in in_df.index:

        dominant_count_T0 = 0
        dominant_count_T40 = 0
        dominant_Species_T0 = "None"
        dominant_Species_T40 = "None"
        Pop1_est_T0_RHt = 0.0
        Pop1_est_T40_RHt = 0.0
        P90_appected_trees_at_Year0 = np.nan
        P90_appected_trees_at_YearN = np.nan
        other_T0 = 0
        stem_T40 = 0
        Pop1_all_T0_count = 0
        site_type = (in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Site_Type']).item()
        line_status = (in_df.loc[in_df.index == (OLnFID, OLnPLT), scenario]).item()
        plot_polygon = in_df.loc[in_df.index == (OLnFID, OLnPLT), 'geometry']
        Est_thr = Est_ht_thresholdB(site_type)
        ref_den = ref_density(site_type)
        ref_ht = ref_ht_func(site_type)
        est_stems_count=0
        shrubs_count=0
        total_species=0
        BioMass_Time0_sum=0.0
        BioMass_Time40_sum=0.0
        est_stems_T40=0
        Pop1_all_T0_height_min=0.0
        Pop1_all_T0_height_max=0.0
        Pop1_all_T40_height_min = 0.0
        Pop1_all_T40_height_max = 0.0

        try:
            if not pd.isna((in_df.loc[in_df.index==(OLnFID,OLnPLT),'Pop1_all_T0_species']).iat[0][0]):
                selected_Lrow = in_Ldf.loc[OLnFID, OLnPLT,:].copy()
                selected_Rrow = in_Rdf.loc[OLnFID, OLnPLT,:].copy()
                L_dominant_T0 = selected_Lrow.loc[selected_Lrow['Plot_area'].idxmax()]['Pop1_est_T0_dominant']
                R_dominant_T0 = selected_Rrow.loc[selected_Rrow['Plot_area'].idxmax()]['Pop1_est_T0_dominant']
                L_Plt_Area_T0 = selected_Lrow.loc[selected_Lrow['Plot_area'].idxmax()]['Plot_area'].item()
                R_Plt_Area_T0 = selected_Rrow.loc[selected_Rrow['Plot_area'].idxmax()]['Plot_area'].item()

                L_SiteType = (selected_Lrow.loc[selected_Lrow.index == (selected_Lrow['Plot_area'].idxmax()), 'Site_Type']).item()
                R_SiteType = (selected_Rrow.loc[selected_Rrow.index == (selected_Rrow['Plot_area'].idxmax()), 'Site_Type']).item()

                if site_type==L_SiteType and site_type==R_SiteType:
                    if L_Plt_Area_T0>=R_Plt_Area_T0:
                        ref_dom_list=[L_dominant_T0]
                        side = 'left'
                        if ref_dom_list == ['None']:
                            ref_dom_list = [R_dominant_T0]
                            side='right'
                    else:
                        ref_dom_list = [R_dominant_T0]
                        side = 'right'

                elif site_type==L_SiteType:
                    ref_dom_list=[L_dominant_T0]
                    side = 'left'
                    if ref_dom_list == ['None']:
                        ref_dom_list = [R_dominant_T0]
                        side = 'right'
                elif site_type==R_SiteType:
                    ref_dom_list = [R_dominant_T0]
                    side = 'right'
                    if ref_dom_list == ['None']:
                        ref_dom_list = [L_dominant_T0]
                        side = 'left'
                else:
                    ref_dom_list = ref_dominance(site_type)
                    side = "None"

                if ref_dom_list==['None']:
                    ref_dom_list = ref_dominance(site_type)
                    side = "None"

                if side=='left':
                    ref_ht = selected_Lrow.loc[selected_Lrow['Plot_area'].idxmax()]['P90_Age_Time0'].item()
                    ref_den= selected_Lrow.loc[selected_Lrow['Plot_area'].idxmax()]['Pop1_est_T0_density (stems/ha)'].item()


                elif side=='right':
                    ref_ht = selected_Rrow.loc[selected_Rrow['Plot_area'].idxmax()]['P90_Age_Time0'].item()
                    ref_den = selected_Rrow.loc[selected_Rrow['Plot_area'].idxmax()][
                                      'Pop1_est_T0_density (stems/ha)'].item()
                else:
                    #deflaut
                    pass

                if ref_ht==0.0:
                    ref_ht = ref_ht_func(site_type)
                if ref_den==0.0:
                    ref_den = ref_density(site_type)

                selected_row = df_tree[
                    ((df_tree['OLnFID'] == OLnFID) & (df_tree['OLnPLT'] == OLnPLT))].copy()
                total_species = len(selected_row)
                appected_shrubs = selected_row[selected_row['Tree_Species'].apply(lambda x: x in shrubs_list)]
                shrubs_count = len(appected_shrubs)
                BioMass_Time0_sum = BioMass_Time0_sum + appected_shrubs['BioMass_Time0'].sum(skipna=True)
                appected_trees_at_Year0 = selected_row[selected_row['Tree_Species'].apply(lambda x: x in trees_list)]
                Pop1_all_T0_count = len(appected_trees_at_Year0)
                if Pop1_all_T0_count > 0:
                    Pop1_all_T0_height_min = appected_trees_at_Year0['Height_T0'].min().item()
                    Pop1_all_T0_height_max = appected_trees_at_Year0['Height_T0'].max().item()

                BioMass_Time0_sum = BioMass_Time0_sum + appected_trees_at_Year0['BioMass_Time0'].sum(
                    skipna=True)
                appected_trees_at_Year0 = appected_trees_at_Year0[appected_trees_at_Year0['Height_T0'] >= Est_thr]
                est_stems_count = len(appected_trees_at_Year0)

                if est_stems_count > 0:
                    # P90_appected_trees_at_Year0 = appected_trees_at_Year0['Height_T0'].describe(percentiles=[0.9]).iloc[-2].item()
                    dominant_count_T0 = appected_trees_at_Year0['Tree_Species'].value_counts().iat[0].item()
                    dominant_Species_T0 = appected_trees_at_Year0['Tree_Species'].value_counts().index[0]
                    Pop1_est_T0_RHt = len(
                        appected_trees_at_Year0[appected_trees_at_Year0['Height_T0'] >= ref_ht*0.5]) / est_stems_count

                other_T0 = total_species - est_stems_count - shrubs_count

                #Time40
                selected_row = df_tree[
                    ((df_tree['OLnFID'] == OLnFID) & (df_tree['OLnPLT'] == OLnPLT))].copy()
                appected_trees_at_YearN = selected_row[selected_row['Tree_Species'].apply(lambda x: x in trees_list)]

                appected_trees_at_YearN = appected_trees_at_YearN[appected_trees_at_YearN['Height_Time40'] >= Est_thr]
                est_stems_T40 = len(appected_trees_at_YearN)

                if est_stems_T40 > 0:
                    Pop1_all_T40_height_min = appected_trees_at_YearN['Height_Time40'].min().item()
                    Pop1_all_T40_height_max = appected_trees_at_YearN['Height_Time40'].max().item()
                    # P90_appected_trees_at_YearN = appected_trees_at_YearN['Height_Time40'].describe(percentiles=[0.9]).iloc[
                    #     -2].item()
                    # dominant_count_T40 = appected_trees_at_YearN['Tree_Species'].value_counts().iat[0].item()
                    dominant_Species_T40 = appected_trees_at_YearN['Tree_Species'].value_counts().index[0]
                    # stem_T40=len(
                    #     appected_trees_at_YearN[appected_trees_at_YearN['Height_Time40'] >= ref_ht])
                    Pop1_est_T40_RHt = len(
                        appected_trees_at_YearN[appected_trees_at_YearN['Height_Time40'] >= ref_ht*0.5]) / est_stems_T40
                BioMass_Time40_sum = BioMass_Time40_sum + appected_trees_at_YearN['BioMass_Time40'].sum(skipna=True)



            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Est_ht_threshold'] = Est_thr
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_Pop2_Pop3_T0'] = total_species
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_all_T0_count'] = Pop1_all_T0_count
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_est_T0_count'] = est_stems_count
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_all_T0_height_min'] = Pop1_all_T0_height_min
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_all_T0_height_max'] = Pop1_all_T0_height_max
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_all_T40_height_min'] = Pop1_all_T40_height_min
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_all_T40_height_max'] = Pop1_all_T40_height_max

            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_est_T0_density (stems/ha)'] = est_stems_count * 100
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Ref_density'] = ref_den
            in_df.at[(OLnFID, OLnPLT), 'Ref_dominance'] = ref_dom_list
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Ref_Ht'] = ref_ht
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_est_T0_dominant'] = dominant_Species_T0
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_est_T0_dominant_Count'] = dominant_count_T0
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_est_T0_RHt'] = Pop1_est_T0_RHt * 100
            if est_stems_count * 100 >= (ref_den*(2/3)):
                RS_CA_Time0 = "Yes"
            else:
                RS_CA_Time0 = "No"

            if Pop1_est_T0_RHt * 100 >= 50.0:
                RS_CB_Time0 = "Yes"
            else:
                RS_CB_Time0 = "No"
            if dominant_Species_T0 in ref_dom_list:
                RS_CC_Time0 = "Yes"
            else:
                RS_CC_Time0 = "No"
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'RS_CA_Time0'] = RS_CA_Time0
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'RS_CB_Time0'] = RS_CB_Time0
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'RS_CC_Time0'] = RS_CC_Time0
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'RS_Status_Time0'] = assign_restortation_status(RS_CA_Time0,
                                                                                                       RS_CB_Time0,
                                                                                                       RS_CC_Time0)

            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop2_T0_count'] = shrubs_count
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop2_T0_density'] = shrubs_count * 100
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop3_T0_count'] = other_T0
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'BioMass_Time0_sum'] = BioMass_Time0_sum
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'BioMass_Time40_sum'] = BioMass_Time40_sum
            ubio, flux = find_flux_underBio(site_type, line_status, 100, plot_polygon)
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'BioMass_Under_T0'] = ubio
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Methane_T0'] = flux

            # in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Soil_Carbon_T0'] = 0

            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_est_T40_count'] = est_stems_T40
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_est_T40_density (stems/ha)'] = est_stems_T40 * 100
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_est_T40_dominant'] = dominant_Species_T40
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Pop1_est_T40_RHt'] = Pop1_est_T40_RHt * 100

            status_T40 = Ass_Status_LenA_B_Time40((Pop1_est_T40_RHt * 100), site_type)
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Ass_Status_Time40'] = status_T40

            if dominant_count_T40 * 100 >= ref_den*(2/3):
                RS_CA_Time40 = "Yes"
            else:
                RS_CA_Time40 = "No"

            if Pop1_est_T40_RHt * 100 >= 50.0:
                RS_CB_Time40 = "Yes"
            else:
                RS_CB_Time40 = "No"
            if dominant_Species_T40 in ref_dom_list:
                RS_CC_Time40 = "Yes"
            else:
                RS_CC_Time40 = "No"

            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'RS_CA_Time40'] = RS_CA_Time40
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'RS_CB_Time40'] = RS_CB_Time40
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'RS_CC_Time40'] = RS_CC_Time40
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'RS_Status_Time40'] = assign_restortation_status(RS_CA_Time40,
                                                                                                        RS_CB_Time40,
                                                                                                        RS_CC_Time40)

            ubio40, flux40 = find_flux_underBio(site_type, status_T40, 100, plot_polygon)
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'BioMass_Under_T40'] = ubio40
            in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Methane_T40'] = flux40
            # in_df.loc[in_df.index == (OLnFID, OLnPLT), 'Soil_Carbon_T40'] = 0

            # in_df.loc[in_df.index == (OLnFID,OLnPLT),'P90_Age_Time0'] = P90_appected_trees_at_Year0
            # in_df.loc[in_df.index == (OLnFID,OLnPLT),'P90_Age_Time40'] = P90_appected_trees_at_YearN
        except Exception as e:
            print(e)
    # in_df = in_df.rename(columns={scenario: 'Ass_Status_Time0', 'Tree_Species': 'Pop1_all_T0_species'})
    in_df= in_df.loc[:, S1b_OnFP_columnsTitles]
    print("Calculating Plot's trees and shrubs density, biomass, and flux @ current year and 40 years later......Done")
    return in_df

def Statis_by_Specie(in_plot,scenario,where):
    in_plot_Tree = in_plot.copy()

    plot_tree_population = in_plot_Tree[
        ['OLnFID','OLnPLT', 'OLnSEG', 'distribute', 'avg_width', 'max_width', 'Plt_Area', 'Site_Type', 'Tree_Species',
         scenario,'BioMass_Time0', 'BioMass_Time40', 'geometry', "Height_T0", "Height_Time40"]]

    print('Calculate Plot Statistic by Specie {}.....'.format(where))
    plot_tree_stat_by_Specie = plot_tree_population.dissolve(

        by=['OLnFID','OLnPLT', 'OLnSEG', 'Site_Type', 'Tree_Species'],

        aggfunc={'OLnFID': "first",
                 'OLnPLT':"first",
                 'OLnSEG': "first",
                 'Site_Type': "first",
                  scenario: "first",
                 'Tree_Species': "first",
                 "avg_width": "mean",
                 "max_width": "mean",
                 "Plt_Area": "mean",
                 "distribute": "count",
                 "Height_T0": ["min", "max"],
                 "BioMass_Time0": "sum",
                 "BioMass_Time40": "sum",
                 "Height_Time40": ["min", "max"],

                 }, skipna=True,

    ).sort_values(by=['OLnFID','OLnPLT', 'OLnSEG'])


    Group_by_Species_columns = [field_name[0] + "_" + field_name[1] if type(field_name) is tuple else field_name for
                                field_name in
                                list(plot_tree_stat_by_Specie.columns.values)]
    plot_tree_stat_by_Specie.columns = Group_by_Species_columns
    plot_tree_stat_by_Specie=plot_tree_stat_by_Specie.rename(columns={'OLnFID_first':'OLnFID','OLnPLT_first':'OLnPLT',
                                             'OLnSEG_first':"OLnSEG",'Site_Type_first':'Site_Type',
                                             scenario+"_first":scenario,'Tree_Species_first':"Tree_Species",
                                             "avg_width_mean":"Plt_width_mean","max_width_mean":"Plt_width_max",
                                             "Plt_Area_mean":"Plot_area","Height_T0_min":"Pop1_all_T0_height_min",
                                             "Height_T0_max":"Pop1_all_T0_height_max","Height_Time40_min":"Pop1_all_T40_height_min",
                                             "Height_Time40_max":"Pop1_all_T40_height_max"})
    plot_tree_stat_by_Specie = plot_tree_stat_by_Specie.reset_index(allow_duplicates=True)
    plot_tree_stat_by_Specie = plot_tree_stat_by_Specie.loc[:, ~plot_tree_stat_by_Specie.columns.duplicated()]
    plot_tree_stat_by_Specie.loc[pd.isna(plot_tree_stat_by_Specie.Pop1_all_T0_height_min), 'distribute_count'] = 0

    print('Calculate Plot Statistic by Specie {}.....Done'.format(where))

    return plot_tree_stat_by_Specie
def plot_statis(in_plot, scenario1A,scenario1B):
    in_plot_Tree=in_plot.copy()

    plot_tree_population = in_plot_Tree[['OLnFID', 'OLnPLT', 'OLnSEG', 'distribute', 'avg_width', 'max_width', 'Plt_Area', 'Site_Type', 'Tree_Species',
                                         scenario1A,scenario1B, 'BioMass_Time0', 'BioMass_Time40', 'geometry', "Height_T0", "Height_Time40"]]

    #print('Summarize Plot Statistic .....')
    plot_sum_statistic=plot_tree_population.dissolve(
        by=['OLnFID','OLnPLT','OLnSEG'],

        aggfunc={'OLnFID':"first",
                 'OLnPLT':"first",
                'OLnSEG':"first",
                 'Site_Type':lambda x:list(set(x)),
                 scenario1A: "first",
                 scenario1B: "first",
                 'Tree_Species': lambda x:list(set(x)),
                 "avg_width":"mean",
                 "max_width":"mean",
                 "Plt_Area":"mean",
                 "distribute": "count",
                 "Height_T0": ["min", "max"],
                 "BioMass_Time0": "sum",
                 "BioMass_Time40":  "sum",
                 "Height_Time40":["min","max"],


                 },skipna=True).sort_values(by=['OLnFID', 'OLnPLT','OLnSEG'])
    Group_by_Species_columns= [field_name[0] + "_" + field_name[1] if type(field_name) is tuple else field_name for
                                field_name in
                                list(plot_sum_statistic.columns.values)]
    plot_sum_statistic.columns = Group_by_Species_columns
    plot_sum_statistic = plot_sum_statistic.rename(
        columns={'OLnFID_first': 'OLnFID', 'OLnPLT_first': 'OLnPLT',
                 'OLnSEG_first': "OLnSEG", 'Site_Type_<lambda>': 'Site_Type',
                 scenario1A + "_first": scenario1A,scenario1B + "_first": scenario1B,
                 'Tree_Species_<lambda>': "Tree_Species",
                 "avg_width_mean": "Plt_width_mean", "max_width_mean": "Plt_width_max",
                 "Plt_Area_mean": "Plot_area", "Height_T0_min": "Pop1_all_T0_height_min",
                 "Height_T0_max": "Pop1_all_T0_height_max","Height_Time40_min":"Pop1_all_T40_height_min",
                                             "Height_Time40_max":"Pop1_all_T40_height_max"})

    plot_sum_statistic=plot_sum_statistic.reset_index(allow_duplicates=True)
    plot_sum_statistic = plot_sum_statistic.loc[:, ~plot_sum_statistic.columns.duplicated()]
    plot_sum_statistic.loc[pd.isna(plot_sum_statistic.Pop1_all_T0_height_min),'distribute_count']=0
    return plot_sum_statistic


def Return_ForestCover(on_FP,Loff_FP,Roff_FP,on_FP_trees):
    if "OLnSEG_first" in on_FP.columns:
        on_FP=on_FP.rename(columns={('OLnFID_first'): "OLnFID", ('OLnSEG_first'): "OLnSEG"})
        on_FP = on_FP.set_index(['OLnFID', 'OLnSEG'])
    if "OLnSEG_first" in Loff_FP.columns:
        Loff_FP=Loff_FP.rename(columns={('OLnFID_first'): "OLnFID", ('OLnSEG_first'): "OLnSEG"})
        Loff_FP = Loff_FP.set_index(['OLnFID', 'OLnSEG'])
    if "OLnSEG_first" in Roff_FP.columns:
        Roff_FP=Roff_FP.rename(columns={('OLnFID_first'): "OLnFID", ('OLnSEG_first'): "OLnSEG"})
        Roff_FP = Roff_FP.set_index(['OLnFID', 'OLnSEG'])
    if "OLnSEG_first" in on_FP_trees.columns:
        on_FP_trees=on_FP_trees.rename(columns={('OLnFID_first'): "OLnFID", ('OLnSEG_first'): "OLnSEG"})
        on_FP_trees = on_FP_trees.set_index(['OLnFID', 'OLnSEG'])
    elif "OLnSEG" in on_FP_trees.columns:
        on_FP_trees = on_FP_trees.set_index(['OLnFID', 'OLnSEG'])

    on_FP['Ca_MinDensity'] = None
    on_FP['Cb_Height'] = None
    on_FP['Cc_Dominance'] = None
    on_FP['Res_Status'] = None


    print("Calculating on FP Plot's 'Return to Forest Cover' Restpration Status......")

    for OLnFID, OLnSEG in on_FP.index:
        if not pd.isna((on_FP.loc[on_FP.index==(OLnFID,OLnSEG),'Species']).iat[0][0]):

            on_FP_EcoType = on_FP.EcositeTyp.loc[OLnFID, OLnSEG][0]
            stemsperha = on_FP['Stems/Ha'].loc[OLnFID, OLnSEG].item()
            selected_trees=on_FP_trees[on_FP_trees.index == (OLnFID,OLnSEG)]
            if on_FP_EcoType in[Ecosite_Type.MESIC_UPLAND.value,Ecosite_Type.DRY_UPLAND.value]:
                if  stemsperha>=800:
                    on_FP.loc[on_FP.index== (OLnFID,OLnSEG),'Ca_MinDensity']='Yes'
                else:
                    on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Ca_MinDensity'] ='No'

                appected_trees_at_YearC=selected_trees[selected_trees['Species'].apply(lambda x: x in trees_list)]
                appected_trees_aboveHt_YearC = appected_trees_at_YearC[appected_trees_at_YearC['Z'] >= Restoration_Cb_Thres.wetland.value]
                if len(appected_trees_at_YearC)>0:
                    if len(appected_trees_aboveHt_YearC)/ len(appected_trees_at_YearC)>=0.5:
                        on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Cb_Height'] ='Yes'
                    else:
                        on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Cb_Height'] = 'No'
                else:
                    on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Cb_Height'] = 'No'

            else:
                if stemsperha >= 1000:
                    on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Ca_MinDensity'] = 'Yes'
                else:
                    on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Ca_MinDensity'] = 'No'
                appected_trees_at_YearC = selected_trees[selected_trees['Species'].apply(lambda x: x in trees_list)]
                appected_trees_at_YearC = appected_trees_at_YearC[
                    appected_trees_at_YearC['Z'] >= Restoration_Cb_Thres.rest.value]
                if len(appected_trees_at_YearC) > 0:
                    if len(appected_trees_aboveHt_YearC) / len(appected_trees_at_YearC) >= 0.5:
                        on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Cb_Height'] = 'Yes'

                    else:
                        on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Cb_Height'] = 'No'
                else:
                    on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Cb_Height'] = 'No'


            if  on_FP.DominantC_Count.loc[OLnFID, OLnSEG].item()>0:
                selected_L = Loff_FP[(Loff_FP.index == (OLnFID,OLnSEG))]
                selected_R = Roff_FP[(Roff_FP.index == (OLnFID,OLnSEG))]
                if len(selected_L)>0:
                    on_LFP_EcoType=selected_L.EcositeTyp.loc[OLnFID, OLnSEG][0]
                if len(selected_R)>0:
                    on_RFP_EcoType = selected_R.EcositeTyp.loc[OLnFID, OLnSEG][0]


                if on_RFP_EcoType==on_FP_EcoType==on_LFP_EcoType:
                    if (on_FP.DominantC.loc[OLnFID, OLnSEG] == selected_R.DominantC.loc[OLnFID, OLnSEG]) or\
                            (on_FP.DominantC.loc[OLnFID, OLnSEG] == selected_L.DominantC.loc[OLnFID, OLnSEG]) :
                        on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Cc_Dominance'] = 'Yes'
                    else:
                        on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Cc_Dominance'] = 'No'

                elif on_LFP_EcoType==on_FP_EcoType:
                    if on_FP.DominantC.loc[OLnFID, OLnSEG] ==selected_L.DominantC.loc[OLnFID, OLnSEG]:
                        on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Cc_Dominance'] = 'Yes'
                    else:
                        on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Cc_Dominance'] = 'No'


                elif on_RFP_EcoType==on_FP_EcoType:
                    if on_FP.DominantC.loc[OLnFID, OLnSEG] == selected_R.DominantC.loc[OLnFID, OLnSEG]:
                        on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Cc_Dominance'] = 'Yes'
                    else:
                        on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Cc_Dominance'] = 'No'


                else:
                    on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Cc_Dominance'] = 'No'
                    print('4')
            else:
                on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Cc_Dominance'] = 'No'

    #
    #         if len(selected_L)>0 and len(selected_R)>0:
    #             if selected_L.EcositeTyp.item()[0] == on_FP_EcoTpe:
    #
    #             elif selected_R.EcositeTyp.item()[0] == on_FP_EcoTpe:

            Ca = on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Ca_MinDensity'].item()
            Cb = on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Cb_Height'].item()
            Cc = on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Cc_Dominance'].item()
            on_FP.loc[on_FP.index == (OLnFID, OLnSEG), 'Res_Status']=assign_restortation_status(Ca,Cb,Cc)
    print("Calculating Plot's trees and shrubs density @ current year and 40 years later......Done")
    return on_FP


def BioMassN_attributes(df_tree, out_tree, yearn):
    YearHt_Heading = 'Height_Time40'
    Year_heading = "Age_Time40"
    BioMass_heading = "BioMass_Time40"

    def Cal_Biomass_YearN(row):
        height = row[YearHt_Heading]
        BiomassT0=row['BioMass_Time0']
        species = row['Tree_Species']

        if ~np.isnan(height):
            if species == Species.Black_Spruce.value:
                biomass_result = BioTMass_Para.Blk_Spruce_a.value * height ** BioTMass_Para.Blk_Spruce_b.value
            elif species == Species.White_Spruce.value:
                biomass_result = BioTMass_Para.Wht_Spruce_a.value * height ** BioTMass_Para.Wht_Spruce_b.value
            elif species == Species.Tamarack.value:
                biomass_result = BioTMass_Para.Tamarack_a.value * height ** BioTMass_Para.Tamarack_b.value
            elif species == Species.Jack_Pine.value:
                biomass_result = BioTMass_Para.Jack_Pine_a.value * height ** BioTMass_Para.Jack_Pine_b.value
            elif species == Species.Balsam_Poplar.value:
                biomass_result = BioTMass_Para.Balsam_Poplar_a.value * height ** BioTMass_Para.Balsam_Poplar_b.value
            elif species == Species.Trembling_Aspen.value:
                biomass_result = BioTMass_Para.Trembling_Aspen_a.value * height ** BioTMass_Para.Trembling_Aspen_b.value
            elif species == Species.White_Birch.value:
                biomass_result = BioTMass_Para.White_Birch_a.value * height ** BioTMass_Para.White_Birch_b.value
            elif species == Species.Alder.value:
                biomass_result = BioTMass_Para.Alder_a.value * height**BioTMass_Para.Alder_b.value
            elif species == Species.Willow.value:
                biomass_result = BioTMass_Para.Willow_a.value * height** BioTMass_Para.Willow_b.value
            else:
                biomass_result =0

        else:
            biomass_result = 0

        return biomass_result

    def Cal_Ht_at_yearN(row):
        YearC=row['Age_Time0']
        heightC = row['Height_T0']
        species = row['Tree_Species']

        EcositeCode = row['Site_Type_Code']

        if YearC!=0: #~np.isnan(heightC) and ~np.isnan(YearC):
            if species == Species.Black_Spruce.value:
                heightN = Black_Spruce_Year_Curve(YearC+yearn, EcositeCode, heightC)
                Yearn = YearC + yearn

            elif species == Species.White_Spruce.value:
                heightN = White_Spruce_Year_Curve(YearC+yearn, EcositeCode, heightC)
                Yearn = YearC + yearn

            elif species == Species.Tamarack.value:
                heightN = Tamarack_Year_Curve(YearC+yearn, EcositeCode, heightC)
                Yearn = YearC + yearn

            elif species == Species.Jack_Pine.value:
                heightN = Jack_Pine_Year_Curve(YearC+yearn, EcositeCode, heightC)
                Yearn = YearC + yearn

            # No study
            elif species == Species.Balsam_Poplar.value:
                # heightN = Black_Spruce_Year_Curve(YearC+yearn, EcositeCode, heightC)
                heightN = 0.0
                Yearn = 0.0

            elif species == Species.Trembling_Aspen.value:
                heightN = Trembling_Aspen_Year_Curve(YearC+yearn, EcositeCode, heightC)
                Yearn = YearC + yearn

            elif species == Species.White_Birch.value:
                heightN = White_Birch_Year_Curve(YearC+yearn, EcositeCode, heightC)
                Yearn = YearC + yearn
            # No study
            elif species == Species.Alder.value:
                # heightN = Black_Spruce_Year_Curve(YearC+yearn, EcositeCode, heightC)
                heightN = 0.0
                Yearn = 0.0
            # No study
            elif species == Species.Willow.value:
                # heightN = Black_Spruce_Year_Curve(YearC+yearn, EcositeCode, heightC)
                heightN = 0.0
                Yearn = 0.0

            else:
                heightN = 0.0
                Yearn = 0.0
        else:
            heightN = 0.0
            Yearn = 0.0


        return pd.Series([heightN, Yearn], index=[YearHt_Heading, Year_heading])

    df_tree[Year_heading] = yearn
    print("Calculating predicted height of tree @ {} Years later......".format(yearn))
    df_tree[[YearHt_Heading, Year_heading]] = df_tree.apply(Cal_Ht_at_yearN, axis=1)
    print("Calculating predicted height of tree @ {} Years later......Done".format(yearn))
    print('%{}'.format(60))
    print("Calculating Biomass @ {} Years later......".format(yearn))
    df_tree[BioMass_heading] = df_tree.apply(Cal_Biomass_YearN, axis=1)
    print("Calculating Biomass @ {} Years later......Done".format(yearn))
    print('%{}'.format(80))
    # df_tree_studied=df_tree[(~np.isnan(df_tree['Age_TimeR']))
    #                         & (df_tree['Age_Time0'] >= df_tree['Age_TimeR']) &
    #                         (df_tree['A']<=100)]
    print("Saving data...........")
    out_path, out_filename = os.path.split(out_tree)
    # out_filename = out_filename.split(".")[0] + "_Filtered.parquet"
    out_filename = out_filename.split(".")[0] + "_Filtered_fr_all.gpkg"
    filtered_out = os.path.join(out_path, out_filename)
    # df_tree_studied.to_parquet(filtered_out, index=False)
    # df_tree.to_parquet(out_tree, index=False)
    # df_tree_studied.to_file(filtered_out, index=False,driver="GPKG")
    df_tree.to_file(out_tree, index=False,driver="GPKG")
    print("Saving data........... Done")
    print('%{}'.format(85))
    return df_tree,out_tree

def assign_ecositecode(in_df):
    def assign_code(row):
        EcositeType = row['Site_Type']
        match EcositeType:
            case Ecosite_Type.EXCLUDED.value:
                return Ecosite_Code.EXCLUDED.value
            case Ecosite_Type.MESIC_UPLAND.value:
                return Ecosite_Code.MESIC_UPLAND.value
            case Ecosite_Type.DRY_UPLAND.value:
                return Ecosite_Code.DRY_UPLAND.value
            case Ecosite_Type.WETLAND_TREED.value:
                return Ecosite_Code.WETLAND_TREED.value
            case Ecosite_Type.WETLAND_LOWDEN.value:
                return Ecosite_Code.WETLAND_LOWDEN.value
            case _:
                return Ecosite_Code.EXCLUDED.value

    in_df['Site_Type_Code']=in_df.apply(assign_code,axis=1)
    return in_df

def BioMassC_attributes(df_tree, out_tree):
    def Cal_Biomass(row):
        height = row['Height_T0']
        species = row['Tree_Species']
        if ~np.isnan(height):
            if species == Species.Black_Spruce.value:
                biomass_result = BioTMass_Para.Blk_Spruce_a.value * height ** BioTMass_Para.Blk_Spruce_b.value
            elif species == Species.White_Spruce.value:
                biomass_result = BioTMass_Para.Wht_Spruce_a.value * height **BioTMass_Para.Wht_Spruce_b.value
            elif species == Species.Tamarack.value:
                biomass_result = BioTMass_Para.Tamarack_a.value * height ** BioTMass_Para.Tamarack_b.value
            elif species == Species.Jack_Pine.value:
                biomass_result = BioTMass_Para.Jack_Pine_a.value * height ** BioTMass_Para.Jack_Pine_b.value
            elif species == Species.Balsam_Poplar.value:
                biomass_result = BioTMass_Para.Balsam_Poplar_a.value * height** BioTMass_Para.Balsam_Poplar_b.value
            elif species == Species.Trembling_Aspen.value:
                biomass_result = BioTMass_Para.Trembling_Aspen_a.value * height**BioTMass_Para.Trembling_Aspen_b.value
            elif species == Species.White_Birch.value:
                biomass_result = BioTMass_Para.White_Birch_a.value * height** BioTMass_Para.White_Birch_b.value
            elif species == Species.Alder.value:
                biomass_result = BioTMass_Para.Alder_a.value * height **BioTMass_Para.Alder_b.value
            elif species == Species.Willow.value:
                biomass_result = BioTMass_Para.Willow_a.value * height ** BioTMass_Para.Willow_b.value
            else:
                biomass_result = 0

        else:
            biomass_result = 0

        # return pd.Series([biomass_result, row['treeID']], index=['BioMassC','treeID'])
        return biomass_result

    def Cal_yearC_fr_curve(row):

        heightC = row['Height_T0']
        species = row['Tree_Species']

        EcositeCode = row['Site_Type_Code']

        if species == Species.Black_Spruce.value:
            yearC = Black_Spruce_Ht_Curve(heightC, EcositeCode)

        elif species == Species.White_Spruce.value:
            yearC = White_Spruce_Ht_Curve(heightC, EcositeCode)

        elif species == Species.Tamarack.value:
            yearC = Tamarack_Ht_Curve(heightC, EcositeCode)

        elif species == Species.Jack_Pine.value:
            yearC = Jack_Pine_Ht_Curve(heightC, EcositeCode)
        elif species == Species.Trembling_Aspen.value:
            yearC = Trembling_Aspen_Ht_Curve(heightC, EcositeCode)

        # No Study
        elif species == Species.Balsam_Poplar.value:
            # yearC = Black_Spruce_Ht_Curve(heightC,EcositeCode)
            yearC = 0.0
        # No Study
        elif species == Species.White_Birch.value:
            yearC = White_Birch_Ht_Curve(heightC,EcositeCode)
            # yearC = 0
        # No Study
        elif species == Species.Alder.value:
            # yearC = Black_Spruce_Ht_Curve(heightC,EcositeCode)
            yearC = 0.0
        # No Study
        elif species == Species.Willow.value:
            # yearC = Black_Spruce_Ht_Curve(heightC,EcositeCode)
            yearC = 0.0
        #other
        else:
            yearC = 0.0

        return yearC


    print("Calculating Current Year ......")
    df_tree['Age_Time0'] = df_tree.apply(Cal_yearC_fr_curve, axis=1)
    print("Calculating Current Year ......Done")
    print('%{}'.format(20))
    print("Calculating Biomass@Current Year ......")
    df_tree['BioMass_Time0'] = df_tree.apply(Cal_Biomass, axis=1)
    print("Calculating Biomass@Current Year ...... Done")
    print('%{}'.format(40))

    return df_tree, out_tree

def BioMass0_attributes(df_tree, out_tree):
    Year0Ht_Heading = 'Height_TR'
    Year0_heading = "Age_TimeR"
    # BioMass0_heading = "BioMass_TimeR"
    df_tree[Year0Ht_Heading] = np.nan
    df_tree[Year0_heading] = np.nan
    # df_tree[BioMass0_heading] = np.nan

    def Assign_year0(row):
        Ht_T0=row['Height_T0']
        species = row['Tree_Species']
        EcositeCode = row['Site_Type_Code']

        if species == Species.Black_Spruce.value:
            year0, Ht0 = Black_Spruce_yearR(EcositeCode)
        elif species == Species.White_Spruce.value:
            year0, Ht0 = White_Spruce_YearR(EcositeCode)

        elif species == Species.Tamarack.value:
            year0, Ht0 = Tamarack_Year0(EcositeCode)

        elif species == Species.Jack_Pine.value:
            year0, Ht0 = Jack_Pine_Year0(EcositeCode)

        elif species == Species.Trembling_Aspen.value:
            year0, Ht0 = Trembling_Aspen_Year0(EcositeCode)

            # No Study
        elif species == Species.Balsam_Poplar.value:
            year0 = 0
            Ht0 = Ht_T0

        elif species == Species.White_Birch.value:
            year0, Ht0 = White_Birch_Year0(EcositeCode)

        # No Study
        elif species == Species.Alder.value:
            year0 = 0
            Ht0 = Ht_T0
        # No Study
        elif species == Species.Willow.value:
            year0 = 0
            Ht0 = Ht_T0

        else:
            year0 = 0
            Ht0 = Ht_T0

        return pd.Series([year0, Ht0], index=[Year0_heading, Year0Ht_Heading])

    def Cal_Biomass_Year0(row):
        height = row[Year0Ht_Heading]
        Year0= row[Year0_heading]
        biomass_result = np.nan
        species = row['Tree_Species']

        if ~np.isnan(height) and ~np.isnan(Year0):
            if species == Species.Black_Spruce.value:
                biomass_result = BioTMass_Para.Blk_Spruce_a.value * height**BioTMass_Para.Blk_Spruce_b.value
            elif species == Species.White_Spruce.value:
                biomass_result = BioTMass_Para.Wht_Spruce_a.value * height** BioTMass_Para.Wht_Spruce_b.value
            elif species == Species.Tamarack.value:
                biomass_result = BioTMass_Para.Tamarack_a.value * height ** BioTMass_Para.Tamarack_b.value
            elif species == Species.Jack_Pine.value:
                biomass_result = BioTMass_Para.Jack_Pine_a.value * height ** BioTMass_Para.Jack_Pine_b.value
            elif species == Species.Balsam_Poplar.value:
                biomass_result = BioTMass_Para.Balsam_Poplar_a.value * height ** BioTMass_Para.Balsam_Poplar_b.value
            elif species == Species.Trembling_Aspen.value:
                biomass_result = BioTMass_Para.Trembling_Aspen_a.value * height** BioTMass_Para.Trembling_Aspen_b.value
            elif species == Species.White_Birch.value:
                biomass_result = BioTMass_Para.White_Birch_a.value * height** BioTMass_Para.White_Birch_b.value
            elif species == Species.Alder.value:
                biomass_result = BioTMass_Para.Alder_a.value * height** BioTMass_Para.Alder_b.value
            elif species == Species.Willow.value:
                biomass_result = BioTMass_Para.Willow_a.value * height** BioTMass_Para.Willow_b.value
        else:
            biomass_result = np.nan

        return biomass_result

    print("Assigning Reference Year for each Tree ......")
    df_tree[[Year0_heading, Year0Ht_Heading]] = df_tree.apply(Assign_year0, axis=1)
    print("Assigning Reference Year for each Tree ......Done")

    print('%{}'.format(60))
    return df_tree, out_tree

def chk_columns(in_df):
    if 'species' in in_df.columns:
        in_df=in_df.rename(columns={"species":"Tree_Species"})
    elif 'Species_' in in_df.columns:
        in_df=in_df.rename(columns={"Species_":"Tree_Species"})
    elif 'Tree_Species' in in_df.columns:
        print("Tree Species column found..")
        pass
    elif 'Tree Species' in in_df.columns:
        in_df=in_df.rename(columns={"Tree Species":"Tree_Species"})

    else:
        print("No 'Species' column found, stop processing..")
        exit()

    if 'landcover_class_name' in in_df.columns:
        in_df=in_df.rename(columns={"landcover_class_name":"landcover"})
    elif 'landcover' in in_df.columns:
        pass
    else:
        print("No 'landcover_class_name' column found, stop processing..")
        exit()
    has_code = False
    if 'gridcode' in in_df.columns:
        in_df=in_df.rename(columns={"gridcode":"Site_Type_Code"})
        has_code=True
    elif 'Ecosite' in in_df.columns:
        in_df=in_df.rename(columns={"Ecosite":"Site_Type_Code"})
        has_code=True
    elif 'Site_Type_Code' or 'Site Type Code' in in_df.columns:
        in_df = in_df.rename(columns={"Site Type Code": "Site_Type_Code"})

        has_code = True

    has_site_type = False
    if 'EcositeType' in in_df.columns:
        in_df = in_df.rename(columns={"EcositeType": "Site_Type"})
        has_site_type=True
    elif 'EcositeTyp' in in_df.columns:
        in_df = in_df.rename(columns={"EcositeTyp": "Site_Type"})
        has_site_type=True
    elif 'Site_Type' or 'Site Type' in in_df.columns:
        in_df = in_df.rename(columns={"Site Type": "Site_Type"})
        has_site_type=True

    if not has_code and has_site_type in in_df.columns:
        in_df=assign_ecositecode(in_df)
    elif not has_code and not has_site_type:
        print("No 'Ecosite Type/ Site Type' or 'Ecosite Code/ Site Type Code' column found, stop processing..")
        exit()

    if 'Z' in in_df.columns:
        in_df = in_df.rename(columns={"Z": "Height_T0"})
    elif 'Height_T0' in in_df.columns:
        pass
    if not 'distribute' in in_df.columns:
        in_df['distribute']=1

    return in_df

def del_joined_index(in_df):
    if 'index_left' in in_df.columns:
        in_df=in_df.drop(['index_left'],axis=1)
    if 'index_right' in in_df.columns:
        in_df = in_df.drop(['index_right'],axis=1)

    return in_df


def save_gpkg(df,filename):
    df.to_file(filename,index=False,driver="GPKG")



