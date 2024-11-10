import numpy as np
from enum import Flag, Enum, IntEnum, unique


@unique
class CenterlineStatus(IntEnum):
    SUCCESS = 1
    FAILED = 2
    REGENERATE_SUCCESS = 3
    REGENERATE_FAILED = 4


@unique
class Ecosite_Type(Enum):
    EXCLUDED = "EXCLUDED"
    MESIC_UPLAND = "MESIC_UPLAND"
    DRY_UPLAND = "DRY_UPLAND"
    TRAN_TREED = "TRAN_TREED"
    WETLAND_TREED = "WETLAND_TREED"
    WETLAND_LOWDEN = "WETLAND_LOWDEN"

Ecosite_Type_list=[]
Ecosite_Type_list.append((Ecosite_Type.EXCLUDED.value))
Ecosite_Type_list.append((Ecosite_Type.MESIC_UPLAND.value))
Ecosite_Type_list.append((Ecosite_Type.DRY_UPLAND.value))
Ecosite_Type_list.append((Ecosite_Type.TRAN_TREED.value))
Ecosite_Type_list.append((Ecosite_Type.WETLAND_TREED.value))
Ecosite_Type_list.append((Ecosite_Type.WETLAND_LOWDEN.value))


@unique
class Ecosite_Code(IntEnum):
    EXCLUDED = 0
    MESIC_UPLAND = 1
    DRY_UPLAND = 2
    TRAN_TREED = 3
    WETLAND_TREED = 4
    WETLAND_LOWDEN = 5

@unique
class Species(Enum):

    Black_Spruce = "Black Spruce"
    White_Spruce = "White Spruce"
    Balsam_Fir="Balsam Fir"
    Tamarack= "Tamarack"
    Jack_Pine= "Jack Pine"
    Balsam_Poplar="Balsam Poplar"
    Trembling_Aspen="Trembling Aspen"
    White_Birch="White Birch"
    Alder="Alder"
    Willow="Willow"

# class trees:
#     def __init__(self,name):
#         self.name=name
trees_list=[]
trees_list.append((Species.Black_Spruce.value))
trees_list.append((Species.White_Spruce.value))
trees_list.append((Species.Balsam_Fir.value))
trees_list.append((Species.Tamarack.value))
trees_list.append((Species.Jack_Pine.value))
trees_list.append((Species.Balsam_Poplar.value))
trees_list.append((Species.Trembling_Aspen.value))
trees_list.append((Species.White_Birch.value))

# class shrubs:
#     def __init__(self,name):
#         self.name=name
shrubs_list=[]
shrubs_list.append((Species.Alder.value))
shrubs_list.append((Species.Willow.value))



class BioTMass_Para(float, Enum):
    Blk_Spruce_a=0.141
    Blk_Spruce_b = 2.323
    Wht_Spruce_a=0.192
    Wht_Spruce_b =2.323
    Balsam_Fir_a=0.147
    Balsam_Fir_b=2.348
    Tamarack_a=0.114
    Tamarack_b=2.385
    Jack_Pine_a=0.087
    Jack_Pine_b=2.464
    Balsam_Poplar_a=0.115
    Balsam_Poplar_b=2.456
    Trembling_Aspen_a=0.127
    Trembling_Aspen_b=2.456
    White_Birch_a=0.084
    White_Birch_b=2.451
    Alder_a=0.047
    Alder_b=2.395
    Willow_a=0.063
    Willow_b=2.325

@unique
class Line_Status(Enum):

    AdvReg = "advanced"
    Reg = "regenerating"
    Arr="arrested"
    Excl = "occluded"
    Treated='treated'
class Under_BioTMass_PlotLevel(float, Enum):
    EXCLUDED = np.nan

    MESIC_UPLAND_AdvReg = 6.06
    MESIC_UPLAND_Reg = 3.88
    MESIC_UPLAND_Arr = 1.70
    MESIC_UPLAND_Treated_Rip=0.17
    MESIC_UPLAND_Treated_Plant = 4.05

    DRY_UPLAND_AdvReg = 11.6
    DRY_UPLAND_Reg = 7.42
    DRY_UPLAND_Arr = 3.25
    DRY_UPLAND_Treated_Screen=0.32
    DRY_UPLAND_Treated_Plant = 7.75

    TRAN_TREED_AdvReg = 27.18
    TRAN_TREED_Reg = 34.49
    TRAN_TREED_Arr = 41.8
    TRAN_TREED_Treated_Inv = 20.56
    TRAN_TREED_Treated_Plant = 37.83

    WETLAND_TREED_AdvReg = 48.31
    WETLAND_TREED_Reg = 65.10
    WETLAND_TREED_Arr = 81.09
    WETLAND_TREED_Treated_Inv = 40.95
    WETLAND_TREED_Treated_Plant = 71.61

    WETLAND_LOWDEN_AdvReg = 81.9
    WETLAND_LOWDEN_Reg = 81.9
    WETLAND_LOWDEN_Arr = 81.9
    WETLAND_LOWDEN_Treated_Inv = 40.95
    WETLAND_LOWDEN_Treated_Plant = 90.09



class Methane_Flux_PlotLevel(float, Enum):
    EXCLUDED = np.nan

    MESIC_UPLAND_AdvReg = 0.00
    MESIC_UPLAND_Reg = 0.00
    MESIC_UPLAND_Arr = 0.00
    MESIC_UPLAND_Treated_Rip = 0.00
    MESIC_UPLAND_Treated_Plant = 0.00

    DRY_UPLAND_AdvReg = 0.00
    DRY_UPLAND_Reg = 0.00
    DRY_UPLAND_Arr = 0.00
    DRY_UPLAND_Treated_Screen = 0.00
    DRY_UPLAND_Treated_Plant = 0.00

    TRAN_TREED_AdvReg = 1.15
    TRAN_TREED_Reg = 1.88
    TRAN_TREED_Arr = 1.88
    TRAN_TREED_Treated_Inv = 2.49
    TRAN_TREED_Treated_Plant = 1.88

    WETLAND_TREED_AdvReg = 3.31
    WETLAND_TREED_Reg = 5.42
    WETLAND_TREED_Arr = 5.42
    WETLAND_TREED_Treated_Inv = 7.19
    WETLAND_TREED_Treated_Plant = 5.42

    WETLAND_LOWDEN_AdvReg = 3.31
    WETLAND_LOWDEN_Reg = 5.42
    WETLAND_LOWDEN_Arr = 5.42
    WETLAND_LOWDEN_Treated_Inv = 7.19
    WETLAND_LOWDEN_Treated_Plant = 5.42

@unique
class ParallelMode(IntEnum):
    SEQUENTIAL = 1
    MULTIPROCESSING = 2
    CONCURRENT = 3
    DASK = 4
    # RAY = 5

PARALLEL_MODE = ParallelMode.MULTIPROCESSING


class OperationCancelledException(Exception):
    pass

class Accp_Trees_MinHt(float, Enum):
    XUp_JPine=0.6
    MUp_T_Asp=1.2
    MUp_Wht_Sp = 0.8
    MUp_Wht_Bi = 1.2
    Tran_J_Pine = 0.8
    Tran_Blk_Sp =0.8
    Tran_Wht_Bi=1.2
    Wlt_T_Tam = 0.65
    Wlt_T_Blk_Sp =0.65
    Wlt_T_Wht_Bi = 1.2

class Establishment_Thres_lenA(float, Enum):
    Treed_wetland=0.65
    LowDens_Tr_Wetland=0.6
    Upland_Dry = 0.6
    Mesic_Upland=0.8
    transitional = 0.8



class Restoration_Cb_Thres(float, Enum):
    wetland=2.0
    rest=5.0

class table_column(Enum):
    col_a='OLnFID'


S1a_OnFP_columnsTitles =['OLnSEG', 'Site_Type', 'Ass_Status_Time0', 'Treated', 'Treatment_Type', 'Pop1_all_T0_species',
                'Plt_width_mean','Plt_width_max','Plot_area','Pop1_Pop2_Pop3_T0','Pop1_all_T0_count','Est_ht_threshold',
                'Pop1_est_T0_count','Pop1_est_T0_density (stems/ha)','Ref_density','Ref_dominance','Ref_Ht','Pop1_all_T0_height_min',
                'Pop1_all_T0_height_max','Pop1_est_T0_dominant','Pop1_est_T0_dominant_Count','Pop1_est_T0_RHt','RS_CA_Time0',
                'RS_CB_Time0',	'RS_CC_Time0',	'RS_Status_Time0',	'Pop2_T0_count',	'Pop2_T0_density',	'Pop3_T0_count',
                'BioMass_Time0_sum',	'BioMass_Under_T0',	'Methane_T0','Soil_Carbon_T0','Pop1_est_T40_count','Pop1_est_T40_density (stems/ha)',
                'Ass_Status_Time40','BioMass_Time40_sum','Pop1_all_T40_height_min','Pop1_all_T40_height_max','Pop1_est_T40_dominant',
                'Pop1_est_T40_RHt','RS_CA_Time40','RS_CB_Time40','RS_CC_Time40','RS_Status_Time40',	'BioMass_Under_T40',
                'Methane_T40','Soil_Carbon_T40','geometry']

S1b_OnFP_columnsTitles =['OLnSEG', 'Site_Type', 'Ass_Status_Time0', 'Treated', 'Treatment_Type', 'Pop1_all_T0_species',
                'Plt_width_mean','Plt_width_max','Plot_area','Pop1_Pop2_Pop3_T0','Pop1_all_T0_count','Est_ht_threshold',
                'Pop1_est_T0_count','Pop1_est_T0_density (stems/ha)','Ref_density','Ref_dominance','Ref_Ht','Pop1_all_T0_height_min',
                'Pop1_all_T0_height_max','Pop1_est_T0_dominant','Pop1_est_T0_dominant_Count','Pop1_est_T0_RHt','RS_CA_Time0',
                'RS_CB_Time0',	'RS_CC_Time0',	'RS_Status_Time0',	'Pop2_T0_count',	'Pop2_T0_density',	'Pop3_T0_count',
                'BioMass_Time0_sum',	'BioMass_Under_T0',	'Methane_T0','Soil_Carbon_T0','Pop1_est_T40_count','Pop1_est_T40_density (stems/ha)',
                'Ass_Status_Time40','BioMass_Time40_sum','Pop1_all_T40_height_min','Pop1_all_T40_height_max','Pop1_est_T40_dominant',
                'Pop1_est_T40_RHt','RS_CA_Time40','RS_CB_Time40','RS_CC_Time40','RS_Status_Time40',	'BioMass_Under_T40',
                'Methane_T40','Soil_Carbon_T40','geometry']

S1b_OffFP_columnsTitles =['Site_Type', 'Ass_Status_Time0', 'Treated', 'Treatment_Type', 'Pop1_all_T0_species',
                'Plt_width_mean','Plt_width_max','Plot_area','Pop1_Pop2_Pop3_T0','Pop1_all_T0_count','Est_ht_threshold',
                'Pop1_est_T0_count','Pop1_est_T0_density (stems/ha)','Ref_density','Ref_dominance','Ref_Ht','Pop1_all_T0_height_min',
                'Pop1_all_T0_height_max','Pop1_est_T0_dominant','Pop1_est_T0_dominant_Count','Pop1_est_T0_RHt','RS_CA_Time0',
                'RS_CB_Time0',	'RS_CC_Time0',	'RS_Status_Time0','P90_Age_Time0','Pop2_T0_count',	'Pop2_T0_density',	'Pop3_T0_count',
                'BioMass_Time0_sum',	'BioMass_Under_T0',	'Methane_T0','Soil_Carbon_T0','Pop1_est_T40_count','Pop1_est_T40_density (stems/ha)',
                'Ass_Status_Time40','BioMass_Time40_sum','Pop1_all_T40_height_min','Pop1_all_T40_height_max','Pop1_est_T40_dominant',
                'Pop1_est_T40_RHt','RS_CA_Time40','RS_CB_Time40','RS_CC_Time40','RS_Status_Time40','P90_Age_Time40',	'BioMass_Under_T40',
                'Methane_T40','Soil_Carbon_T40','geometry']

working_dir= ''