from regen_assess.constants_1 import *
import math as m

def Black_Spruce_Ht_Curve(height,EcositeCode):
    if EcositeCode==Ecosite_Code.TRAN_TREED:
        #height= 2.4734*log(year) - 4.2238
        year=m.exp((height+4.2238)/2.4734)
        if year <= 0.0:
            year = 0.1
    elif EcositeCode==Ecosite_Code.WETLAND_TREED:
        # height=1.0487*log(year) - 1.4801
        year=m.exp((height + 1.4801)/1.0487)
        if year <= 0.0:
            year = 0.1
    # Assume growth as Wetland Treed
    elif EcositeCode==Ecosite_Code.WETLAND_LOWDEN:
        # height=1.0487*log(year) - 1.4801
        year=m.exp((height + 1.4801)/1.0487)
        if year <= 0.0:
            year = 0.1
    else:
        year=0.0

    return year

def Black_Spruce_Year_Curve(yearn,EcositeCode,in_height):
    if EcositeCode==Ecosite_Code.TRAN_TREED:
        height= 2.4734*m.log(yearn) - 4.2238
    elif EcositeCode==Ecosite_Code.WETLAND_TREED:
        height=1.0487*m.log(yearn) - 1.4801
    # Assume growth as Wetland Treed
    elif EcositeCode==Ecosite_Code.WETLAND_LOWDEN:
        height=1.0487*m.log(yearn) - 1.4801
    else:
        height=in_height
    return height

def Black_Spruce_yearR(EcositeCode):
    if EcositeCode==Ecosite_Code.TRAN_TREED:
        year= 8.0
        Ht0=0.8
    elif EcositeCode==Ecosite_Code.WETLAND_TREED:
        year=8.0
        Ht0 = 0.65
    # Assume growth as Wetland Treed
    elif EcositeCode==Ecosite_Code.WETLAND_LOWDEN:
        year=8.0
        Ht0 = 0.65
    else:
        year=0.0
        Ht0=0.0
    return year,Ht0

def White_Spruce_Ht_Curve(height,EcositeCode):
    if EcositeCode == Ecosite_Code.MESIC_UPLAND:
        # height=2.4839*m.log(year) - 3.9902
        year = m.exp((height+3.9902)/2.4839)
        if year <= 0.0:
            year = 0.1
    # # Assume growth as MESIC_UPLAND
    # elif EcositeCode == Ecosite_Code.WETLAND_TREED:
    #     # height=2.4839*m.log(year) - 3.9902
    #     year = m.exp((height+3.9902)/2.4839)
    #     if year <= 0.0:
    #         year = 0.1
    # # Assume growth as MESIC_UPLAND
    # elif EcositeCode == Ecosite_Code.WETLAND_LOWDEN:
    #     # height=2.4839*m.log(year) - 3.9902
    #     year = m.exp((height+3.9902)/2.4839)
    #     if year <= 0.0:
    #         year = 0.1


    else:
        year=0.0

    return year

def White_Spruce_Year_Curve(yearn,EcositeCode,in_height):
    if EcositeCode == Ecosite_Code.MESIC_UPLAND:
        height=2.4839*m.log(yearn) - 3.9902
    # Assume growth as MESIC_UPLAND
    # elif EcositeCode == Ecosite_Code.WETLAND_TREED:
    #     height=2.4839*m.log(yearn) - 3.9902
    # elif EcositeCode == Ecosite_Code.WETLAND_LOWDEN:
    #     height=2.4839*m.log(yearn) - 3.9902
    else:
        height=in_height
    return height

def White_Spruce_YearR(EcositeCode):
    if EcositeCode == Ecosite_Code.MESIC_UPLAND:
        year=7.5
        Ht0=0.8
    # Assume growth as MESIC_UPLAND
    # elif EcositeCode == Ecosite_Code.WETLAND_TREED:
    #     year=7.5
    #     Ht0=0.8
    # elif EcositeCode == Ecosite_Code.WETLAND_LOWDEN:
    #     year=7.5
    #     Ht0=0.8
    else:
        year=0.0
        Ht0=0.0
    return year,Ht0

def Tamarack_Ht_Curve(height,EcositeCode):
    if EcositeCode == Ecosite_Code.WETLAND_TREED:
        #height= 0.9657*log(year) - 0.8784
        year = m.exp((height - 0.5169612) / 1.1093387)
        if year <= 0.0:
            year = 0.1
    # Assume growth as Wetland Treed
    elif EcositeCode == Ecosite_Code.WETLAND_LOWDEN:
        #height= 0.9657*log(year) - 0.8784
        year = m.exp((height - 0.5169612) / 1.1093387)
        if year <= 0.0:
            year = 0.1
    else:
        year = 0.0

    return year


def Tamarack_Year_Curve(yearn,EcositeCode,in_height):
    if EcositeCode == Ecosite_Code.WETLAND_TREED:
        height = 0.9657*m.log(yearn) - 0.8784
    # Assume growth as Wetland Treed
    elif EcositeCode == Ecosite_Code.WETLAND_LOWDEN:
        height = 0.9657*m.log(yearn) - 0.8784
    else:
        height = in_height
    return height

def Tamarack_Year0(EcositeCode):
    if EcositeCode == Ecosite_Code.WETLAND_TREED:
        year = 5.0
        Ht0=0.65
    # Assume growth as Wetland Treed
    elif EcositeCode == Ecosite_Code.WETLAND_LOWDEN:
        year = 5.0
        Ht0=0.65
    else:
        year=0.0
        Ht0=0.0
    return year,Ht0

def Jack_Pine_Ht_Curve(height,EcositeCode):
    if EcositeCode == Ecosite_Code.DRY_UPLAND:
        # height=2.2202*m.log(yearn) - 2.7217
        year = m.exp((height +2.7217) / 2.2202)
        if year <= 0.0:
            year = 0.1
    elif EcositeCode == Ecosite_Code.TRAN_TREED:
        # height = 2.3268*m.log(yearn) - 2.0029
        year = m.exp ((height +2.0029) / 2.3268)
        if year <= 0.0:
            year = 0.1
    else:
        year = 0.0

    return year

def Jack_Pine_Year_Curve(yearn,EcositeCode,in_height):
    if EcositeCode == Ecosite_Code.DRY_UPLAND:
        height =2.2202*m.log(yearn) - 2.7217
    elif EcositeCode == Ecosite_Code.TRAN_TREED:
        height = 2.3268*m.log(yearn) - 2.0029
    else:
        height = in_height
    return height

def Jack_Pine_Year0(EcositeCode):
    if EcositeCode == Ecosite_Code.DRY_UPLAND:
        year =4.5
        Ht0=0.6
    elif EcositeCode == Ecosite_Code.TRAN_TREED:
        year=4.0
        Ht0=0.8
    else:
        year=0.0
        Ht0=0.0
    return year,Ht0



def Trembling_Aspen_Ht_Curve(height,EcositeCode):

    if EcositeCode == Ecosite_Code.MESIC_UPLAND:
        # height = 5.7148*m.log(yearn) - 8.5106
        year = m.exp((height +8.5106) / 5.7148)
        if year <= 0.0:
            year = 0.1
    else:
        year = 0.0

    return year

def Trembling_Aspen_Year_Curve(yearn,EcositeCode,in_height):

    if EcositeCode == Ecosite_Code.MESIC_UPLAND:
        height = 5.7148*m.log(yearn) - 8.5106
    else:
        height = in_height
    return height

def Trembling_Aspen_Year0(EcositeCode):
    if EcositeCode == Ecosite_Code.MESIC_UPLAND:
        # height = 5.7148*m.log(yearn) - 8.5106
        year0=5.5
        Ht0=1.2
    else:
        year0 = 0.0
        Ht0 = 0.0

    return year0,Ht0

def White_Birch_Ht_Curve(height,EcositeCode):

    if EcositeCode == Ecosite_Code.MESIC_UPLAND:
        # height = 4.2791*m.log(yearn) - 6.1205
        year = m.exp((height +6.1205) / 4.2791)
        if year <= 0.0:
            year = 0.1
    elif EcositeCode == Ecosite_Code.TRAN_TREED:
        # height = 1.7341*m.log(yearn) - 1.1937
        year = m.exp((height +1.1937) / 1.7341)
        if year <= 0.0:
            year = 0.1
    elif EcositeCode == Ecosite_Code.WETLAND_TREED:
        # height = 0.6882*m.log(yearn) + 0.0227
        year = m.exp((height +0.0227) / 0.6882)
        if year <= 0.0:
            year = 0.1
    # Assume growth as Wetland Treed
    elif EcositeCode == Ecosite_Code.WETLAND_LOWDEN:
        # height = 0.6882*m.log(yearn) + 0.0227
        year = m.exp((height +0.0227) / 0.6882)
        if year <= 0.0:
            year = 0.1
    else:
        year = 0.0

    return year



def White_Birch_Year_Curve(yearn,EcositeCode,in_height):

    if EcositeCode == Ecosite_Code.MESIC_UPLAND:
        height = 4.2791*m.log(yearn) - 6.1205
    elif EcositeCode == Ecosite_Code.TRAN_TREED:
        height = 1.7341*m.log(yearn) - 1.1937

    elif EcositeCode == Ecosite_Code.WETLAND_TREED:
        height = 0.6882*m.log(yearn) + 0.0227
    # Assume growth as Wetland Treed
    elif EcositeCode == Ecosite_Code.WETLAND_LOWDEN:
        height = 0.6882*m.log(yearn) + 0.0227
    else:
        height = in_height
    return height

def White_Birch_Year0(EcositeCode):
    if EcositeCode == Ecosite_Code.MESIC_UPLAND:
        # height = 4.2791*m.log(yearn) - 6.1205
        year0=6.0
        Ht0=1.2
    elif EcositeCode == Ecosite_Code.TRAN_TREED:
        # height = 1.7341 * m.log(yearn) - 1.1937
        year0 = 6.0
        Ht0 = 1.2

    elif EcositeCode == Ecosite_Code.WETLAND_TREED:
        # height = 0.6882 * m.log(yearn) + 0.0227
        year0 = 6.0
        Ht0 = 1.2
    # Assume growth as Wetland Treed
    elif EcositeCode == Ecosite_Code.WETLAND_LOWDEN:
        # height = 0.6882 * m.log(yearn) + 0.0227
        year0 = 6.0
        Ht0 = 1.2
    else:
        year0 = 0.0
        Ht0 = 0.0

    return year0,Ht0
def assign_restortation_status(Ca,Cb,Cc):

    match Ca,Cb,Cc:
        case 'Yes','Yes','Yes':
            return 'Restored'
        case 'Yes', 'Yes', 'No':
            return 'Conditionally restored'
        case 'Yes', 'No', 'Yes':
            return 'On Track'
        case 'Yes', 'No', 'No':
            return 'Conditionally On Track'
        case _:
            return 'Unrestored'


def Est_ht_thresholdA(site_type):
    if isinstance(site_type,list):
        site_type=site_type[0]
    match site_type:
        case Ecosite_Type.WETLAND_TREED.value:
            return 0.65
        case Ecosite_Type.WETLAND_LOWDEN.value:
            return 0.60
        case Ecosite_Type.DRY_UPLAND.value:
            return 0.6
        case Ecosite_Type.MESIC_UPLAND.value:
            return 0.8
        case Ecosite_Type.TRAN_TREED.value:
            return 0.8
        case _:
            return 0.6

def Est_ht_thresholdB(site_type):
    if isinstance(site_type,list):
        site_type=site_type[0]
    match site_type:
        case Ecosite_Type.WETLAND_TREED.value:
            return 0.65
        case Ecosite_Type.WETLAND_LOWDEN.value:
            return 0.60
        case Ecosite_Type.DRY_UPLAND.value:
            return 0.6
        case Ecosite_Type.MESIC_UPLAND.value:
            return 0.8
        case Ecosite_Type.TRAN_TREED.value:
            return 0.8
        case _:
            return 0.6

def ref_density(site_type):
    if isinstance(site_type,list):
        site_type=site_type[0]
    match site_type:
        case Ecosite_Type.WETLAND_TREED.value:
            return 1000
        case Ecosite_Type.WETLAND_LOWDEN.value:
            return 800
        case Ecosite_Type.DRY_UPLAND.value:
            return 800
        case Ecosite_Type.MESIC_UPLAND.value:
            return 1000
        case Ecosite_Type.TRAN_TREED.value:
            return 1000
        case _:
            return 0

def ref_dominance(site_type):
    if isinstance(site_type,list):
        site_type=site_type[0]
    match site_type:
        case Ecosite_Type.WETLAND_TREED.value:
            return [Species.Black_Spruce.value,Species.Tamarack.value]
        case Ecosite_Type.WETLAND_LOWDEN.value:
            return [Species.Black_Spruce.value,Species.Tamarack.value]
        case Ecosite_Type.DRY_UPLAND.value:
            return [Species.Jack_Pine.value]
        case Ecosite_Type.MESIC_UPLAND.value:
            return [Species.Trembling_Aspen.value,Species.White_Birch.value,Species.White_Spruce.value]
        case Ecosite_Type.TRAN_TREED.value:
            return [Species.Black_Spruce.value,Species.Jack_Pine.value,Species.White_Birch.value]
        case _:
            return []

def ref_ht_func(site_type):
    if isinstance(site_type,list):
        site_type=site_type[0]
    match site_type:
        case Ecosite_Type.WETLAND_TREED.value:
            return 2.0
        case Ecosite_Type.WETLAND_LOWDEN.value:
            return 2.0
        case Ecosite_Type.DRY_UPLAND.value:
            return 5.0
        case Ecosite_Type.MESIC_UPLAND.value:
            return 5.0
        case Ecosite_Type.TRAN_TREED.value:
            return 5.0
        case _:
            return 0.0


def Ass_Status_LenA_B_Time40(in_percent, site_type):
    if not isinstance(in_percent,float):
        in_percent=float(in_percent)
    if isinstance(site_type,list):
        site_type=site_type[0]
    match site_type:
        case Ecosite_Type.WETLAND_TREED.value:
            if in_percent<30:
                return Line_Status.Arr.value
            elif 30<=in_percent<70:
                return Line_Status.Reg.value
            elif in_percent>=70:
                return Line_Status.AdvReg.value
        case Ecosite_Type.WETLAND_LOWDEN.value:
            if in_percent < 30:
                return Line_Status.Arr.value
            elif 30 <= in_percent < 70:
                return Line_Status.Reg.value
            elif in_percent >= 70:
                return Line_Status.AdvReg.value
        case Ecosite_Type.DRY_UPLAND.value:
            if in_percent < 30:
                return Line_Status.Arr.value
            elif 30 <= in_percent < 70:
                return Line_Status.Reg.value
            elif in_percent >= 70:
                return Line_Status.AdvReg.value
        case Ecosite_Type.MESIC_UPLAND.value:
            if in_percent < 30:
                return Line_Status.Arr.value
            elif 30 <= in_percent < 70:
                return Line_Status.Reg.value
            elif in_percent >= 70:
                return Line_Status.AdvReg.value
        case Ecosite_Type.TRAN_TREED.value:
            if in_percent<30:
                return Line_Status.Arr.value
            elif 30<=in_percent<70:
                return Line_Status.Reg.value
            elif in_percent>=70:
                return Line_Status.AdvReg.value
        case _:
            return Line_Status.Excl.value

def find_flux_underBio(ecosite_list,status,target_area,plot):

    plot_area=plot.area.item()
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

    return total_under_bio,total_flux


def accept_trees_fr_ht(ht,ecosite,species):
    accepted = False
    ok = False
    def XUp_JPine(ht):
        if ht >= Accp_Trees_MinHt.XUp_JPine.value:
            ok=True
        else:
            ok=False
            return ok

    def MUp_T_Asp(ht):
        if ht >= Accp_Trees_MinHt.MUp_T_Asp.value:
            ok = True
        else:
            ok = False
            return ok

    def MUp_Wht_Sp(ht):
        if ht >= Accp_Trees_MinHt.MUp_Wht_Sp.value:
            ok = True
        else:
            ok = False
            return ok

    def MUp_Wht_Bi(ht):
        if ht >= Accp_Trees_MinHt.MUp_Wht_Bi.value:
            ok = True
        else:
            ok = False
            return ok

    def Tran_J_Pine(ht):
        if ht >= Accp_Trees_MinHt.Tran_J_Pine.value:
            ok = True
        else:
            ok = False
            return ok

    def Tran_Blk_Sp(ht):
        if ht >= Accp_Trees_MinHt.Tran_Blk_Sp.value:
            ok = True
        else:
            ok = False
            return ok

    def Tran_Wht_Bi(ht):
        if ht >= Accp_Trees_MinHt.Tran_Wht_Bi.value:
            ok = True
        else:
            ok = False
            return ok

    def Wlt_T_Tam(ht):
        if ht >= Accp_Trees_MinHt.Wlt_T_Tam.value:
            ok = True
        else:
            ok = False
            return ok

    def Wlt_T_Blk_Sp(ht):
        if ht >= Accp_Trees_MinHt.Wlt_T_Blk_Sp.value:
            ok = True
        else:
            ok = False
            return ok

    def Wlt_T_Wht_Bi(ht):
        if ht >= Accp_Trees_MinHt.Wlt_T_Wht_Bi.value:
            ok = True
        else:
            ok = False
            return ok
    option={
        Ecosite_Type.DRY_UPLAND.value+"_"+Species.Jack_Pine.value: XUp_JPine,
        Ecosite_Type.MESIC_UPLAND.value + "_" + Species.Trembling_Aspen.value: MUp_T_Asp,
        Ecosite_Type.MESIC_UPLAND.value + "_" + Species.White_Spruce.value: MUp_Wht_Sp,
        Ecosite_Type.MESIC_UPLAND.value + "_" + Species.White_Birch.value: MUp_Wht_Bi,
        Ecosite_Type.TRAN_TREED.value + "_" + Species.Jack_Pine.value: Tran_J_Pine,
        Ecosite_Type.TRAN_TREED.value + "_" + Species.Black_Spruce.value: Tran_Blk_Sp,
        Ecosite_Type.TRAN_TREED.value + "_" + Species.White_Birch.value: Tran_Wht_Bi,
        Ecosite_Type.WETLAND_TREED.value + "_" + Species.Tamarack.value: Wlt_T_Tam,
        Ecosite_Type.WETLAND_TREED.value + "_" + Species.Black_Spruce.value: Wlt_T_Blk_Sp,
        Ecosite_Type.WETLAND_TREED.value + "_" + Species.White_Birch.value: Wlt_T_Wht_Bi,


    }
    accepted=option.get((ecosite+"_"+species),ht)

    return accepted

