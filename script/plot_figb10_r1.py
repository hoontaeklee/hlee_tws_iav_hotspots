'''plot figure b10'''

# =============================================================================
# load libraries and functions
# =============================================================================
import xarray as xr
import regionmask
import numpy as np
import numpy.ma as ma
import pandas as pd
import statsmodels.api as sm
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib as mpl
from config import dir_data, dir_plot
from _shared import plot_timeseries_tws_met_simplified_regions, running_mean, mask_and_weight
import os

# =============================================================================
# load data
# =============================================================================

date_start = '2002-01-01'
date_end = '2017-12-31'
nan_months_start = 3  # 2002-01 ~ 2002-03 == NaN
nan_months_end = 6  # 2017-07 ~ 2017~12 == NaN

# load tws
path = os.path.join(dir_data, 'GRACE_TWS_JPL_RL06Mv1CRIv01_200204_201706_scaleFactorApplied_areaWeighted_180_360_NaNs_for_2001_2019_masked_200201_201712_det.nc')
tws_iav_gra = xr.open_dataset(path)['graceTWS_det'].sel(time=slice(date_start, date_end)).values

path = os.path.join(dir_data, 'sindbad_veg3_det_monthly_180_360_200201_201712.nc')
tws_iav_sin = xr.open_dataset(path).sortby('lat', ascending=False)['wTotal_det'].sel(time=slice(date_start, date_end)).values

path = os.path.join(dir_data, 'h2m3_det_monthly_180_360_200201_201712.nc')
tws_iav_h2m = xr.open_dataset(path).sortby('lat', ascending=False)['tws_det'].sel(time=slice(date_start, date_end)).values

# load met.
path = os.path.join(dir_data, 'gpcp_v01r03_monthly_192_180_360_200201_201712_masked_det.nc')
ppt_iav_gpcp = xr.open_dataset(path).sel(time=slice(date_start, date_end)).precip_det.values

path = os.path.join(dir_data, 'prec.MSWEPv280.360.180.2002-01-01.2017-12-31.det.nc')
ppt_iav_mswep = xr.open_dataset(path).sel(time=slice(date_start, date_end)).prec_det.values

path = os.path.join(dir_data, 'tair.CRUJRA_v2_2.360.180.200201.201712.monthly.masked.det.nc')
tair_iav = xr.open_dataset(path).sel(time=slice(date_start, date_end)).tair_det.values

path = os.path.join(dir_data, 'ceres_syn1deged4a_Rn_6939_180_360_200201_201712_masked_det.nc')
rn_iav = xr.open_dataset(path).sel(time=slice(date_start, date_end)).Rn_det.values

# exclude common nans
path_common_nonnan = os.path.join(dir_data, 'common_nonnan_pixels.nc')
common_nonnan = xr.open_dataset(path_common_nonnan).common_nonnan_pixels.values

tws_iav_gra = np.where(common_nonnan, tws_iav_gra, np.nan)
tws_iav_sin = np.where(common_nonnan, tws_iav_sin, np.nan)
tws_iav_h2m = np.where(common_nonnan, tws_iav_h2m, np.nan)

ppt_iav_gpcp = np.where(common_nonnan, ppt_iav_gpcp, np.nan)
ppt_iav_mswep = np.where(common_nonnan, ppt_iav_mswep, np.nan)
tair_iav = np.where(common_nonnan, tair_iav, np.nan)
rn_iav = np.where(common_nonnan, rn_iav, np.nan)

# =============================================================================
# calc. regional mean time series
# =============================================================================

'''
- North America: 2, 3, 4, 5
- South America: 7, 8, 9, 10
- Africa: 15, 16, 17
- South Asia: 23, 24, 25
'''

# get trimmed-out pixels
file_name = os.path.join(dir_data, 'rsq_change_wo_extrm_norm_res_tws_jpl06v1_non_abs_det_using_20022017_simul_2001_2019.npz')
trimmed_pix = np.load(file_name)['trimmed_pix'][2:, 9, :, :]  # 2, 3: iav_sin, iav_h2m, 10% (idx 9) 

# extract hotspot index (cov, row, col)
hotspot_idx = np.argwhere(trimmed_pix==True)
region_groups = [[5], [7], [15, 16], [23]]  # N AM, S AM, AF, SE AS
trimmed_pix_common = np.logical_and(trimmed_pix[0], trimmed_pix[1])  # get commonly trimmed-out pixels among models
one_weights = np.ones((180, 360))

list_var = [
    'tws',
    'ppt',
    'tair',
    'rn'
]
list_tag = [
    'hotspots',
    'nonhotspots'
]
list_stat = [
    'mean',
    'std'
]

list_ppt = [
    'ppt_iav_gpcp',
    'ppt_iav_mswep'
]  # gpcp, mswep

list_tair = [
    tair_iav
]

list_rn = [
    rn_iav
]

list_tws_prod = [
    'gra',
    'sin',
    'h2m'
]

# create array
for v in range(len(list_var)):
    for t in range(len(list_tag)):
        for s in range(len(list_stat)):
            exec_var_name = 'region_{var}_{tag}_{stat}'.format(var=list_var[v], tag=list_tag[t], stat=list_stat[s])
            exec_create_var = exec_var_name + ' = np.empty((3, len(region_groups), len(tws_iav_sin)))*np.nan'
            if list_var[v] == 'ppt':
                exec_create_var = exec_var_name + ' = np.empty((len(list_ppt), len(region_groups), len(tws_iav_sin)))*np.nan'
            if list_var[v] == 'tair':
                exec_create_var = exec_var_name + ' = np.empty((len(list_tair), len(region_groups), len(tws_iav_sin)))*np.nan'
            if list_var[v] == 'rn':
                exec_create_var = exec_var_name + ' = np.empty((len(list_rn), len(region_groups), len(tws_iav_sin)))*np.nan'
            exec(exec_create_var)

# array to store the number of pixels for each tags
region_cnt = np.empty((1, len(region_groups)))*np.nan
region_cnt_hotspots = np.empty((1, len(region_groups)))*np.nan
region_cnt_nonhotspots = np.empty((1, len(region_groups)))*np.nan

# SREX boolean mask
res = 1.0
lat = np.linspace(-90+(res/2), 90-(res/2), int(180/res))
lon = np.linspace(-180+(res/2), 180-(res/2), int(360/res))
srex_mask = regionmask.defined_regions.srex.mask_3D(lon, lat)  # True: within the region

# load pixel area and land index
area = np.load(os.path.join(dir_data, 'gridAreaInKm2_180_360.npz'))['area']

for r in range(len(region_groups)):
    print('r={r}/{len_r}'.format(r=r, len_r=len(region_groups)-1))
    
    # get index of integrated regions 
    region_idx = srex_mask.sel(region=region_groups[r]).values
    region_idx = np.flip(np.any(region_idx, axis=0), axis=0)  # regionmask has an inverted latitude indexing
    
    # get mask
    mask_all_pix = region_idx  # all pixels
    mask_hotspots = np.logical_and(region_idx, trimmed_pix_common)  # hotspot pixels
    mask_nonhotspots = np.logical_and(region_idx, ~trimmed_pix_common)  # nonhotspot pixels

    # store values    
    
    ## the number of pixels
    region_cnt[0, r] = np.nansum(mask_all_pix)
    region_cnt_hotspots[0, r] = np.nansum(mask_hotspots)
    region_cnt_nonhotspots[0, r] = np.nansum(mask_nonhotspots)

    ## mean and std
    for v in range(len(list_var)):
        for t in range(len(list_tag)):
            for s in range(len(list_stat)):

                exec_var_name = 'region_{var}_{tag}_{stat}'.format(var=list_var[v], tag=list_tag[t], stat=list_stat[s])
                arr_shp_0 = eval(exec_var_name + '.shape[0]')
                
                for i in range(arr_shp_0):
                    exec_subset = '[i, r, :]'

                    # data name
                    data_name = '{v}_iav_{prod}'.format(v=list_var[v], prod=list_tws_prod[i])
                    if list_var[v] == 'ppt':
                        data_name = list_ppt[i]
                    if list_var[v] == 'tair':
                        data_name = 'tair_iav'
                    if list_var[v] == 'rn':
                        data_name = 'rn_iav'

                    if list_tag[t]=='hotspots':
                        mask = 'mask_hotspots'
                    if list_tag[t]=='nonhotspots':
                        mask = 'mask_nonhotspots'

                    weight = 'area'
                    if list_var[v] == 'tair':
                        weight = 'one_weights'

                    func = 'sum'
                    if list_stat[s] == 'std':
                        func = 'std'
                       
                    exec_mask_and_weight = 'mask_and_weight(data_timelatlon={d}, mask_to_retain={m}, weight={w}, func=func)'.format(d=data_name, m=mask, w=weight)
                    exec_assign = exec_var_name + exec_subset + ' = ' + exec_mask_and_weight

                    exec_data_exist = eval("\'" + data_name + "\'" + "in globals()")
                    if exec_data_exist:
                        exec(exec_assign)

# =============================================================================
# plot
# =============================================================================
param_dict = {
    'title_hot': 'Hotspots only',
    'ylim_met_hot': (-8, 8),
    'ylim_tws_hot': (-150, 150),
    'title_non': 'Non-hotspots only',
    'ylim_met_non': (-3, 3),
    'ylim_tws_non': (-60, 60)
}

fig = plot_timeseries_tws_met_simplified_regions(
    arr_tws=region_tws_hotspots_mean,
    arr_met=region_rn_hotspots_mean[0],
    arr_size=region_cnt_hotspots,
    ylim_tws=param_dict['ylim_tws_hot'],
    ylim_met=param_dict['ylim_met_hot'],
    title=param_dict['title_hot'],
    met='rn')
save_name = os.path.join(dir_plot, 'figb10.png')
fig.savefig(save_name,
            dpi=300,
            bbox_inches='tight',
           facecolor='w',
           transparent=False)