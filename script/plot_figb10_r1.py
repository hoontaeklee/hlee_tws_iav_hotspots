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
import os

def running_mean(x, N):
    '''
    N-months running mean (+- 2/N months), e.g. nanmean(month6~month18) for month12
    for the first 6 and last 5 values, less then 12 values will be used for the mean
    x: np.array, N: window length, output: outA: np.array()
    '''
    intV = int(N / 2)
    lenX = len(x)
    outA = np.ones(np.shape(x))
    for _ind in range(lenX):
        _indMin = max(_ind - intV, 0)
        _indMax = min(_ind + intV + 1, lenX)
        if N%2==0:
            _indMax = _indMax - 1
        outA[_ind] = np.nanmean(x[_indMin:_indMax])
    return outA

def mask_and_weight(data_timelatlon, mask_to_retain, weight, func='sum'):
    
    data_timelatlon = np.where(mask_to_retain, data_timelatlon, np.nan)
    weight = np.where(mask_to_retain, weight, np.nan)
    result = np.empty(data_timelatlon.shape[0]) * np.nan
    if func=='sum':       
        result = np.array([np.nansum(_ts * weight)/(np.nansum(weight)) for _ts in data_timelatlon])
    if func=='std':
        result = np.array([np.nanstd(_ts * weight)/(np.nansum(weight)) for _ts in data_timelatlon])
        
    return result

def plot_timeseries_tws_met_simplified_regions(arr_tws, arr_met, arr_size, ylim_tws, ylim_met, title, met='ppt'):
    # ppt or tair or rn
    if met == 'ppt':
        var_met = 'ppt'
        ylab_met = 'PPT IAV (mm)'
    elif met == 'rn':
        var_met = 'rn'
        ylab_met = r'Rn IAV (W/m$^2$)'
    else:
        var_met = 'tair'
        ylab_met = 'Tair IAV (degC)'
    
    # gridspec inside gridspec
    global fig
    fig = plt.figure(constrained_layout=True, figsize=(9, 8))
    subfigs = fig.subfigures(2, 2, wspace=0.00, hspace=0.0)
    subfigs = subfigs.flatten()

    label = ['GRACE', 'SINDBAD', 'H2M']
    palette = dict(zip(label, ['black', 'limegreen', '#bc15b0']))
    handles = [mpl.patches.Patch(color=palette[x], label=x) for x in palette.keys()]
    x_years = pd.date_range(start='2002-01',end='2017-12', freq='1MS')
    x_years_label = np.arange(2002, 2015, 3)
    alphabets = [chr(i) for i in range(ord('a'),ord('z')+1)]
    srex_names = ['North America', 'South America', 'Africa', 'South Asia']  # srex_mask.names.values

    for r_idx in range(4):

        # upper subplots

        ax_box = [0.06, 0.51, 1, 0.41]
        if r_idx in [2, 3]:
            ax_box[1::2] = [x+0.05 for x in ax_box[1::2]]
        ax = subfigs[r_idx].add_axes(ax_box, xticklabels=[], ylim=ylim_tws)
        ax.grid(True, linestyle='--')
        ax.spines['bottom'].set_visible(False)
        ax.annotate(alphabets[r_idx], xy=(0.03, 0.89), xycoords='axes fraction', fontsize=15, weight='bold')       
        tws_mean = []
        tws_mean_smth = []
        for i in range(3):
            tws_mean.append(arr_tws[i, r_idx])
            tws_mean_smth.append(running_mean(arr_tws[i, r_idx], 12))
            ax.plot(x_years, tws_mean_smth[i], label=label[i], color=list(palette.values())[i])
            ax.plot(x_years, tws_mean[i], color=list(palette.values())[i], alpha=0.2)

        if r_idx in [0, 2]:
            ax.set_ylabel('TWS IAV (mm)', rotation='vertical', fontsize=12)
        if r_idx in [1, 3]:
            ax.tick_params(labelleft=False)        

        if r_idx in [0]:
            leg = ax.legend(labels=label, bbox_to_anchor=(1.5, 1.3), ncol=3, fontsize=12)
            leg.legendHandles[0].set_color(list(palette.values())[0])
            leg.legendHandles[1].set_color(list(palette.values())[1])
            leg.legendHandles[2].set_color(list(palette.values())[2])
        ax.set_yticks(np.linspace(ylim_tws[0], ylim_tws[1], num=5))
        ax.set_yticks(ax.get_yticks()[1:-1])

        # lower subplots
        ax_box = [0.06, 0.08, 1, 0.41]
        if r_idx in [2, 3]:
            ax_box[1::2] = [x+0.05 for x in ax_box[1::2]]
        ax = subfigs[r_idx].add_axes(ax_box, ylim=ylim_met)
        ax.set_yticks(np.linspace(ylim_met[0], ylim_met[1], num=5))
        arr_mean = arr_met[r_idx]
        arr_mean_smth = running_mean(arr_met[r_idx], 12)
        
        if met == 'ppt':
            ax.bar(x_years, arr_mean_smth, width=30.5, label='PPT IAV', color='tab:blue', alpha=0.8)
            ax.bar(x_years, arr_mean, width=30.5, color='tab:blue', alpha=0.2)
        elif met == 'rn':
            ax.plot(x_years, arr_mean_smth, label='Rn IAV', color='tab:blue', alpha=0.8)
            ax.plot(x_years, arr_mean, color='tab:blue', alpha=0.2)
        else:
            ax.plot(x_years, arr_mean_smth, label='Tair IAV', color='tab:blue', alpha=0.8)
            ax.plot(x_years, arr_mean, color='tab:blue', alpha=0.2)
        ax.tick_params(axis='y')
        ax.grid(True, linestyle='--')
        ax.spines['top'].set_color('silver')
        ax.set_yticks(ax.get_yticks()[1:-1])

        if r_idx in [0, 2]:
            ax.set_ylabel(ylab_met, rotation='vertical', fontsize=12)
        if r_idx in [0, 1]:
            ax.tick_params(labelbottom=False)
        if r_idx in [1, 3]:
            ax.tick_params(labelleft=False)

        # add r2 and the number of pixels...
        txt_pix_cnt = 'Number of pixels: {cnt}'.format(cnt=int(arr_size[0, r_idx]))
        ax.annotate(txt_pix_cnt, xy=(0.01, 0.02), xycoords='axes fraction')

        rsq_sin = '{:.02f}'.format(pg.corr(tws_mean_smth[1], tws_mean_smth[0])['r'][0]**2)
        rsq_h2m = '{:.02f}'.format(pg.corr(tws_mean_smth[2], tws_mean_smth[0])['r'][0]**2)
        rsq_string = 'R$^2$ (GRACE, Model): {sin}(SINDBAD), {h2m}(H2M)'.format(sin=rsq_sin, h2m=rsq_h2m)
        ax.annotate(rsq_string, xy=(0.01, 0.1), xycoords='axes fraction')

        rsq_gra = '{:.02f}'.format(pg.corr(tws_mean_smth[0], arr_mean_smth)['r'][0]**2)
        rsq_sin = '{:.02f}'.format(pg.corr(tws_mean_smth[1], arr_mean_smth)['r'][0]**2)
        rsq_h2m = '{:.02f}'.format(pg.corr(tws_mean_smth[2], arr_mean_smth)['r'][0]**2)
        rsq_string = 'R$^2$ (TWS, {met}): {gra}(GRACE), {sin}(SINDBAD), {h2m}(H2M)'.format(gra=rsq_gra, sin=rsq_sin, h2m=rsq_h2m, met=var_met)
        ax.annotate(rsq_string, xy=(0.01, 0.2), xycoords='axes fraction')


    fig.text(0, 0.98, title, fontsize=14, weight='bold')
    fig.text(0.5, 0.01, 'Years', ha='center', va='center', fontsize=12)

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

plot_timeseries_tws_met_simplified_regions(
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