'''
- the script used to conduct the covariance matrix analysis
- for quantifying pixel-wise contribution to the global variance

- the example is for TWS IAV (e.g., one used for the figure 3)
- to implement for model errors (e.g., one used for the figure 4), use a new data list (line 89) like...
# data = [
#     tws_iav_sin_scArea - tws_iav_jpl06v1_scArea,
#     tws_iav_h2m_scArea - tws_iav_jpl06v1_scArea,
#     tws_msc_sin_scArea - tws_msc_jpl06v1_scArea,
#     tws_msc_h2m_scArea - tws_msc_jpl06v1_scArea
# ]


'''
# =============================================================================
# load libraries and functions
# =============================================================================

import os as os
import xarray as xr
import pandas as pd
import numpy as np
import numpy.ma as ma
import pingouin as pg
import copy
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
from config import dir_data, dir_plot

# =============================================================================
# load data
# =============================================================================

# load pixel area and land index
area = np.load(os.path.join(dir_data, 'gridAreaInKm2_180_360.npz'))['area']

# target period
date_start = '2002-04-01'
date_end = '2017-06-30'

# GRACE JPL mascon rl06mv1
path = os.path.join(dir_data, 'GRACE_TWS_JPL_RL06Mv1CRIv01_200204_201706_scaleFactorApplied_areaWeighted_180_360_NaNs_for_2001_2019_masked_200201_201712_det.nc')
tws_iav_jpl06v1 = xr.open_dataset(path)['graceTWS_det'].sel(time=slice(date_start, date_end)).values
tws_msc_jpl06v1 = xr.open_dataset(path)['graceTWS_msc'].sel(time=slice(date_start, date_end)).values

# Sindbad tt`
path = os.path.join(dir_data, 'sindbad_veg3_det_monthly_180_360_200201_201712.nc')
tws_iav_sin = xr.open_dataset(path).sortby('lat', ascending=False)['wTotal_det'].sel(time=slice(date_start, date_end)).values
tws_msc_sin = xr.open_dataset(path).sortby('lat', ascending=False)['wTotal_msc'].sel(time=slice(date_start, date_end)).values

# Hybrid bk
path = os.path.join(dir_data, 'h2m3_det_monthly_180_360_200201_201712.nc')
tws_iav_h2m = xr.open_dataset(path).sortby('lat', ascending=False)['tws_det'].sel(time=slice(date_start, date_end)).values
tws_msc_h2m = xr.open_dataset(path).sortby('lat', ascending=False)['tws_msc'].sel(time=slice(date_start, date_end)).values

# exclude union NaNs

# get nan pixels
path_common_nonnan = os.path.join(dir_data, 'common_nonnan_pixels.nc')
common_nonnan = xr.open_dataset(path_common_nonnan).common_nonnan_pixels.values

# exclude union NaNs
tws_iav_jpl06v1 = np.where(common_nonnan, tws_iav_jpl06v1, np.nan)
tws_msc_jpl06v1 = np.where(common_nonnan, tws_msc_jpl06v1, np.nan)
tws_iav_sin = np.where(common_nonnan, tws_iav_sin, np.nan)
tws_msc_sin = np.where(common_nonnan, tws_msc_sin, np.nan)
tws_iav_h2m = np.where(common_nonnan, tws_iav_h2m, np.nan)
tws_msc_h2m = np.where(common_nonnan, tws_msc_h2m, np.nan)
area = np.where(common_nonnan, area, np.nan)

# =============================================================================
# calculate area-weighted global time series
# =============================================================================

print('---start subtracting TWS mean for each pixel---')
tws_iav_jpl06v1_scArea = tws_iav_jpl06v1 * area.reshape(1, 180, 360) * 1000000  # km2 to m2
tws_msc_jpl06v1_scArea = tws_msc_jpl06v1 * area.reshape(1, 180, 360) * 1000000  # km2 to m2
tws_iav_sin_scArea = tws_iav_sin * area.reshape(1, 180, 360) * 1000000  # km2 to m2
tws_msc_sin_scArea = tws_msc_sin * area.reshape(1, 180, 360) * 1000000  # km2 to m2
tws_iav_h2m_scArea = tws_iav_h2m * area.reshape(1, 180, 360) * 1000000  # km2 to m2
tws_msc_h2m_scArea = tws_msc_h2m * area.reshape(1, 180, 360) * 1000000  # km2 to m2

# =============================================================================
# calculate the covariance matrix
# =============================================================================

print('---cov. calc. GRACE SINDBAD scArea start---')

data = [
    tws_iav_jpl06v1_scArea,
    tws_iav_sin_scArea,
    tws_iav_h2m_scArea,
    tws_msc_jpl06v1_scArea,
    tws_msc_sin_scArea,
    tws_msc_h2m_scArea
]


label = [
    'tws_iav_jpl06v1_scArea',
    'tws_iav_sin_scArea',
    'tws_iav_h2m_scArea',
    'tws_msc_jpl06v1_scArea',
    'tws_msc_sin_scArea',
    'tws_msc_h2m_scArea'
]


cnt_mon = tws_iav_sin_scArea.shape[0]
for i in range(len(data)):  # [1]:
    cov = ma.cov(ma.masked_invalid((data[i]).reshape(cnt_mon, -1)).transpose())
    cov_norm = np.nansum(cov, axis=1) / np.nansum(cov)
    print('sum('+label[i]+' - '+label[i]+'.T): '+str((cov-cov.T).sum()))
    
    out_name = os.path.join(dir_data, 'cov_TWS_det_scArea_' + label[i] + '_det_using_20022017_simul_2001_2019.npz')
    # np.savez(out_name,
    #          cov_norm=cov_norm)
    
    # draw cov plot to check the matrix is symmetry or not (cov - cov.T should be equal to 0)
    vmin=np.nanpercentile(cov.data.ravel(), q=25)
    vmax=np.nanpercentile(cov.data.ravel(), q=75)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = ax.flatten()
    im1 = ax[0].imshow(cov, cmap='bwr', vmin=vmin, vmax=vmax)
    plt.colorbar(im1, ax=ax[0], cmap='bwr')
    ax[0].set_title(label[i])
    
    im1 = ax[1].imshow(cov-cov.T, cmap='bwr', vmin=vmin, vmax=vmax)
    plt.colorbar(im1, ax=ax[1], cmap='bwr')
    ax[1].set_title(label[i]+'-'+label[i]+'.T')

    fig_name = os.path.join(dir_plot, 'cov_TWS_det_scArea_' + label[i] + '_det_using_20022017_simul_2001_2019.png')
    # fig.savefig(fig_name, bbox_inches='tight', dpi=300)
        
    del cov
    del cov_norm

# load npz files
label = [
    'tws_iav_jpl06v1_scArea',
    'tws_iav_sin_scArea',
    'tws_iav_h2m_scArea',
    'tws_msc_jpl06v1_scArea',
    'tws_msc_sin_scArea',
    'tws_msc_h2m_scArea'
]
ar = []
for i in range(len(label)):
    file_name = os.path.join(dir_data, 'cov_TWS_det_scArea_' + label[i] + '_det_using_20022017_simul_2001_2019.npz')
    ar.append(np.load(file_name)['cov_norm'])

# save to one npz file
out_name = os.path.join(dir_data, 'cov_TWS_det_scArea_det_using_20022017_simul_2001_2019.npz')
# np.savez(out_name, 
#          norm_tws_iav_jpl06v1=ar[0],  
#          norm_tws_iav_sindbad=ar[0], 
#          norm_tws_iav_h2m=ar[1], 
#          norm_tws_msc_jpl06v1=ar[3],           
#          norm_tws_msc_sindbad=ar[2], 
#          norm_tws_msc_h2m=ar[3])