"""
- the script used to detrend a data set
- examples are for detrending TWS data (GRACE JPL, SINDBAD, and H2M)
"""

# =============================================================================
# load libraries and functions
# =============================================================================

import os
import glob as glob
import copy
import xarray as xr
import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
from config import dir_data, dir_plot


def remove_pix_mean(dat_full, idx_start, idx_end):
    # dat_full: numpy array with the dimension of (time, lat, lon)
    dt = np.ones_like(dat_full) * np.nan
    yrs = np.arange(1, len(dt) + 1)
    for i in range(180):
        for j in range(360):
            dat_p = dat_full[:,i,j]
            nanc=np.sum(~np.isnan(dat_p))
            if nanc > 3:
                dt[:,i,j] = dat_p - np.nanmean(dat_p[idx_start:(idx_end+1)])
    return dt    

def detrend_monthly(dat_full):
    '''
    data - trend
    '''
    # dat_full: numpy array with the dimension of (time, lat, lon)
    dat_yr_mon = dat_full.reshape(-1,12,180,360)
    dt = np.ones_like(dat_yr_mon) * np.nan
    yrs = np.arange(1, len(dt) + 1)
    for mn in range(12):
        dat_mon = dat_yr_mon[:,mn,:,:]
        for i in range(180):
            for j in range(360):
                dat_p = dat_mon[:,i,j]
                nanc=np.sum(~np.isnan(dat_p))
                non_nan_ind = ~np.isnan(dat_p)
                if nanc > 3:
                    dat_p_sel = dat_p[non_nan_ind]
                    yr_sel = yrs[non_nan_ind]
                    p = np.polyfit(yr_sel, dat_p_sel, 1)  # can be replaced with RLM
                    tr = yrs * p[0] + p[1]
                    dt[:,mn,i,j] = dat_p - tr
    dt=dt.reshape(-1,180,360)
    return dt

def get_monthly_msc(dat_full):
    # dat_full: numpy array with the dimension of (time, lat, lon)
    dat_yr_mon = dat_full.reshape(-1,12,180,360)
    dt = np.ones_like(dat_yr_mon) * np.nan
    yrs = np.arange(1, len(dt) + 1)
    for mn in range(12):
        dat_mon = dat_yr_mon[:,mn,:,:]
        for i in range(180):
            for j in range(360):
                dat_p = dat_mon[:,i,j]
                nanc=np.sum(~np.isnan(dat_p))
                non_nan_ind = ~np.isnan(dat_p)
                if nanc > 3:
                    dat_p_sel = dat_p[non_nan_ind] - np.nanmean(dat_p[non_nan_ind])
                    yr_sel = yrs[non_nan_ind]
                    p = np.polyfit(yr_sel, dat_p_sel,1)  # can be replaced with RLM
                    tr = yrs * p[0] + p[1]
                    dt[:,mn,i,j] = np.nanmean(dat_p - tr)  # np.nanmean(dat_p)
    dt=dt.reshape(-1,180,360)
    return dt

def get_trends(dat_full):
    # dat_full: numpy array with the dimension of (time, lat, lon)
    # get linear trend (slope) of full-time series of each pixel
    dt_slp = np.ones(dat_full.shape[1:]) * np.nan
    dt_idx = np.arange(dat_full.shape[0])
    for i in range(180):
        for j in range(360):
            dat_p = dat_full[:,i,j]
            nanc=np.sum(~np.isnan(dat_p))
            non_nan_ind = ~np.isnan(dat_p)
            if nanc > 3:
                dat_p_sel = dat_p[non_nan_ind]
                dt_idx_sel = dt_idx[non_nan_ind]
                p = np.polyfit(dt_idx_sel, dat_p_sel, 1)  # can be replaced with RLM
                dt_slp[i, j] = p[0]
    return dt_slp    

def detrend_dataset(ds_orig, idx_start, idx_end, use_anomaly=False):
    list_var = list(ds_orig.data_vars)  # list of variables in the original xr.dataset
    list_dims = list(ds_orig.dims.keys())  # ['time', 'lat', 'lon'] 
    list_coords = []  # coordinations for dimensions of the xr.dataset
    for i in range(len(list_dims)):
        list_coords.append(ds_orig[list_dims[i]])
    dic_ds = {}  # dictionary for creating the final xr.dataset
    list_label_key = [
        'ano',
        'det',
        'msc',
        'tr',
        'ano_std',
        'msc_amp',
        'det_std'
    ]
    # add dictionary items of each variable
    for i, var in enumerate(list_var):
        arr_var = ds_orig[var].values
        arr_ano = remove_pix_mean(ds_orig[var].values, idx_start=idx_start, idx_end=idx_end)
        if use_anomaly:
            arr_det = detrend_monthly(arr_ano)
            arr_msc = get_monthly_msc(arr_ano)
            arr_tr = get_trends(arr_ano)
        else:
            arr_det = detrend_monthly(arr_var)
            arr_msc = get_monthly_msc(arr_var)
            arr_tr = get_trends(arr_var)
        arr_ano_std = np.nanstd(arr_ano, axis=0)
        arr_msc_amp = np.nanmax(arr_msc, axis=0) - np.nanmin(arr_msc, axis=0)
        arr_det_std = np.nanstd(arr_det, axis=0)
        list_arr = [arr_ano, arr_det, arr_msc, arr_tr, arr_ano_std, arr_msc_amp, arr_det_std]  # should be matched with the label list (list_label_key)
        for k in range(len(list_label_key)):
            if list_arr[k].ndim==3:
                new_da = xr.DataArray(list_arr[k], coords=list_coords, dims=list_dims)  # w/ time
            elif list_arr[k].ndim==2:
                new_da = xr.DataArray(list_arr[k], coords=list_coords[1:], dims=list_dims[1:])  # w/o time
            else:
                raise ValueError('The data is neither 2- nor 3-d array.')
            new_key = var+'_'+list_label_key[k]
            dic_ds[new_key] = new_da
    ds_det = xr.Dataset(data_vars=dic_ds)
    return ds_det

# =============================================================================
# detrend data
# =============================================================================

date_start = '2002-01-01' 
date_end = '2017-12-31'

# load missing timesteps of GRACE
path_gra_invalid_dates = os.path.join(dir_data, 'GRACE_TWS_JPL_RL06Mv1CRIv01_invalid_dates_200201_201712.npy')
gra_invalid_dates = np.load(path_gra_invalid_dates)
gra_invalid_dates_3d = np.tile(gra_invalid_dates, (180, 360, 1)).transpose(2, 0, 1)


# detrend GRACE RL06Mv1CRI (2001-2019 simulation)
# load the original .nc
path_ds = os.path.join(dir_data, 'GRCTellus.JPL.200204_201706.GLO.RL06M.MSCNv01CRIv01.scaleFactorApplied.areaWeighted.192_180_360.with_NaNs_for_2001_2019.masked.nc')

ds = xr.open_dataset(path_ds).sel(time=slice(date_start, date_end))

# detrend
ds_orig = ds  # the original xr.dataset that you want to detrend
ds_det = detrend_dataset(ds_orig=ds_orig, idx_start=0, idx_end=ds.dims['time'])  # idx_start=15, idx_end=197

save_name = os.path.join(dir_data, 'GRACE_TWS_JPL_RL06Mv1CRIv01_200204_201706_scaleFactorApplied_areaWeighted_180_360_NaNs_for_2001_2019_masked_200201_201712_det.nc')
# ds_det.to_netcdf(save_name)


# detrend SINDBAD tws by tt (2001-2019 simulation)
# load the original .nc
path_ds = glob.glob(os.path.join(dir_data, 'SINDBAD_opti_Nov2021', '*.nc'))
ds = xr.open_mfdataset(path_ds).sel(time=slice(date_start, date_end)).resample(time='1M').reduce(np.nanmean)
ds['evapTotal'] = xr.open_mfdataset(path_ds).sel(time=slice(date_start, date_end)).evapTotal.resample(time='1M').reduce(np.nansum)
ds['roTotal'] = xr.open_mfdataset(path_ds).sel(time=slice(date_start, date_end)).roTotal.resample(time='1M').reduce(np.nansum)

# detrend
ds_orig = ds  # the original xr.dataset that you want to detrend
ds_orig = xr.where(~gra_invalid_dates_3d, ds_orig, np.nan)  # apply the same missing dates of GRACE
ds_det = detrend_dataset(ds_orig=ds_orig, idx_start=0, idx_end=ds.dims['time']) 

save_name = os.path.join(dir_data, 'sindbad_veg3_det_monthly_180_360_200201_201712.nc')
# ds_det.to_netcdf(save_name)


# detrend Hybrid tws by bk (2001-2019 simulation)
# load the original .nc
path_ds = os.path.join(dir_data, 'h2m/h2m3_daily_180_360_2001_2019.nc')
ds = xr.open_dataset(path_ds).sel(run=0, time=slice(date_start, date_end)).resample(time='1M').reduce(np.nanmean).sortby('lat')
ds['q'] = xr.open_dataset(path_ds).sel(run=0, time=slice(date_start, date_end)).q.resample(time='1M').reduce(np.nansum).sortby('lat')
ds['et'] = xr.open_dataset(path_ds).sel(run=0, time=slice(date_start, date_end)).et.resample(time='1M').reduce(np.nansum).sortby('lat')

# detrend
ds_orig = ds  # the original xr.dataset that you want to detrend
ds_orig = xr.where(~gra_invalid_dates_3d, ds_orig, np.nan)  # apply the same missing dates of GRACE
ds_det = detrend_dataset(ds_orig=ds_orig, idx_start=0, idx_end=ds.dims['time'])

save_name = os.path.join(dir_data, 'h2m3_det_monthly_180_360_200201_201712.nc')
# ds_det.to_netcdf(save_name)