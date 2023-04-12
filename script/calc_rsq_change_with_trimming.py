'''
- calculate model performance (rsq) changes with trimming
- data was used for the figure 5, ...
'''

# =============================================================================
# load libraries and functions
# =============================================================================
import xarray as xr
import numpy as np
import pingouin as pg
import copy
from config import dir_data, dir_plot
from _shared import running_mean, calc_rsq
import os

def get_trimmed_pix_pos(norm_cov, trim_perc):
    # get threshold based on raw values (no absolute values)
    threshold = np.nanpercentile(norm_cov, 100-trim_perc)
    is_over_threshold = norm_cov >= threshold
    
    return is_over_threshold

def trim_extrm_cov(dat, is_over_threshold, replace=False, dat_replace=None):
    # is_over_threshold: results from get_trimmed_pix_abs() or get_trimmed_pix_pos() 
    _dat = copy.deepcopy(dat)

    if replace==False:
        _dat = np.where(is_over_threshold, np.nan, _dat)
    if replace==True:
        _dat = np.where(is_over_threshold, dat_replace, _dat)
    
    return _dat

# =============================================================================
# load data
# =============================================================================

# load pixel area and land index
area = np.load(os.path.join(dir_data, 'gridAreaInKm2_180_360.npz'))['area']

# data of the 2001-2019 simulation
date_start = '2002-01-01'
date_end = '2017-12-31'
nan_months_start = 3  # 2002-01 ~ 2002-03 == NaN
nan_months_end = 6  # 2017-07 ~ 2017~12 == NaN

# GRACE JPL mascon rl06mv1
path = os.path.join(dir_data, 'GRACE_TWS_JPL_RL06Mv1CRIv01_200204_201706_scaleFactorApplied_areaWeighted_180_360_NaNs_for_2001_2019_masked_200201_201712_det.nc')
tws_iav_gra = xr.open_dataset(path)['graceTWS_det'].sel(time=slice(date_start, date_end)).values
tws_msc_gra = xr.open_dataset(path)['graceTWS_msc'].sel(time=slice(date_start, date_end)).values

# Sindbad tt
path = os.path.join(dir_data, 'sindbad_veg3_det_monthly_180_360_200201_201712.nc')
tws_iav_sin = xr.open_dataset(path).sortby('lat', ascending=False)['wTotal_det'].sel(time=slice(date_start, date_end)).values
tws_msc_sin = xr.open_dataset(path).sortby('lat', ascending=False)['wTotal_msc'].sel(time=slice(date_start, date_end)).values

# Hybrid bk
path = os.path.join(dir_data, 'h2m3_det_monthly_180_360_200201_201712.nc')
tws_iav_h2m = xr.open_dataset(path).sortby('lat', ascending=False)['tws_det'].sel(time=slice(date_start, date_end)).values
tws_msc_h2m = xr.open_dataset(path).sortby('lat', ascending=False)['tws_msc'].sel(time=slice(date_start, date_end)).values

# load norm
path = os.path.join(dir_data, 'cov_TWS_det_resid_scArea_det_using_20022017_simul_2001_2019.npz')
cov_res = np.load(path)
norm_tws_res_iav_sin = cov_res[cov_res.files[0]].reshape(180, 360)
norm_tws_res_iav_h2m = cov_res[cov_res.files[1]].reshape(180, 360)
norm_tws_res_msc_sin = cov_res[cov_res.files[2]].reshape(180, 360)
norm_tws_res_msc_h2m = cov_res[cov_res.files[3]].reshape(180, 360)

# mask
path_common_nonnan = os.path.join(dir_data, 'common_nonnan_pixels.nc')
common_nonnan = xr.open_dataset(path_common_nonnan).common_nonnan_pixels.values

tws_iav_gra = np.where(common_nonnan, tws_iav_gra, np.nan)
tws_iav_sin = np.where(common_nonnan, tws_iav_sin, np.nan)
tws_iav_h2m = np.where(common_nonnan, tws_iav_h2m, np.nan)
tws_msc_gra = np.where(common_nonnan, tws_msc_gra, np.nan)
tws_msc_sin = np.where(common_nonnan, tws_msc_sin, np.nan)
tws_msc_h2m = np.where(common_nonnan, tws_msc_h2m, np.nan)

norm_tws_res_iav_sin = np.where(common_nonnan, norm_tws_res_iav_sin, np.nan)
norm_tws_res_iav_h2m = np.where(common_nonnan, norm_tws_res_iav_h2m, np.nan)
norm_tws_res_msc_sin = np.where(common_nonnan, norm_tws_res_msc_sin, np.nan)
norm_tws_res_msc_h2m = np.where(common_nonnan, norm_tws_res_msc_h2m, np.nan)

# =============================================================================
# calculate rsq change and the sum of contributions with increaing trimming precentage
# =============================================================================

list_tws_obs = [
    tws_msc_gra,
    tws_msc_gra,
    tws_iav_gra,
    tws_iav_gra
]

list_tws_est = [
    tws_msc_sin,
    tws_msc_h2m,
    tws_iav_sin,
    tws_iav_h2m
]

list_norm_cov = [
    norm_tws_res_msc_sin,
    norm_tws_res_msc_h2m,
    norm_tws_res_iav_sin,
    norm_tws_res_iav_h2m
]

list_norm_cov_mask = [
    (norm_tws_res_msc_sin, norm_tws_res_msc_h2m),
    (norm_tws_res_msc_sin, norm_tws_res_msc_h2m),
    (norm_tws_res_iav_sin, norm_tws_res_iav_h2m),
    (norm_tws_res_iav_sin, norm_tws_res_iav_h2m)
]

list_perc = [0.0, 0.1, 0.3, 0.5, 1, 2, 3, 5, 10, 15] + list(np.arange(20, 105, 5))
rsq_monthly = np.empty([len(list_tws_est)]+[len(list_perc)]) * np.nan  # (cov x perc)
rsq_monthly_running = np.empty([len(list_tws_est)]+[len(list_perc)]) * np.nan  # (cov x perc)
rsq_yearly = np.empty([len(list_tws_est)]+[len(list_perc)]) * np.nan  # (cov x perc)
trimmed_pix = np.empty([len(list_norm_cov)]+[len(list_perc)]+list(list_norm_cov[0].shape)) * np.nan  # (cov x perc)
global_tws_obs_monthly = np.empty([len(list_tws_est)]+[len(list_perc)]+[list_tws_obs[0].shape[0]]) * np.nan  # (cov x perc)
global_tws_est_monthly = np.empty([len(list_tws_est)]+[len(list_perc)]+[list_tws_est[0].shape[0]]) * np.nan  # (cov x perc)
global_tws_obs_monthly_running = np.empty([len(list_tws_est)]+[len(list_perc)]+[list_tws_obs[0].shape[0]]) * np.nan  # (cov x perc)
global_tws_est_monthly_running = np.empty([len(list_tws_est)]+[len(list_perc)]+[list_tws_est[0].shape[0]]) * np.nan  # (cov x perc)
global_tws_obs_yearly = np.empty([len(list_tws_est)]+[len(list_perc)]+[int(list_tws_obs[0].shape[0]/12)]) * np.nan  # (cov x perc)
global_tws_est_yearly = np.empty([len(list_tws_est)]+[len(list_perc)]+[int(list_tws_est[0].shape[0]/12)]) * np.nan  # (cov x perc)

for mod in range(len(list_tws_est)):
    print('{mod} / {len}'.format(mod=mod, len=len(list_tws_est)-1))
    for p in range(len(list_perc)):
        # trim n% extreme cov. of TWS residuals
        is_over_threshold_save = get_trimmed_pix_pos(norm_cov=list_norm_cov[mod], 
                                                        trim_perc=list_perc[p])
        is_over_threshold_sin = get_trimmed_pix_pos(norm_cov=list_norm_cov_mask[mod][0], 
                                                        trim_perc=list_perc[p])
        is_over_threshold_h2m = get_trimmed_pix_pos(norm_cov=list_norm_cov_mask[mod][1], 
                                                        trim_perc=list_perc[p])
        is_over_threshold = is_over_threshold_sin * is_over_threshold_h2m  # common mask from SINDBAD and H2M

        tws_obs_trim = trim_extrm_cov(
            dat=list_tws_obs[mod],
            is_over_threshold=is_over_threshold,
            replace=False,
            dat_replace=list_tws_obs[mod]
        )
        tws_est_trim = trim_extrm_cov(
            dat=list_tws_est[mod],
            is_over_threshold=is_over_threshold,
            replace=False,
            dat_replace=list_tws_obs[mod]
        )
        
        # 0% trimming --> change nothing
        # trim_extrm_cov() will otherwise change the largest values...
        if p == 0:
            tws_obs_trim = list_tws_obs[mod]
            tws_est_trim = list_tws_est[mod]
            
        # calculate global time series (monthly, yearly)
        tws_obs_trim_ann = np.nanmean(tws_obs_trim.reshape(-1, 12, 180, 360), axis=1)
        tws_est_trim_ann = np.nanmean(tws_est_trim.reshape(-1, 12, 180, 360), axis=1)            
        
        tws_obs_trim_glo = np.array([np.nansum(_ts * area)/(np.nansum(area)) for _ts in tws_obs_trim])
        tws_est_trim_glo = np.array([np.nansum(_ts * area)/(np.nansum(area)) for _ts in tws_est_trim])
        tws_obs_trim_ann_glo = np.array([np.nansum(_ts * area)/(np.nansum(area)) for _ts in tws_obs_trim_ann])
        tws_est_trim_ann_glo = np.array([np.nansum(_ts * area)/(np.nansum(area)) for _ts in tws_est_trim_ann])

        tws_obs_trim_glo_running = running_mean(tws_obs_trim_glo, 12)
        tws_est_trim_glo_running = running_mean(tws_est_trim_glo, 12)

        # calculate r2
        rsq_monthly[mod, p] = calc_rsq(tws_obs_trim_glo, tws_est_trim_glo)
        rsq_monthly_running[mod, p] = calc_rsq(tws_obs_trim_glo_running, tws_est_trim_glo_running)
        rsq_yearly[mod, p] = calc_rsq(tws_obs_trim_ann_glo, tws_est_trim_ann_glo)
        
        # save data: trimmed pixels, time series
        trimmed_pix[mod, p, :, :] = is_over_threshold_save
        global_tws_obs_monthly[mod, p, :] = tws_obs_trim_glo
        global_tws_est_monthly[mod, p, :] = tws_est_trim_glo
        global_tws_obs_monthly_running[mod, p, :] = tws_obs_trim_glo_running
        global_tws_est_monthly_running[mod, p, :] = tws_est_trim_glo_running
        global_tws_obs_yearly[mod, p, :] = tws_obs_trim_ann_glo
        global_tws_est_yearly[mod, p, :] = tws_est_trim_ann_glo

out_name = os.path.join(dir_data, 'rsq_change_wo_extrm_norm_res_tws_jpl06v1_non_abs_det_using_20022017_simul_2001_2019.npz')
# np.savez(out_name, 
#          trim_perc=np.array(list_perc),
#          trimmed_pix=trimmed_pix,
#          rsq_tws_monthly=rsq_monthly,
#          rsq_tws_monthly_running=rsq_monthly_running,
#          rsq_tws_yearly=rsq_yearly,
#          global_tws_obs_monthly=global_tws_obs_monthly,
#          global_tws_est_monthly=global_tws_est_monthly,
#          global_tws_obs_monthly_running=global_tws_obs_monthly_running,
#          global_tws_est_monthly_running=global_tws_est_monthly_running,
#          global_tws_obs_yearly=global_tws_obs_yearly,
#          global_tws_est_yearly=global_tws_est_yearly
# )