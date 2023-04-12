'''plot figure b19'''

# =============================================================================
# load libraries and functions
# =============================================================================

import xarray as xr
import numpy as np
import pandas as pd
import statsmodels.api as sm
import pingouin as pg
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns
from config import dir_data, dir_plot
from _shared import calc_rsq, calc_nse
import os

# =============================================================================
# load data
# =============================================================================

# load pixel area and land index
area = np.load(os.path.join(dir_data, 'gridAreaInKm2_180_360.npz'))['area']

# load missing timesteps of GRACE
path_gra_invalid_dates = os.path.join(dir_data, 'GRACE_TWS_JPL_RL06Mv1CRIv01_invalid_dates_200201_201712.npy')
gra_invalid_dates = np.load(path_gra_invalid_dates)
gra_invalid_dates_3d = np.tile(gra_invalid_dates, (180, 360, 1)).transpose(2, 0, 1)

date_start = '2002-01-01'
date_end = '2017-12-31'

# GRACE JPL mascon rl06mv1
path = os.path.join(dir_data, 'GRACE_TWS_JPL_RL06Mv1CRIv01_200204_201706_scaleFactorApplied_areaWeighted_180_360_NaNs_for_2001_2019_masked_200201_201712_det.nc')
tws_iav_gra = xr.open_dataset(path)['graceTWS_det'].sel(time=slice(date_start, date_end)).values

# Sindbad
path = os.path.join(dir_data, 'sindbad_veg3_mswep_det_monthly_180_360_200201_201712.nc')  # forced with MSWEP ppt
tws_iav_sin = xr.open_dataset(path).sortby('lat', ascending=False)['wTotal_det'].values

# Hybrid bk
path = os.path.join(dir_data, 'h2m3_mswep_det_monthly_180_360_200201_201712.nc')  # forced with MSWEP ppt
tws_iav_h2m = xr.open_dataset(path).sortby('lat', ascending=False)['tws_det'].values

# =============================================================================
# exclude common NaNs
# =============================================================================

# get nan pixels
path_common_nonnan = os.path.join(dir_data, 'common_nonnan_pixels.nc')
common_nonnan = xr.open_dataset(path_common_nonnan).common_nonnan_pixels.values

# exclude union NaNs
# tws_gra = np.where(common_nonnan, tws_gra, np.nan)
tws_iav_gra = np.where(common_nonnan, tws_iav_gra, np.nan)
tws_iav_sin = np.where(common_nonnan, tws_iav_sin, np.nan)
tws_iav_h2m = np.where(common_nonnan, tws_iav_h2m, np.nan)

area = np.where(common_nonnan, area, np.nan)

# =============================================================================
# calculate annual area-weighted TWS
# =============================================================================

# annual mean
# tws_gra_ann = np.nanmean(tws_gra.reshape(-1,12,180,360), axis=1)
tws_iav_gra_ann = np.nanmean(tws_iav_gra.reshape(-1,12,180,360), axis=1)
tws_iav_sin_ann = np.nanmean(tws_iav_sin.reshape(-1,12,180,360), axis=1)
tws_iav_h2m_ann = np.nanmean(tws_iav_h2m.reshape(-1,12,180,360), axis=1)

tws_iav_gra_glo = np.array([np.nansum(_tws_mon * area)/(np.nansum(area)) for _tws_mon in tws_iav_gra])
tws_iav_sin_glo = np.array([np.nansum(_tws_mon * area)/(np.nansum(area)) for _tws_mon in tws_iav_sin])
tws_iav_h2m_glo = np.array([np.nansum(_tws_mon * area)/(np.nansum(area)) for _tws_mon in tws_iav_h2m])

# filter out invalid timesteps
tws_iav_gra_glo_validonly = tws_iav_gra_glo[~gra_invalid_dates]
tws_iav_sin_glo_validonly = tws_iav_sin_glo[~gra_invalid_dates]
tws_iav_h2m_glo_validonly = tws_iav_h2m_glo[~gra_invalid_dates]

# =============================================================================
# fit regression model
# =============================================================================

# fit a robust regression model
fit_tws_iav_sin = sm.RLM(tws_iav_gra_glo_validonly,
                         sm.add_constant(tws_iav_sin_glo_validonly),
                         M=sm.robust.norms.HuberT()).fit()
fit_tws_iav_h2m = sm.RLM(tws_iav_gra_glo_validonly,
                         sm.add_constant(tws_iav_h2m_glo_validonly),
                         M=sm.robust.norms.HuberT()).fit()

# =============================================================================
# plot
# =============================================================================

nse_sin = np.round(calc_nse(obs=tws_iav_gra_glo_validonly, est=tws_iav_sin_glo_validonly), 2)
nse_h2m = np.round(calc_nse(obs=tws_iav_gra_glo_validonly, est=tws_iav_h2m_glo_validonly), 2)
nse_sinh2m = np.round(calc_nse(obs=tws_iav_h2m_glo_validonly, est=tws_iav_sin_glo_validonly), 2)
x_years = pd.date_range(start='2002-01',end='2017-12', freq='1MS')
x_years_label = np.arange(2002, 2018, 2)
x_data = [x_years[~gra_invalid_dates], tws_iav_gra_glo_validonly]
y_data = [
    [
        tws_iav_gra_glo_validonly,
        tws_iav_sin_glo_validonly,
        tws_iav_h2m_glo_validonly
    ],  # ax0
    [
        tws_iav_sin_glo_validonly,
        tws_iav_h2m_glo_validonly
    ],  # ax1
    [
        tws_iav_sin_glo_validonly - tws_iav_gra_glo_validonly,
        tws_iav_h2m_glo_validonly - tws_iav_gra_glo_validonly
    ]
]
fitted = [
    [],
    [fit_tws_iav_sin, fit_tws_iav_h2m]
]
corr = [
    [calc_rsq(y_data[0][0], y_data[0][1]),
    calc_rsq(y_data[0][0], y_data[0][2]),
    calc_rsq(y_data[0][1], y_data[0][2])
    ]
]
subplot_idx = ['a', 'c', 'b']
anno_text = [
    ['R$^2$(GRACE, SINDBAD): ' + str(corr[0][0]),
    'R$^2$(GRACE, H2M): ' + str(corr[0][1]),
    'R$^2$(SINDBAD, H2M): ' + str(corr[0][2])],
    ['SIN: y={reg_slp}x{sign_int}{reg_int}'.format(reg_slp='{:.2f}'.format(fit_tws_iav_sin.params[1].round(2)), sign_int='+' if fit_tws_iav_sin.params[0].round(2)>=0 else '', reg_int='{:.2f}'.format(fit_tws_iav_sin.params[0].round(2))),
    'H2M: y={reg_slp}x{sign_int}{reg_int}'.format(reg_slp='{:.2f}'.format(fit_tws_iav_h2m.params[1].round(2)), sign_int='+' if fit_tws_iav_h2m.params[0].round(2)>=0 else '', reg_int='{:.2f}'.format(fit_tws_iav_h2m.params[0].round(2)))]
]
anno_nse = [
    'NSE(GRACE, SINDBAD): ' + str(nse_sin),
    'NSE(GRACE, H2M): ' + str(nse_h2m),
    'NSE(SINDBAD, H2M): ' + str(nse_sinh2m)
]
y_lim = [
    [-30, 30],
    [-30, 30]
]
x_lim = [
    [-30, 30],
    [-30, 30]
]
x_labels = ['Years', 'TWS IAV: Model (mm) ', 'TWS IAV error (mm)']
y_labels = ['TWS IAV (mm)', 'TWS IAV: GRACE (mm)', 'Relative frequency (-)']

# set figure
plt.rcParams.update(plt.rcParamsDefault)

color = ['black', 'limegreen', '#bc15b0']  # #bc15b0 for H2M
linestyle = ["--", "-", "-"]

nrow = 2
ncol = 2
fig = plt.figure(figsize=(4.8+2.7+1.0, 2.7*2))
gs = fig.add_gridspec(nrows=nrow, ncols=ncol, figure=fig, width_ratios=[1, 1])


# time series
ax = fig.add_subplot(gs[0, :])
i = 0
ax.plot(x_data[i], y_data[i][0], color=color[0], linestyle=linestyle[0], label='GRACE')
ax.plot(x_data[i], y_data[i][1], color=color[1], linestyle=linestyle[1], label='SINDBAD', alpha=0.75)
ax.plot(x_data[i], y_data[i][2], color=color[2], linestyle=linestyle[2], label='H2M', alpha=0.75)
ax.axhline(y=0, xmin=0, xmax=1, color='black', alpha=0.4)  # y=0
ax.annotate(anno_text[i][0], xy=(0.07, 0.15), xycoords='axes fraction')
ax.annotate(anno_text[i][1], xy=(0.07, 0.09), xycoords='axes fraction')
ax.annotate(anno_text[i][2], xy=(0.07, 0.03), xycoords='axes fraction')

ax.annotate(subplot_idx[i], xy=(0.02, 0.92), xycoords='axes fraction', fontsize=15)
ax.set_ylim(y_lim[i])
ax.set_ylabel(y_labels[i], fontsize=12)
ax.set_xlabel(x_labels[i], fontsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.legend(loc=(0.07, 0.85), ncol=3)

# scatter
ax = fig.add_subplot(gs[1, 1])
i = 1
ax.scatter(y_data[i][0], x_data[i], color=color[1], alpha=0.4, label='SINDBAD')
ax.scatter(y_data[i][1], x_data[i], color=color[2], alpha=0.4, label='H2M')
ax.plot(x_data[i], fitted[i][0].predict(sm.add_constant(x_data[i])), '-', c='limegreen')
ax.plot(x_data[i], fitted[i][1].predict(sm.add_constant(x_data[i])), '-', c='#bc15b0')
ax.plot([0,1],[0,1], '--', transform=ax.transAxes, c='k', alpha=0.6)

ax.annotate(anno_text[i][0], xy=(0.6, 0.10), xycoords='axes fraction')
ax.annotate(anno_text[i][1], xy=(0.6, 0.04), xycoords='axes fraction')
ax.annotate(subplot_idx[i], xy=(0.05, 0.92), xycoords='axes fraction', fontsize=15)
ax.set_xlim(x_lim[i])
ax.set_ylim(y_lim[i])
ax.set_ylabel(y_labels[i], fontsize=12)
ax.set_xlabel(x_labels[i], fontsize=12)
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# histogram
ax = fig.add_subplot(gs[1, 0])
i = 2
shared_bins = np.histogram_bin_edges(y_data[i][0], bins=20)
sns.histplot(x=y_data[i][0], stat='probability', kde=True, bins=shared_bins, ax=ax, color='limegreen', alpha=0.3, label='SINDBAD')
sns.histplot(x=y_data[i][1], stat='probability', kde=True, bins=shared_bins, ax=ax, color='#bc15b0', alpha=0.3, label='H2M')
ax.axvline(x=0, linestyle='dashed', color='black')
ax.annotate(subplot_idx[i], xy=(0.05, 0.92), xycoords='axes fraction', fontsize=15)
txt_mean_sin = '0.0'
txt_std_sin = str(np.nanstd(y_data[i][0]).round(2))
txt_mean_h2m = '0.0'
txt_std_h2m = str(np.nanstd(y_data[i][1]).round(2))
txt_sin = 'SINDBAD: '+txt_mean_sin+r'$\pm$'+txt_std_sin
txt_h2m = 'H2M: '+txt_mean_h2m+r'$\pm$'+txt_std_h2m
ax.annotate(txt_sin, xy=(0.15, 0.96), xycoords='axes fraction')
ax.annotate(txt_h2m, xy=(0.15, 0.9), xycoords='axes fraction')
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)
ax.set_xlabel(x_labels[i], fontsize=12)
ax.set_ylabel(y_labels[i], fontsize=12)


fig.subplots_adjust(top=0.98, bottom=0.02, left=0.02, right=0.98,
                    wspace=0.3, hspace=0.25)

save_name = os.path.join(dir_plot, 'figb19.png')

fig.savefig(
    save_name,
    dpi=600,
    bbox_inches='tight',
    facecolor='w',
    transparent=False
)