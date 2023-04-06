'''plot figure 5'''

# =============================================================================
# load libraries and functions
# =============================================================================
import xarray as xr
import pandas as pd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from config import dir_data, dir_plot
import os

# =============================================================================
# load data
# =============================================================================

path_common_nonnan = os.path.join(dir_data, 'common_nonnan_pixels.nc')
common_nonnan = xr.open_dataset(path_common_nonnan).common_nonnan_pixels.values

# load tws
path = os.path.join(dir_data, 'cov_TWS_det_resid_scArea_det_using_20022017_simul_2001_2019.npz')
cov_res = np.load(path)
norm_tws_res_iav_sin = cov_res[cov_res.files[0]].reshape(180, 360)
norm_tws_res_iav_h2m = cov_res[cov_res.files[1]].reshape(180, 360)
norm_tws_res_msc_sin = cov_res[cov_res.files[2]].reshape(180, 360)
norm_tws_res_msc_h2m = cov_res[cov_res.files[3]].reshape(180, 360)

norm_tws_res_iav_sin = np.where(common_nonnan, norm_tws_res_iav_sin, np.nan)
norm_tws_res_iav_h2m = np.where(common_nonnan, norm_tws_res_iav_h2m, np.nan)
norm_tws_res_msc_sin = np.where(common_nonnan, norm_tws_res_msc_sin, np.nan)
norm_tws_res_msc_h2m = np.where(common_nonnan, norm_tws_res_msc_h2m, np.nan)

file_name = os.path.join(dir_data, 'rsq_change_wo_extrm_norm_res_tws_jpl06v1_non_abs_det_using_20022017_simul_2001_2019.npz')
npz = np.load(file_name)
rsq_tws_monthly = npz['rsq_tws_monthly']
list_perc = npz['trim_perc']

# =============================================================================
# calc. sum of contrib. of each trimming percentage
# =============================================================================
perc = np.arange(0, 105, 5)
arr_contrib = np.arange(2*len(perc)).reshape(2, len(perc)) * np.nan

for p in range(len(perc)):
    arr_contrib[0, p] = np.nansum(np.where(norm_tws_res_iav_sin>np.nanpercentile(norm_tws_res_iav_sin, 100-perc[p]), norm_tws_res_iav_sin, np.nan)).round(3)
    arr_contrib[1, p] = np.nansum(np.where(norm_tws_res_iav_h2m>np.nanpercentile(norm_tws_res_iav_h2m, 100-perc[p]), norm_tws_res_iav_h2m, np.nan)).round(3)

# =============================================================================
# plot
# =============================================================================

list_tws = [rsq_tws_monthly]
label_tws = ['SINDBAD', 'H2M']

fig, ax = plt.subplots(1, 1, figsize=(5, 3))

# left y: R2 change
ln1 = ax.plot(list_perc, list_tws[0][2, :], marker='.', color='limegreen', label=label_tws[0])
ln2 = ax.plot(list_perc, list_tws[0][3, :], marker='.', color='#bc15b0', label=label_tws[1])
ax.set_ylim(0.0, 1.0)
ax.set_ylabel('R$^2$ (-)')
ax.spines['top'].set_visible(False)

# right y: contribution change
ax_y2 = ax.twinx()
color_y2 = 'tab:blue'
ln3 = ax_y2.plot(perc, arr_contrib[0], marker='.', linestyle='dashed', color='limegreen', alpha=0.3, label=label_tws[0]+' IAV')
ln4 = ax_y2.plot(perc, arr_contrib[1], marker='.', linestyle='dashed', color='#bc15b0', alpha=0.3,label=label_tws[1]+' IAV')

ax_y2.set_ylim(-0.5, 2.0)
ax_y2.set_xlabel('trimming %')
ax_y2.set_ylabel('Sum of contributions (-)', color='grey')
ax_y2.spines['top'].set_visible(False)

# labels and legend...
ax.set_xlabel('Trimming %')
ax.set_xlim(0, 50)

sindbad_patch = mpatches.Patch(color='limegreen', label='SINDBAD')
h2m_patch = mpatches.Patch(color='#bc15b0', label='H2M')
ax.legend(
    handles=[sindbad_patch, h2m_patch],
    loc='lower left',
    ncol=2,
    frameon=True
    )

fig_name = os.path.join(dir_plot, 'fig05.png')
fig.savefig(
    fig_name,
    bbox_inches='tight',
    dpi=600,
    facecolor='w',
    transparent=False
)