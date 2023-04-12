'''plot figure b05'''

# =============================================================================
# load libraries and functions
# =============================================================================
import xarray as xr
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import seaborn as sns
from config import dir_data, dir_plot
import os

def get_hotspot_pixels(arr_2d, perc):
    val_perc = np.nanpercentile(arr_2d, perc)
    cond_hot = np.logical_and(arr_2d>val_perc, ~np.isnan(arr_2d)) 
    hotspots = np.where(cond_hot, -999, arr_2d) == -999
    return hotspots

# =============================================================================
# load data
# =============================================================================

# load pixel area and land index
area = np.load(os.path.join(dir_data, 'gridAreaInKm2_180_360.npz'))['area']

date_start = '2002-01-01'
date_end = '2017-12-31'

# GRACE JPL mascon rl06mv1
path = os.path.join(dir_data, 'GRCTellus.JPL.200204_201706.GLO.RL06M.MSCNv01CRIv01.scaleFactorApplied.areaWeighted.192_180_360.with_NaNs_for_2001_2019.masked.nc')
tws_gra_tws = xr.open_dataset(path).sel(time=slice(date_start, date_end)).graceTWS.values
tws_gra_unc = xr.open_dataset(path).sel(time=slice(date_start, date_end)).graceTWS_unc.values

# load tws iav res
path = os.path.join(dir_data, 'cov_TWS_det_resid_scArea_det_using_20022017_simul_2001_2019.npz')  # forcing with gpcp1dd ppt
cov_res = np.load(path)
norm_tws_res_iav_sin = cov_res[cov_res.files[0]].reshape(180, 360)
norm_tws_res_iav_h2m = cov_res[cov_res.files[1]].reshape(180, 360)

# get nan pixels
path_common_nonnan = os.path.join(dir_data, 'common_nonnan_pixels.nc')
common_nonnan = xr.open_dataset(path_common_nonnan).common_nonnan_pixels.values

# =============================================================================
# mask nans and get hotspots
# =============================================================================

# exclude union NaNs
tws_gra_tws = np.where(common_nonnan, tws_gra_tws, np.nan)
tws_gra_unc = np.where(common_nonnan, tws_gra_unc, np.nan)
norm_tws_res_iav_sin = np.where(common_nonnan, norm_tws_res_iav_sin, np.nan)
norm_tws_res_iav_h2m = np.where(common_nonnan, norm_tws_res_iav_h2m, np.nan)

# get hotspots
hotspots_res_sin = get_hotspot_pixels(arr_2d=norm_tws_res_iav_sin, perc=90)
hotspots_res_h2m = get_hotspot_pixels(arr_2d=norm_tws_res_iav_h2m, perc=90)
hotspots_res_common = hotspots_res_sin * hotspots_res_h2m

# =============================================================================
# plot
# =============================================================================

tws_unc_over_amp = np.nanmean(tws_gra_unc, axis=0) / (np.nanmax(tws_gra_tws, axis=0) - np.nanmin(tws_gra_tws, axis=0))
tws_unc_over_amp = np.where(tws_unc_over_amp>1, 1, tws_unc_over_amp)
list_data=[
    tws_unc_over_amp
]

list_title = [
    'GRACE: mean unc. / TWS amp. (-)'
]
list_range = [
    (0.0, 0.2)
]
ndata = len(list_data)

nrow = ndata
ncol = 2
mar_t = 0.01
mar_b = 0.02
mar_l = 0.02
mar_r = 0.01
ht_ratio = np.ones((nrow))
wt_ratio = np.array([2.5, 1])  # np.ones((ncol))
hspace = np.ones((nrow-1)) * 0.015
wspace = np.ones((ncol-1)) * 0.0850
ht = (1 - mar_t - mar_b) / nrow * ht_ratio
ht = ht / np.sum(ht) * ((1 - mar_t - mar_b - np.sum(hspace)) / 1)
wt = (1 - mar_l - mar_r) / ncol * wt_ratio
wt = wt / np.sum(wt) * ((1 - mar_l - mar_r - np.sum(wspace)) / 1)

fig = plt.figure(figsize=(6.9*0.75*ncol, 2.7*0.75*nrow))

# map
r=0
c=0
idx_data = r
ax_box_x = mar_l + np.sum(wt[:c]) + np.sum(wspace[:c])
ax_box_y = 1 - mar_t - np.sum(ht[:(r+1)]) - np.sum(hspace[:r])
ax_box = [ax_box_x, ax_box_y, wt[c], ht[r]]
ax = plt.axes(
    ax_box,
    projection=ccrs.Robinson(central_longitude=0),
    frameon=False)
_data = np.ma.masked_equal(list_data[idx_data][0:150, :], -9999.)
ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
ax.coastlines(linewidth=0.4, color='grey')
im1 = ax.imshow(
    _data,
    cmap='viridis_r',
    vmin=list_range[r][0],
    vmax=list_range[r][1],
    aspect='auto',
    extent=[-180, 180, -60, 90],
    transform=ccrs.PlateCarree())
plt.colorbar(im1, ax=ax, pad=0.027, extend='both').set_label(label=list_title[r], size=9)

ax_box_hist = [ax_box_x+wt[c]*0.1, ax_box_y+ht[r]*0.35, wt[c]*0.12, ht[r]*0.18]
ax_hist = plt.axes(
    ax_box_hist,
    frameon=False)
sns.histplot(_data.ravel(), stat='probability', bins=100, kde=True, color='grey', ax=ax_hist)

if r in [1]:
    ax_hist.set_xlim(0, 1)
ax_hist.set_ylim(0.0, 0.1)

ax_hist.tick_params(axis='both', labelsize=5)
ax_hist.set_ylabel('Probability', fontsize=5)
ax_hist.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
ax_hist.yaxis.major.formatter._useMathText = True
ax_hist.yaxis.get_offset_text().set_fontsize(5)
ax_hist.xaxis.get_offset_text().set_fontsize(5)

# scatter
r=0
c=1
ax_box_x = mar_l + np.sum(wt[:c]) + np.sum(wspace[:c])
ax_box_y = 1 - mar_t - np.sum(ht[:(r+1)]) - np.sum(hspace[:r])
ax_box = [ax_box_x, ax_box_y, wt[c], ht[r]]
ax = plt.axes(
    ax_box,
    frameon=True)
ax.scatter(list_data[r], norm_tws_res_iav_sin, color='limegreen', alpha=0.2, label='SINBAD')
ax.scatter(list_data[r], norm_tws_res_iav_h2m, color='#bc15b0', alpha=0.2, label='H2M')
ax.legend(frameon=False)
ax.set_xlabel(list_title[r])
ax.set_ylabel('Contribution (-)')
ax.annotate(
    'SINDBAD:''$\it{r}$' + '=' + str(pg.corr(list_data[r].ravel(), norm_tws_res_iav_sin.ravel())['r'][0].round(2))  + ', ' + '$\it{p}$' + '<0.001',
    xy=(0.2, 0.11), xycoords='axes fraction')
ax.annotate(
    'H2M:''$\it{r}$' + '=' + str(pg.corr(list_data[r].ravel(), norm_tws_res_iav_h2m.ravel())['r'][0].round(2)) + ', ' + '$\it{p}$' + '<0.05',
    xy=(0.2, 0.03), xycoords='axes fraction')

ax.spines.top.set_visible(False)
ax.spines.right.set_visible(False)

save_name = os.path.join(dir_plot, 'figb05.png')
fig.savefig(
    save_name,
    dpi=300,
    bbox_inches='tight',
    facecolor='w',
    transparent=False
)