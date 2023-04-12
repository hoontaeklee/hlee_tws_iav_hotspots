'''plot figure b04'''

# =============================================================================
# load libraries and functions
# =============================================================================

import xarray as xr
import numpy as np
import cartopy.crs as ccrs
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colorbar as mcbar
import matplotlib.colors as mcolors
import matplotlib.cm as mcm
import _shared_plot as ptool
from _shared import _fit_least_square
import scipy.stats as scst
from config import dir_data, dir_plot
import os

def draw_map_ax(a_x, one_deg_data, title='', vmin=-0.001, vmax=0.001, cm='coolwarm'):
    extent = [-179.5 , 179.5, -89.5 , 89.5]
    _cm=mpl.cm.get_cmap(cm)
    ax = plt.axes(a_x,
                    projection=ccrs.Robinson(central_longitude=0),
                    frameon=False)  #,sharex=right,sharey=all)
    ax=_fix_map(ax)
    im=plt.imshow(np.ma.masked_equal(one_deg_data[0:150, :], -9999.),
                interpolation='none',
                origin='upper', vmin=vmin, vmax=vmax, cmap=_cm,
                transform=ccrs.PlateCarree(),
                extent=[-180, 180, -60, 90])
    ax.set_title(title, fontsize=12)
    return ax

def _fix_map(axis_obj):
    """
    Beautify map object.

    Clean boundaries, coast lines, and removes the outline box/circle.
    """
    # axis_obj.set_global()
    axis_obj.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
    axis_obj.coastlines(linewidth=0.4, color='grey')
    # plt.gca().outline_patch.set_visible(False)
    return axis_obj

# =============================================================================
# load data
# =============================================================================

# nonnan mask
path_common_nonnan = os.path.join(dir_data, 'common_nonnan_pixels.nc')
common_nonnan = xr.open_dataset(path_common_nonnan).common_nonnan_pixels.values

# ano_std and iav_std 
date_start = '2002-01-01'
date_end = '2017-12-31'

path = os.path.join(dir_data, 'GRACE_TWS_JPL_RL06Mv1CRIv01_200204_201706_scaleFactorApplied_areaWeighted_180_360_NaNs_for_2001_2019_masked_200201_201712_det.nc')
tws_iav_std_gra = xr.open_dataset(path)['graceTWS_det_std'].values

path = os.path.join(dir_data, 'sindbad_veg3_det_monthly_180_360_200201_201712.nc')  # forced with GPCP1dd ppt
tws_iav_std_sin = xr.open_dataset(path)['wTotal_det_std'].sortby('lat', ascending=False).values

path = os.path.join(dir_data, 'h2m3_det_monthly_180_360_200201_201712.nc')  # forced with GPCP1dd ppt
tws_iav_std_h2m = xr.open_dataset(path)['tws_det_std'].sortby('lat', ascending=False).values

# =============================================================================
# exclude common NaNs
# =============================================================================
tws_iav_std_gra = np.where(common_nonnan, tws_iav_std_gra, np.nan)
tws_iav_std_sin = np.where(common_nonnan, tws_iav_std_sin, np.nan)
tws_iav_std_h2m = np.where(common_nonnan, tws_iav_std_h2m, np.nan)

# =============================================================================
# plot
# =============================================================================
# diagonal: map of data
# upper diagonal: difference map
# lower diagonal: 2d-histogram in this case

list_data=[
    tws_iav_std_gra,
    tws_iav_std_sin,
    tws_iav_std_h2m
]

list_title = [
    'GRACE',
    'SINDBAD',
    'H2M'
]

# set figure

#FIGURES SETTINGS AND PARAMETER of the figure
nmodels = len(list_data)
x0 = 0.02
y0 = 1.0

wp = 1. / nmodels
hp = wp
xsp = 0.0
aspect_data = 1.65 * 1680. / 4320.
ysp = -0.06
xsp_sca = wp / 3 * (aspect_data)
ysp_sca = hp / 3 * (aspect_data)

wcolo = 0.25
hcolo = 0.085 * hp * nmodels / 7.
cb_off_x = wcolo
cb_off_y = 0.06158
ax_fs = 8  # axis fontsize
alphabets = [chr(i) for i in range(ord('a'),ord('z')+1)][:nmodels**2]

fig = plt.figure(figsize=(11*0.5*1.5, 11*0.5))
for r in range(nmodels):
    for c in range(nmodels):
        # diagonal: maps
        if r == c:
            axl=[x0 + c * wp + c * xsp, y0 -
                    (r * hp + r * ysp), wp, hp]
            if r == 0:
                title = list_title[c]
            else:
                title = ''
            vmin=10
            vmax=90
            ax = draw_map_ax(
                axl,
                one_deg_data=list_data[r],
                title=title,
                vmin=vmin,
                vmax=vmax,
                cm='cividis_r'
                )
        # upper diagonal: maps of difference        
        if c > r:
            axl=[
                    x0 + c * wp + c * xsp, y0 -
                    (r * hp + r * ysp), wp, hp
                ]
            if r == 0:
                title = list_title[c]
            else:
                title = ''
            vmin=-30
            vmax=30
            ax = draw_map_ax(
                axl,
                one_deg_data=list_data[c]-list_data[r],
                title=title,
                vmin=vmin,
                vmax=vmax,
                cm='coolwarm'
                )
        # lower diagonal: 2d-histogram        
        if r > c:
            # draw 2d-histogram
            ax=plt.axes([
                    x0 + c * wp + c * xsp + xsp_sca,
                    y0 - (r * hp + r * ysp) + ysp_sca,
                    0.23, 0.23
                ])
            # ax = fig.add_subplot(gs[r, c])
            cmapBig = mcm.get_cmap('magma_r', 512)
            cmap_sca = mcolors.ListedColormap(cmapBig(np.linspace(0.1, 1.0, 256)))

            lim_dn = np.nanpercentile([list_data[c], list_data[r]], 1).round(4)
            lim_up = np.nanpercentile([list_data[c], list_data[r]], 99).round(4)
            data_x_valid = list_data[c][np.logical_and(~np.isnan(list_data[c]), ~np.isnan(list_data[r]))]
            data_y_valid = list_data[r][np.logical_and(~np.isnan(list_data[c]), ~np.isnan(list_data[r]))] 
            ax.hist2d(data_x_valid, data_y_valid, cmin=1, cmap=cmap_sca, bins=300)
            ax.plot(np.linspace(lim_dn, lim_up+0.0001, num=100), np.linspace(lim_dn, lim_up+0.0001, num=100), 'k--', linewidth=0.52)
            ax.set_xlim(lim_dn, lim_up)
            ax.set_ylim(lim_dn, lim_up)

            # add regression line
            _x_fit_org = list_data[c].reshape(-1)
            _y_fit_org = list_data[r].reshape(-1)
            _mask_x = np.ma.masked_invalid(_x_fit_org).mask
            _mask_y = np.ma.masked_invalid(_y_fit_org).mask
            _mask_common = np.logical_or(_mask_x, _mask_y)
            _x_fit = np.ma.masked_where(_mask_common, _x_fit_org).compressed()
            _y_fit = np.ma.masked_where(_mask_common, _y_fit_org).compressed()
            fit_dat = _fit_least_square(_x_fit,
                                    _y_fit,
                                    _logY=False,
                                    method='quad',
                                    _intercept=True,
                                    _bounds=(-np.inf, np.inf))
            ax.plot(fit_dat['pred']['x'],
                    fit_dat['pred']['y'],
                    c='red',
                    ls='-',
                    lw=0.95,
                    marker=None)
            ax.ticklabel_format(style='sci', scilimits=(0, 0))
            ax.xaxis.major.formatter._useMathText = True
            ax.yaxis.major.formatter._useMathText = True
            ax.xaxis.offsetText.set(size=8)
            ax.yaxis.offsetText.set(size=8)

            # add statistics: corr, regression
            p_r, p_p = scst.pearsonr(data_x_valid, data_y_valid)
            rho, p = scst.spearmanr(data_x_valid, data_y_valid)
            tit_str = "$r$=" + str(round(p_r, 2)) + ", $\\rho$=" + str(
                round(rho, 2))
            reg_int_sign = '+' if fit_dat['coef'][1] >= 0 else '-'
            reg_str = 'y='+str(round(fit_dat['coef'][0], 2))+reg_int_sign+str(abs(round(fit_dat['coef'][1], 3)))
            ax.annotate(tit_str, xy=(0.55, 0.27), xycoords='axes fraction', fontsize=ax_fs * 0.953, ma='left', va='top')
            ax.annotate(reg_str, xy=(0.55, 0.13), xycoords='axes fraction', fontsize=ax_fs * 0.953, color='red', ma='left', va='top')

            # remove redundant ticks and labels
            if c != 0 and r != nmodels - 1:
                ptool.ax_clr(axfs=ax_fs)
                ptool.rotate_labels(which_ax='x', axfs=ax_fs, rot=90)
                if c >= r:
                    ptool.rem_ticklabels()
            elif c == 0 and r != nmodels - 1:
                ptool.ax_clrX(axfs=ax_fs)
                ptool.rotate_labels(which_ax='x', axfs=ax_fs, rot=90)
                if c >= r:
                    ptool.rem_ticklabels()
            elif r == nmodels - 1 and c != 0:
                ptool.ax_clrY(axfs=ax_fs)
                ptool.rotate_labels(which_ax='x', axfs=ax_fs, rot=90)
                if c >= r:
                    ptool.rem_ticklabels()
            if c == 0 and r == nmodels - 1:
                ptool.ax_orig(axfs=ax_fs)
                ptool.rotate_labels(which_ax='x', axfs=ax_fs, rot=90)
                if c >= r:
                    ptool.rem_ticklabels()
# colorbar for tws
cax = plt.axes([0.1, 0.40, 0.3, 0.01])
vmin=10
vmax=90
cb = mcbar.ColorbarBase(
    cax,
    cmap=mcm.get_cmap('cividis_r'),
    norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
    orientation='horizontal',
    extend='both'
)
cb.ax.set_xlabel('TWS IAV std. (mm)', loc='center')

# colorbar for differences
cax = plt.axes([0.7, 0.40, 0.3, 0.01])
vmin=-30
vmax=30
cb = mcbar.ColorbarBase(
    cax,
    cmap=mcm.get_cmap('coolwarm'),
    norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
    orientation='horizontal',
    extend='both'
)
cb.ax.set_xlabel('Difference (Column - Row)', loc='center')

# add row title on the right side
for c in range(nmodels):
    plt.annotate(list_title[c], xy=(1.03, 0.93-c*0.3), xycoords='figure fraction', rotation=90, fontsize=12, va='center', ha='center')

fig.tight_layout()

save_name = os.path.join(dir_plot, 'figb04.png')
fig.savefig(
    save_name,
    dpi=300,
    bbox_inches='tight',
    facecolor='w',
    transparent=False
)