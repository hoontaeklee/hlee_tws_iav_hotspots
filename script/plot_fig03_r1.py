'''plot figure 3'''

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

# tws iav norm. cov.
path = os.path.join(dir_data, 'cov_TWS_det_scArea_det_using_20022017_simul_2001_2019.npz')
cov = np.load(path)

norm_tws_iav_gra = cov[cov.files[0]].reshape(180, 360)
norm_tws_iav_sin = cov[cov.files[1]].reshape(180, 360)
norm_tws_iav_h2m = cov[cov.files[2]].reshape(180, 360)

# =============================================================================
# exclude common NaNs
# =============================================================================

norm_tws_iav_gra = np.where(common_nonnan, norm_tws_iav_gra, np.nan)
norm_tws_iav_sin = np.where(common_nonnan, norm_tws_iav_sin, np.nan)
norm_tws_iav_h2m = np.where(common_nonnan, norm_tws_iav_h2m, np.nan)

# =============================================================================
# plot
# =============================================================================
# diagonal: map of data
# upper diagonal: difference map
# lower diagonal: 2d-histogram in this case

list_data=[
    norm_tws_iav_gra,
    norm_tws_iav_sin,
    norm_tws_iav_h2m
]

list_title = [
    'GRACE',
    'SINDBAD',
    'H2M'
]

# set figure
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
vmin=-0.001
vmax=0.001

cmapBig = mcm.get_cmap('magma_r', 512)
cmap_sca = mcolors.ListedColormap(cmapBig(np.linspace(0.1, 1.0, 256)))

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
            ax = draw_map_ax(
                axl,
                one_deg_data=list_data[r],
                title=title,
                vmin=vmin,
                vmax=vmax,
                cm='coolwarm'
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
            ax = draw_map_ax(
                axl,
                one_deg_data=list_data[c]-list_data[r],
                title=title,
                vmin=vmin,
                vmax=vmax
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
            lim_dn = np.nanpercentile([list_data[c], list_data[r]], 1).round(4)
            lim_up = np.nanpercentile([list_data[c], list_data[r]], 99).round(4)
            data_x_valid = list_data[c][~np.isnan(list_data[c])]
            data_y_valid = list_data[r][~np.isnan(list_data[r])] 
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
            _x_fit = np.ma.masked_where(_mask_x, _x_fit_org).compressed()
            _y_fit = np.ma.masked_where(_mask_x, _y_fit_org).compressed()
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
            ax.xaxis.offsetText.set(size=9)
            ax.yaxis.offsetText.set(size=9)

            # add statistics: corr, regression
            p_r, p_p = scst.pearsonr(data_x_valid, data_y_valid)
            rho, p = scst.spearmanr(data_x_valid, data_y_valid)
            tit_str = "$r$=" + str(round(p_r, 2)) + ", $\\rho$=" + str(
                round(rho, 2))
            reg_int_sign = '+' if fit_dat['coef'][1] >= 0 else '-'
            reg_str = 'y='+str(round(fit_dat['coef'][0], 2))+reg_int_sign+str(abs(round(fit_dat['coef'][1], 5)))
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
# colorbar
cax = plt.axes([0.25, 0.40, 0.6, 0.01])
cb = mcbar.ColorbarBase(
    cax,
    cmap=mcm.get_cmap('coolwarm'),
    norm=mcolors.Normalize(vmin=vmin, vmax=vmax),
    orientation='horizontal',
    extend='both'
)
cb.ax.set_xlabel('Contribution (-)')
cb.formatter.set_powerlimits((0, 0))
cb.formatter._useMathText = True

# add row title on the right side
for c in range(nmodels):
    plt.annotate(list_title[c], xy=(1.03, 0.93-c*0.3), xycoords='figure fraction', rotation=90, fontsize=12, va='center', ha='center')

fig.tight_layout()

save_name = os.path.join(dir_plot, 'fig03.png')
fig.savefig(
    save_name,
    dpi=300,
    bbox_inches='tight',
    facecolor='w',
    transparent=False
)