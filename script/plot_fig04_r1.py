'''plot figure 4'''

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
import seaborn as sns
import _shared_plot as ptool
from config import dir_data, dir_plot
import os

def draw_map_ax(a_x, one_deg_data, title='', vmin=-0.001, vmax=0.001):
    extent = [-179.5 , 179.5, -89.5 , 89.5]
    cm=mpl.cm.get_cmap('coolwarm')
    ax = plt.axes(a_x,
                    projection=ccrs.Robinson(central_longitude=0),
                    frameon=False)  #,sharex=right,sharey=all)
    ax=_fix_map(ax)
    im=plt.imshow(np.ma.masked_equal(one_deg_data[0:150, :], -9999.),
                interpolation='none',
                origin='upper', vmin=vmin, vmax=vmax, cmap=cm,
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

path_common_nonnan = os.path.join(dir_data, 'common_nonnan_pixels.nc')
common_nonnan = xr.open_dataset(path_common_nonnan).common_nonnan_pixels.values

# load tws iav res
path = os.path.join(dir_data, 'cov_TWS_det_resid_scArea_det_using_20022017_simul_2001_2019.npz')  # forced with gpcp1dd ppt
cov_res = np.load(path)
norm_tws_res_iav_sin = cov_res[cov_res.files[0]].reshape(180, 360)
norm_tws_res_iav_h2m = cov_res[cov_res.files[1]].reshape(180, 360)
norm_tws_res_iav_sin = np.where(common_nonnan, norm_tws_res_iav_sin, np.nan)
norm_tws_res_iav_h2m = np.where(common_nonnan, norm_tws_res_iav_h2m, np.nan)

# =============================================================================
# plot
# =============================================================================
# diagonal: map of data
# upper diagonal: difference map
# lower diagonal: 2d-histogram in this case

list_data=[
    norm_tws_res_iav_sin,
    norm_tws_res_iav_h2m
]

list_title = [
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
aspect_data = 1.5 * 1680. / 4320.
ysp = -0.09
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

fig = plt.figure(figsize=(11*0.5*1.5, 11*0.5))
1-(0*(1/3+0*-0.03))
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
                vmax=vmax
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
        # lower diagonal: histogram        
        if r > c:
            # draw histogram
            ax=plt.axes([
                    x0 + c * wp + c * xsp + xsp_sca,
                    y0 - (r * hp + r * ysp) + ysp_sca,
                    0.42, 0.35
                ])
            _data_org = list_data[r].reshape(-1)
            _mask = np.ma.masked_invalid(_data_org).mask
            _data_valid = np.ma.masked_where(_mask, _data_org).compressed()
            shared_bins = np.histogram_bin_edges(_data_valid, bins=50)
            sns.histplot(x=list_data[0].reshape(-1), stat='probability', kde=True, bins=shared_bins, ax=ax, color='limegreen', alpha=0.3, label=list_title[0], line_kws={'label':'_nolegend_'})
            sns.histplot(x=list_data[1].reshape(-1), stat='probability', kde=True, bins=shared_bins, ax=ax, color='#bc15b0', alpha=0.3, label=list_title[1], line_kws={'label':'_nolegend_'})
            ax.axvline(x=0, linestyle='dashed', color='black', label='_nolegend_')
            plt.legend(loc='upper right', frameon=False)
            ax.set_xlim(-0.002, 0.002)

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
                if c >= r:
                    ptool.rem_ticklabels()
                plt.ylabel('Probability (-)')
# colorbar
cax = plt.axes([0.25, 0.55, 0.6, 0.01])
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
    plt.annotate(list_title[c], xy=(1.03, 0.9-c*0.45), xycoords='figure fraction', rotation=90, fontsize=12, va='center', ha='center')

fig.tight_layout()

save_name = os.path.join(dir_plot, 'fig04.png')
fig.savefig(
    save_name,
    dpi=300,
    bbox_inches='tight',
    facecolor='w',
    transparent=False
)