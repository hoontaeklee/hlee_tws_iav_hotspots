'''plot figure b15'''

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
from config import dir_data, dir_plot
import os

def draw_map_ax(a_x, one_deg_data, title='', title_fsize=10, vmin=-0.001, vmax=0.001):
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
    ax.set_title(title, fontsize=title_fsize)
    
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

# get nan pixels
path_common_nonnan = os.path.join(dir_data, 'common_nonnan_pixels.nc')
common_nonnan = xr.open_dataset(path_common_nonnan).common_nonnan_pixels.values

# load tws iav res
path = os.path.join(dir_data, 'cov_TWS_det_resid_scArea_eartH2Observe.npz')
cov_res = np.load(path)

norm_tws_res_iav_csiro = cov_res[cov_res.files[0]].reshape(180, 360)
norm_tws_res_iav_jrc = cov_res[cov_res.files[1]].reshape(180, 360)
norm_tws_res_iav_metfr = cov_res[cov_res.files[2]].reshape(180, 360)
norm_tws_res_iav_uu = cov_res[cov_res.files[3]].reshape(180, 360)

norm_tws_res_iav_csiro = np.where(common_nonnan, norm_tws_res_iav_csiro, np.nan)
norm_tws_res_iav_jrc = np.where(common_nonnan, norm_tws_res_iav_jrc, np.nan)
norm_tws_res_iav_metfr = np.where(common_nonnan, norm_tws_res_iav_metfr, np.nan)
norm_tws_res_iav_uu = np.where(common_nonnan, norm_tws_res_iav_uu, np.nan)

# =============================================================================
# plot
# =============================================================================
# diagonal: map of data
# upper diagonal: difference map

list_data=[
    norm_tws_res_iav_csiro,
    norm_tws_res_iav_jrc,
    norm_tws_res_iav_metfr,
    norm_tws_res_iav_uu
]

list_title = [
    'W3RA',
    'LISFLOOD',
    'SURFEX-TRIP',
    'PCR-GLOBWB'
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

fig = plt.figure(figsize=(11*0.5*1.5, 11*0.5))
for r in range(nmodels):
    for c in range(nmodels):
        # diagonal: maps
        if r == c:
            axl=[
                x0 + c * wp + c * xsp,
                y0 - (r * hp + r * ysp),
                wp,
                hp]
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

# colorbar
cax = plt.axes([0.25, 0.4, 0.6, 0.01])
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
    plt.annotate(list_title[c], xy=(1.03, 0.90-c*0.187), xycoords='figure fraction', rotation=90, fontsize=10, va='center', ha='center')

fig.tight_layout()

save_name = os.path.join(dir_plot, 'figb15.png')
fig.savefig(
    save_name,
    dpi=300,
    bbox_inches='tight',
    pad_inches=0.0,
    facecolor='w',
    transparent=False
)