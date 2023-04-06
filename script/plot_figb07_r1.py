'''plot figure b07'''

# =============================================================================
# load libraries and functions
# =============================================================================
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors
import cartopy
import cartopy.crs as ccrs
from config import dir_data, dir_plot
import os

# =============================================================================
# load data
# =============================================================================
file_name = os.path.join(dir_data, 'rsq_change_wo_extrm_norm_res_tws_jpl06v1_non_abs_det_using_20022017_simul_2001_2019.npz')  # forced with gpcp1dd ppt
# file_name = os.path.join(dir_data, 'rsq_change_wo_extrm_norm_res_tws_jpl06v1_non_abs_det_using_20022017_simul_2001_2019_mswep.npz')  # forced with mswep ppt
npz = np.load(file_name)
rsq_tws_monthly = npz['rsq_tws_monthly']
rsq_tws_monthly_running = npz['rsq_tws_monthly_running']
trimmed_pix = npz['trimmed_pix']
list_perc = npz['trim_perc']

# =============================================================================
# plot
# =============================================================================
# plot hotspots

# values and labels
grace_ver = 'jpl06v1'  # 'grace1', 'grace2', 'JPL06CRI1', ...
fig_name = 'TWS IAV'  # 'TWS MSC' or 'TWS IAV'
dat_idx = 2  # 0: MSC, 2: IAV
perc_idx = 8  # 8: 10% trimming
list_trimPerc = [str(x)+'%' for x in list_perc]
extent = [-180, 180, -60, 90]

# plot
fig = plt.figure(figsize=(5, 3))
ax = plt.axes([0, 0, 1, 1],
            projection=ccrs.Robinson(central_longitude=0),
            frameon=False)  #,sharex=right,sharey=all)

cmap = colors.ListedColormap(['w', 'limegreen'])
bounds=[0, 0.5, 1]
norm = colors.BoundaryNorm(bounds, cmap.N)
im1=plt.imshow(
    trimmed_pix[dat_idx, perc_idx, :, :][:150, :],
    interpolation='none',
    origin='upper',
    vmin=-1,
    vmax=1,
    cmap=cmap,
    norm=norm,
    transform=ccrs.PlateCarree(),
    extent=extent
)

cmap = colors.ListedColormap(['w', '#bc15b0'])
bounds=[0, 0.5, 1]
norm = colors.BoundaryNorm(bounds, cmap.N)
im2=plt.imshow(
    trimmed_pix[dat_idx+1, perc_idx, :, :][:150, :],
    interpolation='none',
    origin='upper',
    vmin=-1,
    vmax=1,
    cmap=cmap,
    norm=norm,
    transform=ccrs.PlateCarree(),
    extent=extent,
    alpha=0.6
)
ax.set_extent(extent, crs=ccrs.PlateCarree())
ax.coastlines()
ax.add_feature(cartopy.feature.LAKES, alpha=0.5)
ax.add_feature(cartopy.feature.RIVERS)

# Create legend handles manually
palette = dict(zip(['SINDBAD', 'H2M'], ['limegreen', '#bc15b0']))
handles = [mpl.patches.Patch(color=palette[x], label=x) for x in palette.keys()]
ax.legend(
    bbox_to_anchor=(0.01, 0.05, 0.3, 0.3),
    ncol=1,
    handles=handles,
    fontsize=11,
    frameon=False
)

save_path = os.path.join(dir_plot, 'figb07.png')
fig.savefig(
    save_path,
    bbox_inches='tight',
    dpi=600,
    facecolor='w',
    transparent=False
)