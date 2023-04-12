'''plot figure b16'''

# =============================================================================
# load libraries and functions
# =============================================================================

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from config import dir_data, dir_plot
import os

# =============================================================================
# load data
# =============================================================================

path = os.path.join(dir_data, 'SINDBAD_opti_Nov2021/VEG_3_studyArea_p_wSoilBase_RD_inDatetime_lonlat_fullPixel.nc')
nc = xr.open_dataset(path).isel(time=0)
soil_cap = np.flip(nc.p_wSoilBase_RD.values, axis=0) + 4.0  # the 1st soil layer has 4mm capacity

path_common_nonnan = os.path.join(dir_data, 'common_nonnan_pixels.nc')
common_nonnan = xr.open_dataset(path_common_nonnan).common_nonnan_pixels.values
soil_cap = np.where(common_nonnan, soil_cap, np.nan)

# =============================================================================
# plot
# =============================================================================
extent = [-179.5 , 179.5, -59.5 , 89.5]
vmin=0
vmax=2000
cmap='viridis'

fig = plt.figure(figsize=(9, 3))
gs = fig.add_gridspec(nrows=1, ncols=2, figure=fig, width_ratios=[1, 0.33])
                      
ax = fig.add_subplot(gs[0, 0], projection=ccrs.Robinson(central_longitude=0), frameon=False)
ax.add_feature(cfeature.LAKES, alpha=0.1, color='black')
ax.add_feature(cfeature.RIVERS, color='black')
ax.coastlines()
im = ax.imshow(soil_cap[:150, :], extent=extent, cmap=cmap, aspect='auto', vmin=vmin, vmax=vmax, transform=ccrs.PlateCarree())

ax_cb = fig.add_axes([0.17, 0.03, 0.5, 0.025])
cb = plt.colorbar(im, cax=ax_cb, extend='max', orientation='horizontal')
cb.ax.set_xlabel('Soil water storage capacity (mm)')
ax.set_extent(extent, crs=ccrs.PlateCarree())

ax_box = [0.84, 0.03, 0.25, 0.97]
ax = plt.axes(ax_box, frameon=True)
ax.hist(soil_cap.ravel(), bins=100, density=True, alpha=0.5, color='green')
ax.set_xlabel('Soil water storage capacity (mm)')
ax.set_ylabel('Density')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.tight_layout()

save_name = os.path.join(dir_plot, 'figb16.png')
fig.savefig(save_name,
            dpi=600,
            bbox_inches='tight', 
            facecolor='w',
            transparent=False)