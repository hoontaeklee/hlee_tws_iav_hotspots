'''plot figure b02'''

import regionmask
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from config import dir_plot
import os

fig = plt.figure(figsize=(5, 3))
ax = plt.axes([0, 0, 1, 1],
            projection=ccrs.Robinson(central_longitude=0),
            frameon=False)  #,sharex=right,sharey=all)
regionmask.defined_regions.srex.plot(ax=ax)
plt.tight_layout()

save_name = os.path.join(dir_plot, 'figb02.png')
fig.savefig(
    save_name,
    dpi=600,
    bbox_inches='tight',
    facecolor='w',
    transparent=False
)