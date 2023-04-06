'''plot figure b01'''

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from config import dir_data, dir_plot
import os

#%% load data
path_in = os.path.join(dir_data, 'GRACE_TWS_JPL_RL06Mv1CRIv01_200204_201706_scaleFactorApplied_areaWeighted_180_360_NaNs_for_2001_2019_masked_200201_201712_det.nc')
ds_tws = xr.open_dataset(path_in)
tws_glo = ds_tws.graceTWS_ano.mean(dim=['lat', 'lon']).values
tws_glo_jan = ds_tws.graceTWS_ano.sel(time=ds_tws.time.dt.month==1).mean(dim=['lat', 'lon']).values
tws_glo_jul = ds_tws.graceTWS_ano.sel(time=ds_tws.time.dt.month==7).mean(dim=['lat', 'lon']).values

#%% fit regression for two example months
time = ds_tws.time.values
x_jan = np.array([np.argwhere(time==e)[0, 0] for e in time if e.astype('datetime64[D]').astype(object).month==1])
x_jul = np.array([np.argwhere(time==e)[0, 0] for e in time if e.astype('datetime64[D]').astype(object).month==7])
x_time = np.array([e.astype('datetime64[Y]').astype(object).year for e in time])

idx_valid = ~np.isnan(tws_glo_jan)
fit_jan = np.polyfit(x_jan[idx_valid], tws_glo_jan[idx_valid], 1)
idx_valid = ~np.isnan(tws_glo_jul)
fit_jul = np.polyfit(x_jul[idx_valid], tws_glo_jul[idx_valid], 1)

tr_x_jan = np.linspace(x_jan[0], x_jan[-1], 100)
tr_jan = tr_x_jan * fit_jan[0] + fit_jan[1]
tr_x_jul = np.linspace(x_jan[0], x_jan[-1], 100)
tr_jul = tr_x_jul * fit_jul[0] + fit_jul[1]

#%% plot
plt.plot(np.arange(time.shape[0]), tws_glo)
plt.plot(x_jan, tws_glo[x_jan], color='blue', marker='o', linestyle='')
plt.plot(x_jul, tws_glo[x_jul], color='red', marker='o', linestyle='')
plt.plot(tr_x_jan, tr_jan, color='blue', label='Regression fit for January')
plt.plot(tr_x_jul, tr_jul, color='red', label='Regression fit for July')

plt.ylabel('TWS anomaly (mm)')
plt.xticks(np.arange(6, x_time.shape[0]-1, 36), x_time[np.arange(6, x_time.shape[0]-1, 36)])
plt.legend(frameon=False)

save_name = os.path.join(dir_plot, 'figb01.png')
plt.savefig(
    save_name,
    dpi=300,
        bbox_inches='tight',
        facecolor='w',
    transparent=False
)
