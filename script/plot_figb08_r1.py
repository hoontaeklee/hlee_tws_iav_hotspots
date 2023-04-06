'''plot figure b08'''

# =============================================================================
# load libraries and functions
# =============================================================================
import xarray as xr
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
from config import dir_data, dir_plot
import os

# %%
# nonnan mask
path_common_nonnan = os.path.join(dir_data, 'common_nonnan_pixels.nc')
common_nonnan = xr.open_dataset(path_common_nonnan).common_nonnan_pixels.values

# %%
# tws norm. cov.
path = os.path.join(dir_data, 'cov_TWS_det_scArea_det_using_20022017_simul_2001_2019.npz')  # forcing with gpcp1dd ppt
# path = os.path.join(dir_data, 'cov_TWS_det_scArea_det_using_20022017_simul_2001_2019_mswep.npz')  # forcing with mswep ppt
cov = np.load(path)

norm_tws_iav_gra = cov[cov.files[0]].reshape(180, 360)
norm_tws_iav_sin = cov[cov.files[1]].reshape(180, 360)
norm_tws_iav_h2m = cov[cov.files[2]].reshape(180, 360)
norm_tws_iav_gra = np.where(common_nonnan, norm_tws_iav_gra, np.nan)
norm_tws_iav_sin = np.where(common_nonnan, norm_tws_iav_sin, np.nan)
norm_tws_iav_h2m = np.where(common_nonnan, norm_tws_iav_h2m, np.nan)

# %%
# load tws iav res
path = os.path.join(dir_data, 'cov_TWS_det_resid_scArea_det_using_20022017_simul_2001_2019.npz')  # forcing with gpcp1dd ppt
# path = os.path.join(dir_data, 'cov_TWS_det_resid_scArea_det_using_20022017_simul_2001_2019_mswep.npz')  # forcing with mswep ppt
cov_res = np.load(path)

norm_tws_res_iav_sin = cov_res[cov_res.files[0]].reshape(180, 360)
norm_tws_res_iav_h2m = cov_res[cov_res.files[1]].reshape(180, 360)
norm_tws_res_iav_sin = np.where(common_nonnan, norm_tws_res_iav_sin, np.nan)
norm_tws_res_iav_h2m = np.where(common_nonnan, norm_tws_res_iav_h2m, np.nan)

# %%
# norm_tws_iav vs. norm_tws_iav_res

list_data = [
    [norm_tws_iav_sin, norm_tws_res_iav_sin],
    [norm_tws_iav_h2m, norm_tws_res_iav_h2m]
]

fig = plt.figure(figsize=(6*0.9*2, 4.5*0.9))
ax1 = plt.axes([0.1, 0.15, 0.4, 0.82])
ax2 = plt.axes([0.1+0.4+0.05, 0.15, 0.4, 0.82])
axes = [ax1, ax2]
cax = plt.axes([0.1+0.4+0.05+0.4+0.03, 0.15, 0.02, 0.82], frameon=True)

for i in range(len(axes)):
    rsq = (pg.corr(list_data[i][0].ravel(), list_data[i][1].ravel())['r'][0]**2).round(2)
    ax = axes[i]
    cbar = False if i == 0 else True
    sc = sns.histplot(
        x=list_data[i][0].ravel(),
        y=list_data[i][1].ravel(),
        stat='count',
        ax=ax,
        cmap='viridis_r',
        cbar=cbar,
        cbar_ax=cax,
        vmin=0,
        vmax=80,
        cbar_kws={'label':'Count'})
    ax.set_xlim(-0.001, 0.001)
    ax.set_ylim(-0.001, 0.001)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.annotate(r'R$^2$={rsq}'.format(rsq=rsq), xy=(0.15, 0.05), xycoords='axes fraction')
    ax.ticklabel_format(style='sci', axis='both', scilimits=(0,0))
    ax.xaxis.major.formatter._useMathText = True
    ax.yaxis.major.formatter._useMathText = True
    ax.xaxis.offsetText.set(size=9)
    ax.yaxis.offsetText.set(size=9)

    ax.vlines(x=0, ymin=-0.001, ymax=0.001, colors='grey', linestyle='--', alpha=0.5)
    ax.hlines(y=0, xmin=-0.001, xmax=0.001, colors='grey', linestyle='--', alpha=0.5)
    ax.plot([0,1],[0,1], '--', transform=ax.transAxes, c='grey', alpha=0.5)

plt.annotate('Contribution to TWS IAV (-)', xy=(0.4, 0.01), xycoords='figure fraction', fontsize=13)
plt.annotate('Contribution to TWS IAV errors (-)', xy=(0.015, 0.2), xycoords='figure fraction', fontsize=13, rotation=90)
axes[1].set_yticklabels('')

save_name = os.path.join(dir_plot, 'figb08.png')
fig.savefig(
    save_name,
    dpi=600,
    bbox_inches='tight',
    facecolor='w',
    transparent=False
)