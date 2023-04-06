'''plot figure b13'''

# =============================================================================
# load libraries and functions
# =============================================================================
import xarray as xr
import numpy as np
import numpy.ma as ma
from numpy.lib.stride_tricks import sliding_window_view
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import regionmask
from config import dir_data, dir_plot
import os

def get_hotspot_pixels(arr_2d, perc):
    if perc==0:  # no masking --> return all pixels
        return arr_2d
    else:  # masking --> mask-out hotspots
        val_perc = np.nanpercentile(arr_2d, perc)
        cond_hot = np.logical_and(arr_2d>=val_perc, ~np.isnan(arr_2d)) 
        hotspots = np.where(cond_hot, -999, arr_2d) == -999
        return hotspots
    
def scale_minmax(data):
    return (data - np.nanmin(data)) / (np.nanmax(data) - np.nanmin(data))

def get_hist_ht_hot_nhot(data, hot_pix, region_pix, rescale_x=False):
    if rescale_x:
        data = scale_minmax(data)

    pix_target = hot_pix * region_pix
    data_hist1 = np.where(pix_target, data, np.nan)
    data_hist1 = data_hist1[~np.isnan(data_hist1)]
    shared_bins1 = pd.qcut(data_hist1, 20, labels=False, retbins=True, duplicates='drop')[1]  # to use bins with equal number of data
    ht_hot = plt.hist(data_hist1, bins=shared_bins1, density=True)[0]
    kde_hot = stats.gaussian_kde(data_hist1, bw_method='scott')

    data_hist2 = np.where(~pix_target, data, np.nan)
    data_hist2 = data_hist2[~np.isnan(data_hist2)] 
    ht_nhot = plt.hist(data_hist2, bins=shared_bins1, density=True)[0]
    kde_nhot = stats.gaussian_kde(data_hist2, bw_method='scott')
    ksp = stats.ks_2samp(data_hist1, data_hist2)

    dic_return = {
        'bins': shared_bins1,
        'ht_hot': ht_hot,
        'ht_nhot': ht_nhot,
        'kde_hot': kde_hot,
        'kde_nhot': kde_nhot,
        'ksp': ksp
    }

    return dic_return

def calc_delta_density(dic_ht, multiply_barwidth=False):
    
    if multiply_barwidth:
        _barwidth = np.diff(dic_ht['bins'])
        _dd = dic_ht['ht_hot']*_barwidth - dic_ht['ht_nhot']*_barwidth
    else:
        _dd = dic_ht['ht_hot'] - dic_ht['ht_nhot']

    return _dd

def create_even_spaced_seq(arr1d, n):
    '''
    from a given 1d-array,
    create an 1d-array of a specified size with an even interval  
    '''
    n_sub = int(n / (arr1d.size - 1))  # number of elements sampled from the range of the bin (e.g., n=100, nbins=20 ==> 5 valued from each bin)
    n_sub_end = 100 - n_sub * (arr1d.size - 2)
    seq_return = np.array([])

    for i in range(arr1d.size-1):
        _seq_sub = np.linspace(arr1d[i], arr1d[i+1], n_sub)

        seq_return = np.concatenate([seq_return, _seq_sub])

    return seq_return

def plot_bar_even_xscale(ht, bins, color, ax='', draw_bar=True, bottom=False, draw_kde=False, kde='', kde_bot='', label='', multiply_width=True):
    # to plot histogram with equal-size binning, not equal-width.
    
    width_bar = np.diff(bins)
    nbins = width_bar.shape[0]
    bar_x = np.average(sliding_window_view(bins, window_shape=2), axis=1)
    bar_ht = ht
    xtick_new = bar_x  # np.cumsum(bar_x)
    xtick_idx = np.arange(1, xtick_new.shape[0]+1)
    if draw_bar!=False:
        if multiply_width:
            bar_ht = ht*width_bar
        if type(bottom) == bool:

            # a hack to "temporarily" solve an issue... that
            # the number of heights for bar plot is less than 20
            # due to, for example, a skewed distribution of data...
            # ht_etsrc3_h2m has only 19 values as there are more than 5% within a bin
            # so pd.qcut() dropped one bin
            # 
            # so... a "zero height" is added in the end position
            if bar_ht.shape[0] < 20:
                diff_nht = 20 - bar_ht.shape[0]
                bar_ht = np.insert(bar_ht, bar_ht.shape[0], np.repeat(0.0, diff_nht))
                # xtick_idx = np.insert(xtick_idx, xtick_idx.shape[0], xtick_idx[-1]+1) 
            #   
            ax.bar(x=xtick_idx, height=bar_ht, width=1.0, edgecolor='black', color=color, alpha=0.3, label=label, linewidth=0)
            ax.set_xticks(xtick_idx[::4])
            if np.max(xtick_new) > 5.0:
                ax.set_xticklabels(xtick_new[::4].astype('int'))
            else:
                ax.set_xticklabels(xtick_new[::4].round(3))
        else:
            if multiply_width:
                ax.bar(x=xtick_idx, height=bar_ht, bottom=bottom*width_bar, width=1.0, edgecolor='black', color=color, alpha=0.3, label=label, linewidth=0)
                ax.set_xticks(xtick_idx[::4])
                if np.max(xtick_new) > 5.0:
                    ax.set_xticklabels(xtick_new[::4].astype('int'))
                else:
                    ax.set_xticklabels(xtick_new[::4].round(3))
            else:
                ax.bar(x=xtick_idx, height=bar_ht, bottom=bottom, width=1.0, edgecolor='black', color=color, alpha=0.3, label=label, linewidth=0)
                ax.set_xticks(xtick_idx[::4])
                if np.max(xtick_new) > 5.0:
                    ax.set_xticklabels(xtick_new[::4].astype('int'))
                else:
                    ax.set_xticklabels(xtick_new[::4].round(3))

    if draw_kde!=False:
        kde_x = create_even_spaced_seq(arr1d=bins, n=100)
        kde_x_idx = np.linspace(0.5, xtick_idx.max()+0.5, 100)
        if multiply_width:
            kde_x_width_idx = (kde_x_idx-0.5).astype('int')  # idx of width for each kde value
            kde_x_width_idx[kde_x_width_idx==nbins] = nbins-1  # the last element --> the width of the last bar
            width_kde = np.array([width_bar[i] for i in kde_x_width_idx])  # bar width for each kde value
            if type(bottom) == bool:
                ax.plot(kde_x_idx, kde(kde_x)*width_kde, color=color)
            else:
                ax.plot(kde_x_idx, kde_bot(kde_x)*width_kde+kde(kde_x)*width_kde, color=color)
                
        else:
            if type(bottom) == bool:
                ax.plot(kde_x_idx, kde(kde_x), color=color)
                # return kde_x, kde_x_idx
            else:
                ax.plot(kde_x_idx, kde_bot(kde_x)+kde(kde_x), color=color)
    return ax
    # return kde_x, kde_x_idx

def draw_a_col_even_xscale(var_ht, idx_ax_model, color_model, idx_ax_diff, name_model, use_another_bins=False, var_bins=''):
    color_model = color_model
    if use_another_bins==False:
        var_bin = var_ht
    else:
        var_bin = var_bins
    exec('dic_bar = {var_ht}'.format(var_ht=var_ht))
    exec('dic_bin = {var_bin}'.format(var_bin=var_bin))
    exec("plot_bar_even_xscale(ht=dic_bar['ht_hot'], bins=dic_bin['bins'], draw_bar=True, draw_kde=True, kde=dic_bar['kde_hot'], color='red', ax=axes[{idx_ax_model}], label='Hotspots', multiply_width=False)".format(idx_ax_model=idx_ax_model))
    exec("plot_bar_even_xscale(ht=dic_bar['ht_nhot'], bins=dic_bin['bins'], draw_bar=True, bottom=False, draw_kde=True, kde=dic_bar['kde_nhot'], kde_bot=dic_bar['kde_hot'], color='blue', ax=axes[{idx_ax_model}], label='Non-hotspots', multiply_width=False)".format(idx_ax_model=idx_ax_model))
    var_diff = 'diff_'+'_'.join(var_ht.split('_')[1:])
    exec("plot_bar_even_xscale(ht={var_diff}, bins=dic_bin['bins'], color=color_model, ax=axes[{idx_ax_diff}], label='hot - nhot, {name_model}', multiply_width=False)".format(var_diff=var_diff, idx_ax_diff=idx_ax_diff, name_model=name_model))


# =============================================================================
# load data
# =============================================================================

# get nonnan pixels
path_common_nonnan = os.path.join(dir_data, 'common_nonnan_pixels.nc')
common_nonnan = xr.open_dataset(path_common_nonnan).common_nonnan_pixels.values

src=os.path.join(dir_data, 'occurrence.fraction.optionA.T0_50.GSW.360.180.nc')  # fraction of subpixels of which occ. is larger than a threshold (50%)
nc=xr.open_dataset(src)
occ=nc.occurrence.values

src=os.path.join(dir_data, 'recurrence.fraction.GSW.nozero.360.180.nc')
nc=xr.open_dataset(src)
rec=nc.recurrence.values

path = os.path.join(dir_data, 'etsources_global_1x1.nc')
etsrc = xr.open_dataset(path, drop_variables=['latitude', 'longitude'])
etsrc_nonnan = np.flip(etsrc.AREA.values != 0.0, axis=0)

# get nonnan pixles of occurrence and recurrence
common_nonnan_occrec = ~np.isnan(occ) * ~np.isnan(rec)

# get common nonnan pixels
common_nonnan = common_nonnan * common_nonnan_occrec * etsrc_nonnan

# load pixel area and land index
area = np.load(os.path.join(dir_data, 'gridAreaInKm2_180_360.npz'))['area']

# SREX boolean mask
res = 1.0
lat = np.linspace(-90+(res/2), 90-(res/2), int(180/res))
lon = np.linspace(-180+(res/2), 180-(res/2), int(360/res))
srex_mask = regionmask.defined_regions.srex.mask_3D(lon, lat)  # True: within the region

# frac. wetlands
path = os.path.join(dir_data, 'CW_TCI.fractions.360.180.nc')
fracwet_rfw = xr.open_dataset(path).RFW.values

# get hotspots
norm_cov = np.load(os.path.join(dir_data, 'cov_TWS_det_resid_scArea_det_using_20022017_simul_2001_2019.npz'))  # forced by gpcp1dd
norm_tws_res_iav_sin = norm_cov[norm_cov.files[0]].reshape(180, 360)
norm_tws_res_iav_h2m = norm_cov[norm_cov.files[1]].reshape(180, 360)

# mask nans
fracwet_rfw = np.where(common_nonnan, fracwet_rfw, np.nan)
norm_tws_res_iav_sin = np.where(common_nonnan, norm_tws_res_iav_sin, np.nan)
norm_tws_res_iav_h2m = np.where(common_nonnan, norm_tws_res_iav_h2m, np.nan)

# =============================================================================
# calc. bin heights of hotspots and non-hotspots
# =============================================================================

hot_sin = get_hotspot_pixels(arr_2d=norm_tws_res_iav_sin, perc=90)
hot_h2m = get_hotspot_pixels(arr_2d=norm_tws_res_iav_h2m, perc=90)
hot_common = hot_sin * hot_h2m

# set hot_pix mask (common hotspots or not)
hot_pix_sin = hot_sin
hot_pix_h2m = hot_h2m

# set idx of regions interested 
region_groups = list(np.arange(26)+1)  # all regions
region_idx = srex_mask.sel(region=region_groups).values
region_idx = np.flip(np.any(region_idx, axis=0), axis=0)

# get heights and bins of histograms
rescale_x = True
ht_fracwet_rfw_sin = get_hist_ht_hot_nhot(data=fracwet_rfw, hot_pix=hot_pix_sin, region_pix=region_idx, rescale_x=rescale_x)
ht_fracwet_rfw_h2m = get_hist_ht_hot_nhot(data=fracwet_rfw, hot_pix=hot_pix_h2m, region_pix=region_idx, rescale_x=rescale_x)

# calc. differences
_mb=False  # multiply barwidths or not: rescaled --> False
diff_fracwet_rfw_sin = calc_delta_density(dic_ht=ht_fracwet_rfw_sin, multiply_barwidth=_mb)
diff_fracwet_rfw_h2m = calc_delta_density(dic_ht=ht_fracwet_rfw_h2m, multiply_barwidth=_mb)

# =============================================================================
# plot
# =============================================================================

def get_pval_sym(pval):
    if pval > 0.05:
        return ''
    if pval < 0.001:
        return '***'
    if pval >= 0.001 and pval < 0.01:
        return '**'
    if pval <= 0.05 and pval >= 0.01:
        return '*'

# set some values
list_dic_ht = [
    'ht_fracwet_rfw_sin',
    'ht_fracwet_rfw_h2m'
]
list_lab_data = [
    'RF Wetlands (-)'
] 
group_name_sel = [
    'rfwetlands'
]
group_ylim_minmaxx = [  # for minmax xscale
    [0, 3], [0, 3], [-6, 9]
]
region_sel = '_'.join([str(i) for i in region_groups])
if len(region_groups)==26:
    region_sel = 'all'

group_ylim = group_ylim_minmaxx

plt.rc('xtick', labelsize=13)    # fontsize of the tick labels
plt.rc('ytick', labelsize=13)    # fontsize of the tick labels

# plot
nrow=3
ncol = int(len(list_dic_ht) / 2)
fig, axes = plt.subplots(nrow, ncol, figsize=(6*ncol*0.7, 3*nrow*0.7))
axes=axes.flatten(order='F')

for c in range(ncol):  # iteratively draw each column
    print('c={c}'.format(c=c))
    draw_a_col_even_xscale(var_ht=list_dic_ht[2*c], idx_ax_model=str(3*c), color_model='green', idx_ax_diff=str(3*c+2), name_model='SINDBAD')
    draw_a_col_even_xscale(var_ht=list_dic_ht[2*c+1], idx_ax_model=str(3*c+1), color_model='purple', idx_ax_diff=str(3*c+2), name_model='H2M', use_another_bins=True, var_bins=list_dic_ht[2*c])

# set aesthetics
    
fontsize=16
for i in [0, 1]:
    axes[i].set_ylabel('Density', fontsize=fontsize)
    axes[i].set_ylabel('Density', fontsize=fontsize)
axes[0].legend()
axes[2].set_ylabel(r'$\Delta$ Density', fontsize=fontsize)
axes[2].legend()

for i in range(ncol):
    # 1st and 2nd rows
    _idx = np.arange(0, nrow*ncol, nrow)  # idx for the first row of each column

    exec("ksp_sin = {dic_ht}['ksp'][1]".format(dic_ht=list_dic_ht[2*i]))
    exec("ksp_h2m = {dic_ht}['ksp'][1]".format(dic_ht=list_dic_ht[2*i+1]))
    sym_sin = get_pval_sym(pval=ksp_sin)
    sym_h2m = get_pval_sym(pval=ksp_h2m)

    _xlim = axes[_idx[i]].get_xlim()
    axes[_idx[i]+1].set_xlim(_xlim)

    y_up = abs(axes[_idx[i]].get_ylim()[1])
    y_expo1 = np.floor(np.log10(y_up)).astype('int')
    y_up = abs(axes[_idx[i]+1].get_ylim()[1])
    y_expo2 = np.floor(np.log10(y_up)).astype('int')
    y_expo = np.nanmax([y_expo1, y_expo2])
    axes[_idx[i]].set_ylim(group_ylim[0][0]*np.float_power(10, y_expo), group_ylim[0][1]*np.float_power(10, y_expo))
    
    axes[_idx[i]+1].set_ylim(group_ylim[1][0]*np.float_power(10, y_expo), group_ylim[1][1]*np.float_power(10, y_expo))

    axes[_idx[i]].set_title('SINDBAD'+sym_sin)
    axes[_idx[i]+1].set_title('H2M'+sym_h2m)
    axes[_idx[i]].set_xticklabels('')
    axes[_idx[i]+1].set_xticklabels('')
    
    # 3rd row
    _idx = np.arange(2, nrow*ncol, nrow)  # idx for the third row of each column
    axes[_idx[i]].set_xlabel(list_lab_data[i], fontsize=fontsize)
    y_up = abs(axes[_idx[i]].get_ylim()[0])
    y_expo = 1
    axes[_idx[i]].set_ylim(group_ylim[2][0]*np.float_power(10, y_expo), group_ylim[2][1]*np.float_power(10, y_expo))

for ax in axes:
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    ax.yaxis.major.formatter._useMathText = True
    ax.yaxis.offsetText.set(size=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

fig.tight_layout()

# save
save_name = os.path.join(dir_plot, 'figb13.png')
fig.savefig(
    save_name,
    dpi=600,
    facecolor='w',
    transparent=False
)