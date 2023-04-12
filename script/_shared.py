import os
import numpy as np
import scipy.stats as scst
import pprint as ppr
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd

def _apply_a_mask(_dat, _mask_dat, _fill_val=np.nan):
    mask_where = np.ma.getmask(np.ma.masked_less(_mask_dat, 1.))
    _dat[mask_where] = _fill_val
    return _dat

def _apply_common_mask_g(*args):
    nargs = len(args)
    in_d = 0
    for ar in range(nargs):
        _dat = args[ar]
        _tmp = np.ones_like(_dat)
        _tmp[_dat == -9999.] = 0
        _tmp_inv_mask = np.ma.masked_invalid(_dat).mask
        _tmp[_tmp_inv_mask] = 0
        if in_d == 0:
            dat_mask = _tmp
        else:
            dat_mask = dat_mask * _tmp
        in_d = in_d + 1
    odat = np.zeros_like(args)
    for ar in range(nargs):
        _dat = args[ar]
        odat[ar] = np.ma.masked_invalid(
            _apply_a_mask(_dat.astype(np.float), dat_mask))
    return odat

def _compress_invalid(_dat):
    odat = np.ma.masked_invalid(_dat).compressed()
    return odat

def _quadfun_res(p, x, y):
    return p[0] * x**2 + p[1] * x + p[2] - y

def _quadfun_res_noIn(p, x, y):
    return p[0] * x**2 + p[1] * x - y

def _linearfun_res(p, x, y):
    return p[0] * x + p[1] - y

def _fit_least_square(_xDat_tr,
                      _yDat_tr,
                      _logY=False,
                      method='quad',
                      _intercept=True,
                      _bounds=None,
                      _loss_func='huber'):
    from scipy.optimize import least_squares
    import scipy.stats as scst
    if method == 'quad':
        pred_x = np.linspace(np.nanpercentile(_xDat_tr, 2),
                             np.nanpercentile(_xDat_tr, 98), 200)
        if _logY:
            y_tar = np.log10(_yDat_tr)
        else:
            y_tar = _yDat_tr
        # _xDat_tr, y_tar = _apply_common_mask_g(_xDat_tr, y_tar)
        # _xDat_tr = _compress_invalid(_xDat_tr)
        # y_tar = _compress_invalid(y_tar)
        if _intercept:
            res_lsq_robust = least_squares(_linearfun_res, [0, 0],
                                           jac='3-point',
                                           ftol=1e-15,
                                           xtol=1e-15,
                                           method='trf',
                                           tr_solver='exact',
                                           loss=_loss_func,
                                           f_scale=1,
                                           bounds=_bounds,
                                           args=(_xDat_tr, y_tar))
            params = res_lsq_robust.x
            pcoff = [params[0], params[1]]
        else:
            res_lsq_robust = least_squares(_quadfun_res_noIn, [0, 0, 0],
                                           jac='3-point',
                                           ftol=1e-15,
                                           xtol=1e-15,
                                           method='trf',
                                           tr_solver='exact',
                                           loss=_loss_func,
                                           f_scale=1,
                                           bounds=_bounds,
                                           args=(_xDat_tr, y_tar))
            params = res_lsq_robust.x
            pcoff = [params[0], params[1], 0]

        if _logY:
            pred_y = 10**(pcoff[0] * pred_x**2 + pcoff[1] * pred_x + pcoff[2])
            pred_y_f = 10**(pcoff[0] * _xDat_tr**2 + pcoff[1] * _xDat_tr +
                            pcoff[2])
        else:
            pred_y = pcoff[0] * pred_x + pcoff[1]
            pred_y_f = pcoff[0] * _xDat_tr + pcoff[1]

        r_mad = np.nanmedian(np.abs(pred_y_f - _yDat_tr)) / (
            np.nanpercentile(_yDat_tr, 75) - np.nanpercentile(_yDat_tr, 25))
        n_rmse = np.sqrt(np.mean(
            (pred_y_f - _yDat_tr)**
            2)) / (np.nanmax(_yDat_tr) - np.nanmin(_yDat_tr))
        ppro = ppr.PrettyPrinter(indent=4)
        ppro.pprint(res_lsq_robust)
        if not _intercept:
            pcoff = np.append(pcoff, 0)

        try:
            pred_y_f, y_tar = _apply_common_mask_g(pred_y_f, y_tar)
            pred_y_f = _compress_invalid(pred_y_f)
            y_tar = _compress_invalid(y_tar)
            r2 = scst.pearsonr(pred_y_f, _yDat_tr)[0]**2
        except:
            r2 = np.nan
        res_ult = {}
        res_ult['pred'] = {}
        res_ult['pred']['x'] = pred_x
        res_ult['pred']['y'] = pred_y
        res_ult['coef'] = pcoff
        res_ult['metr'] = {}
        res_ult['metr']['r2'] = r2
        res_ult['metr']['n_rmse'] = n_rmse
        res_ult['metr']['r_mad'] = r_mad
        print('----------------------------------')
        return res_ult
    if method == 'odr_quad':
        from scipy import odr
        pred_x = np.linspace(np.nanpercentile(_xDat_tr, 2),
                             np.nanpercentile(_xDat_tr, 98), 200)
        if _logY:
            y_tar = np.log10(_yDat_tr)
        else:
            y_tar = _yDat_tr
        _xDat_tr, y_tar = _apply_common_mask_g(_xDat_tr, y_tar)
        _xDat_tr = _compress_invalid(_xDat_tr)
        y_tar = _compress_invalid(y_tar)
        if _intercept:
            fit_model = odr.Model(_quadfun_res)
            fit_data = odr.Data(_xDat_tr, y_tar)
            fit_odr = odr.ODR(fit_data, odr.quadratic, beta0=[1, 1, 1],maxit=1000)
            fit_output = fit_odr.run()
            fit_output.pprint()
            # kera
            # res_lsq_robust = least_squares(_quadfun_res, [1, 1, 1],
            #                                jac='3-point',
            #                                ftol=1e-15,
            #                                xtol=1e-15,
            #                                method='trf',
            #                                tr_solver='exact',
            #                                loss=_loss_func,
            #                                f_scale=1,
            #                                bounds=_bounds,
            #                                args=(_xDat_tr, y_tar))
            params = fit_output.beta
            pcoff = [params[0], params[1], params[2]]
        else:
            fit_model = odr.Model(_quadfun_res_noIn)
            fit_data = odr.Data(_xDat_tr, y_tar)
            fit_odr = odr.ODR(fit_data, odr.quadratic, beta0=[1, 1, 1],maxit=1000)
            fit_output = fit_odr.run()
            fit_output.pprint()
            # res_lsq_robust = least_squares(_quadfun_res_noIn, [1, 1, 1],
            #                                jac='3-point',
            #                                ftol=1e-15,
            #                                xtol=1e-15,
            #                                method='trf',
            #                                tr_solver='exact',
            #                                loss=_loss_func,
            #                                f_scale=1,
            #                                bounds=_bounds,
            #                                args=(_xDat_tr, y_tar))
            params = fit_output.beta
            pcoff = [params[0], params[1], 0]

        if _logY:
            pred_y = 10**(pcoff[0] * pred_x**2 + pcoff[1] * pred_x + pcoff[2])
            pred_y_f = 10**(pcoff[0] * _xDat_tr**2 + pcoff[1] * _xDat_tr +
                            pcoff[2])
        else:
            pred_y = pcoff[0] * pred_x**2 + pcoff[1] * pred_x + pcoff[2]
            pred_y_f = pcoff[0] * _xDat_tr**2 + pcoff[1] * _xDat_tr + pcoff[2]

        r_mad = np.nanmedian(np.abs(pred_y_f - _yDat_tr)) / (
            np.nanpercentile(_yDat_tr, 75) - np.nanpercentile(_yDat_tr, 25))
        n_rmse = np.sqrt(np.mean(
            (pred_y_f - _yDat_tr)**
            2)) / (np.nanmax(_yDat_tr) - np.nanmin(_yDat_tr))
        ppro = ppr.PrettyPrinter(indent=4)
        # ppro.pprint(res_lsq_robust)
        if not _intercept:
            pcoff = np.append(pcoff, 0)

        try:
            pred_y_f, y_tar = _apply_common_mask_g(pred_y_f, y_tar)
            pred_y_f = _compress_invalid(pred_y_f)
            y_tar = _compress_invalid(y_tar)
            r2 = scst.pearsonr(pred_y_f, _yDat_tr)[0]**2
        except:
            r2 = np.nan
        res_ult = {}
        res_ult['pred'] = {}
        res_ult['pred']['x'] = pred_x
        res_ult['pred']['y'] = pred_y
        res_ult['coef'] = pcoff
        res_ult['metr'] = {}
        res_ult['metr']['r2'] = r2
        res_ult['metr']['n_rmse'] = n_rmse
        res_ult['metr']['r_mad'] = r_mad
        print('----------------------------------')
        return res_ult

def calc_rsq(d1, d2):
    rsq = (pg.corr(d1, d2)['r'][0]**2).round(2)
    return rsq

def calc_nse(obs, est):
    return 1-(np.nansum((est-obs)**2)/np.nansum((obs-np.nanmean(obs))**2))

def mask_and_weight(data_timelatlon, mask_to_retain, weight, func='sum'):
    
    data_timelatlon = np.where(mask_to_retain, data_timelatlon, np.nan)
    weight = np.where(mask_to_retain, weight, np.nan)
    result = np.empty(data_timelatlon.shape[0]) * np.nan
    if func=='sum':       
        result = np.array([np.nansum(_ts * weight)/(np.nansum(weight)) for _ts in data_timelatlon])
    if func=='std':
        result = np.array([np.nanstd(_ts * weight)/(np.nansum(weight)) for _ts in data_timelatlon])
        
    return result

def running_mean(x, N):
    '''
    N-months running mean (+- 2/N months), e.g. nanmean(month6~month18) for month12
    for the first 6 and last 5 values, less then 12 values will be used for the mean
    x: np.array, N: window length, output: outA: np.array()
    '''
    intV = int(N / 2)
    lenX = len(x)
    outA = np.ones(np.shape(x))
    for _ind in range(lenX):
        _indMin = max(_ind - intV, 0)
        _indMax = min(_ind + intV + 1, lenX)
        if N%2==0:
            _indMax = _indMax - 1
        outA[_ind] = np.nanmean(x[_indMin:_indMax])
    return outA

def plot_timeseries_tws_met_simplified_regions(arr_tws, arr_met, arr_size, ylim_tws, ylim_met, title, met='ppt'):
    # ppt or tair or rn
    if met == 'ppt':
        var_met = 'ppt'
        ylab_met = 'PPT IAV (mm)'
    elif met == 'rn':
        var_met = 'rn'
        ylab_met = r'Rn IAV (W/m$^2$)'
    else:
        var_met = 'tair'
        ylab_met = 'Tair IAV (degC)'
    
    # gridspec inside gridspec
    global fig
    fig = plt.figure(constrained_layout=False, figsize=(9, 8))
    # subfigs = fig.subfigures(2, 2, wspace=0.00, hspace=0.0)
    # subfigs = subfigs.flatten()

    label = ['GRACE', 'SINDBAD', 'H2M']
    palette = dict(zip(label, ['black', 'limegreen', '#bc15b0']))
    handles = [mpl.patches.Patch(color=palette[x], label=x) for x in palette.keys()]
    x_years = pd.date_range(start='2002-01',end='2017-12', freq='1MS')
    x_years_label = np.arange(2002, 2015, 3)
    alphabets = [chr(i) for i in range(ord('a'),ord('z')+1)]
    srex_names = ['North America', 'South America', 'Africa', 'South Asia']  # srex_mask.names.values

    hsp = 0.03

    for r_idx in range(4):

        # upper subplots

        ax_box = [0.03+0.5*(r_idx%2), 0.04+0.255*(3-2*(r_idx//2))+hsp*(1-1*(r_idx//2)), 0.5, 0.250]
        # if r_idx in [2, 3]:
        #     ax_box[1::2] = [x+0.05 for x in ax_box[1::2]]
        ax = fig.add_axes(ax_box, xticklabels=[], ylim=ylim_tws)
        ax.grid(True, linestyle='--')
        ax.spines['bottom'].set_visible(False)
        ax.annotate(f'({alphabets[r_idx]})', xy=(0.03, 0.89), xycoords='axes fraction', fontsize=15, weight='bold')       
        tws_mean = []
        tws_mean_smth = []
        for i in range(3):
            tws_mean.append(arr_tws[i, r_idx])
            tws_mean_smth.append(running_mean(arr_tws[i, r_idx], 12))
            ax.plot(x_years, tws_mean_smth[i], label=label[i], color=list(palette.values())[i])
            ax.plot(x_years, tws_mean[i], color=list(palette.values())[i], alpha=0.2)

        if r_idx in [0, 2]:
            ax.set_ylabel('TWS IAV (mm)', rotation='vertical', fontsize=12)
        if r_idx in [1, 3]:
            ax.tick_params(labelleft=False)        

        if r_idx in [1]:
            leg = ax.legend(labels=label, bbox_to_anchor=(0.5, 1.25), ncol=3, fontsize=12)
            leg.legendHandles[0].set_color(list(palette.values())[0])
            leg.legendHandles[1].set_color(list(palette.values())[1])
            leg.legendHandles[2].set_color(list(palette.values())[2])
        ax.set_yticks(np.linspace(ylim_tws[0], ylim_tws[1], num=5))
        ax.set_yticks(ax.get_yticks()[1:-1])

        # lower subplots
        ax_box = [0.03+0.5*(r_idx%2), 0.04+0.255*(2-2*(r_idx//2))+hsp*(1-1*(r_idx//2)), 0.5, 0.250]
        # if r_idx in [2, 3]:
        #     ax_box[1::2] = [x+0.05 for x in ax_box[1::2]]
        ax = fig.add_axes(ax_box, ylim=ylim_met)
        ax.set_yticks(np.linspace(ylim_met[0], ylim_met[1], num=5))
        arr_mean = arr_met[r_idx]
        arr_mean_smth = running_mean(arr_met[r_idx], 12)
        
        if met == 'ppt':
            ax.bar(x_years, arr_mean_smth, width=30.5, label='PPT IAV', color='tab:blue', alpha=0.8)
            ax.bar(x_years, arr_mean, width=30.5, color='tab:blue', alpha=0.2)
        elif met == 'rn':
            ax.plot(x_years, arr_mean_smth, label='Rn IAV', color='tab:blue', alpha=0.8)
            ax.plot(x_years, arr_mean, color='tab:blue', alpha=0.2)
        else:
            ax.plot(x_years, arr_mean_smth, label='Tair IAV', color='tab:blue', alpha=0.8)
            ax.plot(x_years, arr_mean, color='tab:blue', alpha=0.2)
        ax.tick_params(axis='y')
        ax.grid(True, linestyle='--')
        ax.spines['top'].set_color('silver')
        ax.set_yticks(ax.get_yticks()[1:-1])

        if r_idx in [0, 2]:
            ax.set_ylabel(ylab_met, rotation='vertical', fontsize=12)
        if r_idx in [0, 1]:
            ax.tick_params(labelbottom=False)
        if r_idx in [1, 3]:
            ax.tick_params(labelleft=False)

        # add r2 and the number of pixels...
        txt_pix_cnt = 'Number of pixels: {cnt}'.format(cnt=int(arr_size[0, r_idx]))
        ax.annotate(txt_pix_cnt, xy=(0.01, 0.02), xycoords='axes fraction')

        rsq_sin = '{:.02f}'.format(pg.corr(tws_mean_smth[1], tws_mean_smth[0])['r'][0]**2)
        rsq_h2m = '{:.02f}'.format(pg.corr(tws_mean_smth[2], tws_mean_smth[0])['r'][0]**2)
        rsq_string = 'R$^2$ (GRACE, Model): {sin}(SINDBAD), {h2m}(H2M)'.format(sin=rsq_sin, h2m=rsq_h2m)
        ax.annotate(rsq_string, xy=(0.01, 0.1), xycoords='axes fraction')

        rsq_gra = '{:.02f}'.format(pg.corr(tws_mean_smth[0], arr_mean_smth)['r'][0]**2)
        rsq_sin = '{:.02f}'.format(pg.corr(tws_mean_smth[1], arr_mean_smth)['r'][0]**2)
        rsq_h2m = '{:.02f}'.format(pg.corr(tws_mean_smth[2], arr_mean_smth)['r'][0]**2)
        rsq_string = 'R$^2$ (TWS, {met}): {gra}(GRACE), {sin}(SINDBAD), {h2m}(H2M)'.format(gra=rsq_gra, sin=rsq_sin, h2m=rsq_h2m, met=var_met)
        ax.annotate(rsq_string, xy=(0.01, 0.2), xycoords='axes fraction')


    fig.text(0, 1.105, title, fontsize=14, weight='bold', transform=fig.transFigure)
    fig.text(0.5, -0.01, 'Years', ha='center', va='center', fontsize=12, transform=fig.transFigure)

    return fig
