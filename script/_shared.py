import os
import xarray as xr
import numpy as np
import pingouin as pg
import matplotlib.pyplot as plt
import pandas as pd
import scipy.stats as scst
import _shared_plot as ptool
import scipy.odr as scodr
import json
import pprint as ppr


def _get_set():
    co_settings = json.load(open("settings_common.json"))
    obs_set_name = os.environ['c_cycle_obs_set']
    co_settings["exp_suffix"] = co_settings["exp_suffix"] + obs_set_name
    obs_settings = json.load(open("settings_obs_tau_"+ obs_set_name +".json"))



    # obs_settings = json.load(open(co_settings["obs_settings_file"]))
    co_settings["obs_dict"] = obs_settings
    co_settings['fig_settings']['fig_dir'] = co_settings['fig_settings'][
        'fig_dir'] + co_settings['exp_suffix'] + '/'
    os.makedirs(co_settings['fig_settings']['fig_dir'] + 'diag_figs_inp',
                exist_ok=True)
    os.makedirs(co_settings['fig_settings']['fig_dir'], exist_ok=True)
    return co_settings


def _rem_nan(tmp, _fill_val=np.nan):
    whereisNan = np.isnan(tmp)
    tmp[whereisNan] = _fill_val
    whereisNan = np.isinf(tmp)
    tmp[whereisNan] = _fill_val
    tmp[tmp == 0.] = _fill_val
    return tmp


def _apply_a_mask(_dat, _mask_dat, _fill_val=np.nan):
    mask_where = np.ma.getmask(np.ma.masked_less(_mask_dat, 1.))
    _dat[mask_where] = _fill_val
    return _dat


def _get_data(_whichVar,
              _co_settings,
              _co_mask=None,
              _get_full_obs=False,
              _fill_val=np.nan,
              _get_model_time_series=False):
    top_dir = _co_settings['top_dir']
    obs_dict = _co_settings['obs_dict']
    pr_unit_corr = _co_settings['pr_unit_corr']
    reso = _co_settings['reso']
    syear = _co_settings['syear']
    eyear = _co_settings['eyear']

    if _fill_val is None:
        _fill_val = _co_settings['fill_val']
    _scn = _co_settings['_scn']
    _models = _co_settings['model']['names']

    all_mod_dat = {}
    for _md in _models:
        # print(_md)

        if _md == 'obs':
            datfile = os.path.join(top_dir,
                                    obs_dict[_whichVar]['obs_file'])
            datVar = obs_dict[_whichVar]['obs_var']
            mod_dat_f = xr.open_dataset(datfile, decode_times=False)
            if _get_full_obs:
                mod_dat0 = mod_dat_f[datVar].values.reshape(-1, 360, 720)
            else:
                mod_dat0 = np.nanmedian(mod_dat_f[datVar].values.reshape(
                    -1, 360, 720),axis=0)
            # elif _co_settings[
            #         "obs_settings_file"] in ["settings_obs_tau_extended_cube.json", "settings_obs_tau_carvalhais2014_cube.json"]:
            #     datfile = os.path.join(top_dir,
            #                            obs_dict[_whichVar]['obs_file'])
            #     datVar = obs_dict[_whichVar]['obs_var']
            #     mod_dat_f = xr.open_dataset(datfile, decode_times=False)
            #     if _get_full_obs:
            #         mod_dat0 = mod_dat_f[datVar].values.reshape(-1, 360, 720)
            #     else:
            #         mod_dat0 = np.nanmedian(mod_dat_f[datVar].values.reshape(
            #             -1, 360, 720),axis=0)
            # else:
            #     print(
            #         "observation configuration not implemented in _shared _get_data"
            #     )
            #     exit
            datCorr = 1
        else:

            if _get_model_time_series:
                syear_ts_data = str(
                    _co_settings['model_dict'][_md]['start_year_ts'])
                eyear_ts_data = str(
                    _co_settings['model_dict'][_md]['end_year_ts'])
                syear_an = str(_co_settings['syear_ts'])
                eyear_an = str(_co_settings['eyear_ts'])

                datVar = obs_dict[_whichVar]['model_var']
                datCorr = obs_dict[_whichVar]['corr_factor_model']
                datfile = os.path.join(top_dir + 'Models/harmonized_' + reso +
                                       '/' + _md + '_' + _scn + '_' + datVar +
                                       '_ts_' + syear_ts_data + '-' +
                                       eyear_ts_data + '_harmonized_' + reso +
                                       '.nc')
                mod_dat_f = xr.open_dataset(datfile)
                mod_dat_f = mod_dat_f.sel(year=slice(syear_an, eyear_an))
                mod_dat0 = mod_dat_f[datVar].values
                mod_dat = _rem_nan(mod_dat0) * datCorr
                all_mask, arI, area_dat = _get_aux_data(_co_settings)
                if _whichVar in _co_settings['pgc_vars']:
                    mod_dat = np.array(
                        [_mod_dat * area_dat for _mod_dat in mod_dat])
                mod_dat_f[datVar].values = mod_dat
                all_mod_dat[_md] = mod_dat_f
                continue
            else:
                datVar = obs_dict[_whichVar]['model_var']
                datfile = os.path.join(
                    top_dir + 'Models/harmonized_' + reso,
                    _md + '_' + _scn + '_' + datVar + '_mean_' + syear + '-' +
                    eyear + '_harmonized_' + reso + '.nc')
                mod_dat_f = xr.open_dataset(datfile, decode_times=False)
                mod_dat0 = mod_dat_f[datVar].values  #*datCorr
            if _whichVar == 'pr':
                datCorr = pr_unit_corr[_md]
            else:
                datCorr = obs_dict[_whichVar]['corr_factor_model']
        # plt.show()
        mod_dat_f.close()
        if _whichVar.startswith('tas'):
            if _md != 'obs':
                mod_dat0 = mod_dat0 - 273.15
        mod_dat = _rem_nan(mod_dat0)
        mod_dat = mod_dat * datCorr
        if _co_mask is not None:
            if mod_dat.ndim == 3:
                mod_dat = np.array([_apply_a_mask(_mod_dat, _co_mask) for _mod_dat in mod_dat])
            else:
                mod_dat = _apply_a_mask(mod_dat, _co_mask)



        print('-----------------------------------------------')
        print("Reading data:")
        print("     variable:", datVar)
        print("     Model:", _md)
        tmp_fig = os.path.join(
            _co_settings['fig_settings']['fig_dir'] + 'diag_figs_inp/',
            datVar + '_' +_md + _co_settings['exp_suffix'] + '.png')
        if not os.path.exists(tmp_fig):
            import cartopy.crs as ccrs
            _ax = plt.subplot(1,
                            1,
                            1,
                            projection=ccrs.Robinson(central_longitude=0),
                            frameon=False)
            plt.imshow(np.ma.masked_less(mod_dat.reshape(-1, 360, 720).mean(0)[:300, :], -999.),
                    vmin=_co_settings['obs_dict'][_whichVar]['plot_range_map'][0],
                    vmax=_co_settings['obs_dict'][_whichVar]['plot_range_map'][1],
                    cmap=_co_settings['obs_dict'][_whichVar]['color_scale'],
                    origin='upper',
                    transform=ccrs.PlateCarree(),
                    extent=[-180, 180, -60, 90])
            _ax.set_extent([-180, 180, -60, 90], crs=ccrs.PlateCarree())
            _ax.coastlines(linewidth=0.4, color='grey')
            plt.gca().outline_patch.set_visible(False)
            plt.colorbar(pad=0.0316, aspect=35 ,orientation='horizontal', shrink = 0.676, extend='both')
            plt.title(_md + ':   ' + datVar +' ('+_co_settings['obs_dict'][_whichVar]['title']+', '+_co_settings['obs_dict'][_whichVar]['unit']+')')
            plt.savefig(tmp_fig,
            bbox_inches='tight', dpi=_co_settings['fig_settings']['fig_dpi'])
            plt.close()

        all_mod_dat[_md] = mod_dat
    return all_mod_dat


def _get_aux_data(co_settings):
    top_dir = co_settings['top_dir']
    all_mask = xr.open_dataset(
        os.path.join(top_dir, co_settings['auxiliary']['mask_file']))[
            co_settings['auxiliary']['mask_var']].values

    arI = xr.open_dataset(
        os.path.join(top_dir, co_settings['auxiliary']['aridity_file']))[
            co_settings['auxiliary']['aridity_var']].values

    arI = _apply_a_mask(arI, all_mask)
    # get the gridded area and fraction
    area_dat = xr.open_dataset(
        os.path.join(top_dir, co_settings['auxiliary']['area_file']))[
            co_settings['auxiliary']['area_var']].values
    area_dat = _apply_a_mask(area_dat, all_mask)

    landFrac = xr.open_dataset(
        os.path.join(top_dir, co_settings['auxiliary']['land_frac_file']))[
            co_settings['auxiliary']['land_frac_var']].values
    area_dat = area_dat * landFrac * 0.01
    return all_mask, arI, area_dat


def _get_obs_percentiles(_whichVar, _co_settings, _perc_range, _co_mask=None):
    top_dir = _co_settings['top_dir']
    obs_dict = _co_settings['obs_dict']
    datfile = os.path.join(top_dir, obs_dict[_whichVar]['obs_file'])
    datVar = obs_dict[_whichVar]['obs_var']
    mod_dat_f = xr.open_dataset(datfile, decode_times=False)
    obs_full = mod_dat_f[datVar].values
    mod_dat_5 = np.nanpercentile(obs_full, _perc_range[0], axis=0)
    mod_dat_95 = np.nanpercentile(obs_full, _perc_range[1], axis=0)

    if _co_mask is not None:
        mod_dat_5 = _apply_a_mask(mod_dat_5, _co_mask)
        mod_dat_95 = _apply_a_mask(mod_dat_95, _co_mask)

    return mod_dat_5, mod_dat_95


def _get_colormap_info(_whichVar, _co_settings, isratio=False, isdiff=False):
    import matplotlib as mpl
    obs_dict = _co_settings['obs_dict']
    if isratio:
        border = 0.9
        ncolo = 128
        num_gr = int(ncolo // 4)
        num_col = num_gr - 4

        _bounds_rat = np.concatenate(
            (np.geomspace(0.2, 0.25,
                          num=num_col), np.geomspace(0.25, 0.33, num=num_col),
             np.geomspace(0.33, 0.5,
                          num=num_col), np.geomspace(0.5, border, num=num_col),
             np.linspace(border, 1 / border,
                         num=num_gr), np.geomspace(1 / border, 2, num=num_col),
             np.geomspace(2, 3, num=num_col), np.geomspace(3, 4, num=num_col),
             np.geomspace(4, 5, num=num_col)))

        cb_ticks = [0.2, 0.25, 0.33, 0.5, 0.9, 1.1, 2, 3, 4, 5]

        cb_labels = [
            '  $\\dfrac{1}{5}$', '  $\\dfrac{1}{4}$', '  $\\dfrac{1}{3}$',
            '  $\\dfrac{1}{2}$', ' $\\dfrac{1}{1.1}$', ' $1.1$', ' $2$',
            ' $3$', ' $4$', ' $5$'
        ]

        # combine them and build a new colormap
        colors1 = plt.cm.Blues(np.linspace(0.15, 0.998, (num_col) * 4))[::-1]
        colorsgr = np.tile(np.array([0.8, 0.8, 0.8, 1]),
                           num_gr).reshape(num_gr, -1)
        colors2 = plt.cm.Reds(np.linspace(0.15, 0.998, (num_col) * 4))

        # combine them and build a new colormap
        colors1g = np.vstack((colors1, colorsgr))
        colors = np.vstack((colors1g, colors2))
        cm_rat_c = mpl.colors.LinearSegmentedColormap.from_list(
            'my_colormap', colors)
        norm = mpl.colors.BoundaryNorm(boundaries=_bounds_rat,
                                       ncolors=len(_bounds_rat))

        col_map = mpl.colors.LinearSegmentedColormap.from_list(
            'my_colormap', colors)
        bo_unds = _bounds_rat
    elif isdiff:
        valrange_md = [-np.max(obs_dict[_whichVar]['plot_range_map']), np.max(obs_dict[_whichVar]['plot_range_map'])]
        _bounds_rat=np.linspace(valrange_md[0],valrange_md[1],100)
        cb_tit='Difference (Column - Row)'
        ncolo=128

        colors1 = plt.cm.plasma(np.linspace(0.3, 0.8, ncolo))#[::-1]
        colorsgr=np.tile(np.array([0.8,0.8,0.8,1]),int(ncolo/4)).reshape(int(ncolo/4),-1)
        colors2 = plt.cm.viridis(np.linspace(0.2, 0.76, ncolo))[::-1]

        colors1 = plt.cm.Blues(np.linspace(0.15, 0.998, ncolo))[::-1]
        colorsgr=np.tile(np.array([0.8,0.8,0.8,1]),int(ncolo/4)).reshape(int(ncolo/4),-1)
        colors2 = plt.cm.Reds(np.linspace(0.15, 0.998, ncolo))#[::-1]


        # combine them and build a new colormap
        colors1g = np.vstack((colors1, colorsgr))
        colors = np.vstack((colors1g,colors2))
        col_map = mpl.colors.LinearSegmentedColormap.from_list('my_colormap', colors)

        cb_ticks = np.linspace(valrange_md[0], valrange_md[1],10)

        cb_labels = cb_ticks
        bo_unds = _bounds_rat

    else:
        valrange_md = obs_dict[_whichVar]['plot_range_map']
        cbName = obs_dict[_whichVar]['color_scale']
        cb_ticks = []
        cb_labels = []
        if _whichVar == 'tau_c':
            bo_unds = np.concatenate(
                ([1], np.linspace(8, 16, num=10)[:-1],
                 np.linspace(16, 32, num=10)[:-1], np.linspace(32, 64,
                                                               num=10)[:-1],
                 np.linspace(64, 128,
                             num=10)[:-1], np.linspace(128, 256, num=10)[:-1],
                 np.linspace(256, 1000, num=2, endpoint=True)))
            cb_ticks = np.array([1, 8, 16, 32, 64, 128, 256])
            color_list = ptool.get_colomap(cbName, bo_unds, lowp=0., hip=1.0)
            col_map = mpl.colors.ListedColormap(color_list)

        if _whichVar == 'gpp':
            bo_unds = np.linspace(min(valrange_md), max(valrange_md), 100)
            color_list = ptool.get_colomap(cbName, bo_unds, lowp=0., hip=1)
            col_map = mpl.colors.ListedColormap(color_list)
            cb_ticks = np.linspace(min(valrange_md), max(valrange_md), 8)
            cb_ticks = np.array([0.6, 1.2, 1.8, 2.4, 3, 3.6, 4.2])
        if _whichVar == 'c_total':
            bo_unds = np.linspace(min(valrange_md), max(valrange_md), 100)
            color_list = ptool.get_colomap(cbName, bo_unds, lowp=0., hip=1)
            cb_ticks = np.array([10, 20, 30, 40, 50, 60, 70])
            col_map = mpl.colors.ListedColormap(color_list)
        if _whichVar == 'c_soil':
            bo_unds = np.linspace(min(valrange_md), max(valrange_md), 100)
            color_list = ptool.get_colomap(cbName, bo_unds, lowp=0., hip=1)
            cb_ticks = np.array([0, 5, 10, 20, 30, 40, 50])
            col_map = mpl.colors.ListedColormap(color_list)
        if _whichVar == 'c_veg':
            bo_unds = np.linspace(min(valrange_md), max(valrange_md), 100)
            color_list = ptool.get_colomap(cbName, bo_unds, lowp=0., hip=1)
            cb_ticks = np.array([0, 5, 10, 15, 20, 25, 30])
            col_map = mpl.colors.ListedColormap(color_list)

    return bo_unds, col_map, cb_ticks, cb_labels


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


def _apply_common_mask_d(_data):
    in_d = 0
    for _var, _dat in _data.items():
        _tmp = np.ones_like(_dat)
        _tmp[_dat == -9999.] = 0
        _tmp_inv_mask = np.ma.masked_invalid(_dat).mask
        _tmp[_tmp_inv_mask] = 0
        if in_d == 0:
            dat_mask = _tmp
        else:
            dat_mask = dat_mask * _tmp
        in_d = in_d + 1
    o_data = {}
    for _var, _dat in _data.items():
        o_data[_var] = np.ma.masked_invalid(
            _apply_a_mask(_dat.astype(np.float), dat_mask))
    return o_data


def _compress_invalid(_dat):
    odat = np.ma.masked_invalid(_dat).compressed()
    return odat


def _draw_legend_aridity(_co_settings, loc_a='best', ax_fs=None, is_3d=False):

    if ax_fs is None:
        ax_fs = _co_settings['fig_settings']['ax_fs'] * 0.8
    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D, proj3d
        f = lambda x, y, z: proj3d.proj_transform(x, y, z,
                                                  plt.gca().get_proj())[:2]
        leg = plt.gca().legend(
            loc="lower left",
            bbox_to_anchor=f(loc_a[0], loc_a[1], loc_a[2]),
            bbox_transform=plt.gca().transData,
            ncol=4,
            fontsize=ax_fs,
            fancybox=True,
            columnspacing=1.605,
            #  labelspacing=0.02,
            handletextpad=0.0,
            handlelength=0,
            markerscale=0)
    else:
        leg = plt.legend(
            loc=loc_a,
            ncol=4,
            fontsize=ax_fs,
            fancybox=True,
            columnspacing=1.605,
            # labelspacing=0.02,
            handletextpad=0.0,
            handlelength=0,
            markerscale=0)
    leg.get_frame().set_linewidth(0)
    leg.get_frame().set_facecolor('#dddddd')
    leg.legendPatch.set_alpha(0.45)
    texts = leg.get_texts()
    color_list = _co_settings[_co_settings['fig_settings']
                              ['eval_region']]['colors']
    if ax_fs is None:
        ax_fs = _co_settings['fig_settings']['ax_fs']
    for ttI in range(len(texts)):
        tt = texts[ttI]
        tt.set_color(color_list[ttI])
    return leg


def _draw_legend_models(_co_settings,
                        loc_a='best',
                        ax_fs=None,
                        inc_mme=True,
                        inc_obs=True,
                        is_3d=False):
    if ax_fs is None:
        ax_fs = _co_settings['fig_settings']['ax_fs'] * 0.8
    if is_3d:
        from mpl_toolkits.mplot3d import Axes3D, proj3d
        f = lambda x, y, z: proj3d.proj_transform(x, y, z,
                                                  plt.gca().get_proj())[:2]
        leg = plt.gca().legend(loc="lower left",
                               bbox_to_anchor=f(loc_a[0], loc_a[1], loc_a[2]),
                               bbox_transform=plt.gca().transData,
                               fontsize=ax_fs,
                               ncol=9,
                               columnspacing=0.085,
                               fancybox=True,
                               handlelength=0,
                               markerscale=0)
    else:
        leg = plt.legend(loc=loc_a,
                         fontsize=ax_fs,
                         ncol=9,
                         columnspacing=0.085,
                         fancybox=True,
                         handlelength=0,
                         markerscale=0)

    leg.get_frame().set_linewidth(0)
    leg.get_frame().set_facecolor('#dddddd')
    leg.legendPatch.set_alpha(0.45)
    texts = leg.get_texts()
    if inc_mme:
        texts[len(texts) - 1].set_color('blue')
        ltext = len(texts) - 1
    else:
        ltext = len(texts)
    mod_colors = _co_settings['model']['colors']
    lmin = 1
    if inc_obs is False:
        mod_colors = _co_settings['model']['colors']
        lmin = 0
    models = _co_settings['model']['names']
    for ttI in range(lmin, ltext):
        tt = texts[ttI]
        col = mod_colors[models[ttI]]
        tt.set_color(col)
    return leg


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


def _fLin(B, x):
    '''Linear function y = m*x + b'''
    return B[0] * x + B[1]


def _fit_odr(x, y):
    # fit orthogonal (total) least squares
    xmask = ~(np.ma.masked_invalid(x).mask)
    ymask = ~(np.ma.masked_invalid(y).mask)
    xymask = xmask * ymask
    xdat = x[xymask]
    ydat = y[xymask]
    print('ODR1', xdat.max(), xdat.min())
    print('ODR2', ydat.max(), ydat.min())
    linearF = scodr.Model(_fLin)
    mydata = scodr.Data(xdat, ydat)
    myodr = scodr.ODR(mydata, linearF, beta0=[1, 0], ifixb=[1, 0], maxit=100)
    myoutput = myodr.run()
    return myoutput.beta[0], myoutput.beta[1]


def _density_estimation(m1, m2):
    xmin = np.nanpercentile(m1, 1)
    xmax = np.nanpercentile(m1, 99)
    ymin = np.nanpercentile(m2, 1)
    ymax = np.nanpercentile(m2, 99)
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])
    values = np.vstack([m1, m2])
    kernel = scst.gaussian_kde(values, bw_method='silverman')
    Z = np.reshape(kernel(positions).T, X.shape)
    return X, Y, Z


#-------------------------------------------
# Zonal correlations
#-------------------------------------------


def _zonal_correlation(data_dict, var_dict, zonal_set):
    _lats = zonal_set['lats']
    bandsize_corr = zonal_set['bandsize_corr']
    _latint = abs(_lats[1] - _lats[0])
    _z_corr = zonal_set["z_corr"]
    windowSize = np.int(np.round(bandsize_corr / (_latint * 2)))

    __dat = np.ones((len(_lats))) * np.nan
    print(var_dict['x'], var_dict['y'], var_dict['z'])

    data_dict_com = _apply_common_mask_d(data_dict)
    in_d = 0
    for _var, _dat in data_dict_com.items():
        _tmp = ~np.ma.masked_invalid(_dat).mask
        if in_d == 0:
            v_mask = _tmp
        else:
            v_mask = v_mask * _tmp
        in_d = in_d + 1
        data_dict_com[_var] = np.ma.masked_invalid(_dat)
    nvalids = np.sum(v_mask, axis=1)
    lat_indices = np.argwhere(nvalids > 0)
    for _latInd in lat_indices:
        lat_dict = {}
        li = _latInd[0]
        istart = max(0, li - windowSize)
        iend = min(np.size(_lats), li + windowSize + 1)
        for _var, _dat in data_dict_com.items():
            lat_dict[_var] = _dat[istart:iend, :].compressed()
        df = pd.DataFrame(lat_dict)
        df = df[(np.abs(scst.zscore(df)) < _z_corr).all(axis=1)]
        r12_3 = _partial_corr(df,
                              x_var=var_dict['x'],
                              y_var=var_dict['y'],
                              z_var=var_dict['z'],
                              _method=zonal_set['method_corr'],
                              _p_thres=zonal_set['p_thres'])
        __dat[li] = r12_3
    return __dat


def _zonal_correlation_percentiles(data_dict_full, dict_vars, zonal_set):
    perc_range = zonal_set['perc_range']
    nLats = len(zonal_set['lats'])
    x_dat = data_dict_full[dict_vars['x']]
    y_dat = data_dict_full[dict_vars['y']]
    if x_dat.ndim > 2:
        nMemb_x = len(x_dat)
    else:
        nMemb_x = 1
    if y_dat.ndim > 2:
        nMemb_y = len(y_dat)
    else:
        nMemb_y = 1
    if nMemb_x != 1 and nMemb_y != 1 and nMemb_x != nMemb_y:
        nMemb = nMemb_x * nMemb_y
        full_comb = True
    else:
        nMemb = nMemb_x
        full_comb = False
    print(nMemb_x, nMemb_y, dict_vars['x'], full_comb)
    # kera
    # nMemb = 2
    r_x_y_z_full = np.ones((nLats, nMemb)) * np.nan
    data_dict_zone = {}
    if full_comb:
        memb = 0
        for memb_x in range(nMemb_x):
            print('corr obs, memb_x: ', memb_x)
            for memb_y in range(nMemb_y):
                print('corr obs, memb_y: ', memb_y)
                for _var, _dat in data_dict_full.items():
                    # print (_var, dict_vars.items())
                    if _dat.ndim > 2:
                        if _var == dict_vars['x']:
                            data_dict_zone[_var] = _dat[memb_x]
                        else:
                            data_dict_zone[_var] = _dat[memb_y]
                    else:
                        data_dict_zone[_var] = _dat
                corr_zone = _zonal_correlation(data_dict_zone, dict_vars, zonal_set)
                r_x_y_z_full[:, memb] = corr_zone
                memb = memb + 1
    else:
        for memb in range(nMemb):
            print('corr obs, memb: ', memb, 'out of', nMemb)
            for _var, _dat in data_dict_full.items():
                if _dat.ndim > 2:
                    data_dict_zone[_var] = _dat[memb]
                else:
                    data_dict_zone[_var] = _dat
            corr_zone = _zonal_correlation(data_dict_zone, dict_vars, zonal_set)
            r_x_y_z_full[:, memb] = corr_zone
            print('--------------------')

    r_x_y_z_full = np.clip(r_x_y_z_full, -0.99, 0.99)

    r_x_y_z_5 = np.nanpercentile(r_x_y_z_full, perc_range[0], axis=1)
    r_x_y_z_95 = np.nanpercentile(r_x_y_z_full, perc_range[1], axis=1)

    zs_x_y_z_full = _fisher_z(r_x_y_z_full)
    zs_x_y_z_full[np.isinf(zs_x_y_z_full)] = np.nan
    zs_x_y_z_mean = np.nanmean(zs_x_y_z_full, axis=1)
    r_x_y_z_mean = _inverse_fisher_z(zs_x_y_z_mean)

    return np.column_stack((r_x_y_z_5, r_x_y_z_mean, r_x_y_z_95))


def _fisher_z(_rdat):
    _zdat = 0.5 * (np.log(1 + _rdat) - np.log(1 - _rdat))
    return _zdat


def _inverse_fisher_z(_zdat):
    _rdat = (np.exp(2 * _zdat) - 1) / (np.exp(2 * _zdat) + 1)
    return _rdat


def _partial_corr(df,
                  x_var='x',
                  y_var='y',
                  z_var=[],
                  _method='pearson',
                  _p_thres=0.05):
    if len(z_var) == 0:
        pcorr = pg.corr(df.loc[:, x_var].values,
                        df.loc[:, y_var].values,
                        method=_method)
    else:
        pcorr = pg.partial_corr(data=df,
                                x=x_var,
                                y=y_var,
                                covar=z_var,
                                method=_method)
    r = pcorr.iloc[0]['r']
    p = pcorr.iloc[0]['p-val']
    if p > _p_thres:
        r = np.nan
    return r


def _plot_mm_norm_mean_r(_sp, allmodels_r, zonal_set, fig_set):
    allmodels_z = _fisher_z(allmodels_r)
    allmodels_z[np.isinf(allmodels_r)] = np.nan

    zmm_ens = np.nanmean(allmodels_z, axis=1)
    zmm_ens_std = np.nanstd(allmodels_z, axis=1)
    r_mmod = _inverse_fisher_z(zmm_ens)
    r_mmod_std_low = _inverse_fisher_z(zmm_ens - zmm_ens_std)
    r_mmod_std_hi = _inverse_fisher_z(zmm_ens + zmm_ens_std)

    _sp.plot(np.ma.masked_equal(r_mmod, np.nan),
             zonal_set['lats'],
             color='blue',
             ls='-',
             lw=fig_set['lwMainLine'],
             label='Model Ensemble',
             zorder=9)

    _sp.fill_betweenx(zonal_set['lats'],
                      np.ma.masked_equal(r_mmod_std_low, np.nan),
                      np.ma.masked_equal(r_mmod_std_hi, np.nan),
                      facecolor='#42d4f4',
                      alpha=0.25)
    return


def _get_sel_data_dict(full_data_dict, sel_var_info, model):
    sel_dat = {}
    sel_dat[sel_var_info['x']] = full_data_dict[sel_var_info['x']][model]
    sel_dat[sel_var_info['y']] = full_data_dict[sel_var_info['y']][model]
    for _zv in sel_var_info['z']:
        sel_dat[_zv] = full_data_dict[_zv][model]
    return sel_dat


def _plot_correlations(_sp, full_dat, var_info, zonal_set, fig_set, co_set):
    # get the correlation percentiles for observed tau
    models = co_set['model']['names']
    nmodels = len(models)
    lats = zonal_set['lats']
    sel_dat = _get_sel_data_dict(full_dat, var_info, 'obs')
    r_obs = _zonal_correlation_percentiles(sel_dat, var_info, zonal_set)
    # plotting observed tau temperature correlation
    _sp.plot(r_obs[:, 1],
             lats,
             color='k',
             lw=fig_set['lwMainLine'],
             label='Obs-based',
             zorder=10)
    _sp.fill_betweenx(lats,
                      r_obs[:, 0],
                      r_obs[:, 2],
                      facecolor='grey',
                      alpha=0.40)

    # correlation for each model as well as multimodel normalized mean r
    # define arrays
    r_allmodels = np.zeros((len(lats), nmodels - 1))
    for row_m in range(1, nmodels):
        row_mod = co_set['model']['names'][row_m]
        # get the zonal correlation for a model
        sel_dat = _get_sel_data_dict(full_dat, var_info, row_mod)
        print('corr: model: ', row_mod)
        mzone = _zonal_correlation(sel_dat, var_info, zonal_set)
        r_mod = mzone
        r_allmodels[:, row_m - 1] = r_mod

        # plot the model correlation betwen tau and tas
        _sp.plot(np.ma.masked_equal(r_mod, np.nan),
                 lats,
                 color=fig_set['mod_colors'][row_mod],
                 lw=fig_set['lwModLine'],
                 label=co_set['model_dict'][row_mod]['model_name'])
        print('--------------------')

    ### calculate and plot the normalized mean r of all models
    _plot_mm_norm_mean_r(_sp, r_allmodels, zonal_set, fig_set)

    plt.xlim(-1, 1)
    plt.ylim(-60, 85)
    plt.axhline(y=0, lw=0.48, color='grey')
    plt.axvline(x=0, lw=0.48, color='grey')
    ptool.rem_axLine(['top', 'right'])

    return
