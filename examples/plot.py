import numpy as np
import bokeh.plotting as bkp
import bokeh.layouts as bkl
import bokeh.palettes
from bokeh.io.export import export_png
from bokeh.models import FuncTickFormatter
from bokeh.models.tickers import FixedTicker
import os
import sys

legend = True
pngs = False

logFmtr = FuncTickFormatter(code="""
var trns = [
'\u2070',
'\u00B9',
'\u00B2',
'\u00B3',
'\u2074',
'\u2075',
'\u2076',
'\u2077',
'\u2078',
'\u2079'];
if (tick <= 0){
  return '';
}
var tick_power = Math.floor(Math.log10(tick));
var tick_mult = Math.pow(10, Math.log10(tick) - tick_power);
var ret = '';
if (tick_mult > 1.) {
  if (Math.abs(tick_mult - Math.round(tick_mult)) > 0.05){
    ret = tick_mult.toFixed(1) + '\u22C5';
  } else {
    ret = tick_mult.toFixed(0) +'\u22C5';
  }
}
ret += '10';
if (tick_power < 0) {
  ret += '\u207B';
  tick_power = tick_power*(-1);
}
power_digits = [];
while (tick_power > 9){
  power_digits.push( tick_power - Math.floor(tick_power/10)*10 );
  tick_power = Math.floor(tick_power/10);
}

power_digits.push(tick_power);
for (i = power_digits.length-1; i >= 0; i--){
  ret += trns[power_digits[i]];
}
return ret;
""")


def errorbar(fig, x, y, xerr=None, yerr=None, color='red',
             point_kwargs={}, error_kwargs={}):
  if xerr is not None:
      x_err_x = []
      x_err_y = []
      for px, py, err in zip(x, y, xerr):
          x_err_x.append((px - err, px + err))
          x_err_y.append((py, py))
      fig.multi_line(x_err_x, x_err_y, color=color, **error_kwargs)

  if yerr is not None:
      y_err_x = []
      y_err_y = []
      for px, py, err in zip(x, y, yerr):
          y_err_x.append((px, px))
          y_err_y.append((py - err, py + err))
      fig.multi_line(y_err_x, y_err_y, color=color, **error_kwargs)

anms_dict = {'subset_regressors' : 'SoR', 'variational_inducing' : 'VFE', 'fisher_inducing' : 'pF-DTC', 'full' : 'exact', 'subsample' : 'subsample'}

dset_sizes = dict(synthetic=1000, delays10k=8000, ccpp=7568,
                  abalone=3177, airfoil=1103, wine=3898)

#dnms = ['synthetic', 'delays10k', 'abalone', 'airfoil'] #, 'kin8nm']
#dnms = ['synthetic', 'delays10k', 'abalone', 'airfoil', 'wine', 'ccpp']
dnms = [sys.argv[1]]
d_seed = 1

font_size = '54pt'

colors=['blue', 'red', 'green', 'orange', 'purple', 'cyan']
for k in range(len(dnms)):
    res = np.load('results/'+dnms[k]+'_'+str(d_seed)+'_results.npz')
    anms=res['anms']
    n_inducing=res['n_inducing']
    pretrain_cputs=res['pretrain_cputs']
    train_cputs=res['train_cputs']
    pred_cputs=res['pred_cputs']
    pred_errs=res['pred_errs']
    post_mean_errs=res['post_mean_errs']
    post_sig_errs=res['post_sig_errs']
    lsc_errs = res['lsc_errs']
    kvar_errs = res['kvar_errs']
    lvar_errs = res['lvar_errs']

    res = np.load('results/'+dnms[k]+'_'+str(d_seed)+'_full_results.npz')
    mu_pred_full = res['mu_pred_full']
    sig_pred_full = res['sig_pred_full']
    pred_err_full = res['pred_err_full']
    pred_cput_full = res['pred_cput_full']
    train_cput_full = res['train_cput_full']

    f_pe_vs_ni = bkp.figure(plot_width=1250, plot_height=1250, y_axis_type='log', y_axis_label='RMSE Test Prediction Error', x_axis_type='log', x_axis_label='# Inducing Pts')
    f_pme_vs_ni = bkp.figure(plot_width=1250, plot_height=1250, y_axis_type='log', y_axis_label='RMSE Posterior Mean Error', x_axis_type='log', x_axis_label='# Inducing Pts')
    f_pse_vs_ni = bkp.figure(plot_width=1250, plot_height=1250, y_axis_type='log', y_axis_label='RMSE Posterior StdDev Error', x_axis_type='log', x_axis_label='# Inducing Pts')
    f_pme_best_vs_ni = bkp.figure(plot_width=1250, plot_height=1250, y_axis_type='log', y_axis_label='RMSE Posterior Mean Error (best)', x_axis_type='log', x_axis_label='# Inducing Pts')
    f_pse_best_vs_ni = bkp.figure(plot_width=1250, plot_height=1250, y_axis_type='log', y_axis_label='RMSE Posterior StdDev Error (best)', x_axis_type='log', x_axis_label='# Inducing Pts')

    #f_pe_vs_cput = bkp.figure(plot_width=1250, plot_height=1250, y_axis_type='log', y_axis_label='RMSE Test Prediction Error', x_axis_type='log', x_axis_label='# Inducing Pts')
    #f_pme_vs_cput = bkp.figure(plot_width=1250, plot_height=1250, y_axis_type='log', y_axis_label='RMSE Posterior Mean Error', x_axis_type='log', x_axis_label='# Inducing Pts')
    #f_pse_vs_cput = bkp.figure(plot_width=1250, plot_height=1250, y_axis_type='log', y_axis_label='RMSE Posterior StdDev Error', x_axis_type='log',  x_axis_label='# Inducing Pts')

    hypchg_vs_ni = bkp.figure(plot_width=1250, plot_height=1250, y_axis_type='log', y_axis_label='Relative HyperParam Change', x_axis_type='log', x_axis_label='# Inducing Pts')

    for f in [f_pe_vs_ni, f_pme_vs_ni, f_pme_best_vs_ni, f_pse_vs_ni, f_pse_best_vs_ni, hypchg_vs_ni]:
        f.xaxis.axis_label_text_font_size= font_size
        f.xaxis.major_label_text_font_size= font_size
        f.xaxis.formatter = logFmtr
        f.yaxis.axis_label_text_font_size= font_size
        f.yaxis.major_label_text_font_size= font_size
        f.yaxis.formatter = logFmtr
        f.title.text_font_size = font_size



    if legend:
        f_pe_vs_ni.line(n_inducing, pred_err_full*np.ones(n_inducing.shape[0]), legend=anms_dict['full'], line_width=7, line_color='black')
    else:
        f_pe_vs_ni.line(n_inducing, pred_err_full*np.ones(n_inducing.shape[0]), line_width=7, line_color='black')

    for j in range(len(anms)):
        if anms[j] == b'linear':
            continue
        pe_mean = pred_errs[j, :, :].mean(axis=1)
        pe_std = pred_errs[j, :, :].std(axis=1)

        cput_mean = train_cputs[j, :, :].mean(axis=0)
        cput_std = train_cputs[j, :, :].std(axis=0)

        if legend:
            anm_legend = anms_dict[anms[j].decode("utf-8")]
        else:
            anm_legend = None

        if anms[j] == b'subsample':
            eff_num_inducing = n_inducing**1.5 / np.sqrt(dset_sizes[dnms[k]])
        else:
            eff_num_inducing = n_inducing

        f_pe_vs_ni.line(eff_num_inducing, pe_mean, legend=anm_legend, line_width=7, line_color=colors[j])
        #f_pe_vs_cput.line(cput_mean, pe_mean, legend=anm_legend, line_width=7, line_color=colors[j])

        pme_mean = post_mean_errs[j, :, :].mean(axis=1)
        pme_std = post_mean_errs[j, :, :].std(axis=1)
        pme_best = post_mean_errs[j, :, :].min(axis=1)
        pse_mean = post_sig_errs[j, :, :].mean(axis=1)
        pse_std = post_sig_errs[j, :, :].std(axis=1)
        pse_best = post_sig_errs[j, :, :].min(axis=1)

        f_pme_vs_ni.line(eff_num_inducing[:-1], pme_mean[:-1], legend=anm_legend, line_width=7, line_color=colors[j])
        f_pse_vs_ni.line(eff_num_inducing[:-1], pse_mean[:-1], legend=anm_legend, line_width=7, line_color=colors[j])
        f_pme_best_vs_ni.line(eff_num_inducing[:-1], pme_best[:-1], legend=anm_legend, line_width=7, line_color=colors[j])
        f_pse_best_vs_ni.line(eff_num_inducing[:-1], pse_best[:-1], legend=anm_legend, line_width=7, line_color=colors[j])
        #f_pme_vs_cput.line(cput_mean, pme_mean, legend=anm_legend, line_width=7, line_color=colors[j])
        #f_pse_vs_cput.line(cput_mean, pse_mean, legend=anm_legend, line_width=7, line_color=colors[j])

        if anms[j] == 'variational_inducing':
            hypchg_vs_ni.line(n_inducing, lsc_errs[0, :, :].mean(axis=1), legend=anm_legend, line_width=7, line_color=colors[j])
            hypchg_vs_ni.line(n_inducing, kvar_errs[0, :, :].mean(axis=1), legend=anm_legend, line_width=7, line_color=colors[j])
            hypchg_vs_ni.line(n_inducing, lvar_errs[0, :, :].mean(axis=1), legend=anm_legend, line_width=7, line_color=colors[j])
        if anms[j] == 'fisher_inducing':
            hypchg_vs_ni.line(n_inducing, lsc_errs[1, :, :].mean(axis=1), legend=anm_legend, line_width=7, line_color=colors[j])
            hypchg_vs_ni.line(n_inducing, kvar_errs[1, :, :].mean(axis=1), legend=anm_legend, line_width=7, line_color=colors[j])
            hypchg_vs_ni.line(n_inducing, lvar_errs[1, :, :].mean(axis=1), legend=anm_legend, line_width=7, line_color=colors[j])

    for f in [f_pe_vs_ni, f_pme_vs_ni, f_pse_vs_ni, hypchg_vs_ni]:
        f.legend.label_text_font_size= font_size
        f.legend.glyph_width=100
        f.legend.glyph_height=40
        f.legend.spacing=20

    for f in [f_pme_vs_ni, f_pse_vs_ni]:
        f.legend.location = 'bottom_left'

    if dnms[k] == 'delays10k':
        f_pme_vs_ni.yaxis.ticker = FixedTicker(ticks=[0.1, 0.4, 0.7])
    elif dnms[k] == 'airfoil':
        f_pe_vs_ni.yaxis.ticker = FixedTicker(ticks=[0.3, 0.6, 0.9])
        f_pse_vs_ni.yaxis.ticker = FixedTicker(ticks=[0.1, 0.5, 1])


    #bkp.show(bkl.gridplot([[f_pe_vs_ni, f_pme_vs_ni, f_pse_vs_ni], [f_pe_vs_cput, f_pme_vs_cput, f_pse_vs_cput]]))
    figs = [f_pe_vs_ni, f_pme_vs_ni, f_pme_best_vs_ni, f_pse_vs_ni, f_pse_best_vs_ni,  hypchg_vs_ni]
    if pngs:
        for i, f in enumerate(figs):
            export_png(f, 'figures/%s%d.png' % (dnms[k], i+1))
    else:
        bkp.output_file(dnms[k] + '.html')
        bkp.show(bkl.gridplot([figs]))
