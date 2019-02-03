import numpy as np
import cPickle as cpk
import bokeh.plotting as bkp
import bokeh.layouts as bkl
import bokeh.palettes
from bokeh.models import FuncTickFormatter
from bokeh.models.tickers import FixedTicker
import os

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

dnms = ['synthetic', 'delays10k', 'abalone', 'airfoil'] #, 'kin8nm']
dnms = ['airfoil']
d_seed = 1

font_size = '60pt'

f_pme_vs_obj = bkp.figure(plot_width=1250, plot_height=800, y_axis_type='log', y_axis_label='pF Divergence', x_axis_type='log', x_axis_label='RMS Posterior Mean Error')
f_pse_vs_obj = bkp.figure(plot_width=1250, plot_height=800, y_axis_type='log', y_axis_label='pF Divergence', x_axis_type='log', x_axis_label='RMS Posterior StdDev Error')
f_iter_vs_obj = bkp.figure(plot_width=1250, plot_height=800, y_axis_type='log', y_axis_label='pF Divergence', x_axis_type='log', x_axis_label='Iteration')

for f in [f_pme_vs_obj, f_pse_vs_obj, f_iter_vs_obj]:
  f.xaxis.axis_label_text_font_size= font_size
  f.xaxis.major_label_text_font_size= font_size
  f.xaxis.formatter = logFmtr
  f.yaxis.axis_label_text_font_size= font_size
  f.yaxis.major_label_text_font_size= font_size
  f.yaxis.formatter = logFmtr
  f.title.text_font_size = font_size

f_pme_vs_obj.xaxis.ticker = FixedTicker(ticks=[0.05, 0.1, 0.2])
f_pse_vs_obj.xaxis.ticker = FixedTicker(ticks=[0.3, 0.6, 0.9])



for k in range(len(dnms)):
  f = open('results/'+dnms[k]+'_'+str(d_seed)+'_igp_traces.cpk', 'rb')
  mu_err_traces, sig_err_traces, obj_traces = cpk.load(f)
  f.close()

  iters = np.arange(obj_traces[0].size) + 1
  # print mu_err_traces
  # print sig_err_traces
  # print obj_traces

  for j in range(len(mu_err_traces)):
    f_pme_vs_obj.line(mu_err_traces[j], obj_traces[j], line_width=7, line_color='purple')
    f_pse_vs_obj.line(sig_err_traces[j], obj_traces[j], line_width=7, line_color='purple')
    f_iter_vs_obj.line(iters, obj_traces[j], line_width=7, line_color='purple')

#for f in [f_pme_vs_ni, f_pse_vs_ni]:
#  f.legend.location = 'bottom_left'

bkp.show(bkl.gridplot([[f_pme_vs_obj, f_pse_vs_obj, f_iter_vs_obj]]))
