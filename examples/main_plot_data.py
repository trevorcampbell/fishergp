import numpy as np
from utils import kl_gaussian, gen_synthetic, gen_linear, gen_from_file, standardize
import bokeh.plotting as bkp
import bokeh.layouts as bkl
import bokeh.palettes 
import bokeh.io as bki
import pickle as pk
import time
import os
import GPy


##create dataset generators
dnms = ['synthetic', 'abalone', 'airfoil', 'wine', 'ccpp', 'delays10k']
n_pretrain = [100, 300, 100, 300, 700, 800]
datasets = [lambda s : gen_synthetic(1000, 1000, s),
            lambda s : gen_from_file('abalone', 3177, 1000, s),
            lambda s : gen_from_file('airfoil', 1103, 400, s),
            lambda s : gen_from_file('wine', 3898, 1000, s),
            lambda s : gen_from_file('ccpp', 7568, 2000, s),
            lambda s : gen_from_file('delays10k', 8000, 2000, s)]

dnms = ['synthetic']
datasets = [lambda s : gen_synthetic(1000, 1000, s)]
            


d_seed =1
for k in range(len(datasets)):
  res = np.array([])
  dnm = dnms[k] 
  dst = datasets[k]
  #load/standardize data
  X, Y, Xt, Yt = dst(d_seed)
  #note that this function modifies in-place; the output isn't used
  #Xmu, XZ, Ymu, YZ = standardize(X, Y, Xt, Yt)

  plots = []
  for axis in range(X.shape[1]):
    xorder = np.argsort(X[:, axis])
    X = X[xorder, :]
    Y = Y[xorder, :]

    xorder = np.argsort(Xt[:, axis])
    Xt = Xt[xorder, :]
    Yt = Yt[xorder, :]

    fres = np.load('results/'+dnm+'_'+str(d_seed)+'_full_results.npz')

    mu_pred_full = fres['mu_pred_full'].flatten()[xorder]
    sig_pred_full = fres['sig_pred_full'].flatten()[xorder]
    #pred_err_full = fres['pred_err_full'].flatten()[xorder]


    fig = bkp.figure()
    fig.circle(X[:, axis], Y[:, 0])
    fig.line(Xt.flatten(), mu_pred_full, line_color='blue')
    fig.line(Xt.flatten(), mu_pred_full+sig_pred_full, line_color='red')
    fig.line(Xt.flatten(), mu_pred_full-sig_pred_full, line_color='red')
    plots.append(fig)
  bkp.output_file(dnms[k] + '.html')
  bki.save(bkl.gridplot([plots]))









