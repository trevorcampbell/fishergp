import numpy as np
import cPickle as cpk
from gen_data import gen_synthetic, gen_linear, gen_from_file, standardize
from gpcoreset import SubsampleGP, SubsetRegressorsGP, NystromGP, InducingGP, VariationalGP, Linear, optimize_hyperparameters
import bokeh.plotting as bkp
import bokeh.layouts as bkl
import bokeh.palettes 
import time
import os

##create dataset generators
d_seed = 1
dnms = ['synthetic', 'airfoil', 'abalone', 'delays10k'] #, 'kin8nm']
n_pretrain = [100, 100, 300, 600, 800]
datasets = [lambda s : gen_synthetic(1000, 1000, s),
            lambda s : gen_from_file('airfoil', 1103, 400, s),
            lambda s : gen_from_file('abalone', 3177, 1000, s),
            lambda s : gen_from_file('delays10k', 8000, 2000, s)]
            #lambda s : gen_from_file('kin8nm', 6192, 2000, s),

dnms = ['airfoil']
n_pretrain = [100]
datasets = [lambda s : gen_from_file('airfoil', 1103, 400, s)]


dnms = ['synthetic']
n_pretrain = [100]
datasets = [lambda s : gen_synthetic(1000, 1000, s)]




n_trials = 10
n_inducing = 200
#n_inducing = np.unique(np.logspace(0, 2, 5, dtype=np.int))
n_inducing_hyperopt = 200

#run trials, loading each dataset
for k in range(len(datasets)):
  dnm = dnms[k] 
  dst = datasets[k]
  print('Dataset: '+dnm)
  #load/standardize data
  print('Loading data...')
  X, Y, Xt, Yt = dst(d_seed)
  print('Standardizing...')
  Xmu, XZ, Ymu, YZ = standardize(X, Y, Xt, Yt)

  #load/optimize kernel parameters
  krnprm_fn = 'results/'+dnm+'_krn_'+str(d_seed)+'.npy'
  if os.path.exists(krnprm_fn):
    print('Loading parameters')
    krnprms = np.load(krnprm_fn)
    likelihood_variance = krnprms[0]
    kernel_variance = krnprms[1]
    sq_length_scales = krnprms[2:]
  else:
    print('No saved parameters found. Optimizing...')
    sq_length_scales, kernel_variance, likelihood_variance = optimize_hyperparameters(X, Y, n_inducing_hyperopt)
    np.save(krnprm_fn, np.hstack((likelihood_variance, kernel_variance, sq_length_scales)))

  print 'Lvar: ' + str(likelihood_variance)
  print 'Kvar: ' + str(kernel_variance)
  print 'lengths: ' + str(sq_length_scales)

  #create the model objects
  print('Creating models')
  gp = SubsampleGP(X, Y, sq_length_scales, kernel_variance, likelihood_variance)
  igp = InducingGP(X, Y, sq_length_scales, kernel_variance, likelihood_variance)

  print('Training full GP')
  full_results_fn = 'results/'+dnm+'_'+str(d_seed)+'_full_results.npz'
  if os.path.exists(full_results_fn):
    print('Found full GP results. Loading...')
    fres = np.load(full_results_fn)
    mu_pred_full = fres['mu_pred_full']
    sig_pred_full = fres['sig_pred_full']
    pred_err_full = fres['pred_err_full']
    pred_cput_full = fres['pred_cput_full']
    train_cput_full = fres['train_cput_full']
  else:
    print('No Full GP results found. Training...')
    t0 = time.time()
    gp.train(np.arange(X.shape[0]))
    train_cput_full = time.time()-t0
    t0 = time.time()
    mu_pred_full, sig_pred_full = gp.predict_f(Xt, cov_type='diag')
    sig_pred_full = np.sqrt(sig_pred_full)
    pred_cput_full = time.time()-t0 
    pred_err_full = np.sqrt(((mu_pred_full - Yt)**2).mean())
    np.savez(full_results_fn, mu_pred_full=mu_pred_full, sig_pred_full=sig_pred_full, 
                             train_cput_full=train_cput_full, pred_cput_full=pred_cput_full, pred_err_full=pred_err_full)
     
  #run algs
  print('Running inference')
  mu_err_traces = []
  sig_err_traces = []
  obj_traces = []
  for t in range(n_trials):
    print('Trial ' + str(t+1)+'/'+str(n_trials))

    #get index subsets for training
    idcs = np.arange(X.shape[0])
    np.random.shuffle(idcs)
    subsample_idcs = idcs[:n_inducing].copy()
    np.random.shuffle(idcs)
    pretrain_subsample_idcs = idcs[:n_pretrain[k]].copy()
    
    #train with tracing
    igp.pretrain(pretrain_subsample_idcs)
    igp.train(subsample_idcs, Xt=Xt, mu_full=mu_pred_full, sig_full=sig_pred_full)
    #extract output
    mu_err_traces.append(np.array(igp.mu_err_trace))
    sig_err_traces.append(np.array(igp.sig_err_trace))
    obj_traces.append(np.array(igp.obj_trace))
          
  f = open('results/'+dnm+'_'+str(d_seed)+'_igp_traces.cpk', 'wb')  
  cpk.dump((mu_err_traces, sig_err_traces, obj_traces), f)
  f.close()

