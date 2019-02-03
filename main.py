import numpy as np
from gen_data import gen_synthetic, gen_linear, gen_from_file, standardize
from gpcoreset import SubsampleGP, SubsetRegressorsGP, NystromGP, InducingGP, VariationalGP, Linear, optimize_hyperparameters, optimize_hyperparameters_post
import bokeh.plotting as bkp
import bokeh.layouts as bkl
import bokeh.palettes 
import time
import os

##create dataset generators
d_seed = 1
#dnms = ['synthetic', 'delays10k', 'abalone', 'kin8nm', 'airfoil']
#n_pretrain = [100, 100, 100, 100, ]
#datasets = [lambda s : gen_synthetic(1000, 1000, s),
#            lambda s : gen_from_file('delays10k', 1000, 1000, s),
#            lambda s : gen_from_file('abalone', 1000, 1000, s),
#            lambda s : gen_from_file('kin8nm', 1000, 1000, s),
#            lambda s : gen_from_file('airfoil', 1000, 1000, s)]


#dnms = ['synthetic', 'delays10k', 'abalone', 'kin8nm', 'airfoil']
#n_pretrain = [100, 800, 300, 600, 100]
#datasets = [lambda s : gen_synthetic(1000, 1000, s),
#            lambda s : gen_from_file('delays10k', 8000, 2000, s),
#            lambda s : gen_from_file('abalone', 3177, 1000, s),
#            lambda s : gen_from_file('kin8nm', 6192, 2000, s),
#            lambda s : gen_from_file('airfoil', 1103, 400, s)]


#dnms = ['airfoil']
#n_pretrain = [100]
#datasets = [lambda s : gen_from_file('airfoil', 1103, 400, s)]


dnms = ['wine', 'ccpp', 'kin8nm'] #, 'sarcos']
n_pretrain = [300, 700, 600] #, 4000]
datasets = [lambda s : gen_from_file('wine', 3898, 1000, s),
            lambda s : gen_from_file('ccpp', 7568, 2000, s),
            lambda s : gen_from_file('kin8nm', 6192, 2000, s)]
            #lambda s : gen_from_file('sarcos', 30484, 14000, s)]

dnms = ['synthetic']
n_pretrain = [100]
datasets = [lambda s : gen_synthetic(1000, 1000, s)]




            

n_trials = 10
n_inducing = np.unique(np.logspace(0, 3, 10, dtype=np.int))
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
  lin = Linear(X, Y)
  sgp = SubsampleGP(X, Y, sq_length_scales, kernel_variance, likelihood_variance)
  srgp = SubsetRegressorsGP(X, Y, sq_length_scales, kernel_variance, likelihood_variance)
  vgp = VariationalGP(X, Y, sq_length_scales, kernel_variance, likelihood_variance)
  igp = InducingGP(X, Y, sq_length_scales, kernel_variance, likelihood_variance)

  #store algs in a list
  anms = ['linear', 'subsample', 'subset_regressors', 'variational_inducing', 'fisher_inducing']
  algs = [lin, sgp, srgp, vgp, igp]
  
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
     
  #initialize results matrices
  pretrain_cputs = np.zeros((len(algs), n_inducing.shape[0], n_trials))
  train_cputs = np.zeros((len(algs), n_inducing.shape[0], n_trials))
  pred_cputs = np.zeros((len(algs), n_inducing.shape[0], n_trials))
  pred_errs = np.zeros((len(algs), n_inducing.shape[0], n_trials))
  post_mean_errs = np.zeros((len(algs), n_inducing.shape[0], n_trials))
  post_sig_errs = np.zeros((len(algs), n_inducing.shape[0], n_trials))
  lsc_errs = np.zeros((2, n_inducing.shape[0], n_trials))
  kvar_errs = np.zeros((2, n_inducing.shape[0], n_trials))
  lvar_errs = np.zeros((2, n_inducing.shape[0], n_trials))

  #run algs
  print('Running inference')
  for i in range(n_inducing.shape[0]):
    print('# Inducing pts: ' +str(n_inducing[i]))
    for t in range(n_trials):
      print('Trial ' + str(t+1)+'/'+str(n_trials))

      #get index subsets for training
      idcs = np.arange(X.shape[0])
      np.random.shuffle(idcs)
      subsample_idcs = idcs[:n_inducing[i]].copy()
      np.random.shuffle(idcs)
      pretrain_subsample_idcs = idcs[:n_pretrain[k]].copy()
      
      #run on each algorithm
      for j in range(len(algs)):
        print('Training ' + anms[j])
        #pretrain if required
        if getattr(algs[j], 'pretrain', None) is not None:
          t0 = time.time()
          algs[j].pretrain(pretrain_subsample_idcs)
          pretrain_cputs[j, i, t] = time.time()-t0
        #train
        t0 = time.time()
        algs[j].train(subsample_idcs)
        train_cputs[j, i, t] = time.time()-t0
        #predict
        t0 = time.time()
        mu_pred, sig_pred = algs[j].predict_f(Xt, cov_type='diag')
        sig_pred = np.sqrt(sig_pred)
        pred_cputs[j, i, t] = time.time() - t0
        #evaluate
        pred_errs[j, i, t] = np.sqrt(((mu_pred - Yt)**2).mean())
        post_mean_errs[j, i, t] = np.sqrt(((mu_pred_full-mu_pred)**2).mean())
        post_sig_errs[j, i, t] = np.sqrt(((sig_pred_full-sig_pred)**2).mean())
        if j >= len(algs)-2:
          print 'before post hyperopt: '
          print sq_length_scales, kernel_variance, likelihood_variance
          lsc, kvar, lvar = optimize_hyperparameters_post(X, Y, algs[j].X_ind, sq_length_scales, kernel_variance, likelihood_variance)
          print 'after post hyperopt: '
          print lsc, kvar, lvar
          print 'reldiffs: '
          print np.sqrt( ((lsc - sq_length_scales)**2).sum())/np.sqrt((sq_length_scales**2).sum())
          print np.fabs(kvar-kernel_variance)/np.fabs(kernel_variance)
          print np.fabs(lvar - likelihood_variance)/np.fabs(likelihood_variance)
          lsc_errs[j-(len(algs)-2), i, t] =  np.sqrt( ((lsc - sq_length_scales)**2).sum())/np.sqrt((sq_length_scales**2).sum())
          kvar_errs[j-(len(algs)-2), i, t] =  np.fabs(kvar-kernel_variance)/np.fabs(kernel_variance)
          lvar_errs[j-(len(algs)-2), i, t] =  np.fabs(lvar - likelihood_variance)/np.fabs(likelihood_variance)
          
  np.savez('results/'+dnm+'_'+str(d_seed)+'_results.npz', n_inducing=n_inducing, anms=anms,  
                                pretrain_cputs=pretrain_cputs, train_cputs=train_cputs, pred_cputs=pred_cputs, 
                                pred_errs=pred_errs, post_mean_errs=post_mean_errs, post_sig_errs=post_sig_errs,    
                                lsc_errs=lsc_errs, kvar_errs=kvar_errs, lvar_errs=lvar_errs)


