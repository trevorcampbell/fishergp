import numpy as np
from utils import kl_gaussian, gen_synthetic, gen_linear, gen_from_file, standardize
from fishergp import SubsampleGP, SubsetRegressorsGP, NystromGP, FisherGP, VariationalGP, Linear
from fishergp.utils import optimize_hyperparameters
from fishergp.kernels import GaussianKernel
import bokeh.plotting as bkp
import bokeh.layouts as bkl
import bokeh.palettes 
import pickle as pk
import time
import os
import sys
import GPy


##create dataset generators
d_seed = 1
#dnms = ['synthetic', 'delays10k', 'abalone', 'kin8nm', 'airfoil']
#n_pretrain = [100, 800, 300, 600, 100]
#datasets = [lambda s : gen_synthetic(1000, 1000, s),
#            lambda s : gen_from_file('delays10k', 8000, 2000, s),
#            lambda s : gen_from_file('abalone', 3177, 1000, s),
#            lambda s : gen_from_file('kin8nm', 6192, 2000, s),
#            lambda s : gen_from_file('airfoil', 1103, 400, s)]

dnms = ['synthetic', 'airfoil',  'ccpp','abalone','wine',  'delays10k']
n_pretrain = [100, 100, 700, 300,300,  800]
datasets = [lambda s : gen_synthetic(1000, 1000, s),
            lambda s : gen_from_file('airfoil', 1103, 400, s),
            lambda s : gen_from_file('ccpp', 7568, 2000, s),
            lambda s : gen_from_file('abalone', 3177, 1000, s),
            lambda s : gen_from_file('wine', 3898, 1000, s),
            lambda s : gen_from_file('delays10k', 8000, 2000, s)]

configs = dict(zip(dnms, zip(n_pretrain, datasets)))

#if below is True, runs a lot of extra optimization to make sure that
#fixing hyperparameters between all methods is justified;
#not needed for use in practice
check_hyper_stability = False

#if below is true, save objective values from fishergp and variationalgp
save_objs = True

#number of points to use for hyperparameter optimization
n_inducing_hyperopt = 200

full = True
try:
  dnm = sys.argv[1]
  if len(sys.argv) == 4:
    full = False
    n_inducing = int(sys.argv[2])
    n_trial = int(sys.argv[3])
  if len(sys.argv) != 2 or len(sys.argv) != 4:
    raise
except:
  print('Need to call this script as: python3 main_single.py [dataset_name] [num_inducing] [trial_num] OR python3 main_single.py [dataset_name]')

if dnm not in configs or n_inducing <= 0 or n_trial < 0:
  print('Command line arg error')
  print('Need valid dataset name, n_inducing > 0, and n_trial >= 0')
  print('dataset name: ' + dnm)
  print('n_trial: ' + str(n_trial))
  print('n_inducing: ' + str(n_inducing))
  quit()


n_pt, dst = configs[dnm]
print('Dataset: '+dnm)
#load/standardize data
print('Loading data...')
X, Y, Xt, Yt = dst(d_seed)
print('Standardizing...')
#note that this function modifies in-place; the output isn't used
Xmu, XZ, Ymu, YZ = standardize(X, Y, Xt, Yt)

#just tune kernel and full gp results
if full:
  #load/optimize kernel parameters
  krnprm_fn = 'results/'+dnm+'_krn_'+str(d_seed)+'.npy'
  if os.path.exists(krnprm_fn):
    print('Loading parameters')
    krnprms = np.load(krnprm_fn)
    likelihood_variance = krnprms[0]
    kernel_variance = krnprms[1]
    length_scales = krnprms[2:]
  else:
    print('No saved parameters found. Optimizing...')
    kern, like  = optimize_hyperparameters(X, Y, n_inducing_hyperopt,
                                      		GPy.kern.RBF(input_dim=X.shape[1], ARD=True), 
                                                GPy.likelihoods.Gaussian()
                                                )
    length_scales = kern.lengthscale
    kernel_variance = kern.variance
    likelihood_variance = like.variance
    np.save(krnprm_fn, np.hstack((likelihood_variance, kernel_variance, length_scales)))

  print('Lvar: ' + str(likelihood_variance))
  print('Kvar: ' + str(kernel_variance))
  print('lengths: ' + str(length_scales))

  kern = GaussianKernel(length_scales, kernel_variance)
  
  #create the model object
  gp = SubsampleGP(X, Y, kern, likelihood_variance)
  
  print('Training full GP')
  full_results_fn = 'results/'+dnm+'_'+str(d_seed)+'_full_results.npz'
  if os.path.exists(full_results_fn):
    print('Found full GP results. Loading...')
    fres = np.load(full_results_fn)
    mu_pred_full = fres['mu_pred_full']
    sig_pred_full = fres['sig_pred_full']
    var_pred_full = fres['var_pred_full']
    pred_err_full = fres['pred_err_full']
    pred_cput_full = fres['pred_cput_full']
    train_cput_full = fres['train_cput_full']
  else:
    print('No Full GP results found. Training...')
    t0 = time.time()
    gp.train(np.arange(X.shape[0]))
    train_cput_full = time.time()-t0
    t0 = time.time()
    mu_pred_full, var_pred_full = gp.predict_f(Xt, cov_type='full')
    sig_pred_full = np.sqrt(np.diag(var_pred_full))
    pred_cput_full = time.time()-t0 
    pred_err_full = np.sqrt(((mu_pred_full - Yt)**2).mean())
    np.savez(full_results_fn, mu_pred_full=mu_pred_full, sig_pred_full=sig_pred_full, var_pred_full=var_pred_full,
                             train_cput_full=train_cput_full, pred_cput_full=pred_cput_full, pred_err_full=pred_err_full)
else:
  #run a single trial here
  if os.path.exists(krnprm_fn):
    print('Loading parameters')
    krnprms = np.load(krnprm_fn)
    likelihood_variance = krnprms[0]
    kernel_variance = krnprms[1]
    length_scales = krnprms[2:]
  else:
    raise Exception('No kernel parameters found! rerun python3 main.py '+dnm)
  full_results_fn = 'results/'+dnm+'_'+str(d_seed)+'_full_results.npz'
  if os.path.exists(full_results_fn):
    print('Found full GP results. Loading...')
    fres = np.load(full_results_fn)
    mu_pred_full = fres['mu_pred_full']
    sig_pred_full = fres['sig_pred_full']
    var_pred_full = fres['var_pred_full']
    pred_err_full = fres['pred_err_full']
    pred_cput_full = fres['pred_cput_full']
    train_cput_full = fres['train_cput_full']
  else:
    raise Exception('No full GP results found! rerun python3 main.py '+dnm)
  results_fn = 'results/'+dnm+'_'+str(d_seed)+'_'+str(n_inducing)+'_'+str(n_trial)+'_results.npz'
  inducing_fn = 'results/'+dnm+'_'+str(d_seed)+'_'+str(n_inducing)+'_'+str(n_trial)+'_inducing.pk'
  if os.path.exists(results_fn) and os.path.exists(inducing_fn):
    print('results files for '+dnm+' with seed='+str(d_seed)+', n_ind='+str(n_inducing)+', trial='+str(n_trial)+' exists, quitting')
    quit()
  
  kern = GaussianKernel(length_scales, kernel_variance)
  gpykern = GPy.kern.RBF(input_dim=X.shape[1], lengthscale=length_scales, ARD=True, variance=kernel_variance)
  
  #create the model objects
  print('Creating models')
  gp = SubsampleGP(X, Y, kern, likelihood_variance)
  lin = Linear(X, Y, likelihood_variance)
  sgp = SubsampleGP(X, Y, kern, likelihood_variance)
  srgp = SubsetRegressorsGP(X, Y, kern, likelihood_variance)
  vgp = VariationalGP(X, Y, gpykern, likelihood_variance)
  igp = FisherGP(X, Y, kern, likelihood_variance)
  
  #store algs in a list
  anms = ['linear', 'subsample', 'subset_regressors', 'variational_inducing', 'fisher_inducing']
  algs = [lin, sgp, srgp, vgp, igp]
  
    
  #initialize results matrices
  pretrain_cputs = np.zeros(len(algs))
  train_cputs = np.zeros(len(algs))
  pred_cputs = np.zeros(len(algs))
  pred_errs = np.zeros(len(algs))
  post_mean_errs = np.zeros(len(algs))
  post_sig_errs = np.zeros(len(algs))
  kl_divergences = np.zeros(len(algs))
  lsc_errs = np.zeros(2)
  kvar_errs = np.zeros(2)
  lvar_errs = np.zeros(2)
  opt_objs = np.zeros(2)
  
  inducing_pts = []
  for i in range(len(algs)):
    inducing_pts.append([])
  
  #run algs
  print('Running inference')
  print('Dataset: ' + dnm) 
  print('# Inducing pts: ' +str(n_inducing)) 
  print('Trial ' + str(n_trial))
  
  #get index subsets for training
  idcs = np.arange(X.shape[0])
  np.random.shuffle(idcs)
  subsample_idcs = idcs[:n_inducing].copy()
  np.random.shuffle(idcs)
  pretrain_subsample_idcs = idcs[:n_pt].copy()
  
  #run on each algorithm
  for j in range(len(algs)):
    print('Training ' + anms[j])
    #pretrain if required
    if getattr(algs[j], 'pretrain', None) is not None:
      t0 = time.time()
      algs[j].pretrain(pretrain_subsample_idcs)
      pretrain_cputs[j] = time.time()-t0
    #train
    t0 = time.time()
    algs[j].train(subsample_idcs)
    train_cputs[j] = time.time()-t0
    inducing_pts[j].append(algs[j].X_ind)
    #predict
    t0 = time.time()
    mu_pred, var_pred = algs[j].predict_f(Xt, cov_type='full')
    sig_pred = np.sqrt(np.diag(var_pred))
    pred_cputs[j] = time.time() - t0
    #evaluate
    pred_errs[j] = np.sqrt(((mu_pred - Yt)**2).mean())
    post_mean_errs[j] = np.sqrt(((mu_pred_full-mu_pred)**2).mean())
    post_sig_errs[j] = np.sqrt(((sig_pred_full-sig_pred)**2).mean())
    kl_divergences[j] = kl_gaussian(mu_pred_full, var_pred_full, mu_pred, var_pred)
  
    if anms[j] == 'fisher_inducing' and save_objs:
      opt_objs[j-(len(algs)-2)] = algs[j]._objective(algs[j].X_ind, 0, 1e-9)
    if anms[j] == 'variational_inducing' and save_objs:
      opt_objs[j-(len(algs)-2)] = algs[j].model.objective_function()
      
    if (anms[j] == 'variational_inducing' or anms[j] == 'fisher_inducing') and check_hyper_stability:
      print('before post hyperopt: ')
      print(length_scales)
      print(kernel_variance)
      print(likelihood_variance)
  
      kern, like  = optimize_hyperparameters(X, Y, algs[j].X_ind,
                                  		GPy.kern.RBF(input_dim=X.shape[1], lengthscale=length_scales, variance=kernel_variance, ARD=True), 
                                            GPy.likelihoods.Gaussian(variance=likelihood_variance)
                                            )
      lsc = kern.lengthscale
      kvar = kern.variance
      lvar = like.variance
      #lsc, kvar, lvar = optimize_hyperparameters_post(X, Y, algs[j].X_ind, length_scales, kernel_variance, likelihood_variance)
  
      print('after post hyperopt: ')
      print(lsc)
      print(kvar)
      print(lvar)
      print('reldiffs: ')
      print(np.sqrt( ((lsc - length_scales)**2).sum())/np.sqrt((length_scales**2).sum()))
      print(np.fabs(kvar-kernel_variance)/np.fabs(kernel_variance))
      print(np.fabs(lvar - likelihood_variance)/np.fabs(likelihood_variance))
      lsc_errs[j-(len(algs)-2)] =  np.sqrt( ((lsc - length_scales)**2).sum())/np.sqrt((length_scales**2).sum())
      kvar_errs[j-(len(algs)-2)] =  np.fabs(kvar-kernel_variance)/np.fabs(kernel_variance)
      lvar_errs[j-(len(algs)-2)] =  np.fabs(lvar - likelihood_variance)/np.fabs(likelihood_variance)
          
  np.savez(results_fn, n_inducing=n_inducing, anms=anms,  
                                pretrain_cputs=pretrain_cputs, train_cputs=train_cputs, pred_cputs=pred_cputs, 
                                pred_errs=pred_errs, post_mean_errs=post_mean_errs, post_sig_errs=post_sig_errs, kl_divergences=kl_divergences,    
                                lsc_errs=lsc_errs, kvar_errs=kvar_errs, lvar_errs=lvar_errs, opt_objs=opt_objs)
  f = open(inducing_fn, 'wb')
  pk.dump(inducing_pts, f)
  f.close()

