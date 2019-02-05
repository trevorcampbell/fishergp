import numpy as np
from gen_data import gen_synthetic, gen_linear, gen_from_file, standardize
from fishergp import SubsampleGP, SubsetRegressorsGP, NystromGP, FisherGP, VariationalGP, Linear
from fishergp.utils import optimize_hyperparameters
from fishergp.kernels import GaussianKernel
import bokeh.plotting as bkp
import bokeh.layouts as bkl
import bokeh.palettes 
import pickle as pk
import time
import os
import GPy

def kl_gaussian(mu1, Sig1, mu2, Sig2, ridge=1e-9):
  r = ridge*np.eye(mu1.shape[0])
  #print(np.linalg.eigvalsh(Sig1+r).min())
  #print(np.linalg.eigvalsh(Sig2+r).min())
  k1 = np.trace(np.linalg.solve(Sig2+r, Sig1+r))
  k2 = (mu1-mu2).T.dot( np.linalg.solve(Sig2+r, mu1-mu2   ))
  k3 = np.linalg.slogdet(Sig2+r)[1] - np.linalg.slogdet(Sig1+r)[1]
  return 0.5*( k1 + k2 + k3 - mu1.shape[0] )


#GPy.kern.RBF(input_dim=X.shape[1], lengthscale=self.k.gamma, ARD=True, variance=self.k.sigma)


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


n_trials = 10
n_inducing = np.array([2, 5, 10], dtype=np.int64)




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
     
  #initialize results matrices
  pretrain_cputs = np.zeros((len(algs), n_inducing.shape[0], n_trials))
  train_cputs = np.zeros((len(algs), n_inducing.shape[0], n_trials))
  pred_cputs = np.zeros((len(algs), n_inducing.shape[0], n_trials))
  pred_errs = np.zeros((len(algs), n_inducing.shape[0], n_trials))
  post_mean_errs = np.zeros((len(algs), n_inducing.shape[0], n_trials))
  post_sig_errs = np.zeros((len(algs), n_inducing.shape[0], n_trials))
  kl_divergences = np.zeros((len(algs), n_inducing.shape[0], n_trials))
  lsc_errs = np.zeros((2, n_inducing.shape[0], n_trials))
  kvar_errs = np.zeros((2, n_inducing.shape[0], n_trials))
  lvar_errs = np.zeros((2, n_inducing.shape[0], n_trials))
  inducing_pts = []
  for i in range(len(algs)):
    inducing_pts.append([])
    for j in range(n_inducing.shape[0]):
      inducing_pts[i].append([])

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
        inducing_pts[j][i].append(algs[j].X_ind)
        #predict
        t0 = time.time()
        mu_pred, var_pred = algs[j].predict_f(Xt, cov_type='full')
        sig_pred = np.sqrt(np.diag(var_pred))
        pred_cputs[j, i, t] = time.time() - t0
        #evaluate
        pred_errs[j, i, t] = np.sqrt(((mu_pred - Yt)**2).mean())
        post_mean_errs[j, i, t] = np.sqrt(((mu_pred_full-mu_pred)**2).mean())
        post_sig_errs[j, i, t] = np.sqrt(((sig_pred_full-sig_pred)**2).mean())
        kl_divergences[j,i,t] = kl_gaussian(mu_pred_full, var_pred_full, mu_pred, var_pred)
        if j >= len(algs)-2:
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
          lsc_errs[j-(len(algs)-2), i, t] =  np.sqrt( ((lsc - length_scales)**2).sum())/np.sqrt((length_scales**2).sum())
          kvar_errs[j-(len(algs)-2), i, t] =  np.fabs(kvar-kernel_variance)/np.fabs(kernel_variance)
          lvar_errs[j-(len(algs)-2), i, t] =  np.fabs(lvar - likelihood_variance)/np.fabs(likelihood_variance)
          
  np.savez('results/'+dnm+'_'+str(d_seed)+'_results.npz', n_inducing=n_inducing, anms=anms,  
                                pretrain_cputs=pretrain_cputs, train_cputs=train_cputs, pred_cputs=pred_cputs, 
                                pred_errs=pred_errs, post_mean_errs=post_mean_errs, post_sig_errs=post_sig_errs, kl_divergences=kl_divergences,    
                                lsc_errs=lsc_errs, kvar_errs=kvar_errs, lvar_errs=lvar_errs)
  f = open('results/'+dnm+'_'+str(d_seed)+'_inducing_pts.pk', 'wb')
  pk.dump(inducing_pts, f)
  f.close()

