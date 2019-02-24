import numpy as np
import os
import sys

d_seed = 1

if len(sys.argv) < 3:
    sys.exit('usage: ' + sys.argv[0] + ' dataset_name num_trials [n_algs]')

dnm = sys.argv[1]
n_trial = int(sys.argv[2])
if len(sys.argv) > 3:
    n_algs = int(sys.argv[3])
else:
    n_algs = 5

n_inducing = np.unique(np.logspace(0, 3, 10, dtype=np.int))[1:]


pretrain_cputs = np.zeros((n_algs, n_inducing.shape[0], n_trials))
train_cputs = np.zeros((n_algs, n_inducing.shape[0], n_trials))
pred_cputs = np.zeros((n_algs, n_inducing.shape[0], n_trials))
pred_errs = np.zeros((n_algs, n_inducing.shape[0], n_trials))
post_mean_errs = np.zeros((n_algs, n_inducing.shape[0], n_trials))
post_sig_errs = np.zeros((n_algs, n_inducing.shape[0], n_trials))
kl_divergences = np.zeros((n_algs, n_inducing.shape[0], n_trials))
lsc_errs = np.zeros((2, n_inducing.shape[0], n_trials))
kvar_errs = np.zeros((2, n_inducing.shape[0], n_trials))
lvar_errs = np.zeros((2, n_inducing.shape[0], n_trials))
opt_objs = np.zeros((2, n_inducing.shape[0], n_trials))

inducing_pts = [list() for i in range(n_algs)]

for t in range(n_trial):
    for i in range(n_inducing.shape[0]):
        results_fn = 'results/'+dnm+'_'+str(d_seed)+'_'+str(n_inducing[i])+'_'+str(t+1)+'_results.npz'
        # inducing_fn = 'results/'+dnm+'_'+str(d_seed)+'_'+str(n_inducing[i])+'_'+str(t+1)+'_inducing.pk'

        res = np.load(results_fn)
        pretrain_cputs[:,i,t] = res['pretrain_cputs']
        train_cputs[:,i,t] = res['train_cputs']
        pred_cputs[:,i,t] = res['pred_cputs']
        pred_errs[:,i,t] = res['pred_errs']
        post_mean_errs[:,i,t] = res['post_mean_errs']
        post_sig_errs[:,i,t] = res['post_sig_errs']
        kl_divergences[:,i,t] = res['kl_divergences']
        lsc_errs[:,i,t] = res['lsc_errs']
        kvar_errs[:,i,t] = res['kvar_errs']
        lvar_errs[:,i,t] = res['lvar_errs']
        opt_objs[:,i,t] = res['opt_objs']


 np.savez('results/'+dnm+'_'+str(d_seed)+'_results.npz', n_inducing=n_inducing, anms=anms,
          pretrain_cputs=pretrain_cputs, train_cputs=train_cputs, pred_cputs=pred_cputs,
          pred_errs=pred_errs, post_mean_errs=post_mean_errs, post_sig_errs=post_sig_errs,
          kl_divergences=kl_divergences, lsc_errs=lsc_errs, kvar_errs=kvar_errs,
          lvar_errs=lvar_errs, opt_objs=opt_objs)
