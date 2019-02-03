import numpy as np
from fishergp import SubsampleGP, SubsetRegressorsGP, NystromGP, InducingGP, VariationalGP, Linear, optimize_hyperparameters
import bokeh.plotting as bkp
import bokeh.layouts as bkl
import bokeh.palettes 
import time
import os


N = 300
N_knots = 9
N_srknots = 100
idcs = np.arange(N)
np.random.shuffle(idcs)
idcs = idcs[:N_knots]
likelihood_var = 0.05
kernel_var = 1.0
gamma = 0.14142
ridge = 1e-6

X = np.random.rand(N,1)
Y = np.sin(12*X) + 0.66*np.cos(25*X) + np.random.randn(N,1)*np.sqrt(likelihood_var) + 3
xg = np.linspace(-0.1, 1.1, 1000)
fbtx = np.append(xg, xg[::-1])
yg = np.sin(12*xg) + 0.66*np.cos(25*xg) + 3


#fig_full    = bkp.figure(plot_width=1250, plot_height=1250)
#fig_nystrom    = bkp.figure(plot_width=1250, plot_height=1250)
#fig_subsample    = bkp.figure(plot_width=1250, plot_height=1250)
fig_subset_regressors    = bkp.figure(plot_width=1250, plot_height=600)
#fig_coreset    = bkp.figure(plot_width=1250, plot_height=1250)
fig_inducing    = bkp.figure(plot_width=1250, plot_height=600)
fig_variational    = bkp.figure(plot_width=1250, plot_height=600)
#fig_mean_err    = bkp.figure(plot_width=1250, plot_height=1250, y_axis_type='log')
#fig_sig_err    = bkp.figure(plot_width=1250, plot_height=1250, y_axis_type='log')
#fig_kl    = bkp.figure(plot_width=1250, plot_height=1250, y_axis_type='log')

print('Training full GP')
gp = SubsampleGP(X, Y, gamma, kernel_var, likelihood_var)
gp.train(subsample_idcs=np.arange(X.shape[0]), ridge=ridge)
mu_full, sig_full = gp.predict_y(xg[:, np.newaxis], cov_type='diag')
mu_full = mu_full.flatten()
fbty_full = np.append( mu_full-np.sqrt(sig_full), (mu_full+np.sqrt(sig_full))[::-1] )

fig_variational.circle(X.T.flatten(), Y.T.flatten(), fill_color='black', line_color=None, size=10)
#fig_variational.line(xg, yg, line_color='black', line_width=2)
fig_variational.line(xg, mu_full, line_color='blue', line_width=4)
fig_variational.patch(fbtx, fbty_full, color='blue', fill_alpha=0.1, line_color=None)

fig_inducing.circle(X.T.flatten(), Y.T.flatten(), fill_color='black', line_color=None, size=10)
#fig_inducing.line(xg, yg, line_color='black', line_width=2)
fig_inducing.line(xg, mu_full, line_color='blue', line_width=4)
fig_inducing.patch(fbtx, fbty_full, color='blue', fill_alpha=0.1, line_color=None)

fig_subset_regressors.circle(X.T.flatten(), Y.T.flatten(), fill_color='black', line_color=None, size=10)
fig_subset_regressors.line(xg, mu_full, line_color='blue', line_width=4)
fig_subset_regressors.patch(fbtx, fbty_full, color='blue', fill_alpha=0.1, line_color=None)



print('Training variational GP')
gp = VariationalGP(X, Y, gamma*np.ones(X.shape[1]), kernel_var, likelihood_var)
gp.train(idcs)
mu_var, sig_var = gp.predict_y(xg[:, np.newaxis], cov_type='diag')
mu_Xind, _ = gp.predict_y(gp.X_ind, cov_type='diag')
mu_var = mu_var.flatten()
fbty_var = np.append( mu_var-np.sqrt(sig_var), (mu_var+np.sqrt(sig_var))[::-1] )

#fig_variational.circle(X.T.flatten(), Y.T.flatten(), fill_color='black', line_color=None, size=10)
#fig_variational.line(xg, yg, line_color='black', line_width=2)
fig_variational.line(xg, mu_var, line_color='red', line_width=4)
fig_variational.circle(gp.X_ind.flatten(), mu_Xind.flatten(), fill_color='red', line_color=None, size=25)
fig_variational.patch(fbtx, fbty_var, color=None, line_color='red', line_width=4, line_dash='dashed')

print('Training inducing GP')
sridcs = np.arange(N)
np.random.shuffle(sridcs)
sridcs = sridcs[:N_srknots]
#plot inducing pts 
gp = InducingGP(X, Y, gamma*np.ones(X.shape[1]), kernel_var, likelihood_var)
gp.pretrain(sridcs)
gp.train(idcs)
mu_ind, sig_ind = gp.predict_y(xg[:, np.newaxis], cov_type='diag')
mu_Xind, _ = gp.predict_y(gp.X_ind, cov_type='diag')
mu_ind = mu_ind.flatten()
fbty_ind = np.append( mu_ind-np.sqrt(sig_ind), (mu_ind+np.sqrt(sig_ind))[::-1] )

fig_inducing.line(xg, mu_ind, line_color='red', line_width=4)
fig_inducing.circle(gp.X_ind.flatten(), mu_Xind.flatten(), fill_color='red', line_color=None, size=25)
fig_inducing.patch(fbtx, fbty_ind, color=None, line_color='red', line_width=4, line_dash='dashed')


print('Training subset regressors GP')
gp = SubsetRegressorsGP(X, Y, gamma, kernel_var, likelihood_var)
gp.train(subsample_idcs=idcs, ridge=ridge)
mu_sr, sig_sr = gp.predict_y(xg[:, np.newaxis], cov_type='diag')
mu_sr = mu_sr.flatten()
fbty_sr = np.append( mu_sr-np.sqrt(sig_sr), (mu_sr+np.sqrt(sig_sr))[::-1] )

fig_subset_regressors.circle(X[idcs, :].T.flatten(), Y[idcs, :].T.flatten(), size=25, fill_color='red')
#fig_subset_regressors.line(xg, yg, line_color='black', line_width=2)
fig_subset_regressors.line(xg, mu_sr, line_color='red', line_width=4)
fig_subset_regressors.patch(fbtx, fbty_sr, color=None, line_color='red', line_width=4, line_dash='dashed')


font_size=None
logFmtr=None
for f in [fig_variational, fig_inducing, fig_subset_regressors]:
    f.xaxis.axis_label_text_font_size= font_size
    f.xaxis.major_label_text_font_size= font_size
    f.yaxis.axis_label_text_font_size= font_size
    f.yaxis.major_label_text_font_size= font_size
    f.title.text_font_size = font_size



bkp.show(bkl.gridplot([[fig_inducing, fig_variational, fig_subset_regressors]]))

#
#
#fig_variational.circle(X.T.flatten(), Y.T.flatten(), fill_color='black', line_color=None, size=10)
#fig_variational.line(xg, yg, line_color='black', line_width=2)
#fig_variational.line(xg, mu_vgp, line_color='red', line_width=2)
#fig_variational.patch(fbtx, fbty, color='red', fill_alpha=0.1)
#fig_variational.circle(np.asarray(gp.model.inducing_inputs).flatten(), mu_Xind.flatten(), size=10, fill_color='red')
#
#print('Training inducing GP')
#sridcs = np.arange(N)
#np.random.shuffle(sridcs)
#sridcs = sridcs[:N_srknots]
##plot inducing pts 
#igp = InducingGP(X, Y, gamma*np.ones(X.shape[1]), kernel_var, likelihood_var)
#igp.pretrain(sridcs)
#igp.train(idcs)
#mu_ind, sig_ind = igp.predict_f(xg[:, np.newaxis], cov_type='diag')
#mu_Xind, _ = igp.predict_f(igp.X_ind, cov_type='diag')
#mu_ind = mu_ind.flatten()
#print('RMSE: ' + str(np.sqrt(((mu_ind-mu)**2).mean())))
#fbty = np.append( mu_ind-np.sqrt(sig_ind), (mu_ind+np.sqrt(sig_ind))[::-1] )
#
#fig_inducing.circle(X.T.flatten(), Y.T.flatten(), fill_color='black', line_color=None, size=10)
#fig_inducing.line(xg, yg, line_color='black', line_width=2)
#fig_inducing.line(xg, mu_ind, line_color='red', line_width=2)
#fig_inducing.patch(fbtx, fbty, color='red', fill_alpha=0.1)
#fig_inducing.circle(X[sridcs, :].T.flatten(), Y[sridcs, :].T.flatten(), size=10)
#fig_inducing.circle(igp.X_ind.flatten(), mu_Xind.flatten(), size=10, fill_color='red')
##
#











#print('Training subsample GP')
#gp = SubsampleGP(X, Y, gamma, kernel_var, likelihood_var)
#gp.train(subsample_idcs=idcs, ridge=ridge)
#mu_s, sig_s = gp.predict_f(xg[:, np.newaxis], cov_type='diag')
#mu_s = mu_s.flatten()
#print('RMSE: ' + str(np.sqrt(((mu_s-mu)**2).mean())))
#fbty = np.append( mu_s-np.sqrt(sig_s), (mu_s+np.sqrt(sig_s))[::-1] )
#
#fig_subsample.circle(X.T.flatten(), Y.T.flatten(), fill_color='black', line_color=None, size=10)
#fig_subsample.circle(X[idcs, :].T.flatten(), Y[idcs, :].T.flatten(), size=10)
#fig_subsample.line(xg, yg, line_color='black', line_width=2)
#fig_subsample.line(xg, mu_s, line_color='blue', line_width=2)
#fig_subsample.patch(fbtx, fbty, color='blue', fill_alpha=0.1)
#
#print('Training nystrom GP')
#gp = NystromGP(X, Y, gamma, kernel_var, likelihood_var)
#gp.train(subsample_idcs=idcs, ridge=ridge)
#mu_n, sig_n = gp.predict_f(xg[:, np.newaxis], cov_type='diag')
#mu_n = mu_n.flatten()
#print('RMSE: ' + str(np.sqrt(((mu_n-mu)**2).mean())))
#fbty = np.append( mu_n-np.sqrt(sig_n), (mu_n+np.sqrt(sig_n))[::-1] )
#
#fig_nystrom.circle(X.T.flatten(), Y.T.flatten(), fill_color='black', line_color=None, size=10)
#fig_nystrom.circle(X[idcs, :].T.flatten(), Y[idcs, :].T.flatten(), size=10)
#fig_nystrom.line(xg, yg, line_color='black', line_width=2)
#fig_nystrom.line(xg, mu_n, line_color='blue', line_width=2)
#fig_nystrom.patch(fbtx, fbty, color='blue', fill_alpha=0.1)
#

#
##print('Training coreset GP')
###plot coresets
##cgp = CoresetGP(X, Y, kern, likelihood_var)
##cgp.train(idcs, N_subsample, preconditioned=False, ridge=ridge)
##mu_cst, sig_cst = cgp.predict_f(xg[:, np.newaxis], cov_type='diag')
##mu_cst = mu_cst.flatten()
##fig_coreset.circle(X.T.flatten(), Y.T.flatten(), fill_color='black', line_color=None, size=10)
##fig_coreset.circle(X[idcs, :].T.flatten(), Y[idcs, :].T.flatten(), size=10)
##fig_coreset.line(xg, yg, line_color='black', line_width=2)
##fig_coreset.line(xg, mu_cst, line_color='blue', line_width=2)
##fbty = np.append( mu_cst-np.sqrt(sig_cst), (mu_cst+np.sqrt(sig_cst))[::-1] )
##fig_coreset.patch(fbtx, fbty, color='blue', fill_alpha=0.1)
##
##
#
#fig_mean_err.line(xg, np.fabs(mu_n-mu)/np.fabs(mu), line_color='blue', line_width=2, legend='nystrom')
#fig_mean_err.line(xg, np.fabs(mu_s-mu)/np.fabs(mu), line_color='green', line_width=2, legend='subsample')
#fig_mean_err.line(xg, np.fabs(mu_sr-mu)/np.fabs(mu), line_color='red', line_width=2, legend='subset_regresors')
#fig_mean_err.line(xg, np.fabs(mu_ind-mu)/np.fabs(mu), line_color='purple', line_width=2, legend='inducing')
##fig_mean_err.line(xg, np.fabs(mu_cst-mu)/np.fabs(mu), line_color='orange', line_width=2, legend='coreset')
#fig_mean_err.line(xg, np.fabs(mu_vgp-mu)/np.fabs(mu), line_color='black', line_width=2, legend='variational')
#
#fig_sig_err.line(xg, np.fabs(sig_n-sig)/np.fabs(sig), line_color='blue', line_width=2, legend='nystrom')
#fig_sig_err.line(xg, np.fabs(sig_s-sig)/np.fabs(sig), line_color='green', line_width=2, legend='subsample')
#fig_sig_err.line(xg, np.fabs(sig_sr-sig)/np.fabs(sig), line_color='red', line_width=2, legend='subset_regressors')
#fig_sig_err.line(xg, np.fabs(sig_ind-sig)/np.fabs(sig), line_color='purple', line_width=2, legend='inducing')
##fig_sig_err.line(xg, np.fabs(sig_cst-sig)/np.fabs(sig), line_color='orange', line_width=2, legend='coreset')
#fig_sig_err.line(xg, np.fabs(sig_vgp-sig)/np.fabs(sig), line_color='black', line_width=2, legend='variational')
#
#fig_kl.line(xg, 0.5*(np.log(sig_n/sig) + (sig+(mu-mu_n)**2)/(sig_n) - 1.), line_color='blue', line_width=2, legend='nystrom')
#fig_kl.line(xg, 0.5*(np.log(sig_s/sig) + (sig+(mu-mu_s)**2)/(sig_s) - 1.), line_color='green', line_width=2, legend='subsample')
#fig_kl.line(xg, 0.5*(np.log(sig_sr/sig) + (sig+(mu-mu_sr)**2)/(sig_sr) - 1.), line_color='red', line_width=2, legend='subset_regressors')
#fig_kl.line(xg, 0.5*(np.log(sig_ind/sig) + (sig+(mu-mu_ind)**2)/(sig_ind) - 1.), line_color='purple', line_width=2, legend='inducing')
##fig_kl.line(xg, np.fabs(sig_cst-sig)/np.fabs(sig), line_color='orange', line_width=2, legend='coreset')
#fig_kl.line(xg, 0.5*(np.log(sig_vgp/sig) + (sig+(mu-mu_vgp)**2)/(sig_vgp) - 1.), line_color='black', line_width=2, legend='variational')


#bkp.show(bkl.gridplot([[fig_full], [fig_nystrom, fig_subsample, fig_subset_regressors], [fig_inducing, fig_coreset, fig_variational], [fig_mean_err, fig_sig_err, fig_kl]]))


