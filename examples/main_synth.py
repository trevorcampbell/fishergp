import numpy as np
from fishergp import SubsampleGP, SubsetRegressorsGP, NystromGP, FisherGP, VariationalGP, Linear
from fishergp.kernels import GaussianKernel
import bokeh.plotting as bkp
import bokeh.layouts as bkl
import bokeh.palettes 
import time
import os
import GPy


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

kern = GaussianKernel(gamma, kernel_var)
gpykern = GPy.kern.RBF(input_dim=X.shape[1], lengthscale=gamma, ARD=True, variance=kernel_var)


fig_subset_regressors    = bkp.figure(plot_width=1250, plot_height=600)
fig_inducing    = bkp.figure(plot_width=1250, plot_height=600)
fig_variational    = bkp.figure(plot_width=1250, plot_height=600)

print('Training full GP')
gp = SubsampleGP(X, Y, kern, likelihood_var)
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
gp = VariationalGP(X, Y, gpykern, likelihood_var)
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
gp = FisherGP(X, Y, kern, likelihood_var)
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
gp = SubsetRegressorsGP(X, Y, kern, likelihood_var)
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


