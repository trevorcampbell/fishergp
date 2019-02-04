import autograd.numpy as np
from fishergp import SubsampleGP, SubsetRegressorsGP, NystromGP,  FisherGP, VariationalGP
from fishergp.utils import optimize_hyperparameters
from fishergp.kernels import GaussianKernel
import bokeh.plotting as bkp
import bokeh.layouts as bkl
import bokeh.palettes 
import GPy

#load data
print('loading data')
data = np.load('datasets/delays.npy')
#shuffle order
print('shuffling data')
np.random.shuffle(data)


print('extracting training/test sets')
N_train = 10000
N_test = 10000


data = data[:1000, :]
N_train = 800
N_test = 200


#extract train, test sets
X = data[:-N_test, :-1]
Y = data[:-N_test, -1][:, np.newaxis]
Xt = data[-N_test:, :-1]
Yt = data[-N_test:, -1][:, np.newaxis]
Ytnorm = np.sqrt((Yt**2).sum())

print('standardizing using training data')
#standardize input/output
Ymu = Y.mean(axis=0)
Xmu = X.mean(axis=0)
X -= Xmu
Y -= Ymu
Xt -= Xmu
Yt -= Ymu
Xcov = np.cov(X, rowvar=False)
u, V = np.linalg.eigh(Xcov)
X = X.dot(V)/np.sqrt(u)
Xt = Xt.dot(V)/np.sqrt(u)

print('getting training subsample for nystrom/SR/subsample and inducing initialization')
#select subsample idcs
N_subsample = 1000


N_subsample = 20


N = X.shape[0]
idcs = np.arange(N)
np.random.shuffle(idcs)
idcs = idcs[:N_subsample]

print('constructing kernel')
ridge = 1e-9
batch_size=500
#optimize_hyperparameters(X, Y, idcs, batch_size)
#optimize_hyperparameters(X, Y, idcs, batch_size)
#optimize_hyperparameters(X, Y, idcs, batch_size)
likelihood_variance = 296.16
kernel_variance = 17.66
length_scales = np.array([0.7224, 11.17, 0.8665, 3.69081, 5.23420, 13.162661, 1.84575, 0.6019])
kern = GaussianKernel(length_scales, kernel_variance)
gpykern = GPy.kern.RBF(input_dim=X.shape[1], lengthscale=length_scales, ARD=True, variance=kernel_variance)

print('Naive mean error')
mu_mean = Y.mean()
print('Normalized Error: ' + str(np.sqrt( (( mu_mean - Yt)**2).sum())/Ytnorm))
print('RMS Error: ' + str(np.sqrt( (( mu_mean - Yt)**2).mean())))

print('Training subsample GP')
gp = SubsampleGP(X, Y, kern, likelihood_variance)
gp.train(subsample_idcs=idcs, ridge=ridge)
mu_s, sig_s = gp.predict_y(Xt, cov_type='diag')
mu_s = mu_s.flatten()
print('Normalized Error: ' + str(np.sqrt( (( mu_s - Yt)**2).sum())/Ytnorm))
print('RMS Error: ' + str(np.sqrt( (( mu_s - Yt)**2).mean())))

gp = VariationalGP(X, Y, gpykern, likelihood_variance)
print('Training sparse variational GP')
gp.train(idcs)
mu_svgp, sig_svgp = gp.predict_y(Xt, cov_type='diag')
mu_svgp = mu_svgp.flatten()
print(np.sqrt(((mu_svgp - Yt)**2).sum())/Ytnorm)


print('Training nystrom GP')
gp = NystromGP(X, Y, kern, likelihood_variance)
gp.train(subsample_idcs=idcs, ridge=ridge)
mu_n, sig_n = gp.predict_y(Xt, cov_type='diag')
mu_n = mu_n.flatten()
print('Normalized Error: ' + str(np.sqrt( (( mu_n - Yt)**2).sum())/Ytnorm))
print('RMS Error: ' + str(np.sqrt( (( mu_n - Yt)**2).mean())))



print('Training subset regressors GP')
gp = SubsetRegressorsGP(X, Y, kern, likelihood_variance)
gp.train(subsample_idcs=idcs, ridge=ridge)
mu_sr, sig_sr = gp.predict_y(Xt, cov_type='diag')
mu_sr = mu_sr.flatten()
print('Normalized Error: ' + str(np.sqrt( (( mu_sr - Yt)**2).sum())/Ytnorm))
print('RMS Error: ' + str(np.sqrt( (( mu_sr - Yt)**2).mean())))


igp = FisherGP(X, Y, kern, likelihood_variance)
print('Training inducing GP')
igp.pretrain(subsample_idcs=idcs)
igp.train(subsample_idcs=idcs, ridge=ridge)
mu_ind, sig_ind = igp.predict_y(Xt, cov_type='diag')
mu_ind = mu_ind.flatten()
print(np.sqrt(((mu_ind - Yt)**2).sum())/Ytnorm)



