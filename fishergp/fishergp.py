import autograd.numpy as np
from autograd import grad
from scipy.optimize import minimize, Bounds
from .utils import ProgressBar
import GPy

##Base GP class
class GP(object):
  def __init__(self, X, Y, kernel, likelihood_variance): #length_scale, kernel_variance, likelihood_variance):
    self.X = X
    self.Y = Y
    self.lvar = likelihood_variance
    self.k = kernel #lambda U, V=None, diag=False : expkern(U, V, length_scale, kernel_variance, diag)

  def train(self, subsample_idcs=None, ridge=1e-9):
    raise NotImplementedError

  def predict_y(self, Xt, cov_type='full'):
    mu, sig = self.predict_f(Xt, cov_type)
    if cov_type == 'full':
      sig += np.diag(self.lvar*np.ones(sig.shape[0]))
    else:
      sig += self.lvar
    return mu, sig

  def predict_f(self, Xt, cov_type='full'):
    raise NotImplementedError

class SubsampleGP(GP):

  def train(self, subsample_idcs=None, ridge=1e-9):
    self.idcs = np.sort(subsample_idcs)
    self.X_ind = self.X[self.idcs, :]

    Kxx = self.k(self.X[self.idcs, :])
    self.alpha = np.linalg.solve(Kxx+self.lvar*np.eye(self.idcs.shape[0]), self.Y[self.idcs, :])
    self.V = Kxx+self.lvar*np.eye(self.idcs.shape[0])

  def predict_f(self, Xt, cov_type='full'):
    Kxtx = self.k(Xt, self.X[self.idcs, :])
    mu = Kxtx.dot(self.alpha)

    if cov_type=='full':
        sig = self.k(Xt) - Kxtx.dot(np.linalg.solve(self.V, Kxtx.T))
    else:
        sig = self.k(Xt, diag=True) - (Kxtx*(np.linalg.solve(self.V, Kxtx.T).T)).sum(axis=1)
    return mu, sig

class SubsetRegressorsGP(GP):

  def train(self, subsample_idcs=None, ridge=1e-9, max_storage=1e8):
    self.idcs = np.sort(subsample_idcs)
    self.X_ind = self.X[self.idcs, :]

    self.chunk_size  = int(max_storage / float(self.idcs.shape[0]))
    if self.chunk_size == 0:
      self.chunk_size = 1
    csz = self.chunk_size

    #can handle len(idcs)**2 memory, len(idcs)**3 time cost
    Kxx = self.k(self.X[self.idcs, :])
    Ysum = np.zeros((self.idcs.shape[0], self.Y.shape[1]))
    Ksum = np.zeros((self.idcs.shape[0], self.idcs.shape[0]))

    pbar = ProgressBar('Chunk sum', 0, self.X.shape[0])
    j = 0
    csz = self.idcs.shape[0]
    while j*csz < self.X.shape[0]:
      pbar.update(j*csz)
      Kchunk = self.k(self.X[j*csz:(j+1)*csz, :], self.X[self.idcs, :])
      Ychunk = self.Y[j*csz:(j+1)*csz, :]
      Ysum += Kchunk.T.dot(Ychunk)
      Ksum += Kchunk.T.dot(Kchunk)
      j += 1
    pbar.finish()

    self.V = Ksum + self.lvar*Kxx
    self.V += ridge*np.fabs(self.V).max()*np.eye(self.V.shape[0])
    self.alpha = np.linalg.solve(self.V, Ysum)
    self.V /= self.lvar

  def predict_f(self, Xt, cov_type='full'):
    Kxtxsr = self.k(Xt, self.X[self.idcs, :])
    mu = Kxtxsr.dot(self.alpha)

    if cov_type=='full':
      sig = Kxtxsr.dot(np.linalg.solve(self.V, Kxtxsr.T))
    else:
      sig = (Kxtxsr*(np.linalg.solve(self.V, Kxtxsr.T).T)).sum(axis=1)
    return mu, sig

class NystromGP(GP):

  def train(self, subsample_idcs=None, ridge=1e-9, max_storage=1e8):
    self.idcs = np.sort(subsample_idcs)
    self.X_ind = self.X[self.idcs, :]

    self.chunk_size  = int(max_storage / float(self.idcs.shape[0]))
    if self.chunk_size == 0:
      self.chunk_size = 1
    csz = self.chunk_size

    #can handle len(idcs)**2 memory, len(idcs)**3 time cost
    Ysum = np.zeros((self.idcs.shape[0], self.Y.shape[1]))
    Ksum = np.zeros((self.idcs.shape[0], self.idcs.shape[0]))

    pbar = ProgressBar('Computing Kxxi^TKxxi, Kxxi^TY ', 0, self.X.shape[0])
    j = 0
    while j*csz < self.X.shape[0]:
      pbar.update(j*csz)
      Kchunk = self.k(self.X[j*csz:(j+1)*csz, :], self.X[self.idcs, :])
      Ychunk = self.Y[j*csz:(j+1)*csz, :]
      Ysum += Kchunk.T.dot(Ychunk)
      Ksum += Kchunk.T.dot(Kchunk)
      j += 1
    pbar.finish()

    self.V = Ksum+ self.lvar*self.k(self.X[self.idcs, :])
    self.V += ridge*np.fabs(self.V).max()*np.eye(self.V.shape[0])
    beta = np.linalg.solve(self.V, Ysum)
    self.alpha = self.Y.copy()

    pbar = ProgressBar('Computing alpha', 0, self.X.shape[0])
    j = 0
    while j*csz < self.X.shape[0]:
      pbar.update(j*csz)
      Kchunk = self.k(self.X[j*csz:(j+1)*csz, :], self.X[self.idcs, :])
      self.alpha[j*csz:(j+1)*csz, :] -= Kchunk.dot(beta)
      j += 1
    pbar.finish()

    self.alpha /= self.lvar

  def predict_f(self, Xt, cov_type='full'):

    mu = np.zeros((Xt.shape[0], self.alpha.shape[1]))
    UTk = np.zeros((self.idcs.shape[0], Xt.shape[0]))
    kTk = np.zeros((Xt.shape[0], Xt.shape[0]))

    j = 0
    csz = self.chunk_size
    pbar = ProgressBar('Computing mu, Kxxt^TKxxi, Kxxt^TKxxt', 0, self.X.shape[0])
    while j*csz < self.X.shape[0]:
      pbar.update(j*csz)
      Kchunk = self.k(self.X[j*csz:(j+1)*csz, :], self.X[self.idcs, :])
      Kchunk_t = self.k(self.X[j*csz:(j+1)*csz, :], Xt)

      mu += Kchunk_t.T.dot(self.alpha[j*csz:(j+1)*csz, :])
      UTk += Kchunk.T.dot(Kchunk_t)
      kTk += Kchunk_t.T.dot(Kchunk_t)
      j += 1

    K2 = UTk.T.dot(np.linalg.solve(self.V, UTk))
    if cov_type == 'full':
      sig = self.k(Xt) - kTk/self.lvar + K2/self.lvar
    else:
      sig = self.k(Xt, diag=True) - np.diag(kTk)/self.lvar + np.diag(K2)/self.lvar

    return mu, sig

class Linear(SubsampleGP):
  def __init__(self, X, Y, likelihood_variance):
    self.X = X
    self.Y = Y
    self.k = lambda U, V=None : U.dot(V.T) if V is not None else U.dot(U.T)
    self.lvar = likelihood_variance

  #def train(self, subsample_idcs):
  #  XTX = np.dot(self.X.T, self.X)
  #  XTY = np.dot(self.X.T, self.Y)
  #  self.alpha = np.linalg.solve(XTX, XTY)

  #def predict_f(self, Xt, cov_type='full'):
  #  mu = np.dot(Xt, self.alpha)
  #  sig = np.zeros((mu.shape[0], mu.shape[0]))
  #  if cov_type == 'full':
  #    return mu, sig
  #  else:
  #    return mu, np.diag(sig).copy()


#this class is essentially a wrapper around GPy
class VariationalGP(GP):
  
  def train(self, subsample_idcs):
    self.idcs = np.sort(subsample_idcs)
    self.model = GPy.core.SparseGP(self.X, self.Y, self.X[self.idcs, :].copy(),
                           self.k,
                           GPy.likelihoods.Gaussian(variance=self.lvar))
    self.model.kern.fix()
    self.model.likelihood.variance.fix()
    self.model.inducing_inputs.constrain_bounded(self.X.min(), self.X.max())
    self.model.optimize('lbfgsb', max_iters=10000, messages=True, ipython_notebook=False)
    #self.model.optimize('fmin_tnc', max_iters=10000, messages=True, ipython_notebook=False)
    self.X_ind = np.asarray(self.model.inducing_inputs)

  def predict_f(self, Xt, cov_type='full'):
    mu, sig = self.model.predict_noiseless(Xt, full_cov=True if cov_type=='full' else False)

    if cov_type == 'full':
      return mu, sig
    else:
      return mu, sig.flatten()


class FisherGP(GP):

  def pretrain(self, subsample_idcs, ridge=1e-9):
    self.sridcs = np.sort(subsample_idcs)

    #get matrices necessary to compute khat(x,x') and muhat(x)
    #using subset of regressors
    srgp = SubsetRegressorsGP(self.X, self.Y, self.k, self.lvar)
    srgp.train(self.sridcs, ridge)
    self.pre_alpha = srgp.alpha
    self.V = srgp.V
    self.V += ridge*np.fabs(self.V).max()*np.eye(self.sridcs.shape[0])

  def train(self, subsample_idcs, ridge=1e-9, Xt=None, mu_full=None, sig_full=None):
    #chunk size
    csz = max(self.sridcs.shape[0], subsample_idcs.shape[0])

    #create storage for mean/var/obj trace
    self.mu_err_trace = []
    self.sig_err_trace = []
    self.obj_trace = []

    #compute muhat(X) - Y using chunked method to avoid high memory expense
    self.dYx = np.zeros(self.Y.shape)
    j = 0
    while j*csz < self.X.shape[0]:
      Kchunk = self.k(self.X[j*csz:(j+1)*csz, :], self.X[self.sridcs, :])
      self.dYx[j*csz:(j+1)*csz, :] = Kchunk.dot(self.pre_alpha) - self.Y[j*csz:(j+1)*csz, :]
      j += 1

    #optimize inducing pts
    Z = self.X[subsample_idcs, :]
    self.Zshape = Z.shape
    if Xt is not None and mu_full is not None and sig_full is not None:
      __cbk = lambda x : self._cbk(x, Xt, mu_full, sig_full, ridge)
    else:
      __cbk = None
    self.X_ind = minimize(fun=lambda x : self._objective(x, 0, ridge),
                        x0=Z.flatten(),
                        jac=grad(lambda x : self._objective(x, 0, ridge)),
                        bounds=Bounds(self.X.min(), self.X.max()),
                        method='L-BFGS-B', options ={'disp' : True, 'maxiter':10000},
                        #method='TNC', options ={'disp' : True, 'maxiter':10000},
                        callback = __cbk,
                        ).x.reshape(self.Zshape)
    #after optimization is done, compute alpha / C for testing
    self.posttrain()

  def posttrain(self, ridge=1e-9):
    #compute constants for prediction
    Kxi = self.k(self.X_ind)
    self.C = Kxi.copy()
    j = 0
    KY = np.zeros((self.X_ind.shape[0], self.Y.shape[1]))
    KK = np.zeros((self.X_ind.shape[0],self.X_ind.shape[0]))
    csz = max(self.sridcs.shape[0], self.X_ind.shape[0])
    while j*csz < self.X.shape[0]:
      Kchunk = self.k(self.X[j*csz:(j+1)*csz, :], self.X_ind)
      KY += np.dot(Kchunk.T, self.Y[j*csz:(j+1)*csz, :])
      KK += np.dot(Kchunk.T, Kchunk)
      j += 1
    self.C += KK/self.lvar
    self.C += ridge*np.fabs(self.C).max()*np.eye(self.C.shape[0])
    self.alpha = np.linalg.solve(self.C, KY)/self.lvar
    Kxi += ridge*np.fabs(Kxi).max()*np.eye(Kxi.shape[0])
    self.C = np.linalg.inv(self.C) - np.linalg.inv(Kxi)
    #self.C = np.linalg.solve(Kxi.T, (np.linalg.solve(self.C, Kxi) - np.eye(self.C.shape[0])).T).T

  def _cbk(self, x, Xt, mu_full, sig_full, ridge):
    self.X_ind = x.reshape(self.Zshape)
    self.posttrain()
    mu_ind, var_ind = self.predict_f(Xt, 'diag')
    sig_ind = np.sqrt(var_ind)
    self.mu_err_trace.append(np.sqrt(((mu_ind - mu_full)**2).sum())/np.sqrt((mu_full**2).sum()))
    self.sig_err_trace.append(np.sqrt(((sig_ind - sig_full)**2).sum())/np.sqrt((sig_full**2).sum()))
    self.obj_trace.append(self._objective(x, 0, ridge, True))

  def _objective(self, X_i, itr, ridge, compute_constant=False):
    #chunk size
    csz = max(self.sridcs.shape[0], self.Zshape[0])
    #inducing pt size
    isz = self.Zshape[0]
    #subsample size
    srsz = self.sridcs.shape[0]

    #reshape inducing pt matrix (scipy.minimize uses a flattened version)
    X_i = X_i.reshape(self.Zshape)

    #compute useful constants
    Kxi = self.k(X_i)
    Kxi += ridge*np.fabs(Kxi).max()*np.eye(isz)
    Kxixsr = self.k(X_i, self.X[self.sridcs, :])
    Khxi = np.dot(Kxixsr, np.linalg.solve(self.V, Kxixsr.T))

    Kxi_inv_muxi = np.linalg.solve(Kxi, np.dot(Kxixsr, self.pre_alpha))
    

    KxxiTKxxi = np.zeros((isz,isz))
    KxxsrTKxxi = np.zeros((srsz,isz))
    dYxi_chunks = [] #need this because autograd doesn't support array asgnmt
    KdY = np.zeros((isz, self.Y.shape[1]))
    KdYi = np.zeros((isz, self.Y.shape[1]))
    j = 0
    while j*csz < self.X.shape[0]:
      Kchunk_sr = self.k(self.X[j*csz:(j+1)*csz, :], self.X[self.sridcs, :])
      Kchunk_i = self.k(self.X[j*csz:(j+1)*csz, :], X_i)
      KxxiTKxxi += np.dot(Kchunk_i.T, Kchunk_i)
      KxxsrTKxxi += np.dot(Kchunk_sr.T, Kchunk_i)
      dYxi_chunk = np.dot(Kchunk_i, Kxi_inv_muxi) - self.Y[j*csz:(j+1)*csz, :]
      dYxi_chunks.append(dYxi_chunk)
      KdY += np.dot(Kchunk_i.T, self.dYx[j*csz:(j+1)*csz, :])
      KdYi += np.dot(Kchunk_i.T, dYxi_chunk)
      j += 1
    dYxi = np.vstack(dYxi_chunks)

    Kxi_inv_KxxiTKxxi = np.linalg.solve(Kxi, KxxiTKxxi)

    B = np.linalg.solve(Kxi.T, Kxi_inv_KxxiTKxxi.T).T / self.lvar
    B = 0.5*(B+B.T) #enforce symmetry (lin solver doesn't guarantee)

    Xi = np.linalg.solve(np.eye(isz) + np.dot(B, Kxi), B)
    Xi = 0.5*(Xi+Xi.T) #enforce symmetry (lin solver doesn't guarantee)

    ##################
    ##Compute ||L||^2
    ##################

    #compute C3 (used for S3 = Kx,xi*C3*Kxi,x )
    C3 = np.dot(Xi, np.dot(Kxi, Xi)) - 2.*Xi
    L = np.trace(np.dot(KxxsrTKxxi.T, np.linalg.solve(self.V, np.dot(KxxsrTKxxi, C3))))
    L += np.trace(np.dot(KdY.T, np.dot(C3, KdY)))

    ##this code computes objective function constants that aren't needed for optimization
    if compute_constant:
      Kx = self.k(self.X)
      Kxxsr = self.k(self.X, self.X[self.sridcs, :])
      Khxx = np.dot(Kxxsr, np.linalg.solve(self.V, Kxxsr.T))
      L += np.trace(np.dot(Khxx, Kx))
      L += np.dot(self.dYx.T, np.dot(Kx, self.dYx))

    ##################
    ##Compute ||L_i||^2
    ##################

    #compute C1 (used for S1 = Kxxi*C1*Kxix)
    A1 = np.linalg.solve(Kxi, np.eye(isz) - np.dot(Kxi, Xi))
    C1 = np.dot(A1, np.dot(Kxi, A1.T))
    L_i = np.trace(np.dot(Khxi, np.dot(Kxi_inv_KxxiTKxxi, np.dot(C1, Kxi_inv_KxxiTKxxi.T))))
    L_i += np.trace(np.dot(KdYi.T, np.dot(C1, KdYi)))

    ##################
    ##Compute <L, L_i>
    ##################
    #compute C2 (used for S2 = Kxxi*C2*Kxix)
    A2 = np.eye(isz) - np.dot(Xi, Kxi)
    C2 = np.dot(A2, np.linalg.solve(Kxi, A2.T).T)
    L_L_i = np.trace( np.dot(np.linalg.solve(self.V.T, Kxixsr.T).T, np.dot(KxxsrTKxxi, np.dot(C2, Kxi_inv_KxxiTKxxi.T))))
    L_L_i += np.trace(np.dot(KdY.T, np.dot(C2, KdYi)))

    return (L + L_i - 2*L_L_i).sum() #.sum() converts a 1x1 array to scalar (1x1 arr causes problems for scipy.minimize)


  def predict_f(self, Xt, cov_type='full'):
    Kxtxi = self.k(Xt, self.X_ind)
    mu = Kxtxi.dot(self.alpha)
    if cov_type=='full':
      sig = self.k(Xt) + Kxtxi.dot( self.C.dot(Kxtxi.T))
      return mu, sig
    else:
      sig = self.k(Xt, diag=True) +np.diag(Kxtxi.dot( self.C.dot(Kxtxi.T)))
      return mu, sig

