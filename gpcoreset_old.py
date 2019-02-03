import autograd.numpy as np
from autograd.misc.optimizers import adam
from scipy.linalg import solve_triangular

def _nystrom(X, idcs, k, ridge=1e-9):
  Kx = k(X, X[idcs, :])
  Kxx = Kx[idcs, :]
  lmb, U = np.linalg.eigh(Kxx)
  lmb += ridge*lmb.max()
  return Kx.dot(U)/np.sqrt(lmb)

class GP(object):

  def __init__(self, X, Y, kern, likelihood_variance):
    self.X = X
    self.Y = Y
    self.k = kern
    self.lvar = likelihood_variance

  #options are nystrom, subsample, full
  def train(self, method='full', subsample_idcs=None, ridge=1e-9):
    #below uses the representation K=UDU^T and Woodbury Matrix Id
    #(K+sI)^{-1} = s^{-1}(I - U(s D^{-1} + U^TU)^{-1}U^T)
    self.method = method
    self.idcs = subsample_idcs

    #the prediction f = kx^T alpha, cov = 1[add_kxx]*kxx + kx^TVkx
    if method == 'full':
      self.alpha = np.linalg.solve(self.k(self.X)+self.lvar*np.eye(self.X.shape[0]), self.Y)
      self.V = -np.linalg.inv(self.k(self.X)+self.lvar*np.eye(self.X.shape[0]))
      self.Xpred = self.X
      self.add_kxx = True
      self.use_V = True
    elif method == 'subsample':
      if self.idcs is None:
        raise ValueError()
      self.alpha = np.linalg.solve(self.k(self.X[self.idcs, :)+self.lvar*np.eye(self.X[self.idcs,:].shape[0]), self.Y[self.idcs, :])
      self.V = -np.linalg.inv(self.k(self.X[self.idcs,:])+self.lvar*np.eye(self.X[self.idcs,:].shape[0]))
      self.Xpred = self.X[self.idcs, :]
      self.add_kxx = True
      self.use_V = True
    elif method == 'subset_regressors':
      Kx = self.k(self.X, self.X[self.idcs, :])
      Kxx = Kx[self.idcs, :]
      C = Kx.T.dot(Kx) + self.lvar*Kxx
      lmbmax = np.linalg.eigvalsh(C).max()
      C += ridge*lmbmax*np.eye(C.shape[0])
      self.V =  self.lvar*np.linalg.inv(C)
      self.alpha = self.V.dot(Kx.T.dot(self.Y))/self.lvar
      self.Xpred = self.X[self.idcs, :]
      self.add_kxx = False
      self.use_V = True
    elif method == 'nystrom':
      if self.idcs is None:
        raise ValueError()
      Z = _nystrom(self.X, self.idcs, self.k, ridge)
      self.Z = Z
      self.C = np.linalg.inv(self.lvar*np.eye(Z.shape[1]) + Z.T.dot(Z))
      self.alpha = (self.Y - Z.dot(self.C.dot(Z.T.dot(self.Y))))/self.lvar
      self.Xpred = self.X
      self.use_V = False
      self.add_kxx = True
    else:
      raise NotImplementedError()
      
  def predict_y(self, Xt, cov_type='full'):
    mu, sig = self.predict_f(Xt, cov_type)
    if cov_type == 'full':
      sig += np.diag(lvar*np.ones(sig.shape[0]))
    else:
      sig += self.lvar
    return mu, sig
  
  def predict_f(self, Xt, cov_type='full'):
      XtX = self.k(Xt, self.Xpred)
      mu = XtX.dot(self.alpha)
      if self.use_V:
        sig = XtX.dot(self.V).dot(XtX.T)
      else:
        sig = XtX.dot(XtX.T)/self.lvar - XtX.dot(self.C.dot(self.Z.T.dot(XtX.T)))/self.lvar
      if self.add_kxx: 
        sig += self.k(Xt)
    if cov_type=='full':
      return mu, sig
    else:
      return mu, np.diag(sig).copy()

class CoresetGP(object):

  def __init__(self, X, Y, kern, likelihood_variance, kern_prime=None):
    self.X = X
    self.Y = Y
    self.k = kern
    self.kp = kern_prime if kern_prime is not None else kern
    self.lvar = likelihood_variance
    self.precond = preconditioned

  def train(self, n_coreset_itrs=100, khat_project_method='subset_regressors', k_project_method='nystrom', subsample_idcs=None, preconditioned=True, ridge=1e-9):

    self.khat_method = khat_project_method
    self.k_method = k_project_method
    self.idcs = subsample_idcs

    #get low rank approx of K = ZZ^T and K' = Z'Z'^T
    if k_project_method == 'full':
      lmb, U = np.linalg.eigh(self.k(self.X))
      lmb += ridge*lmb.max()
      Z = U*np.sqrt(lmb)
      lmb, U = np.linalg.eigh(self.kp(self.X))
      lmb += ridge*lmb.max()
      Zp = U*np.sqrt(lmb)
    elif k_project_method == 'nystrom':
      #get nystrom approximation of K: K = ZZ^T
      Z = _nystrom(self.X, self.idcs, self.k, ridge)
      Zp = _nystrom(self.X, self.idcs, self.kp, ridge)
    else:
      raise NotImplementedError()

    #get low rank approx of posterior mean, covariance 
    if khat_project_method == 'full':
      K = self.k(self.X)
      M = K.dot( np.linalg.solve(K+self.lvar*np.eye(K.shape[0]), self.Y) )
      V = K - K.dot(np.linalg.solve(K+self.lvar*np.eye(K.shape[0]), K))
    elif khat_project_method == 'nystrom':
      ZTZ = Z.T.dot(Z)
      sIZTZinv_ZTZ = np.linalg.solve(self.lvar*np.eye(ZTZ.shape[0]) + ZTZ, ZTZ)
      C = np.eye(ZTZ.shape[0]) - 1.0/self.lvar*ZTZ + 1.0/self.lvar*ZTZ.dot(sIZTZinv_ZTZ)
      V = Z.dot(np.linalg.cholesky(C))
      ZTY = Z.T.dot(self.Y)
      M = 1.0/self.lvar*Z.dot( (np.eye(ZTZ.shape[0]) - sIZTZinv_ZTZ.T).dot(ZTY) )
    elif khat_project_method == 'subsample':
      Kx = self.k(self.X, self.X[self.idcs, :])
      Kxx = Kx[self.idcs, :]
      #TODO: bug here? should this not have s^2*I in it?
      M = Kx.dot(np.linalg.inv(Kxx+ 1e-9*np.eye(Kxx.shape[0])).dot(self.Y[self.idcs, :]))
      lmb, U = np.linalg.eigh(Kxx + 1e-9*np.eye(Kxx.shape[0]))
      V = Kx.dot(U)*np.sqrt(1./lmb - 1./(lmb+self.lvar))
    elif khat_project_method == 'subset_regressors':
      Kx = self.k(self.X, self.X[self.idcs, :])
      Kxx = Kx[self.idcs, :]
      V =  solve_triangular( np.linalg.cholesky(Kx.T.dot(Kx) + self.lvar*Kxx + 1e-9*np.eye(Kxx.shape[0])), Kx.T ).T
      M = V.dot(V.T.dot(self.Y))
    else:
      raise NotImplementedError()
    Zhat = np.hstack((V, M-self.Y))

    

    #run coreset construction
    self.wts = np.zeros(self.X.shape[0])
    zw = np.zeros((Zhat.shape[1], Zp.shape[1]))
    if not preconditioned:
      #compute N x J x J representation of N data, normalize
      Zc = Zhat[:, :, np.newaxis]*Zp[:, np.newaxis, :]
      norms = np.sqrt((Zc**2).sum(axis=(1, 2)))
      Zc /= norms[:, np.newaxis, np.newaxis]
      #compute J x J sum of vectors, normalize
      zs = Zc.sum(axis=0)
      snorm = np.sqrt((zs**2).sum())
      zs /= snorm
    for i in range(n_coreset_itrs):
      #compute current preconditioned data
      #if not preconditioned, just leave Zc and Zs alone
      if preconditioned:
        ZXiZp = Z[self.wts>0, :].T.dot(np.linalg.solve(self.k(self.X[self.wts > 0, :]) + self.lvar*1./self.wts[self.wts>0], Zp[self.wts>0, :]))
        Zc = Zhat[:, :, np.newaxis]*((Zp - Z.dot(ZXiZp))[:, np.newaxis, :])
        zs = Zc.sum(axis=0)
        snorm = np.sqrt((zs**2).sum())
        zs /= snorm
        norms = np.sqrt((Zc**2).sum(axis=(1,2)))
        Zc /= norms[:, np.newaxis, np.newaxis]
        zw = (self.wts[:, np.newaxis, np.newaxis]*Zc).sum(axis=0)
      
      #current residual
      r = snorm*zs - zw

      #find max aligned data vec
      f = (Zc*r[np.newaxis, :, :]).sum(axis=(1, 2)).argmax()

      #get line search val
      gamma = (Zc[f, :, :]*r).sum()
      
      #update wts, zw
      self.wts[f] += gamma
      zw += gamma*Zc[f, :, :]

    #make the weights apply to original unnormalized data
    self.wts /= norms

    #use the coreset to train the GP
    self.Xw = self.X[self.wts > 0, :]
    self.Yw = self.Y[self.wts > 0, :]
    self.wts = self.wts[self.wts > 0]
    self.invKw = np.linalg.inv(self.k(self.Xw) + self.lvar*np.diag(1.0/self.wts))
    self.invKwY = self.invKw.dot(self.Yw)

    
  def predict_y(self, Xt, cov_type='full'):
    mu, sig = self.predict_f(Xt, cov_type)
    if cov_type == 'full':
      sig += np.diag(lvar*np.ones(sig.shape[0]))
    else:
      sig += self.lvar
    return mu, sig
  
  def predict_f(self, Xt, cov_type='full'):
    XtXw = self.k(Xt, self.Xw)
    XtXt = self.k(Xt)
    mu = XtXw.dot(self.invKwY)
    sig = XtXt - XtXw.dot(self.invKw).dot(XtXw.T)
    return mu, sig

class InducingGP(object):
  
  def __init__(self, X, Y, kern, likelihood_variance, kern_prime=None):
    self.X = X
    self.Y = Y
    self.k = kern
    self.kp = kern_prime if kern_prime is not None else kern
    self.lvar = likelihood_variance
    self.precond = preconditioned

  def train(self, n_coreset_itrs=100, khat_project_method='subset_regressors', subsample_idcs=None, preconditioned=True):
    self.idcs = subsample_idcs
    self.khat_method = khat_project_method

    #get matrices necessary to compute khat(x,x') and muhat(x)
    if khat_project_method == 'full':
      K = self.k(self.X)
      M = np.linalg.solve(K+self.lvar*np.eye(K.shape[0]), self.Y)
      V = np.linalg.inv(K+self.lvar*np.eye(K.shape[0]))
      self.Xpred = self.X[self.idcs, :]
    elif khat_project_method == 'nystrom':
      #maybe TODO; but since nystrom can give nonPD covariances, maybe not
      raise NotImplementedError()
    elif khat_project_method == 'subsample':
      Kx = self.k(self.X, self.X[self.idcs, :])
      Kxx = Kx[self.idcs, :]
      M = np.linalg.solve(Kxx+ self.lvar*np.eye(Kxx.shape[0]), self.Y[self.idcs, :])
      V = np.linalg.inv(Kxx+self.lvar*np.eye(Kxx.shape[0])
      self.Xpred = self.X[self.idcs, :]
    elif khat_project_method == 'subset_regressors':
      Kx = self.k(self.X, self.X[self.idcs, :])
      Kxx = Kx[self.idcs, :]
      M =  np.linalg.solve(Kx.T.dot(Kx) + self.lvar*Kxx + 1e-9*np.eye(Kxx.shape[0]), Kx.T.dot(self.Y) )
      V = self.lvar*np.linalg.inv(Kx.T.dot(Kx) + self.lvar*Kxx + 1e-9*np.eye(Kxx.shape[0]))
      self.Xpred = self.X[self.idcs, :]
    else:
      raise NotImplementedError()

    #create objective function
    def _objective():
      pass

    #output print callback
    def callback(prms, t, g):
      print('Iteration {} lower bound {}'.format(t, -objective(prms, t)))
    
    #optimize
    adam(gradient, init_params, step_size=0.1, num_iters=1000, callback=cbk)
   

  def predict_y(self, Xt, cov_type='full'):
    mu, sig = self.predict_f(Xt, cov_type)
    if cov_type == 'full':
      sig += np.diag(lvar*np.ones(sig.shape[0]))
    else:
      sig += self.lvar
    return mu, sig
  
  def predict_f(self, Xt, cov_type='full'):
    XtXw = self.k(Xt, self.Xw)
    XtXt = self.k(Xt)
    mu = XtXw.dot(self.invKwY)
    sig = XtXt - XtXw.dot(self.invKw).dot(XtXw.T)
    return mu, sig
