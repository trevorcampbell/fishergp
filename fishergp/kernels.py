import autograd.numpy as np

class Kernel:
  def __call__(self, X, Y):
    raise NotImplementedError  

class GaussianKernel(Kernel):
  def __init__(self, gamma, sigma):
    self.gamma = gamma
    self.sigma = sigma
  
  def __call__(self, X, Y=None, diag=False):
    if diag:
      return self.sigma*np.ones(X.shape[0])

    if Y is None:
      Y = X

    Ynsq = (Y**2/self.gamma**2).sum(axis=1)
    Xnsq = (X**2/self.gamma**2).sum(axis=1)
    return self.sigma*np.exp(-0.5*(Xnsq[:, np.newaxis] + Ynsq - 2.*np.dot(X/self.gamma**2, Y.T)))

