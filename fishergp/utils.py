import sys
import time
import GPy

class ProgressBar(object):

  def __init__(self, message, xmin, xmax, dt=1.):
    self.t0 = time.time()
    self.tp = self.t0
    self.update_dt = dt
    self.message = message
    self.xmax = xmax
    self.xmin = xmin
    self.width = 32
    self.bar_prefix = ' |'
    self.bar_suffix = '| '
    self.bar_char = '#'
    self.empty_fill = ' '
    self.max_chs = len(str(self.xmax))
    self.update(self.xmin)
    self.wrote = False

  def update(self, x):
    t = time.time()
    if t - self.tp > self.update_dt:
      self.wrote = True
      self.x = x
      self.tp = t
      self.progress = float(self.x - self.xmin)/float(self.xmax-self.xmin)
      nbar = int(self.progress*self.width)
      nempty = self.width - nbar

      suffix = str(self.x).zfill(self.max_chs) + '/' + str(self.xmax)
      bar = self.bar_char * nbar
      empty = self.empty_fill *nempty
      line = ''.join([self.message, self.bar_prefix, bar, empty, self.bar_suffix, suffix])
      sys.stdout.write('\r')
      sys.stdout.write(line)
      sys.stdout.write(' '*50)
      sys.stdout.flush()

  def finish(self):
    self.update(self.xmax)
    if self.wrote:
     sys.stdout.write('\n')
     sys.stdout.flush()

def optimize_hyperparameters(X, Y, num_inducing):
  m = GPy.core.SparseGP(X, Y, X[np.random.randint(X.shape[0], size=num_inducing), :].copy(),
                        GPy.kern.RBF(input_dim=X.shape[1], ARD=True),
                        GPy.likelihoods.Gaussian())
  m.optimize('bfgs', max_iters=10000, messages=True)
  return np.asarray(m.rbf.lengthscale), np.asscalar(m.rbf.variance), np.asscalar(m.likelihood.variance)

def optimize_hyperparameters_post(X, Y, Z, sq_length_scales, kernel_var, likelihood_var):
  m = GPy.core.SparseGP(X, Y, Z,
                        GPy.kern.RBF(input_dim=X.shape[1], lengthscale=sq_length_scales.copy(), variance=kernel_var, ARD=True),
                        GPy.likelihoods.Gaussian(variance=likelihood_var))
  m.inducing_inputs.fix()
  m.optimize('bfgs', max_iters=10000, messages=True)
  return np.asarray(m.rbf.lengthscale), np.asscalar(m.rbf.variance), np.asscalar(m.likelihood.variance)


