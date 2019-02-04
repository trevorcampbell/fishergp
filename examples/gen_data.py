import numpy as np

def gen_linear(N_train, N_test, seed):
  np.random.seed(seed)
  print('generating synthetic data')
  N=N_train+N_test
  likelihood_var = 0.01
  X = 2*np.random.rand(N,1)
  Y = 3*X + np.random.randn(N,1)*np.sqrt(likelihood_var)
  Xt = X[-N_test:, :]
  Yt = Y[-N_test:, :]
  X = X[:N_train, :]
  Y = Y[:N_train, :]
  return X, Y, Xt, Yt

def gen_synthetic(N_train, N_test, seed):
  np.random.seed(seed)
  print('generating synthetic data')
  N=N_train+N_test
  likelihood_var = 0.01
  X = 2*np.random.rand(N,1)
  Y = np.sin(12*X) + 0.66*np.cos(25*X) + np.random.randn(N,1)*np.sqrt(likelihood_var) + 3
  Xt = X[-N_test:, :]
  Yt = Y[-N_test:, :]
  X = X[:N_train, :]
  Y = Y[:N_train, :]
  return X, Y, Xt, Yt

def gen_from_file(dataset_name, N_train, N_test, seed):
  np.random.seed(seed)
  print('loading ' + dataset_name + ' data')
  data = np.load('datasets/'+dataset_name+'.npy')
  print('shuffling data')
  np.random.shuffle(data)
  
  print('extracting training/test sets')
  #extract train, test sets
  X = data[:N_train, :-1]
  Y = data[:N_train, -1][:, np.newaxis]
  Xt = data[N_train:N_train+N_test, :-1]
  Yt = data[N_train:N_train+N_test, -1][:, np.newaxis]
  return X, Y, Xt, Yt

def standardize(X, Y, Xt, Yt):
  print('standardizing using training data')
  #standardize input/output
  Ymu = Y.mean(axis=0)
  Xmu = X.mean(axis=0)
  Xcov = np.atleast_2d(np.cov(X, rowvar=False))
  Ycov = np.atleast_2d(np.cov(Y, rowvar=False))
  X -= Xmu
  Y -= Ymu
  Xt -= Xmu
  Yt -= Ymu
  u, V = np.linalg.eigh(Xcov)
  XZ = V/np.sqrt(u)
  X[:] = X.dot(XZ)
  Xt[:] = Xt.dot(XZ)
  u, V = np.linalg.eigh(Ycov)
  YZ = V/np.sqrt(u)
  Y[:] = Y.dot(YZ)
  Yt[:] = Yt.dot(YZ)
  return Xmu, XZ, Ymu, YZ	

