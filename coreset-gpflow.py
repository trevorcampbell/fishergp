from __future__ import absolute_import, print_function

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import gpflow
import numpy as np
import tensorflow as tf
from gpflow import likelihoods
from gpflow import settings
from gpflow.params import DataHolder
from gpflow.decors import name_scope, autoflow, params_as_tensors
from gpflow.densities import multivariate_normal
from gpflow.models.model import GPModel

class CoresetGPR(GPModel):
    """
    Coreset Gaussian Process Regression.
    Multiple columns of Y are treated independently.
    Low-rank Nystrom-like approximation at Z
    """
    def __init__(self, X, Y, Z, M, kern, mean_function=None, **kwargs):
        """
        wts is a weight vector, length N
        X is a data matrix, size N x D
        Z is a data matrix, size M x D, M << N
        Y is a data matrix, size N x R
        kern, mean_function are appropriate GPflow objects
        """
        likelihood = likelihoods.Gaussian()
        X = DataHolder(X)
        Y = DataHolder(Y)
        GPModel.__init__(self, X, Y, kern, likelihood, mean_function, **kwargs)
        self.num_latent = Y.shape[1]
        self.coreset_size = M
        self.num_coarse = Z.shape[0]
        self.num_train_data = X.shape[0]
        self.Z = DataHolder(Z)
        
    @name_scope('likelihood')
    @params_as_tensors
    def _build_likelihood(self):
        #raise NotImplementedError()
        """
        Construct a tensorflow function to compute the likelihood.
            \log p(Y | theta).
        """
        K = self.kern.K(self.X) +tf.eye(tf.shape(self.X)[0], dtype=settings.tf_float)*self.likelihood.variance #+ tf.diag(self.inv_wts) * self.likelihood.variance
        L = tf.cholesky(K)
        m = self.mean_function(self.X)

        return multivariate_normal(self.Y, m, L)

    @name_scope('predict')
    @params_as_tensors
    def _build_predict(self, Xnew, full_cov=False):
        """
        Xnew is a data matrix, point at which we want to predict
        This method computes
            p(F* | Y )
        where F* are points on the GP at Xnew, Y are noisy observations at X.
        """

        ######################
        #first, compute coarse approximate solution khat, muhat
        ######################
        Kzz = self.kern.K(self.Z) + 1e-8*tf.eye(tf.shape(self.Z)[0], dtype=settings.tf_float)
        Kzx = self.kern.K(self.Z, self.X)

        B = tf.matrix_triangular_solve(tf.cholesky(Kzz), Kzx)
        # B is num_coarse x num_train_data
        #K(X, X) approx = B^T B

        BB = tf.matmul(B, B, transpose_b=True)
        C = self.likelihood.variance*tf.eye(tf.shape(Kzz)[0], dtype=settings.tf_float) + BB
        D = tf.matrix_triangular_solve(tf.cholesky(C), B)
        ###B^T(s^2 I + B B^T)^{-1} B approx = D^T D

        DDY = tf.matmul(D, tf.matmul(D, self.Y), transpose_a = True)
        muhat = 1./self.likelihood.variance*tf.matmul(B, tf.matmul(B, self.Y - DDY), transpose_a=True)

        F = tf.eye(tf.shape(Kzz)[0], dtype=settings.tf_float)
        F -= 1./self.likelihood.variance*BB
        DB = tf.matmul(D, B, transpose_b = True)
        F += 1./self.likelihood.variance*tf.matmul(DB, DB, transpose_a=True)
        L = tf.cholesky(F)
        Lhat = tf.matmul(B, L, transpose_a = True)
        #khat = Lhat * Lhat^T 

        #construct matrix of feature vectors for coreset construction
        #for two low-rank matrices B = QQ^T and C= LL^T, if A_nm = B_nm * C_nm, then A = MM^T where cols of M are all pairwise hadamard products of cols of L and Q
        featvecs_k = tf.concat([ tf.tile(tf.expand_dims(Lhat[:, i], -1), [1, self.num_coarse]) for i in range(self.num_coarse)], axis=1) #tf.tile(tf.transpose(B), [1, self.num_coarse])  * 
        featvecs_y = tf.concat([ tf.tile(tf.expand_dims(muhat[:, i], -1) - tf.expand_dims(self.Y[:, i], -1), [1, self.num_coarse]) for i in range(self.num_latent)], axis=1) #tf.tile(tf.transpose(B), [1, self.num_latent])  
        featvecs = 1./self.likelihood.variance**2*tf.concat([featvecs_k, featvecs_y], axis=1)

        #run coreset construction with weights w, inverse weights invw, and data subset Xw, Yw
        wts = self.construct_coreset(featvecs, self.coreset_size)
        
        idcs = tf.not_equal(wts, 0)
        self.w = tf.boolean_mask(wts, idcs)
        self.invw = 1./self.w
        self.Xw = tf.boolean_mask(self.X, idcs)
        self.Yw = tf.boolean_mask(self.Y, idcs)

        ##self.Xw = self.X
        ##self.Yw = self.Y
        ##self.w = tf.ones([self.num_train_data], dtype=settings.tf_float)
        ##self.invw = tf.ones([self.num_train_data], dtype=settings.tf_float)

        #run weighted gp regression
        Kx = self.kern.K(self.Xw, Xnew)
        K = self.kern.K(self.Xw) + tf.diag(self.invw) * self.likelihood.variance
        #K = self.kern.K(self.X) + tf.eye(tf.shape(self.X)[0], dtype=settings.tf_float) * self.likelihood.variance
        L = tf.cholesky(K)
        A = tf.matrix_triangular_solve(L, Kx, lower=True)
        V = tf.matrix_triangular_solve(L, self.Yw - self.mean_function(self.Xw))
        fmean = tf.matmul(A, V, transpose_a=True) + self.mean_function(Xnew)
        if full_cov:
            fvar = self.kern.K(Xnew) - tf.matmul(A, A, transpose_a=True)
            shape = tf.stack([1, 1, tf.shape(self.Yw)[1]])
            fvar = tf.tile(tf.expand_dims(fvar, 2), shape)
        else:
            fvar = self.kern.Kdiag(Xnew) - tf.reduce_sum(tf.square(A), 0)
            fvar = tf.tile(tf.reshape(fvar, (-1, 1)), [1, tf.shape(self.Yw)[1]])
        return fmean, fvar

    @name_scope('coreset')
    @params_as_tensors
    def construct_coreset(self, V, M):
        # sigma_n
        norms = tf.norm(V, axis=1)
        # sigma
        norm_sum = tf.reduce_sum(norms)
        # sum vector L and weighted sum Lw
        L = tf.expand_dims(tf.reduce_sum(V, axis=0), -1)
        Lw = tf.zeros_like(L)
        #weights
        w = tf.zeros_like(norms)
        for i in range(M):
            r = L - Lw
            idx = tf.argmax(tf.matmul(V / tf.expand_dims(norms, -1), r))
            q = norm_sum/norms[idx[0]]*tf.expand_dims(V[idx[0], :], -1) - Lw
            gamma = tf.matmul(tf.transpose(q), r) / tf.square(tf.norm(q))
            w *= (1.-gamma)
            w += gamma*norm_sum/norms[idx[0]]*tf.one_hot(idx[0], self.num_train_data, dtype=settings.tf_float)
            Lw = Lw + gamma*q
        return tf.squeeze(w)
    

def plot_GP(m, wts=None):
    xx = np.linspace(-0.2, 2.2, 200)[:,None]
    mean, var = m.predict_y(xx)
    plt.figure(figsize=(12, 6))
    plt.plot(X, Y, 'kx', mew=2)
    plt.plot(xx, mean, 'b', lw=2)
    if wts is not None:
        plt.plot(X[wts > 0], Y[wts > 0], 'ro', mew=2)
    plt.fill_between(xx[:,0], mean[:,0] - 2*np.sqrt(var[:,0]), mean[:,0] + 2*np.sqrt(var[:,0]), color='blue', alpha=0.2)
    plt.xlim(-0.2, 2.2)
    
    plt.figure()
    plt.plot(xx, np.sqrt(var[:,0]))
    plt.yscale('log')


N = 200
N_inducing = 50
M = 40
idcs = np.arange(N)
np.random.shuffle(idcs)
idcs = idcs[:N_inducing]
X = 2*np.random.rand(N,1)
Z = X[idcs, :]
Z = X.copy()
Y = np.sin(12*X) + 0.66*np.cos(25*X) + np.random.randn(N,1)*0.1 + 3

#plt.figure()
#plt.plot(X, Y, 'kx', mew=2)

#m = gpflow.models.GPR(X, Y, kern=gpflow.kernels.RBF(1, lengthscales=[.1]))
#m.likelihood.variance = 0.01
#m.compile()
#plot_GP(m)


cm = CoresetGPR(X, Y, Z, M, kern=gpflow.kernels.RBF(1, lengthscales=[.1]))
cm.likelihood.variance = 0.01
cm.compile()

plot_GP(cm)

#
plt.show()

