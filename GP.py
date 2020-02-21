import numpy as np
from car import Car
import random

class GP:
    """
    General purpose Gaussian process model
    :param X: input observations
    :param Y: output observations
    :param kernel: a GP kernel, defaults to squared exponential
    :param likelihood: a GPy likelihood
    :param inference_method: The :class:`~GPy.inference.latent_function_inference.LatentFunctionInference` inference method to use for this GP
    :rtype: model object
    :param Norm normalizer:
        normalize the outputs Y.
        Prediction will be un-normalized using this normalizer.
        If normalizer is True, we will normalize using Standardize.
        If normalizer is False, no normalization will be done.
    .. Note:: Multiple independent outputs are allowed using columns of Y
    """
    def __init__(self, X, Y, kernel='SE', omega=None, l=None, sigma=None, noise=None):
        self.X = X
        self.Y = Y
        self.kernel = kernel
        self.omega = omega
        self.l = l
        self.sigma = sigma
        self.noise = noise

    # Set/update input data for model
    def set_XY(self, X, Y):
        """
        Set the input data of the model
        :param X: input observations
        :type X: np.ndarray
        """
        self.X = X
        self.Y = Y
    
    # Evaluate kernel (squared exponential)
    def evaluate_kernel(self, x1, x2):
        diff = np.linalg.norm(x1 - x2)
        return self.sigma**2 * np.exp(-diff**2 / (2*self.l**2))

    def get_covariance(self):
        N = len(self.X)
        K = np.empty((N, N))
        for i in range(N):
            for j in range(i, N):
                val = self.evaluate_kernel(self.X[i,:], self.X[j,:])
                if (i == j):
                    K[i, i] = val
                else:
                    K[i, j] = val
                    K[j, i] = val
        return K

    def get_X_cov(self, Xnew):
        N = len(self.X)
        K_star = np.empty((N,1))
        for i in range(N):
            K_star[i,:] = self.evaluate_kernel(X[i,:], Xnew)
        return K_star

    def predict(self, Xnew):
        """
        Predict the function(s) at the new point Xnew. This includes the
        likelihood variance added to the predicted underlying function
        (usually referred to as f).
        """
        N = len(self.X)
        K = self.get_covariance()
        K_inv = np.inv(K + self.noise*np.eye(N))
        k_star = self.get_X_cov(Xnew)
        mean = np.matmul(np.transpose(np.matmul(K_inv, k_star)), self.Y)
        var = 0.0

        return mean, var

    def log_likelihood(self):
        raise NotImplementedError("this needs to be implemented to use the model class")

    def _log_likelihood_gradients(self):
        return self.gradient#.copy()

    def objective_function(self):
        """
        The objective function for the given algorithm.
        This function is the true objective, which wants to be minimized.
        Note that all parameters are already set and in place, so you just need
        to return the objective function here.
        For probabilistic models this is the negative log_likelihood
        (including the MAP prior), so we return it here. If your model is not
        probabilistic, just return your objective to minimize here!
        """
        return -float(self.log_likelihood()) - self.log_prior()

    def objective_function_gradients(self):
        """
        The gradients for the objective function for the given algorithm.
        The gradients are w.r.t. the *negative* objective function, as
        this framework works with *negative* log-likelihoods as a default.
        You can find the gradient for the parameters in self.gradient at all times.
        This is the place, where gradients get stored for parameters.
        This function is the true objective, which wants to be minimized.
        Note that all parameters are already set and in place, so you just need
        to return the gradient here.
        For probabilistic models this is the gradient of the negative log_likelihood
        (including the MAP prior), so we return it here. If your model is not
        probabilistic, just return your *negative* gradient here!
        """
        return -(self._log_likelihood_gradients() + self._log_prior_gradients())

def process_data(dat, dat_u):
    data_all = np.zeros(( 1, 8, len(dat[0]) ))
    data_u_all = np.zeros(( 1, 2, len(dat[0]) ))
    dat_u = np.array(dat_u)
    for i in range(len(dat)):
        Na = int(len(dat[i][0]) / 4)
        data_np = np.zeros((Na-1, 8, len(dat[i])))
        data_u_np = np.zeros((Na-1, 2, len(dat[i])))
        for j in range(len(dat[i])):
            for k in range(Na-1):
                data_np[k,0:4,j] = dat[i][j][0:4]
                data_np[k,4:8,j] = dat[i][j][4*(k+1):4*(k+2)]
                data_u_np[k,:,j] = dat_u[i,j,:]
        data_all = np.concatenate((data_all,data_np), axis=0)
        data_u_all = np.concatenate((data_u_all, data_u_np), axis=0)
    data_all = data_all[1:,:,:]
    data_u_all = data_u_all[1:,:,:]
    return data_all, data_u_all

def get_XY_from_data(dat, dat_u):
    # [p, v, ph, vh] -> [dp, dv, dp_h, dv_h]
    car = Car(0.0, 0.0)
    X = np.zeros(( dat.shape[0]*(dat.shape[2]-1), dat.shape[1] ))
    Y = np.zeros(( dat.shape[0]*(dat.shape[2]-1), dat.shape[1] ))
    for i in range(dat.shape[0]):
        for j in range(dat.shape[2] - 1):
            xr, xh = dat[i,0:4,j], dat[i,4:8,j]
            xr_n, xh_n = dat[i,0:4,j+1], dat[i,4:8,j+1]
            fp, gp, fv, gv = car.get_dynamics(xr)
            fp_h, _ , fv_h, _ = car.get_dynamics_human(xh)
            u = dat_u[i,:,j]
            dr = xr_n - np.concatenate([fp, fv]) - np.matmul(np.vstack([gp, gv]), u)
            dh = xr_n - np.concatenate([fp_h, fv_h])
            d = np.concatenate([dr, dh])
            x = np.concatenate([xr_n, xh_n])
            X[i*(dat.shape[2]-1) + j, :] = x
            Y[i*(dat.shape[2]-1) + j, :] = d

    return X, Y

def sample_data(X, Y, n_samples=1000):
    N = X.shape[0]
    idx = random.sample(range(0, N), min(n_samples, N))
    return X[idx,:], Y[idx,:]

if __name__ == '__main__':
    dat = np.load('train_data_i6.npy', allow_pickle=True)
    dat_u = np.load('train_data_u_i6.npy', allow_pickle=True)
    data_all, data_u_all = process_data(dat, dat_u)
    X, Y = get_XY_from_data(data_all, data_u_all)
    print("Imported Data")

    X_s, Y_s = sample_data(X, Y)
    gp = GP(X_s, Y_s, omega = np.eye(8), l = 1.0, sigma = 1.0, noise = 0.0)
    K = gp.get_covariance()
    print(K.shape)