import numpy as np
import random
from scipy.stats import chi2
import pickle

from car import Car

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
    def __init__(self, X, Y, kernel='SE', omega=None, l=None, sigma=None, noise=None, horizon=20):
        self.X = X
        self.Y = Y
        self.X_s = X
        self.Y_s = Y
        self.kernel = kernel
        self.omega = omega
        self.l = l
        self.sigma = sigma
        self.noise = noise
        self.K = None                                   # Train GP
        self.K_obs = np.empty((horizon, horizon))       # Observation GP
        self.K_star = np.empty(horizon)
        self.N_data = 0
        self.horizon = horizon
        self.X_obs = []
        self.Y_obs = []
        self.count = -1

    def load_parameters(self, file_name):
        # open a file, where you stored the pickled data
        file = open(file_name, 'rb')
        data = pickle.load(file)
        file.close()
        self.omega = data['omega']
        self.sigma = data['sigma']
        self.l = data['l']

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

    # Add observed data to GP
    def add_data(self, x, y):
        self.X_obs.append(np.copy(x))
        self.Y_obs.append(np.copy(y))
        if (len(self.X_obs) != len(self.Y_obs)):
            print("ERROR: Input/output data dimensions don't match")
        if (len(self.X_obs) > self.horizon):
            self.X_obs.pop(0)
            self.Y_obs.pop(0)
        self.N_data = len(self.X_obs)
        self.count += 1
        if (self.count >= self.horizon):
            self.count = 0

    # Get K*
    def get_X_cov(self, Xnew):
        N = self.N_data
        for i in range(N):
            self.K_star[i] = self.evaluate_kernel(self.X_obs[i], Xnew)
        return self.K_star[0:N]

    # Get covariance matrix given current dataset
    def get_obs_covariance(self):
        N = self.N_data
        K = np.empty((N, N))
        for i in range(N):
            for j in range(i, N):
                val = self.evaluate_kernel(self.X_obs[i], self.X_obs[j])
                if (i == j):
                    K[i, i] = val + self.noise
                else:
                    K[i, j] = val
                    K[j, i] = val
        self.K_obs = K
        return K

    # Update covariance matrix given new data (run after add_data)
    def update_obs_covariance(self):
        N = self.N_data
        x = self.X_obs[-1]
        for i in range(N):
            val = self.evaluate_kernel(x, self.X_obs[i])
            if (i == N-1):
                self.K_obs[N-1, N-1] = val + self.noise
            else:
                self.K_obs[i, N-1] = val
                self.K_obs[N-1, i] = val
        return self.K_obs[0:N, 0:N]

    '''
    # Get covariance matrix given current dataset
    def update_obs_covariance(self):
        N = self.N_data
        for i in range(N):
            val = self.evaluate_kernel(self.X_obs[self.count], self.X_obs[i])
            if (self.count == i):
                self.K_obs[i, i] = val + self.noise
            else:
                self.K_obs[i, self.count] = val
                self.K_obs[self.count, i] = val
        return self.K_obs[0:N,0:N]
    '''

    # Predict function at new point Xnew
    def predict(self, Xnew):
        N = self.N_data
        # K = self.get_covariance()
        K_inv = np.linalg.inv(self.K_obs[0:N,0:N])
        k_star = self.get_X_cov(Xnew)
        mean = np.matmul(np.transpose(np.matmul(K_inv, k_star)), self.Y_obs[0:N])
        Sigma = self.evaluate_kernel(Xnew, Xnew) + self.noise - np.matmul(np.transpose(np.matmul(K_inv, k_star)), k_star)
        cov = np.kron(Sigma, self.omega)
        return mean, cov
    
    def extract_norms(self, cov_d, p_threshold=0.01):
        # Extract chi2 value
        Nd = 2
        kd = chi2.isf(p_threshold, Nd)
        D_p, _ = np.linalg.eig(cov_d[0:2,0:2])
        # D_p = np.abs(D_p)
        lamda_max = 1 / (np.min(D_p))
        zp = np.sqrt(kd / lamda_max)
        D_v, _ = np.linalg.eig(cov_d[2:4,2:4])
        # D_v = np.abs(D_v)
        lamda_max = 1 / (np.min(D_v))
        zv = np.sqrt(kd / lamda_max)
        return zp, zv

    def extract_box(self, cov_d, p_threshold=0.01):
        # Extract chi2 value
        Nd = 4
        kd = chi2.isf(p_threshold, Nd)
        D, V = np.linalg.eig(cov_d)

        # Populate bounding polytope
        G = np.zeros((8, 4))
        g = np.zeros(8)
        for i in range(4):
            G[2*i,:] = -V[:,i]
            g[2*i] = np.sqrt(kd*abs(D[i])) #  - np.dot(V[:,i], m_d)
            G[2*i+1,:] = V[:,i]
            g[2*i+1] = np.sqrt(kd*abs(D[i])) # + np.dot(V[:,i], m_d)

        # Extract norm
        # d_norm = np.sqrt(kd / (1 / np.min(D)))

        return G, g

# Helper Function to Load in Data
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

if __name__ == '__main__':
    print("GP Program Run")