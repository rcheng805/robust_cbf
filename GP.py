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
        self.X_s = X
        self.Y_s = Y
        self.kernel = kernel
        self.omega = omega
        self.l = l
        self.sigma = sigma
        self.noise = noise
        self.K = None

    # Sample subset of data for gradient computation
    def resample(self, n_samples=120):
        N = self.X.shape[0]
        idx = random.sample(range(0, N), min(n_samples, N))
        self.X_s, self.Y_s = self.X[idx,:], self.Y[idx,:]
        return self.X_s, self.Y_s

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

    # Evaluate derivative of kernel (w.r.t. length scale)
    def dk_dl(self, x1, x2):
        diff = np.linalg.norm(x1 - x2)
        return self.sigma**2 * np.exp(-diff**2 / (2*self.l**2)) * (diff*2 / (self.l**3))

    # Evaluate derivative of kernel (w.r.t. sigma)
    def dk_ds(self, x1, x2):
        diff = np.linalg.norm(x1 - x2)
        return 2*self.sigma * np.exp(-diff**2 / (2*self.l**2))

    # Get covariance matrix given current dataset
    def get_covariance(self):
        N = len(self.X_s)
        K = np.empty((N, N))
        for i in range(N):
            for j in range(i, N):
                val = self.evaluate_kernel(self.X_s[i,:], self.X_s[j,:])
                if (i == j):
                    K[i, i] = val
                else:
                    K[i, j] = val
                    K[j, i] = val
        self.K = K
        return K

    # Get derivative of covariance matrix (w.r.t. length scale and sigma)
    def get_dK(self):
        N = len(self.X_s)
        Kl = np.empty((N, N))
        Ks = np.empty((N, N))
        for i in range(N):
            for j in range(i, N):
                val_l = self.dk_dl(self.X_s[i,:], self.X_s[j,:])
                val_s = self.dk_ds(self.X_s[i,:], self.X_s[j,:])
                if (i == j):
                    Kl[i, i] = val_l
                    Ks[i, i] = val_s
                else:
                    Kl[i, j] = val_l
                    Kl[j, i] = val_l
                    Ks[i, j] = val_s
                    Ks[j, i] = val_s
        return Kl, Ks

    # Get gradient of negative log likelihood (w.r.t. length scale, sigma, omega)
    def likelihood_gradients(self):
        n = self.X_s.shape[0]
        d = self.Y_s.shape[1]
        K = self.get_covariance()
        Kl, Ks = self.get_dK()
        Kinv = np.linalg.inv(K)
        omegainv = np.linalg.inv(self.omega)
        A = np.matmul(Kinv, np.matmul(self.Y_s, np.matmul(omegainv, np.transpose(self.Y_s))))
        dL_dl = (d/2)*np.trace(np.matmul(Kinv, Kl)) + (1/2)*np.trace(np.matmul(-Kinv, np.matmul(Kl, A)))
        dL_ds = (d/2)*np.trace(np.matmul(Kinv, Ks)) + (1/2)*np.trace(np.matmul(-Kinv, np.matmul(Ks, A)))
        dL_domega = (n/2)*np.transpose(omegainv) - (1/2)*np.matmul(np.matmul(np.matmul(np.matmul(np.transpose(omegainv), np.transpose(self.Y_s)), np.transpose(Kinv)), self.Y_s), np.transpose(omegainv))
        return dL_dl, dL_ds, dL_domega

    def get_X_cov(self, Xnew):
        N = len(self.X)
        K_star = np.empty((N,1))
        for i in range(N):
            K_star[i,:] = self.evaluate_kernel(self.X[i,:], Xnew)
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

    # Compute negative log likelihood
    def log_likelihood(self):
        n = self.X_s.shape[0]
        d = self.Y_s.shape[1]
        self.get_covariance()
        A = np.matmul(np.matmul(np.matmul(np.linalg.inv(self.K), self.Y_s), self.omega), np.transpose(self.Y_s))
        L = (n*d/2)*np.log(2*np.pi) + (d/2)*np.log(np.linalg.det(self.K)) + (n/2)*np.log(np.linalg.det(self.omega)) + (1/2)*np.trace(A)
        return L

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

# Process data to get X,Y (training data)
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


if __name__ == '__main__':
    # Import dataset
    dat = np.load('train_data_i7.npy', allow_pickle=True)
    dat_u = np.load('train_data_u_i7.npy', allow_pickle=True)
    data_all, data_u_all = process_data(dat, dat_u)
    X, Y = get_XY_from_data(data_all, data_u_all)
    print("Imported Data")
    print(X.shape)

    # Initialize GP with random hyperparameters
    omega_init = 0.1*(np.random.rand(8,8) - 0.5)
    omega_init = np.eye(8) + (omega_init + omega_init.T)/2
    gp = GP(X, Y, omega = omega_init, l = 5*np.random.rand(), sigma = 5*np.random.rand(), noise = 0.0)

    # Define gradient descent parameters
    vals = []
    params_omega, params_sigma, params_l = [], [], []
    cur_o, cur_s, cur_l = gp.omega, gp.sigma, gp.l 
    iters, max_iters = 0, 2000
    grad_max = 25.0
    rate = 0.0004
    while iters < max_iters:
        prev_o, prev_s, prev_l = gp.omega, gp.sigma, gp.l
        
        # Get Gradients
        gp.resample()
        dL_dl, dL_ds, dL_domega = gp.likelihood_gradients()
        dL_dl = np.clip(dL_dl, -grad_max, grad_max)
        dL_ds = np.clip(dL_ds, -grad_max, grad_max)
        if (np.amax(dL_domega) > grad_max or np.amin(dL_domega) < grad_max):
            max_val = max(np.amax(dL_domega), abs(np.amin(dL_domega)))
            dL_domega = dL_domega * (grad_max / max_val)

        # Gradient descent
        cur_o = cur_o - rate * dL_domega
        cur_l = cur_l - rate * dL_dl
        cur_s = cur_s - rate * dL_ds

        # Update parameters
        gp.omega, gp.sigma, gp.l = cur_o, cur_s, cur_l

        iters = iters+1 #iteration count
        value = gp.log_likelihood()

        # Store and save updated parameters
        params_omega.append(gp.omega)
        params_sigma.append(gp.sigma)
        params_l.append(gp.l)
        vals.append(value)
        if (iters % 10 == 0):
            print(iters)
            print("sigma: ", gp.sigma)
            print("length: ", gp.l)
            print("Likelihood for this dataset: ", value)
        if (iters % 50 == 0):
            np.save('likelihood_vals_v7', vals)
            np.save('parameters_v7', [params_omega, params_sigma, params_l])