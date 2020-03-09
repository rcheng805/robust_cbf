import numpy as np
from car import Car
import random

kSave = True

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
    def __init__(self, X, Y, kernel='SE', omega=None, L=None, l=None, sigma=None, noise=None, horizon=50):
        self.X = X
        self.Y = Y
        self.X_s = X
        self.Y_s = Y
        self.kernel = kernel
        self.omega = omega
        self.L = L      # Cholesky factorization of omega
        self.l = l
        self.sigma = sigma
        self.noise = noise
        self.K = None                       # Train GP
        self.K_obs = np.empty((horizon, horizon))       # Observation GP
        self.K_star = np.empty(horizon)
        self.N_data = 0
        self.horizon = horizon
        self.X_obs = []
        self.Y_obs = []

    # Sample subset of data for gradient computation
    def resample(self, n_samples=80):
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
        return self.sigma**2 * np.exp(-diff**2 / (2*self.l**2)) * (diff**2 / (self.l**3))

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
                    K[i, i] = val + self.noise
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
        # self.omega = np.matmul(self.L, np.transpose(self.L))
        omegainv = np.linalg.inv(self.omega)
        A = np.matmul(Kinv, np.matmul(self.Y_s, np.matmul(omegainv, np.transpose(self.Y_s))))
        dL_dl = (d/2)*np.trace(np.matmul(Kinv, Kl)) + (1/2)*np.trace(np.matmul(-Kinv, np.matmul(Kl, A)))
        dL_ds = (d/2)*np.trace(np.matmul(Kinv, Ks)) + (1/2)*np.trace(np.matmul(-Kinv, np.matmul(Ks, A)))
        dL_domega = (n/2)*np.transpose(omegainv) - (1/2)*np.matmul(np.matmul(np.matmul(np.matmul(np.transpose(omegainv), np.transpose(self.Y_s)), np.transpose(Kinv)), self.Y_s), np.transpose(omegainv))
        # dL_dL = n*np.linalg.pinv(self.L) - np.matmul(np.matmul(np.matmul(np.matmul(np.matmul(np.transpose(self.L), omegainv), np.transpose(self.Y_s)), Kinv), self.Y_s), omegainv)
        # dL_dL = np.tril(dL_dL)
        return dL_dl, dL_ds, dL_domega

    # Compute negative log likelihood
    def log_likelihood(self):
        n = self.X_s.shape[0]
        d = self.Y_s.shape[1]
        # self.omega = np.matmul(self.L, np.transpose(self.L))
        self.get_covariance()
        A = np.matmul(np.matmul(np.matmul(np.linalg.inv(self.K), self.Y_s), np.linalg.inv(self.omega)), np.transpose(self.Y_s))
        L = (n*d/2)*np.log(2*np.pi) + (d/2)*np.log(np.linalg.det(self.K)) + (n/2)*np.log(np.linalg.det(self.omega)) + (1/2)*np.trace(A)
        return L

    def add_data(self, x, y):
        self.X_obs.append(x)
        self.Y_obs.append(y)
        if (len(self.X_obs) != len(self.Y_obs)):
            print("ERROR: Input/output data dimensions don't match")
        if (len(self.X_obs) > self.horizon):
            self.X_obs = self.X_obs.pop(0)
            self.Y_obs = self.Y_obs.pop(0)
        self.N_data = len(self.X_obs)

    # Get K*
    def get_X_cov(self, Xnew):
        N = self.N_data
        for i in range(N):
            self.K_star[i] = self.evaluate_kernel(self.X_obs[i], Xnew)
        return self.K_star[0:N]

    # Update covariance matrix given new data (run after add_data)
    def update_obs_covariance(self):
        N = self.N_data
        x = self.X_obs[-1]
        for i in range(N):
            val = self.evaluate_kernel(x, self.X_obs[i])
            if (i == N-1):
                self.K_obs[N-1, N-1] = val
            else:
                self.K_obs[i, N-1] = val
                self.K_obs[N-1, i] = val
        return self.K_obs[0:N, 0:N]

    # Predict function at new point Xnew
    def predict(self, Xnew):
        N = self.N_data
        # K = self.get_covariance()
        K_inv = np.inv(self.K_obs[0:N,0:N] + self.noise*np.eye(N))
        k_star = self.get_X_cov(Xnew)
        mean = np.matmul(np.transpose(np.matmul(K_inv, k_star)), self.Y[0:N])
        Sigma = self.evaluate_kernel(Xnew, Xnew) - np.matmul(np.transpose(np.matmul(K_inv, k_star)), k_star)
        cov = np.kron(Sigma, self.omega)
        return mean, var

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
    '''
    # Import dataset
    dat = np.load('train_data_i6.npy', allow_pickle=True)
    dat_u = np.load('train_data_u_i6.npy', allow_pickle=True)
    data_all, data_u_all = process_data(dat, dat_u)
    X1, Y1 = get_XY_from_data(data_all, data_u_all)
    print("Imported Data")
    dat = np.load('train_data_i7.npy', allow_pickle=True)
    dat_u = np.load('train_data_u_i7.npy', allow_pickle=True)
    data_all, data_u_all = process_data(dat, dat_u)
    X2, Y2 = get_XY_from_data(data_all, data_u_all)
    print("Imported Data")
    dat = np.load('train_data_i8.npy', allow_pickle=True)
    dat_u = np.load('train_data_u_i8.npy', allow_pickle=True)
    data_all, data_u_all = process_data(dat, dat_u)
    X3, Y3 = get_XY_from_data(data_all, data_u_all)
    print("Imported Data")
    dat = np.load('train_data_i9.npy', allow_pickle=True)
    dat_u = np.load('train_data_u_i9.npy', allow_pickle=True)
    data_all, data_u_all = process_data(dat, dat_u)
    X4, Y4 = get_XY_from_data(data_all, data_u_all)
    print("Imported Data")

    X = np.concatenate((X1, X2, X3, X4), axis=0)
    Y = np.concatenate((Y1, Y2, Y3, Y4), axis=0)
    print(X.shape)
    print(Y.shape)
    np.save('train_data_all.npy', [X, Y])
    '''
    dat = np.load('train_data.npy', allow_pickle=True)
    d = 4
    X_r, Y_r, X_h, Y_h = dat[0], dat[1], dat[2], dat[3]

    for iteration in range(5, 10):
        # Initialize GP with random hyperparameters
        L_init = 0.2*(np.random.rand(d,d) - 0.5)
        L_init = np.eye(d) + np.tril(L_init)
        omega_init = np.matmul(L_init, np.transpose(L_init))
        D,V = np.linalg.eig(omega_init)
        gp = GP(X_h, Y_h, omega = omega_init, L = L_init, l = 10.0 + 70.0*np.random.rand(), sigma = 8.0 + 16.0*np.random.rand(), noise = 0.001)

        # Define gradient descent parameters
        vals = []
        params_omega, params_sigma, params_l = [], [], []
        cur_o, cur_s, cur_l = gp.omega, gp.sigma, gp.l 
        iters, alter_iter, max_iters = 0, 30, 15000
        grad_max = 50.0
        omega_grad_max = 40.0
        rate = 0.0005
        var = np.random.randint(3)
        while iters < max_iters:
            prev_o, prev_s, prev_l = gp.omega, gp.sigma, gp.l
            
            if (iters == 5000):
                rate = 0.0003
            if (iters == 10000):
                rate = 0.0002

            # Get Gradients
            gp.resample()
            dL_dl, dL_ds, dL_domega = gp.likelihood_gradients()
            dL_domega = (dL_domega + np.transpose(dL_domega))/2
            dL_dl = np.clip(dL_dl, -grad_max, grad_max)
            dL_ds = np.clip(dL_ds, -grad_max, grad_max)
            if (np.amax(dL_domega) > omega_grad_max or np.amin(dL_domega) < omega_grad_max):
                max_val = max(np.amax(dL_domega), abs(np.amin(dL_domega)))
                dL_domega = dL_domega * (omega_grad_max / max_val)
            
            # Gradient descent
            eps = 0.0005
            if (var == 0):
                cur_o = cur_o - rate * dL_domega
                D, V = np.linalg.eig(cur_o)
                for i in range(len(D)):
                    if (D[i] <= eps):
                        D[i] = eps
                cur_o = np.matmul(np.matmul(V, np.diag(D)), np.linalg.inv(V))
            elif (var == 1):
                cur_l = cur_l - rate * dL_dl
            elif (var == 2):
                cur_s = cur_s - rate * dL_ds
            else:
                print("Error in parameter update")

            # Update parameters
            gp.omega, gp.sigma, gp.l = cur_o, cur_s, cur_l
            gp.omega = (gp.omega + np.transpose(gp.omega))/2

            iters = iters+1 #iteration count
            value = gp.log_likelihood()

            # Store and save updated parameters
            params_omega.append(gp.omega)
            params_sigma.append(gp.sigma)
            params_l.append(gp.l)
            vals.append(value)

            if (iters % alter_iter == 0):
                if (var < 2):
                    var += 1
                elif (var == 2):
                    var = 0
                else:
                    print("Error in setting variable to update.")

            if (iters % 10 == 0):
                print(iters)
                print("sigma: ", gp.sigma)
                print("length: ", gp.l)
                print("Likelihood for this dataset: ", value)
            if (iters % 50 == 0 and kSave):
                np.save('likelihood_vals_human_v' + str(iteration), vals)
                np.save('parameters_human_v' + str(iteration), [params_omega, params_sigma, params_l])

        # Initialize GP with random hyperparameters
        L_init = 0.2*(np.random.rand(d,d) - 0.5)
        L_init = np.eye(d) + np.tril(L_init)
        omega_init = np.matmul(L_init, np.transpose(L_init))
        D,V = np.linalg.eig(omega_init)
        gp = GP(X_r, Y_r, omega = omega_init, l = 0.2 + 3.0*np.random.rand(), sigma = 0.01 + 0.2*np.random.rand(), noise = 0.001)

        # Define gradient descent parameters
        vals = []
        params_omega, params_sigma, params_l = [], [], []
        cur_o, cur_s, cur_l = gp.omega, gp.sigma, gp.l 
        iters, alter_iter, max_iters = 0, 30, 10000
        grad_max = 50.0
        omega_grad_max = 40.0
        rate = 0.0002
        var = np.random.randint(3)
        while iters < max_iters:
            prev_o, prev_s, prev_l = gp.omega, gp.sigma, gp.l
            
            if (iters == 8000):
                rate = 0.0001
            if (iters == 12000):
                rate = 0.0001

            # Get Gradients
            gp.resample()
            dL_dl, dL_ds, dL_domega = gp.likelihood_gradients()
            dL_dl = np.clip(dL_dl, -grad_max, grad_max)
            dL_ds = np.clip(dL_ds, -grad_max, grad_max)
            if (np.amax(dL_domega) > omega_grad_max or np.amin(dL_domega) < omega_grad_max):
                max_val = max(np.amax(dL_domega), abs(np.amin(dL_domega)))
                dL_domega = dL_domega * (omega_grad_max / max_val)

            # Gradient descent
            eps = 0.0005
            if (var == 0):
                cur_o = cur_o - rate * dL_domega
                D, V = np.linalg.eig(cur_o)
                for i in range(len(D)):
                    if (D[i] <= eps):
                        D[i] = eps
                cur_o = np.matmul(np.matmul(V, np.diag(D)), np.linalg.inv(V))
            elif (var == 1):
                cur_l = cur_l - rate * dL_dl
                if (cur_l < eps):
                    cur_l = eps
            elif (var == 2):
                cur_s = cur_s - rate * dL_ds
                if (cur_s < eps):
                    cur_s = eps
            else:
                print("Error in parameter update")

            # Update parameters
            gp.omega, gp.sigma, gp.l = cur_o, cur_s, cur_l
            gp.omega = (gp.omega + np.transpose(gp.omega))/2


            iters = iters+1 #iteration count
            value = gp.log_likelihood()

            # Store and save updated parameters
            params_omega.append(gp.omega)
            params_sigma.append(gp.sigma)
            params_l.append(gp.l)
            vals.append(value)

            if (iters % alter_iter == 0):
                if (var < 2):
                    var += 1
                elif (var == 2):
                    var = 0
                else:
                    print("Error in setting variable to update.")

            if (iters % 10 == 0):
                print(iters)
                print("sigma: ", gp.sigma)
                print("length: ", gp.l)
                print("Likelihood for this dataset: ", value)
            if (iters % 50 == 0 and kSave):
                np.save('likelihood_vals_robot_v' + str(iteration), vals)
                np.save('parameters_robot_v' + str(iteration), [params_omega, params_sigma, params_l])
