import numpy as np

class GP(Model):
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
            K_star[i,:] = evaluate_kernel(X[i,:], Xnew)
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