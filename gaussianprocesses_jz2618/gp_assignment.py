import numpy as np
from scipy.optimize import minimize


# ##############################################################################
# LoadData takes the file location for the yacht_hydrodynamics.data and returns
# the data set partitioned into a training set and a test set.
# the X matrix, deal with the month and day strings.
# Do not change this function!
# ##############################################################################
def loadData(df):
    data = np.loadtxt(df)
    Xraw = data[:,:-1]
    # The regression task is to predict the residuary resistance per unit weight of displacement
    yraw = (data[:,-1])[:, None]
    X = (Xraw-Xraw.mean(axis=0))/np.std(Xraw, axis=0)
    y = (yraw-yraw.mean(axis=0))/np.std(yraw, axis=0)

    ind = range(X.shape[0])
    test_ind = ind[0::4] # take every fourth observation for the test set
    train_ind = list(set(ind)-set(test_ind))
    X_test = X[test_ind]
    X_train = X[train_ind]
    y_test = y[test_ind]
    y_train = y[train_ind]

    return X_train, y_train, X_test, y_test

# ##############################################################################
# Returns a single sample from a multivariate Gaussian with mean and cov.
# ##############################################################################
def multivariateGaussianDraw(mean, cov):
    sample = np.zeros((mean.shape[0], )) # This is only a placeholder
    # Task 1:
    # TODO: Implement a draw from a multivariate Gaussian here
    sample = np.random.multivariate_normal(mean, cov)
    # plt.plot(x, y, 'x')
    # plt.axis('equal')
    # plt.show()

    # Return drawn sample
    return sample

# ##############################################################################
# RadialBasisFunction for the kernel function
# k(x,x') = s2_f*exp(-norm(x,x')^2/(2l^2)). If s2_n is provided, then s2_n is
# added to the elements along the main diagonal, and the kernel function is for
# the distribution of y,y* not f, f*.
# ##############################################################################
class LinearPlusRBF():
    def __init__(self, params):
        self.ln_sigma_b = params[0]
        self.ln_sigma_v = params[1]
        self.ln_sigma_f = params[2]
        self.ln_length_scale = params[3]
        self.ln_sigma_n = params[4]

        self.sigma2_b = np.exp(2*self.ln_sigma_b)
        self.sigma2_v = np.exp(2*self.ln_sigma_v)
        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def setParams(self, params):
        self.ln_sigma_b = params[0]
        self.ln_sigma_v = params[1]
        self.ln_sigma_f = params[2]
        self.ln_length_scale = params[3]
        self.ln_sigma_n = params[4]

        self.sigma2_b = np.exp(2*self.ln_sigma_b)
        self.sigma2_v = np.exp(2*self.ln_sigma_v)
        self.sigma2_f = np.exp(2*self.ln_sigma_f)
        self.sigma2_n = np.exp(2*self.ln_sigma_n)
        self.length_scale = np.exp(self.ln_length_scale)

    def getParams(self):
        return np.array([self.ln_sigma_b, self.ln_sigma_v, self.ln_sigma_f, self.ln_length_scale, self.ln_sigma_n])

    def getParamsExp(self):
        return np.array([self.sigma2_b, self.sigma2_v, self.sigma2_f, self.length_scale, self.sigma2_n])

    # ##########################################################################
    # covMatrix computes the covariance matrix for the provided matrix X using
    # the RBF. If two matrices are provided, for a training set and a test set,
    # then covMatrix computes the covariance matrix between all inputs in the
    # training and test set.
    # ##########################################################################
    def covMatrix(self, X, Xa=None):
        if Xa is not None:
            X_aug = np.zeros((X.shape[0]+Xa.shape[0], X.shape[1]))
            X_aug[:X.shape[0], :X.shape[1]] = X
            X_aug[X.shape[0]:, :X.shape[1]] = Xa
            X = X_aug

        n = X.shape[0]
        covMat1 = np.zeros((n, n))
        covMat2 = np.zeros((n, n))

        print('X input shape', X.shape)
        # print(self.ln_sigma_b)
        # print(self.ln_sigma_n)
        # print(self.ln_length_scale)

        # Task 2:
        # TODO: Implement the covariance matrix here
        X_train, y_train, X_test, y_test = loadData('boston_housing.txt')
        covMat1 += (self.sigma2_b**2) + self.sigma2_v**2 * np.dot(X, X.T)

        # diffAbs = abs(np.dot((X_train - X), (X_train - X).T))
        # diffAbs = np.linalg.norm(X-X_train)
        #
        # covMat2 = (self.ln_sigma_f**2 * np.exp(-(1/(2 * self.ln_length_scale**2)) * diffAbs)+(self.ln_sigma_n**2))*np.identity(n)

        for i in range(n):
            for j in range(n):
                diffAbs = np.linalg.norm(X[i] - X[j])**2
                covMat2[i][j] = self.sigma2_f**2 * np.exp(-diffAbs/(2 * self.length_scale**2))


        covMat = covMat1 + covMat2 # n by n 379 by 379

        # If additive Gaussian noise is provided, this adds the sigma2_n along
        # the main diagonal. So the covariance matrix will be for [y y*]. If
        # you want [y f*], simply subtract the noise from the lower right
        # quadrant.
        if self.sigma2_n is not None:
            covMat += self.sigma2_n*np.identity(n)

        # Return computed covariance matrix K = K(A,A) is the kernel matrix computed for a set of points,

        return covMat


class GaussianProcessRegression():
    def __init__(self, X, y, k):
        self.X = X
        self.n = X.shape[0]
        self.y = y
        self.k = k
        self.K = self.KMat(self.X)
        self.L = np.linalg.cholesky(self.K)

    # ##########################################################################
    # Recomputes the covariance matrix and the inverse covariance
    # matrix when new hyperparameters are provided.
    # ##########################################################################
    def KMat(self, X, params=None):
        if params is not None:
            self.k.setParams(params)
        K = self.k.covMatrix(X)
        self.K = K
        self.L = np.linalg.cholesky(self.K)
        return K

    # ##########################################################################
    # Computes the posterior mean of the Gaussian process regression and the
    # covariance for a set of test points.
    # NOTE: This should return predictions using the 'clean' (not noisy) covariance
    # ##########################################################################
    def predict(self, Xa):
        """
        :param set of test point Xa (379,13):
        :return posterior mean_f*, covariance, cov(f*):
        """
        mean_fa = np.zeros((Xa.shape[0], 1))  # (379, 1)
        cov_fa = np.zeros((Xa.shape[0], Xa.shape[0])) # (379,379)
        # Task 3:
        # TODO: compute the mean and covariance of the prediction
        # k_XaX = self.k.covMatrix(Xa, self.X)
        k_XXa = self.k.covMatrix(self.X, Xa)[0:self.n, self.n:]
        k_XaX = self.k.covMatrix(self.X, Xa)[self.n:, 0:self.n]
        k_XaXa = self.k.covMatrix(self.X, Xa)[self.n:, self.n:] - self.k.sigma2_n*np.identity(self.n)

        # print(self.k.covMatrix(self.X, Xa)[0:self.n, 0:self.n] == self.K)
        # print(k_XaXa == self.k.covMatrix(Xa))
        cov_yy = np.linalg.inv(self.K) # sigma is 0

        # print(k_XaX.shape)
        # print(k_XXa.shape)
        # print(cov_yy.shape)
        # print(self.y.shape)
        # print(self.K.shape)
        # print((self.y - np.mean(self.X, axis=1)).shape)
        # print(np.dot(k_XaX, cov_yy).shape)
        mean_fa +=  np.dot(np.dot(k_XXa.T, cov_yy), self.y)
        cov_fa += k_XaXa - np.dot(np.dot(k_XXa.T, cov_yy), k_XXa)

        # Return the mean and covariance
        return mean_fa, cov_fa

    # ##########################################################################
    # Return negative log marginal likelihood of training set. Needs to be
    # negative since the optimiser only minimises.
    # Note: our optimizer, provided for you in the function optimize
    # minimizes the target function, so please return the negative log marginal likelihood:
    # ##########################################################################
    def logMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)

        mll = 0
        # Task 4:
        # TODO: Calculate the log marginal likelihood ( mll ) of self.y
        cov_yy = np.linalg.inv(self.K)  # sigma is 0
        mll += 0.5*np.dot(np.dot(self.y.T, cov_yy), self.y) + 0.5*np.log(np.linalg.det(self.K)) + self.n*0.5*np.log(2*np.pi)
        # Return mll
        return mll

    # ##########################################################################
    # Computes the gradients of the negative log marginal likelihood wrt each
    # hyperparameter.
    # ##########################################################################
    def gradLogMarginalLikelihood(self, params=None):
        if params is not None:
            K = self.KMat(self.X, params)

        grad_ln_sigma_b = grad_ln_sigma_v = grad_ln_sigma_f = grad_ln_length_scale = grad_ln_sigma_n = 0
        # Task 5:
        # TODO: calculate the gradients of the negative log marginal likelihood
        # wrt. the hyperparameters
        cov_yy = np.linalg.inv(self.K)  # sigma is 0
        alpha = np.dot(cov_yy, self.y) # sigma is 0

        dk_dln_sigma_b = 2*np.exp(2*self.k.ln_sigma_b) * np.ones((self.n, self.n))
        grad_ln_sigma_b += 0.5*np.trace(np.dot((cov_yy - np.dot(alpha, alpha.T)), dk_dln_sigma_b))

        dk_dln_sigma_v = 2*np.exp(2*self.k.ln_sigma_v) * np.dot(self.X, self.X.T)
        grad_ln_sigma_v += 0.5 * np.trace(np.dot((cov_yy - np.dot(alpha, alpha.T)), dk_dln_sigma_v))

        normal = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                normal[i][j] = np.linalg.norm(self.X[i] - self.X[j])**2

        dk_dln_sigma_f = 2*np.exp(2*self.k.ln_sigma_f) * np.exp((-1/(2*np.exp(2*self.k.ln_length_scale)))*normal)
        grad_ln_sigma_f += 0.5 * np.trace(np.dot((cov_yy - np.dot(alpha, alpha.T)), dk_dln_sigma_f))

        dk_dln_length_scale = dk_dln_sigma_f/2* normal * np.exp(-2*self.k.ln_length_scale)
        grad_ln_length_scale = 0.5 * np.trace(np.dot((cov_yy - np.dot(alpha, alpha.T)), dk_dln_length_scale))

        dk_dln_sigma_n = 2*np.exp(2*self.k.ln_sigma_n) * np.identity(self.n)
        grad_ln_sigma_n += 0.5 * np.trace(np.dot((cov_yy - np.dot(alpha, alpha.T)), dk_dln_sigma_n))




        # Combine gradients
        gradients = np.array([grad_ln_sigma_b, grad_ln_sigma_v, grad_ln_sigma_f, grad_ln_length_scale, grad_ln_sigma_n])

        # Return the gradients
        return gradients

    # ##########################################################################
    # Computes the mean squared error between two input vectors.
    # ##########################################################################
    def mse(self, ya, fbar):
        mse = 0
        # Task 7:
        # TODO: Implement the MSE between ya and fbar
        for i in range(ya.shape[0]):
            mse += (ya[i][0] - fbar[i][0]) ** 2
        # Return mse
        mse *= 1 / (float(ya.shape[0]))
        return mse

    # ##########################################################################
    # Computes the mean standardised log loss.
    # ##########################################################################
    def msll(self, ya, fbar, cov):
        msll = 0
        # Task 7:
        # TODO: Implement MSLL of the prediction fbar, cov given the target ya
        for i in range(ya.shape[0]):
            sigma2_x = cov[i, i] + self.k.sigma2_n
            msll += 0.5 * np.log(2 * np.pi * sigma2_x) + ((ya[i][0] - fbar[i][0]) ** 2) / (2 * sigma2_x)

        msll *= 1 / (float(ya.shape[0]))
        # Return msll
        return msll

    # ##########################################################################
    # Minimises the negative log marginal likelihood on the training set to find
    # the optimal hyperparameters using BFGS.
    # ##########################################################################
    def optimize(self, params, disp=True):
        res = minimize(self.logMarginalLikelihood, params, method ='BFGS', jac = self.gradLogMarginalLikelihood, options = {'disp':disp})
        return res.x

if __name__ == '__main__':

    np.random.seed(42)

    ##########################
    # You can put your tests here - marking
    # will be based on importing this code and calling
    # specific functions with custom input.
    ##########################
    X_train, y_train, X_test, y_test = loadData('boston_housing.txt')
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)

    mean = np.mean(X_train, axis=1)
    cov = np.cov(X_train)
    #
    sample = multivariateGaussianDraw(mean, cov)


    # L = LinearPlusRBF([1,2,3,4,5])
    # L.covMatrix(X_train, X_train)
    # self.ln_sigma_b = params[0]
    # self.ln_sigma_v = params[1]
    # self.ln_sigma_f = params[2]
    # self.ln_length_scale = params[3]
    # self.ln_sigma_n = params[4]

    k = LinearPlusRBF([0, 0, 0, np.log(0.1), 0.5*np.log(0.5)])
    print('X_train shape', X_train.shape)
    print('X_test shape', X_test.shape)
    covmat = k.covMatrix(X_train, Xa=X_test)
    print('cov mat shape', covmat.shape)


    GP = GaussianProcessRegression(X_train, y_train, k)
    gradients = GP.gradLogMarginalLikelihood()
    print(gradients)

    print(np.dot(np.array([[1], [1], [1]]), np.array([[1], [1], [1]]).T))






