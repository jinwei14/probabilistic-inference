import numpy as np
import numpy.random as rn
from scipy import optimize, stats
import scipy.linalg as linalg


# ##############################################################################
# load_data generates a binary dataset for visualisation and testing using two
# parameters:
# * A **jitter** parameter that controls how noisy the data are; and
# * An **offset** parameter that controls the separation between the two classes.
#
# Do not change this function!
# ##############################################################################
def load_data(N=50, jitter=0.7, offset=1.2):
    # Generate the data
    x = np.vstack([rn.normal(0, jitter, (N // 2, 1)),
                   rn.normal(offset, jitter, (N // 2, 1))])
    y = np.vstack([np.zeros((N // 2, 1)), np.ones((N // 2, 1))])
    x_test = np.linspace(-2, offset + 2).reshape(-1, 1)

    # Make the augmented data matrix by adding a column of ones
    x_train = np.hstack([np.ones((N, 1)), x])
    x_test = np.hstack([np.ones((N, 1)), x_test])
    return x_train, y, x_test


# ##############################################################################
# predict takes a input matrix X and parameters of the logistic regression theta
# and predicts the output of the logistic regression.
# ##############################################################################
def predict(X, theta):
    # X: K x D matrix of test inputs
    # theta: D x 1 vector of parameters
    # returns: prediction of f(X); K x 1 vector
    prediction = np.zeros((X.shape[0], 1))

    # ######################## Task 1:##########################
    # TODO: Implement the prediction of a logistic regression here.
    # print(X.shape)
    # print(theta.shape)

    x = np.dot(X,theta)
    prediction = 1/(1+np.exp(-x))
    

    return prediction


def predict_binary(X, theta):
    # X: K x D matrix of test inputs
    # theta: D x 1 vector of parameters
    # returns: binary prediction of f(X); K x 1 vector; should be 0 or 1

    prediction = 1. * (predict(X, theta) > 0.5)

    return prediction


# ##############################################################################
# log_likelihood takes data matrices x and y and parameters of the logistic
# regression theta and returns the log likelihood of the data given the logistic
# regression.
# ##############################################################################
def log_likelihood(X, y, theta):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # theta: parameters (D x 1)
    # returns: log likelihood, scalar


    L = 0

    # ########################  Task 2:############################
    # TODO: Calculate the log-likelihood of a dataset
    # given a value of theta.
    # N x 1 vector in the log
    L += np.dot(y.T,np.log(predict(X,theta))) + np.dot((1-y).T, np.log(1-predict(X,theta)))


    return np.asscalar(L)


# ##############################################################################
# max_lik_estimate takes data matrices x and y ands return the maximum
# likelihood parameters of a logistic regression.
# ##############################################################################
def max_lik_estimate(X, y):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # returns: maximum likelihood parameters (D x 1)

    N, D = X.shape
    theta_init = rn.rand(D, 1)
    theta_ml = theta_init

    # ########################  Task 3:#######################
    # TODO: Optimize the log-likelihood function you've
    # written above an obtain a maximum likelihood estimate

    def likelihood(theta):
        #
        # print('theta shape',theta.shape)
        # theta.reshape(50,2)
        # print('y shape', y.shape)
        L = -np.dot(y.T, np.log(predict(X, theta))) - np.dot((1 - y).T, np.log(1 - predict(X, theta)))
        return L

    res = optimize.minimize(likelihood, theta_init, method='BFGS',options={'disp': False})
    return res.x



# ##############################################################################
# neg_log_posterior takes data matrices x and y and parameters of the logistic
# regression theta as well as a prior mean m and covariance S and returns the
# negative log posterior of the data given the logistic regression.
# ##############################################################################
def neg_log_posterior(theta, X, y, m, S):
    # theta: D x 1 matrix of parameters
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: scalar negative log posterior
    N, D = X.shape


    negative_log_posterior = 0

    # #######################  Task 4:  ###########################
    # TODO: Calculate the log-posterior
    # print('linalg.inv(S)',linalg.inv(S).shape)
    # print('(X-m.T).T',(X-m.T).T.shape)
    # print('(X-m.T)',(X-m.T).shape)
    # print(' np.sum((X-m.T), axis=0)', np.sum((X-m.T), axis=0).shape)
    # np.sum([[0, 1], [0, 5]], axis=0)


    part1 = 0.5 * np.dot(np.dot((theta - m).T, linalg.inv(S)),(theta - m))


    negative_log_posterior += part1  - log_likelihood(X,y,theta)

    return np.asscalar(negative_log_posterior)



# ##############################################################################
# map_estimate takes data matrices x and y as well as a prior mean m and
# covariance  and returns the maximum a posteriori parameters of a logistic
# regression.
# ##############################################################################
def map_estimate(X, y, m, S):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: maximum a posteriori parameters (D x 1)

    N, D = X.shape

    theta_init = rn.rand(D, 1)
    theta_map = theta_init

    ##########################   Task 5:  #########################
    # TODO: Optimize the log-posterior function you've
    # written above an obtain a maximum a posteriori estimate
    def negLogPosterior(theta):
        theta = theta.reshape(2,1)

        return neg_log_posterior(theta, X, y, m, S)

    res = optimize.minimize(negLogPosterior, theta_map, method='BFGS',options={'disp': False})
    return res.x



# ##############################################################################
# laplace_q takes an array of points z and returns an array with Laplace
# approximation q evaluated at all points in z.
# ##############################################################################
def laplace_q(z):
    # z: double array of size (T,) (50,)
    # returns: array with Laplace approximation q evaluated
    #          at all points in z
    D = z.shape[0]
    q = np.zeros_like(z)

    # Task 6:
    # TODO: Evaluate the Laplace approximation $q(z)$.

    z_star = 2
    pTilde =z_star*np.exp(-z_star*0.5)
    A =  z_star**(-2)
    # q += pTilde*np.exp(-0.5*np.dot(np.dot((z-z_star).T,A),(z-z_star)))
    q = stats.multivariate_normal.pdf(z, mean=z_star, cov=A**(-1))
    # q = (2*np.pi)**(-0.5*D)*linalg.det(linalg.inv(A))**(-0.5)*np.exp(-0.5*np.dot(np.dot((z-z_star).T,A),(z-z_star)))

    return q


# ##############################################################################
# get_posterior takes data matrices x and y as well as a prior mean m and
# covariance and returns the maximum a posteriori solution to parameters
# of a logistic regression as well as the covariance approximated with the
# Laplace approximation.
# ##############################################################################
def get_posterior(X, y, m, S):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: maximum a posteriori parameters (D x 1)
    #          covariance of Laplace approximation (D x D)

    mu_post = np.zeros_like(m)
    S_post = np.zeros_like(S)
    N = X.shape[0]
    D = X.shape[1]
    # ##################################### Task 7: #############################
    # TODO: Calculate the Laplace approximation of p(theta | X, y)

    mu_post = map_estimate(X, y, m, S).reshape(2,1)
    # print('mu:,theta ',mu_post.shape)
    # print(linalg.inv(S).T.shape)
    # print('predict(X.T,mu_post)',predict(X,mu_post).shape)
    # print('(1-predict(X,mu_post)).T', (1-predict(X,mu_post)).T.shape)

    S_post = linalg.inv(S) + np.dot(X.T, predict(X,mu_post)*(1-predict(X,mu_post))*X)

    return mu_post, linalg.inv(S_post)


# ##############################################################################
# metropolis_hastings_sample takes data matrices x and y as well as a prior mean
# m and covariance and the number of iterations of a sampling process.
# It returns the sampling chain of the parameters of the logistic regression
# using the Metropolis algorithm.
# ##############################################################################
def metropolis_hastings_sample(X, y, m, S, nb_iter):
    # X: N x D matrix of training inputs
    # y: N x 1 vector of training targets/observations
    # m: D x 1 prior mean of parameters
    # S: D x D prior covariance of parameters
    # returns: nb_iter x D matrix of posterior samples

    D = X.shape[1]
    samples = np.zeros((nb_iter, D))



    # ###################  Task 8:  #######################################
    # TODO: Write a function to sample from the posterior of the
    # parameters of the logistic regression p(theta | X, y) using the
    # Metropolis algorithm.
    xt = np.zeros(D)
    rn.seed(5)
    for i in range(nb_iter):
        # generate 1 X D vector
        x0 = np.random.multivariate_normal(xt.reshape(D,), S, 1).reshape(D, 1)
        u = np.random.uniform(0, 1, 1)
        # print(u)
        px1 = np.exp(-neg_log_posterior(x0.reshape((D,1)), X, y, m, S))
        pxt = np.exp(-neg_log_posterior(xt.reshape((D,1)), X, y, m, S))
        # print(px1/pxt)
        if u <= (px1/pxt):
            # print(' samples[i] = x0', samples[i].shape, x0.shape)
            samples[i] = x0.reshape(D,)
            xt = x0
        else:
            # print(' samples[i] = xt', samples[i].shape, xt.shape)
            samples[i] = xt.reshape(D,)
            xt = xt

    return samples


if __name__ == '__main__':

    np.random.seed(42)

    ##########################################################################
    # You can put your tests here - marking
    # will be based on importing this code and calling
    # specific functions with custom input.
    ##########################################################################
    x_train, y, x_test = load_data()
    print('x_train.shape', x_train.shape)
    print('y.shape',y.shape)
    print('x_test.shape', x_test.shape)

    ############################# testing of task 3#############################
    theta = np.array([[2],[9]])
    l = log_likelihood(x_train, y, theta)
    print('L',l)
    res = max_lik_estimate(x_train, y)
    print('res',res)

    ############################# testing of task 5#############################
    # # theta: D x 1 matrix of parameters
    # # X: N x D matrix of training inputs
    # # y: N x 1 vector of training targets/observations
    # # m: D x 1 prior mean of parameters
    # # S: D x D prior covariance of parameters
    # # returns: scalar negative log posterior
    # theta = rn.rand(2,1)
    # m = rn.rand(2, 1)
    # S = rn.rand(2, 2)
    # print(map_estimate(x_train,y,m,S))



    # ######################## try out sum along axis  #######################
    # # x1 = np.arange(12.0).reshape((3, 4))
    # # x2 = np.arange(4.0).reshape(4,1)
    # #
    # # print(x1)
    # # print(x2)
    # # print(x1- x2.T)
    # a = np.array([[0, 1], [0, 5],[0, 6]])
    # print(a.shape)
    # print( np.sum(a, axis=0))

    ############################  task 6 testing   #############################
    # z = rn.rand(50,)
    # print('laplace',laplace_q(z))

    ############################  task 7 testing   #############################

    D = x_train.shape[1]
    m = rn.rand(2, 1)
    S = 5*np.eye(D)
    res = get_posterior(x_train, y, m, S)
    print(res[1])
    ############################  task 8 testing   #############################

    u1 = np.random.uniform(0, 1, 1)
    u2 = np.random.uniform(0, 1, 1)
    print(u1,u2)
    print('############################  task 8 testing   #############################')
    x0 = np.random.multivariate_normal(np.zeros(D), S, 1)
    print('x0',x0)
    print('np.zeros(D)', np.zeros((1,D)))

    metropolis_hastings_sample(x_train, y, m, S, 1000)


