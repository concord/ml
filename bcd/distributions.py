from scipy import stats
import numpy as np


class Gaussian(object):
    """ Adapted from http://engineering.richrelevance.com/
    bayesian-analysis-of-normal-distributions-with-python/,
    http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

        Parameters stored in array
    """

    def __init__(self, kappa, mu, alpha, beta):

        self.kappa0 = self.kappa = np.array([kappa])  # Certainty
        self.mu0 = self.mu = np.array([mu])  # Mean
        self.alpha0 = self.alpha = np.array([alpha])  # Gamma
        self.beta0 = self.beta = np.array([beta])  # Sum of Squares

    def pdf(self, data):
        return stats.norm.pdf(x=data,
                              loc=self.mu,
                              scale=np.sqrt(self.beta / self.kappa))

    def update(self, data):
        new_kappa = self.kappa + 1
        new_mu = (self.kappa * self.mu + data) / (self.kappa + 1)
        new_alpha = (self.alpha + 0.5)
        new_beta = self.beta + ((self.kappa * (data - self.mu) ** 2) /
                                (2 * self.kappa + 1))

        self.kappa = np.concatenate(self.kappa, new_kappa)
        self.mu = np.concatenate(self.mu, new_mu)
        self.alpha = np.concatenate(self.alpha, new_alpha)
        self.beta = np.concatenate(self.beta, new_beta)


class StudentT(object):
    """ Taken from https://github.com/hildensia/bayesian_changepoint_detection

        Parameters stored in array
    """
    def __init__(self, kappa, mu, alpha, beta):
        self.kappa0 = self.kappa = np.array([kappa])  # Certainty
        self.mu0 = self.mu = np.array([mu])  # Mean
        self.alpha0 = self.alpha = np.array([alpha])  # Gamma
        self.beta0 = self.beta = np.array([beta])  # Sum of Squares

    def pdf(self, data):
        return stats.t.pdf(x=data,
                           df=2*self.alpha,
                           loc=self.mu,
                           scale=np.sqrt(self.beta * (self.kappa+1) /
                                         (self.alpha * self.kappa)))

    def update(self, data):
        kappa_update = np.concatenate((self.kappa0, self.kappa + 1.))
        mu_update = np.concatenate((self.mu0, (self.kappa *
                                   self.mu + data) / (self.kappa + 1)))
        alpha_update = np.concatenate((self.alpha0, self.alpha + 0.5))
        beta_update = np.concatenate((self.beta0, self.beta + (self.kappa *
                                     (data - self.mu)**2) /
                                    (2. * (self.kappa + 1.))))

        self.kappa = kappa_update
        self.mu = mu_update
        self.alpha = alpha_update
        self.beta = beta_update
