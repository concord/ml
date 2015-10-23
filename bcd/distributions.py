""" Defines prior distributions for bayesian changepoint detection
"""

from scipy import stats
import numpy as np


class Distribution(object):
    """Base class for probability distributions used in BCD.

    Note that each instance represents _multiple_ different pdfs, each
    assuming a different run length. So, after calling `update` t times,
    there are (t + 1) different possible values of r (0 <= r <= t).
    `pdf` should thus return an array of length (t + 1).

    We thus store (t + 1) copies of each hyperparameter.
    hyperparameter[0] is our prior distribution, hyperparameter[1] is
    the distribution fitted on 1 observation, hyperparameter[2] is the
    distribution fitted on 2 observations, and so on.
    """

    def pdf(self, data):
        """ Get the prob density of the observation conditioned on run length.

        Note that calling pdf _does not_ automatically call update. You
        must do this manually.

        Args:
            data: Latest observtion
        Returns:
            A 1-D array of floats, where pdf[t] is the probability
            density of data assuming that the run length is t and given
            our fitted hyperparameters.
            pdf[t] = p(data | r=t, hyperparameters)
        """
        raise NotImplementedError

    def update(self, data):
        """ Update the hyperparameters given the current observation
        Args:
            data: The latest observation
        Returns:
            None
        """

        raise NotImplementedError


class Gaussian(Distribution):
    """ Gaussian Distribution

    Adapted from http://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf
    and http://goo.gl/gfyils.

    Args:
        kappa: The prior for kappa
        mu:    The prior for mu
        alpha: The prior for alpha
        beta:  The prior for beta
    """

    def __init__(self, kappa, mu, alpha, beta):
        """ Initialize the prior distribution """
        self.kappa0 = self.kappa = np.array([kappa])  # Certainty
        self.mu0 = self.mu = np.array([mu])           # Mean
        self.alpha0 = self.alpha = np.array([alpha])  # Gamma
        self.beta0 = self.beta = np.array([beta])     # Sum of Squares

    def pdf(self, data):
        """ Calculate probability density at data at all possible run lengths.

        This does _not_ automatically call update.

        Args:
            data: float that is the latest observtion
        Returns:
            A 1-D array of floats, whose length is equal to the number
            of times update is called
        """
        scale = np.sqrt(self.beta / self.kappa)
        return stats.norm.pdf(x=data, loc=self.mu, scale=scale)

    def update(self, data):
        """ Update the hyperparameters

        Args:
            data: float that is the latest observation
        Returns:
            None
        """

        new_kappa = self.kappa + 1
        new_mu = (self.kappa * self.mu + data) / (self.kappa + 1)
        new_alpha = (self.alpha + 0.5)
        new_beta = self.beta + ((self.kappa * (data - self.mu) ** 2) /
                                (2 * self.kappa + 1))

        self.kappa = np.concatenate([self.kappa0, new_kappa])
        self.mu = np.concatenate([self.mu0, new_mu])
        self.alpha = np.concatenate([self.alpha0, new_alpha])
        self.beta = np.concatenate([self.beta0, new_beta])


class StudentT(Distribution):
    """ Student's T distribution prior.

    Adapted from https://github.com/hildensia/bayesian_changepoint_detection

    Args:
        kappa: The prior for kappa
        mu:    The prior for mu
        alpha: The prior for alpha
        beta:  The prior for beta
    """
    def __init__(self, kappa, mu, alpha, beta):
        self.kappa0 = self.kappa = np.array([kappa])  # Certainty
        self.mu0 = self.mu = np.array([mu])           # Mean
        self.alpha0 = self.alpha = np.array([alpha])  # Gamma
        self.beta0 = self.beta = np.array([beta])     # Sum of Squares

    def pdf(self, data):
        """ Calculate probability density at data at all possible run lengths.

        This does _not_ automatically call update.

        Args:
            data: float that is the latest observtion
        Returns:
            A 1-D array of floats, whose length is equal to the number
            of times update is called
        """
        scale = np.sqrt(self.beta * (self.kappa + 1) /
                        (self.alpha * self.kappa))
        return stats.t.pdf(x=data, df=2*self.alpha, loc=self.mu, scale=scale)

    def update(self, data):
        """ Update the hyperparameters given the current observation

        Args:
            data: float that is the latest observation
        Returns:
            None
        """
        new_kappa = self.kappa + 1
        new_mu = (self.kappa * self.mu + data) / (self.kappa + 1)
        new_alpha = self.alpha + 0.5
        new_beta = self.beta + (self.kappa * (data - self.mu ** 2) /
                                (2 * self.kappa + 2))

        self.kappa = np.concatenate([self.kappa0, new_kappa])
        self.mu = np.concatenate([self.mu0, new_mu])
        self.alpha = np.concatenate([self.alpha0, new_alpha])
        self.beta = np.concatenate([self.beta0, new_beta])
