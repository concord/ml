"""Local implementation for Bayesian Changepoint Detection"""
import numpy as np
from concord.computation import (
    Computation,
    Metadata
)

# import logging
# logging.basicConfig()
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class BayesianChangepointDetection(Computation):
    """ Bayesian Changepoint Detection implementation
    Based on algorithm from http://arxiv.org/abs/0710.3742.
    """
    def __init__(self, hazard, distribution, istream=None, ostream=None):
        """ Constructor for BCD computation

        Args:
            hazard: A function that maps from the length of the current
                    run to the probability of a changepoint.
                    Callable[[int], float]
            distribution: The distribution that the time series is assumed to
                      follow. Should follow interface defined in
                      `concord_ml.bcd.distributions.Distribution`.
            istream (List[str]): istreams the computation should listen to.
            ostream (List[str]): ostreams the computation should produce to.
        """
        self.hazard = hazard
        self.distribution = distribution
        self.istream = istream
        self.ostream = ostream
        self.time = 0
        self.Pr = np.ones(1)

    def init(self, ctx):
        pass

    def process_timer(self, ctx, key, time):
        raise Exception('process_timer not implemented')

    def process_record(self, ctx, record):
        """
        Returns:
            A 1-D np.array of floats representing the probability
            distribution of the current run length. Output[r_0] is the
            probability that r=r_0 at time r_0.
        """
        result = self.step(float(record.data))
        if self.ostream is not None:
            for stream in self.ostream:
                ctx.produce_record(stream, '~', result)

    def metadata(self):
        return Metadata(
            name='BayesianChangepointDetection',
            istreams=[] if self.istream is None else self.istream,
            ostreams=[] if self.ostream is None else self.ostream)

    def step(self, observation):
        """ Calculates the probability of a changepoint at the current time

        Args:
            data: The latest observation. The type can be anything that
                  the distribution's pdf accepts.
        Returns:
            A 1-D np.array of floats representing the probability
            distribution of the current run length. Output[r_0] is the
            probability that r=r_0 at the current time.
        """
        self.time += 1

        H = self.hazard(np.arange(self.time))
        old = self.Pr
        self.Pr = np.zeros(self.time + 1)  # +1 because 0 <= r <= time

        predprob = self.distribution.pdf(observation)
        self.distribution.update(observation)

        self.Pr[1:] = old * predprob * (1 - H)
        self.Pr[0] = np.sum(old * predprob * H)
        self.Pr /= np.sum(self.Pr)

        return self.Pr


def offline_changepoint_detection(data, hazard_function, distribution):
    """ Reference changepoint detection implementation

    Assumes that length of data is known beforehand. Implementation from
    https://github.com/hildensia/bayesian_changepoint_detection.

    Args:
        data: A list of floats that represents the data to analyze.
              data[t] should be the observed data @ time t.
        hazard_function: A function that returns the (prior) probability
                         of a changepoint given the current run length.
                         Callable[[int], float]
        distribution: The distribution that the time series is assumed
                      to follow. Should follow interface defined in
                      `bcd.distributions.Distribution`.
    Returns:
        A two-dimensional numpy array Pr, where Pr[r_0, t] is the
        probability that r = r_0 @ time t.
    """

    length = len(data)
    H = hazard_function(np.arange(length + 1))
    Pr = np.zeros((length + 1, length + 1))    # length + 1 b/c 0 is for priors

    Pr[0, 0] = 1
    for t, x in enumerate(data):
        t += 1    # t=0 is for priors stuff. First data point is @ t=1
        predprobs = distribution.pdf(x)
        Pr[1:t + 1, t] = Pr[:t, t - 1] * predprobs * (1 - H[:t])
        Pr[0, t] = np.sum(Pr[:t, t - 1] * predprobs * H[:t])
        Pr[:, t] /= np.sum(Pr[:, t])    # normalize probabilities
        distribution.update(x)
    return Pr
