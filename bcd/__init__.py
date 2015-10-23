import numpy as np


class BayesianChangepointDetection(object):
    def __init__(self, hazard, distribution):
        self.hazard = hazard
        self.distribution = distribution
        self.time = 0
        self.Pr = np.ones(1)

    def step(self, x):
        self.time += 1

        H = self.hazard(np.arange(self.time))
        old = self.Pr
        self.Pr = np.zeros(self.time + 1)   # +1 because 0 <= r <= time

        predprob = self.distribution.pdf(x)
        self.distribution.update(x)

        self.Pr[1:] = old * predprob * (1 - H)
        self.Pr[0] = np.sum(old * predprob * H)
        self.Pr /= np.sum(self.Pr)

        return self.Pr


def offline_changepoint_detection(data, hazard_function, distribution):
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
