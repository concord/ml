from __future__ import print_function, division
import pytest

from bcd import offline_changepoint_detection, BayesianChangepointDetection
from bcd.distributions import StudentT


@pytest.fixture(scope="module")
def hazard():
    def hazard_func(t):
        return 0.5 ** t
    return hazard_func


def test_bcd_against_reference(hazard):
    data = [0] * 100 + [2] * 100
    detector = BayesianChangepointDetection(hazard, StudentT(0.1, 0.1, 1, 1))

    reference = offline_changepoint_detection(data,
                                              hazard, StudentT(0.1, 0.1, 1, 1))

    for i, observation in enumerate(data):
        pr = detector.step(observation)
        assert (pr == reference[:i + 2, i]).all()
