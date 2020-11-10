import numpy

from assignment_1.distribution import SingleParameterDistribution


class ExponentialDistribution(SingleParameterDistribution):
    def __init__(self, theta: float):
        super().__init__(theta)

    @staticmethod
    def name() -> str:
        return "exponential"

    def get_samples(self, n_samples: int) -> numpy.ndarray:
        return numpy.random.exponential(scale=self.theta, size=n_samples)

    def get_mom_estimated_theta(self, samples: numpy.ndarray, moment: int) -> float:
        empirical_mean = numpy.power(samples, moment).mean()
        root_term = empirical_mean
        for k in range(moment, 0, -1):
            root_term /= k
        return numpy.power(root_term, 1. / moment)
