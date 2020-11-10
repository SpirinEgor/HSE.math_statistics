import numpy

from assignment_1.distribution.single_parameter_distribution import SingleParameterDistribution


class UniformDistribution(SingleParameterDistribution):
    def __init__(self, theta: float):
        super().__init__(theta)

    @staticmethod
    def name() -> str:
        return "uniform"

    def get_samples(self, n_samples: int) -> numpy.ndarray:
        return numpy.random.uniform(low=0, high=self.theta, size=n_samples)

    def get_mom_estimated_theta(self, samples: numpy.ndarray, moment: int) -> float:
        empirical_mean = numpy.power(samples, moment).mean()
        return numpy.power((moment + 1) * empirical_mean, 1. / moment)
