from typing import List

from assignment_1.distribution import UniformDistribution, SingleParameterDistribution, ExponentialDistribution


class Experiment:

    _distribution: SingleParameterDistribution

    def __init__(self, distribution_name: str, theta: float):
        if distribution_name == UniformDistribution.name():
            self._distribution = UniformDistribution(theta)
        elif distribution_name == ExponentialDistribution.name():
            self._distribution = ExponentialDistribution(theta)
        else:
            raise ValueError(f"Unknown distribution name: {distribution_name}")

    def _get_squared_error(self, moment: int, n_samples: int) -> float:
        samples = self._distribution.get_samples(n_samples)
        estimated_theta = self._distribution.get_mom_estimated_theta(samples, moment)
        difference = self._distribution.theta - estimated_theta
        return difference * difference

    def _estimate_moment_mse(self, moment: int, n_samples, n_runs: int) -> float:
        squared_errors = [self._get_squared_error(moment, n_samples) for _ in range(n_runs)]
        return sum(squared_errors) / n_runs

    def run(self, n_moments: int, n_samples: int, n_runs: int) -> List[float]:
        return [self._estimate_moment_mse(i, n_samples, n_runs) for i in range(1, n_moments + 1)]
