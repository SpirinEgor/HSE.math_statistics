from abc import ABC, abstractmethod

import numpy


class SingleParameterDistribution(ABC):
    def __init__(self, theta: float):
        self._theta = theta

    @staticmethod
    def name() -> str:
        raise NotImplementedError("Abstract class does not have a name.")

    @property
    def theta(self) -> float:
        return self._theta

    @abstractmethod
    def get_samples(self, n_samples: int) -> numpy.ndarray:
        pass

    @abstractmethod
    def get_mom_estimated_theta(self, samples: numpy.ndarray, moment: int) -> float:
        pass
