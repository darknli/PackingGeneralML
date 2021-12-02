import numpy as np


class StdNorm:
    """
    标能化
    """
    def __init__(self):
        self._mean = None
        self._std = None
        self.eps = 1e-6

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2
        if self._mean is None or self._std is None:
            self._mean = np.mean(x, 0)
            self._std = np.maximum(np.std(x, 0), self.eps)
        y = (x - self._mean) / self._std
        return y


class MinMaxNorm:
    """`
    Min-Max归一化
    """
    def __init__(self):
        self._min = None
        self._margin = None
        self.eps = 1e-6

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert len(x.shape) == 2
        if self._min is None or self._margin is None:
            self._min = np.min(x, 0)
            self._margin = np.maximum(np.max(x, 0) - self._min, self.eps)
        y = (x - self._min) / self._margin
        return y
