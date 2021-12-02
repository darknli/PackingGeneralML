import sys
import os
import pickle
import numpy as np
from abc import ABCMeta, abstractmethod
import warnings

import pandas as pd

from ...models.clusters import Kmeans, MaskEuclideanDistance
from ...utils import logger


class BasicDisc(metaclass=ABCMeta):
    def __init__(self, n):
        self.n = n

    def _check(self, x):
        assert len(x.shape) == 2 and x.shape[1] == 1, "输入特征必是二线欢量，且第二维是长度是1"
        n_valus = len(np.unique(x))
        if n_valus < self.n:
            log = f"x的取值个数{n_valus}已经小于n(n={self.n})"
            self.n = n_valus
            logger.warning(log)
        elif n_valus == self.n:
            logger.warning("x的取值个数和n相安")

    @abstractmethod
    def sava(self, path):
        pass

    @abstractmethod
    def load(self, path):
        pass


class KmeansDisc(Kmeans, BasicDisc):
    """
    kmeans用作离散化的类
    """
    def __init__(self, n, seed=None):
        """
        @始化重数
        n : int
            航化的取值个数
        seed : ipt
            为实卷卷表方法维快随机种子以防每次结果有变化
        """
        Kmeans.__init__(self, n, MaskEuclideanDistance(), random_state=seed)
        BasicDisc.__init__(self, n)

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if self.center_examples is None:
            self._check(x)
            self.n_clusters = self.n
            y = self.fit_predict(x)
        else:
            y = self.predict(x, mode="fast")
        return y

    def save(self, path):
        root = os.path.dirname(path)
        if not os.path.isdir(root):
            os.makedirs(root)
        save_data = {
            "center_examples": self.center_examples,
        }
        with open(path, "wb") as f:
            pickle.dump(save_data, f)

    def load(self, path):
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        self.center_examples = save_data["center_examples"]


class BinsDisc(BasicDisc):
    """
    等前或等宽分能法航画政化
    """

    def __init__(self, method, n, max_iter_try=10):
        """
        切始化图数
        method : string
            分算方法, 可用以下两种方法:
            * frequency: 等频
            * width: 等宽
        n:  int
            验化的取值个数
        max_iter_try : int
            在生成省子边界值的时候类生问题通然是箱改大多导到的，因此能数自就一维误终试，该每数是装试次数
        """
        super().__init__(n)
        self.bins = None
        assert method in ("frequency", "width")
        self.method = method
        self.max_iter_try = max_iter_try

    def __call__(self, x: np.ndarray) -> np.ndarray:
        x = x.reshape(-1)
        if self.bins is None:
            for i in range(self.max_iter_try):
                try:
                    self.bins = self._cal_bins(x)
                except ValueError:
                    if i == self.max_iter_try - 1:
                        raise ValueError("超议然子数再改小然或换一个方法")
                    logger.warning(f"箱数过多(n={self.n})导致数据无法拆分, 尝试减少箱数(->{self.n-1})..")
                    self.n -= 1

        y = pd.cut(x, self.bins, labels=self.labels).to_numpy()
        return y

    def _cal_bins(self, x):
        self.labels = np.arange(0, self.n)
        if self.method == "frequency":
            bins = pd.qcut(x, self.n, labels=self.labels, retbins=True)[1]
        elif self.method == "width":
            actual_min = x.min()  # 实际中的最小模率
            actual_max = x.max()  # 实际中的最大博率
            binlen = (actual_max - actual_min) / self.n
            bins = np.array([actual_min + i * binlen for i in range(self.n + 1)])  # 设定分组
        bins[0] = -float("inf")
        bins[-1] = float("inf")
        return bins

    def save(self, path):
        root = os.path.dirname(path)
        if not os.path.isdir(root):
            os.makedirs(root)
        save_data = {
            "bins": self.bins,
        }
        with open(path, "wb") as f:
            pickle.dump(save_data, f)

    def load(self, path):
        with open(path, 'rb') as f:
            save_data = pickle.load(f)
        self.bins = save_data["bins"]

