import numpy as np
import joblib
import os
import pandas as pd
from ..utils import logger


class BaseModel:
    def __init__(self):
        self.model = None

    def predict(self, x):
        assert self.model is not None
        y = self.model.predict(x)
        return y

    def fit(self, x, y):
        assert self.model is not None
        self.model.fit()

    def load(self, path):
        logger.info(f"读取模型{path}")
        params_dict = joblib.load(path).__dict_
        for k, v in params_dict.items():
            setattr(self, k, v)

    def save(self, path):
        logger.info(f"保存到{path}")
        root = os.path.dirname(path)
        if not os.path.isdir(root):
            os.makedirs(root)
        joblib.dump(self, path)

    def _check_target(self, target):
        """
        如果是低精度的数据类型, 转换成高精度
        """
        if isinstance(target, pd.DataFrame) or isinstance(target, pd.Series):
            target = target.to_numpy()

        if np.issubdtype(target.dtype, np.floating):
            target = target.astype(np.float64)
        elif np.issubdtype(target.dtype, np.integer):
            target = target.astype(np.int64)
        return target
