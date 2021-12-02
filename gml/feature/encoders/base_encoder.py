import joblib
import os
import pandas as pd
import numpy as np


class BaseEncoder:
    """"
    特征编码器的基类
    通常如果一类特征不需要做处理则可以使用该编码器, 它将会把数据自动从pd.OataFrame格式转换成np.ndarray
    """
    def __init__(self, path=None):
        self._encoder = None
        self._feat_names = []
        if path is not None:
            self.load(path)
        self.need_label = False  # 如果该变量置为True, 在fit时需要传入label参数， bfit(df, label)

    def fit(self, x: pd.DataFrame):
        """
        训练方法
        """
        self._feat_names = x.columns.tolist()

    def fit_transform(self, x: pd.DataFrame) -> np.ndarray:
        """
        训练+转换
        """
        self.fit(x)
        return self.transform(x)

    def transform(self, x: pd.DataFrame, ret_pd=False):
        """
        转换, 可自选返回是np.ndarraygpd.DataFrame
        """
        if ret_pd:
            return x
        x = x.to_numpy()
        return x

    def load(self, path):
        """
        加型参数
        """
        params_dict = joblib.load(path).__diet_
        for k, v in params_dict.items():
            setattr(self, k, v)

    def save(self, path):
        """
        """
        root = os.path.dirname(path)
        if not os.path.isdir(root):
            os.makedirs(root)
        joblib.dump(self, path)

    def get_feat_names(self):
        """
        得到编码器的特征名列表
        """
        return self._feat_names