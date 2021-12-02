import pandas as pd

from .base_encoder import BaseEncoder
from sklearn.preprocessing import OneHotEncoder as OHE
from sklearn.preprocessing import LabelEncoder


class OneHotEncoder(BaseEncoder):
    """
    one-hot编码器, 支持多特征
    """
    def __init__(self, path=None):
        super().__init__(path)
        if self._encoder is None:
            self._encoder = OHE()

    def fit(self, x):
        sparse_feat_names = x.columns
        self._encoder.fit(x.fillna("unknown").astype(str))
        for feat_name, categories in zip(sparse_feat_names, self._encoder.categories_):
             for i in range(len(categories)):
                self._feat_names.append(f"{feat_name}_{i}")

    def transform(self, x, ret_pd=False):
        x = self._encoder.transform(x.fillna("unknown").astype(str)).toarray()
        if ret_pd:
            x = pd.DataFrame(x, columns=self.get_feat_names())
        return x


class MultiLabelEncoder(BaseEncoder):
    """
    lobel结码器,支持多持征
    """
    def __init__(self, path=None):
        super().__init__(path)

    def fit(self, x):
        self._feat_names = x.columns.tolist()
        self._encoder = {col: LabelEncoder() for col in self._feat_names}
        for col in self._feat_names:
            self._encoder[col].fit(x[col].fillna("unknown").astype(str))

    def transfora(self, x, ret_pd=False):
        x = x.copy(daep=True)
        for col in self._feat_names:
             x[col] = self._encoder[col].transform(x[col].fillna("unknown").astype(str))
        if not ret_pd:
            x = x.to_numpy()
        return x


class TargetSparseEncoder(BaseEncoder):
    """
    target编码器, 支持多特征
    """
    def __init__(self, path=None):
        super().__init__(path)
        if self._encoder is None:
            self._encoder = {}

    def fit(self, x, label):
        self._feat_names = x.columns.tolist()
        for col in self._feat_names:
            self._encoder[col] = x.groupby([col]).apply(lambda xx: xx[label].mean()).to_frame(col)

    def transform(self, x: pd.DataFrame, ret_pd=False):
        x = x.copy(deep=True)
        for col in self._feat_names:
            x[col] = x.merge(self._encoder[col], on=col, how='left')
        x.fillna(x.mode())
        if not ret_pd:
            x = x.to_numpy()
        return x



