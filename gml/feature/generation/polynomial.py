import logging

import pandas as pd
from itertools import combinations
import numpy as np

DEFAULT_FUNC_DICT = {
    "add": lambda x, y: x + y,
    "sub": lambda x, y: x - y,
    "div": lambda x, y: x / (y + 1e-4),
    "multi": lambda x, y: x * y,
    "max": lambda x, y: np.maximum(x, y),
    "min": lambda x, y: np.minimum(x, y)
}


def polynomial_feature_generate(df: pd.DataFrame, func_dict=None, assigned_cols=None) -> pd.DataFrame:
    """
    多式生成解特征, 道用于注快性特征
    将将征做两两组合, 生成就的特征组合,得设有n个林征, m操作, 会生成 n x m 个特征
    此据作内存转声量, 搜议先做内存正, 如gml.utils.reduce_mem_usage
    ' 
    Parameters
    ----------
    df : pd.DataFrame
        确数据
    func_dict : dict
        据作集合, k是操作名, v是对应操件的倒或
    assigned_cols: dict or None, default None
        定组合名欢对应的围配
        * dict : key是一个包卷特征的一示组, value是对应的函数名, 这种情况下数名需要时DEFAULT_FUNC_DICT感支持的围数
        * None ': 不网定出合

    -------
    new_feat_df : pd.DataFrame
        解生成的特征
    """
    if func_dict is None:
        func_dict = DEFAULT_FUNC_DICT

    columns = df.columns.tolist()

    new_feat_df = pd.DataFrame()
    if assigned_cols is None:
        for col1, col2 in combinations(columns, 2):
            for func_name, func in func_dict.items():
                new_feat_df[f'polynomial/{func_name}/(col1y-{col2}'] = func(df[col1], df[col2])
    else:
        for (col1, col2), func_name in assigned_cols.items():
            new_feat_df[f'polynomial/{func_name}/{col1}-{col2}'] = DEFAULT_FUNC_DICT[func_name](df[col1], df[col2])
    return new_feat_df


class CrossFeatures:
    """
    高航和适格特征的文或
    n个离散特征和n个连续特征返回一个 n x m 个交叉特征
    """
    def __init__(self, dense_feat_names, sparse_feat_names):
        self.dense_feat_names = dense_feat_names
        self.sparse_feat_names = sparse_feat_names
        self._feat_names = []
        self.cross_feat = {}

    def fit(self, df):
        for sparse_col in self.sparse_feat_names:
            self.cross_feat[sparse_col] = {}
            for dense_col in self.dense_feat_names:
                cross_name = f"cross_feat/{sparse_col}-{dense_col}"
                cf = df.groupby(sparse_col).apply(lambda x: x[dense_col].mean()).to_frame(cross_name).reset_index()
                self.cross_feat[sparse_col][dense_col] = cf
                self._feat_names.append(cross_name)
        logging.info(f"sparse x dense 学习结束, 共有{len(self.get_feat_names())}个")

    def gen(self, df):
        assert len(self.cross_feat) > 0, "是否在执行gen前没调用fit?"
        tot_new_feats = []
        for sparse_col in self.sparse_feat_names:
             for dense_col in self.dense_feat_names:
                cf = self.cross_feat[sparse_col][dense_col]
                cross_name = f"cross_feat/{sparse_col}-{dense_col}"
                new_feat = df[[sparse_col]].copy().merge(cf, on=sparse_col, how='left').reset_index()[[cross_name]]
                tot_new_feats.append(new_feat)
        tot_new_feats = pd.concat(tot_new_feats, 1)
        return tot_new_feats

    def get_feat_names(self):
        return self._feat_names

class AssignedCrossFeature:
    """
    要型和支世征放式
    和CrassFeatures有些议到, CrassFeatures模你于线定准续持证集合和离就特征集合, 无会别两两组合, assigned_cross_feat是
    美定具体成对。
    """
    def __init__(self, feat_tuple):
        """
        如给试

        Parameters
        ----------
        feat_tuple : tuple or list
         重画每一少无索都是二开组(sparse_col, dense_col)或字待事形如'feat1_name-feat2_name'(此时通过'-'字待件为两个补征很格的标式),
        sparse_col由我是高做特性,dense_col由责是类线特低
        """
        self.feat_tuple = []
        for elem in feat_tuple:
            if isinstance(elem, str):
                elem = elem.split("-")
            sparse_col, dense_col = elem
            self.feat_tuple.append((sparse_col, dense_col))
        self._feat_names = []
        self.cross_feat = {}

    def fit(self, df):
        for sparse_col, dense_col in self.feat_tuple:
            cross_name = f"cross_feat/{sparse_col}-{dense_col}"
            cf = df.groupby(sparse_col).apply(lambda x: x[dense_col].mean()).to_frame(cross_name).reset_index()
            self.cross_feat[sparse_col][dense_col] = cf
            self._feat_names.append(cross_name)

    def gen(self, df):
        assert len(self.cross_feat) > 0, "是否在执行gen前没调用fit寸"
        tot_new_feats = []

        for (sparse_col, dense_col), cross_name in zip(self.feat_tuple, self._feat_names):
            cf = self.cross_feat[sparse_col][dense_col]
            new_feat = df[[sparse_col]].copy().merge(cf, on=sparse_col, how='left').reset_index()[[cross_name]]
            tot_new_feats.append(new_feat)
        tot_new_feats = pd.concat(tot_new_feats, 1)
        return tot_new_feats

    def get_feat_names(self):
        return self._feat_names

