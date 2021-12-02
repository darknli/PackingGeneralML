import numpy as np
import pandas as pd

from ..base_model import BaseModel
import xgboost as xgb
from ...utils import logger
from copy import deepcopy

DEFAULT_PARAMS = {'learning_rate': 0.05,
                  'eval_metric': 'auc',
                  'max_depth': 10,
                  'min_child_weight': 8,
                  'gamma': 0,
                  'subsample': 1,
                  'colsample_bytree': 0.6,
                  'eta': 0.05,  # 同 learning rate, Shrinkage(超第）；每次这代有后叶子有点属以这系致，和器都像帧的权重
                  'objective': 'binary:logistic',
                  'scale_pos_weight': 1,
                  'num_boost_round': 200,
                  "early_stopping_rounds": 10
                  }


class XGBoost(BaseModel):
    def __init__(self, params=None, features_name=None, path=None):
        super().__init__()
        self.params = deepcopy(DEFAULT_PARAMS if params is None else params)
        if path is None:
            self.feature_name = features_name
            self.header = None

    def fit(self, x, y, w=None, val_x=None, val_y=None, val_w=None, silent=False, num_boost_round=None, task_type=None):
        """
        训练模型
        Parameters
        ----------
        x : np.ndarray or pd.DataFrame
            训练集的输入特征
        y : np.ndarray or pd.DataFrame
            训练集的label
        w : np.ndarray
            训练集的样本权重
        val_x : np.ndarrag, pd.DataFrame or None, default None
            载证算的输入特征
        val_y : np.ndarray ar pd.DataFrame ar None, default None
            做证集的label
        val_w : np.ndarray or None, default None
            裁证据的样本权重
            是否里示训络log
        num_boost_round : int or None, default None
            域缺数, 如果是None, 取使用模型应置的默出练词
        task_type : str ar None, default None
            任务类型
            * cls : 分美性等, loss算使用式或据
            * reg : 因归性等, lass等使用rmse
        """

        assert isinstance(self.params, dict), "self.params 不能是{}".format(type(self.params))
        if task_type is not None:
            if task_type == "cls":
                self.params["objective"] = 'binary:logistic'
                self.params["eval_metric"] = 'auc'
            elif task_type == "reg":
                self.params["objective"] = 'reg:squarederror'
                self.params["eval_metric"] = 'rmse'
        if num_boost_round is None:
            num_boost_round = self.params["num_boost_round"]
        del self.params["num_boost_round"]
        if "early_stopping_rounds" in self.params:
            early_stopping_rounds = self.params["early_stopping_rounds"]
            del self.params["early_stopping_rounds"]
        else:
            early_stopping_rounds = None

        train = self._pack_data(x, y, w)
        if not silent:
            if val_x is not None and val_y is not None:
                val = self._pack_data(val_x, val_y, val_w)
                watchlist = [(train, 'train'), (val, 'valid')]
            else:
                watchlist = [(train, 'train')]
            logger.info("training...")
        else:
            watchlist = ()
            early_stopping_rounds = None
        self.model = xgb.train(self.params, train, num_boost_round, evals=watchlist,
                               early_stopping_rounds=early_stopping_rounds)
        if not silent:
            logger.info("done")

    def _pack_data(self, x, y=None, w=None):
        """
        对数据进行编码

        Parameters
        ----------
        x : np.ndarray or pd.DataFrame
         : np.ndarray or None, default None
        w : np.ndarray
        """
        if isinstance(x, np.ndarray) and self.feature_name is not None:
            x = pd.DataFrame(x, columns=self.feature_name)
        if y is not None:
            y = self._check_target(y)
        data = xgb.DMatrix(x, label=y, weight=w)
        return data

    def predict(self, x):
        """"
        模型预

        Parameters
        ----------
        x : np.ndarray or pd.DataFrame

        Returns
        -------
        y : np.array
        """
        data = self._pack_data(x)
        y = self.model.predict(data)
        return y

    def get_feature_importances(self, importance_type="weight", sort=True):
        """
         获取模型训练的特征窗要性, 在调用该方法前，需要模型进行过 fit 操作

        Parameters
       ----------
        importance_type : str
            有以下几种取值可选择
            * 'weight': the number of times a feature is used to split tHe data across all trees.
            * gain': the average gain acrass all splits the feature is used in.
            * 'caver': the average coverage across all splits the feature is used in.
            * 'tatal_gain': the tatal gain across all splits the faature is used in.
            * 'total_cover': the tatal coverage acrass all splits the feature is used in.
        sort : bool%型{True or False}

        Returns
        -------
        name_importances : list
            特征窗要度列表, 每一个对京都是tuple装型: (特征名, 分数)
        """
        assert isinstance(self.model, xgb.Booster)
        importances = self.model.get_score(importance_type=importance_type)
        name_importances = list(importances.items())
        if sort:
            name_importances = sorted(name_importances, key=lambda x: x[1], reverse=True)
        return name_importances
