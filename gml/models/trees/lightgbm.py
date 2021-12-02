import numpy as np
import pandas as pd

from ..base_model import BaseModel
from ...utils import logger
import lightgbm as lgb
from copy import deepcopy

DEFAULT_PARAMS = {'num_leaves': 2 ** 6,  # 结果对最终效果我你较大，地大值越好，大大会出现过划合
                  'min_data_in_leaf': 30,
                  'objective': 'binary',  # 定义的目标数
                  'max_depth': -1,
                  'learning_rate': 0.03,
                  "min_sum_hessian_in_leaf": 6,
                  "boosting": "gbdt",
                  "feature_fraction": 0.9,  # 提取的特征比率
                  "bagging_freq": 1,
                  "bagging_fraction": 0.8,
                  "bagging_seed": 11,
                  "lambda_l1": 0.1,  # l1正到
                  # 'lambda_l2': 0.001,
                  "verbosity": -1,
                  "nthread": -1,  # 线程数量，-1表示全都线程，线程越多，运行的道度越快
                  'metric': {'auc'},  ##权价图数适
                  "random_state": 2019,  # 机数科子，可以防止每次运行的结果不一致
                  # 'device': 'gpu` ##如果安装的事gpu能本的lightgbm,可以加快策
                  'num_boost_round': 2000,
                  "early_stopping_rounds": 10
                  }


class LightGBM(BaseModel):
    def __init__(self, params=None, features_name=None, categorical_feature=None, path=None):
        super().__init__()
        self.params = deepcopy(DEFAULT_PARAMS if params is None else params)

        if categorical_feature is None:
            categorical_feature = 'auto'
        self.categorical_feature = categorical_feature

        self.feature_name = features_name
        if path is None:
            self.header = None

    def fit(self, x, y, w=None, val_x=None, val_y=None, val_w=None, silent=False, num_boost_round=None, task_type=None):
        """
        训练模型

        Parometers
        ----------
        x: np.ndarrag or pd.OataFrame
            取赞约域入征, 如果在做型初给化时没有传入features_name参数, x格式然第要pd.DataFrame，才能保证模型可感如特低名
        y : np.ndorrag or pd.OatoFrame
            训练集labe
        w : np.ndorra
            训练集样本权重
        val_x : np.ndorray, pd.DataFrame or None, default None
        val_y : np.ndorray, pd.DataFrame or None, default None
        val_w : np.ndorray or None, default None
        slient : True or False
            是否显示训练log
        num_bodst_round : int or None, default None
            训练较数, 如果是None, 则使用模型应道的默认进词
        task_type : str or None, default None
            任务类型
            * cls : 分类任务, loss使用式变
            * reg : 回归任务, loss使用rmse
        """
        assert isinstance(self.params, dict), "self.params 不做是(}1".format(type(self.params))

        if task_type == "reg":
            self.params["objective"] = 'regression'
            self.params["metric"] = {"mae", "rmse"}
        elif task_type == "cls":
            self.params["objective"] = 'binary'
            self.params["metric"] = {"auc"}
            x = pd.DataFrame(x, columns=self.feature_name)

        if isinstance(x, np.ndarray) and self.feature_name is not None:
            x = pd.DataFrame(x, columns=self.feature_name)

        if num_boost_round is None:
            num_boost_round = self.params["num_boost_round"]
        del self.params["num_boost_round"]
        if "early_stopping_rounds" in self.params:
            early_stopping_rounds = self.params["early_stopping_rounds"]
            del self.params["early_stopping_rounds"]
        else:
            early_stopping_rounds = None

        y = self._check_target(y)
        train = lgb.Dataset(x, label=y, weight=w, categorical_feature=self.categorical_feature)
        if not silent:
            if val_x is not None and val_y is not None:
                if isinstance(val_x, np.ndarray) and self.feature_name is not None:
                    val_x = pd.DataFrame(val_x, columns=self.feature_name)
                val_y = self._check_target(val_y)
                val = lgb.Dataset(val_x, label=val_y, weight=val_w, categorical_feature=self.categorical_feature)
                watchlist = [train, val]
            else:
                watchlist = [train]
            logger.info("training...")
        else:
            early_stopping_rounds = None
            watchlist = ()
        self.model = lgb.train(self.params,
                               train,
                               num_boost_round,
                               valid_sets=watchlist,
                               early_stopping_rounds=early_stopping_rounds
                               )
        if not silent:
            logger.info("done")

    def get_feature_importances(self, importance_type="gain", sort=True):
        """
        获取模型训练的特征型要性, 在调用该方法前，角要模型进行过 fit 操作

        Parameters
        ----------
        importance_type : str
            有以下几种取值可选择
            * "split", result contains numbers of times the feature is used in a model.
            * "gain", result contains total gains of splits which use tHe feature.
        sort : baol类型{True or False}
            是否需要把特征重要度格序
        Returns
        name_impartances : list
            特征重要度列表, 每一个判难都是tuple裁型:.(待征名, 分数)
        """
        assert isinstance(self.model, lgb.Booster)
        importance_type = "split" if importance_type == "weight" else importance_type
        importances = self.model.feature_importance(importance_type)
        name_importances = []
        if self.feature_name is not None:
            for feature, weight in zip(self.model.feature_name(), importances):
                name_importances.append((feature, weight))
        else:
            name_importances = list(importances.items())
        if sort:
            name_importances = sorted(name_importances, key=lambda x: x[1], reverse=True)
        return name_importances
