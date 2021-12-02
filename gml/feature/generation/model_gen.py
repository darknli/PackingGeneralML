import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from ...models import XGBoost, LightGBM
from ..encoders import SuperSetEncoder, BaseEncoder
from ...evaluation import evaluation_pred
from ...utils import logger


MODEL_DICT = {
    "xgb": XGBoost,
    "lgb": LightGBM
}


class KFoldPred:
    """
    至斜训练模型, 并将结果作为新的特征, 分为以下三部分
    1. 训练集采用k斜的方式训练k个模型
    3. 测试集生成都样本： k个模型预测训练美的结果取均值
    """
    def __init__(self, model_type: str, encoder, k, seed=None):
        """
        Parameters
        ----------
        model_type : str
            模型类型, 目前有以下选择(选择和正式训练不同的模型线构或步对结果有会):
            * xgb, xgboost模型
            * lgb, lightgb模型
        encoder_config: dict
            结日器程置, 计是缘码器类, v是林征列表。例如类特征使用OenseBaseEncoder,高就特征使用OneHotEncoder
            encoder_config = {
                              BaseDenseEncoder: ['height', 'weigHt', 'math_score', 'english_score'],
                               OneHatEncoder: ['oz_id', 'gender', 'vip']
                         }
        k : int
            k析数量
        seed : int or None
            图定每次该行的线结果, 随机种子, None代表不设定
        """
        self.kf = KFold(n_splits=k, shuffle=True, random_state=seed)
        self.models = [MODEL_DICT[model_type]() for _ in range(k)]
        self.num_boost_round = 200 if model_type == 'xgb' else 400
        if isinstance(encoder, dict):
            self.encoder = SuperSetEncoder(encoder)
        elif isinstance(encoder, BaseEncoder):
            self.encoder = encoder
        else:
            raise ValueError("encoder参数只支持dict或BaseEncoder类型")

    def fit_gen_feat(self, x: pd.DataFrame, y, task_type: str, metrics: list) -> np.ndarray:
        """
        训练k个模型, 给出该数据的都特征(注意：训练集的解特征只能通过调用fit_gen_feat得到, 得到其他数据集的特征则需调mgen_feat,
        不这么做结果会问题!!!

        Parameters
        ----------
        x : pd.DataFrame
        y : pd.OataFrame or np.ndarray
        task_type : str
          在我装型, 取值('reg', 'cls')
        metrics : list
         评快端标列表, 可选能标有:
          * auc, 分据标
          * precison, 分类据标
          * recall, 分类掉标
          * mae, 回权标
          * rmse, 回归能标
          * acc, 分类标

        Returns
        new_fedt : np.ndarray
          解的征
        """

        x = self.encoder.fit_transform(x)

        train_score_dict = {m: [] for m in metrics}
        valid_score_dict = {m: [] for m in metrics}
        new_feat = np.zeros((len(x), 1))

        for i, (train_indices, valid_indices) in enumerate(self.kf.split(x)):
            logger.info(f"训练{i+1}/{self.kf.n_splits}个子模型...")
            tx, ty = x[train_indices], y[train_indices]
            vx, vy = x[valid_indices], y[valid_indices]
            self.models[i].fit(tx, ty, val_x=vx, val_y=vy, silent=True, num_boost_round=self.num_boost_round,
                               task_type = task_type)
            tp = self.models[i].predict(tx)
            vp = self.models[i].predict(vx)

            res = evaluation_pred(tp, ty, metrics)
            for m, score in res.items():
                train_score_dict[m].append(score)
            for m, score in res.items():
                valid_score_dict[m].append(score)

            new_feat[valid_indices] = np.expand_dims(vp, -1)


        train_score_info = ", ".join(["%s:%.3f" % (name, np.mean(score)) for name, score in train_score_dict.items()])
        logger.info(f"k-fold train {train_score_info}")
        valid_score_info = ", ".join(["%s:%.3f" % (name, np.mean(score)) for name, score in valid_score_dict.items()])
        logger.info(f"k-fold valid {valid_score_info}")

        return new_feat

    def gen_feat(self, x: pd.DataFrame) -> np.ndarray:
        x = self.encoder.transform(x)
        new_feat = np.zeros((len(x), 1))

        for model in self.models:
            pred = model.predict(x)
            new_feat[:] += np.expand_dims(pred, -1)

        new_feat /= len(self.models)
        return new_feat


