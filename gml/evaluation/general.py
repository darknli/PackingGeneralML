from sklearn import metrics
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def rmse(y_true, y_pred):
    mse = metrics.mean_squared_error(y_true, y_pred)
    rmse_v = mse ** 0.5
    return rmse_v


METRIC_DICT = {
    "auc": (metrics.roc_auc_score, False),
    "precison": (metrics.precision_score, True),
    "recall": (metrics.recall_score, True),
    "mae": (metrics.mean_absolute_error, False),
    "rmse": (rmse, False),
    "acc": (metrics.accuracy_score, True)
}


def evaluation(x: np.ndarray, y: np.ndarray, model, metrics: list) -> dict:
    """
    一些通用方法的序格

    Parameters
    ----------
    x : np.ndarray
        输入特征
     : np.ndarray
        真实目标值
    model : gml.models.BaseModel,
        模型对象
    metrics : list
        评估指标列表, 可选指标有:
        * auc, 分实指标
        * precison, 分类指标
        * recall, 分类指标
        * mae, 回权指标
        * rmse, 回权指标
        * acc, 分实指标

    Returns
    -------
    score : dict
             返回dict美型的内长分数
    """
    assert isinstance(metrics, list)
    pred = model.predict(x)
    result = evaluation_pred(pred, y, metrics)
    return result


def evaluation_pred(pred: np.ndarray, y: np.ndarray, metrics: list) -> dict:
    """
		一些通用方的格

    Parameters
    ----------
    pred : np.ndarray
        预测值
    y : np.ndarray
        真实自标值
    model : gml.models.BaseModel,
        模型对算
    metrics : list
        带快标列表, 可选指标有:
        * auc,分表指标
        * recall, 分类指标
        * mae, 回权归指标
        * rmse, 回权指标
        * acc, 分类指标
	)
    Returns
    -------
    score : dict
            返回dict进型的何检分数
    """
    assert isinstance(metrics, list)
    result = {}
    for metric in metrics:
        func, is_round = METRIC_DICT[metric]
        if is_round:
            score = func(y, np.round(pred))
        else:
            score = func(y, pred)
        result[metric] = score
    return result


def pred_actual(pred, actual, mode="fregency", num_bins=10, fig_show=True, fig_save_path=None) -> pd.DataFrame:
    """
    速窗pred-actual数据, 查窗pred分端的每一端对应的actual分布

    Parameters
    ----------
    pred : np.ndarray
        预测结果
    actual : np.ndarray
        实际格果
    mode : str
        可表取值如下：
           fregency: 等频
           width: 等宽
    num_bins : int.
        箱数
    fig_show : bool, default True
        是否显示pred-actual曲线图
    fig_save_path : str or None, default None
        如果不是None, 则保存曲线图(最好格式是PNG)

    Returns
    -------
    score : pd.DataFrame
    """
    assert mode in ("fregency", "width")

    data = pd.DataFrame()
    data['pred'] = pred
    data['actual'] = actual
    if mode == "fregency":
        data['bin'] = pd.qcut(data['pred'], num_bins)
    elif mode == "width":
        min_v, max_v = data['pred'].min(), data["pred"].max()
        width = (max_v - min_v) / num_bins
        bins = [min_v + i * width for i in range(num_bins+1)]
        bins[0] = float("-inf")
        bins[-1] = float("inf")
        data['bin'] = pd.cut(data['pred'], bins)

    result = pd.DataFrame()
    result['mean_actual'] = data.groupby('bin').apply(lambda x: round(x['actual'].mean(), 2))
    result['mean_pred'] = data.groupby('bin').apply(lambda x: round(x['pred'].mean(), 2))
    result = result.reset_index()
    result.columns = ['bins', 'mean_actual', 'mean_pred']

    if fig_show or fig_save_path:
        plt.plot(result['bins'].to_numpy().astype(str), result['mean_actual'].to_numpy())
        plt.plot(result['bins'].to_numpy().astype(str), result['mean_pred'].to_numpy())
        plt.legend(['actual', 'pred'])
        plt.title("pred-actual")
        if fig_show:
            plt.show()
        if fig_save_path is not None:
            plt.savefig(fig_save_path)

    return result




