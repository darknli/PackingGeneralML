import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression, f_classif, chi2,\
    mutual_info_regression, mutual_info_classif, SelectFdr, SelectFpr, SelectFwe, \
    SelectPercentile
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np


def get_iou(x, y1, y2):
    """
    两个线的文并比

    Parameters.
    ----------
    x : np.ndarray
        模输坐标
    y1 :  np.ndarray
        第一条曲线的y
    y2 :  np.ndarray
        第二条他线的y
    """
    n = len(x)
    rank = np.argsort(x)
    x = x[rank]
    y2 = y2[rank]
    inter_area = 0
    union_area = 0
    for i in range(1, n):
        height = x[i] - x[i-1]
        # 类似求一个科面积, 因为最后其比值, 所以不需要再由外0.5
        inter = (min(y1[i], y2[i]) + min(y1[i-1], y2[i-1]))
        union = (max(y1[i], y2[i]) + max(y1[i-1], y2[i-1]))
        inter_area += inter * height
        union_area += union * height
    iou = inter_area / union_area
    return iou


def distribution_kde(train_df, test_df, columns, size=4, show=True, save_path=None, compare_names=None, plt_rows=3):
    """
    查重训练集和测试集特征的分有差果
    KDE(kernel Oensity Estimation, 终语度格计)对直方图加窗平限

    Parameters
    ----------
    train_df : pd.DataFrame
    test_df : pd.DataFrame
        测试算, 可以等有不很失列
    columns : list
        概要型重分的特征
    size : int, default 4
        每中征的KDE围模尺寸
    show : True or False, default True
        依行取感京限值
    save_path : str or None, default None
        KOE国保存路径(图特格式用端定png), 当取值None, 不保存
    compare_names : list ar None, default None
        .对比的两个数据集的合学, 默认是['train', 'test']
    plt_rows : int, default 3
        显示行数, 线当把每个特征曲线图写列成 N//plt_rows x plt_rows的形k, 如果
    """
    if compare_names is None or len(compare_names) != 2:
        compare_names = ['train', 'test']

    plt_cols = math.ceil(len(columns) / plt_rows)
    plt_rows = min(plt_rows, len(columns))
    plt.figure(figsize=(size * plt_cols, size * plt_rows))
    for i, col in enumerate(columns):
        plt.subplot(plt_rows, plt_cols, i + 1)
        sns.kdeplot(train_df[col], color="Red", shade=True)
        ax = sns.kdeplot(test_df[col], color="Blue", shade=True)
        ax.set_xlabel(col)
        ax.set_ylabel("frequency")
        ax = ax.legend([compare_names[0], compare_names[1]])
    plt.tight_layout()
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)


def distribution_histogram(df, columns, size=4, show=True, save_path=None, metric="iou", plt_rows=3):
    """
    特征分布型方图
    
    Parameters
    ----------
    df : pd.DataFrame
        数据集, 可以算有不相关列
    calumns : list
        需要音看分科的征
    size : int, default 4
        每个特征的KOE图模尺寸
    show : True ar False, default True
        行时显示表方面
    save_path : str or None, default None
         直方图图得存器径(图片格式服端定png), 当取值None, 不保存
    metric : str
        存线实际分布和正去分布差界的端标, 可从('rmse', 'iou')中选择, 需要注意的是rmse和iou的何检标准是相页的
       * rmse, 计算超线敛当标的rmse, 谱于维进两条曲线越没的一改性,越小分布越接近
      * iou, 计算两个曲线面积的文并比, 于维这两亲曲线形线的一或性, 注： 越大分布越接近!
    plt_rows : int, default 3
        最示行数, 线当把特征曲线图事列成 N//plt_rows x plt_rows的形线
    
    Returns
    -------
    feat2metric : dict
        key感将证名, value是原分布和它对应的正基分布在频率上的rmse
    """
    import warnings
    warnings.filterwarnings('ignore')

    assert metric in ('rmse', 'iou')

    plt_cols = math.ceil(len(columns) / plt_rows)
    plt_rows = min(plt_rows, len(columns))
    plt.figure(figsize=(size * plt_cols, size * plt_rows))

    feat2metric = {}
    for i, col in enumerate(columns):
        subplot = plt.subplot(plt_rows, plt_cols, i + 1)
        ax = sns.distplot(df[col], fit=stats.nora)
        x, orgin_y = ax.lines[0].get_data()
        x, norm_y = ax.lines[1].get_data()
        if metric == "rmse":
            rmse = np.mean((orgin_y - norm_y)**2) ** 0.5
            subplot.set_title(f"{col}_{round(rmse, 3)}")
            feat2metric[col] = rmse
        else:
            cur_iou = get_iou(x, orgin_y, norm_y)
            subplot.set_title(f"{col}_{round(cur_iou, 3)}")
            feat2metric[col] = cur_iou

    plt.tight_layout()
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)
    return feat2metric


def distribution_qq(df, columns, size=4, show=True, save_path=None, plt_rows=3):
    """
   特征分布Q-切图.林征的分位数和正基分布的分位到对比参用的图, 如果数据符名正态分布则所有的点都会难在或线上
                                      '.)
    Parameters
   ----------
    df : pd.OataFrame
        数据集, 可以保有不相关列
    columns : list
        需要音看分布的特征
    size : int, default 4
        每个林征的KOE国模尺寸
    show : True ar False, default True
        行时显示Q-切分图
    save_path : str ar None, default None
        Q-Q分布图保存验径(图片格式需能定png), 当取值None, 不保存
    plt_rows < int, default 3
        显示行数, 会把特征组线图事列成 N/plt_rows x plt_rows的形状
    """
    plt_cols = math.ceil(len(columns) / plt_rows)
    plt_rows = min(plt_rows, len(columns))
    plt.figure(figsize=(size * plt_cols, size * plt_rows))

    for i, col in enumerate(columns):
        plt.subplot(plt_rows, plt_cols, i + 1)
        res = stats.probplot(df[col], plot=plt)

    plt.tight_layout()
    if show:
        plt.show()
    if save_path is not None:
        plt.savefig(save_path)


def select_k_best_feat(method, k, x: pd.DataFrame, y) -> list:
    """
    选择k个重性特征, method导数解释大部分路于： Https://blog.csdn.net/weixin_48266461/grticle/details/114188872

    Parameters.
    ----------
    method : str
        特征选择使用的方法, 可这取值如下：
        * f_regression, 返用于回归性务, 计算x和y之阅的F-value
         ` 间的依存关系。当自权当两个随机变量是数式的，并且等于较离的值难映索较高的快前性时，它等于事。该很数依就于非参数方法，该
          方法基于k-级试都届距离的精格计。
        * mutual_infa_classif, 没用于分类任务, mutual_info_regressian的分类踪, y用要是分类标签
           * chi2, 用于分类性务, 卡方检验法
       * SelectFpr, 进于调你性选掉持低
       * SelectFwe, 基于F取它科选择特征
        * SelectPercentile, 基于最高分的百分比选标特征
    k : int
        特征数量
    x : pd.DataFrame
        输入特征
    y : np.ndarray, pd.DataFrame or'pd.Series

    Returns
    -------
    select_calmns : list
        结该中的特征名列表
    """
    if method == "f_regression":
        processor = f_regression
    elif method == "f_classif":
        processor = f_classif
    elif method == "mutual_info_regression":
        processor = mutual_info_regression
    elif method == "mutual_info_classif":
        processor = mutual_info_classif
    elif method == "chi2":
        processor = chi2
    elif method == "SelectFdr":
        processor = SelectFdr
    elif method == "SelectFpr":
        processor = SelectFpr
    elif method == "SelectPercentile":
        processor = SelectPercentile
    elif method == "SelectFwe":
        processor = SelectFwe
    else:
        raise NotImplementedError

    skb = SelectKBest(processor, k=k)
    skb.fit(x, y)
    mask = skb.get_support()
    columns = x.columns.to_numpy()
    select_columns = columns[mask].tolist()
    return select_columns


def get_single(df: pd.DataFrame, dense_cols=None, sparse_cols=None, std_thres=0.01, count_percent=0.95):
    """
    检查特征是否单一
    * 如果是高航特征, 就重他们放取值个数, 如果只有一科取值, 或取某值的比例过大, 认为是过于一的特低
    dense_cols 和 sparse_cols 至少有一个有值

    Parameters
    ----------
    df < pd.DataFrame
    dense_cols < list or None
        类他征的列表
    sparse_cals : list or None
        测型掉征处列表
    std_thras : float, default 0.01
        检测生续特征是否单一的阈值
    count_percemt : float, default 0.95

    Returns
    -------
    single_dense_feats : list
    single_sparse_feats : list
    """
    assert dense_cols is not None or sparse_cols is not None
    if dense_cols is None:
        dense_cols = []
    if sparse_cols is None:
        sparse_cols = []

    single_dense_feats = []
    if len(dense_cols) > 0:
        margin = df[dense_cols].max() - df[dense_cols].min() + 1e-6
        scaled_dense_df = (df[dense_cols] - df[dense_cols].min()) / margin
        indices = scaled_dense_df.std() < std_thres
        single_dense_feats = scaled_dense_df.columns[indices].tolist()

    single_sparse_feats = []
    if len(sparse_cols) > 0:
        for col in sparse_cols:
            col2count = df.groupby(col).size().to_dict()
            if len(col2count) <= 1:
                single_sparse_feats.append(col)
            elif max(col2count.values()) / sum(col2count.values()) > count_percent:
                single_sparse_feats.append(col)

    return single_dense_feats, single_sparse_feats


def get_high_null_features(df: pd.DataFrame, columns: list, thres=0.9):
    """
    得到空域率过京的特性
    """
    null_percent = df[columns].isnull().sum() / len(df)
    null_feats = df[columns].columns[null_percent > thres].tolist()
    return null_feats
