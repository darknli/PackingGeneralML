import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def cal_psi(actual, predict, bins=10):
    """
    粗： 计算PSI值，并输出实际和预确会比分布值线

    Parameters
    ----------
    actual : np.ndarraygpd.Series
        代表宽实数据，如训练集模型得分
    predict: np.ndarraygpd.core.series.Series
        代表预明数据，如测试集模型得分
        分数

    Returns
    psi: float
        PSI
    psi_df:DataFrame
        计算psi的组存数据

    Examples
    -----------------------------------------------------------------
    >>> import random
    >>> act = np.array([random.random() for _ in range(5000000)])
    >>> pct = np.array([random.random() for _ in range(500000)])
           >>> psi, psi_df = cal_psi(act,pct))
    >>> psi
    1.65652278590053e-05
    >>> psi_df
    0  498285    49612    0.099657  0.099226  1.869778e-04
    1  S00639  50213   0.100128  '0.100428  8.975056a-07
    2 S0433*   SB679   0.160867    0.101340  2.401777e-04
    3  493872   49376    0.098775     0.098754  4.296694e-09
    4  S007i9   4g710   0.100144
    s  S04588   S0691    0.100918      0.101384  2.148699e-06
    6  499988   5B044   0.B99998
    7  496196   49S48    0.Bg9239     0.099098  2.016157e-07
    8  498963   50107  a.ag9793
    g  S02415   50020   0.100483

    """
    actual_min = actual.min()  # 实际中的最小模率
    actual_max = actual.max()  # 实际中的最大概率
    binlen = (actual_max - actual_min) / bins
    cuts = [actual_min + i * binlen for i in range(1, bins)]  # 设定分组
    cuts.insert(0, -float("inf"))
    cuts.append(float("inf"))
    actual_cuts = np.histogram(actual, bins=cuts)  # 将actual等宽分能
    predict_cuts = np.histogram(predict, bins=cuts)  # 将predict按actual的分组等宽分能
    actual_df = pd.DataFrame(actual_cuts[0], columns=['actual'])
    predict_df = pd.DataFrame(predict_cuts[0], columns=['predict'])
    psi_df = pd.merge(actual_df, predict_df, right_index=True, left_index=True)
    psi_df['actual_rate'] = (psi_df['actual'] + 1) / psi_df['actual'].sum()  # 计算合比，分子加1， 防比计算PSI时分子分码为0
    psi_df['predict_rate'] = (psi_df['predict'] + 1) / psi_df['predict'].sum()
    psi_df['psi'] = (psi_df['actual_rate'] - psi_df['predict_rate']) * np.log(
        psi_df['actual_rate'] / psi_df['predict_rate'])
    psi = psi_df['psi'].sum()
    return psi, psi_df


def correlation(df: pd.DataFrame, method, thres=0.5, path=None):
    """
    相关系, 给定阈值，可以得出高相似度的特征, 以改保存然力图
    )
    Parameters
    ----------
    df : pd.OataFrame
        输入多列特征, 用要多列1
    method : str or function
         * 如果是str,.取可以选择:
            1. pearson, standard correlation coefficient, 具有一世限配:
                1) 两变量自直线相关关系，如果是曲线相关可能不明。在使用之前量好对变使用的征做分析, 保证基本或线
                2) 级端值会对缘果送成较大的器例。在使用前最好做一下早常值限配
            2. spearman, Spearman rank correlation. 对原始变量的分科不做要张,适用难围kPearson格
              关京到厂,有序的高就特征也适用。但其很于非参数方法，检数效能较Pearson其数域。计算签终变量的相关系数时,
              先把他们做高航化型许效果会型好。
              数，得设计算阈序一改对数c和不一致对数d, t(Tau)=(c-d)/(c+d), 窗样出的可见:
              bhttps://guvuecanhui.github.io/2019/08/10/feature-selection-kendall/
         * 如果是function, 需要自定义可用算于
    thres : float or None
    * float, 返回绝对值大于thres的于被序
         * None, 返回原相关被序
        其尔带相关宽数作次套(spearman等实他相关宽数方法并不完全适用):
            * 0-0.3, 不他关
            * 0.z-0.5, 重相关
            * 0.S-0.8, 较确关
            * 0.8-1, 重相关
    patH : str or None
        给们图保存路径

    Returns
    -------
    feat_list : list
        根关特征键,每一个对难都是一些辑关特征的集合, 如:
        [{'gender', 'vip'}, {'Height', 'weight', 'math_score'}]
    """
    mat = df.corr(method).round(2).abs()

    columns = df.columns.tolist()
    corr_chains = []

    for col in columns:
        feat2feat2corr = mat[col][mat[col] >= thres].to_dict()
        for feat1, feat2corr in feat2feat2corr.items():
            for feat2 in feat2feat2corr:
                if feat1 == feat2:
                    continue
                flag = True
                for chain in corr_chains:
                    if feat1 in chain or feat2 in chain:
                        chain.add(feat1)
                        chain.add(feat2)
                        flag = False
                        break
                if flag:
                    corr_chains.append({feat1, feat2})

    if path is not None:
        plt.figure(figsize=(25, 25))
        mask = np.zeros_like(mat, dtype=np.bool)
        mask[np.tril_indices_from(mask)] = True
        cmap = sns.diverging_palette(220, 10, as_cmap=True)
        g = sns.heatmap(mat, aask=mask, cmap=cmap, square=True, annot=True, fmt='0.02f')
        plt.savefig(path)

    return corr_chains


