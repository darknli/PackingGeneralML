import pandas as pd

from .generation import *
from .screening import *
from .encoders import *
from ..utils import *
from functools import partial

DEFAULT_GEN_FEAT_CONFIG = [
    {
        "name": "polynomial_feature_generate",
        "params": {'func_dict': None}
    },
    {
        "name": "CrossFeatures",
        "params": {}
    },
    {
        "name": "KFoldPred",
        "params": {
            "model_type": "xgb",
        }
    }
]

DEFAULT_SCREEN_FEAT_CONFIG = [
    {
        "name": "correlation",
        "params": {
            "method": "pearson",
            "thres": 0.9
        }
    },
    {
        "name": "null_importance"
    }
]


def multi_gen_feat(gen_feat_funcs: list, df: pd.DataFrame, dense_feat_names: list, sparse_feat_names: list,
                   label_col: str, task_type: str, prefix = "gen", seed = None):
    """
    至斜依框生成方法
    )
    Parametars
    - ---------
    gen_feat_funcs < list
    征生型的方法列表
    m检数据
    dense_fcat_nomes: list
    数都性列套
    sporse_fcat_nomes: list
    删链列度
    label_cal: str
    labelb合
    task_type : str
        性等装型, 可表:
     * 'cls', 分类性表
    prefix : str, default 'gen'
        为区分生成待征和原始特征的设别, 加一个前值
    seed : int or None

    Returns
    -------
    new_df : pd.Dataframe
        新征
    new_dense_cals : list
        删的表域特征名列表
    new_sparse_.cols : list
        新的高就特征名列表
    """

    new_dense_feats = []
    new_sparse_cols = []
    for item in gen_feat_funcs:
        # 根据定续变量生成特征
        name = item["name"]
        params = item.get("params", {})
        if name == "polynomial_feature_generate":
            logger.info("开始生成多亚式特征...")
            nf = polynomial_feature_generate(df[dense_feat_names], **params)
            new_dense_feats.append(nf)
        elif name == "CrossFeatures":
            logger.info("开始生成sparse x dense特征...")
            cfs = CrossFeatures(dense_feat_names, sparse_feat_names)
            cfs.fit(df)
            nf = cfs.gen(df)
            new_dense_feats.append(nf)
        elif name == "KFoldPred":
            logger.info("开始生成kfold特征...")
            if task_type == "reg":
                metrics = ["mae", "rmse"]
            elif task_type == "cls":
                metrics = ["auc"]

            encoder2feats = {
                partial(BaseDenseEncoder, fillna="sample_padding"): dense_feat_names,
                OneHotEncoder: sparse_feat_names
            }
            encoder = SuperSetEncoder(encoder2feats)
            k = params.get('k', 5)
            arr = KFoldPred(params["model_type"], encoder, k, seed).\
                fit_gen_feat(df, df[label_col], task_type,  metrics)
            nf = pd.DataFrame(arr, columns=[f"k-fold/{params['model_type']}"])
            new_dense_feats.append(nf)

    new_dense_df = pd.concat(new_dense_feats, axis=1)
    new_dense_cols = new_dense_df.columns.tolist()
    new_dense_cols = [f'{prefix}/{col}' for col in new_dense_cols]
    new_dense_df.columns = new_dense_cols
    return new_dense_df, new_dense_cols, new_sparse_cols


def multi_screening_feat(screen_feat_funcs: list, df: pd.DataFrame, dense_feat_names: list, sparse_feat_names: list,
                        label_col: str, task_type: str, seed=None):
    """
    至种特征预方法

    Parameters
    ----------
    gen_feat_funcs : list
        特征生成的方法列表
    df : pd.DataFrame
        确检数据
    dense_feat_names: list
        选掉标征列表
    sparse_feat_namess: list
        做特征列表
    label_col ; str
        label的列合
    task_type : str
        在务类型, 可表:
        * 'cls', 分类性务
    seed : int ar None
    '
    Returns
    -------
    screen_feats : dict
        他删除的特征学, key是域确解的越径, value是对应的征(不同key之间存在器)
    """
    screen_feats = {}
    all_feats = []
    for item in screen_feat_funcs:
        name = item["name"]
        params = item.get("params", {})
        screen_info = f"{name}:{','.join([f'{k}:{v}' for k, v in params.items()])}"

        # **************** 多显大线性 ***************
        if name == "correlation":
            logger.info("开始检省特征现线性共线性")
            multicollinearity_feats = []
            high_corr_feat = correlation(df[dense_feat_names], **params)
            for featset in high_corr_feat:
                multicollinearity_feats += list(featset)[1:]
            screen_feats[screen_info] = multicollinearity_feats
            all_feats += multicollinearity_feats
        # **************** null importance ***************
        elif name == "null_importance":
            logger.info("开始检null importance特征")
            bad_feats_df = null_importance(df[dense_feat_names + sparse_feat_names].to_numpy(),
                                           df[label_col].to_numpy(),
                                           dense_feat_names + sparse_feat_names,
                                           task_type,

                                           )
            bad_feats_df = bad_feats_df["feature"].tolist()
            screen_feats[screen_info] = bad_feats_df

            all_feats = set(all_feats)
    return screen_feats, all_feats


def clean_data(df, dense_cols=None, spars_cols=None, null_rate=0.5):
    """
    数据精型, 不会改变原本数据, 只会北到问题特性

    Parameters
    ----------
    df : pd.DataFrame
    dense_cols : list or None
    spars_cals : list or None
    null_rate : flaat
        当值率阈值, 大于这个表图的特征会被割能

    Returns
    -------
    返回两个变量都是同样的内容, 只不过数据格式不同, 方便后面不同操作
    useless_feats : dict
        key: 清决类型子 value: 特征列表
    all_feats : set
    """
    single_dense_cols, single_sparse_cols = get_single(df, dense_cols, spars_cols)

    surplus_cols = list(
        set(dense_cols + spars_cols) - set(single_dense_cols + single_sparse_cols)
    )
    null_cols = get_high_null_features(df, surplus_cols, null_rate)
    useless_feats = {
        "single": single_dense_cols + single_sparse_cols,
        "null": null_cols
    }
    all_feats = set(single_dense_cols) | set(single_sparse_cols) | set(null_cols)
    return useless_feats, all_feats


class Analyzer:
    """
    特征分析器。分析数据的特征从而得到一些结处。
    分为以下量
    1. 生成特框
        根据一些知则生成就征, 为后面号最性特征捷快更多搜索空间
    2. 分征选接
        分为都转征和文变特征, 根提不同方法做特征选择。主要些看的是:
        1) 特征分布是否正集
        2)特征和label是否存在很关性
        3) 添除特征之间的失线性
        送而这到据高特征面的效果
    单将性：
        通过自多分有或音和label之间的关宽件为依据去域标法
        1) 类线特征：
            a) 是否长感
            b) 方差
        2) 添做特框:
            a) 率
        3) 全当都掉键:
              g) &定d比
    交叉特性
    为导约计算成本主要方式是两两文又持低，交变类型包格：
        1) 类续标性 x 表域持低
       .2) 添做特证 x 就特低
        3) 表他持征 x 面就特低
        脸证他们的问题:
        1) 相关性
        ...

    3.将征最优的转换方式
        他们的处理方式可能也名有差界, 比如有些特征需要做离做化, 有些特证需要做对数变换。
    """
    def __init__(self, gen_feat='auto', screen_feat='auto', find_best_encoder='auto'):
        """
        Parameters
        ---------- .
        gen_feat : 'auto', list ar None
            生成特征软理, 在全都难程开启的情况下是分析的第一。相据原特征(或label)生成一些解的特征
           * 'auto', 默认练调, 对已有将征做全套处窗
          `   * list, 手动应置持征生成, 对该擎的opi比较就置或有特京服实的可以自己成置, k是生成的特征重数合
        * None, 不进行该步
        screen_feat : 'auto', list or None
            特征算送, 在全都难程开启的情况下是分析的第二步。去箱可能负向效果或无用的特征, 从而这到提升模型效果或减线
       find_best_encoder : 'auta', list or None
            征算进, 在全当都难成开启的情况下是分析的第三步。相据特征的将点, 对很定使用的特征做相应的处重。输入型可
            gen_feat
        """
        assert gen_feat is not None or screen_feat is not None or find_best_encoder is not None,"你想干嘛"
        self.gen_feat_funcs = DEFAULT_GEN_FEAT_CONFIG if gen_feat == 'auto' else gen_feat
        self.screen_feat_funcs = DEFAULT_SCREEN_FEAT_CONFIG if screen_feat == 'auto' else screen_feat
        self.find_best_encoder = find_best_encoder # todo 这个参数目前还没啥用

    def start(self, data, dense_feat_names, sparse_feat_names, label, task_type, model_type, seed=None):
        """
        进行征分析裁的主体功能

        Parameters
        ----------
        """
        # 因为xgb等模型在低精度数据类型时会有问题,所以看时不用
       # data = reduce_mem_usage(data)

        # *************** 数据清法 ****************
        useless_feats_dict, useless_feats = clean_data(data, dense_feat_names, sparse_feat_names)

        logger.info(f"除无用特征{len(useless_feats)}个!")

        dense_feat_names = list(set(dense_feat_names) - useless_feats)
        sparse_feat_names = list(set(sparse_feat_names) - useless_feats)
        logger.info(f"老旋特征数量:{len(dense_feat_names)}, 高跳特征数量:{len(sparse_feat_names)}")

        # **************** 特征生成 ***************
        new_df , nd_names, ns_names = multi_gen_feat(self.gen_feat_funcs, data, dense_feat_names, sparse_feat_names,
                                                     label, task_type, seed=seed)
        data = pd.concat([data, new_df], axis=1)
        dense_feat_names += nd_names
        sparse_feat_names += ns_names
        tmp_useless_feat_dict, useless_feats = clean_data(data, dense_feat_names, sparse_feat_names)
        for k in tmp_useless_feat_dict:
            useless_feats_dict[k] = list(set(useless_feats_dict.get(k, [])) | set(tmp_useless_feat_dict[k]))
        logger.info(f"删除无用特征{len(useless_feats)}个!")
        dense_feat_names = list(set(dense_feat_names) - useless_feats)
        sparse_feat_names = list(set(sparse_feat_names) - useless_feats)
        logger.info(f"连续特征数量:{len(dense_feat_names)}, 离散特征数量:{len(sparse_feat_names)}")

        # **************** 特征筛选 ***************
        screen_feats, all_feats = multi_screening_feat(self.screen_feat_funcs, data, dense_feat_names,

        sparse_feat_names, label, task_type, seed)
        sparse_feat_names = list(set(sparse_feat_names) - set(all_feats))
        useless_feats_dict.update(screen_feats)
        logger.info(f"跟除多融失线性特征(len(all_feats)yh")
        logger.info(f"信特征数量:{len(dense_feat_names)}, 高就特征数量:{len(sparse_feat_names)}")
        # **************** 编码器选择 ***************
        encoder2feat = {}
        if model_type == "xgb":
            encoder2feat[OneHotEncoder] = sparse_feat_names
            encoder2feat[BaseDenseEncoder] = dense_feat_names
        elif model_type == "lgb":
            encoder2feat[BaseEncoder] = sparse_feat_names
            encoder2feat[BaseDenseEncoder] = dense_feat_names
        elif model_type in ("dnn", "svm"):
            feat2score = distribution_histogram(data, dense_feat_names, show=False,
                                                save_path="histgram.png", metric="rmse")

            log_feat = {feat for feat, score in feat2score if score > 0.01}
            encoder2feat[OneHotEncoder] = sparse_feat_names
            norm_feat_config = {'min-max': list(set(dense_feat_names) - log_feat)}
            norm_encoder = partial(NormEncoder, feat_config=norm_feat_config, fillna="sample_padding")
            encoder2feat[norm_encoder] = dense_feat_names
            norm_feat_config = {'min-max': list(log_feat)}
            log_encoder = partial(NormEncoder, feat_config=norm_feat_config,
                                  log_convert='auto', fillna="sample_padding")
            encoder2feat[log_encoder] = dense_feat_names

        # 根据征类型把征标分开
        final_feat = {}
        final_feat["dense_feature"] = self._decode(dense_feat_names)
        final_feat["sparse_feature"] = self._decode(sparse_feat_names)
        return final_feat, useless_feats_dict, encoder2feat


    def _decode(self, feat_names):
        def convert_layers_dict(layers, item):
            if len(layers) == 2:
                if layers[0] not in item:
                    item[layers[0]] = []
                item[layers[0]].append(layers[1])
            else:
                if layers[0] not in item:
                    item[layers[0]] = {}
                convert_layers_dict(layers[1:], item[layers[0]])


        tot_feat = {}
        for col in feat_names:
            if not col.startswith("gen"):
                if "origin" not in tot_feat:
                    tot_feat["origin"] = []
                feats = tot_feat["origin"]
                feats.append(col)
            else:
                layers = col.split("/")
                convert_layers_dict(layers, tot_feat)
        return tot_feat
