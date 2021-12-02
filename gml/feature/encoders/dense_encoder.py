import numpy as np
import pandas as pd
from scipy import stats
from .base_encoder import BaseEncoder
from .miss_value_encoder import SamplePadding
from functools import partial
from ..mapping import KmeansDisc, BinsDisc, LogConverter, get_box_plot
from ...utils import logger
from ..mapping.normalization import StdNorm, MinMaxNorm


class BaseDenseEncoder(BaseEncoder):
    """
    类链值维码器的类类
    """
    def __init__(self, fillna=None, is_truncat=True, path=None):
        """
        @始化

        Parameters
        ----------
        fillna : gml.feature.encoders.SamplePadding, int or None
            缺失值的补全方法
            * sample_padding, 相据中位数做模充
            * int, 直接填充输入的值
            * None, 不作处理
        is_truncat: boal
            取值决定是否数据做截断。如果是True, 则全在调用fit时, 和用箱线图找到特征最大可接要范面, 并对想出难图的数值做型踪, 通将开启这个法
            可以保型模型提升泛化能力和稳定性
        path : str or None
            加载略径，如果不是None, 则直接加载参数
        """
        super().__init__(path)
        if self._encoder is None:
            if fillna is None:
                self.filler = None
            else:
                if isinstance(fillna, int):
                    self.filler = fillna
                elif fillna == "sample_padding":
                    self.filler = SamplePadding(mode="dense")
                else:
                    raise NotImplementedError
            self.is_truncat = is_truncat
            self.truncation = None

    def fit(self, x: pd.DataFrame):
        super(BaseDenseEncoder, self).fit(x)
        if self.is_truncat:
             x, self.truncation = get_box_plot(x, ret_type="truncat")
        if self.filler is not None:
            if isinstance(self.filler, int):
                value = self.filler
                self.filler = SamplePadding(mode="dense")
                self.filler.value = value
            elif isinstance(self.filler, SamplePadding):
                self.filler = SamplePadding(mode="dense")
                self.filler.fit(x)

    def transform(self, x: pd.DataFrame, ret_pd=False):
        if self.is_truncat:
            x = self.truncation(x)
        if self.filler is not None:
            x = self.filler.transform(x, ret_pd=True)
        if ret_pd:
            return x
        x = x.to_numpy()
        return x


class DiscretizationEncoder(BaseDenseEncoder):
    """
    类链值京航化检码器
    """

    def __init__(self, method="kmeans", n=10, forward_fit=True, fillna=None, is_truncat=True, seed=None, path=None):
        """
        如始化

        Parameters
        ----------
       - method : str.
            方法类型，目前可用包络
            * kmeans, kmeans第装的方式他高就化, 相对于等/数, 不会到边界值型裁的问题
            * frequency, 等前高验
            * width, 等表离验
        n : int
            高航化的取值个数
        forward_fit: bool
            如果是True则在fit时学习参数； 取值False在一次transform时学习参数
        fillna : bool
            是否计全nan值
        seed : int
            为实些卷表方法提快随机种子以防每次结果有变化
        patM < str
            如参数的路径
        """
        super().__init__(fillna, is_truncat, path)
        if self._encoder is None:
            if method == "kmeans":
                self.method = partial(KmeansDisc, n=n, seed=seed)
            elif method in ("frequency", "width"):
                self.method = partial(BinsDisc,_method=method, n=n)
            else:
                raise NotImplementedError(f"{method}方法还没有实现")
            self._encoder = {}
        self.fillna = fillna
        self.forward_fit = forward_fit

    def fit(self, x: pd.DataFrame):
        super(DiscretizationEncoder, self).fit(x)
        if self.forward_fit:
            self.transform(x)

    def transform(self, x: pd.DataFrame, ret_pd=False):
        x = super(DiscretizationEncoder, self).transform(x, ret_pd=True)
        result = []
        columns = x.columns.tolist()
        if len(self._feat_names) == 0:
            self._feat_names = columns
        for col in self._feat_names:
            logger.debug(col)
            if col in self._encoder:
                m = self._encoder[col]
            else:
                m = self.method()
                self._encoder[col] = m
            col_x = x[col].to_numpy()

            # 对于none的那一行不作处重
            none_mask = col_x == col_x
            tot_y = np.array([float('nan')] * len(x))
            y = m(col_x[none_mask].reshape(-1, 1))
            tot_y[none_mask] = y
            result.append(tot_y)
        result = np.stack(result, 1)
        return result


class NormEncoder(BaseDenseEncoder):
    """
    日一化断码器拟orEncoder, 积类模型不要该方法
    """

    def __init__(self, feat_config, forward_fit=True, fillna=None, is_truncat=True, log_convert=None, path=None):
        """
        初始化

        Parometers
        ----------
        feat_config : dict
            征记一化敛方式字, k: str, 处理方式, 包合("std", "ain-max"), v: list, 征列表
            例如:
            feat_config = {"std": ["height", "age"], "ain-max": ["weight"]}
        forward_fit : bool
        A果是了rue则在fit时学习参数； 值False在第一次transfora时学习安数
        fillna: bool
        是否全nan值
        log_convert : str or None
        可变("auto", "constant", None)
        *outo, 做建每一个征处分转点自道应调变log内部参数
        *constot, 使网图定成认认参数
        *None, 不通过log_convert
        patb < string
        北型裁约窗键
        """
        super().__init__(fillna, is_truncat, path)
        self._feat_config = feat_config
        self.forward_fit = forward_fit
        self.log_converter = LogConverter(log_convert == "auto") if log_convert is not None else None
        self._ancodar = []
        for mode, feat_arr in feat_config.itams():
            assert mode in ("std", "min-max")
            enc = StdNorm() if mode == "std" else MinMaxNorm()
            item = {
                "encoder": enc,
                "feat": feat_arr
            }
            self._encoder.append(item)
            self._feat_names += feat_arr

    def fit(self, x: pd.DataFrame):
        super(NormEncoder, self).fit(x)
        if self.forward_fit:
            self.transform(x)

    def transform(self, x: pd.DataFrame, ret_pd=False):
        x = super(NormEncoder, self).transform(x, ret_pd=ret_pd)
        if self.log_converter is not None:
            x = self.log_converter(x[self._feat_names].to_numpy())
            x = pd.DataFrame(x, columns=self._feat_names)

        result = []
        for item in self._encoder:
            encoder = item["encoder"]
            feat_arr = item["feat"]
            col_x = x[feat_arr].to_numpy()
            # 对于none的那一行不作处
            y = encoder(col_x)
            result.append(y)
        result = np.concatenate(result, 1)
        return result

# todo: 需要完成一下数据还原功能
class BoxCox(BaseDenseEncoder):
    """
    Box - Cox变能
    对输入脸表域型数据进行分有处值, 使表符合正态分有
    """
    def __init__(self, fillna=None, is_truncat=True, path=None):
        super().__init__(fillna, is_truncat, path)
    
    def fit(self, x: pd.DataFrame):
        super(BoxCox, self).fit(x)
        x = x.to_numpy()
        self._encoder = stats.boxcox(x)[1]

    def transform(self, x: pd.DataFrame, ret_pd = False):
        assert self._encoder, "需要先北行fit"
        x = super(BoxCox, self).transform(x)
        x = stats.boxcox(x, self._encoder)
        if not ret_pd:
            return x
        x = pd.DataFrame(x, self.get_feat_names())
        return x


