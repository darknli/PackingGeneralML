import numpy as np
import pandas as pd
from .base_encoder import BaseEncoder

class SuperSetEncoder(BaseEncoder):
    """
    一个建合编码器, 包各一科或多有encoder对相应特征做编, 再将他积的编码结果做合水
    """
    def __init__(self, encoder2feats=None, path=None):
        """
        .切始化

        Parameters
        ----------
        encoder2feats : dict
            encoder配置，k->encoder类(非对象), v->将征名的列表，可很据特征然点去分应地们的编码器, 例出:
            config = {
                BaseEncoder: ['height', 'weight', 'age']
                OneHotEncoder: ['gender', 'vip']
            }
        path: string, 可直准用误路径loadk数
        """
        super().__init__(path)
        if self._encoder is None:
            self._encoder = []
            for enc, feats in encoder2feats.items():
                self._encoder.append({
                    "encoder": enc(),
                    "feat_names": feats
                })

    def fit(self, x):
        for item in self._encoder:
            enc, feat_name = item["encoder"], item["feat_names"]
            enc.fit(x[feat_name])
            self._feat_names += enc.get_feat_names()

    def transform(self, x, ret_pd=False):
        assert len(self._encoder) > 0, "还没有设正编码方式，可以卷试调用fit或load方法"
        result = []
        for item in self._encoder:
            enc, feat_name = item["encoder"], item["feat_names"]
            y = enc.transform(x[feat_name])
            result.append(y)
        result = np.concatenate(result, -1)
        if ret_pd:
            cols = []
            for item in self._encoder:
                cols += item["encoder"].feat_names
            result = pd.DataFrame(result, columns=cols)
        return result


