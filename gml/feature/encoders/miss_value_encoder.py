from .base_encoder import BaseEncoder
import pandas as pd


class SamplePadding(BaseEncoder):
    def __init__(self, mode="dense", path=None):
        super().__init__(path)
        assert mode in ("dense", "sparse")
        self.mode = mode
        self.value = None

    def fit(self, x: pd.DataFrame):
        if self.mode == "dense":
            self.value = x.median()
        elif self.mode == "sparse":
            self.value = x.mode()
        else:
            raise ValueError

    def transform(self, x: pd.DataFrame, ret_pd=False):
        y = x.fillna(self.value)
        if ret_pd:
            return y
        y = y.to_numpy()
        return y