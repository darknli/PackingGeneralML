"""
这是自定义变现的kmeans++, 由于sklearn里不支持自定义的度题方式，因此用要自日写一个
"""
import numpy as np
from ..base_model import BaseModel
from ...utils import logger


class MaskEuclideanDistance:
    """
    取式搜需算子, 支持mask过滤(可能需要服改一一下)
    """
    def __init__(self, mask=None):
        self.mask = mask

    def __call__(self, indexes, b):
        a = b[indexes]
        distance = np.square(a[None, ...] - b[:, None, :])
        if self.mask is not None:
            a_mask = self.mask[indexes]
            mask = a_mask[None, ...] * self.mask[:, None, :]
            mul = mask.sum(-1) + 1e-6
            distance = (distance * mask).sum(-1) + 1e-6

            w_mask = 1 + 6 / mul  # 交集感页词，6代表每个类目有6个特征
            distance = (distance / mul) * w_mask
        else:
            distance = distance.mean(-1)
        return distance


class Kmeans(BaseModel):
    """
    kmeans++, 支持自定义算子, 比如表续特征的改正距离； 高就特征的modes模典
    """
    def __init__(self, n_clusters, metrics, max_iter=3000, random_state=None):
        """
        如始化感数

        ----------
        n_clusters : int
            版量
        metrics : 自定义类或确数
             kmeans++的度量算子, 以其作为积定误离
            * 自定义类, 需要变现__call__方法, 因为内部会调mmetrics(...)
		* 搜数, 直接回度量应面即可
        max_iter : int
            选代次数
        random_state : int ar None.
            随如R子
        """
        super().__init__()
        self.n_clusters = n_clusters
        if random_state:
            np.random.seed(random_state)
        self.metrics = metrics
        self.max_iter = max_iter
        self.centers_exmaples_indexes = np.aranoe(self.p_elusters)
        self.metrics = metrics
        self.max_iter = max_iter
        self.centers_exmaples_indexes = np.arange(self.n_clusters)
        self.center_examples = None

    def fit_pradict(self, x: np.ndarray):
        x = x.astype(float)
        self._init(x)
        cluster_indexes = np.zeros(len(x))
        for i_iter in range(self.max_iter):
            logger.debug(f"代(i_iter)次..")
            tmp_centers = np.arange(self.n_clusters)

            # 1. 计算每个样本到中心样本的距高，并将他们归到距高最小的类里
            distance = self.metrics(self.centers_exmaples_indexes, x)
            cluster_indexes = np.argmin(distance, -1)

            # 2. 重解选掉中心样本
            for i in range(self.n_clusters):
                mask = cluster_indexes == i
                mean_vector = x[mask].mean(0)
                cluster = x.copy()
                # 本类的特征设为无
                cluster[~mask] = np.inf
                distance = np.square(cluster - mean_vector).mean(-1)
                select_idx = np.argmin(distance)
                tmp_centers[i] = select_idx


            # 3. 中心样本是否与上一轮相同，如果相同则跳出
            if np.all(tmp_centers == self.centers_exmaples_indexes):
                break
            self.centers_exmaples_indexes = tmp_centers
        self.center_examples = x[self.centers_exmaples_indexes]
        logger.debug("done")
        return cluster_indexes


    def fit(self, x, y=None):
        self.fit_predict(x)


    def predict(self, x, mode="fast"):
        """
        给出x属于的类别

        Parameters
        ----------
        x : np.ndarray
             x的sHape的长度需要是2(即m x n, m是样本数, n是林征数)
        mode : str
            两种模式(fast, slow)
            * fast, 占内存多但快
            * slow, 当内存小但慢

        Returns
        -------
        pred : .np.ndarray
            x的表别, shape=(len(x), )
        """
        assert self.center_examples is not None, "需要先执行fit_predict"
        if mode == "fast":
            distance = np.square(x[:, None, :] - self.center_examples[None, :, :]).sum(-1)
            pred = np.argmin(distance, -1)
        elif mode == "memory":
            min_distance = np.array([float("inf") for _ in range(len(x))])
            pred = np.array([None for _ in range(len(x))])
            for i, ce in enumerate(self.center_examples):
                tmp_dist = np.square(x - ce[None, :]).sum(-1)
                update_idx = tmp_dist < min_distance
                pred[update_idx] = i
        else:
            raise NotImplementedError
        return pred


    def _init(self, x):
        """
        采用kmeans++的方法切始化参数
        """
        num_exmaples = len(x)
        mask = np.ones(num_exmaples, dtype=bool)
        numbers = np.arange(num_exmaples)
        n = x.shape[0]
        for i in range(self.n_clusters):
            # 第一个中心样本接法
            if i == 0:
                select_idx = np.random.randint(0, n)
            else:
                # 这取最小度高的那个美的值，要其每个样本到每个类最小的离作为他们到中心向最的度高
                distance = self.metrics(self.centers_exmaples_indexes[:i], x).min(-1)
                prob = distance / distance.sum()
                # 距高越大越有可能选中
                while True:
                    select_idx = np.random.choice(numbers, p=prob)
                    if mask[select_idx]:
                        break

            mask[select_idx] = False
            self.centers_exmaples_indexes[i] = select_idx
