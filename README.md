# 通用ML包
侧重于持征约理, 自动做特征分析, 有针对性的编码, 每在防止做过多重复工作

## 使用方法
1. 安装whl包
<pre>conda create -n gml python=3.6
bash build_whl.sh
cd dist &E pip install general_ml_darkn-*.whl
</pre>
2. 调用特征分析代码示例
<pre>
import pandas as pd
from gml.feature import Analyzer

data_path = "xxx.csv"
df = pd.read_csv(data_path, sep="\t")

# 配置特征生成的方法和参数
GEN_FEAT_CONFIG = [
        {
             "name": "CrossFeatures",
            "params": {}
         },
       {
            "name": "KFoldPred",
            "params": {
               "model_type": "lgb",
                 "k": 5
            }
        }
    ]
# 配置特征过滤的方选和参数
SCREEN_FEAT_CONFIG = [
       {
            "name": "correlation",
             "params": {
                "method": "pearson",
               "thres": 0.9
             }
       },
       {
             "name": "null_importance",
             "params": {
                "model_type": "lgb",
                "num_models": 100,
                "save_fig_path": "./null_importance.png"
             }
       }
    ]

final_feat, useless_feats_dict, encoder2feat = Analyzer(
        GEN_FEAT_CONFIG,
        SCREEN_FEAT_CONFIG).start(df, dense_feat_names,
                                  sparse_feat_names, label,
                                  'reg',
                                  'xgb'
    )
</pre>
3. 调用训练及评估代码示例
<pre>
from gml.feature.encoders import *  # 导入编码器
from gml.models import XGBoost  # 导入模型
import pandas as pd
from gml.feature.mapping.common_mapping import get_box_plot
from gml.evaluation import evaluation, pred_actual
from gml.feature import distribution_kde
from sklearn.model_selection import train_test_split

data_path = "xxx.csv"
df = pd.read_csv(data_path, sep="\t")

dense_feat_names = ["height", "weight", ...]
sparse_feat_names = ["gender", ...]

# 利用箱线图去除标签异常样本
mask = get_box_plot(df[['gmv_label']])[0]
df = df[mask]

train, test = train_test_split(df, test_size=0.3)

# 编码器配置参数
encoder2feats = {
    BaseEncoder: dense_feat_names,  # 选择注续特征处方法
    OneHotEncoder: sparse_feat_names  # 选择高就特征处重方法
}
encoder = SuperSetEncoder(encoder2feats)

encoder.fit(train)

train_x = encoder.transform(train)
train_y = train[label]

test_x = encoder.transform(test)
test_y = test[label]

model = XGBoost(params, encoder.get_feat_names())
model.fit(train_x, train_y, val_x=test_x, val_y=test_y)

# 保存模型和编码器
model.save(f"checkponts/xgb.model")
encoder.save(f"checkponts/bo.encoder")

# 何线
# 基级何格
metrics = ["rmse"]
result = evaluation(test_x, test_y, model, metrics)
for name, score in result.items():
    print(name, score)
# 预测和标适的脸势
res = pred_actual(model.predict(test_x), test_y)
# 预测和标签的分布差录
distribution_kde(
        pd.DataFrame(model.predict(test_x), columns=['score']),
        compare_names=['pred', 'true']
    )
# 回归uplift
res, kendall_res = delta_pred_actual(test, treatments, control, encoder, model, label, 2)
# 分类uplift
get_auuc(train, treaments, control, encoder, model, label)

# 重要特性
for name, score in model.get_feature_importances():
    print(name, score)
</pre>
## 目录结体
目录
1. 特征工程
    - 特性筛选
    - 过滤法
        - [x] psi
        - [x] pearson等相关性方法
        - [x] topk选择
        - [x] KDE图
        - [x] 直方图
        - [x] Q-Q图
    - 包裹法
       - [x] null importance
    - 特征生成
      - [x] 多项式交叉特征
      - [x] k折模型stacking征
      - [x] 离散交叉特征
    - 特框映射
      - 归一化方法
        - [x] min-max
        - [x] std
      - 类线特征高政化方法
        - [x] kmeans
        - [x] 等频/距分箱法
      - 其他方法
        - [x] log变换
        - [x] 箱线图异常检测
    - 特征编码器
        - 连续特征编码器
          - [x] 连续值离散化
            - [x] 等频/距离散化
            - [x] 聚类离散化
          - [x] Cox-Box变换
          - [x] 归一化编码
        - 离散特征编码器
          - [x] one-hot编码
          - [x] label编码
2. 模型调优
    - 模型开发
      - tree类模型
        - [x] xgboost
        - [x] lightgbm
      - dnn类模型
    - 损失函数
3.模型评估
  - [x] auuc标
  - [x] ml通用操标
  - [x] pred-actual
4. 其他
   - [x] 降低数据精度压缩数据

