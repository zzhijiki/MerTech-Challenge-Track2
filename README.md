# README

MarTech Challenge Track 2 点击反欺诈预测

https://aistudio.baidu.com/aistudio/competition/detail/22

提供约50万次点击数据，请预测用户的点击行为是否为正常点击，还是作弊行为。

标签：反欺诈预测比赛时间：2020/05/10 - 2020/07/05

## 结果：

排名：12/424（2020-06-18）

正确率：89.11%

## process&EDA.ipynb

process：

- 对数据类型进行合理的转换，由于数据占内存不大，所以没有做内存优化（uint的转换）

EDA：

- 缺失值的展示（lan）
- 在后面的EDA中得知，package=0，android=0，dev_xx=0的值都是缺失，用rf做了缺失值的预测。
- 描述统计的展示：.
  - object：count，unique，top，freq。
  - numerical：min，mean，max，25%，50%，75%
- 分布展示：media_id，location，label，fea_hash，fea1_hash
- 箱线图：
- 相关及其显著性检验，包括相关矩阵
- 各列与label关系的数据透析

## final_feature.ipynb

对原始数据进行预处理：

- object：选取topk，label_encoder
- 特殊：osv→正则表达式匹配出正确的x.x.x版本

构造统计特征（groupby）：

- count，top，freq

classfication：

- lightgmb，10折fold

对于特征工程和预处理做了不一样的改进

用rf预测了ppi，package等缺失信息，对于dev的信息做了两个新的特征工程

对类别特征进行了交叉特征信息，使用catboost进行预测，对类别信息，catboost的处理更好。

使用android和ntt信息找test和train中的相似信息，直接transform一部分label，相当于使用leak的信息，对于结果有不错的提升。

## 展望：

对于leak，找到更多train和test相近的信息，对于答案直接transform，对于预测剔除这些数据后重新做训练集

做stacking和后续特征选择的处理



