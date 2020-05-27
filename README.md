# README

MarTech Challenge Track 2 点击反欺诈预测

https://aistudio.baidu.com/aistudio/competition/detail/22

提供约50万次点击数据，请预测用户的点击行为是否为正常点击，还是作弊行为。

标签：反欺诈预测比赛时间：2020/05/10 - 2020/07/05



## 概要：

这里主要进行了EDA和预处理，并进行了初步的feature_engineer，初步的分类

后续还要继续修改，使得正确率进入90%。

## process&EDA.ipynb

process：

- 对数据类型进行合理的转换，由于数据占内存不大，所以没有做内存优化（uint的转换）

EDA：

- 缺失值的展示（lan）

- 描述统计的展示：.
  - object：count，unique，top，freq。
  - numerical：min，mean，max，25%，50%，75%
- 分布展示：media_id，location，label，fea_hash，fea1_hash

- 箱线图：
- 相关及其显著性检验，包括相关矩阵
- 各列与label关系的数据透析

## feature.ipynb

对原始数据进行预处理：

- object：选取topk，label_encoder
- 特殊：osv→正则匹配第一个数字

构造用户特征（groupby）：

- count，unique，top，freq，min，mean，max，25%，50%，75%

classfication：

- lightgmb，5折fold