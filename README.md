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
- 特殊：osv→正则表达式匹配出正确的x.x.x版本

构造用户特征（groupby）：

- count，unique，top，freq，min，mean，max，25%，50%，75%

classfication：

- lightgmb，10折fold



## 更新2：

对于特征工程和预处理做了不一样的改进

用rf预测了ppi，对于dev的信息做了两个新的特征工程

回到了前10的榜首

## submission_记录

06090206：89.054

```python
model = lgb.LGBMClassifier(boosting_type='gbdt',num_leaves=100,max_depth=6,learning_rate=0.02,n_estimators=10000,subsample=0.6,feature_fraction=0.4,reg_alpha=0.8,reg_lambda=0.8,random_state=220,metric=None,n_jobs=-1,save_binary=True,
                        #    max_bin=255
                        #   cat_smooth=30
                        #    colsample_bytree=0.8
                        #    metric=["binary_error"]
                        #    max_bin=10,
                        #    device="gpu"
                           )
```

[1410]	train's binary_logloss: 0.263427	train's binary_error: 0.100902	train's auc: 0.957133	valid's binary_logloss: 0.289595	==valid's binary_error: 0.10818==	valid's auc: 0.944363

[1478]	train's binary_logloss: 0.261809	train's binary_error: 0.10064	train's auc: 0.957748	valid's binary_logloss: 0.291009	==valid's binary_error: 0.11042==	valid's auc: 0.944567

[2180]	train's binary_logloss: 0.246513	train's binary_error: 0.0953089	train's auc: 0.963621	valid's binary_logloss: 0.292026	==valid's binary_error: 0.10954==	valid's auc: 0.943822

06090223:

```python
model = lgb.LGBMClassifier(boosting_type='gbdt',num_leaves=100,max_depth=6,learning_rate=0.02,n_estimators=10000,subsample=0.6,feature_fraction=0.5,reg_alpha=1.2,reg_lambda=1.2,random_state=220,metric=None,n_jobs=-1,save_binary=True,cat_smooth=30
# ,max_bin=255
                        #    colsample_bytree=0.8
                        #    metric=["binary_error"]
                        #    max_bin=10,
                        #    device="gpu"
                           )
```

[1539]	train's auc: 0.958638	train's binary_logloss: 0.260266	train's binary_error: 0.100158	valid's auc: 0.944116	valid's binary_logloss: 0.289976	==valid's binary_error: 0.10888==

[1600]	train's auc: 0.959285	train's binary_logloss: 0.258401	train's binary_error: 0.0995867	valid's auc: 0.943531	valid's binary_logloss: 0.293063  ==valid's binary_error: 0.11024==

[1000]	train's auc: 0.961707	train's binary_logloss: 0.252705	train's binary_error: 0.0975089	valid's auc: 0.943773	valid's binary_logloss: 0.292188	valid's binary_error: 0.11026