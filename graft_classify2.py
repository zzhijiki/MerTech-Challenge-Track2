import pandas as pd
from sklearn import preprocessing
import numpy as np
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from catboost import CatBoostClassifier
import warnings

from test2_code.process import process_category, read, dict_category, special_category, process_osv, rf_cast, feature

warnings.filterwarnings("ignore")


def train(train_path="../processed_data/df_train3.pkl", test_path="../processed_data/df_test3.pkl", graft_path=None):
    print(train_path, test_path, graft_path)
    df_train = pd.read_pickle(train_path)
    df_test = pd.read_pickle(test_path)
    if graft_path:
        print("开始嫁接!")
        df_graft = pd.read_pickle(graft_path)
        df_train = df_train.append(df_graft)
        # df_train = pd.concat([df_train, df_graft])
        print(df_train.shape)
        del df_graft
    for col in ["dev_height", "dev_width", "hw_ratio", "hw_matrix", "inch", "lan"]:
        if col in df_train.columns:
            df_train[col] = df_train[col].astype("float64")
            df_test[col] = df_test[col].astype("float64")
    cate_feature = ['apptype', 'carrier', 'media_id', 'os', 'osv', 'package', 'version', 'location', 'cus_type',
                    "fea1_hash", "fea_hash", "ntt", "os", 'fea1_hash_ntt_combine', 'fea_hash_carrier_combine',
                    'cus_type_osv_combine', 'fea1_hash_apptype_combine', 'fea_hash_media_id_combine',
                    'cus_type_version_combine', 'apptype_ntt_combine', 'media_id_carrier_combine',
                    'version_osv_combine', 'package_lan_combine', 'lan']

    # x_col=df_importance.head(27).column

    y_col = 'label'
    x_col = ['apptype', 'carrier', 'dev_height',
             'dev_width', 'lan', 'media_id', 'ntt', 'osv', 'package',
             'timestamp', 'version', 'fea_hash', 'location', 'fea1_hash', 'cus_type',
             'hour', 'minute',
             '160_height',
             'hw_ratio', 'hw_matrix', 'inch']
    cate_feature = [x for x in cate_feature if x in x_col]
    for item in cate_feature:
        print(item)
        if item in ['fea1_hash_ntt_combine', 'fea_hash_carrier_combine', 'cus_type_osv_combine',
                    'fea1_hash_apptype_combine', 'fea_hash_media_id_combine', 'cus_type_version_combine',
                    'apptype_ntt_combine', 'media_id_carrier_combine', 'version_osv_combine', 'package_lan_combine',
                    ]:
            set4 = set(df_train[item].value_counts().head(300).index)

            def process_fea_hash(x):
                if x in set4:
                    return x
                else:
                    return -1

            df_train[item] = df_train[item].apply(process_fea_hash).astype("str")
            df_test[item] = df_test[item].apply(process_fea_hash).astype("str")
        le = preprocessing.LabelEncoder()
        df_train[item] = le.fit_transform(df_train[item])
        df_test[item] = le.transform(df_test[item])

    df_prediction = df_test[x_col]
    df_prediction['label'] = 0
    model2 = CatBoostClassifier(loss_function="Logloss",
                                eval_metric="Accuracy",
                                task_type="GPU",
                                learning_rate=0.03,
                                iterations=10000,
                                random_seed=42,
                                od_type="Iter",
                                metric_period=10,
                                depth=10,
                                early_stopping_rounds=500,
                                use_best_model=True,
                                bagging_temperature=0.7,
                                leaf_estimation_method="Newton",
                                )

    oof = []
    df_importance_list = []
    n = 10
    # kfold = GroupKFold(n_splits=n)
    weight = [0.1, 0.11, 0.1, 0.11, 0.11, 0.11, 0.05, 0.11, 0.1, 0.1]
    assert sum(weight) == 1 and len(weight) == n
    kfold = KFold(n_splits=n, shuffle=True, random_state=220)
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(df_train[x_col], df_train[y_col])):
        X_train = df_train.iloc[trn_idx][x_col]
        Y_train = df_train.iloc[trn_idx][y_col]

        X_val = df_train.iloc[val_idx][x_col]
        Y_val = df_train.iloc[val_idx][y_col]
        # print(X_train.shape, Y_train.shape, X_val.shape, Y_val.shape)
        print('\nFold_{} Training ================================\n'.format(fold_id + 1))
        cat_model = model2.fit(
            X_train,
            Y_train,
            cat_features=cate_feature,
            # # eval_names=['train', 'valid'],
            eval_set=(X_val, Y_val),
            verbose=100,

            # plot=True
            # eval_metric=["auc","binary_logloss","binary_error"],
            # early_stopping_rounds=400
        )

        pred_val = cat_model.predict_proba(X_val, thread_count=-1)[:, 1]
        df_oof = df_train.iloc[val_idx].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)

        pred_test = cat_model.predict_proba(df_test[x_col], thread_count=-1)[:, 1]
        df_prediction['label'] += weight[fold_id] * pred_test
        # prediction['label'] += pred_test

        df_importance = pd.DataFrame({
            'column': x_col,
            'importance': cat_model.feature_importances_,
        })
        df_importance_list.append(df_importance)
    return df_prediction, oof, df_train, df_test


def graft(valid_path='../test1.csv'):
    # read
    df1, df2 = read(test_path=valid_path)
    for col in ["location", "os", "ntt", "cus_type"]:
        df1, df2 = process_category(df1, df2, col)
    for col, dict1 in zip(["carrier"], [{0.0: 0, 46000.0: 1, 46001.0: 2, 46003.0: 3, -1.0: -1}]):
        df1, df2 = dict_category(df1, df2, col, dict1)
    for col in ["apptype", "media_id", "version", "lan", "package", "fea1_hash", "fea_hash"]:
        df1, df2 = special_category(df1, df2, col)
    df1, df2 = process_osv(df1, df2)
    df1, df2 = rf_cast(df1, df2)
    df1, df2 = feature(df1, df2)
    df1.to_pickle("../processed_data/graft_train.pkl")
    df2.to_pickle("../processed_data/graft_test.pkl")
    # train
    df_prediction, oof, feature_train, feature_test = train(train_path="../processed_data/graft_train.pkl",
                                                            test_path="../processed_data/graft_test.pkl")
    # graft
    feature_train = df1
    feature_test = df2
    df_copy = df_prediction[["label"]]

    def fun1(x):
        if x > 0.9:
            return 1
        elif x < 0.1:
            return 0
        else:
            return x

    df_copy.label = df_copy.label.apply(fun1)
    df_copy = df_copy[(df_copy.label == 1) | (df_copy.label == 0)]
    df_graft = feature_test.loc[df_copy.index]
    df_graft.label = df_copy.label.values
    df_graft.to_pickle("../processed_data/graft_for_train.pkl")
    print("保存了graft！")
    # feature_train=feature_train.append(df_graft)
    # feature_train.to_pickle("../processed_data/graft_for_train2.pkl")
    return df_graft


def save(file_path, pred, df1, df2, threshold=0.5):
    a = pd.DataFrame(pred.index)
    a['label'] = pred["label"].values

    a.label = a.label.apply(lambda x: 1 if x > threshold else 0)
    user_label = pd.DataFrame()

    user_label["uid"] = df1.android_id.values
    user_label["ntt"] = df1.ntt.values
    temp = pd.DataFrame(df1.groupby(["android_id", "ntt"]).label.mean())
    temp = temp.reset_index()
    temp.rename(columns={"android_id": "uid", "label": "label_prior"}, inplace=True)
    user_label = pd.merge(user_label, temp, on=["uid", "ntt"], how="left")
    user_label.drop_duplicates(inplace=True)
    a["uid"] = df2.android_id.values
    a["ntt"] = df2.ntt.values
    a = pd.merge(a, user_label, how="left", on=["uid", "ntt"])

    def post(label, prior):
        n = len(label)
        count = 0
        for i in range(n):
            if 0 <= prior[i] <= 0.1 and label[i] == 1:
                label[i] = 0
                count += 1
                # print(i)
            elif 0.9 <= prior[i] <= 1 and label[i] == 0:
                label[i] = 1
                count += 1
                # print(i)
            else:
                pass
        print(count)
        return label.values

    a.label = post(a.label, a.label_prior)
    a = a[["sid", "label"]]
    a.to_csv(file_path, index=False)
    return a


def search_threshold(df):
    ans_threshold = 0
    top_score = 0
    for threshold in np.arange(0.3, 0.7, 0.01):
        df_copy = df.copy()
        a = df_copy.label.values
        b = df_copy.pred.apply(lambda x: 1 if x > threshold else 0).values
        score = np.sum(a == b) / len(a)
        if score > top_score:
            top_score = score
            ans_threshold = threshold
        print(score, threshold)
    return top_score, ans_threshold


if __name__ == "__main__":
    filename = './submission_0713_1550.csv'
    # prediction, oof, feature_train, feature_test = train()
    df_graft = graft()
    prediction, oof, feature_train, feature_test = train(graft_path="../processed_data/graft_for_train.pkl")
    df_oof = pd.concat(oof)
    top, ans = search_threshold(df_oof)
    save(filename, prediction, feature_train, feature_test, threshold=ans)
    print("结束了！,graft-classify")  # 89.2533
