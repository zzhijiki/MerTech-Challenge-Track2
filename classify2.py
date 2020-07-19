import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from catboost import CatBoostClassifier
import warnings

warnings.filterwarnings("ignore")


def train(train_path="../processed_data/df_train3.pkl", test_path="../processed_data/df_test3.pkl"):
    feature_train = pd.read_pickle(train_path)
    feature_test = pd.read_pickle(test_path)
    for col in ["dev_height", "dev_width", "hw_ratio", "hw_matrix", "inch", "lan"]:
        if col in feature_train.columns:
            feature_train[col] = feature_train[col].astype("float64")
            feature_test[col] = feature_test[col].astype("float64")
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
                    'apptype_ntt_combine', 'media_id_carrier_combine', 'version_osv_combine', 'package_lan_combine']:
            set4 = set(feature_train[item].value_counts().head(300).index)

            def process_fea_hash(x):
                if x in set4:
                    return x
                else:
                    return -1

            feature_train[item] = feature_train[item].apply(process_fea_hash).astype("str")
            feature_test[item] = feature_test[item].apply(process_fea_hash).astype("str")
        le = preprocessing.LabelEncoder()
        feature_train[item] = le.fit_transform(feature_train[item])
        feature_test[item] = le.transform(feature_test[item])

    df_prediction = feature_test[x_col]
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
    kfold = KFold(n_splits=n, shuffle=True, random_state=220)
    # weight = [0.1, 0.11, 0.1, 0.11, 0.11, 0.11, 0.05, 0.11, 0.1, 0.1]
    # assert sum(weight) == 1 and len(weight) == n
    for fold_id, (trn_idx, val_idx) in enumerate(kfold.split(feature_train[x_col], feature_train[y_col])):
        X_train = feature_train.iloc[trn_idx][x_col]
        Y_train = feature_train.iloc[trn_idx][y_col]

        X_val = feature_train.iloc[val_idx][x_col]
        Y_val = feature_train.iloc[val_idx][y_col]

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
        df_oof = feature_train.iloc[val_idx].copy()
        df_oof['pred'] = pred_val
        oof.append(df_oof)

        pred_test = cat_model.predict_proba(feature_test[x_col], thread_count=-1)[:, 1]
        df_prediction['label'] += pred_test / n
        # prediction['label'] += pred_test

        df_importance = pd.DataFrame({
            'column': x_col,
            'importance': cat_model.feature_importances_,
        })
        df_importance_list.append(df_importance)
    return df_prediction, oof, feature_train, feature_test


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


if __name__ == "__main__":
    filename = './submission_0710_0350.csv'
    df_prediction, oof, feature_train, feature_test = train()
    oof = pd.concat(oof)
    oof.to_pickle("./oof.pkl")
    save(filename, df_prediction, feature_train, feature_test)
    print("????,classify2")  # 89.2393%
