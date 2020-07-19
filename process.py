import lightgbm as lgb
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import gc
import re
import warnings

warnings.filterwarnings("ignore")


def read(train_path='../train.csv', test_path='../test2.csv'):
    train = pd.read_csv(train_path)
    test1 = pd.read_csv(test_path)
    # labels = train['label']
    df1 = train.drop(['Unnamed: 0'], axis=1)
    df2 = test1.drop(['Unnamed: 0'], axis=1)

    df2["label"] = -1

    for col in ["android_id", "apptype", "carrier", "ntt", "media_id", "cus_type", "package", 'fea1_hash', "location"]:
        df1[col] = df1[col].astype("object")
        df2[col] = df2[col].astype("object")

    for col in ["fea_hash"]:
        df1[col] = df1[col].map(lambda x: 0 if len(str(x)) > 16 else int(x))
        df2[col] = df2[col].map(lambda x: 0 if len(str(x)) > 16 else int(x))

    for col in ["dev_height", "dev_ppi", "dev_width", "fea_hash", "label"]:
        df1[col] = df1[col].astype("int64")
        df2[col] = df2[col].astype("int64")

    df2["label"] = None

    df1["truetime"] = pd.to_datetime(df1['timestamp'], unit='ms', origin=pd.Timestamp('1970-01-01'))
    df2["truetime"] = pd.to_datetime(df2['timestamp'], unit='ms', origin=pd.Timestamp('1970-01-01'))

    df1["day"] = df1.truetime.dt.day
    df2["day"] = df2.truetime.dt.day

    df1["hour"] = df1.truetime.dt.hour
    df2["hour"] = df2.truetime.dt.hour

    df1["minute"] = df1.truetime.dt.minute
    df2["minute"] = df2.truetime.dt.minute

    df1.set_index("sid", drop=True, inplace=True)
    df2.set_index("sid", drop=True, inplace=True)

    df1.dev_height[df1.dev_height == 0] = None
    df1.dev_width[df1.dev_width == 0] = None
    df1.dev_ppi[df1.dev_ppi == 0] = None
    df2.dev_height[df2.dev_height == 0] = None
    df2.dev_width[df2.dev_width == 0] = None
    df2.dev_ppi[df2.dev_ppi == 0] = None

    return df1, df2


def process_category(df1, df2, col):
    print("{}_cate start!".format(col))

    le = preprocessing.LabelEncoder()
    df1[col] = le.fit_transform(df1[col])
    df1[col] = df1[col].astype("object")

    df2[col] = le.transform(df2[col])
    df2[col] = df2[col].astype("object")

    return df1, df2


def dict_category(df1, df2, col, dict1):
    print("{}_dict start!".format(col))
    print(col, dict1)
    df1[col] = df1[col].map(dict1)
    df1[col] = df1[col].astype("object")
    df2[col] = df2[col].map(dict1)
    df2[col] = df2[col].astype("object")
    return df1, df2


def filter_value(df1, df2, col, top, other=-1):
    set1 = set(df1[col].value_counts().head(top).index)

    def process_temp(x):
        if x in set1:
            return x
        else:
            return other

    df1[col] = df1[col].apply(process_temp)
    df2[col] = df2[col].apply(process_temp)
    return df1, df2


def special_category(df1, df2, col):
    print("{} start!".format(col))
    if col == "apptype":
        df1, df2 = filter_value(df1, df2, col, 75, -1)

    if col == "media_id":
        df1, df2 = filter_value(df1, df2, col, 200, -1)

    if col == "version":
        df2[col] = df2[col].replace("20", "0").replace("21", "0")

    if col == "lan":
        def foreign_lan(x):
            set23 = {'zh-CN', 'zh', 'cn', 'zh_CN', 'Zh-CN', 'zh-cn', 'ZH', 'CN', 'zh_CN_#Hans'}
            if x in set23:
                return 0
            elif x == "unk":
                return 2
            else:
                return 1

        df1["vpn"] = df1["lan"].apply(foreign_lan)
        df2["vpn"] = df2["lan"].apply(foreign_lan)

        set12 = {'zh-CN', 'zh', 'cn', 'zh_CN', 'Zh-CN', 'zh-cn', 'ZH', 'CN', 'tw', 'en', 'zh_CN_#Hans', 'ko'}

        def process_lan(x):
            if x in set12:
                return x
            else:
                return "unk"

        df1[col] = df1[col].apply(process_lan)
        df2[col] = df2[col].apply(process_lan)

    if col == "package":
        df1, df2 = filter_value(df1, df2, col, 800, -1)

    if col == "fea1_hash":
        df1, df2 = filter_value(df1, df2, col, 850, -1)

    if col == "fea_hash":
        df1, df2 = filter_value(df1, df2, col, 850, -1)

    # cate
    df1, df2 = process_category(df1, df2, col)
    print("cate end")
    print("--------------")
    return df1, df2


def feature(df1, df2):
    def divided(x):
        if x % 40 == 0:
            return 2
        elif not x:
            return 1
        else:
            return 0

    df1["160_height"] = df1.dev_height.apply(divided)
    df2["160_height"] = df2.dev_height.apply(divided)
    df1["160_width"] = df1.dev_width.apply(divided)
    df2["160_width"] = df2.dev_width.apply(divided)
    df1["160_ppi"] = df1.final_ppi.apply(divided)
    df2["160_ppi"] = df2.final_ppi.apply(divided)
    df1["hw_ratio"] = df1.dev_height / df1.dev_width
    df2["hw_ratio"] = df2.dev_height / df2.dev_width
    df1["hw_matrix"] = df1.dev_height * df1.dev_width
    df2["hw_matrix"] = df2.dev_height * df2.dev_width
    df1["inch"] = (df1.dev_height ** 2 + df1.dev_width ** 2) ** 0.5 / df1.final_ppi
    df2["inch"] = (df2.dev_height ** 2 + df2.dev_width ** 2) ** 0.5 / df2.final_ppi
    return df1, df2


def rf_cast(df1, df2):
    c1 = df1.dev_width.notnull()
    c2 = df1.dev_height.notnull()
    c3 = df1.dev_ppi.isna()
    c4 = df1.dev_ppi.notnull()
    df1["mynull1"] = c1 & c2 & c3
    df1["mynull2"] = c1 & c2 & c4

    predict = df1[
        ["apptype", "carrier", "dev_height", "dev_ppi", "dev_width", "media_id", "ntt", "mynull1", "mynull2"]]

    df_notnans = predict[predict.mynull2 == True]

    # Split into 75% train and 25% test
    X_train, X_test, y_train, y_test = train_test_split(
        df_notnans[["apptype", "carrier", "dev_height", "dev_width", "media_id", "ntt"]], df_notnans["dev_ppi"],
        train_size=0.75, random_state=6)

    regr_multirf = RandomForestClassifier(n_estimators=100, max_depth=40, random_state=0, n_jobs=-1)

    # Fit on the train data
    regr_multirf.fit(X_train, y_train)

    # Check the prediction score
    score = regr_multirf.score(X_test, y_test)
    print("The prediction score on the test data is {:.2f}%".format(score * 100))
    df_nans = predict[predict.mynull1 == True].copy()
    df_nans["dev_ppi_pred"] = regr_multirf.predict(
        df_nans[["apptype", "carrier", "dev_height", "dev_width", "media_id", "ntt"]])
    df1 = pd.merge(df1, df_nans[["dev_ppi_pred"]], on="sid", how="left")
    c1 = df2.dev_width.notnull()
    c2 = df2.dev_height.notnull()
    c3 = df2.dev_ppi.isna()
    c4 = df2.dev_ppi.notnull()
    df2["mynull1"] = c1 & c2 & c3
    df2["mynull2"] = c1 & c2 & c4

    predict_test = df2[
        ["apptype", "carrier", "dev_height", "dev_ppi", "dev_width", "media_id", "ntt", "mynull1", "mynull2"]]
    df_nans = predict_test[predict_test.mynull1 == True].copy()
    df_nans["dev_ppi_pred"] = regr_multirf.predict(
        df_nans[["apptype", "carrier", "dev_height", "dev_width", "media_id", "ntt"]])
    df2 = pd.merge(df2, df_nans[["dev_ppi_pred"]], on="sid", how="left")

    def recol_ppi(df):
        a = df.dev_ppi.fillna(0).values
        b = df.dev_ppi_pred.fillna(0).values
        c = []
        # print(a,b)
        for i in range(len(a)):
            c.append(max(a[i], b[i]))
        c = np.array(c)
        df["final_ppi"] = c
        df["final_ppi"][df["final_ppi"] == 0] = None
        return df

    df1 = recol_ppi(df1)
    df2 = recol_ppi(df2)
    gc.collect()
    return df1, df2


def process_osv(df1, df2):
    def process_osv1(x):
        x = str(x)
        if not x:
            return -1
        elif x.startswith("Android"):
            x = str(re.findall("\d{1}\.*\d*\.*\d*", x)[0])
            return x
        elif x.isdigit():
            return x
        else:
            try:
                x = str(re.findall("\d{1}\.\d\.*\d*", x)[0])
                return x
            except:
                return 0

    df1.osv = df1.osv.apply(process_osv1)
    df2.osv = df2.osv.apply(process_osv1)
    set3 = set(df1["osv"].value_counts().head(70).index)

    def process_osv2(x):
        if x in set3:
            return x
        else:
            return 0

    df1["osv"] = df1["osv"].apply(process_osv2)
    df2["osv"] = df2["osv"].apply(process_osv2)

    le8 = preprocessing.LabelEncoder()
    df1.osv = le8.fit_transform(df1.osv.astype("str"))
    df1["osv"] = df1["osv"].astype("object")

    df2.osv = le8.transform(df2.osv.astype("str"))
    df2["osv"] = df2["osv"].astype("object")
    return df1, df2


if __name__ == "__main__":
    df1, df2 = read()

    for col in ["location", "os", "ntt", "cus_type"]:
        df1, df2 = process_category(df1, df2, col)

    for col, dict1 in zip(["carrier"], [{0.0: 0, 46000.0: 1, 46001.0: 2, 46003.0: 3, -1.0: -1}]):
        df1, df2 = dict_category(df1, df2, col, dict1)

    for col in ["apptype", "media_id", "version", "lan", "package", "fea1_hash", "fea_hash"]:
        df1, df2 = special_category(df1, df2, col)
    df1, df2 = process_osv(df1, df2)
    df1, df2 = rf_cast(df1, df2)
    df1, df2 = feature(df1, df2)
    df1.to_pickle("../processed_data/df_train3.pkl")
    df2.to_pickle("../processed_data/df_test3.pkl")
    print("保存完毕！")
