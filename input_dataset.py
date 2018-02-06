import numpy as np
import pandas as pd
from My_finding_donors_project import visuals as vs
from sklearn.preprocessing import MinMaxScaler

def get_data():
    data = pd.read_csv("census.csv")


    # 将数据切分成特征和对应的标签
    income_raw = data['income']
    features_raw = data.drop('income', axis=1)

    # 对于倾斜的数据使用Log转换
    skewed = ['capital-gain', 'capital-loss']
    features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

    # 可视化对数转换后 'capital-gain'和'capital-loss' 两个特征
    vs.distribution(features_raw, transformed=True)

    # 初始化一个 scaler，并将它施加到特征上
    scaler = MinMaxScaler()
    numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
    features_raw[numerical] = scaler.fit_transform(data[numerical])


    # TODO：使用pandas.get_dummies()对'features_raw'数据进行独热编码
    features = pd.get_dummies(features_raw)

    # TODO：将'income_raw'编码成数字值
    income = income_raw.map({'<=50K': 0, '>50K': 1})

    from sklearn.model_selection import train_test_split

    # 将'features'和'income'数据切分成训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(features, income, test_size=0.2, random_state=0,
                                                        stratify=income)
    # 将'X_train'和'y_train'进一步切分为训练集和验证集
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0,
                                                      stratify=y_train)

    accuracy = float(len(y_val[income == 1])) / len(y_val)

    # TODO： 计算查准率 Precision
    precision = float(len(y_val[income == 1])) / len(y_val)

    # TODO： 计算查全率 Recall
    recall = float(len(y_val[income == 1])) / len(y_val[income == 1])

    # TODO： 使用上面的公式，设置beta=0.5，计算F-score
    fscore = (1 + 0.5 ** 2) * precision * recall / (0.5 ** 2 * precision + recall)
    # 显示切分的结果
    print("Training set has {} samples.".format(X_train.shape[0]))
    print("Validation set has {} samples.".format(X_val.shape[0]))
    print("Testing set has {} samples.".format(X_test.shape[0]))

    return X_train, X_val, X_test, y_train, y_val, y_test, accuracy, fscore
