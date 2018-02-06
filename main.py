#-*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from time import time
from IPython.display import display # 允许为DataFrame使用display()

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import fbeta_score, accuracy_score

# 导入人口普查数据
data = pd.read_csv("census.csv")

# 将数据切分成特征和对应的标签
income_raw = data['income']
features_raw = data.drop('income', axis = 1)

# 对于倾斜的数据使用Log转换
skewed = ['capital-gain', 'capital-loss']
features_raw[skewed] = data[skewed].apply(lambda x: np.log(x + 1))

# 初始化一个 scaler，并将它施加到特征上
scaler = MinMaxScaler()
numerical = ['age', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
features_raw[numerical] = scaler.fit_transform(data[numerical])

# 使用pandas.get_dummies()对'features_raw'数据进行独热编码
features = pd.get_dummies(features_raw)

# 将'income_raw'编码成数字值
income = income_raw.map({'<=50K':0,'>50K':1})

# 将'features'和'income'数据切分成训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, income, test_size = 0.2, random_state = 0,
                                                    stratify = income)
# 将'X_train'和'y_train'进一步切分为训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0,
                                                    stratify = y_train)

# 显示切分的结果
print("Training set has {} samples.".format(X_train.shape[0]))
print("Validation set has {} samples.".format(X_val.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

# 计算准确率
accuracy = float(len(y_val[income==1]))/len(y_val)

# 计算查准率 Precision
precision = float(len(y_val[income==1]))/len(y_val)

# 计算查全率 Recall
recall = float(len(y_val[income==1]))/len(y_val[income==1])

# 使用上面的公式，设置beta=0.5，计算F-score
fscore = (1+0.5**2)*precision*recall/(0.5**2*precision+recall)

print("Naive Predictor on validation data: \n \
    Accuracy score: {:.4f} \n \
    Precision: {:.4f} \n \
    Recall: {:.4f} \n \
    F-score: {:.4f}".format(accuracy, precision, recall, fscore))

def train_predict(learner, sample_size, X_train, y_train, X_val, y_val):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_val: features validation set
       - y_val: income validation set
    '''

    results = {}

    # Fit the learner to the training data using slicing with 'sample_size'
    start = time()  # 获得程序开始时间
    learner = learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()  # 获得程序结束时间
    results['train_time'] = end - start  # 计算训练时间

    # 得到在验证集上的预测值
    start = time()  # 获得程序开始时间
    predictions_val = learner.predict(X_val)
    predictions_train = learner.predict(X_train[:300])
    end = time()  # 获得程序结束时间

    # 计算预测用时
    results['pred_time'] = end - start

    # 计算在最前面的300个训练数据的准确率
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

    # 计算在验证上的准确率
    results['acc_val'] = accuracy_score(y_val, predictions_val)

    # 计算在最前面300个训练数据上的F-score
    results['f_train'] = fbeta_score(y_train[:300], predictions_train, beta=0.5)

    # 计算验证集上的F-score
    results['f_val'] = fbeta_score(y_val, predictions_val, beta=0.5)

    # 成功
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    return results

from sklearn import svm, tree, ensemble

# 初始化
clf_A = svm.SVC(random_state=0)
clf_B = tree.DecisionTreeClassifier(max_depth=8, random_state=0)
clf_C = ensemble.AdaBoostClassifier(random_state=0)

# 计算1%， 10%， 100%的训练数据分别对应多少点
samples_1 = int(0.01 * len(X_train))
samples_10 = int(0.1 * len(X_train))
samples_100 = len(X_train)

results = {}
for clf in [clf_A, clf_B, clf_C]:
    clf_name = clf.__class__.__name__
    results[clf_name] = {}
    for i, samples in enumerate([samples_1, samples_10, samples_100]):
        results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_val, y_val)

# 对选择的三个模型得到的评价结果进行可视化

# 模型调优
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer,fbeta_score
from sklearn import ensemble
# 初始化分类器
clf = ensemble.AdaBoostClassifier(random_state=0)

# 创建调节参数列表
parameters ={ 'n_estimators':[100,500,1000], 'learning_rate':[0.1,0.5,1.0]}

# 创建一个fbeta_score打分对象
scorer = make_scorer(fbeta_score,beta=0.5)

# 在分类器上使用网格搜索，使用'scorer'作为评价函数
grid_obj = GridSearchCV(clf,parameters,scoring=scorer)

# 用训练数据拟合网格搜索对象并找到最佳参数
grid_obj=grid_obj.fit(X_train,y_train)
# 得到estimator
best_clf = grid_obj.best_estimator_

# 使用没有调优的模型做预测
predictions = (clf.fit(X_train, y_train)).predict(X_val)
best_predictions = best_clf.predict(X_val)

# 汇报调参前和调参后的分数
print("Unoptimized model\n------")
print("Accuracy score on validation data: {:.4f}".format(accuracy_score(y_val, predictions)))
print("F-score on validation data: {:.4f}".format(fbeta_score(y_val, predictions, beta = 0.5)))
print("\nOptimized Model\n------")
print("Final accuracy score on the validation data: {:.4f}".format(accuracy_score(y_val, best_predictions)))
print("Final F-score on the validation data: {:.4f}".format(fbeta_score(y_val, best_predictions, beta = 0.5)))


accuracy_final=accuracy_score(y_test, best_clf.predict(X_test))
F_score_final=fbeta_score(y_test, best_clf.predict(X_test), beta = 0.5)
print(accuracy_final,F_score_final)