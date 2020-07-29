import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn_pandas import DataFrameMapper
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import sys
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.feature_selection import f_classif
from feature_selector import FeatureSelector
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE  # doctest: +NORMALIZE_WHITESPACE
from collections import Counter
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import imblearn
from imblearn.under_sampling import RandomUnderSampler



train = pd.read_csv(sys.argv[1])
test1 = pd.read_csv(sys.argv[2])

test = test1.drop(['index'], axis=1)
labels = train['TARGET']
train = train.drop(['TARGET'], axis=1)
col = train.columns
print(train.columns)
print(test.columns)
print(train.head())
ros = RandomUnderSampler()
train, labels = ros.fit_sample(train, labels)
train = pd.DataFrame(train, columns=col)
labels = pd.DataFrame(labels, columns=['TARGET'])

X_train, X_test, y_train, y_test = train_test_split(
    train, labels, test_size=0.1, stratify=labels)

category_cols = []
continuous_cols = []
for a in col:
    if len(train[a].value_counts()) < 3:
        category_cols.append(a)
        # print(train[a].value_counts())
    else:
        continuous_cols.append(a)

scaler = RobustScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(X_train[continuous_cols]), columns=continuous_cols)
X_test = pd.DataFrame(
    scaler.transform(X_test[continuous_cols]), columns=continuous_cols)
test = pd.DataFrame(
    scaler.transform(test[continuous_cols]), columns=continuous_cols)

fs = FeatureSelector(data=X_train, labels=y_train)
fs.identify_single_unique()
single_unique = fs.ops['single_unique']

fs.identify_collinear(correlation_threshold=0.95)
correlated_features = fs.ops['collinear']

fs.identify_zero_importance(
    task='classification',
    eval_metric='auc',
    n_iterations=10,
    early_stopping=True)
zero_importance_features = fs.ops['zero_importance']

fs.identify_low_importance(cumulative_importance=0.99)
low_importance_features = fs.ops['low_importance']

X_train = fs.remove(methods='all', keep_one_hot=False)

X_test = X_test.drop(columns=fs.removed_features)
test = test.drop(columns=fs.removed_features)

		
clf1 = RandomForestClassifier(n_estimators=8000, max_depth=8)
clf1.fit(X_train, np.ravel(y_train))
pred = clf1.predict(X_test)
score = roc_auc_score(y_test, pred)
#print(est, md, score)
final = clf1.predict(test)
		

final = pd.Series(final)
answer = pd.concat([test1['index'], final], axis=1)
answer.columns = ['index', 'TARGET']
answer.to_csv("submission.csv", index=False)
