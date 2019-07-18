# coding: utf-8

import pandas as pd

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier


# SET OF FUNCTIONS TO CREATE CLASSIFICATION MODELS (Step 2) #

# Balance data set in case of imbalanced learning.
def balance_split(X, Y, sampling='UNDER', test_size=0.25):
    if sampling == 'OVER':
        X_s, Y_s = SMOTE(ratio='minority').fit_sample(X, Y)
    elif sampling == 'UNDER':
        X_s, Y_s = RandomUnderSampler(return_indices=False).fit_sample(X, Y)
    else:
        X_s, Y_s = X, Y
    return train_test_split(X_s, Y_s, test_size=test_size)


# Train classifier given the model desired.
def trained_model(model, X, Y):
    if model == 'NBAYES':
        return GaussianNB().fit(X, Y)
    elif model == 'ADAB':
        return AdaBoostClassifier(n_estimators=100).fit(X, Y)
    elif model == 'DTREE':
        return DecisionTreeClassifier(max_depth=100).fit(X, Y)
    elif model == 'KNN':
        return KNeighborsClassifier(n_neighbors=3).fit(X, Y)
    elif model == 'ANN':
        return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1000, 500, 50)).fit(X, Y)
    elif model == 'SVM':
        return SVC(kernel='rbf', gamma='scale').fit(X, Y)
    else:
        return LogisticRegression(multi_class='auto', solver='lbfgs', max_iter=1000).fit(X, Y)


# Computer performance metrics of the classifier.
def evaluate_model(model, X_train, X_test, Y_train, Y_test):
    Y_pred = model.predict(X_test)
    Y_pre = model.predict(X_train)
    acc = accuracy_score(Y_test, Y_pred, normalize=True), accuracy_score(Y_train, Y_pre, normalize=True)
    pr = precision_score(Y_test, Y_pred, average='macro'), precision_score(Y_train, Y_pre, average='macro')
    rec = recall_score(Y_test, Y_pred, average='macro'), recall_score(Y_train, Y_pre, average='macro')
    f1 = f1_score(Y_test, Y_pred, average='macro'), f1_score(Y_train, Y_pre, average='macro')
    perf = pd.DataFrame([acc, pr, rec, f1])
    perf.columns = ['Test', 'Train']
    perf.index = ['Accuracy', 'Precision', 'Recall', 'F1-score']
    return perf
