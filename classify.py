#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords
from gensim.models import LdaMulticore, LsiModel, HdpModel, LdaSeqModel, TfidfModel, word2vec
from gensim.matutils import corpus2dense
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.wrappers.dtmmodel import DtmModel

from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler, SMOTE, ADASYN
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

from utils import no_paren, normalize, listate, drop_journa, my_remove_stopwords, my_stemmer
from sklearn.datasets import fetch_20newsgroups


def load_embeddings(project, embedding):
    pass


def load_labels(label_path):
    pass


def balance_split(X, Y, sampling='under', test_size=0.25):
    if sampling == 'over':
        X_s, Y_s = SMOTE(ratio='minority').fit_sample(X, Y)
    elif sampling == 'under':
        X_s, Y_s = RandomUnderSampler(return_indices=False).fit_sample(X, Y)
    else:
        X_s, Y_s = X, Y
        return train_test_split(X_s, Y_s, test_size=test_size)


def trained_model(model, X, Y):
    if model == 'logit':
        print(">>> Logit")
        return LogisticRegression(solver='lbfgs', max_iter=1000).fit(X_train, Y_train)
    if model == 'NB':
        print(">>> Naive Bayes")
        return GaussianNB().fit(X_train, Y_train)
    if model == 'adab':
        print(">>> AdaBoost")
        return AdaBoostClassifier(n_estimators=100).fit(X_train, Y_train)
    if model == 'DT':
        print(">>> Decision Tree")
        return DecisionTreeClassifier(max_depth=100).fit(X_train, Y_train)
    if model == 'KNN':
        print(">>> KNN")
        return KNeighborsClassifier(n_neighbors=3).fit(X_train, Y_train)
    if model == 'ann':
        print(">>> RNA")
        return MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(1000, 500, 50)).fit(X_train, Y_train)
    if model == 'SVM':
        print(">>> SVM")
        return SVC(kernel='rbf', gamma='scale').fit(X_train, Y_train)


if __name__ == '__main__':
    freeze_support()

    # Parameters
    MODEL = 'logit'
    PROJECT = 'ESSAI'
    EMBEDDINGS = ['BOW']
    SAMPLING = 'over'
    ANALYSEFALSE = True

    # Load data
    X = []
    for e in embeddings:
        x = load_embeddings(project, embedding)
        X.append(x)
    Y = load_labels(label_path)

    # Verify integrity
    if any(len(x) != len(Y) for x in X):
        pass
    else:
        print("X shape:{}. Y shape: {}.".format(X.shape, Y.shape))

    # Split and sample
    X_train, X_test, Y_train, Y_test = balance_split(X, Y, sampling=SAMPLING, test_size=0.15)
    print("X train and X test shapes:", X_train.shape, X_test.shape)

    # Train model
    model = trained_model(MODEL, X_train, Y_train)

    # Evaluate model
    Y_pred = model.predict(X_test)
    Y_pre = model.predict(X_train)
    acc = accuracy_score(Y_test, Y_pred, normalize=True)
    ac = accuracy_score(Y_train, Y_pre, normalize=True)
    prr = precision_score(Y_test, Y_pred, average='macro')
    pr = precision_score(Y_train, Y_pre, average='macro')
    recc = recall_score(Y_test, Y_pred, average='macro')
    rec = recall_score(Y_train, Y_pre, average='macro')
    f11 = f1_score(Y_test, Y_pred, average='macro')
    f1 = f1_score(Y_train, Y_pre, average='macro')
    print("Train accuracy: {}%\t\t Test accuracy: {}%".format(round(100 * ac, 3), round(100 * acc, 3)))
    print("Train precision: {}%\t Test precision: {}%".format(round(100 * pr, 3), round(100 * prr, 3)))
    print("Train recall: {}%\t\t Test recall: {}%".format(round(100 * rec, 3), round(100 * recc, 3)))
    print("Train F1 score: {}%\t\t Test F1 score: {}%".format(round(100 * f1, 3), round(100 * f11, 3)))

    # Interpret
    if type(model).__name__ == 'LogisticRegression':
        coef = model.coef_[0]
        imp = np.argsort(coef)[::-1]

        # Interpretation BoW
        ind = [imp[i] for i in range(20)]
        for i in ind:
            print(dictionary[i])
        ind = [imp[-i - 1] for i in range(20)]
        for i in ind:
            print(dictionary[i])

        # Interpretation LDA
        ind = [imp[i] for i in range(5)]
        topi = pd.DataFrame([[dictionary[x[0]] for x in mod.get_topic_terms(x, topn=10)] for x in ind]).T
        topi.columns = ind
        topi.loc[-1] = [model.coef_[0][x] for x in ind]
        topi = topi.sort_index()
        print(topi)

        ind = [imp[-i - 1] for i in range(5)]
        topi = pd.DataFrame([[dictionary[x[0]] for x in mod.get_topic_terms(x, topn=10)] for x in ind]).T
        topi.columns = ind
        topi.loc[-1] = [model.coef_[0][x] for x in ind]
        topi = topi.sort_index()
        print(topi)

    if ANALYSEFALSE:
        data['pred_expert'] = model.predict(X)
        data['prob_expert'] = model.predict_proba(X)[:, 1]

        falsepos = data[(data['is_expert'] == 0) & (data['pred_expert'] == 1)].sort_values(by='prob_expert',
                                                                                           ascending=False)
        falseneg = data[(data['is_expert'] == 1) & (data['pred_expert'] == 0)].sort_values(by='prob_expert',
                                                                                           ascending=True)

        sum1 = falsepos['invité'].value_counts()[:15]
        sum1 = pd.concat([sum1.reset_index()], axis=1)
        sum2 = falseneg['invité'].value_counts()[:15]
        sum1 = pd.concat([sum1.reset_index(), sum2.reset_index()], axis=1)
        del sum1['level_0']
        sum1.columns = ['Intervenant (FP)', 'Faux positifs', 'Intervenant (FN)', 'Faux négatifs']
        sum1['Total (FP)'] = pd.Series([sum(data['invité'] == x) for x in sum1['Intervenant (FP)']])
        sum1['Total (FN)'] = pd.Series([sum(data['invité'] == x) for x in sum1['Intervenant (FN)']])
        sum1 = sum1[
            ['Intervenant (FP)', 'Faux positifs', 'Total (FP)', 'Intervenant (FN)', 'Faux négatifs', 'Total (FN)']]
        print(sum1)

    print("Done!")
