#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
from multiprocessing import freeze_support

from gensim.corpora.dictionary import Dictionary
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords
from gensim.models import LdaMulticore, LsiModel, HdpModel, TfidfModel, word2vec
from gensim.matutils import corpus2dense
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.wrappers.dtmmodel import DtmModel

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

from utils import my_remove_stopwords, transcorp2matrix
from sklearn.datasets import fetch_20newsgroups


# STM
def load_stm_encoding():
    dat = pd.read_csv('../Data Russie/encodings/RADIO_STM_ENCODING_1000.csv', encoding='utf-8')
    X = np.array(dat[[c for c in dat.columns if 'Topic' in c]])
    del dat
    return X


# CTM
def load_ctm_encoding():
    odata = pd.read_csv('../Data Russie/encodings/RADIO_CTM_ENCODING_2.csv', encoding='utf-8')
    X = np.array(odata[[c for c in odata.columns if 'V' in c]])
    del odata
    return X


# HTLM
def load_htlm_encoding():
    odata = pd.read_json('../Data Russie/encodings/RADIO_HTLM_ENCODING_1000.json', encoding='utf-8')
    res = np.zeros((len(corpus), len(odata)))
    for i, top in enumerate(odata['doc']):
        for doc in top:
            res[int(doc[0])][i] = doc[1] + 0.01 * doc[2]
    som = res.sum(axis=1)[:, None]
    res = np.divide(res, som, out=np.zeros_like(res), where=som != 0)
    del odata
    return res


# PTM
def load_ptm_encoding():
    X = np.loadtxt("../Data Russie/encodings/RADIO_PTM_ENCODING_1000.txt")
    return X


# Doc2Vec
def create_d2v_encoding(corpus, vector_size):
    d2v_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
    d2v = Doc2Vec(d2v_corpus, vector_size=vector_size, window=5, min_count=2, workers=3)
    return np.array([d2v.docvecs[i] for i in range(len(d2v.docvecs))])


# Pool (BoE)
def create_pool_encoding(corpus, vector_size):
    w2v = word2vec.Word2Vec(corpus, size=vector_size, window=5, min_count=1, workers=3, sg=0)
    return np.array([w2v.wv[c].mean(0) if len(c) > 0 else np.zeros(vector_size) for c in corpus])


# BoREP
def create_borep_encoding(corpus, vector_size, dim=200):
    w2v = word2vec.Word2Vec(corpus, size=dim, window=5, min_count=1, workers=3, sg=0)
    W = np.random.uniform(-1 / np.sqrt(dim), 1 / np.sqrt(dim), (vector_size, dim))
    return np.vstack([np.apply_along_axis(lambda x: W.dot(x), 1, w2v.wv[c]).mean(0) if len(c) > 0
                      else np.zeros(vector_size) for c in corpus])


# LDA
def create_lda_encoding(corpus, vector_size, dictionary):
    bow_corpus = [dictionary.doc2bow(x) for x in corpus]
    mod = LdaMulticore(bow_corpus, num_topics=vector_size, workers=3)
    transcorp = mod[bow_corpus]
    return transcorp2matrix(transcorp, bow_corpus, vector_size)


# LSA
def create_lsa_encoding(corpus, vector_size, dictionary):
    bow_corpus = [dictionary.doc2bow(x) for x in corpus]
    mod = LsiModel(bow_corpus, num_topics=vector_size)
    transcorp = mod[bow_corpus]
    return transcorp2matrix(transcorp, bow_corpus, vector_size)


# HDP
def create_hdp_encoding(corpus, vector_size, dictionary):
    bow_corpus = [dictionary.doc2bow(x) for x in corpus]
    mod = HdpModel(bow_corpus, id2word=dictionary)
    vector_size = mod.get_topics().shape[0]
    transcorp = mod[bow_corpus]
    return transcorp2matrix(transcorp, bow_corpus, vector_size)


# BoW
def create_bow_encoding(corpus, vector_size, dictionary):
    dictionary.filter_extremes(keep_n=vector_size)
    bow_corpus = [dictionary.doc2bow(x) for x in corpus]
    mod = TfidfModel(bow_corpus, dictionary=dictionary)
    corpus_tfidf = mod[bow_corpus]
    return corpus2dense(corpus_tfidf, num_terms=vector_size).T


# DTM
def create_dtm_encoding(corpus, vector_size, dictionary, slices):
    mod_path = "./external/dtm_bin/dtm-win64.exe"
    dictionary.filter_extremes(keep_n=5000)
    bow_corpus = [dictionary.doc2bow(x) for x in corpus]
    dtm = DtmModel(mod_path, corpus=bow_corpus, id2word=dictionary, time_slices=slices, num_topics=vector_size)
    return dtm.gamma_


def construct_corpus(corpus, method='BOW', vector_size=200, slices=None):
    dictionary = Dictionary(corpus)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    # Reference models
    if method == 'DOC2VEC':
        X = create_d2v_encoding(corpus, vector_size)
    elif method == 'POOL':
        X = create_pool_encoding(corpus, vector_size)
    elif method == 'BOREP':
        X = create_borep_encoding(corpus, vector_size, dim=200)
    elif method == 'LSA':
        X = create_lsa_encoding(corpus, vector_size, dictionary)
    # Topic models
    elif method == 'LDA':
        X = create_lda_encoding(corpus, vector_size, dictionary)
    elif method == 'HDP':
        print("Warning: HDP is hierarchical hence vector_size is ignored.")
        X = create_hdp_encoding(corpus, vector_size, dictionary)
    elif method == 'HTLM':
        print("Warning: HTLM is hierarchical hence vector_size is ignored.")
        print("Warning: HTLM loads pre-computed embeddings using https://github.com/kmpoon/hlta")
        X = load_htlm_encoding()
    elif method == 'DTM':
        X = create_dtm_encoding(corpus, vector_size, dictionary, slices)
    elif method == 'STM':
        print("STM embedding requires to drop observations. Y variable will be changed and vector_size ignored.")
        print("Warning: STM uses pre-computed embeddings using https://cran.r-project.org/web/packages/stm/")
        X = load_stm_encoding()
    elif method == 'CTM':
        print("CTM loads pre-computed embeddings using https://cran.r-project.org/web/packages/topicmodels/")
        X = load_ctm_encoding()
    elif method == 'PTM':
        print("PTM loads pre-computed embeddings using https://github.com/qiang2100/STTM")
        X = load_ptm_encoding()
    # Default: Bag of Words
    else:
        print("Using default encoding: Bag of Words.")
        X = create_bow_encoding(corpus, vector_size, dictionary)
    return X


def load_embeddings(PROJECT, EMBEDDING, K):
    filename = './encodings/{}/{}_embedding_{}.csv'.format(PROJECT, EMBEDDING, K)
    return np.genfromtxt(filename, delimiter=',')


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


def evaluate_model(model):
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


def interpret_bow(dictionary):
    ind = [imp[i] for i in range(20)]
    print("Most positive:")
    for i in ind:
        print(dictionary[i])
    ind = [imp[-i - 1] for i in range(20)]
    print("Most negative:")
    for i in ind:
        print(dictionary[i])


def interpret_lda(dictionary, mod):
    ind = [imp[i] for i in range(5)]
    topi = pd.DataFrame([[dictionary[x[0]] for x in mod.get_topic_terms(x, topn=10)] for x in ind]).T
    topi.columns = ind
    topi.loc[-1] = [model.coef_[0][x] for x in ind]
    topi = topi.sort_index()
    print("Most positive:")
    print(topi)

    ind = [imp[-i-1] for i in range(5)]
    topi = pd.DataFrame([[dictionary[x[0]] for x in mod.get_topic_terms(x, topn=10)] for x in ind]).T
    topi.columns = ind
    topi.loc[-1] = [model.coef_[0][x] for x in ind]
    topi = topi.sort_index()
    print("Most negative:")
    print(topi)


def analyse_errors(data, model):
    data['pred_expert'] = model.predict(X)
    data['prob_expert'] = model.predict_proba(X)[:, 1]
    falsepos = data[(data['label'] == 0) & (data['pred_expert'] == 1)].sort_values(by='prob_expert',
                                                                                       ascending=False)
    falseneg = data[(data['label'] == 1) & (data['pred_expert'] == 0)].sort_values(by='prob_expert',
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


if __name__ == '__main__':
    freeze_support()

    INPUT = './datasets/dataset_ina_radio.csv'
    PROJECT = 'INA'
    EMBEDDING = 'LDA'
    ALGO = 'logit'
    K = 100
    PREPROCESS = True
    MODE = 'all'
    ANALYSE = True
    SAMPLING = 'over'

    # First mode: create embedding
    if MODE == 'all' or MODE == 'encode':
        print('Loading data...')
        if INPUT == '20News':
            source = fetch_20newsgroups(subset='all', remove=('headers', 'footers'))  # , 'quotes'
            res = pd.Series(source.data, name='res')
            # label = pd.Series(source.target, name='label')
            if PREPROCESS:
                print("Pre-processing text...")
                corpus = [preprocess_string(remove_stopwords(x)) for x in res]
            else:
                corpus = [x.split() for x in res]

        elif not os.path.exists(INPUT):
            print("No such file: '{}'".format(INPUT))
            exit(1)

        elif not INPUT[-4:] == '.csv':
            print("Currently supported inputs: '20News' or .csv file containing a column called 'text'.")
            exit(1)

        else:
            data = pd.read_csv(INPUT, encoding='utf-8')
            if 'text' not in data.columns:
                print("Column containing text must be called 'text'. Please check your input format.")
                exit(1)
            if PREPROCESS:
                print("Pre-processing text...")
                corpus = [preprocess_string(my_remove_stopwords(x)) for x in data['text']]
            else:
                corpus = [x.split() for x in data['text'].tolist()]
            if 'toremove' in data.columns:
                rm = [preprocess_string(my_remove_stopwords(' '.join(x))) for x in data['toremove'].apply(eval)]
                corpus = [[y for y in x if y not in rm[i]] for i, x in enumerate(corpus)]

        slices = None
        if EMBEDDING == 'DTM':
            if INPUT[-4:] == '.csv':
                slices = data['year'].value_counts().sort_index().tolist()
            else:
                print("DTM cannot be used with this input as time information is required. Try .csv file with 'year' column")
                exit(1)

        print("Creating embedding using {} with size {}...".format(EMBEDDING, K))
        X = construct_corpus(corpus, method=EMBEDDING, vector_size=K, slices=slices)

        path = './encodings/{}/'.format(PROJECT)
        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of directory '{}' failed".format(path))

        filename = './encodings/{}/{}_embedding_{}.csv'.format(PROJECT, EMBEDDING, K)
        np.savetxt(filename, X, delimiter=",")

    # Second mode: train model
    if MODE == 'all' or MODE == 'classify':
        print('Loading data...')
        if INPUT == '20News':
            source = fetch_20newsgroups(subset='all', remove=('headers', 'footers'))
            Y = pd.Series(source.target, name='label')

        elif not os.path.exists(INPUT):
            print("No such file: '{}'".format(INPUT))
            exit(1)

        elif not INPUT[-4:] == '.csv':
            print("Currently supported inputs: '20News' or .csv file containing a column called 'label'.")
            exit(1)

        else:
            data = pd.read_csv(INPUT, encoding='utf-8')
            if 'label' not in data.columns:
                print("Column called 'label' is required to perform classification task.")
                exit(1)
            else:
                Y = data['label']

        X = load_embeddings(PROJECT, EMBEDDING, K)

        # Split and sample
        X_train, X_test, Y_train, Y_test = balance_split(X, Y, sampling=SAMPLING, test_size=0.15)
        print("X train and X test shapes:", X_train.shape, X_test.shape)

        # Train model
        model = trained_model(ALGO, X_train, Y_train)

        # Evaluate model
        evaluate_model(model)

        # Interpret
        if type(model).__name__ == 'LogisticRegression':
            coef = model.coef_[0]
            imp = np.argsort(coef)[::-1]
            dictionary = Dictionary(corpus)
            dictionary.filter_extremes(no_below=2, no_above=0.5)
            # Interpretation BoW
            if EMBEDDING == 'BOW':
                interpret_bow(dictionary)
            # Interpretation LDA
            if EMBEDDING == 'LDA':
                interpret_lda(dictionary, mod)

        if ANALYSE:
            analyse_errors(data, model)
