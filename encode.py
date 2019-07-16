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


# DMM
def load_dmm_encoding():
    X = np.loadtxt("../Data Russie/encodings/RADIO_DMM_ENCODING_1000.txt")
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
    global mod
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
    global mod
    dictionary.filter_extremes(keep_n=vector_size)
    bow_corpus = [dictionary.doc2bow(x) for x in corpus]
    mod = TfidfModel(bow_corpus, dictionary=dictionary)
    corpus_tfidf = mod[bow_corpus]
    return corpus2dense(corpus_tfidf, num_terms=vector_size).T


# DTM
def create_dtm_encoding(corpus, vector_size, dictionary):
    modpath = "./external/dtm/dtm-win64.exe"
    md = data.sort_values(by='year')
    rm_names = [preprocess_string(my_remove_stopwords(' '.join(x))) for x in md['invités'].apply(eval)]
    slices = md['year'].value_counts().sort_index().tolist()
    corpus = [preprocess_string(my_remove_stopwords(x)) for x in md['res']]
    corpus = [[y for y in x if y not in rm_names[i]] for i, x in enumerate(corpus)]
    dictionary = Dictionary(corpus)
    dictionary.filter_extremes(no_below=2, no_above=0.5, keep_n=5000)
    bow_corpus = [dictionary.doc2bow(x) for x in corpus]
    dtm = DtmModel(modpath, corpus=bow_corpus, id2word=dictionary, time_slices=slices, num_topics=vector_size)
    return dtm.gamma_


def construct_corpus(corpus, method='BOW', vector_size=200):
    dictionary = Dictionary(corpus)
    dictionary.filter_extremes(no_below=2, no_above=0.5)
    ###### BENCHMARKS ######
    if method == 'DOC2VEC':
        X = create_d2v_encoding(corpus, vector_size)
    elif method == 'POOL':
        X = create_pool_encoding(corpus, vector_size)
    elif method == 'BOREP':
        X = create_borep_encoding(corpus, vector_size, dim=200)
    elif method == 'LSA':
        X = create_lsa_encoding(corpus, vector_size, dictionary)
    ###### TOPIC MODELS ######
    elif method == 'LDA':
        X = create_lda_encoding(corpus, vector_size, dictionary)
    elif method == 'HDP':
        print("Warning: HDP is hierachical hence vector_size is ignored.")
        X = create_hdp_encoding(corpus, vector_size, dictionary)
    elif method == 'HTLM':
        print("Warning: HTLM is hierachical hence vector_size is ignored.")
        print("Warning: HTLM loads pre-computed embeddings using https://github.com/kmpoon/hlta")
        X = load_htlm_encoding()
    elif method == 'DTM':
        X = create_dtm_encoding(corpus, vector_size, dictionary)
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
    elif method == 'DMM':
        print("DMM loads pre-computed embeddings using https://github.com/qiang2100/STTM")
        X = load_dmm_encoding()
    ###### DEFAULT IS BOW ######
    else:
        print("Using default encoding: Bag of Words.")
        X = create_bow_encoding(corpus, vector_size, dictionary)
    return X


if __name__ == '__main__':
    freeze_support()

    INPUT = './datasets/dataset_ina_radio.csv'
    PROJECT = 'INA'
    EMBEDDING = 'STM'
    K = 200

    print('Loading data...')
    if INPUT == '20News':
        source = fetch_20newsgroups(subset='all', remove=('headers', 'footers'))  # , 'quotes'
        res = pd.Series(source.data, name='res')
        # label = pd.Series(source.target, name='label')
        corpus = [preprocess_string(remove_stopwords(x)) for x in res]

    elif not os.path.exists(INPUT):
        print("No such file: '{}'".format(INPUT))
        exit()

    elif INPUT[-4:] == '.txt':
        with open(INPUT, encoding='utf-8') as f:
            corpus = f.read().splitlines()

    elif INPUT[-4:] == '.csv':
        data = pd.read_csv(INPUT, encoding='utf-8')
        corpus = [preprocess_string(my_remove_stopwords(x)) for x in data['text']]
        if 'toremove' in data.columns:
            rm = [preprocess_string(my_remove_stopwords(' '.join(x))) for x in data['toremove'].apply(eval)]
            corpus = [[y for y in x if y not in rm[i]] for i, x in enumerate(corpus)]

    else:
        print("Currently supported inputs: '20News', .csv file or .txt file.")
        exit()

    print("Creating embedding using {} with size {}...".format(EMBEDDING, K))
    X = construct_corpus(corpus, method=EMBEDDING, vector_size=200)

    path = './encodings/{}/'.format(PROJECT)
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError:
            print("Creation of directory '{}' failed".format(path))

    filename = './encodings/{}/{}_embedding_{}.csv'.format(PROJECT, EMBEDDING, K)
    np.savetxt(filename, X, delimiter=",")
