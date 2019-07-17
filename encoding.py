#!/usr/bin/env python
# coding: utf-8

import os
import subprocess
import pandas as pd
import numpy as np

from gensim.models import LdaMulticore, LsiModel, HdpModel, TfidfModel, word2vec
from gensim.matutils import corpus2dense
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.wrappers.dtmmodel import DtmModel

from utils import *


# STM
def load_stm_encoding():
    dat = pd.read_csv('../Data Russie/encodings/RADIO_STM_ENCODING_1000.csv', encoding='utf-8')
    X = np.array(dat[[c for c in dat.columns if 'Topic' in c]])
    del dat
    return X, None


# CTM
def create_ctm_encoding(vector_size, input, language):
    subprocess.call("Rscript ./external/ctm.R {} {} {}".format(input, vector_size, language), shell=True)
    X = np.loadtxt('./external/raw_embeddings/tmp_{}_EMBEDDING_{}.csv'.format('CTM', vector_size))
    return X, None


# PTM
def load_ptm_encoding():
    X = np.loadtxt("../Data Russie/encodings/RADIO_PTM_ENCODING_1000.txt")
    return X, None


# Doc2Vec
def create_d2v_encoding(corpus, vector_size):
    d2v_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
    mod = Doc2Vec(d2v_corpus, vector_size=vector_size, window=5, min_count=2, workers=3)
    return np.array([mod.docvecs[i] for i in range(len(mod.docvecs))]), mod


# Pool (BoE)
def create_pool_encoding(corpus, vector_size):
    mod = word2vec.Word2Vec(corpus, size=vector_size, window=5, min_count=1, workers=3, sg=0)
    return np.array([mod.wv[c].mean(0) if len(c) > 0 else np.zeros(vector_size) for c in corpus]), mod


# BoREP
def create_borep_encoding(corpus, vector_size, dim=200):
    w2v = word2vec.Word2Vec(corpus, size=dim, window=5, min_count=1, workers=3, sg=0)
    W = np.random.uniform(-1 / np.sqrt(dim), 1 / np.sqrt(dim), (vector_size, dim))
    res = np.vstack([np.apply_along_axis(lambda x: W.dot(x), 1, w2v.wv[c]).mean(0) if len(c) > 0
                     else np.zeros(vector_size) for c in corpus])
    return res, None


# LDA
def create_lda_encoding(corpus, vector_size, dictionary):
    bow_corpus = [dictionary.doc2bow(x) for x in corpus]
    mod = LdaMulticore(bow_corpus, num_topics=vector_size, workers=3)
    transcorp = mod[bow_corpus]
    return transcorp2matrix(transcorp, bow_corpus, vector_size), mod


# LSA
def create_lsa_encoding(corpus, vector_size, dictionary):
    bow_corpus = [dictionary.doc2bow(x) for x in corpus]
    mod = LsiModel(bow_corpus, num_topics=vector_size)
    transcorp = mod[bow_corpus]
    return transcorp2matrix(transcorp, bow_corpus, vector_size), mod


# HDP
def create_hdp_encoding(corpus, vector_size, dictionary):
    bow_corpus = [dictionary.doc2bow(x) for x in corpus]
    mod = HdpModel(bow_corpus, id2word=dictionary)
    vector_size = mod.get_topics().shape[0]
    transcorp = mod[bow_corpus]
    return transcorp2matrix(transcorp, bow_corpus, vector_size), mod


# BoW
def create_bow_encoding(corpus, vector_size, dictionary):
    dictionary.filter_extremes(keep_n=vector_size)
    bow_corpus = [dictionary.doc2bow(x) for x in corpus]
    mod = TfidfModel(bow_corpus, dictionary=dictionary)
    corpus_tfidf = mod[bow_corpus]
    return corpus2dense(corpus_tfidf, num_terms=vector_size).T, mod


# DTM
def create_dtm_encoding(corpus, vector_size, dictionary, slices):
    mod_path = "./external/dtm_bin/dtm-win64.exe"
    dictionary.filter_extremes(keep_n=5000)
    bow_corpus = [dictionary.doc2bow(x) for x in corpus]
    mod = DtmModel(mod_path, corpus=bow_corpus, id2word=dictionary, time_slices=slices, num_topics=vector_size)
    return mod.gamma_, mod


def construct_corpus(corpus, dictionary, method='BOW', vector_size=200, input=None, slices=None, language='english'):
    if method == 'DOC2VEC':
        X, mod = create_d2v_encoding(corpus, vector_size)
    elif method == 'POOL':
        X, mod = create_pool_encoding(corpus, vector_size)
    elif method == 'BOREP':
        X, mod = create_borep_encoding(corpus, vector_size, dim=200)
    elif method == 'LSA':
        X, mod = create_lsa_encoding(corpus, vector_size, dictionary)
    elif method == 'LDA':
        X, mod = create_lda_encoding(corpus, vector_size, dictionary)
    elif method == 'HDP':
        print("HDP is hierarchical hence parameter K is ignored.")
        X, mod = create_hdp_encoding(corpus, vector_size, dictionary)
    elif method == 'DTM':
        X, mod = create_dtm_encoding(corpus, vector_size, dictionary, slices)
    elif method == 'STM':
        print("STM is going to run a R subprocess to construct embedding...")
        X, mod = load_stm_encoding()
    elif method == 'CTM':
        print("CTM is going to run a R subprocess to construct embedding...")
        X, mod = create_ctm_encoding(vector_size, input, language)
    elif method == 'PTM':
        print("PTM loads pre-computed embeddings using https://github.com/qiang2100/STTM")
        X, mod = load_ptm_encoding()
    # Default: Bag of Words
    else:
        X, mod = create_bow_encoding(corpus, vector_size, dictionary)
    return X, mod

