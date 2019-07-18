# coding: utf-8

import sys
import subprocess
import numpy as np

from gensim.models import LdaMulticore, LsiModel, HdpModel, TfidfModel, word2vec
from gensim.matutils import corpus2dense
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.wrappers.dtmmodel import DtmModel

from utils import transcorp2matrix


# SET OF FUNCTIONS TO CREATE EMBEDDINGS (Step 1) #

# Create STM embeddings by calling R script.
def create_stm_encoding(vector_size, datafile, language):
    subprocess.call("Rscript ./external/stm.R {} {} {}".format(datafile, vector_size, language), shell=True)
    x = np.loadtxt('./external/raw_embeddings/tmp_{}_EMBEDDING_{}.csv'.format('STM', vector_size))
    return x, None


# Create CTM embeddings by calling R script.
def create_ctm_encoding(vector_size, datafile, language):
    subprocess.call("Rscript ./external/ctm.R {} {} {}".format(datafile, vector_size, language), shell=True)
    x = np.loadtxt('./external/raw_embeddings/tmp_{}_EMBEDDING_{}.csv'.format('CTM', vector_size))
    return x, None


# Loads PTM embeddings if provided by the user as specified.
def load_ptm_encoding(vector_size):
    filename = './external/raw_embeddings/tmp_{}_EMBEDDING_{}.csv'.format('PTM', vector_size)
    try:
        x = np.loadtxt("../Data Russie/encodings/RADIO_PTM_ENCODING_1000.txt")
    except OSError:
        print('No such file: {}'.format(filename))
        sys.exit(1)
    return x, None


# Created Doc2Vec embedding using gensim.
def create_d2v_encoding(corpus, vector_size):
    d2v_corpus = [TaggedDocument(doc, [i]) for i, doc in enumerate(corpus)]
    mod = Doc2Vec(d2v_corpus, vector_size=vector_size, window=5, min_count=2, workers=3)
    return np.array([mod.docvecs[i] for i in range(len(mod.docvecs))]), mod


# Create Pool embedding using word2vec.
def create_pool_encoding(corpus, vector_size):
    mod = word2vec.Word2Vec(corpus, size=vector_size, window=5, min_count=1, workers=3, sg=0)
    return np.array([mod.wv[c].mean(0) if len(c) > 0 else np.zeros(vector_size) for c in corpus]), mod


# Create BoREP embedding using random projection of words' embeddings.
def create_borep_encoding(corpus, vector_size, dim=200):
    w2v = word2vec.Word2Vec(corpus, size=dim, window=5, min_count=1, workers=3, sg=0)
    w = np.random.uniform(-1 / np.sqrt(dim), 1 / np.sqrt(dim), (vector_size, dim))
    res = np.vstack([np.apply_along_axis(lambda x: w.dot(x), 1, w2v.wv[c]).mean(0) if len(c) > 0
                     else np.zeros(vector_size) for c in corpus])
    return res, None


# Create LDA embedding using gensim's multicore implementation. Change 'workers' to suit your specs.
def create_lda_encoding(corpus, vector_size, dictionary):
    bow_corpus = [dictionary.doc2bow(x) for x in corpus]
    mod = LdaMulticore(bow_corpus, num_topics=vector_size, workers=3)
    transcorp = mod[bow_corpus]
    return transcorp2matrix(transcorp, bow_corpus, vector_size), mod


# Create LSA embedding using gensim.
def create_lsa_encoding(corpus, vector_size, dictionary):
    bow_corpus = [dictionary.doc2bow(x) for x in corpus]
    mod = LsiModel(bow_corpus, num_topics=vector_size)
    transcorp = mod[bow_corpus]
    return transcorp2matrix(transcorp, bow_corpus, vector_size), mod


# Create HDP embedding using gensim.
def create_hdp_encoding(corpus, vector_size, dictionary):
    bow_corpus = [dictionary.doc2bow(x) for x in corpus]
    mod = HdpModel(bow_corpus, id2word=dictionary)
    vector_size = mod.get_topics().shape[0]
    transcorp = mod[bow_corpus]
    return transcorp2matrix(transcorp, bow_corpus, vector_size), mod


# Create Bag-of-Words with TF-IDF.
def create_bow_encoding(corpus, vector_size, dictionary):
    dictionary.filter_extremes(keep_n=vector_size)
    bow_corpus = [dictionary.doc2bow(x) for x in corpus]
    mod = TfidfModel(bow_corpus, dictionary=dictionary)
    corpus_tfidf = mod[bow_corpus]
    return corpus2dense(corpus_tfidf, num_terms=vector_size).T, mod


# Create DTM embedding using Blei's original binary. Change to suit your specs.
def create_dtm_encoding(corpus, vector_size, dictionary, slices):
    mod_path = "./external/dtm_bin/dtm-win64.exe"
    dictionary.filter_extremes(keep_n=5000)
    bow_corpus = [dictionary.doc2bow(x) for x in corpus]
    mod = DtmModel(mod_path, corpus=bow_corpus, id2word=dictionary, time_slices=slices, num_topics=vector_size)
    return mod.gamma_, mod


# Main function to centralize the call for embedding construction.
def construct_corpus(corpus, dictionary, method='BOW', vector_size=200, datafile=None, slices=None, language='english'):
    if method == 'DOC2VEC':
        x, mod = create_d2v_encoding(corpus, vector_size)
    elif method == 'POOL':
        x, mod = create_pool_encoding(corpus, vector_size)
    elif method == 'BOREP':
        x, mod = create_borep_encoding(corpus, vector_size, dim=200)
    elif method == 'LSA':
        x, mod = create_lsa_encoding(corpus, vector_size, dictionary)
    elif method == 'LDA':
        x, mod = create_lda_encoding(corpus, vector_size, dictionary)
    elif method == 'HDP':
        print("HDP is hierarchical hence parameter K is ignored.")
        x, mod = create_hdp_encoding(corpus, vector_size, dictionary)
    elif method == 'DTM':
        x, mod = create_dtm_encoding(corpus, vector_size, dictionary, slices)
    elif method == 'STM':
        print("STM is going to run a R subprocess to construct embedding...")
        x, mod = create_stm_encoding(vector_size, datafile, language)
    elif method == 'CTM':
        print("CTM is going to run a R subprocess to construct embedding...")
        x, mod = create_ctm_encoding(vector_size, datafile, language)
    elif method == 'PTM':
        print("PTM loads pre-computed embeddings using https://github.com/qiang2100/STTM")
        x, mod = load_ptm_encoding(vector_size)
    else:
        x, mod = create_bow_encoding(corpus, vector_size, dictionary)
    return x, mod
