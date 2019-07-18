# coding: utf-8

import sys
import os
import time
import joblib
import argparse
import pandas as pd
import numpy as np

from gensim import utils
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords

from sklearn.datasets import fetch_20newsgroups


# NLP TOOLS #

# A custom stopwords remover based on gensim's but allowing non-english languages + special stopwords
def my_remove_stopwords(s, language):
    if language == 'english':
        return remove_stopwords(s)
    path = "./datasets/stopwords-{}.txt".format(language)
    if not os.path.exists(path):
        print("{} is not a built-in language yet. Please provide '{}' file containing appropriate stopwords \
(one word by line, lower case).".format(language.capitalize(), path))
        sys.exit(1)
    with open(path, encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    specials = "./datasets/special-stopwords.txt"
    if os.path.exists(specials):
        with open(specials, encoding='utf-8') as f:
            specials = f.read().splitlines()
        stopwords = set(stopwords + specials)
    s = utils.to_unicode(s)
    return " ".join(w for w in s.split() if w.lower() not in stopwords)


# From a sparse transformed corpus of gensim, i.e. [(0, 12), (1, 15)], return matrix format: [12, 15].
def transcorp2matrix(transcorp, bow_corpus, vector_size):
    x = np.zeros((len(bow_corpus), vector_size))
    for i, doc in enumerate(transcorp):
        for topic in doc:
            x[i][topic[0]] = topic[1]
    return x


# DATA IMPORT TOOLS #

# Load user's .csv file data set or 20News data set
def load_corpus(datafile, embedding, preprocess=True, language='english'):
    corpus, slices, data = None, None, None
    if datafile == '20News':
        source = fetch_20newsgroups(subset='all', remove=('headers', 'footers'))  # , 'quotes'
        res = pd.Series(source.data, name='res')
        if preprocess:
            print("Pre-processing text...")
            corpus = [preprocess_string(remove_stopwords(x)) for x in res]
        else:
            corpus = [x.split() for x in res]
    elif not os.path.exists(datafile):
        print("No such file: '{}'".format(datafile))
        sys.exit(1)
    elif not datafile[-4:] == '.csv':
        print("Currently supported inputs: '20News' or .csv file containing a column called 'text'.")
        sys.exit(1)
    else:
        data = pd.read_csv(datafile, encoding='utf-8')
        if 'text' not in data.columns:
            print("Column containing text must be called 'text'. Please check your datafile format.")
            sys.exit(1)
        if preprocess:
            print("Pre-processing text...")
            corpus = [preprocess_string(my_remove_stopwords(x, language)) for x in data['text']]
        else:
            corpus = [x.split() for x in data['text'].tolist()]
        if 'toremove' in data.columns:
            rm = [preprocess_string(my_remove_stopwords(' '.join(x), language)) for x in data['toremove'].apply(eval)]
            corpus = [[y for y in x if y not in rm[i]] for i, x in enumerate(corpus)]
    if embedding == 'DTM':
        if datafile == '20News':
            print("DTM cannot be used with '20News' datafile as time information is not provided.")
            sys.exit(1)
        elif datafile[-4:] == '.csv':
            if 'year' in data.columns:
                slices = data['year'].value_counts().sort_index().tolist()
            else:
                print("DTM cannot be used with this datafile as time information is required.\
                        Try .csv file with 'year' column.")
                sys.exit(1)
        else:
            print("DTM cannot be used with this datafile as time information is required.\
                    Try .csv file with 'year' column.")
            sys.exit(1)
    return corpus, slices


# Loading labels from user's file or 20News
def load_labels(datafile):
    if datafile == '20News':
        source = fetch_20newsgroups(subset='all', remove=('headers', 'footers'))
        y = pd.Series(source.target, name='label')
    elif not os.path.exists(datafile):
        print("No such file: '{}'".format(datafile))
        sys.exit(1)
    elif not datafile[-4:] == '.csv':
        print("Currently supported inputs: '20News' or .csv file containing a column called 'label'.")
        sys.exit(1)
    else:
        data = pd.read_csv(datafile, encoding='utf-8')
        if 'label' not in data.columns:
            print("Column called 'label' is required to perform classification task.")
            sys.exit(1)
        else:
            y = data['label']
    return y


# Loading embeddings computed in Step 1
def load_embeddings(project, embedding, k):
    filename = './results/{}/embeddings/{}_embedding_{}.csv'.format(project, embedding, k)
    try:
        res = np.genfromtxt(filename, delimiter=',')
        return res
    except OSError:
        print("The embedding you're trying to work with is not computed. \
Try to use the same command with '-mode encode' first (instead of '-mode classify') to compute it.")
        sys.exit(1)


# Loading models used in Step 1
def load_model(project, embedding, algo, k):
    filename = './results/{}/classifiers/{}_{}_{}.joblib'.format(project, embedding, algo, k)
    try:
        model = joblib.load(filename)
        return model
    except OSError:
        print("The classifier you're trying to work with is not computed. \
Try to use the same command with '-mode classify' first (instead of '-mode interpret') to compute it.")
        sys.exit(1)


# A parser to get user's inputs
def read_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-mode',
                        choices=['all', 'encode', 'classify', 'interpret'],
                        required=True,
                        help="Step you want to perform (can be 'all').")
    parser.add_argument('-input',
                        type=str,
                        required=True,
                        help="Path to your .csv input file, or '20News'.")
    parser.add_argument('-embed',
                        choices=['BOW', 'DOC2VEC', 'POOL', 'BOREP', 'LSA', 'LDA', 'HDP', 'DTM', 'STM', 'CTM', 'PTM'],
                        required=True,
                        help='Embedding to use.')
    parser.add_argument('-project',
                        type=str,
                        default=time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()),
                        help='Name of the project where to find the results later.')
    parser.add_argument('-k',
                        type=int,
                        default=200,
                        help='Size of your embedding vectors.')
    parser.add_argument('-prep',
                        type=bool,
                        default=True,
                        help='Specify if you want to pre-process text (i.e. lowercase, lemmatize...).')
    parser.add_argument('-langu',
                        type=str,
                        default='english',
                        help='Language to use for text pre-processing.')
    parser.add_argument('-algo',
                        choices=['LOGIT', 'NBAYES', 'ADAB', 'DTREE', 'KNN', 'ANN', 'SVM'],
                        default='LOGIT',
                        help='Classifier to use.')
    parser.add_argument('-samp',
                        choices=['OVER', 'UNDER', 'NONE'],
                        default='NONE',
                        help='Sampling to use to prevent imbalanced data sets.')
    return parser.parse_args()
