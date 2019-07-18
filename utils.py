import sys
import os
import re
import time
import unicodedata
import joblib
import argparse
import pandas as pd
import numpy as np

from gensim import utils
from gensim.parsing.preprocessing import preprocess_string, remove_stopwords

from sklearn.datasets import fetch_20newsgroups


# STRING PRE-PROCESSING #

def no_paren(string):
    if not pd.isnull(string):
        res = re.sub(r"\s?\(.*\)\)?", '', string)
        return re.sub(r'\[.*\]\s?', '', res)


def normalize(string, rm_space=True, rm_punct=True, lower=True):
    if pd.isnull(string):
        return None
    s = unicodedata.normalize('NFKD', string).encode('ascii', 'ignore').decode("utf-8")
    s = re.compile('[^a-zA-Z\s\.\,\:\;]').sub('', s)
    if rm_space:
        s = re.compile('[^a-zA-Z\.\,\:\;]').sub('', s)
    if rm_punct:
        s = re.compile('[^a-zA-Z\s]').sub('', s)
    if lower:
        s = s.lower()
    s = re.sub(' +', ' ', s).lstrip().rstrip()
    return s


def listate(l):
    if not isinstance(l, (list,)):
        return None
    res = []
    for i in range(0, len(l), 2):
        res.append((l[i], l[i + 1]))
    return res


def drop_journa(l):
    if not isinstance(l, (list,)):
        return None
    keep = []
    for g in l:
        if g[0] in ['PAR']:
            keep.append(g[1])
    if len(keep) > 0:
        return keep
    else:
        return None


# NLP #

def my_remove_stopwords(s, language):
    if language == 'english':
        return remove_stopwords(s)
    filepath = "./datasets/stopwords-{}.txt".format(language)
    if not os.path.exists(filepath):
        print("{} is not a built-in language yet. Please provide '{}' file containing appropriate stopwords \
(one word by line, lower case).".format(language.capitalize(), filepath))
        sys.exit(1)
    with open(filepath, encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    with open("./datasets/special-stopwords.txt", encoding='utf-8') as f:
        specials = f.read().splitlines()
    stopwords = set(stopwords + specials)
    s = utils.to_unicode(s)
    return " ".join(w for w in s.split() if w.lower() not in stopwords)


def transcorp2matrix(transcorp, bow_corpus, vector_size):
    X = np.zeros((len(bow_corpus), vector_size))
    for i, doc in enumerate(transcorp):
        for topic in doc:
            X[i][topic[0]] = topic[1]
    return X


# DATA IMPORT #

# Import data
def load_corpus(INPUT, EMBEDDING, PREPROCESS=True, LANGUAGE='english'):
    corpus, slices, data = None, None, None
    if INPUT == '20News':
        source = fetch_20newsgroups(subset='all', remove=('headers', 'footers'))  # , 'quotes'
        res = pd.Series(source.data, name='res')
        if PREPROCESS:
            print("Pre-processing text...")
            corpus = [preprocess_string(remove_stopwords(x)) for x in res]
        else:
            corpus = [x.split() for x in res]
    elif not os.path.exists(INPUT):
        print("No such file: '{}'".format(INPUT))
        sys.exit(1)
    elif not INPUT[-4:] == '.csv':
        print("Currently supported inputs: '20News' or .csv file containing a column called 'text'.")
        sys.exit(1)
    else:
        data = pd.read_csv(INPUT, encoding='utf-8')
        if 'text' not in data.columns:
            print("Column containing text must be called 'text'. Please check your input format.")
            sys.exit(1)
        if PREPROCESS:
            print("Pre-processing text...")
            corpus = [preprocess_string(my_remove_stopwords(x, LANGUAGE)) for x in data['text']]
        else:
            corpus = [x.split() for x in data['text'].tolist()]
        if 'toremove' in data.columns:
            rm = [preprocess_string(my_remove_stopwords(' '.join(x), LANGUAGE)) for x in data['toremove'].apply(eval)]
            corpus = [[y for y in x if y not in rm[i]] for i, x in enumerate(corpus)]
    if EMBEDDING == 'DTM':
        if INPUT == '20News':
            print("DTM cannot be used with '20News' input as time information is not provided.")
            sys.exit(1)
        elif INPUT[-4:] == '.csv':
            if 'year' in data.columns:
                slices = data['year'].value_counts().sort_index().tolist()
            else:
                print("DTM cannot be used with this input as time information is required.\
                        Try .csv file with 'year' column.")
                sys.exit(1)
        else:
            print("DTM cannot be used with this input as time information is required.\
                    Try .csv file with 'year' column.")
            sys.exit(1)
    return corpus, slices


# Loading labels
def load_labels(INPUT):
    Y = None
    if INPUT == '20News':
        source = fetch_20newsgroups(subset='all', remove=('headers', 'footers'))
        Y = pd.Series(source.target, name='label')
    elif not os.path.exists(INPUT):
        print("No such file: '{}'".format(INPUT))
        sys.exit(1)
    elif not INPUT[-4:] == '.csv':
        print("Currently supported inputs: '20News' or .csv file containing a column called 'label'.")
        sys.exit(1)
    else:
        data = pd.read_csv(INPUT, encoding='utf-8')
        if 'label' not in data.columns:
            print("Column called 'label' is required to perform classification task.")
            sys.exit(1)
        else:
            Y = data['label']
    return Y


def load_embeddings(PROJECT, EMBEDDING, K):
    filename = './results/{}/embeddings/{}_embedding_{}.csv'.format(PROJECT, EMBEDDING, K)
    try:
        res = np.genfromtxt(filename, delimiter=',')
        return res
    except OSError:
        print("The embedding you're trying to work with is not computed. \
Try to use the same command with '-mode encode' first (instead of '-mode classify') to compute it.")
        sys.exit(1)


def load_model(PROJECT, EMBEDDING, ALGO, K):
    filename = './results/{}/classifiers/{}_{}_{}.joblib'.format(PROJECT, EMBEDDING, ALGO, K)
    try:
        model = joblib.load(filename)
        return model
    except OSError:
        print("The classifier you're trying to work with is not computed. \
Try to use the same command with '-mode classify' first (instead of '-mode interpret') to compute it.")
        sys.exit(1)


# Parser to get user's inputs
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
