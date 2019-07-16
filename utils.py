import re
import unicodedata
import spacy
import pandas as pd
import numpy as np
from gensim import utils


# STRING PRE-PROCESSING #

def no_paren(string):
    if not pd.isnull(string):
        res = re.sub(r'\s?\(.*\)\)?', '', string)
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

def my_remove_stopwords(s):
    with open("./datasets/stopwords-fr.txt", encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    with open("./datasets/special-stopwords.txt", encoding='utf-8') as f:
        specials = f.read().splitlines()
    stopwords = set(stopwords + specials)
    s = utils.to_unicode(s)
    return " ".join(w for w in s.split() if w.lower() not in stopwords)


def my_stemmer(doc):
    nlp = spacy.load("fr_core_news_sm")
    doc_lem = nlp(doc)
    return ' '.join([d.lemma_ for d in doc_lem])


def transcorp2matrix(transcorp, bow_corpus, vector_size):
    X = np.zeros((len(bow_corpus), vector_size))
    for i, doc in enumerate(transcorp):
        for topic in doc:
            X[i][topic[0]] = topic[1]
    return X
