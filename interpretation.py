#!/usr/bin/env python
# coding: utf-8

import os
import pandas as pd
import numpy as np
import joblib

from utils import *


def interpret_bow(dictionary, imp):
    ind = [imp[i] for i in range(20)]
    x, y = [], []
    for i in ind:
        x.append(dictionary[i])
    ind = [imp[-i - 1] for i in range(20)]
    for i in ind:
        y.append(dictionary[i])
    res = pd.concat([pd.Series(x), pd.Series(y)], axis=1)
    res.columns = ['Most positive', 'Most negative']
    return res


def interpret_lda(dictionary, mod, imp, coef):
    ind = [imp[i] for i in range(5)]
    top1 = pd.DataFrame([[dictionary[x[0]] for x in mod.get_topic_terms(x, topn=10)] for x in ind]).T
    top1.columns = ind
    top1.loc[-1] = [coef[x] for x in ind]
    top1 = top1.sort_index()
    ind = [imp[-i - 1] for i in range(5)]
    top2 = pd.DataFrame([[dictionary[x[0]] for x in mod.get_topic_terms(x, topn=10)] for x in ind]).T
    top2.columns = ind
    top2.loc[-1] = [coef[x] for x in ind]
    top2 = top2.sort_index()
    return pd.concat([top1, top2], axis=1)
