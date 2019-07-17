#!/usr/bin/env python
# coding: utf-8

import sys
import os
import pandas as pd
import numpy as np
import joblib
from multiprocessing import freeze_support
from gensim.corpora.dictionary import Dictionary

from utils import *
from encoding import *
from classification import *
from interpretation import *


if __name__ == '__main__':
    freeze_support()

    INPUT = './datasets/dataset_ina_radio.csv'
    PROJECT = 'INA'
    EMBEDDING = 'CTM'
    ALGO = 'LOGIT'
    K = 2
    PREPROCESS = True
    MODE = 'all'
    SAMPLING = 'OVER'
    LANGUAGE = 'french'

    check_intergrity(MODE, PROJECT, INPUT, EMBEDDING, K, ALGO, PREPROCESS, SAMPLING, LANGUAGE)

    # First mode: create embedding
    if MODE == 'all' or MODE == 'encode':

        print('Loading data...')
        corpus, slices = load_corpus(INPUT, EMBEDDING, PREPROCESS, LANGUAGE)
        print('Data loaded, contains {} rows.'.format(len(corpus)))

        path = './results/{}/'.format(PROJECT)
        if not os.path.exists(path):
            try:
                os.mkdir(path)
            except OSError:
                print("Creation of directory '{}' failed".format(path))
                sys.exit(1)
            try:
                os.mkdir(path + 'embeddings/')
                os.mkdir(path + 'models/')
                os.mkdir(path + 'classifiers/')
                os.mkdir(path + 'performances/')
                os.mkdir(path + 'interpretation/')
            except OSError:
                print("Creation of subdirectories of '{}' failed".format(path))
                sys.exit(1)

        print("Creating embedding using {} with size {}...".format(EMBEDDING, K))
        dictionary = Dictionary(corpus)
        dictionary.filter_extremes(no_below=2, no_above=0.5)
        X, mod = construct_corpus(corpus, dictionary, method=EMBEDDING, vector_size=K, input=INPUT, slices=slices, language=LANGUAGE)

        filename = './results/{}/embeddings/{}_embedding_{}.csv'.format(PROJECT, EMBEDDING, K)
        np.savetxt(filename, X, delimiter=",")
        if mod is not None:
            filename = './results/{}/models/{}_model_{}.model'.format(PROJECT, EMBEDDING, K)
            mod.save(filename)
        print('Encoding done!')

    # Second mode: train model
    if MODE == 'all' or MODE == 'classify':

        print('Loading labels...')
        Y = load_labels(INPUT)

        print('Loading embedding...')
        X = load_embeddings(PROJECT, EMBEDDING, K)

        print('Splitting into test and train, sampling...')
        X_train, X_test, Y_train, Y_test = balance_split(X, Y, sampling=SAMPLING, test_size=0.15)

        print('Training {} on {} embedding with size {}...'.format(ALGO, EMBEDDING, K))
        model = trained_model(ALGO, X_train, Y_train)
        filename = './results/{}/classifiers/{}_{}_{}.joblib'.format(PROJECT, EMBEDDING, ALGO, K)
        joblib.dump(model, filename)

        print('Computing performance metrics...')
        perf = evaluate_model(model, X_train, X_test, Y_train, Y_test)
        filename = './results/{}/performances/{}_perf_{}.csv'.format(PROJECT, EMBEDDING, K)
        perf.to_csv(filename)

        print('Classification done!')

    # Interpret
    if MODE == 'all' or MODE == 'interpret':

        print('Loading data...')
        corpus, slices = load_corpus(INPUT, EMBEDDING, PREPROCESS, LANGUAGE)
        model = load_model(PROJECT, EMBEDDING, ALGO, K)

        print('Providing interpretation...')
        if type(model).__name__ == 'LogisticRegression':

            dictionary = Dictionary(corpus)
            dictionary.filter_extremes(no_below=2, no_above=0.5)

            for i, coef in enumerate(model.coef_):
                imp = np.argsort(coef)[::-1]

                # Interpretation BoW
                if EMBEDDING == 'BOW':
                    dictionary.filter_extremes(keep_n=K)
                    interp = interpret_bow(dictionary, imp)
                    filename = './results/{}/interpretation/{}_interpretation_{}_class_{}.csv'.format(PROJECT, EMBEDDING, K, i)
                    interp.to_csv(filename, encoding='utf-8')

                # Interpretation LDA
                elif EMBEDDING == 'LDA':
                    filename = './results/{}/models/{}_model_{}.model'.format(PROJECT, EMBEDDING, K)
                    mod = LdaMulticore.load(filename)
                    interp = interpret_lda(dictionary, mod, imp, coef)
                    filename = './results/{}/interpretation/{}_interpretation_{}_class_{}.csv'.format(PROJECT, EMBEDDING, K, i)
                    interp.to_csv(filename, encoding='utf-8')

                else:
                    print('Cannot provide interpretation for embeddings other than BOW and LDA for now.')
                    sys.exit(1)
        else:
            print("Cannot provide interpretation for classifier other than logistic regression (LOGIT).")
            sys.exit(1)

        print('Interpretation done!')
