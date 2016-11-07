from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import dill
import cPickle

sys.path.append('../common')
from util import batch_index_generator, split_data
import activation
import cosine_mlp, thin_cosine_mlp

# to generate a training dataset
import numpy as np
from gensim import models
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


def read_data_from_file(file):
    with open(file) as fp:
        data = fp.read()
    data = data.split('\r\n')
    data = [i for i in data if len(i) > 0]

    num_data = len(data)
    model = models.Word2Vec.load_word2vec_format(
        '../storage/pos_tagger/GoogleNews-vectors-negative300.bin',
        binary=True)
    vectorizer = lambda x: model[x] if x in model else np.zeros(300)

    X = np.zeros((num_data, 300))
    y = np.zeros((num_data, 300))

    all_words = []
    for i in range(len(data)):
        lemma, word = data[i].split('\t')
        all_words.append(word)
        word_vec = vectorizer(word)
        lemma_vec = vectorizer(lemma)

        X[i, :] = word_vec
        y[i, :] = lemma_vec

    word_pos = [p[1]for p in pos_tag(all_words)]
    pos_set = list(set(word_pos))
    pos_encoder = one_hot_encoding(pos_set)

    P = np.zeros((num_data, len(pos_set)))
    for i in range(len(data)):
        P[i, :] = pos_encoder[word_pos[i]]

    return X, y, P


def one_hot_encoding(pos_set):
    num_uniq_pos = len(pos_set)
    d = dict()
    for i,pos in enumerate(pos_set):
        d[pos] = np.zeros(num_uniq_pos)
        d[pos][i] = 1

    return d


def train_thin_lemmatizer(
    inputs,
    outputs,
    postags=None,
    batch_size=256,
    param_scale=0.01,
    num_epochs=250,
    step_size=0.001,
    l1_lambda=0,
    l2_lambda=0
):

    if not postags is None:
        inputs = np.hstack((inputs, postags))

    ''' use a thinly-connected NN '''
    pred_fun, loglike_fun, trained_weights = \
        thin_cosine_mlp.train_mlp(inputs,
                                  outputs,
                                  batch_size=batch_size,
                                  param_scale=param_scale,
                                  num_epochs=num_epochs,
                                  step_size=step_size,
                                  l1_lambda=l1_lambda,
                                  l2_lambda=l2_lambda,
                                  nonlinearity=activation.identity)

    return pred_fun, loglike_fun, trained_weights


def train_fat_lemmatizer(
    inputs,
    outputs,
    postags=None,
    batch_size=256,
    param_scale=0.01,
    num_epochs=250,
    step_size=0.001,
    l1_lambda=0,
    l2_lambda=0
):

    if not postags is None:
        print(inputs.shape, postags.shape)
        inputs = np.hstack((inputs, postags))

    ''' Use a fully-connected NN '''
    pred_fun, loglike_fun, trained_weights = \
        cosine_mlp.train_mlp(inputs,
                             outputs,
                             [1200, 1200],
                             batch_size=batch_size,
                             param_scale=param_scale,
                             num_epochs=num_epochs,
                             step_size=step_size,
                             l1_lambda=l1_lambda,
                             l2_lambda=l2_lambda,
                             nonlinearity=activation.sigmoid)

    return pred_fun, loglike_fun, trained_weights
