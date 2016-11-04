""" GRU for lemmatization. State-of-the-art lemmatization and
    stemming seems to work through large dictionary lookups and
    hardcoded chopping. Instead we can train a character-level
    NN to map a word to its own lemma.

    There will be a feature per character (padded to some max).
    The POS should also be a feature as it provides info.

"""

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import dill
import cPickle

sys.path.append('../common')
from util import batch_index_generator, split_data, devectorize, train_LSHForest
from featurizers import window_featurizer

import thin_cosine_mlp

# to generate a training dataset
import numpy as np
from gensim import models
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

ZERO_EPSILON=1e-7


def treebank_to_simple(penn_tag, default=None):
    morphy_tag = {'NN':wordnet.NOUN,
                  'JJ':wordnet.ADJ,
                  'VB':wordnet.VERB,
                  'RB':wordnet.ADV}
    penn_pre = penn_tag[:2]
    if penn_pre in morphy_tag:
        return morphy_tag[penn_pre]
    return default


def prepare_sentence(words,
                     vectorizer=None,
                     lemmatizer=None,
                     max_words=78,
                     return_output=True):
    X = np.ones((max_words, 300))*ZERO_EPSILON
    if return_output:
        y = np.ones((max_words, 300))*ZERO_EPSILON
        raw_pos = [p[1]for p in pos_tag(words)]
        pos     = [str(treebank_to_simple(p, default=wordnet.NOUN)) for p in raw_pos]
        lemmas  = [str(lemmatizer(w, pos=p)) for (w,p) in zip(words, pos)]

    num_words = len(words) if len(words) <= max_words else max_words

    for word_i in range(num_words):
        word_vector = vectorizer(words[word_i])
        X[word_i, :] = word_vector

        if return_output:
            lemma_vector = lemmas[word_i]
            y[word_i, :] = vectorizer(lemma_vector)

    if return_output:
        return X, y
    return X


def gen_dataset(sentences,
                max_words=78,
                train_test_split=True):
    ''' Generate a dataset of (input, output) pairs where the
        input is an embedded vector and output is
        an embedded vector for the lemmatized form.

        Args
        ----
        sentences : list
                    list of sentences where each sentence is list of tokens
        max_words : integer
                    maximum number of words allowed in sentence
        train_test_split : boolean
                           whether to split data into 2 sets
    '''

    num_sentences = len(sentences)
    model = models.Word2Vec.load_word2vec_format(
        '../storage/GoogleNews-vectors-negative300.bin',
        binary=True)
    vectorizer = lambda x: model[x] if x in model else np.ones(300)*ZERO_EPSILON
    lemmatizer = WordNetLemmatizer().lemmatize

    X = np.zeros((num_sentences, max_words, 300))
    y = np.zeros((num_sentences, max_words, 300))
    K = np.zeros(num_sentences)
    I = np.arange(num_sentences)

    param_dict = {}
    param_dict['max_words'] = max_words

    for sent_i, words in enumerate(sentences):
        if sent_i % 1000 == 0:
            print("{} sentences parsed. {} remaining.".format(
                sent_i, num_sentences - sent_i - 1))

        X[sent_i, :, :], y[sent_i, :, :] = \
            prepare_sentence(words,
                             vectorizer=vectorizer,
                             lemmatizer=lemmatizer,
                             max_words=max_words)

        K[sent_i] = len(words)  # keep track of num words in sentence

    if train_test_split:
        (X_train, X_test), (I_train, I_test) = split_data(
            X, out_data=I, frac=0.80)
        y_train, y_test = y[I_train], y[I_test]
        K_train, K_test = K[I_train], K[I_test]

        return (X_train, X_test), (y_train, y_test), (K_train, K_test), param_dict
    return (X, y, K), param_dict


def train_lemmatizer(
    obs_set,
    out_set,
    count_set,
    window_size=[1,1],
    include_identity=True,
    batch_size=256,
    param_scale=0.01,
    num_epochs=250,
    step_size=0.001,
    l2_lambda=0
):
    ''' function to train the NN for mapping vectorized
        characters + POS --> a vectorized lemma

        Args
        ----
        obs_set : np array
                  created by gen_dataset
        out_set : np.array
                  created by gen_dataset
        count_set : np.array
                    created by gen_dataset
        window_size : integer
                      group nearby vecvtors
        batch_size : integer
                     size of batch in learning
        param_scale : float
                      size of weights if none
        num_epochs : int
                     number of epochs to train
        step_size : float
                    initial step size
    '''

    param_set = {}
    param_set['window_size'] = window_size
    param_set['batch_size'] = batch_size
    param_set['param_scale'] = param_scale
    param_set['num_epochs'] = num_epochs
    param_set['step_size'] = step_size

    obs_lst, out_lst = [], []

    # loop through each sentence and window featurize it
    for sent_i in range(obs_set.shape[0]):
        obs_slice = obs_set[sent_i, :, :][:count_set[sent_i]]
        out_slice = out_set[sent_i, :, :][:count_set[sent_i]]
        obs_window = window_featurizer(obs_slice, size=window_size,
                                                  epsilon=ZERO_EPSILON)

        obs_lst.append(obs_window)
        out_lst.append(out_slice)

    # flatten vectors
    inputs = np.concatenate(obs_lst)
    outputs = np.concatenate(out_lst)

    if not include_identity:
        inputs_dim = inputs.shape[-1]
        outputs_dim = outputs.shape[-1]
        mid_idx = int(np.ceil(outputs_dim / float(inputs_dim)) + 1)
        axis_sums = np.sum(
            inputs[:,outputs_dim*(mid_idx-1):outputs_dim*mid_idx]-outputs, axis=1)
        axis_slice = np.where(axis_sums != 0)
        if axis_slice[0].size > 0:
           inputs = inputs[axis_slice[0], :]
           outputs = outputs[axis_slice[0], :]

    pred_fun, loglike_fun, trained_weights = \
        thin_cosine_mlp.train_mlp(inputs,
                                  outputs,
                                  batch_size=batch_size,
                                  param_scale=param_scale,
                                  num_epochs=num_epochs,
                                  l2_lambda=l2_lambda)

    param_set['pred_fun'] = pred_fun
    param_set['loglike_fun'] = loglike_fun
    param_set['trained_weights'] = trained_weights

    return param_set


class NeuralLemmatizer(object):
    ''' Dummy class as a wrapper to easy load the weights and use
        them with one call. Must have a trained nn lemmatizer already.
    '''
    def __init__(self,
                 gen_param_set_loc,
                 nn_param_set_loc):

        print('loading NN params')
        with open(nn_param_set_loc) as fp:
            nn_param_set = dill.load(fp)
            self.pred_fun = nn_param_set['pred_fun']
            self.loglike_fun = nn_param_set['loglike_fun']
            self.window_size = nn_param_set['window_size']
            self.weights = nn_param_set['trained_weights']

        print('loading GEN params')
        with open(gen_param_set_loc) as fp:
            gen_param_set = cPickle.load(fp)
            self.max_words = gen_param_set['max_words']

        print('loading Word2Vec model')
        self.model = models.Word2Vec.load_word2vec_format(
            '../storage/GoogleNews-vectors-negative300.bin',
            binary=True)

        self.vectorizer = lambda x: self.model[x] \
            if x in self.model else np.ones(300)*ZERO_EPSILON

        print('training LSH Forest model')
        self.lsh_forest = train_LSHForest(self.model)


    def lemmatize(self, sentence):
        X = prepare_sentence(sentence,
                             vectorizer=self.vectorizer,
                             max_words=self.max_words,
                             return_output=False)
        X = X[:len(sentence)]
        X = window_featurizer(X, size=self.window_size)
        y = self.pred_fun(self.weights, X)
        return devectorize(y, self.lsh_forest)

