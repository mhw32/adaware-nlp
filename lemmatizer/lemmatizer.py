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
from util import batch_index_generator, split_data

sys.path.append('../models')
import nn_regressor

# to generate a training dataset
import numpy as np
from gensim import models
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


def treebank_to_simple(penn_tag, default=None):
    morphy_tag = {'NN':wordnet.NOUN,
                  'JJ':wordnet.ADJ,
                  'VB':wordnet.VERB,
                  'RB':wordnet.ADV}
    penn_pre = penn_tag[:2]
    if penn_pre in morphy_tag:
        return morphy_tag[penn_pre]
    return default


def pad_array(array, max_size):
    a = np.zeros(max_size)
    if len(array) > max_size:
        a = array[:max_size]
    else:
        a[:len(array)] = array
    return a


def prepare_sentence(words,
                     char_dict,
                     pos_dict,
                     lemmatizer=None,
                     max_chars=25,
                     max_words=78,
                     return_output=True):
    X = np.zeros((max_words, max_chars+1))
    if return_output:
        y = np.zeros((max_words, 300))

    raw_pos = [p[1]for p in pos_tag(words)]
    pos     = [str(treebank_to_simple(p, default=wordnet.NOUN)) for p in raw_pos]
    if return_output:
        lemmas  = [str(lemmatizer(w, pos=p)) for (w,p) in zip(words, pos)]

    num_words = len(words) if len(words) <= max_words else max_words

    for word_i in range(num_words):
        X[word_i, :max_chars] = ascii_vectorizer(words[word_i], char_dict)
        X[word_i, -1] = pos_dict[raw_pos[word_i]]
        if return_output:
            y[word_i, :] = ascii_vectorizer(lemmas[word_i], char_dict)

    if return_output:
        return X, y
    return X


def ascii_vectorizer(s, char_dict):
    return np.array([char_dict[ch] for ch in s])


def ascii_devectorizer(x, rev_char_dict):
    return ''.join([rev_char_dict[n]for n in x])


def gen_dataset(sentences,
                train_test_split=True,
                max_words=78,
                max_chars=25):
    ''' Generate a dataset of (input, output) pairs where the
        input is a vector of characters + POS and output is
        a vector of characters for the lemmatized form.

        Args
        ----
        sentences : list of sentences where each sentence is list of tokens
        max_words : maximum number of words allowed in sentence
    '''

    num_sentences = len(sentences)

    # define a list of all possible chars
    char_set = set()
    for sentence in sentences:
        char_set = char_set.union(set(''.join(sentence)))
    char_dict = dict(zip(list(char_set), range(len(char_set))))

    # define all the POS
    with open('../storage/one_hot_list') as f:
        pos_list = cPickle.load(f)
        pos_dict = {}
        for i, pos in enumerate(pos_list):
            pos_dict[pos] = i

    lemmatizer = WordNetLemmatizer().lemmatize
    X = np.zeros((num_sentences, max_words, max_chars+1))
    y = np.zeros((num_sentences, max_words, max_chars))

    param_dict = {}
    param_dict['max_words'] = max_words
    param_dict['char_dict'] = char_dict
    param_dict['pos_dict'] = pos_dict

    for sent_i, words in enumerate(sentences):
        if sent_i % 1000 == 0:
            print("{} sentences parsed. {} remaining.".format(
                sent_i, num_sentences - sent_i - 1))

        X[sent_i, :, :], y[sent_i, :, :] = prepare_sentence(
            words, char_dict, pos_dict, lemmatizer,
            max_words=max_words, max_chars=max_chars)

    if train_test_split:
        (X_train, X_test), (y_train, y_test) = split_data(
            X, out_data=y, frac=0.80)

        return (X_train, X_test), (y_train, y_test), param_dict
    return (X, y), param_dict


def window_featurizer(X, y=None, size=[1,1]):
    ''' Given some time series of data, it might be a good idea
        to include some temporal information by adding neighboring
        vectors.

        Args
        ----
        X : 2D numpy
            inputs matrix
        y : 2D numpy
            outputs matrix
        size : list of 2
               first is number prior, second is number after
    '''

    if sum(size) <= 0:
        return (X, y) if not y is None else X

    window_X = np.zeros((X.shape[0], X.shape[1]*(sum(size)+1)))
    if not y is None:
        window_y = np.zeros((y.shape[0], y.shape[1]))

    # prepend + postpend with 0's
    X = np.vstack((np.zeros((size[0], X.shape[1])), X, np.zeros((size[1], X.shape[1]))))

    for i in range(size[0],X.shape[0]-size[1]-1):
        for j,k in enumerate(range(i-size[0],i+size[1]+1)):
            window_X[i-size[0], j*X.shape[1]:(j+1)*X.shape[1]] = X[k, :]
        if not y is None:
            window_y[i-size[0], :] = y[i, :]

    return (window_X, window_y) if not y is None else window_X


def train_lemmatizer(
    obs_set,
    out_set,
    num_hiddens,
    window_size=[1,1],
    batch_size=256,
    param_scale=0.01,
    num_epochs=250,
    step_size=0.001
):
    ''' function to train the NN for mapping vectorized
        characters + POS --> a vectorized lemma

        Args
        ----
        X_train : np array
                  created by gen_dataset
        y_train : np.array
                  created by gen_dataset
        X_test : np.array
                 created by gen_dataset
        y_test : np.array
                 created by gen_dataset
        num_hiddens : integer
                      LSTM hidden nodes
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
    param_set['num_hiddens'] = num_hiddens
    param_set['window_size'] = window_size
    param_set['batch_size'] = batch_size
    param_set['param_scale'] = param_scale
    param_set['num_epochs'] = num_epochs
    param_set['step_size'] = step_size

    obs_set = obs_set.reshape(-1, obs_set.shape[-1])
    out_set = out_set.reshape(-1, out_set.shape[-1])

    obs_set, out_set = window_featurizer(obs_set, y=out_set, size=window_size)

    pred_fun, loglike_fun, trained_weights = \
        nn_regressor.train_nn_regressor(obs_set,
                                        out_set,
                                        [1000, 500],
                                        batch_size=batch_size,
                                        param_scale=param_scale,
                                        num_epochs=num_epochs,
                                        step_size=step_size)

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

        with open(nn_param_set_loc) as fp:
            nn_param_set = dill.load(fp)
            self.pred_fun = nn_param_set['pred_fun']
            self.loglike_fun = nn_param_set['loglike_fun']
            self.window_size = nn_param_set['window_size']
            self.weights = nn_param_set['trained_weights']

        with open(gen_param_set_loc) as fp:
            gen_param_set = cPickle.load(fp)
            self.max_words = gen_param_set['max_words']
            self.max_chars = gen_param_set['max_chars']
            self.pos_dict = gen_param_set['pos_dict']
            self.char_dict = gen_param_set['char_dict']
            self.rev_char_dict = {v: k for k, v in self.char_dict.items()}

    def lemmatize(self, sentence):
        X = prepare_sentence(sentence,
                             self.char_dict,
                             self.pos_dict,
                             max_chars=self.max_chars,
                             max_words=self.max_words,
                             return_output=False)

        X = window_featurizer(X, size=self.window_size)
        y = self.pred_fun(self.weights, X)
        return [ascii_devectorizer(word, self.rev_char_dict) for word in y]
