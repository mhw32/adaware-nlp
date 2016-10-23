""" GRU for lemmatization. State-of-the-art lemmatization and
    stemming seems to work through large dictionary lookups and
    hardcoded chopping. Instead we can train a character-level
    NN to map a word to its own lemma.

    There will be a feature per character (padded to some max).
    The POS should also be a feature as it provides info.

"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import cPickle

sys.path.append('../common')
from util import batch_index_generator, split_data

sys.path.append('../models')
import nn

# to generate a training dataset
import numpy as np
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


def gen_dataset(sentences, train_test_split=True, max_size=25):
    ''' Generate a dataset of (input, output) pairs where the
        input is a vector of characters + POS and output is
        a vector of characters for the lemmatized form.

        Args
        ----
        sentences : list of sentences where each sentence is list of tokens
    '''

    lemmatizer = WordNetLemmatizer()
    X, P, y = [], [], []
    num_sentences = len(sentences)

    for sent_i, words in enumerate(sentences):
        if sent_i % 1000 == 0:
            print("{} sentences parsed. {} remaining.".format(sent_i, num_sentences - sent_i - 1))
        raw_pos = [p[1]for p in pos_tag(words)]
        pos = [str(treebank_to_simple(p, default=wordnet.NOUN)) for p in raw_pos]
        lemmas = [str(lemmatizer.lemmatize(w, pos=p)) for (w,p) in zip(words, pos)]

        X.extend(words)
        P.extend(raw_pos)
        y.extend(lemmas)

    pos_set = np.unique(P)
    char_set = list(set(' '.join(X)))
    pos_to_ix = { po:i for i,po in enumerate(pos_set) }
    char_to_ix = { ch:i for i,ch in enumerate(char_set) }
    word_to_ixs = lambda w: [char_to_ix[l] for l in w]

    word_char_arr = np.zeros((len(X), max_size + 1))
    lemma_char_arr = np.zeros((len(y), max_size))

    for i, (word, pos, lemma) in enumerate(zip(X, P, y)):
        word_char_arr[i, :max_size] = pad_array(word_to_ixs(word), max_size)
        word_char_arr[i, max_size] = pos_to_ix[pos]
        lemma_char_arr[i, :] = pad_array(word_to_ixs(lemma), max_size)

    if train_test_split:
        (X_train, X_test), (y_train, y_test) = split_data(
            word_char_arr, out_data=lemma_char_arr, frac=0.80)

        return (X_train, X_test), (y_train, y_test), char_set, pos_set, max_size

    return (word_char_arr, lemma_char_arr), char_set, pos_set, max_size


def train_lemmatizer(
    obs_set,
    out_set,
    num_hiddens,
    batch_size=256,
    param_scale=0.01,
    num_epochs=1000,
    step_size=0.001,
    L2_reg=0.1
):
    ''' function to train the Bi-LSTM for mapping vectorized
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
        num_hiddens : list of integers
                      number of hidden nodes
                      i.e. [30, 50, 1]
        batch_size : integer
                     size of batch in learning
        param_scale : float
                      size of weights if none
        num_epochs : int
                     number of epochs to train
        step_size : float
                    initial step size
        L2_reg : float
                 regularization constant
    '''

    trained_weights = nn.train_nn(obs_set,
                                  out_set,
                                  num_hiddens,
                                  batch_size=batch_size,
                                  param_scale=param_scale,
                                  num_epochs=num_epochs,
                                  step_size=step_size,
                                  L2_reg=L2_reg)

    return trained_weights


class NeuralLemmatizer(object):
    ''' Dummy class as a wrapper to easy load the weights and use
        them with one call. Must have a trained nn lemmatizer already.
    '''
    def __init__(self,
                 weights_loc,
                 char_set_loc,
                 pos_set_loc,
                 max_size):
        self.max_size = max_size
        self.weights = np.load(weights_loc)
        with open(char_set_loc) as fp:
            self.char_set = cPickle.load(fp)
        with open(pos_set_loc) as fp:
            self.pos_set = cPickle.load(fp)

    def lemmatize(self, word, pos='NN'):
        pos_to_ix = { po:i for i,po in enumerate(pos_set) }
        char_to_ix = { ch:i for i,ch in enumerate(char_set) }
        ix_to_char = { i:ch for i,ch in enumerate(char_set) }
        word_to_ixs = lambda w: [char_to_ix[l] for l in w]

        word_input = pad_array(word_to_ixs(word), self.max_size)
        pos_input = pos_to_ix[pos]

        word_input = np.concatenate((word_input, pos_input))
        lemma_output = nn.neural_net_predict(self.weights, word_input)

        return ''.join([ix_to_char[i] for i in lemma_output if i > 0])
