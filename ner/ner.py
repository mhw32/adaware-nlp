''' Name Entity Recognition

    Model: Convolutional Neural Network
        - Conv --> Pool --> Conv --> Pool --> FNN
        - input: n-gram*300
        - conv(1): 4 kernels (2*41 size)
        - pool(1): horiz. and vert pooling dim = 2
        - conv(2): 8 kernels (1*21 size)
        - pool(2): horiz = 2, vert = 1
        - 256 HU w/ 0.5 proba dropout

    Heuristics:
    1. Encoding: Use Word2Vec/GloVe
    2. N-gram
        Train 3,5,7,9 and use a mixture of
        experts when doing predictions

        Bayesian Hierarchical Mixtures of Experts
    3. Possible data augmentation:
        Replace words with synonyms

    Training dataset: look in the drive for a
    news_tagged_data.txt.
'''

import os
import sys
local_ref = lambda x: os.path.join(os.path.dirname(__file__),  x)
sys.path.append(local_ref('../common'))
sys.path.append(local_ref('../models'))

import util
import cPickle
import numpy as np
import featurizers
from gensim import models
import cnn
import pdb

def read_wordvec_from_file(in_file, out_file):
    with open(in_file) as fp:
        raw_data = fp.read()
    data = raw_data.split('\n')
    num_data = len(data)

    model = {}
    for i in range(num_data):
        cur_data = data[i].split('\t')[:-1]
        key = cur_data[0]
        val = np.array(cur_data[1:]).astype(np.float64)
        model[key] = val

    with open(out_file, 'w') as fp:
        cPickle.dump(model, fp)


def read_dataset_from_file(file):
    with open(file) as fp:
        raw_data = fp.read()
    data = raw_data.split('\n\n')[:-1]
    num_data = len(data)

    text = []
    category = []

    for i in range(num_data):
        if i % 100 == 0:
            print('processed {} lines'.format(i))
        line = data[i].split('\n')
        num_words = len(line)

        cur_text, cur_category = [], []
        for j in range(num_words):
            t, c = line[j].split('\t')
            cur_text.append(t)
            cur_category.append(c)

        text.append(cur_text)
        category.append(cur_category)

    return text, category


def prepare_sentence(words,
                     categories=None,
                     vectorizer=None,
                     encoder=None,
                     max_words=78):
    num_keys = len(encoder.keys())
    X = np.zeros((max_words, 300))
    if not categories is None:
        y = np.zeros((max_words, num_keys))
    num_words = len(words) if len(words) <= max_words else max_words

    for word_i in range(max_words):
        if word_i < num_words:
            word_vector = vectorizer(words[word_i])
            X[word_i, :] = word_vector
            if not categories is None:
                y[word_i, encoder[categories[word_i]]] = 1
        else:
            if not categories is None:
                y[word_i, encoder['O']] = 1  # default

    if not categories is None:
        return X, y
    return X


def one_hot_encoding(categories):
    uniq = list(set(np.concatenate(categories)))
    encoding = dict(zip(uniq, range(len(uniq))))
    return encoding


def gen_dataset(sentences,
                categories,
                max_words=78,
                train_test_split=True):
    ''' Generate a dataset of (input, output) pairs where the
        input is an embedded vector and output the category (one-hotted)

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
        local_ref('../storage/pos_tagger/GoogleNews-vectors-negative300.bin'),
        binary=True)
    vectorizer = lambda x: model[x] if x in model else np.zeros(300)
    encoder = one_hot_encoding(categories)

    X = np.zeros((num_sentences, max_words, 300))
    y = np.zeros((num_sentences, max_words, len(encoder.keys())))
    K = np.zeros(num_sentences)
    I = np.arange(num_sentences)

    param_dict = {}
    param_dict['max_words'] = max_words
    param_dict['encoder'] = encoder

    for sent_i in I:
        words = sentences[sent_i]
        cats = categories[sent_i]

        if sent_i % 1000 == 0:
            print("{} sentences parsed. {} remaining.".format(
                sent_i, num_sentences - sent_i - 1))

        X[sent_i, :, :], y[sent_i, :, :] = \
            prepare_sentence(words, categories=cats,
                                    vectorizer=vectorizer,
                                    encoder=encoder,
                                    max_words=max_words)

        K[sent_i] = len(words)  # keep track of num words in sentence

    if train_test_split:
        (X_train, X_test), (I_train, I_test) = util.split_data(
            X, out_data=I, frac=0.80)
        y_train, y_test = y[I_train], y[I_test]
        K_train, K_test = K[I_train], K[I_test]

        return (X_train, X_test), (y_train, y_test), (K_train, K_test), param_dict
    return (X, y, K), param_dict


def train_ner(
    obs_set,
    out_set,
    count_set,
    window_size=[1,1],
    batch_size=256,
    param_scale=0.01,
    num_epochs=250,
    step_size=0.001,
    l2_lambda=0
):
    ''' function to train the NN for word vectors to
        NER category.

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
        obs_slice = obs_set[sent_i, :, :][:int(count_set[sent_i])]
        out_slice = out_set[sent_i, :, :][:int(count_set[sent_i])]
        obs_window = featurizers.window_featurizer(obs_slice, size=window_size)
        obs_window = obs_window.reshape(obs_slice.shape[0], sum(window_size)+1, obs_slice.shape[-1])

        obs_lst.append(obs_window)
        out_lst.append(out_slice)

    # flatten vectors
    inputs = np.concatenate(obs_lst)
    inputs = np.expand_dims(inputs, axis=1)
    outputs = np.concatenate(out_lst)

    layer_specs = [cnn.conv_layer((2, 41), 4),
                   cnn.maxpool_layer((2, 2)),
                   cnn.conv_layer((1, 21), 8),
                   cnn.maxpool_layer((1, 2)),
                   cnn.tanh_layer(256),
                   cnn.softmax_layer(9)]

    pred_fun, loglike_fun, trained_weights = \
        cnn.train_cnn(inputs,
                      outputs,
                      layer_specs,
                      batch_size=batch_size,
                      param_scale=param_scale,
                      num_epochs=num_epochs,
                      L2_reg=l2_lambda)

    param_set['pred_fun'] = pred_fun
    param_set['loglike_fun'] = loglike_fun
    param_set['trained_weights'] = trained_weights

    return param_set


class NeuralNER(object):
    ''' Dummy class as a wrapper to easy load the weights and use
        them with one call. Must have a trained cnn already. '''
    def __init__(self,
                 gen_param_set,
                 nn_param_set,
                 wordvec_embedder):

        self.pred_fun = nn_param_set['pred_fun']
        self.loglike_fun = nn_param_set['loglike_fun']
        self.window_size = nn_param_set['window_size']
        self.weights = nn_param_set['trained_weights']

        self.max_words = gen_param_set['max_words']
        self.encoder = gen_param_set['encoder']
        self.reverse_encoder = {v: k for k, v in self.encoder.iteritems()}
        self.embedder = wordvec_embedder

        self.vectorizer = lambda x: self.embedder[x] \
            if x in self.embedder else np.zeros(300)

    def ner(self, sentence):
        X = prepare_sentence(sentence,
                             vectorizer=self.vectorizer,
                             encoder=self.encoder,
                             max_words=self.max_words)
        X = X[:len(sentence)]
        X_f = featurizers.window_featurizer(X, size=self.window_size)
        X = X_f.reshape(X.shape[0], sum(self.window_size)+1, X.shape[-1])
        X = X[:, np.newaxis, ...]

        layer_specs = [cnn.conv_layer((2, 41), 4),
                       cnn.maxpool_layer((2, 2)),
                       cnn.conv_layer((1, 21), 8),
                       cnn.maxpool_layer((1, 2)),
                       cnn.tanh_layer(256),
                       cnn.softmax_layer(9)]
        L2_reg = 0
        num_weights, pred_fun, loss_fun, frac_err = cnn.build(X.shape[1:], layer_specs, L2_reg)
        logproba = pred_fun(self.weights, X)
        return [self.reverse_encoder[i] for i in np.argmax(logproba, axis=1)]

