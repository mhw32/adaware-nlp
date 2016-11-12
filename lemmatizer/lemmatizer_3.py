import sys
sys.path.append('../common')
import util
import cPickle
import numpy as np

sys.path.append('../common')
sys.path.append('../models')

import featurizers
from gensim import models
from nltk.stem import WordNetLemmatizer
import cnn_mlp

ZERO_EPSILON = 1e-5

def prepare_sentence(words,
                     vectorizer=None,
                     lemmatizer=None,
                     max_words=78):
    X = np.ones((max_words, 300)) * ZERO_EPSILON
    y = np.ones((max_words, 300)) * ZERO_EPSILON
    num_words = len(words) if len(words) <= max_words else max_words

    for word_i in range(max_words):
        if word_i < num_words:
            word_vector = vectorizer(words[word_i])
            X[word_i, :] = word_vector
            lemma_vector = vectorizer(lemmatizer(words[word_i]))
            y[word_i, :] = lemma_vector

    return X, y


def gen_dataset(sentences,
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
        '../storage/pos_tagger/GoogleNews-vectors-negative300.bin',
        binary=True)
    vectorizer = lambda x: model[x] if x in model else np.ones(300)*ZERO_EPSILON
    wordnet_lemmatizer = WordNetLemmatizer()
    lemmatizer = lambda x:  wordnet_lemmatizer.lemmatize(x)

    X = np.zeros((num_sentences, max_words, 300))
    y = np.zeros((num_sentences, max_words, 300))
    K = np.zeros(num_sentences)
    I = np.arange(num_sentences)

    param_dict = {}
    param_dict['max_words'] = max_words

    for sent_i in I:
        words = sentences[sent_i]

        if sent_i % 1000 == 0:
            print("{} sentences parsed. {} remaining.".format(
                sent_i, num_sentences - sent_i - 1))

        X[sent_i, :, :], y[sent_i, :, :] = \
            prepare_sentence(words, vectorizer=vectorizer,
                                    lemmatizer=lemmatizer,
                                    max_words=max_words)

        K[sent_i] = len(words)  # keep track of num words in sentence

    if train_test_split:
        (X_train, X_test), (I_train, I_test) = util.split_data(
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
    positive_samples_only=False,
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
    outputs = np.concatenate(out_lst)

    # keep only if lemma != word
    if positive_samples_only:
        _s_idx = []
        _s = window_size[0]
        for i in range(inputs.shape[0]):
            if np.sum(inputs[i, _s, :] - outputs[i, :]) != 0:
                _s_idx.append(i)

        _s_idx = np.array(_s_idx)
        inputs = inputs[_s_idx]
        outputs = outputs[_s_idx]

    inputs = np.expand_dims(inputs, axis=1)
    layer_specs = [cnn_mlp.conv_layer((2, 41), 4),
                   cnn_mlp.maxpool_layer((2, 2)),
                   # cnn_mlp.conv_layer((1, 21), 8),
                   # cnn_mlp.maxpool_layer((1, 2)),
                   cnn_mlp.identity_layer(300)]
    print(layer_specs)
    pred_fun, loglike_fun, trained_weights = \
        cnn_mlp.train_cnn_mlp(inputs,
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

