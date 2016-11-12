"""BLSTM for POS tagging"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import cPickle
# gensim and word2vec will be removed later
from gensim import models

sys.path.append('../common')
from util import batch_index_generator

sys.path.append('../models')
from blstm import init_blstm_params, blstm_predict, log_likelihood


def train_pos_tagger_blstm(
    X_train,
    Y_train,
    X_test,
    Y_test,
    cap_train,
    cap_test,
    train_mask,
    test_mask,
    num_hiddens=100,
    batch_size=15,
    L1_REG=1e-5,
    step_size=0.001,
    num_iters=5000,
    init_params=None,
    one_hot=None
):
    '''
        X_train and X_test are N x W x D arrays where N is the number of
        sentences, W is the max number of words in a sentence, and D is the
        vector representation for a word.

        For Y_train and Y_test the third dimension D is the one hot
        representation of the part of speech, which is described by the dict
        one_hot.

        For cap_train and cap_test the third dimension is the cap_vector.
        train_mask and test_mask are N dimensional vectors where each item i
        represents the number of words in sentence i
    '''
    if one_hot is None:
        with open('storage/pos_tagger/one_hot_list', 'rb') as f:
            one_hot = cPickle.load(f)

    X_train = np.swapaxes(X_train, 0, 1)
    Y_train = np.swapaxes(Y_train, 0, 1)
    X_test = np.swapaxes(X_test, 0, 1)
    Y_test = np.swapaxes(Y_test, 0, 1)
    cap_train = np.swapaxes(cap_train, 0, 1)
    cap_test = np.swapaxes(cap_test, 0, 1)

    index_generator = batch_index_generator(X_train.shape[1],
                                            batch_size=batch_size)
    if init_params is None:
        init_params = init_blstm_params(input_size=X_train.shape[2],
                                        output_size=Y_train.shape[2],
                                        state_size=num_hiddens,
                                        param_scale=0.05)

    def training_loss(params, iter):
        sample_indices = index_generator.next()
        log_lik = -log_likelihood(params,
                                  X_train[:, sample_indices, :],
                                  Y_train[:, sample_indices, :],
                                  train_mask[sample_indices],
                                  cap_train[:, sample_indices, :])
        return log_lik + L1_REG*l1_norm(params)

    trained_params = adam(training_loss_grad,
                          init_params,
                          step_size=step_size,
                          num_iters=num_iters)
    return trained_params


def text_to_vector(sentence_list, MAX_SENTENCE=78, model=None):
    if model is None:
        model = models.Word2Vec.load_word2vec_format(
            '../storage/pos_tagger/GoogleNews-vectors-negative300.bin', binary=True)
    X = np.zeros((MAX_SENTENCE, len(sentence_list), 300))
    capitals = np.zeros((MAX_SENTENCE, len(sentence_list), 3))
    vectorize = lambda x: model[x] if x in model else np.zeros(300)
    mask = []
    for i, sentence in enumerate(sentence_list):
        for j, word in enumerate(sentence):
            if j == MAX_SENTENCE:
                j -= 1
                break
            X[j][i] = vectorize(word)
            capitals[j][i] = cap_vector(word)
        mask.append(j + 1)
    mask = np.array(mask)
    return X, capitals, mask


def cap_vector(word):
    ''' Returns vector (x,y,z), wh ere x = 1 if word is all lowercase,
        y = 1 if all uppercase, z = 1 if leads with capital.
    '''
    x = int(word.lower() == word)
    y = int(word.upper() == word)
    z = int(word[0].upper() == word[0])
    return np.array((x, y, z))


def probability_to_pos(logprobs, mask):
    logprobs = np.array(logprobs)
    with open('../storage/pos_tagger/one_hot_list', 'rb') as f:
        one_hot = pickle.load(f)
    sentences = []
    for i in range(logprobs.shape[1]):
        sentence = []
        for j in range(mask[i]):
            sentence.append(one_hot[np.argmax(logprobs[:, i, :][j])])
        sentences.append(sentence)
    return sentences


def predict_from_sentences(sentence_list, params=None, model=None):
    if params is None:
        params = dict(np.load('storage/pos_tagger/pos_trained_weights.npz'))

    X, capitals, mask = text_to_vector(sentence_list, model=model)
    logprobs = np.array(blstm_predict(params, X, capitals))
    return probability_to_pos(logprobs, mask)
