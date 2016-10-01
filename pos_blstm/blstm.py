"""BLSTM for POS tagging"""

from __future__ import absolute_import
from __future__ import print_function
from builtins import range
from os.path import dirname, join
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.scipy.misc import logsumexp
import sys
from optimizers import batch_adam, adam
from rnn import string_to_one_hot, one_hot_to_string,\
                build_dataset, sigmoid, concat_and_multiply
import pdb
import cPickle as pickle
import utils

def init_lstm_params(input_size, state_size, output_size,
                     param_scale=0.01, rs=npr.RandomState(0)):
    def rp(*shape): # returns a np array sampled from normal distribution of specified shape
        return rs.randn(*shape) * param_scale

    if load:
        return dict(np.load('tmp/trained_weights.npz'))
    return {'capitals':     rp(3, input_size),
            'init cells_f':   rp(1, state_size),
            'init hiddens_f': rp(1, state_size),
            'change_f':       rp(input_size + state_size + 1, state_size),
            'forget_f':       rp(input_size + state_size + 1, state_size),
            'ingate_f':       rp(input_size + state_size + 1, state_size),
            'outgate_f':      rp(input_size + state_size + 1, state_size),
            'init cells_b':   rp(1, state_size),
            'init hiddens_b': rp(1, state_size),
            'change_b':       rp(input_size + state_size + 1, state_size),
            'forget_b':       rp(input_size + state_size + 1, state_size),
            'ingate_b':       rp(input_size + state_size + 1, state_size),
            'outgate_b':      rp(input_size + state_size + 1, state_size),
            'predict':      rp(state_size*2 + 1, output_size)}

def lstm_predict(params, inputs, capitals):
    def update_forward_lstm(_input, hiddens, cells):
        change  = np.tanh(concat_and_multiply(params['change_f'], _input, hiddens))
        forget  = sigmoid(concat_and_multiply(params['forget_f'], _input, hiddens))
        ingate  = sigmoid(concat_and_multiply(params['ingate_f'], _input, hiddens))
        outgate = sigmoid(concat_and_multiply(params['outgate_f'], _input, hiddens))
        cells   = cells * forget + ingate * change
        hiddens = outgate * np.tanh(cells)
        return hiddens, cells

    def update_backward_lstm(_input, hiddens, cells):
        change  = np.tanh(concat_and_multiply(params['change_b'], _input, hiddens))
        forget  = sigmoid(concat_and_multiply(params['forget_b'], _input, hiddens))
        ingate  = sigmoid(concat_and_multiply(params['ingate_b'], _input, hiddens))
        outgate = sigmoid(concat_and_multiply(params['outgate_b'], _input, hiddens))
        cells   = cells * forget + ingate * change
        hiddens = outgate * np.tanh(cells)
        return hiddens, cells

    def hiddens_to_output_probs(hiddens):
        output = concat_and_multiply(params['predict'], hiddens)
        return output - logsumexp(output, axis=1, keepdims=True) # Normalize log-probs.

    num_sequences = inputs.shape[1]
    hiddens1 = np.repeat(params['init hiddens_f'], num_sequences, axis=0)
    cells1   = np.repeat(params['init cells_f'],   num_sequences, axis=0)

    hiddens2 = np.repeat(params['init hiddens_b'], num_sequences, axis=0)
    cells2   = np.repeat(params['init cells_b'],   num_sequences, axis=0)
    output = []
    for i,_input in enumerate(inputs):  # Iterate over time steps.
        _input = _input + np.dot(capitals[i], params['capitals'])
        hiddens1, cells1 = update_forward_lstm(_input, hiddens1, cells1)
        hiddens2, cells2 = update_backward_lstm(_input[:,::-1], hiddens2, cells2)
        output.append(hiddens_to_output_probs(np.hstack((hiddens1,hiddens2))))
    return output

def lstm_log_likelihood(params, inputs, targets, mask, capitals):
    logprobs = lstm_predict(params, inputs, capitals)
    loglik = 0.0
    num_time_steps, num_examples, _ = inputs.shape
    for t in range(num_time_steps):
        loglik += np.sum(logprobs[t] * targets[t])
    return loglik / sum(mask)


if __name__ == '__main__':
    num_chars = 128
    load = 'load' in sys.argv[1:]
    save = 'save' in sys.argv[1:]
    with open('tmp/one_hot_list') as f:
        one_hot = pickle.load(f)
    path = 'tmp/'
    X_train = np.load(path + 'X_train.npy')
    X_train = np.swapaxes(X_train, 0, 1)
    Y_train = np.load(path + 'Y_train.npy')
    Y_train = np.swapaxes(Y_train, 0, 1)
    X_test = np.load(path + 'X_test.npy')
    X_test = np.swapaxes(X_test, 0, 1)
    Y_test = np.load(path + 'Y_test.npy')
    Y_test = np.swapaxes(Y_test, 0, 1)
    cap_train = np.load(path + 'cap_train.npy')
    cap_train = np.swapaxes(cap_train, 0, 1)
    cap_test = np.load(path + 'cap_test.npy')
    cap_test = np.swapaxes(cap_test, 0, 1)
    train_mask = np.load(path + 'train_mask.npy')
    test_mask = np.load(path + 'test_mask.npy')

    index_generator = utils.batch_index_generator(X_train.shape[1], batch_size=100)
    init_params = init_lstm_params(input_size=X_train.shape[2], output_size=Y_train.shape[2],
                                   state_size=100, param_scale=0.05)


    def print_training_prediction(weights):
        num_sentences = 20
        logprobs = lstm_predict(weights, X_train[:,:num_sentences,:], cap_train[:,:num_sentences,:])
        sentence = []
        correct = []
        for i in xrange(num_sentences):
            for j in xrange(train_mask[i]):
                sentence.append(one_hot[np.argmax(np.array(logprobs)[:,i,:][j])])
                correct.append(one_hot[np.argmax(Y_train[j,i,:])])
        print('train acc:',sum(np.array(sentence) == np.array(correct))/float(len(sentence)))


        logprobs = lstm_predict(weights, X_test[:,:num_sentences,:], cap_test[:,:num_sentences,:])
        sentence = []
        correct = []
        for i in xrange(num_sentences):
            for j in xrange(test_mask[i]):
                sentence.append(one_hot[np.argmax(np.exp(logprobs)[:,i,:][j])])
                correct.append(one_hot[np.argmax(Y_test[j,i,:])])
        print('valid acc:',sum(np.array(sentence) == np.array(correct))/float(len(sentence)))

    def training_loss(params, iter):
        # sample_indices = np.random.choice(X_train.shape[1], 200, replace=False)
        sample_indices = index_generator.next()
        return -lstm_log_likelihood(params, X_train[:,sample_indices,:], Y_train[:,sample_indices,:], train_mask[sample_indices], cap_train[:,sample_indices,:])

    def callback(weights, iter, gradient):
        if iter % 10 == 0:
            print("Iteration", iter, "Train loss:", training_loss(weights, 0))
            print_training_prediction(weights)
        if iter % 250 == 0 and save:
            np.savez('tmp/trained_weights.npz', **weights)

    # Build gradient of loss function using autograd.
    training_loss_grad = grad(training_loss)

    print("Training LSTM...")
    trained_params = adam(training_loss_grad, init_params, step_size=0.001, num_iters=1000, callback=callback)


    num_sentences = 200
    logprobs = lstm_predict(trained_params, X_train[:,:num_sentences,:], cap_train[:,:num_sentences,:])
    sentence = []
    correct = []
    for i in xrange(num_sentences):
        for j in xrange(train_mask[i]):
            sentence.append(one_hot[np.argmax(np.exp(logprobs)[:,i,:][j])])
            correct.append(one_hot[np.argmax(Y_train[j,i,:])])
    print(sum(np.array(sentence) == np.array(correct))/float(len(sentence)))

    logprobs = lstm_predict(trained_params, X_test[:,:num_sentences,:], cap_test[:,:num_sentences,:])
    sentence = []
    correct = []
    for i in xrange(num_sentences):
        for j in xrange(test_mask[i]):
            sentence.append(one_hot[np.argmax(np.exp(logprobs)[:,i,:][j])])
            correct.append(one_hot[np.argmax(Y_test[j,i,:])])
    print(sum(np.array(sentence) == np.array(correct))/float(len(sentence)))

    if save:
        np.savez('tmp/trained_weights.npz', **trained_params)