from __future__ import absolute_import
from __future__ import print_function

import sys
from builtins import range
from os.path import dirname, join

sys.path.append('../common')
import util

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad
from autograd.scipy.misc import logsumexp
from optimizers import adam


def sigmoid(x):
    return 0.5*(np.tanh(x) + 1.0)   # Output ranges from 0 to 1.


def concat_and_multiply(weights, *args):
    cat_state = np.hstack(args + (np.ones((args[0].shape[0], 1)),))
    return np.dot(cat_state, weights)


def init_lstm_params(input_size, state_size, output_size,
                     param_scale=0.01, rs=npr.RandomState(0)):
    def rp(*shape):
        return rs.randn(*shape) * param_scale

    return {'init cells':   rp(1, state_size),
            'init hiddens': rp(1, state_size),
            'change':       rp(input_size + state_size + 1, state_size),
            'forget':       rp(input_size + state_size + 1, state_size),
            'ingate':       rp(input_size + state_size + 1, state_size),
            'outgate':      rp(input_size + state_size + 1, state_size),
            'predict':      rp(state_size + 1, output_size)}


def lstm_predict(params, inputs):
    def update_lstm(input, hiddens, cells):
        change  = np.tanh(concat_and_multiply(params['change'], input, hiddens))
        forget  = sigmoid(concat_and_multiply(params['forget'], input, hiddens))
        ingate  = sigmoid(concat_and_multiply(params['ingate'], input, hiddens))
        outgate = sigmoid(concat_and_multiply(params['outgate'], input, hiddens))
        cells   = cells * forget + ingate * change
        hiddens = outgate * np.tanh(cells)
        return hiddens, cells

    def hiddens_to_output_probs(hiddens):
        output = concat_and_multiply(params['predict'], hiddens)
        return output - logsumexp(output, axis=1, keepdims=True) # Normalize log-probs.

    num_sequences = inputs.shape[1]
    hiddens = np.repeat(params['init hiddens'], num_sequences, axis=0)
    cells   = np.repeat(params['init cells'],   num_sequences, axis=0)

    output = []
    for input_i, input in enumerate(inputs):  # Iterate over time steps.
        hiddens, cells = update_lstm(input, hiddens, cells)
        _output = hiddens_to_output_probs(hiddens)
        output.append(_output)
    return np.array(output)


def lstm_log_likelihood(params, inputs, targets):
    logprobs = lstm_predict(params, inputs)
    loglik = 0.0
    num_time_steps, num_examples, _ = inputs.shape
    for t in range(num_time_steps):
        loglik += np.sum(logprobs[t] * targets[t])
    return loglik / (num_time_steps * num_examples)


def accuracy(params, inputs, targets):
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(lstm_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)


def train_lstm(inputs,
               outputs,
               state_size,
               batch_size=256,
               param_scale=0.001,
               num_epochs=5,
               step_size=0.001):

    # split data (again) into a training and a validation set
    (tr_inputs, va_inputs), (tr_outputs, va_outputs) = util.split_data(
        inputs, out_data=outputs, frac=0.80)

    input_size = tr_inputs.shape[2]
    output_size = tr_outputs.shape[2]

    init_params = init_lstm_params(input_size,
                                   state_size,
                                   output_size,
                                   param_scale=param_scale,
                                   rs=npr.RandomState(0))

    num_batches = int(np.ceil(tr_inputs.shape[1] / batch_size))

    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # Define training objective
    def objective(params, iter):
        idx = batch_indices(iter)
        return -lstm_log_likelihood(
            params, tr_inputs[:, idx, :], tr_outputs[:, idx, :])

    # Get gradient of objective using autograd.
    objective_grad = grad(objective)

    print(
        "     Epoch     |    Train accuracy  |    Train log-like  |  Holdout accuracy  |  Holdout log-like  ")

    def print_perf(params, iter, gradient):
        train_acc = accuracy(params, tr_inputs, tr_outputs)
        train_ll = -lstm_log_likelihood(params, tr_inputs, tr_outputs)
        valid_acc = accuracy(params, va_inputs, va_outputs)
        valid_ll = -lstm_log_likelihood(params, va_inputs, va_outputs)
        print("{:15}|{:20}|{:20}|{:20}|{:20}".format(
            iter//num_batches, train_acc, train_ll, valid_acc, valid_ll))

    # The optimizers provided can optimize lists, tuples, or dicts of parameters.
    optimized_params = adam(objective_grad,
                            init_params,
                            step_size=step_size,
                            num_iters=num_epochs,
                            callback=print_perf)

    return optimized_params
