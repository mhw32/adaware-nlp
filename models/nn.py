from __future__ import absolute_import
from __future__ import print_function
from builtins import range

from copy import copy
import sys

sys.path.append('../common')
import util

# load in libraries for NN
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad, value_and_grad
from autograd.util import flatten
from optimizers import adam


'''
Neural network setup to map tokens into sentence
or non-sentence endings (binary classification)

'''


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """
    Build a list of (weights, biases) tuples,
    one for each layer in the net.
    """
    return [(scale * rs.randn(m, n),   # weight matrix
             scale * rs.randn(n))      # bias vector
            for m, n in zip(layer_sizes[:-1], layer_sizes[1:])]


def neural_net_predict(params, inputs):
    """
    Implements a deep neural network for classification.
    params is a list of (weights, bias) tuples.
    inputs is an (N x D) matrix.
    returns normalized class log-probabilities.
    """
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = sigmoid(outputs)
    return outputs - logsumexp(outputs, axis=1, keepdims=True)


def l2_norm(params):
    """
    Computes l2 norm of params by flattening them into a vector.
    """
    flattened, _ = flatten(params)
    return np.dot(flattened, flattened)


def log_posterior(params, inputs, targets, L2_reg):
    log_prior = -L2_reg * l2_norm(params)
    log_lik = np.sum(neural_net_predict(params, inputs) * targets)
    return log_prior + log_lik


def accuracy(params, inputs, targets):
    target_class = np.argmax(targets, axis=1)
    predicted_class = np.argmax(neural_net_predict(params, inputs), axis=1)
    return np.mean(predicted_class == target_class)


def train_nn(
        inputs, outputs, num_hiddens,  # don't include inputs and outputs
        batch_size=256, param_scale=0.1,
        num_epochs=5, step_size=0.001, L2_reg=1.0):

    # split data (again) into a training and a validation set
    (tr_inputs, va_inputs), (tr_outputs, va_outputs) = util.split_data(
        inputs, out_data=outputs, frac=0.80)

    num_input_dims = tr_inputs.shape[1]
    num_output_dims = tr_outputs.shape[1]
    layer_sizes = [num_input_dims] + num_hiddens + [num_output_dims]
    init_params = init_random_params(param_scale, layer_sizes)
    num_batches = int(np.ceil(tr_inputs.shape[0] / batch_size))

    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # Define training objective
    def objective(params, iter):
        idx = batch_indices(iter)
        return -log_posterior(
            params, tr_inputs[idx], tr_outputs[idx], L2_reg)

    # Get gradient of objective using autograd.
    objective_grad = grad(objective)

    print(
        "     Epoch     |    Train accuracy  |    Train log-like  |  Holdout accuracy  |  Holdout log-like  ")

    def print_perf(params, iter, gradient):
        if iter % num_batches == 0:
            train_acc = accuracy(params, tr_inputs, tr_outputs)
            train_ll = log_posterior(params, tr_inputs, tr_outputs, L2_reg)
            valid_acc = accuracy(params, va_inputs, va_outputs)
            valid_ll = log_posterior(params, va_inputs, va_outputs, L2_reg)
            print("{:15}|{:20}|{:20}|{:20}|{:20}".format(
                iter//num_batches, train_acc, train_ll, valid_acc, valid_ll))

    # The optimizers provided can optimize lists, tuples, or dicts of
    # parameters.
    optimized_params = adam(
        objective_grad, init_params, step_size=step_size,
        num_iters=num_epochs * num_batches, callback=print_perf)

    return optimized_params
