from __future__ import absolute_import
from __future__ import print_function

from copy import copy
import sys

sys.path.append('../common')
import util

# load in libraries for NN regressor
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd import grad

from optimizers import adam
from sklearn.metrics import mean_squared_error
from math import sqrt

# different activation functions
rbf  = lambda x: np.exp(-x**2)
relu = lambda x: np.maximum(x, 0.0)


def rms(preds, targets):
    return sqrt(mean_squared_error(targets, preds))


def build(layer_sizes,
          weight_scale=10.0,
          noise_scale=0.1,
          nonlinearity=np.tanh):
    """These functions implement a standard multi-layer perceptron."""

    shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    num_weights = sum((m+1)*n for m, n in shapes)

    def unpack_layers(weights):
        for m, n in shapes:
            cur_layer_weights = weights[:m*n]     .reshape((m, n))
            cur_layer_biases  = weights[m*n:m*n+n].reshape((1, n))
            yield cur_layer_weights, cur_layer_biases
            weights = weights[(m+1)*n:]

    def predictions(weights, inputs):
        for W, b in unpack_layers(weights):
            outputs = np.dot(inputs, W) + b
            inputs = nonlinearity(outputs)
        return outputs

    def logprob(weights, inputs, targets):
        log_prior = np.sum(norm.logpdf(weights, 0, weight_scale))
        preds = predictions(weights, inputs)
        log_lik = np.sum(norm.logpdf(preds, targets, noise_scale))
        return log_prior + log_lik

    return num_weights, predictions, logprob


def train_mlp(inputs,
              outputs,
              layer_sizes,  # don't include inputs and outputs
              batch_size=256,
              init_weights=None,
              param_scale=0.1,
              num_epochs=5,
              step_size=0.001):

    # split data (again) into a training and a validation set
    (tr_inputs, va_inputs), (tr_outputs, va_outputs) = util.split_data(
        inputs, out_data=outputs, frac=0.80)

    # define num of batches
    num_batches = int(np.ceil(tr_inputs.shape[0] / batch_size))

    # define nn arch
    num_input_dims = tr_inputs.shape[1]
    num_output_dims = tr_outputs.shape[1]
    layer_sizes = [num_input_dims] + layer_sizes + [num_output_dims]

    num_weights, predictions, logprob = \
        build(layer_sizes=layer_sizes, nonlinearity=rbf)

    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # Define training objective
    def objective(weights, iter):
        idx = batch_indices(iter)
        return -logprob(weights, tr_inputs[idx], tr_outputs[idx])

    # Get gradient of objective using autograd.
    objective_grad = grad(objective)

    print(
        "     Epoch     |    Train error  |    Train log-like  |  Holdout error  |  Holdout log-like  ")

    def print_perf(weights, iter, gradient):
        # make predictions
        tr_preds = predictions(weights, tr_inputs)
        va_preds = predictions(weights, va_inputs)
        # get accuracy measurements
        train_acc = rms(tr_preds, tr_outputs)
        valid_acc = rms(va_preds, va_outputs)
        # get log likelihoods
        train_ll = -logprob(weights, tr_inputs, tr_outputs)
        valid_ll = -logprob(weights, va_inputs, va_outputs)
        print("{:15}|{:20}|{:20}|{:20}|{:20}".format(
                iter//num_batches, train_acc, train_ll, valid_acc, valid_ll))

    # define init weights
    if init_weights is None:
        init_weights = param_scale * np.random.randn(num_weights)

    # optimize parameters
    trained_weights = adam(objective_grad,
                           init_weights,
                           step_size=step_size,
                           num_iters=num_epochs*num_batches,
                           callback=print_perf)

    return predictions, logprob, trained_weights
