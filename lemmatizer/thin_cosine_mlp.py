''' Thin Cosine Similarity MLP

    With N training samples, generate a new dataset
    where the i-th training sample is (w_i(1), w_i(2), w_i(3)).

    The input to the MLP is 600 node layer with 200 node
    layer (lemma vector). Not all connections persist. Character
    level connections.

    o_{i}(j) ~ (w_{i}^{1}(j)m w_{i}^{2}(j)m w_{i}^{3}(j))

    The MLP uses identity function (no bounded nonlinearities).
    Cost function uses cosine similiarity between real lemma and
    predicted word.

    Use an LSH to convert vector back to word.
'''

from __future__ import absolute_import
from __future__ import print_function

import autograd.numpy as np
import autograd.scipy.stats.norm as norm
from autograd import grad

import sys
sys.path.append('../models')
sys.path.append('../common')
import util
from optimizers import adam
from util import identity, sigmoid


def cms(preds, targets):
    return np.abs(np.sum(mat_cosine_dist(preds, targets))) / targets.shape[0]


def mat_cosine_dist(X, Y):
    prod = np.diagonal(np.dot(X, Y.T),
        offset=0, axis1=-1, axis2=-2)
    len1 = np.sqrt(np.diagonal(np.dot(X, X.T),
        offset=0, axis1=-1, axis2=-2))
    len2 = np.sqrt(np.diagonal(np.dot(Y, Y.T),
        offset=0, axis1=-1, axis2=-2))
    return np.divide(np.divide(prod, len1), len2)


def build(input_count,
          output_count,
          nonlinearity=identity):
    ''' Builds the multi-layer perceptron. Assume that any/all
        one-hot encoding has already been done. This supports
        continuous regression only.
        Args
        ----
        input_count : integer
                      number of features in observations
        output_count : integer
                       number of features in outputs
        state_counts : list
                       list of internal hidden units. Each index represents
                       a new hidden layer.
        Returns
        -------
        prediction: lambda function
                    inputs are (weights, observation)
        log_likelihood: lambda function
                        inputs are (weights, observation, outputs)
        num_weights: integer
                     number of weights in general
    '''

    # only connections from the same letters exist
    # no hidden nodes (directly fully-connected)
    layer_sizes = [input_count, output_count]
    num_reps = input_count / output_count
    num_weights = (num_reps+1) * output_count  # bias
    base_idx = np.array([i for i in range(num_weights) if i % output_count == 0])

    def outputs(weights, inputs):
        targets = []
        for i in range(output_count):
            mask = weights[base_idx+i]
            W, b = mask[:-1], mask[-1]
            I = inputs[:, base_idx[:-1]+i]
            targets.append(np.dot(I, W) + b)
        targets = np.array(targets).T
        targets = nonlinearity(targets)
        return targets

    def log_likelihood(weights, inputs, targets):
        ''' Measure likelihoods by 1 - cosine similiarity between
            the predicted and the real lemma vectors '''
        preds = outputs(weights, inputs)
        log_lik = np.log(cms(preds, targets))
        return log_lik

    return outputs, log_likelihood, num_weights


def train_mlp(
        inputs,
        outputs,
        init_weights=None,
        num_epochs=100,
        step_size=0.001,
        batch_size=128,
        param_scale=0.01,
        l1_lambda=0,
        l2_lambda=0,
        nonlinearity=identity):

    # split data (again) into a training and a validation set
    (tr_inputs, va_inputs), (tr_outputs, va_outputs) = util.split_data(
        inputs, out_data=outputs, frac=0.80)

    num_batches = int(np.ceil(tr_inputs.shape[0] / float(batch_size)))

    input_count = tr_inputs.shape[-1]
    output_count = tr_outputs.shape[-1]

    pred_fun, loglike_fun, num_weights = build(
        input_count, output_count, nonlinearity=nonlinearity)

    if init_weights is None:
        init_weights = np.random.randn(num_weights) * param_scale

    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    def loss(weights, x, y):
        return -loglike_fun(weights, x, y) \
            + l1_lambda * np.sum(np.abs(weights)) \
            + l2_lambda * np.sum(np.power(weights, 2))

    def batch_loss(weights, iter):
        idx = batch_indices(iter)
        return loss(weights, tr_inputs[idx, :], tr_outputs[idx, :])

    print(
        "     Epoch     |    Train cosine  |    Train log-like  |  Holdout cosine  |  Holdout log-like  ")

    def print_perf(weights, iter, gradient):
        # make predictions
        tr_preds = pred_fun(weights, tr_inputs)
        va_preds = pred_fun(weights, va_inputs)
        # get accuracy measurements
        train_acc = cms(tr_preds, tr_outputs)
        valid_acc = cms(va_preds, va_outputs)
        # get log likelihoods
        train_ll = -loglike_fun(weights, tr_inputs, tr_outputs)
        valid_ll = -loglike_fun(weights, va_inputs, va_outputs)
        print("{:15}|{:20}|{:20}|{:20}|{:20}".format(
                iter//num_batches, train_acc, train_ll, valid_acc, valid_ll))


    grad_fun = grad(batch_loss)
    trained_weights = adam(grad_fun,
                           init_weights,
                           step_size=step_size,
                           callback=print_perf,
                           num_iters=num_epochs*num_batches)

    return pred_fun, loglike_fun, trained_weights

