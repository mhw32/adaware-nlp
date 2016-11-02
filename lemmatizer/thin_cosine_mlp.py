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
from optimizers import adam


def cosine_dist(v1, v2):
    prod = np.dot(v1, v2.T)
    len1 = np.sqrt(np.dot(v1, v1.T))
    len2 = np.sqrt(np.dot(v2, v2.T))
    return prod / (len1 * len2)


def mat_cosine_dist(X, Y):
    prod = np.diagonal(np.dot(X, Y.T),
        offset=0, axis1=-1, axis2=-2)
    len1 = np.sqrt(np.diagonal(np.dot(X, X.T),
        offset=0, axis1=-1, axis2=-2))
    len2 = np.sqrt(np.diagonal(np.dot(Y, Y.T),
        offset=0, axis1=-1, axis2=-2))
    return np.divide(np.divide(prod, len1), len2)


def identity(X): return X


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
        log_lik = np.log(np.sum(1 - mat_cosine_dist(preds, targets)) / inputs.shape[0])
        return log_lik

    return outputs, log_likelihood, num_weights


def train_mlp(
        obs_set,
        out_set,
        init_weights=None,
        num_epochs=100,
        batch_size=128,
        param_scale=0.01,
        l2_lambda=0):

    input_count = obs_set.shape[-1]
    output_count = out_set.shape[-1]
    num_batches = int(np.ceil(obs_set.shape[0] / float(batch_size)))

    pred_fun, loglike_fun, num_weights = build(
        input_count, output_count)

    if init_weights is None:
        init_weights = np.random.randn(num_weights) * param_scale

    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    def loss(weights, x, y):
        return -loglike_fun(weights, x, y) + l2_lambda * np.sum(np.power(weights, 2))

    def batch_loss(weights, iter):
        idx = batch_indices(iter)
        return loss(weights, obs_set[idx, :], out_set[idx, :])

    def callback(x, i, g):
        print('iter {}  |  training loss {}'.format(
            i, loss(x, obs_set, out_set)))

    grad_fun = grad(batch_loss)
    trained_weights = adam(grad_fun,
                           init_weights,
                           callback=callback,
                           num_iters=num_epochs)

    return pred_fun, loglike_fun, trained_weights

