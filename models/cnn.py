''' Conv Nets (similar style to LeNet-5)
    http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf
'''

from __future__ import absolute_import
from __future__ import print_function

import pdb
import sys
sys.path.append('../common')
sys.path.append('../models')
from optimizers import adam
import util

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.signal import convolve
from autograd import grad
from builtins import range


class WeightsParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.N = 0

    def add_weights(self, name, shape):
        start = self.N
        self.N += np.prod(shape)
        self.idxs_and_shapes[name] = (slice(start, self.N), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[idxs], shape)


def logsumexp(X, axis, keepdims=False):
    ''' normalizing in log space hack '''
    max_X = np.max(X)
    return max_X + np.log(np.sum(np.exp(X - max_X), axis=axis, keepdims=keepdims))


def build(input_shape, layer_specs, L2_reg):
    parser = WeightsParser()
    cur_shape = input_shape
    pdb.set_trace()

    for layer in layer_specs:
        N_weights, cur_shape = layer.build_weights_dict(cur_shape)
        parser.add_weights(layer, (N_weights,))

    def predictions(W_vect, inputs):
        """Outputs normalized log-probabilities.
        shape of inputs : [data, color, y, x]"""
        cur_units = inputs
        for layer in layer_specs:
            cur_weights = parser.get(W_vect, layer)
            cur_units = layer.forward_pass(cur_units, cur_weights)
        return cur_units

    def loss(W_vect, X, T):
        log_prior = -L2_reg * np.dot(W_vect, W_vect)
        log_lik = np.sum(predictions(W_vect, X) * T)
        return - log_prior - log_lik

    def frac_err(W_vect, X, T):
        return np.mean(np.argmax(T, axis=1) != np.argmax(predictions(W_vect, X), axis=1))

    return parser.N, predictions, loss, frac_err


class conv_layer(object):
    def __init__(self, kernel_shape, num_filters):
        self.kernel_shape = kernel_shape
        self.num_filters = num_filters

    def forward_pass(self, inputs, param_vector):
        # Input dimensions:  [data, 1, y, x]
        # Params dimensions: [1, filter, y, x]
        # Output dimensions: [data, filter, y, x]
        params = self.parser.get(param_vector, 'params')
        biases = self.parser.get(param_vector, 'biases')
        conv = convolve(inputs, params, axes=([2, 3], [2, 3]), dot_axes = ([1], [0]), mode='valid')
        return conv + biases

    def build_weights_dict(self, input_shape):
        # Input shape : [1, y, x] (don't need to know number of data yet)
        self.parser = WeightsParser()
        self.parser.add_weights('params', (input_shape[0], self.num_filters)
                                          + self.kernel_shape)
        self.parser.add_weights('biases', (1, self.num_filters, 1, 1))
        output_shape = (self.num_filters,) + \
                       self.conv_output_shape(input_shape[1:], self.kernel_shape)
        return self.parser.N, output_shape

    def conv_output_shape(self, A, B):
        return (A[0] - B[0] + 1, A[1] - B[1] + 1)


class maxpool_layer(object):
    def __init__(self, pool_shape):
        self.pool_shape = pool_shape

    def build_weights_dict(self, input_shape):
        # input_shape dimensions: [filter, y, x] (don't need to know number of data yet)
        output_shape = list(input_shape)
        for i in [0, 1]:
            assert input_shape[i + 1] % self.pool_shape[i] == 0, \
                "maxpool shape ({}) should tile input ({}) exactly".format(
                    self.pool_shape[i], input_shape[i + 1])
            output_shape[i + 1] = input_shape[i + 1] / self.pool_shape[i]
        return 0, output_shape

    def forward_pass(self, inputs, param_vector):
        new_shape = inputs.shape[:2]
        for i in [0, 1]:
            pool_width = self.pool_shape[i]
            img_width = inputs.shape[i + 2]
            new_shape += (pool_width, img_width / pool_width)
        result = inputs.reshape(new_shape)
        return np.max(np.max(result, axis=2), axis=3)


class full_layer(object):
    def __init__(self, size):
        self.size = size

    def build_weights_dict(self, input_shape):
        # Input shape is anything (all flattened)
        input_size = np.prod(input_shape, dtype=int)
        self.parser = WeightsParser()
        self.parser.add_weights('params', (input_size, self.size))
        self.parser.add_weights('biases', (self.size,))
        return self.parser.N, (self.size,)

    def forward_pass(self, inputs, param_vector):
        params = self.parser.get(param_vector, 'params')
        biases = self.parser.get(param_vector, 'biases')
        if inputs.ndim > 2:
            inputs = inputs.reshape((inputs.shape[0], np.prod(inputs.shape[1:])))
        return self.nonlinearity(np.dot(inputs[:, :], params) + biases)


class tanh_layer(full_layer):
    def nonlinearity(self, x):
        return np.tanh(x)


class softmax_layer(full_layer):
    def nonlinearity(self, x):
        return x - logsumexp(x, axis=1, keepdims=True)


def train_cnn(inputs,
              outputs,
              layer_specs,
              init_weights=None,
              param_scale=0.1,
              step_size=0.001,
              batch_size=128,
              num_epochs=50,
              L2_reg=1.0):
    ''' wrapper function to train the convnet '''

    (tr_inputs, va_inputs), (tr_outputs, va_outputs) = util.split_data(
        inputs, out_data=outputs, frac=0.80)

    input_shape = tr_inputs.shape
    num_data = tr_inputs.shape

    # number of batches
    num_batches = int(np.ceil(tr_inputs.shape[0] / batch_size))

    def batch_indices(iter):
        idx = iter % num_batches
        return slice(idx * batch_size, (idx+1) * batch_size)

    # build CNN
    num_weights, pred_fun, loss_fun, frac_err = build(input_shape[1:], layer_specs, L2_reg)

    def batch_loss(weights, iter):
        idx = batch_indices(iter)
        return loss_fun(weights, tr_inputs[idx], tr_outputs[idx])

    loss_grad = grad(batch_loss)

    # init weights
    if init_weights is None:
        rs = npr.RandomState()
        init_weights = rs.randn(num_weights) * param_scale

    print("    Epoch      |    Train err  |   Validation error  ")
    def print_perf(weights, epoch, gradients):
        va_perf = frac_err(weights, va_inputs, va_outputs)
        tr_perf = frac_err(weights, tr_inputs, tr_outputs)
        print("{0:15}|{1:15}|{2:15}".format(epoch, tr_perf, va_perf))

    # optimize parameters
    trained_weights = adam(loss_grad,
                           init_weights,
                           step_size=step_size,
                           num_iters=num_epochs*num_batches,
                           callback=print_perf)

    return pred_fun, loss_fun, trained_weights
