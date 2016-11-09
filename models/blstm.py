from __future__ import absolute_import
from __future__ import print_function

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.util import flatten
from autograd import grad
from autograd.scipy.misc import logsumexp

from util import sigmoid, concat_and_multiply


def init_blstm_params(input_size,
                      state_size,
                      output_size,
                      param_scale=0.01,
                      rs=npr.RandomState(0)):
    '''
        Initialize blstm weights
    '''
    def rp(*shape):
        return rs.randn(*shape) * param_scale

    return {'capitals':       rp(3, input_size),
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
            'predict':        rp(state_size*2 + 1, output_size)}


def blstm_predict(params, inputs, capitals):
    def update_forward_lstm(_input, hiddens, cells):
        change = np.tanh(
            concat_and_multiply(params['change_f'], _input, hiddens))
        forget = sigmoid(
            concat_and_multiply(params['forget_f'], _input, hiddens))
        ingate = sigmoid(
            concat_and_multiply(params['ingate_f'], _input, hiddens))
        outgate = sigmoid(
            concat_and_multiply(params['outgate_f'], _input, hiddens))
        cells = cells * forget + ingate * change
        hiddens = outgate * np.tanh(cells)
        return hiddens, cells

    def update_backward_lstm(_input, hiddens, cells):
        change = np.tanh(
            concat_and_multiply(params['change_b'], _input, hiddens))
        forget = sigmoid(
            concat_and_multiply(params['forget_b'], _input, hiddens))
        ingate = sigmoid(
            concat_and_multiply(params['ingate_b'], _input, hiddens))
        outgate = sigmoid(
            concat_and_multiply(params['outgate_b'], _input, hiddens))
        cells = cells * forget + ingate * change
        hiddens = outgate * np.tanh(cells)
        return hiddens, cells

    def hiddens_to_output_probs(hiddens):
        output = concat_and_multiply(params['predict'], hiddens)
        return output - logsumexp(output, axis=1, keepdims=True)

    num_sequences = inputs.shape[1]
    hiddens1 = np.repeat(params['init hiddens_f'], num_sequences, axis=0)
    cells1 = np.repeat(params['init cells_f'],   num_sequences, axis=0)

    hiddens2 = np.repeat(params['init hiddens_b'], num_sequences, axis=0)
    cells2 = np.repeat(params['init cells_b'],   num_sequences, axis=0)
    output = []
    for i, _input in enumerate(inputs):  # Iterate over sentence words
        _input = _input + np.dot(capitals[i], params['capitals'])
        hiddens1, cells1 = update_forward_lstm(_input, hiddens1, cells1)
        hiddens2, cells2 = update_backward_lstm(
            _input[:, ::-1], hiddens2, cells2)
        output.append(hiddens_to_output_probs(np.hstack((hiddens1, hiddens2))))
    return output


def log_likelihood(params, inputs, targets, mask, capitals):
    logprobs = blstm_predict(params, inputs, capitals)
    loglik = 0.0
    num_time_steps, num_examples, _ = inputs.shape
    for t in range(num_time_steps):
        loglik += np.sum(logprobs[t] * targets[t])
    return loglik / sum(mask)

# Seems like there's a bug in the autograd flatten function for DictNodes,
# unless I'm using it wrong.

# yup -- autograd does not work well with most data types


def l1_norm(params):
    if isinstance(params, dict):
        return np.sum(np.absolute(flatten(params)[0]))
    return np.sum(np.absolute(flatten(params.value)[0]))
