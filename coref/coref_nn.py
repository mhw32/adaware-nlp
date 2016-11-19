from __future__ import absolute_import


import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd import grad
import sys
import itertools

sys.path.append('../common')
from util import split_data, batch_index_generator

sys.path.append('../models')
import nn
from optimizers import adam

from util import sigmoid, relu, concat_and_multiply
import pdb

def init_coref_nn_params(input_size,
                         hidden1_size,
                         hidden2_size,
                         output_size,
                         param_scale=0.01,
                         rs=npr.RandomState(0)):
    '''
        Initialize blstm weights
    '''
    def rp(*shape):
        return rs.randn(*shape) * param_scale

    return {'hidden1':       rp(input_size + 1, hidden1_size),
            'hidden2':       rp(hidden1_size + 1, hidden2_size),
            'predict':        rp(hidden2_size, output_size)}


def train_coref_nn(
                X,
                Y,
                batch_size=200,
                step_size=0.001,
                num_iters=5000,
                init_params=None,
                seed=None
):

    num_samples = X.shape[0]
    if seed:
        np.random.seed(seed)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    training_indices = indices[num_samples//100:]
    validation_indices = indices[:num_samples//100]

    index_gen = batch_index_generator(len(training_indices), batch_size=batch_size)

    if init_params is None:
        init_params = init_coref_nn_params(input_size=X.shape[1],
                                           hidden1_size=300,
                                           hidden2_size=100,
                                           output_size=1,
                                           )


    def callback(params, t, g):
        if t % (num_samples//10000) == 0:
            print "Iteration {} log likelihood {}".format(t, training_loss(params, t))
            print np.sum((coref_nn_predict(params, X[validation_indices]) > 0.5) == Y[validation_indices])#/len(validation_indices)

    def training_loss(params, iter):
        loss = 0
        batch_indices = training_indices[index_gen.next()]
        predict = coref_nn_predict(params, X[batch_indices])
        targets = Y[batch_indices].reshape(-1,1)
        loss = -np.sum(np.log(predict*targets + (1-predict)*(1-targets)))
        return loss

    def coref_nn_predict(params, inputs):
        # returns probability of coref
        h1 = relu(concat_and_multiply(params['hidden1'], inputs))
        h2 = relu(concat_and_multiply(params['hidden2'], h1))
        return sigmoid(np.dot(h2,params['predict']))

    return adam(grad(training_loss), init_params,
                            step_size=step_size, num_iters=num_iters, callback=callback)



if __name__ == '__main__':
    XY_DATA = np.load('coref_data.npy')
    X = XY_DATA[:,:-1]
    Y = XY_DATA[:,-1]
    del XY_DATA
    optimized_params = train_coref_nn(X, Y, seed=123, step_size=0.0001,batch_size=200)
