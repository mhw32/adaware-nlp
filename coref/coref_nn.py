from __future__ import absolute_import


import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd import grad

from autograd.optimizers import adam
import itertools

sys.path.append('../common')
from util import split_data

sys.path.append('../models')
import nn

from util import sigmoid, relu, concat_and_multiply

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



def coref_nn_predict(params, inputs):
    # returns probability of merge being good
    h1 = relu(
        concat_and_multiply(params['hidden1'], inputs))
    h2 = relu(
        concat_and_multiply(params['hidden2'], h1))
    return sigmoid(np.dot(params['predict'], h2))



def train_coref_nn(
                train_mentions,
                train_mention_ids,
                valid_mentions,
                valid_mention_ids,
                step_size=0.001,
                num_iters=5000,
                init_params=None
):
    def training_loss(params, mention_pairs, target_dict):    
        loss = 0
        for i, pair in enumerate(mention_pairs):
            x = coref_nn_predict(params, pair)
            loss += -np.log(x*target_dict[i] + (1-x)(1-target_dict[i]))
        return loss

    if init_params is None:
        init_params = init_coref_nn_params(input_size=X_train.shape[2],
                                           hidden1_size=300,
                                           hidden2_size=100,
                                           output_size=1,
                                           )
    


if __name__ == '__main__':

    # Specify inference problem by its unnormalized log-posterior.


    # Implement a 3-hidden layer neural network.
    num_weights, predictions, logprob = \
        make_nn_funs(layer_sizes=[1, 20, 20, 1], nonlinearity=rbf)

    inputs, targets = build_toy_dataset()
    objective = lambda weights, t: -logprob(weights, inputs, targets)

    # Set up figure.
    fig = plt.figure(figsize=(12,8), facecolor='white')
    ax = fig.add_subplot(111, frameon=False)
    plt.show(block=False)

    def callback(params, t, g):
        print "Iteration {} log likelihood {}".format(t, -objective(params, t))

    rs = npr.RandomState(0)
    init_params = 0.02 * rs.randn(num_weights)

    print "Optimizing network parameters..."
    optimized_params = adam(grad(objective), init_params,
                            step_size=0.01, num_iters=1000, callback=callback)
