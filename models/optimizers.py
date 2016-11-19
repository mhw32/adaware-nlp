"""
Some standard gradient-based stochastic optimizers.

These are just standard routines that don't make any use of autograd,
though you could take gradients of these functions too if you want
to do meta-optimization.

These routines can optimize functions whose inputs are structured
objects, such as dicts of numpy arrays.
"""

from __future__ import absolute_import

import autograd.numpy as np
from autograd.util import flatten_func
from builtins import range


def sgd(grad, init_params, callback=None, num_iters=200, step_size=0.1, mass=0.9):
    """
    Stochastic gradient descent with momentum.
    grad() must have signature grad(x, i), where i is the iteration number.

    """
    flattened_grad, unflatten, x = flatten_func(grad, init_params)

    velocity = np.zeros(len(x))
    for i in range(num_iters):
        g = flattened_grad(x, i)
        if callback:
            callback(unflatten(x), i, unflatten(g))
        velocity = mass * velocity - (1.0 - mass) * g
        x = x + step_size * velocity
    return unflatten(x)


def rmsprop(grad, init_params, callback=None, num_iters=100,
            step_size=0.1, gamma=0.9, eps=10**-8):
    """
    Root mean squared prop: See Adagrad paper for details.

    """
    flattened_grad, unflatten, x = flatten_func(grad, init_params)

    avg_sq_grad = np.ones(len(x))
    for i in range(num_iters):
        g = flattened_grad(x, i)
        if callback:
            callback(unflatten(x), i, unflatten(g))
        avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
        x = x - step_size * g/(np.sqrt(avg_sq_grad) + eps)
    return unflatten(x)


def adam(grad, init_params, callback=None, num_iters=100,
         step_size=0.001, b1=0.9, b2=0.999, eps=10**-8):
    """
    Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms.

    """
    flattened_grad, unflatten, x = flatten_func(grad, init_params)

    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = flattened_grad(x, i)
        if callback:
            callback(unflatten(x), i, unflatten(g))
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
    return unflatten(x)



def adagrad(f_grad, x0, xdata, ydata=None, stepsize=1e-2, fudge_factor=1e-6, max_it=1000,
            minibatchsize=None, minibatch_ratio=0.01, callback=None, epsilon=1):
    """ Adaptive Stochastic Gradient Descent (minibatches)
        f_grad          : gradient of the loss function
        x0              : initial params (starting point for optim)
        data            : training data
        stepsize        : master stepsize for Adagrad
        fudge_factor    : to prevent numerical instability
        minibatchsize   : number of training samples to consider each iter
        minibatch_ratio : if minibatchsize is not set this ratio will be
                            used to determine the batch size dependent on
                            the length of the training data
        callback        : function to do something every iter.
                            This gets passed the iter num & current weights

    """

    # d-dimensional vector representing diag(Gt)
    # to store a running total of the squares of the gradients.

    gti = np.zeros(x0.shape[0])
    num_row = xdata.shape[-1]

    if minibatchsize is None:
        minibatchsize = int(ceil(num_row * minibatch_ratio))

    w = x0  # initial weights
    t = 0
    norm_w = 2  # some big number
    while (t < max_it) and (norm_w > epsilon):
        s = sample(xrange(num_row), minibatchsize)
        sdx = xdata[..., s]
        sdy = ydata[..., s] if not ydata is None else ydata
        val, grad = f_grad(w, sdx, sdy)
        gti += grad**2
        adjusted_grad = grad / (fudge_factor + np.sqrt(gti))
        w = w - stepsize * adjusted_grad
        t = t + 1
        norm_w = l2(w)
        if not callback is None:
            callback(t, w, norm_w)
    return w


def batch_adam(
        grad, init_params, callback=None, max_iters=1e5,
        step_size=0.001, b1=0.9, b2=0.999, eps=10**-8,
        validation_grad=None, stop_criterion=1e-3, patience=50,
        early_stop_freq=1):
    """
    Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms.

    """
    flattened_grad, unflatten, x = flatten_func(grad, init_params)
    # initial settings for variables
    m, v = np.zeros(len(x)), np.zeros(len(x))
    cur_iter = 0
    reset_patience = patience
    oldg, g = 0, 1
    # early stop on patience, old gradients not too diff.
    while (cur_iter < max_iters) and (l2(oldg - g) > stop_criterion) and (patience > 0):
        oldg = copy(g)  # save last iter grad
        g = flattened_grad(x, cur_iter)  # pass iter for batch training
        if callback:
            callback(unflatten(x), cur_iter, unflatten(g))
        m = (1 - b1) * g + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(cur_iter + 1))    # Bias correction.
        vhat = v / (1 - b2**(cur_iter + 1))
        x = x - step_size*mhat/(np.sqrt(vhat) + eps)
        # check the validation error
        if (not validation_grad is None) and \
                (((cur_iter % early_stop_freq) == 0) or (cur_iter+1 == max_iters)):

            valoss, _ = validation_grad(x)
            # we want to save the best one (in case of bad regions)
            if cur_iter == 0:
                best_loss = valoss
                best_x = x
            else:
                if valoss < best_loss:
                    best_loss = valoss
                    best_x = x

            # update patience
            patience = patience - 1 if valoss > best_loss else reset_patience

        else:  # if no validation_grad, always save
            best_x = x

        cur_iter += 1
    return unflatten(best_x)
