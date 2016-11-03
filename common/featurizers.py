''' With NLP, there's a lot of different features that
    can be extracted from a list of words. We can push all
    of them here so we can reuse them across modules.
'''

import numpy as np

def window_featurizer(X, y=None, pad=True, size=[1,1]):
    ''' Given some time series of data, it might be a good idea
        to include some temporal information by adding neighboring
        vectors.

        Args
        ----
        X : 2D numpy
            inputs matrix
        y : 2D numpy
            outputs matrix
        pad : boolean
              whether not to add zeros to the beginning and ends of
              each sentence to keep 1st and last word
        size : list of 2
               first is number prior, second is number after
    '''

    if sum(size) <= 0:
        return (X, y) if not y is None else X

    window_X = np.zeros((X.shape[0], X.shape[1]*(sum(size)+1)))
    if not y is None:
        window_y = np.zeros((y.shape[0], y.shape[1]))

    if pad:
        # prepend + postpend with 0's
        X = np.vstack((np.ones((size[0], X.shape[1]))*ZERO_EPSILON,
            X, np.ones((size[1], X.shape[1]))*ZERO_EPSILON))

    for i in range(size[0],X.shape[0]-size[1]):
        for j,k in enumerate(range(i-size[0],i+size[1]+1)):
            window_X[i-size[0], j*X.shape[1]:(j+1)*X.shape[1]] = X[k, :]
        if not y is None:
            window_y[i-size[0], :] = y[i, :]

    return (window_X, window_y) if not y is None else window_X
