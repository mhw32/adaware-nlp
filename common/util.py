from collections import defaultdict
import numpy as np
import json


def is_int(x):
    try:
        int(x)
        return True
    except:
        return False


def FreqDist(tags):
    freq = defaultdict(lambda: 0)
    for tag in tags:
        freq[tag] += 1
    return freq


def batch_index_generator(n, batch_size=50):
    i = 0
    shuffled_indices = np.random.permutation(n)
    while True:
        if i + batch_size > n:
            shuffled_indices = np.random.permutation(n)
            i = 0
        yield shuffled_indices[i:i+batch_size]
        i += batch_size


def sigmoid(x):
    return 0.5*(np.tanh(x) + 1.0)


def relu(x):
    return np.maximum(0, x)


def concat_and_multiply(weights, *args):
    cat_state = np.hstack(args + (np.ones((args[0].shape[0], 1)),))
    return np.dot(cat_state, weights)


def pp_json(json_thing, sort=True, indents=4):
    if type(json_thing) is str:
        print(json.dumps(json.loads(json_thing),
                         sort_keys=sort,
                         indent=indents))
    else:
        print(json.dumps(json_thing,
                         sort_keys=sort,
                         indent=indents))
    return None


def split_data(in_data, out_data=None, frac=0.80):
    ''' splitting dataset into training/testing sets

        Args
        ----
        in_data : np array
                  input features x rows

        out_data : np array
                   output features x rows

        frac : float
               %  to keep in training set

        Returns
        -------
        (training_inputs, testing_inputs),
        [(training_outputs, testing_outputs)]
    '''

    num_data = in_data.shape[0]
    num_tr_data = int(np.floor(num_data * frac))
    indices = np.random.permutation(num_data)

    tr_idx, te_idx = indices[:num_tr_data], indices[num_tr_data:]
    tr_in_data, te_in_data = in_data[tr_idx, :], in_data[te_idx, :]

    if not out_data is None:
        tr_out_data, te_out_data = out_data[tr_idx, :], out_data[te_idx, :]
        return (tr_in_data, te_in_data), (tr_out_data, te_out_data)

    return (tr_in_data, te_in_data)
