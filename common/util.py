from collections import defaultdict
import numpy as np
import json
from sklearn.neighbors import LSHForest
import itertools

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
    tr_in_data, te_in_data = in_data[tr_idx], in_data[te_idx]

    if not out_data is None:
        tr_out_data, te_out_data = out_data[tr_idx], out_data[te_idx]
        return (tr_in_data, te_in_data), (tr_out_data, te_out_data)

    return (tr_in_data, te_in_data)


def grouper(iterable, n): # Might combine this with batch_index_generator
    it = iter(iterable)
    while True:
       chunk = tuple(itertools.islice(it, n))
       if not chunk:
           return
       yield chunk


def train_LSHForest(model, batch_size=1000, n_candidates=50, n_estimators=10):
    ''' Given a large wordvec or GloVe model, we need to efficiently be able
        to get a word back from a vector. Current methods rely on
        inefficient search algorithms.

        Args
        ----
        model : gensim.model
                pretrained WordVec model
        batch_size : int
        n_candidates : int
                       number of candidates for LSH to generate
        n_estimators : number of LSH trees in forest

        Returns
        -------
        lshf : LSHForest
    '''
    lshf = LSHForest(n_candidates=n_candidates, n_estimators=n_estimators)
    for batch in grouper(model.index2word, batch_size):
        array = np.array([model[word] for word in batch])
        lshf.partial_fit(array)
    return lshf


def devectorize(vectors, lsh_forest, neighbors=1):
    ''' Returns an (n x neighbors) array of words, where row i contains the
        nearest neighbor words for the word vector i
    '''
    dists, indices = lsh_forest.kneighbors(vectors, n_neighbors=neighbors)
    vec_i2w = np.vectorize(lambda index:model.index2word[index])
    return vec_i2w(indices)
