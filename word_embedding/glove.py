""" GloVe (Global Vectors for Word Representation)

    Simple test cases :
        1) king - man + woman ~ queen
        2) brought - bring + seek ~ sought

    Background:

    - define co-occurence matrix X (Xij = measure of how
      often the word i appears in the context of word j)

    - loop thru data -> make co-occurrence X
    - assign continuous values for each word in corpus
    - given word i + j
        - w_i * w_j + bias_i + bias_j = log X_ij
    - minimize an objective function
        - J = sum sum f(X_ij)(w_i * w_j + bias_i + bias_j - log X_ij)^2
        - f prevents common word pairs
        - f(X_ij) = (X_ij / x_max)^alpha if X_ij < x_max else 1

    Implementation:

    - build co-occurrences by sliding n-gram window
    - train GloVe to optimize parameters

    Paper:
    http://www-nlp.stanford.edu/projects/glove/
    http://www-nlp.stanford.edu/pubs/glove.pdf

"""

import os
import math
import codecs
import cPickle
import itertools
import msgpack

import numpy as np
from scipy import sparse

from collections import Counter
from functools import wraps
from random import shuffle


def listify(fn):
    """ converts tuple to list for a generator """

    @wraps(fn)
    def listified(*args, **kwargs):
        return list(fn(*args, **kwargs))

    return listified


def make_id2word(vocab):
    return dict((id, word) for word, (id, _) in vocab.iteritems())


def load_object(path, build_fn, *args, **kwargs):
    """ load from serialized form or build an object, saving the built
        object; kwargs provided to `build_fn`.
    """

    save = False
    obj = None

    if path is not None and os.path.isfile(path):
        with open(path, 'rb') as obj_f:
            obj = msgpack.load(obj_f, use_list=False, encoding='utf-8')
    else:
        save = True

    if obj is None:
        obj = build_fn(*args, **kwargs)

        if save and path is not None:
            with open(path, 'wb') as obj_f:
                msgpack.dump(obj, obj_f)

    return obj


def merge_main_context(W,
                       merge_fun=lambda m, c: np.mean([m, c], axis=0),
                       normalize=True):
    """ merge the main-word and context-word vectors for a weight matrix
        using a provided merge function

        By default, `merge_fun` returns the mean of the two vectors.

        Args
        ----
        W : numpy array
            weight matrix

        normalize : bool
                    normalize probs

        Returns
        -------
        W : numpy array
            merged weight matrix
    """

    vocab_size = len(W) / 2
    for i, row in enumerate(W[:vocab_size]):
        merged = merge_fun(row, W[i + vocab_size])
        if normalize:
            merged /= np.linalg.norm(merged)
        W[i, :] = merged

    return W[:vocab_size]


def get_vector(word, W, vocab):
    """ Given a word, return its vector repr """

    assert len(W) == len(vocab)
    if word in vocab:
        return W[vocab[word][0]]
    else:
        return np.zeros(W.shape[-1])


def most_similar(W, vocab, id2word, word, n=15):
    """ Given a word, find the n most similar words.

        Returns
        -------
        a list of word strings.
    """

    assert len(W) == len(vocab)

    word_id = vocab[word][0]

    dists = np.dot(W, W[word_id])
    top_ids = np.argsort(dists)[::-1][:n + 1]

    return [id2word[id] for id in top_ids if id != word_id][:n]


def build_vocab(corpus):
    """ given a corpus, build a dictionary that maps word to a
        id and frequency value

        corpus : file object
    """

    vocab = Counter()
    for line in corpus:
        tokens = line.strip().split()
        vocab.update(tokens)

    d = {word: (i, freq) for i, (word, freq) in enumerate(vocab.iteritems())}
    return d


@listify
def build_cooccur_mat(vocab,
                      corpus,
                      window_size=10,
                      min_count=None):
    """ construct the cooccurence matr ix .

        Args
        ----
        vocab : dict
                mapping words to integer ids
        corpus : <iterator>
                 iterators over sentence
        window_size : int
        min_count : int
                    cooccurrence pairs that occurs less than min_count

        Returns
        -------
        tuple generator, where each element is
        (i_main, i_context, cooccurrence)

        - i_main is the ID of the main word
        - i_context is the ID of the context word

    """
    vocab_size = len(vocab)
    id2word = dict((i, word) for word, (i, _) in vocab.iteritems())

    # Collect cooccurrences internally as a sparse matrix
    cooccurrences = sparse.lil_matrix((vocab_size, vocab_size),
                                      dtype=np.float64)

    for i, line in enumerate(corpus):
        if i % 1000 == 0:
            print("Building cooccurrence matrix: on line {}".format(i))

        tokens = line.strip().split()
        token_ids = [vocab[word][0] for word in tokens]

        for center_i, center_id in enumerate(token_ids):

            # collect all word IDs in left window of center word
            context_ids = token_ids[max(0, center_i - window_size): center_i]
            contexts_len = len(context_ids)

            for left_i, left_id in enumerate(context_ids):
                # distance from center word
                distance = contexts_len - left_i

                # weight by inverse of distance between words
                increment = 1.0 / float(distance)

                # build co-occurrence matrix symmetrically (pretend we
                # are calculating right contexts as well)
                cooccurrences[center_id, left_id] += increment
                cooccurrences[left_id, center_id] += increment

    # yield tuple sequence
    for i, (row, data) in enumerate(itertools.izip(cooccurrences.rows,
                                                   cooccurrences.data)):
        if min_count is not None and vocab[id2word[i]][1] < min_count:
            continue

        for data_idx, j in enumerate(row):
            if min_count is not None and vocab[id2word[j]][1] < min_count:
                continue

            yield i, j, data[data_idx]


def train_glove(vocab,
                cooccurrences,
                callback=None,
                vector_size=100,
                iterations=25,
                **kwargs):
    """ Given a feed of cooccurrences as (i, j, x_ij)
        train the GloVe vectors.

        Args
        ----
        vocab : dict
        cooccurrences : iterator of tuples
        callback : function
                   called after each iteration

        Returns
        -------
        W : array
            word vector matrix
    """

    vocab_size = len(vocab)

    # Word vector matrix. This matrix is (2V) * d, where N is the size
    # of the corpus vocabulary and d is the dimensionality of the word
    # vectors. All elements are initialized randomly in the range (-0.5,
    # 0.5]. We build two word vectors for each word: one for the word as
    # the main (center) word and one for the word as a context word.
    #
    # Pennington et al. (2014) suggest adding or averaging the
    # two for each word, or discarding the context vectors.
    W = (np.random.rand(vocab_size * 2, vector_size) - 0.5) / \
        float(vector_size + 1)

    # bias terms, each associated with a single vector. An array of size
    # $2V$, initialized randomly in the range (-0.5, 0.5].
    biases = (np.random.rand(vocab_size * 2) - 0.5) / float(vector_size + 1)

    # Training is done via adaptive gradient descent (AdaGrad). To make
    # this work we need to store the sum of squares of all previous
    # gradients.
    #
    # Like `W`, this matrix is (2V) * d.
    #
    # Initialize all squared gradient sums to 1 so that our initial
    # adaptive learning rate is simply the global learning rate.
    gradient_squared = np.ones((vocab_size * 2, vector_size),
                               dtype=np.float64)

    # Sum of squared gradients for the bias terms.
    gradient_squared_biases = np.ones(vocab_size * 2, dtype=np.float64)

    data = [(W[i_main], W[i_context + vocab_size],
             biases[i_main: i_main + 1],
             biases[i_context + vocab_size: i_context + vocab_size + 1],
             gradient_squared[i_main], gradient_squared[
                 i_context + vocab_size],
             gradient_squared_biases[i_main: i_main + 1],
             gradient_squared_biases[
                 i_context + vocab_size: i_context + vocab_size + 1],
             cooccurrence)
            for i_main, i_context, cooccurrence in cooccurrences]

    for i in range(iterations):
        cost = update_glove(vocab, data, **kwargs)
        if callback is not None:
            callback(W)

    return W


def update_glove(vocab, data, learning_rate=0.05, x_max=100, alpha=0.75):
    """
    run a single iteration of GloVe training using the given
    cooccurrence data and the previously computed weight vectors /
    biases and accompanying gradient histories.

    Args
    ----
    data: custom tuple
          a pre-fetched data / weights list where each element is:
            (v_main, v_context,
             b_main, b_context,
             gradsq_W_main, gradsq_W_context,
             gradsq_b_main, gradsq_b_context,
             cooccurrence)

    alpha : double
           learning rate

    Returns
    -------
    cost associated with the given weight assignments and
    updates the weights by online AdaGrad in place.

    """

    global_cost = 0

    # iterate over data randomly
    shuffle(data)

    for (v_main, v_context, b_main, b_context, gradsq_W_main, gradsq_W_context,
         gradsq_b_main, gradsq_b_context, cooccurrence) in data:

        weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1

        # intermediate component of cost function
        # --> J' = w_i^Tw_j + b_i + b_j - log(X_{ij}) $$
        cost_inner = (v_main.dot(v_context)
                      + b_main[0] + b_context[0]
                      - math.log(cooccurrence))

        # compute cost --> J = f(X_{ij}) (J')^2 $$
        cost = weight * (cost_inner ** 2)

        # Add weighted cost to the global cost tracker
        global_cost += 0.5 * cost

        # Compute gradients for word vector terms.
        grad_main = weight * cost_inner * v_context
        grad_context = weight * cost_inner * v_main

        # Compute gradients for bias terms
        grad_bias_main = weight * cost_inner
        grad_bias_context = weight * cost_inner

        # Now perform adaptive updates
        v_main -= (learning_rate * grad_main / np.sqrt(gradsq_W_main))
        v_context -= (learning_rate * grad_context / np.sqrt(gradsq_W_context))

        b_main -= (learning_rate * grad_bias_main / np.sqrt(gradsq_b_main))
        b_context -= (learning_rate * grad_bias_context / np.sqrt(
            gradsq_b_context))

        # Update squared gradient sums
        gradsq_W_main += np.square(grad_main)
        gradsq_W_context += np.square(grad_context)
        gradsq_b_main += grad_bias_main ** 2
        gradsq_b_context += grad_bias_context ** 2

    return global_cost


def _load_pretrained_glove_from_file(path):
    with open(path) as fp:
        data = fp.read().split('\n')

    vocab = {}
    weights = []
    for i, line in enumerate(data):
        if i % 1000 == 0:
            print('{} tokens processed.'.format(i))
        line = line.split(' ')
        if len(line) > 0:
            word = line[0]
            data = [float(j) for j in line[1:]]
            vocab[line[0]] = (i, None)
            weights.append(data)

    weights = np.array(weights)
    return weights, vocab
