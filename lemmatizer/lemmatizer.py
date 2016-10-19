""" BLSTM for lemmatization. State-of-the-art lemmatization and
    stemming seems to work through large dictionary lookups and
    hardcoded chopping. Instead we can train a character-level
    RNN to map a word to its own lemma.

    There will be a feature per character (padded to some max).
    The POS should also be a feature as it provides info.

"""

from __future__ import absolute_import
from __future__ import print_function

import sys
import cPickle

sys.path.append('../common')
from util import batch_index_generator, split_data

sys.path.append('../models')
from blstm import init_blstm_params, blstm_predict, log_likelihood

# to generate a training dataset
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

def gen_dataset(sentences, train_test_split=True):
    ''' Generate a dataset of (input, output) pairs where the
        input is a vector of characters + POS and output is
        a vector of characters for the lemmatized form.

        Args
        ----
        sentences : list of sentences where each sentence is a string
    '''

    tagger = pos_tag
    lemmatizer = WordNetLemmatizer()
    X, y = [], []

    for sentence in sentences:
        resp = tagger(sentence)
        words, tags = zip(*resp)
        lemmas = [lemmatizer.lemmatize(w, pos=p) for (w,p) in resp]

        X.extend(words)
        y.extend(lemmas)

    X = np.array(X)
    y = np.array(y)

    if train_test_split:
        (X_train, X_test), (y_train, y_test) = split_data(
            X, out_data=y, frac=0.80)

    return (X_train, X_test), (y_train, y_test)
