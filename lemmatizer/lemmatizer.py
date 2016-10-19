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
from nltk import pos_tag, word_tokenize
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


def treebank_to_simple(penn_tag, default=None):
    morphy_tag = {'NN':wordnet.NOUN,
                  'JJ':wordnet.ADJ,
                  'VB':wordnet.VERB,
                  'RB':wordnet.ADV}
    penn_pre = penn_tag[:2]
    if penn_pre in morphy_tag:
        return morphy_tag[penn_pre]
    return default


def pad_array(array, max_size):
    a = np.zeros(max_size)
    a[:len(array)] = array
    return a


def gen_dataset(sentences, train_test_split=True, max_size=25):
    ''' Generate a dataset of (input, output) pairs where the
        input is a vector of characters + POS and output is
        a vector of characters for the lemmatized form.

        Args
        ----
        sentences : list of sentences where each sentence is a string
    '''

    lemmatizer = WordNetLemmatizer()
    X, P, y = [], [], []

    for sentence in sentences:
        words = word_tokenize(sentence)
        raw_pos = [p[1]for p in pos_tag(words)]
        pos = [str(treebank_to_simple(p, default=wordnet.NOUN)) for p in raw_pos]
        lemmas = [str(lemmatizer.lemmatize(w, pos=p)) for (w,p) in zip(words, pos)]

        X.extend(words)
        P.extend(raw_pos)
        y.extend(lemmas)

    pos_set = np.unique(P)
    char_set = list(set(' '.join(X)))
    pos_to_ix = { po:i for i,po in enumerate(pos_set) }
    char_to_ix = { ch:i for i,ch in enumerate(char_set) }
    word_to_ixs = lambda w: [char_to_ix[l] for l in w]

    word_char_arr = np.zeros((len(X), max_size + 1))
    lemma_char_arr = np.zeros((len(y), max_size))

    for i, (word, pos, lemma) in enumerate(zip(X, P, y)):
        word_char_arr[i, :max_size] = pad_array(word_to_ixs(word), max_size)
        word_char_arr[i, max_size] = pos_to_ix[pos]
        lemma_char_arr[i, :] = pad_array(word_to_ixs(lemma), max_size)

    if train_test_split:
        (X_train, X_test), (y_train, y_test) = split_data(
            word_char_arr, out_data=lemma_char_arr, frac=0.80)

    return (X_train, X_test), (y_train, y_test)


def train_lemmatizer_blstm(
    X_train,
    y_train,
    X_test,
    y_test,
    num_hiddens,
    batch_size=64,
    L1_REG=1e-5,
    step_size=0.001,
    num_iters=5000,
    init_params=None
):
