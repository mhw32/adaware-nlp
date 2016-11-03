''' Name Entity Recognition

    Model: Convolutional Neural Network
        - Conv --> Pool --> Conv --> Pool --> FNN
        - input: n-gram*300
        - conv(1): 4 kernels (2*41 size)
        - pool(1): horiz. and vert pooling dim = 2
        - conv(2): 8 kernels (1*21 size)
        - pool(2): horiz = 2, vert = 1
        - 256 HU w/ 0.5 proba dropout

    Heuristics:
    1. Encoding: Use Word2Vec/GloVe
    2. N-gram
        Train 3,5,7,9 and use a mixture of
        experts when doing predictions

        Bayesian Hierarchical Mixtures of Experts
    3. Possible data augmentation:
        Replace words with synonyms

    Training dataset: look in the drive for a
    news_tagged_data.txt.
'''

import numpy as np
import sys
sys.path.append('../common')
import util
import cPickle


def read_wordvec_from_file(in_file, out_file):
    with open(in_file) as fp:
        raw_data = fp.read()
    data = raw_data.split('\n')
    num_data = len(data)

    model = {}
    for i in range(num_data):
        cur_data = data[i].split('\t')[:-1]
        key = cur_data[0]
        val = np.array(cur_data[1:]).astype(np.float64)
        model[key] = val

    with open(out_file, 'w') as fp:
        cPickle.dump(model, fp)


def read_dataset_from_file(file):
    with open(file) as fp:
        raw_data = fp.read()
    data = raw_data.split('\n\n')[:-1]
    num_data = len(data)

    text = []
    category = []

    for i in range(num_data):
        if i % 100 == 0:
            print('processed {} lines'.format(i))
        line = data[i].split('\n')
        num_words = len(line)

        cur_text, cur_category = [], []
        for j in range(num_words):
            t, c = line[j].split('\t')
            cur_text.append(t)
            cur_category.append(c)

        text.append(cur_text)
        category.append(cur_category)

    (tr_text, te_text), (tr_category, te_category) = \
        util.split_data(text, out_data=category, frac=0.80)

    return (tr_text, te_text), (tr_category, te_category)


def train_ner():
    layer_spaces = [conv_layer((2, 41), 4),
                    maxpool_layer(2, 2),
                    conv_layer((1, 21), 8),
                    maxpool_layer(2, 1),
                    full_layer(256),
                    softmax_layer(2)]


class NeuralNER(object):
    def __init__(self):
        pass
