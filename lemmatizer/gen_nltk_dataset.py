import os
import cPickle
import numpy as np
from nltk.corpus import brown
from lemmatizer import gen_dataset

def gen_brown_dataset(output_folder, num=None):
    sentences = brown.sents()

    if num:
        if num > len(sentences):
            num = len(sentences)
        sentences = sentences[:num]

    (X_train, X_test), (y_train, y_test), char_set, pos_set, max_size = \
        gen_dataset(sentences, train_test_split=True, max_size=25)

    params = {}
    params['max_size'] = max_size

    if output_folder:
        np.save(os.path.join(output_folder, 'X_train.npy'), X_train)
        np.save(os.path.join(output_folder, 'X_test.npy'), X_test)
        np.save(os.path.join(output_folder, 'y_train.npy'), y_train)
        np.save(os.path.join(output_folder, 'y_test.npy'), y_test)
        with open(os.path.join(output_folder, 'char_set.pkl'), 'w') as f:
            cPickle.dump(char_set, f)
        with open(os.path.join(output_folder, 'pos_set.pkl'), 'w') as f:
            cPickle.dump(pos_set, f)
        with open(os.path.join(output_folder, 'params.pkl'), 'w') as f:
            cPickle.dump(params, f)
