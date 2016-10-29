import os
import cPickle
import numpy as np
from nltk.corpus import brown
from lemmatizer import gen_dataset, train_lemmatizer

def gen_brown_dataset(output_folder, num=None):
    sentences = brown.sents()

    if num:
        if num > len(sentences):
            num = len(sentences)
        sentences = sentences[:num]

    (X_train, X_test), (y_train, y_test), param_dict = gen_dataset(sentences)

    if output_folder:
        np.save(os.path.join(output_folder, 'X_train.npy'), X_train)
        np.save(os.path.join(output_folder, 'X_test.npy'), X_test)
        np.save(os.path.join(output_folder, 'y_train.npy'), y_train)
        np.save(os.path.join(output_folder, 'y_test.npy'), y_test)

        with open(os.path.join(output_folder, 'gen_param_dict.pkl'), 'w') as f:
            cPickle.dump(param_dict, f)


def train_brown_lemmatizer(output_folder):
    obs_set = np.load('_example/100/X_train.npy')
    out_set = np.load('_example/100/y_train.npy')
    nn_param_set = train_lemmatizer(
        obs_set,
        out_set,
        [1000, 500],
        window_size=[1,1],
        batch_size=256,
        param_scale=0.01,
        num_epochs=10,
        step_size=0.001)

    if output_folder:
        with open(os.path.join(output_folder, 'nn_param_dict.pkl'), 'w') as f:
            cPickle.dump(nn_param_dict, f)
