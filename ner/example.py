''' Try the train the NER on a set of News Data '''
import numpy as np
import cPickle
import ner
import dill
import sys
import os

local_ref = lambda x: os.path.join(os.path.dirname(__file__),  x)

obs_set = np.load(local_ref('../storage/ner/X_train.npy'))
out_set = np.load(local_ref('../storage/ner/y_train.npy'))
count_set = np.load(local_ref('../storage/ner/K_train.npy'))

with open(local_ref('../storage/ner/gen_params_set.pkl')) as fp:
    params_dict = cPickle.load(fp)

nn_param_dict = ner.train_ner(obs_set,
                              out_set,
                              count_set,
                              window_size=[1,1],
                              batch_size=128,
                              param_scale=0.01,
                              num_epochs=2500,
                              step_size=0.001,
                              l2_lambda=0)

with open(local_ref('../storage/ner/nn_params_set.dill', 'w')) as fp:
    dill.dump(nn_param_dict, fp)
