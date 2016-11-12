''' Try the train the NER on a set of News Data '''
import numpy as np
import cPickle
import ner

obs_set = np.load('../storage/ner/X_train.npy')
out_set = np.load('../storage/ner/y_train.npy')
count_set = np.load('../storage/ner/K_train.npy')

with open('../storage/ner/gen_params_set.pkl') as fp:
    params_dict = cPickle.load(fp)

nn_param_dict = ner.train_ner(obs_set,
                              out_set,
                              count_set,
                              window_size=[1,1],
                              batch_size=128,
                              param_scale=0.01,
                              num_epochs=2000,
                              step_size=0.001,
                              l2_lambda=0)

with open('../storage/ner/nn_params_set.pkl', 'w') as fp:
    cPickle.dump(nn_param_dict, fp)
