#!/usr/bin/env python

'''
    Executable to generate parameters for sentence
    disambiguation weights, namely the files:
    - storage/data_nn_disambiguator/X_train.npy
    - storage/data_nn_disambiguator/X_test.npy
    - storage/data_nn_disambiguator/y_train.npy
    - storage/data_nn_disambiguator/y_test.npy
    - storage/trained_weights.npy
'''

import os
import sys
project_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(project_path)

from disambiguator import create_features_labels, main
from prompt import confirm_params_override

class SentenceWeightParams:
    
    @staticmethod
    def write_params():
        confirm_params_override([
            'storage/data_nn_disambiguator/X_train.npy', \
            'storage/data_nn_disambiguator/X_test.npy', \
            'storage/data_nn_disambiguator/y_train.npy', \
            'storage/data_nn_disambiguator/y_test.npy', \
            'storage/trained_weights.py'])
        create_features_labels(save_to_disk=True)
        main()


if __name__ == '__main__':
    SentenceWeightParams.write_params()