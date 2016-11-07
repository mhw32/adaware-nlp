import os
import dill
import cPickle
import numpy as np

class DictionaryManager(object):
    ''' Load the right weights necessary for any of our models '''
    def __init__(self):
        self.paths = []
        self.key_store = {}

    def add_file(self, key, path):
        file_name, file_ext = os.path.splitext(path)

        if file_ext == 'npy':
            data = np.load(path)
        elif file_ext == 'pkl':
            with open(file_ext) as fp:
                data = cPickle.load(fp)
        else:  # dill is most generic
            with open(file_ext) as fp:
                data = dill.load(fp)

        # add a path by the key
        self.key_store[key] = data
