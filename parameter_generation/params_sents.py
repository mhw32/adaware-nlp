#!/usr/bin/env python

'''
    Executable to generate parameters for sentence
    disambiguation, namely the files:
    - storage/brown_tag_order.pkl
    - storage/brown_tag_distribution.pkl
'''

import os
import sys
project_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(project_path)

from disambiguator import *
from prompt import *

class SentenceParams:

    @staticmethod
    def write_params():
        confirm_params_override(['storage/brown_tag_order.pkl', 'storage/brown_tag_distribution.pkl'])
        init_prior_pos_proba(lexicon=None, simplify_tags=True, save_to_disk=True)

if __name__ == '__main__':
    SentenceParams.write_params()