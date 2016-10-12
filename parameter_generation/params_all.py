#!/usr/bin/env python

'''
    Executable to generate all parameters from scratch
    This will remove all files in the directory:
    - storage/
'''

import os
import glob

from prompt import confirm_params_override
from params_sents import SentenceParams
from params_sent_weights import SentenceWeightParams

if __name__ == '__main__':
    confirm_params_override(['storage/'], ensure_prompt=True)
    params_dir = os.path.dirname(os.path.abspath(__file__))
    
    for f in glob.glob('../storage/*'):
        os.remove(os.path.join(params_dir, f))

    SentenceParams.write_params()
    SentenceWeightParams.write_params()
