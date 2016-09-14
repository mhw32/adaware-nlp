'''
Given many different datasets, use this function
as a platform for loading them all.

'''

import os
import nltk
import util

IGNORE_FILES = [
    '.DS_Store',
    'CONTENTS',
    'README',
    'categories.pickle'
]

DEFAULT_TAG = 'unk'


def safe_split(s, char='/'):
    if char in s:
        arr = s.split(char)
        if util.is_int(arr[-1]) or arr[-1].strip() == '':
            return (s, DEFAULT_TAG)
        return (arr[:-1], arr[-1])
    return (s, DEFAULT_TAG)


def brown_generator(loc='datasets/brown/raw'):
    ''' Establish a generator that returns the
        contents of a document as a generator.

        Raw text from:
        http://clu.uni.no/icame/manuals/BROWN/INDEX.HTM

        Args
        ----
        loc : string
              location of the raw files
    '''
    findex = 0
    for root, dirs, files in os.walk(
            loc, topdown=False):

        for fname in files:
            # bad files
            if fname in IGNORE_FILES:
                continue

            fpath = os.path.join(root, fname)
            with open(fpath, 'rb') as fp:
                fcontent = fp.read()
                yield (findex, fcontent)

            findex += 1


def load_data(
        generator, num_docs=None, return_tags=True):
    ''' For each document tokenize the words and
        separate the conglomerate into words and
        tags.

        Args
        ----
        generator : Python generator
                    obj that returns the content
                    for each document.

        num_docs : inn / None
                   number of documents to process
                   if None, return all

        return_tags : boolean
                      if true, tags are returned w/
                      words in a tuple

        Returns
        -------
        data : Python generator
               yields a single token (w/ POS)
    '''

    for (doc_id, doc_con) in generator:
        if num_docs:  # early stop
            if doc_id > num_docs:
                break

        doc_tokens = tokens = nltk.tokenize.word_tokenize(doc_con)
        for cur_doc_tokens in doc_tokens:
            cur_doc_word, cur_doc_pos = \
                safe_split(cur_doc_tokens, char='/')

            if return_tags:
                yield((cur_doc_word, cur_doc_pos))
            else:
                yield cur_doc_word


