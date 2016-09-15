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
    'categories.pickle',
    'cats.txt'
]

DEFAULT_TAG = 'UNK'


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
                print "-"*30
                print "reading: {}".format(fpath)
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

        # fuck nltk. they word_tokenize off punctuation as well
        # doc_tokens = tokens = nltk.tokenize.word_tokenize(doc_con)

        for cur_doc_token in doc_con.split():
            cur_doc_word, cur_doc_pos = \
                nltk.tag.str2tuple(cur_doc_token, sep='/')

            if cur_doc_pos is None:
                print "found an item with no tag: {}".format(cur_doc_token)
                cur_doc_pos = DEFAULT_TAG

            if util.is_int(cur_doc_pos) or cur_doc_pos == '':
                cur_doc_pos = DEFAULT_TAG

            if return_tags:
                yield((cur_doc_word, cur_doc_pos))
            else:
                yield cur_doc_word


