'''
Given many different datasets, use this function
as a platform for loading them all.

'''

import os
import sys
import nltk
local_ref = lambda x: os.path.join(os.path.dirname(__file__),  x)
sys.path.append(local_ref('../common'))
import util
import numpy as np

IGNORE_FILES = [
    '.DS_Store',
    'CONTENTS',
    'README',
    'categories.pickle',
    'cats.txt'
]

DEFAULT_TAG = 'UNK'


def brown_generator(loc=local_ref('../storage/sentence_disambiguation/raw_brown')):
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
        generator, num_docs=None, return_tags=False,
        return_sent_labels=False):
    ''' For each document tokenize the words and
        separate the conglomerate into words and
        tags.

        Args
        ----
        generator : Python generator
                    obj that returns the content
                    for each document.

        num_docs : int / None
                   number of documents to process
                   if None, return all

        return_tags : boolean
                      if true, tags are returned w/
                      words in a tuple

        return_sent_labels : boolean
                            if true, a boolean is
                            returned for whether the
                            token repr. EOS.
        Returns
        -------
        data : Python generator
               yields a single token (w/ POS)
    '''

    default_splitter = nltk.data.load(
        'tokenizers/punkt/english.pickle')

    for (doc_id, doc_con) in generator:
        if num_docs:  # early stop
            if doc_id > num_docs:
                break

        # generate labels using nltk for sentence splitting
        doc_sents = default_splitter.tokenize(doc_con.strip())

        for doc_sent in doc_sents:
            doc_sent = doc_sent.split()
            num_tokens_in_sent = len(doc_sent)

            for cur_token_id, cur_doc_token in enumerate(doc_sent):
                cur_doc_word, cur_doc_pos = \
                    nltk.tag.str2tuple(cur_doc_token, sep='/')

                if cur_doc_pos is None:
                    print "found an item with no tag: {}".format(cur_doc_token)
                    cur_doc_pos = DEFAULT_TAG

                if util.is_int(cur_doc_pos) or cur_doc_pos == '':
                    cur_doc_pos = DEFAULT_TAG

                # last token in sentence
                if cur_token_id == num_tokens_in_sent - 1:
                    cur_sent_label = 1
                else:
                    cur_sent_label = 0

                if return_tags and return_sent_labels:
                    yield((cur_doc_word, cur_doc_pos, cur_sent_label))
                elif return_tags:
                    yield((cur_doc_word, cur_doc_pos))
                elif return_sent_labels:
                    yield((cur_doc_word, cur_sent_label))
                else:
                    yield cur_doc_word

