'''
Given many different datasets, use this function
as a platform for loading them all.

'''

import os
import nltk
import util
import numpy as np
# from sklearn.metrics import roc_curve

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


def split_data(in_data, out_data=None, frac=0.80):
    ''' splitting dataset into training/testing sets

        Args
        ----
        in_data : np array
                  input features x rows

        out_data : np array
                   output features x rows

        frac : float
               %  to keep in training set

        Returns
        -------
        (training_inputs, testing_inputs),
        [(training_outputs, testing_outputs)]
    '''

    num_data = in_data.shape[1]
    num_tr_data = int(np.floor(num_data * frac))
    indices = np.random.permutation(num_data)

    tr_idx, te_idx = indices[:num_tr_data], indices[num_tr_data:]
    tr_in_data, te_in_data = in_data[:, tr_idx], in_data[:, te_idx]

    if not out_data is None:
        tr_out_data, te_out_data = out_data[:, tr_idx], out_data[:, te_idx]
        return (tr_in_data, te_in_data), (tr_out_data, te_out_data)

    return (tr_in_data, te_in_data)

def get_auc(outputs, probas):
    ''' AUC is a common metric for binary classification
        methods by comparing true & false positive rates

        Args
        ----
        outputs : numpy array
                 true outcomes (OxTxN)

        probas : numpy array
                 predicted probabilities (OxTxN)

        Returns
        -------
        auc : integer
    '''

    fpr, tpr, _ = roc_curve(outputs, probas[:, 1])
    return auc(fpr, tpr)
