import os
import sys
import numpy as np
import cPickle
import dill
from datetime import datetime
from gensim import models


local_ref = lambda x: os.path.join(os.path.dirname(__file__),  x)


class ParamBank(object):
    ''' Store all weights in this class ... this is a nightmare
        to load but what can we do? '''

    def __init__(self):
        self.bank = {}

    def load(self):
        # disambiguator params
        print('[{}] Loading <disambiguator_weights>'.format(str(datetime.now())))
        disambiguator_weights = np.load(local_ref('../storage/sentence_disambiguation/trained_weights.npy'))
        print('[{}] Loading <disambiguator_tag_counts>'.format(str(datetime.now())))
        with open(local_ref('../storage/sentence_disambiguation/brown_tag_distribution.pkl')) as fp:
            disambiguator_tag_counts = cPickle.load(fp)
        print('[{}] Loading <disambiguator_tag_order>'.format(str(datetime.now())))
        with open(local_ref('../storage/sentence_disambiguation/brown_tag_order.pkl')) as fp:
            disambiguator_tag_order = cPickle.load(fp)

        # glove embedding params
        print('[{}] Loading <embedder_weights>'.format(str(datetime.now())))
        embedder_weights = np.load(local_ref('../storage/word_embedding/glove_weights_300d.npy'))
        print('[{}] Loading <embedder_vocab>'.format(str(datetime.now())))
        with open(local_ref('../storage/word_embedding/glove_vocab_300d.pkl')) as fp:
            embedder_vocab = cPickle.load(fp)

        # part-of-speech params
        print('[{}] Loading <pos_tagger_weights>'.format(str(datetime.now())))
        pos_tagger_weights = dict(np.load(local_ref('../storage/pos_tagger/pos_trained_weights.npz')))
        print('[{}] Loading <wordvec_model>'.format(str(datetime.now())))
        wordvec_model = models.Word2Vec.load_word2vec_format(
            local_ref('../storage/pos_tagger/GoogleNews-vectors-negative300.bin'), binary=True)

        # NER params
        print('[{}] Loading <ner_gen_params>'.format(str(datetime.now())))
        with open(local_ref('../storage/ner/gen_params_set.pkl')) as fp:
            ner_gen_params = cPickle.load(fp)
        print('[{}] Loading <ner_nn_params>'.format(str(datetime.now())))
        with open(local_ref('../storage/ner/nn_params_set.dill')) as fp:
            ner_nn_params = dill.load(fp)

        # stanford dep parser params
        print('[{}] Loading <dep_path_to_jar>'.format(str(datetime.now())))
        dep_path_to_jar = local_ref('../storage/dependency_parsing/stanford-parser.jar')
        print('[{}] Loading <dep_path_to_models_jar>'.format(str(datetime.now())))
        dep_path_to_models_jar = local_ref('../storage/dependency_parsing/stanford-parser-3.5.2-models.jar')

        self.bank['disambiguator_weights'] = disambiguator_weights
        self.bank['disambiguator_tag_counts'] = disambiguator_tag_counts
        self.bank['disambiguator_tag_order'] = disambiguator_tag_order
        self.bank['embedder_weights'] = embedder_weights
        self.bank['embedder_vocab'] = embedder_vocab
        self.bank['pos_tagger_weights'] = pos_tagger_weights
        self.bank['wordvec_model'] = wordvec_model
        self.bank['ner_gen_params'] = ner_gen_params
        self.bank['ner_nn_params'] = ner_nn_params
        self.bank['dep_path_to_jar'] = dep_path_to_jar
        self.bank['dep_path_to_models_jar'] = dep_path_to_models_jar

    def get(self, key):
        if key in self.bank:
            return self.bank[key]
        return None
