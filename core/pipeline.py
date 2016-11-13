import os
import sys
import nlp
import dill
import cPickle
import numpy as np
from gensim import models
from datetime import datetime

local_ref = lambda x: os.path.join(os.path.dirname(__file__),  x)

class AdaTextPipeline(object):
    def __init__(self, bank):
        step_sequence = ['Disambiguation', 'Tokenization', 'Lemmatization', 'Embedding', 'POSTagging', \
                         'NERClassification', 'DependencyParsing']

        print('[{}] AdaWordPipeline Execution Path:'.format(str(datetime.now())))
        for step in step_sequence:
            print('[{}] {}'.format(str(datetime.now())), step)

        disambiguator_weights = bank.get(['disambiguator_weights'])
        disambiguator_tag_counts = bank.get(['disambiguator_tag_counts'])
        disambiguator_tag_order = bank.get(['disambiguator_tag_order'])

        self.disambiguator = nlp.AdaSentenceDisambiguator(
            disambiguator_weights, disambiguator_tag_counts, disambiguator_tag_order)

        self.AdaSentencePipeline = AdaSentencePipeline()

    def do(self, text):
        ada = self.AdaSentencePipeline()
        sentences = self.disambiguator(text)

        resp = {}
        for i, tokens_lst in enumerate(sentences):
            sent_i_json = ada._do_tokens(tokens_lst)
            resp['sentence_{}'.format(i)] = sent_i_json
        return resp


class AdaSentencePipeline(object):
    def __init__(self, bank):
        step_sequence = ['Tokenization', 'Lemmatization', 'Embedding', 'POSTagging', \
                         'NERClassification','DependencyParsing']  # CoRefClassication

        print('[{}] AdaWordPipeline Execution Path:'.format(str(datetime.now())))
        for step in step_sequence:
            print('[{}] {}'.format(str(datetime.now())), step)

        embedder_weights = bank.get(['embedder_weights'])
        embedder_vocab = bank.get(['embedder_vocab'])
        pos_tagger_weights = bank.get(['pos_tagger_weights'])
        wordvec_model = bank.get(['wordvec_model'])
        ner_gen_params = bank.get(['ner_gen_params'])
        ner_nn_params = bank.get(['ner_nn_params'])

        self.tokenizer = nlp.AdaWordTokenizer()
        self.lemmatizer = nlp.AdaLemmatizer()
        self.embedder = nlp.AdaWordEmbedder(embedder_weights, embedder_vocab)
        self.pos_tagger = nlp.AdaPosTagger(pos_tagger_weights, wordvec_model)
        self.ner_classifier = nlp.AdaNerClassifier(ner_gen_params, ner_nn_params)
        # self.coref_classifier = nlp.AdaCoRefClassifier()
        self.dep_parser = nlp.AdaDependencyParser()

    def do(self, sentence):
        tokens = self.tokenizer.do(sentence)
        lemmas = self.lemmatizer.do_all(tokens)
        embeddings = self.embedder.do_all(tokens)
        postags = self.pos_tagger.do(tokens)
        nertags = self.ner_classifier.do(tokens)
        deptags = self.dep_parser.do(tokens)

        resp = { 'tokens' : tokens,
                 'lemmas' : lemmas,
                 'embeddings' : embeddings,
                 'pos_tags' : postags,
                 'ner_tags' : nertags,
                 'dep_tags' : deptags,
                 '_timestamp' : str(datetime.now()) }

    def _do_tokens(self, tokens):
        lemmas = self.lemmatizer.do_all(tokens)
        embeddings = self.embedder.do_all(tokens)
        postags = self.pos_tagger.do(tokens)
        nertags = self.ner_classifier.do(tokens)
        deptags = self.dep_parser.do(tokens)

        resp = { 'lemmas' : lemmas,
                 'embeddings' : embeddings,
                 'pos_tags' : postags,
                 'ner_tags' : nertags,
                 'dep_tags' : deptags,
                 '_timestamp' : str(datetime.now()) }


class AdaWordPipeline(object):
    def __init__(self):
        step_sequence = ['Lemmatization', 'Embedding']

        print('[{}] AdaWordPipeline Execution Path:'.format(str(datetime.now())))
        for step in step_sequence:
            print('[{}] {}'.format(str(datetime.now())), step)

        embedder_weights = bank.get(['embedder_weights'])
        embedder_vocab = bank.get(['embedder_vocab'])

        self.lemmatizer = nlp.AdaLemmatizer()
        self.embedder = nlp.AdaWordEmbedder(embedder_weights, embedder_vocab)

    def do(self, word):
        lemma = self.lemmatizer.do(word)
        embedding = self.embedder.do(word)
        resp = { 'token' : word,
                 'lemma' : lemma,
                 'embedding' : embedding,
                 '_timestamp' : str(datetime.now()) }
        return resp
