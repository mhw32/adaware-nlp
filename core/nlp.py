import numpy as np

# -- replace me after wrap in a pip library --
import os
import sys

local_ref = lambda x: os.path.join(os.path.dirname(__file__),  x)
sys.path.append(local_ref('../sentence_disambiguator'))
sys.path.append(local_ref('../word_tokenizer'))
sys.path.append(local_ref('../word_embedding'))
sys.path.append(local_ref('../pos_tagging'))
sys.path.append(local_ref('../lemmatizer'))
sys.path.append(local_ref('../ner'))
sys.path.append(local_ref('../coref'))
sys.path.append(local_ref('../dependency_parsing'))

import disambiguator
import treebank_tokenizer
import glove
import nltk_lemmatizer
import pos_tagger
from ner import NeuralNER
import nltk_dep_parser


class AdaWordTokenizer(object):
    def __init__(self):
        self.tokenizer = treebank_tokenizer.TreeBankTokenizer()

    def do(self, sentence):
        ''' sentence : string, text with spaces '''
        return self.tokenizer.tokenize(sentence)

    def do_all(self, sentences):
        ''' sentences : list, many text with spaces '''
        for sentence in sentences:
            yield self.do(sentence)


class AdaSentenceDisambiguator(object):
    def __init__(self, weights, tag_counts, tag_order):
        self.weights = weights
        self.tag_counts = tag_counts
        self.tag_order = tag_order
        self.tokenizer = AdaWordTokenizer()

    def do(self, sentence):
        ''' sentence : string '''
        tokens = self.tokenizer.do(sentence)
        didx, proba = disambiguator.predict_from_tokens(
            tokens, self.weights, self.tag_counts, self.tag_order)
        return disambiguator.split_into_sentences(tokens, didx, proba)

    def do_all(self, sentences):
        ''' sentence : list, list of list of words '''
        for sentence in sentences:
            yield self.do(sentence)


class AdaWordEmbedder(object):
    def __init__(self, weights, vocab):
        self.weights = weights
        self.vocab = vocab

    def do(self, word):
        ''' word : string, a single token '''
        return glove.get_vector(word, self.weights, self.vocab)

    def do_all(self, words):
        ''' words : list, many tokens '''
        for word in words:
            yield self.do(word)


class AdaLemmatizer(object):
    def __init__(self):
        self.model = nltk_lemmatizer.Lemmatizer()

    def do(self, word, pos='n'):
        ''' word : string, a single token '''
        return self.model.lemmatize(word, pos)

    def do_all(self, words, poses=None):
        ''' words : list, many tokens '''
        num_words = len(words)
        return [self.do(words[i], pos=poses[i] if poses else 'n') for i in range(num_words)]


class AdaPOSTagger(object):
    def __init__(self, weights, wordvec):
        self.weights = weights
        self.wordvec = wordvec

    def do(self, sentence):
        ''' sentence : list, list of tokens '''
        return pos_tagger.predict_from_sentences([sentence], params=self.weights, model=self.wordvec)

    def do_all(self, sentences):
        ''' sentence : 2D list, list of list of tokens '''
        return pos_tagger.predict_from_sentences(sentences, params=self.weights, model=self.wordvec)


class AdaNERClassifier(object):
    def __init__(self, gen_params, nn_params, wordvec):
        self.model = NeuralNER(gen_params, nn_params, wordvec)

    def do(self, sentence):
        ''' sentence : list, list of tokens '''
        return self.model.ner(sentence)

    def do_all(self, sentences):
        ''' sentence : 2D list, list of list of tokens '''
        for i, sentence in enumerate(sentences):
            return self.model.ner(sentence)


class AdaCoRefClassifier(object):
    def __init__(self):
        return 'Coming Soon! This is not yet available.'


class AdaDependencyParser(object):
    def __init__(self, path_to_jar, path_to_models_jar):
        self.model = nltk_dep_parser.DependencyParser(path_to_jar, path_to_models_jar)

    def do(self, sentence):
        ''' sentence : list, list of tokens '''
        return self.model.lst_parse(sentence)

    def do_all(self, sentences):
        ''' sentence : 2D list, list of list of tokens '''
        for i, sentence in enumerate(sentences):
            self.do(sentence)
