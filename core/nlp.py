import numpy as np

# -- replace me after wrap in a pip library --
import sys
sys.path.append('../sentence_disambiguator')
sys.path.append('../word_tokenizer')
sys.path.append('../word_embedding')
sys.path.append('../pos_tagger')
sys.path.append('../lemmatizer')
sys.path.append('../ner')
sys.path.append('../coref')
sys.path.append('../dependency_parsing')

import disambiguator
import treebank_tokenizer
import glove
import nltk_lemmatizer
import pos_tagger


class AdaWordTokenizer(object):
    def __init__(self):
        self.tokenizer = treebank_tokenizer.TreeBankTokenizer

    def do(self, sentence):
        ''' sentence : string, text with spaces '''
        return self.tokenizer(sentence)

    def do_all(self, sentences):
        ''' sentences : list, many text with spaces '''
        for sentence in sentences:
            yield self.do(sentence)


class AdaSentenceDisambiguator(object):
    def __init__(self, weights, tag_counts, tag_order):
        self.weights = weights
        self.tag_counts = tag_counts
        self.tag_order = tag_order

    def do(self, sentence):
        ''' sentence : list, list of words '''
        labels = disambiguator.predict_from_tokens(
            sentence, tag_counts, tag_order)

        return disambiguator.split_into_sentences(sentence, labels)

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
        self.model = nltk_lemmatizer.Lemmatizer

    def do(self, word, pos='n'):
        ''' word : string, a single token '''
        return self.model.lemmatize(word, pos)

    def do_all(self, words, poses=None):
        ''' words : list, many tokens '''
        for i, word in enumerate(words):
            self.do(word, pos=poses[i] if poses else 'n')


class AdaPOSTagger(object):
    def __init__(self, weights, wordvec):
        self.weights = weights
        self.wordvec = wordvec

    def do(self, sentence):
        ''' sentence : list, list of tokens '''
        return pos_tagger.predict_from_sentences([sentence])

    def do_all(self, sentences):
        ''' sentence : 2D list, list of list of tokens '''
        return pos_tagger.predict_from_sentences(sentences)


class AdaNERClassifier(object):
    def __init__(self, gen_params, nn_params):
        self.model = NeuralNER(gen_params, nn_params)

    def do(self, sentence):
        ''' sentence : list, list of tokens '''
        return self.model.ner(sentence)

    def do_all(self, sentences):
        ''' sentence : 2D list, list of list of tokens '''
        for i, sentence in enumerate(sentences):
            return self.model.ner(sentence)


class AdaCoRefClassifier(object)
    return 'Coming Soon! This is not yet available.'


class AdaDependencyParser(object):
    def __init__(self):
        self.model = nltk_dep_parser.DependencyParser

    def do(self, sentence):
        ''' sentence : list, list of tokens '''
        return self.model.lst_parse(sentence)

    def do_all(self, sentences):
        ''' sentence : 2D list, list of list of tokens '''
        for i, sentence in enumerate(sentences):
            self.do(sentence)
