import nlp
import numpy as np
from datetime import datetime

class AdaTextPipeline(object):
    def __init__(self):
        step_sequence = ['Disambiguating', 'Tokenization', 'Lemmatization', 'Embedding', 'POSTagging', \
                         'NERClassifying', 'CoRefClassifying', 'DependencyParsing']

        print('[{}] AdaWordPipeline Execution Path:\n{}'.format(
            str(datetime.now()),))

        embedder_weights = np.load('../storage/word_embedding/glove_weights_300d.npy')
        with open('../storage/word_embedding/glove_vocab_300d.pkl') as fp:
            embedder_vocab = cPickle.load(fp)

        disambiguator_weights = np.load('../storage/sentence_disambiguator/trained_weights.npy')

        with open('../storage/sentence_disambiguator/brown_tag_distribution.pkl') as fp:
            disambiguator_tag_counts = cPickle.load(fp)

        with open('../storage/sentence_disambiguator/brown_tag_order.pkl') as fp:
            disambiguator_tag_order = cPickle.load(fp)

        pos_tagger_weights = np.load('../storage/pos_tagger/trained_weights.npy')

        self.disambiguator = nlp.AdaSentenceDisambiguator(
            disambiguator_weights, disambiguator_tag_counts, disambiguator_tag_order)
        self.lemmatizer = nlp.AdaLemmatizer()
        self.embedder = nlp.AdaEmbedder(embedder_weights, embedder_vocab)
        self.tokenizer = nlp.AdaTokenizer()
        self.pos_tagger = nlp.AdaPosTagger(pos_tagger_weights)
        # self.ner_classifier = nlp.AdaNerClassifier()
        # self.coref_classifier = nlp.AdaCoRefClassifier()
        self.dep_parser = nlp.AdaDependencyParser()


    def do(self, text):
        pass


class AdaSentencePipeline(object):
    def __init__(self):
        step_sequence = ['Tokenization', 'Lemmatization', 'Embedding', 'POSTagging', \
                         'NERClassifying', 'CoRefClassifying', 'DependencyParsing']

        print('[{}] AdaWordPipeline Execution Path:\n{}'.format(
            str(datetime.now()), '\t\n'.join(step_sequence)))

        embedder_weights = np.load('../storage/word_embedding/glove_weights_300d.npy')
        with open('../storage/word_embedding/glove_vocab_300d.pkl') as fp:
            embedder_vocab = cPickle.load(fp)

        self.lemmatizer = nlp.AdaLemmatizer()
        self.embedder = nlp.AdaEmbedder(embedder_weights, embedder_vocab)
        self.tokenizer = nlp.AdaTokenizer()
        self.pos_tagger = nlp.AdaPosTagger(pos_tagger_weights)
        # self.ner_classifier = nlp.AdaNerClassifier()
        # self.coref_classifier = nlp.AdaCoRefClassifier()
        self.dep_parser = nlp.AdaDependencyParser()

    def do(self, sentence):
        pass


class AdaWordPipeline(object):
    def __init__(self):
        step_sequence = ['Lemmatization', 'Embedding']

        print('[{}] AdaWordPipeline Execution Path:\n{}'.format(
            str(datetime.now()), '\t\n'.join(step_sequence)))

        embedder_weights = np.load('../storage/word_embedding/glove_weights_300d.npy')
        with open('../storage/word_embedding/glove_vocab_300d.pkl') as fp:
            embedder_vocab = cPickle.load(fp)

        self.lemmatizer = nlp.AdaLemmatizer()
        self.embedder = nlp.AdaEmbedder(embedder_weights, embedder_vocab)

    def do(self, word):
        lemma = self.lemmatizer.do(word)
        embedding = self.embedder.do(word)
        resp = { 'token' : word,
                 'lemma' : lemma,
                 'embedding' : embedding
                 '_timestamp' : str(datetime.now()) }
        return resp
