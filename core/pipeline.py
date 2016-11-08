import nlp
import dill
import cPickle
import numpy as np
from datetime import datetime

class AdaTextPipeline(object):
    def __init__(self):
        step_sequence = ['Disambiguation', 'Tokenization', 'Lemmatization', 'Embedding', 'POSTagging', \
                         'NERClassification', 'DependencyParsing']

        print('[{}] AdaWordPipeline Execution Path:\n{}'.format(
            str(datetime.now()),))

        disambiguator_weights = np.load('../storage/sentence_disambiguator/trained_weights.npy')
        with open('../storage/sentence_disambiguator/brown_tag_distribution.pkl') as fp:
            disambiguator_tag_counts = cPickle.load(fp)
        with open('../storage/sentence_disambiguator/brown_tag_order.pkl') as fp:
            disambiguator_tag_order = cPickle.load(fp)

        self.disambiguator = nlp.AdaSentenceDisambiguator(
            disambiguator_weights, disambiguator_tag_counts, disambiguator_tag_order)

        self.AdaSentencePipeline = AdaSentencePipeline()

    def do(self, text):
        ada = self.AdaSentencePipeline()
        sentences = self.disambiguator(text)

        resp = {}
        for i, sentence in enumerate(sentences):
            sent_i_json = ada.do(sentence)
            resp['sentence_{}'.format(i)] = sent_i_json
        return resp


class AdaSentencePipeline(object):
    def __init__(self):
        step_sequence = ['Tokenization', 'Lemmatization', 'Embedding', 'POSTagging', \
                         'NERClassification','DependencyParsing']  # CoRefClassication

        print('[{}] AdaWordPipeline Execution Path:\n{}'.format(
            str(datetime.now()), '\t\n'.join(step_sequence)))

        embedder_weights = np.load('../storage/word_embedding/glove_weights_300d.npy')
        with open('../storage/word_embedding/glove_vocab_300d.pkl') as fp:
            embedder_vocab = cPickle.load(fp)

        pos_tagger_weights = np.load('../storage/pos_tagger/trained_weights.npy')

        with open('../storage/ner/gen_params_set.pkl') as fp:
            ner_gen_params = cPickle.load(fp)
        with open('../storage/ner/nn_params_set.pkl') as fp:
            ner_nn_params = dill.load(fp)

        self.tokenizer = nlp.AdaTokenizer()
        self.lemmatizer = nlp.AdaLemmatizer()
        self.embedder = nlp.AdaEmbedder(embedder_weights, embedder_vocab)
        self.pos_tagger = nlp.AdaPosTagger(pos_tagger_weights)
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
                 'embedding' : embedding,
                 '_timestamp' : str(datetime.now()) }
        return resp
