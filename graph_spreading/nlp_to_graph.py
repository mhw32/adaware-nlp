''' Convert JSONs of semantic information into a graph
    of nodes with semantically related edges.
'''

import pdb
import numpy as np
from graph import Graph
from collections import defaultdict
from datetime import datetime


def semantic_priors():
    priors = defaultdict(lambda: 0)
    priors['adjacency-1'] = 0.15
    priors['adjacency-2'] = 0.1
    priors['adjacency-3'] = 0.08
    priors['lemma'] = 0.24
    priors['name'] = 0.75
    priors['coref'] = 0.85
    priors['pos'] = 0.05
    priors['dependency'] = 1.0
    priors['sentence'] = 0.05
    return priors


class SemanticGraph(object):
    def __init__(self):
        self.graph = Graph()
        self.priors = semantic_priors()

        # these are really inefficient (ignore it)
        self.lemma_hash = defaultdict(lambda: [])
        self.name_hash = defaultdict(lambda: [])
        self.pos_hash = defaultdict(lambda: [])

    def doc_json_to_graph(self, json):
        ''' Given an AdaTextPipeline response <see pipeline.py>,
            parse the json into nodes and edges in the graph
        '''
        num_sentences = json['num_sentences']
        for sent_i in range(num_json):
            self.sent_json_to_graph(json['sentence_{}'.format(sent_i)])

    def sent_json_to_graph(self, json, sent_i=0):
        ''' Given an AdaSentencePipeline response <see pipeline.py>,
            parse the json into nodes and edges in the graph
        '''
        num_tokens = len(json['tokens'])
        tokens = json['tokens']
        lemmas = json['lemmas']
        pos_tags = json['pos_tags'][0]
        ner_tags = json['ner_tags']
        dep_tags = json['dep_tags']

        # store how many nodes are in currently
        offset = self.graph.num_nodes

        # add each token as a node first
        for token_i in range(num_tokens):
            self.graph.add_node(tokens[token_i])

        # add sentence tags
        for token_i in range(num_tokens-1):
            true_token_i = offset + token_i
            for token_j in range(token_i+1, num_tokens):
                true_token_j = offset + token_j
                self._add_link(
                    true_token_i,
                    true_token_j,
                    self.priors['sentence'],
                    'sentence-{}'.format(sent_i))

        # add special "alpha" semantic tags
        for token_i in range(num_tokens):
            # token_i = relative loc in sentence
            # true_token_i = absolute loc in graph
            true_token_i = offset + token_i

            # add adjacency-1 through adjacency-3 links
            for adj_i in [0, 1, 2]:
                if token_i > adj_i and token_i < num_tokens - (adj_i+1):
                    self._add_link(
                        true_token_i,
                        true_token_i+adj_i+1,
                        self.priors['adjacency-{}'.format(adj_i+1)],
                        'adjacency-{}'.format(adj_i+1))
                    self._add_link(
                        true_token_i,
                        true_token_i-(adj_i+1),
                        self.priors['adjacency-{}'.format(adj_i+1)],
                        'adjacency-{}'.format(adj_i+1))

            # add lemma links
            lemma = lemmas[token_i]
            for lemma_i in self.lemma_hash[lemma]:
                self._add_link(
                    true_token_i,
                    lemma_i,
                    self.priors['lemma'],
                    'lemma')
            self.lemma_hash[lemma].append(true_token_i)

            # add pos links
            pos = pos_tags[token_i]
            for pos_i in self.pos_hash[pos]:
                self._add_link(
                    true_token_i,
                    pos_i,
                    self.priors['pos'],
                    'pos-{}'.format(pos.lower()))
            self.pos_hash[pos].append(true_token_i)

            # add ner links
            ner = ner_tags[token_i]
            if ner != 'O':  # ignore OTHER tags
                for ner_i in self.name_hash[ner]:
                    self._add_link(
                        true_token_i,
                        ner_i,
                        self.priors['name'],
                        'ner-{}'.format(ner.lower()))
                self.name_hash[ner].append(true_token_i)

            # add dep links
            # FIXME: for now I'm treating all dependencies
            # the same -- definitely not okay.
            cur_deps_dict = dep_tags[token_i]['deps']
            for dep_key, dep_vals in cur_deps_dict.iteritems():
                for dep_val in dep_vals:
                    self._add_link(
                        true_token_i,
                        offset + dep_val - 1,  # 1 offset for some reason
                        self.priors['dependency'],
                        'dependency-{}'.format(dep_key))

        print('[{}] Sentence ({}) Added to Graph'.format(str(datetime.now()), sent_i))
        print('[{}] Graph State ({} nodes | {} edges)'.format(
            str(datetime.now()),
            self.graph.num_nodes,
            self.graph.num_edges))

    def _add_link(self, i, j, prior, link_type):
        self.graph.add_edge(i, j, prior, edge_type=link_type)

    def run(self, source_list):
        ''' list of indexes for source nodes '''
        print('[{}] Begin Spreading Activation'.format(str(datetime.now())))
        similar_nodes, dissimilar_nodes = self.graph.spreading_activation(source_list)
        print('[{}] Spreading Activation Converged'.format(str(datetime.now())))
        return similar_nodes, dissimilar_nodes

    def api_run(self, source_list):
        similar_nodes, dissimilar_nodes = self.run(source_list)

        similar_indexs = [node.index for node in similar_nodes]
        similar_activation = [node.activation for node in similar_nodes]
        similar_tokens = [node.value for node in similar_nodes]

        dissimilar_indexs = [node.index for node in dissimilar_nodes]
        dissimilar_activation = [node.activation for node in dissimilar_nodes]
        dissimilar_tokens = [node.value for node in dissimilar_nodes]

        response = {
            '_timestamp' : str(datetime.now()),
            'similar_indexes' : similar_indexs,
            'similar_activation' : similar_activation,
            'similar_tokens' : similar_tokens,
            'dissimilar_indexes' : dissimilar_indexs,
            'dissimilar_activation' : dissimilar_activation,
            'dissimilar_tokens' : dissimilar_tokens,
        }

        return response
