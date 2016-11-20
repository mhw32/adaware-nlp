''' Convert JSONs of semantic information into a graph
    of nodes with semantically related edges.
'''

from graph import GraphNode, Graph
from spreading import spreading_activation


def semantic_priors():
    priors = defaultdict(lambda: 0)
    priors['adjacency-1'] = 0.15
    priors['adjacency-2'] = 0.1
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

    def sent_json_to_graph(self, json);
        ''' Given an AdaSentencePipeline response <see pipeline.py>,
            parse the json into nodes and edges in the graph
        '''
        num_tokens = json['num_tokens']
        tokens = json['tokens']
        lemmas = json['lemmas']
        pos_tags = json['pos_tags']
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
                    'sentence')

        # add special "alpha" semantic tags
        for token_i in range(num_tokens):
            # token_i = relative loc in sentence
            # true_token_i = absolute loc in graph
            true_token_i = offset + token_i

            # add adjacency-1 links
            if token_i > 0 and token_i < num_tokens - 1:
                self._add_link(
                    true_token_i,
                    true_token_i+1,
                    self.priors['adjacency-1'],
                    'adjacency-1')
                self._add_link(
                    true_token_i,
                    true_token_i-1,
                    self.priors['adjacency-1'],
                    'adjacency-1')

            # add adjacency-1 links
            if token_i > 1 and token_i < num_tokens - 2:
                self._add_link(
                    true_token_i,
                    true_token_i+2,
                    self.priors['adjacency-2'],
                    'adjacency-2')
                self._add_link(
                    true_token_i,
                    true_token_i-2,
                    self.priors['adjacency-2'],
                    'adjacency-2')

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
                    self.priors['lemma'],
                    'pos')
            self.pos_hash[pos].append(true_token_i)

            # add ner links
            ner = ner_tags[token_i]
            for ner_i in self.ner_hash[ner]:
                self._add_link(
                    true_token_i,
                    ner_i,
                    self.priors['lemma'],
                    'ner')
            self.ner_hash[ner].append(true_token_i)

            # add dep links
            # TODO

    def _add_link(self, i, j, prior, link_type):
        self.graph.add_edge(i, j, prior, edge_type=link_type)
