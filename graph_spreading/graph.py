''' Functions to define and easily work with graphs.
    In our case, graphs will be defined by two structures:
        1) an array of GraphNode classes
        2) a hash of edge pairs to weights
'''

from collections import defaultdict


class GraphNode(object):
    ''' a node contains a single value which it stores.
        this may represent a name or a property, etc. A
        graph node also has an activation, which shows
        if the graph is active or not '''

    def __init__(self, value, index=None):
        self.value = value
        self.index = index
        self.activation = 0
        # list of GraphNode objects
        self.edges = []


class Graph(object):
    def __init__(self):
        self.nodes = []
        self.map_name_to_index = {}
        self.num_nodes = 0
        # b/c our large graphs will be fairly sparse, we
        # use a hash instead of 2x2 array
        self.edges = defaultdict(lambda: 0)

    def add_node(self, name):
        ''' nodes by default have no edges '''
        node = GraphNode(name, index=self.num_nodes)
        self.map_name_to_index[name] = self.num_nodes
        self.nodes.append(node)
        self.num_nodes += 1

    def get_node(self, name=None, index=None):
        if name is not None:
            index = self.map_name_to_index[name]

        if index is not None:
            return self.nodes[index]
        return None

    def del_node(self, name=None, index=None):
        ''' WARNING: deleting a node deletes all related edges '''
        if name is not None:
            index = self.map_name_to_index[name]

        if index is not None:
            node_i = self.get_node(index=index)
            # delete all connections
            for j in node_i.edges:
                node_j = self.get_node(index=j)
                self.del_edge(node_i, node_j)

            self.nodes.pop(index)
            return True
        return False

    def _edge_hash(self, i, j):
        # i, j are integers
        if i <= j:
            return 'e_{}_{}'.format(str(i), str(j))
        else:
            return 'e_{}_{}'.format(str(j), str(i))

    def add_edge(self, node_i, node_j, weight):
        ''' node_i and node_j are GraphNode objects '''
        node_i.edges.append(node_j.index)  # store index cuz cheaper
        node_j.edges.append(node_i.index)

        # use smaller one as first hash key
        hash_key = self._edge_hash(node_i.index, node_j.index)
        self.edges[hash_key] = weight

    def get_edge(self, node_i, node_j):
        ''' node_i and node_j are GraphNode objects '''
        hash_key = self._edge_hash(node_i.index, node_j.index)
        return self.edges[hash_key]

    def del_edge(self, node_i, node_j):
        ''' node_i and node_j are GraphNode objects '''
        node_i.remove(node_j.index)
        node_j.remove(node_i.index)
        hash_key = self._edge_hash(node_i.index, node_j.index)
        del self.edges[hash_key]
