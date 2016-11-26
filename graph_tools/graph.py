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
        self.label = 0
        # list of GraphNode objects
        self.edges = []


class Graph(object):
    def __init__(self):
        self.nodes = []
        self.map_name_to_index = {}
        self.num_nodes = 0
        self.num_edges = 0
        # b/c our large graphs will be fairly sparse, we
        # use a hash instead of 2x2 array
        self.edges = defaultdict(lambda: defaultdict(lambda: 0))

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

    def add_edge(self, i, j, weight, edge_type=None):
        ''' i, j are integers
            node_i and node_j are GraphNode objects '''
        node_i = self.get_node(index=i)
        node_j = self.get_node(index=j)
        node_i.edges.append(j)  # store index cuz cheaper
        node_j.edges.append(i)

        # use smaller one as first hash key
        hash_key = self._edge_hash(i, j)
        self.edges[hash_key][edge_type] += weight
        self.num_edges += 1

    def get_edge(self, i, j, edge_type=None):
        ''' i, j are integers '''
        hash_key = self._edge_hash(i, j)
        if edge_type is not None:
            return self.edges[hash_key][edge_type]
        return self.edges[hash_key]

    def del_edge(self, i, j, edge_type=None):
        ''' i, j are integers
            node_i and node_j are GraphNode objects '''

        hash_key = self._edge_hash(i, j)
        should_remove_node = False

        if edge_type is not None:
            self.edges[hash_key][edge_type] = 0
            if len(self.edges[hash_key].values()) == 0:
                should_remove_node = True
            self.num_edges -= 1
        else:  # delete all
            should_remove_node = True
            self.num_edges -= len(self.edges[hash_key].keys())
            self.edges[hash_key] = defaultdict(lambda: 0)

        if should_remove_node:
            # removes real edge
            node_i = self.get_node(index=i)
            node_j = self.get_node(index=j)
            node_i.remove(j)
            node_j.remove(i)

    def spreading_activation(self, origins_i, threshold=0.5, decay=0.05):
        ''' origins_i must be a list of integers representing indexes of
            source nodes '''
        origins = [self.get_node(index=node_i) for node_i in origins_i]

        for o in origins:
            o.activation = 1.0

        already_fired = set()
        need_firing = origins

        while len(need_firing) > 0:
            root = need_firing.pop(0)
            # skip if this node has already been activated
            if root in already_fired:
                continue

            # "fire" the node
            for n in root.edges:
                neighbor_node = self.get_node(index=n)
                edge_hash = self._edge_hash(root.index, n)
                weight = sum(self.edges[edge_hash].values())  # naive linear sum

                if weight == 0:
                    continue

                neighbor_node.activation += (root.activation * weight * decay)
                neighbor_node.activation = max(1.0, neighbor_node.activation)

                if neighbor_node.activation >= threshold:
                    need_firing.append(neighbor_node)

            already_fired.add(root)
            root.label = 1
