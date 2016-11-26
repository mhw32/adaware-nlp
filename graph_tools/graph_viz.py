''' Toolkit for visualizing a semantic graph '''

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cStringIO import StringIO


def visualize_graph(graph, savepath=None):
    ''' pass in a Graph object '''

    node_objs = graph.nodes
    edge_hash = graph.edges

    G = nx.Graph()  # init network obj

    # add all nodes
    node_activations = []
    for node in node_objs:
        G.add_node(node.value)
        node_activations.append(node.activation)

    for k,v in edge_hash.iteritems():
        _, nd_i, nd_j = k.split('_')
        node_i = graph.get_node(index=int(nd_i))
        node_j = graph.get_node(index=int(nd_j))

        weight = 0
        for k2, v2 in v.iteritems():
            weight += float(v2)

        G.add_edge(
            node_i.value,
            node_j.value,
            weight=weight
        )

    node_labels = {node:node for node in G.nodes()}
    edge_labels=dict([((u,v,),d['weight'])
                     for u,v,d in G.edges(data=True)])

    edge_colors = ['black' if float(d['weight']) < 1.0 else 'red' for _, _,d in G.edges(data=True)]

    pos=nx.spring_layout(G)
    nx.draw_networkx_labels(
        G,
        pos,
        labels=node_labels,
        font_color='w'
    )
    nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=edge_labels
    )
    nx.draw(
        G,
        pos,
        node_color=node_activations,
        node_size=1500,
        edge_color=edge_colors,
        edge_cmap=plt.cm.Reds
    )

    if savepath:
        plt.savefig(savepath)
        return
    plt.show()
