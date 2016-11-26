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

    G = nx.MultiGraph()  # init network obj

    # add all nodes
    for node in node_objs:
        G.add_node(node.value)

    for k,v in edge_hash.iteritems():  
        _, nd_i, nd_j = k.split('_')
        node_i = graph.get_node(index=nd_i)
        node_j = graph.get_node(index=nd_j)

        for k2, v2 in v.iteritems():
            G.add_edge(
                node_i.value, 
                node_j.value, 
                weight=v2, 
                key=k2
            )


    d = nx.to_pydot(g)
    png_str = d.create_png()
    sio = StringIO()
    sio.write(png_str)
    sio.seek(0)

    img = mpimg.imread(sio)
    imgplot = plt.imshow(img)
