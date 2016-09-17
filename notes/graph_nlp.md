# Graph spreading

Link to paper: https://www.aaai.org/Papers/AAAI/1997/AAAI97-097.pdf

Need the user to provide a topic. The initially is composed of several nodes. Spreading activation groups semantically related nodes together.

Concepts are described in a doc (words, phrases, proper names) as nodes. Edges are semantic and topological relations between concepts.

	- specialiation relationships
	- association relationships
	- coreference relationships

Salient nodes are defined by a topic. Given a topic, the graph is searched for nodes semantically related to the topic --> spreading.

Set of nodes activated is a functionf link type and distance from the entry node.

Given a doc and 2 documents, nodes in each document are semantically related, then these nodes and relationships can be compared to establish similarities and differences

Given two graphs, intersection of actvated concepts between topics <-- too similar.