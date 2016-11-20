#!/usr/bin/env python

def spreading_activation(graph, origin):
	threshold=0.5
	decay=0.1

	origin.activation = 1.0
	already_fired = set()
	need_firing = set([origin])

	while len(need_firing) > 0:
		root = need_firing.pop()
		# skip if this node has already been activated
		if root in already_fired:
			continue

		# "fire" the node
		for n in root.neighbors:
			weight = weights[origin, n]
			if weight == 0:
				continue
			n.activation += (root.activation * weight * decay)
			n.activation = max(1.0, n.activation)
			if n.activation >= threshold:
				need_firing.append(n)
		already_fired.append(root)

	return already_fired, set(graph.nodes) - already_fired