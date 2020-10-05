import sys
import random

random_nodes = [0, 1, 2]
random_conns = {(0, 1)}
valid_node_coll = {0: [True, True], 1: [True, True], 2: [False, False]}
while len(random_conns) < 20:
    r_conn_start, r_conn_end = random.sample(random_nodes, k=2)

    if r_conn_start == 1 or r_conn_end == 0:
        continue

    if (r_conn_start, r_conn_end) not in random_conns:
        random_conns.add((r_conn_start, r_conn_end))

        if not valid_node_coll[r_conn_start][0]:
            valid_node_coll[r_conn_start][0] = True

        if not valid_node_coll[r_conn_end][1]:
            valid_node_coll[r_conn_end][1] = True

        if all([valid[0] == valid[1] for valid in valid_node_coll.values()]):
            r_conn_start, r_conn_end = random.sample(random_nodes, k=2)
            while r_conn_start == 1 or r_conn_end == 0:
                r_conn_start, r_conn_end = random.sample(random_nodes, k=2)

            new_node = max(random_nodes) + 1
            random_nodes.append(new_node)

            random_conns.add((r_conn_start, new_node))
            random_conns.add((new_node, r_conn_end))

            valid_node_coll[new_node] = [False, False]

node_dependencies = dict()
for conn in random_conns:
    if conn[1] in node_dependencies:
        node_dependencies[conn[1]].add(conn[0])
    else:
        node_dependencies[conn[1]] = {conn[0]}

########

from graphviz import Digraph
# Create Digraph, setting name and graph orientaion
dot = Digraph(name='tempgraph', graph_attr={'rankdir': 'TB'})

# Traverse all bp graph genes, adding the nodes and edges to the graph
for conn in random_conns:
    dot.edge(str(conn[0]), str(conn[1]))

# Render created dot graph, optionally showing it
#dot.render(filename='tempgraph.svg', view=True, cleanup=True, format='svg')


#node_dependencies = {2: {0, 3, 4, 5, 7}, 1: {0, 2, 3, 6}, 7: {0}, 5: {2, 3, 4}, 3: {0, 2, 4, 5}, 4: {0, 2, 5}, 6: {2}}

print(node_dependencies)

node_deps = node_dependencies.copy()
node_deps[0] = set()

recurrent_conns = list()
graph_topology = list()

while True:
    print(f"node_deps: {node_deps}\n")

    # find all nodes in graph having no dependencies in current iteration
    dependencyless = set()
    for node, dep in node_deps.items():
        if len(dep) == 0:
            dependencyless.add(node)

    if not dependencyless:
        if node_deps:
            node_deps_min = None
            node_deps_excluded = None
            min_deps_count = sys.maxsize

            for node, node_deps_set in node_deps.items():
                if node == 1:
                    continue
                if len(node_deps_set) < min_deps_count:
                    min_deps_count = len(node_deps_set)
                    node_deps_min = {node: node_deps_set}
                    node_deps_excluded = {node}
                elif len(node_deps_set) == min_deps_count:
                    node_deps_min[node] = node_deps_set
                    node_deps_excluded.add(node)

            print("node_deps: {}".format(node_deps))
            print("node_deps_min: {}".format(node_deps_min))
            print("node_deps_excluded: {}".format(node_deps_excluded))

            node_deps_min_adj = node_deps_min.copy()
            for node in node_deps_min.keys():
                node_deps_min_adj[node] -= node_deps_excluded

            if not node_deps_min_adj:
                node_deps_min_adj = node_deps_min
            print(f"node_deps_min_adj: {node_deps_min_adj}")

            '''
            if len(node_deps_min_adj) > 1:
                remaining_nodes = set(node_deps_min_adj.keys())
                presence_count_in_other_deps = dict()
                for node in remaining_nodes:
                    other_remaining_nodes = remaining_nodes - {node}
                    presence_count_in_other_deps[node] = 0
                    for other_remaining_node in other_remaining_nodes:
                        if node in node_deps_min_adj[other_remaining_node]:
                            presence_count_in_other_deps[node] += 1
    
                node_to_clear = max(presence_count_in_other_deps.keys(), key=presence_count_in_other_deps.get)
            else:
                node_to_clear = next(iter(node_deps_min_adj))
            '''
            for node, node_deps_set in node_deps_min_adj.items():
                for dep_nodes in node_deps_set:
                    recurrent_conns.append((dep_nodes, node))
                node_deps[node] = set()

            print(f"node_deps: {node_deps}")
            print(f"recurrent_conns: {recurrent_conns}")

            print("")
            continue

        break


        '''
        if node_deps:
            min_deps_nodes = None
            min_deps_nodes_exclusion = None
            min_deps_count = sys.maxsize
            for node, node_deps_set in node_deps.items():
                if len(node_deps_set) < min_deps_count:
                    min_deps_nodes = node_deps_set
                    min_deps_count = len(node_deps_set)
                    if node not in node_deps_set:
                        min_deps_nodes_exclusion = {node}
                    else:
                        min_deps_nodes_exclusion = set()
                elif len(node_deps_set) == min_deps_count:
                    min_deps_nodes = min_deps_nodes.union(node_deps_set)
                    if node not in node_deps_set:
                        min_deps_nodes_exclusion.add(node)

            min_deps_nodes = min_deps_nodes - min_deps_nodes_exclusion
            if not min_deps_nodes:
                nodes_deps_len_dict = dict()
                for node_deps_set in node_deps.values():
                    if len(node_deps_set) == min_deps_count:
                        if node_deps_set in nodes_deps_len_dict:
                            nodes_deps_len_dict[node_deps_set] += 1
                        else:
                            nodes_deps_len_dict[node_deps_set] = 1
                min_deps_nodes = min(nodes_deps_len_dict.keys(), key=nodes_deps_len_dict.get)

            for node, node_deps_set in node_deps.items():
                node_deps[node] = node_deps_set - min_deps_nodes
                if not node_deps[node]:
                    for dep_node in min_deps_nodes:
                        recurrent_conns.append((dep_node, node))
            continue
        '''
    # Add dependencyless nodes of current generation to list
    graph_topology.append(dependencyless)

    # remove keys with empty dependencies and remove all nodes that are considered dependencyless from the
    # dependencies of other nodes in order to create next iteration
    for node in dependencyless:
        del node_deps[node]
    for node, dep in node_deps.items():
        node_deps[node] = dep - dependencyless


print("RESULTS:")
print(graph_topology)
print(random_conns)
print(recurrent_conns)
print("EXIT")
exit()