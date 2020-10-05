import sys
import tensorflow as tf
from .deepneat_genes import DeepNEATNode, DeepNEATConn


class DeepNEATModel(tf.keras.Model):
    """"""

    def __init__(self,
                 genome_graph,
                 preprocessing_layers,
                 input_shape,
                 input_layers,
                 output_layers,
                 recurrent_stateful,
                 recurrent_init,
                 input_scaling,
                 merge_method,
                 dtype,
                 *args,
                 **kwargs):
        """"""
        super().__init__(dtype=dtype, *args, **kwargs)
        # !FIXME disregard preprocessing layers for first prototype
        # !FIXME disregard input layers for first prototype

        node_dependencies = dict()
        node_configs = dict()

        for gene in genome_graph.values():
            if isinstance(gene, DeepNEATNode):
                node_configs[gene.node] = gene.layer
            elif gene.enabled:
                if gene.conn_end in node_dependencies:
                    node_dependencies[gene.conn_end].add(gene.conn_start)
                else:
                    node_dependencies[gene.conn_end] = {gene.conn_start}
        del node_configs[1]

        ########
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

        print(node_dependencies)
        ########

        from graphviz import Digraph
        # Create Digraph, setting name and graph orientaion
        dot = Digraph(name='tempgraph', graph_attr={'rankdir': 'TB'})

        # Traverse all bp graph genes, adding the nodes and edges to the graph
        for conn in random_conns:
            dot.edge(str(conn[0]), str(conn[1]))

        # Render created dot graph, optionally showing it
        dot.render(filename='tempgraph.svg', view=True, cleanup=True, format='svg')

        node_deps = node_dependencies.copy()
        node_deps[0] = set()

        recurrent_conns = list()
        graph_topology = list()

        while True:
            # find all nodes in graph having no dependencies in current iteration
            dependencyless = set()
            for node, dep in node_deps.items():
                if len(dep) == 0:
                    dependencyless.add(node)

            if not dependencyless:
                # If node_dependencies not empty, though no dependencyless node was found then a CircularDependencyError
                # occured
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
                        min_deps_nodes = max(nodes_deps_len_dict.keys(), key=nodes_deps_len_dict.get)

                    for node, node_deps_set in node_deps.items():
                        for dep_node in min_deps_nodes:
                            if dep_node in node_deps_set:
                                recurrent_conns.append((dep_node, node))
                        node_deps[node] = node_deps_set - min_deps_nodes
                    continue
                # Otherwise if no dependencyless nodes exist anymore and node_deps is empty, exit dependency loop
                # regularly
                break
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

        '''
        self.node_dependencies = dict()
        # Create set of species (self.species, set), assignment of nodes to their species (self.node_species, dict) as
        # well as the assignment of nodes to the nodes they depend upon (self.node_dependencies, dict)
        for gene in self.blueprint_graph.values():
            if isinstance(gene, CoDeepNEATBlueprintNode):
                self.node_species[gene.node] = gene.species
                self.species.add(gene.species)
            elif gene.enabled:  # and isinstance(gene, CoDeepNEATBlueprintConn):
                # Only consider a connection for the processing if it is enabled
                if gene.conn_end in self.node_dependencies:
                    self.node_dependencies[gene.conn_end].add(gene.conn_start)
                else:
                    self.node_dependencies[gene.conn_end] = {gene.conn_start}
        # Remove the 'None' species assigned to Input node
        self.species.remove(None)

        # Topologically sort the graph and save into self.graph_topology as a list of sets of levels, with the first
        # set being the layer dependent on nothing and the following sets depending on the values of the preceding sets
        node_deps = self.node_dependencies.copy()
        node_deps[1] = set()  # Add Input node 1 to node dependencies as dependent on nothing
        while True:
            # find all nodes in graph having no dependencies in current iteration
            dependencyless = set()
            for node, dep in node_deps.items():
                if len(dep) == 0:
                    dependencyless.add(node)

            if not dependencyless:
                # If node_dependencies not empty, though no dependencyless node was found then a CircularDependencyError
                # occured
                if node_deps:
                    raise ValueError("Circular Dependency Error when sorting the topology of the Blueprint graph.\n"
                                     "Parent mutation: {}".format(self.parent_mutation))
                # Otherwise if no dependencyless nodes exist anymore and node_deps is empty, exit dependency loop
                # regularly
                break
            # Add dependencyless nodes of current generation to list
            self.graph_topology.append(dependencyless)

            # remove keys with empty dependencies and remove all nodes that are considered dependencyless from the
            # dependencies of other nodes in order to create next iteration
            for node in dependencyless:
                del node_deps[node]
            for node, dep in node_deps.items():
                node_deps[node] = dep - dependencyless


        for gene in genome_graph:
            if isinstance(gene, CoDeepNEATBlueprintNode):
                self.node_species[gene.node] = gene.species
                self.species.add(gene.species)
            elif gene.enabled:  # and isinstance(gene, CoDeepNEATBlueprintConn):
                # Only consider a connection for the processing if it is enabled
                if gene.conn_end in self.node_dependencies:
                    self.node_dependencies[gene.conn_end].add(gene.conn_start)
                else:
                    self.node_dependencies[gene.conn_end] = {gene.conn_start}
        # Remove the 'None' species assigned to Input node
        self.species.remove(None)

        # Topologically sort the graph and save into self.graph_topology as a list of sets of levels, with the first
        # set being the layer dependent on nothing and the following sets depending on the values of the preceding sets
        node_deps = self.node_dependencies.copy()
        node_deps[1] = set()  # Add Input node 1 to node dependencies as dependent on nothing
        while True:
            # find all nodes in graph having no dependencies in current iteration
            dependencyless = set()
            for node, dep in node_deps.items():
                if len(dep) == 0:
                    dependencyless.add(node)

            if not dependencyless:
                # If node_dependencies not empty, though no dependencyless node was found then a CircularDependencyError
                # occured
                if node_deps:
                    raise ValueError("Circular Dependency Error when sorting the topology of the Blueprint graph.\n"
                                     "Parent mutation: {}".format(self.parent_mutation))
                # Otherwise if no dependencyless nodes exist anymore and node_deps is empty, exit dependency loop
                # regularly
                break
            # Add dependencyless nodes of current generation to list
            self.graph_topology.append(dependencyless)

            # remove keys with empty dependencies and remove all nodes that are considered dependencyless from the
            # dependencies of other nodes in order to create next iteration
            for node in dependencyless:
                del node_deps[node]
            for node, dep in node_deps.items():
                node_deps[node] = dep - dependencyless
        '''

        #### ACTUAL IMPLEMENTATION #####################################################################################

        self.output_layers = [tf.keras.layers.deserialize(layer) for layer in output_layers]

    def __call__(self, inputs, *args, **kwargs) -> tf.Tensor:
        x = tf.cast(x=inputs, dtype=self.dtype)

        for layer in self.output_layers:
            x = layer(x)

        return x
