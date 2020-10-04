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
