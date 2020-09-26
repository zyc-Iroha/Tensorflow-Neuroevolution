from tfne.encodings.base_encoding import BaseEncoding
from .deepneat_genome import DeepNEATGenome, DeepNEATNode, DeepNEATConn


class DeepNEATEncoding(BaseEncoding):
    """"""

    def __init__(self, initial_state=None):
        """"""
        # Initialize internal counter variables
        self.genome_id_counter = 0
        self.gene_id_counter = 0

        # Initialize container that maps a gene parameters to a gene-id, presuming that the encoding encountered the
        # gene parameters before and assigned those parameters a specific id. If the gene parameters are novel, assign
        # the gene a new gene-id, ensuring that each gene parameter has a single unique ID.
        self.gene_to_gene_id = dict()

        # Initialize a counter for nodes and a connection split history, assigning the tuple of connection start and end
        # a previously assigned node or new node if not yet present in history.
        self.node_counter = 2
        self.conn_split_history = dict()

        # If an initial state is supplied, then the encoding was deserialized. Recreate this initial state.
        if initial_state is not None:
            raise NotImplementedError("TODO")

    def create_node_gene(self, node, layer) -> (int, DeepNEATNode):
        """"""
        gene_key = (node,)
        if gene_key not in self.gene_to_gene_id:
            self.gene_id_counter += 1
            self.gene_to_gene_id[gene_key] = self.gene_id_counter

        gene_id = self.gene_to_gene_id[gene_key]
        return gene_id, DeepNEATNode(gene_id, node, layer)

    def create_conn_gene(self, conn_start, conn_end) -> (int, DeepNEATConn):
        """"""
        gene_key = (conn_start, conn_end)
        if gene_key not in self.gene_to_gene_id:
            self.gene_id_counter += 1
            self.gene_to_gene_id[gene_key] = self.gene_id_counter

        gene_id = self.gene_to_gene_id[gene_key]
        return gene_id, DeepNEATConn(gene_id, conn_start, conn_end)

    def get_node_for_split(self, conn_start, conn_end) -> int:
        """"""
        conn_key = (conn_start, conn_end)
        if conn_key not in self.conn_split_history:
            self.node_counter += 1
            self.conn_split_history[conn_key] = self.node_counter

        return self.conn_split_history[conn_key]

    def create_genome(self,
                      parent_mutation,
                      generation,
                      input_shape,
                      genome_graph,
                      preprocessing_layers,
                      output_layers,
                      optimizer,
                      dtype) -> (int, DeepNEATGenome):
        """"""
        self.genome_id_counter += 1
        # Genome genotype: (input_shape, genome_graph, preprocessing_layers, output_layers, optimizer)
        return self.genome_id_counter, DeepNEATGenome(genome_id=self.genome_id_counter,
                                                      parent_mutation=parent_mutation,
                                                      generation=generation,
                                                      input_shape=input_shape,
                                                      genome_graph=genome_graph,
                                                      preprocessing_layers=preprocessing_layers,
                                                      output_layers=output_layers,
                                                      optimizer=optimizer,
                                                      dtype=dtype)

    def serialize(self) -> dict:
        """"""
        raise NotImplementedError("TODO")
