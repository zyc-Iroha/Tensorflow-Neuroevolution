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

        # INSERT RECURRENCY GRAPH DETECTION

        #### ACTUAL IMPLEMENTATION #####################################################################################

        self.output_layers = [tf.keras.layers.deserialize(layer) for layer in output_layers]

    def __call__(self, inputs, *args, **kwargs) -> tf.Tensor:
        x = tf.cast(x=inputs, dtype=self.dtype)

        for layer in self.output_layers:
            x = layer(x)

        return x
