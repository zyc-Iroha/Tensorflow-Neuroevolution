import sys
import tensorflow as tf


class DeepNEATModel(tf.keras.Model):
    """"""

    def __init__(self,
                 graph_topology,
                 recurrent_conns,
                 genome_nodes,
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

        # INSERT RECURRENCY GRAPH DETECTION

        #### ACTUAL IMPLEMENTATION #####################################################################################

        self.output_layers = [tf.keras.layers.deserialize(layer) for layer in output_layers]

    def __call__(self, inputs, *args, **kwargs) -> tf.Tensor:
        x = tf.cast(x=inputs, dtype=self.dtype)

        for layer in self.output_layers:
            x = layer(x)

        return x
