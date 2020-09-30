import tensorflow as tf


class DeepNEATModel(tf.keras.Model):
    """"""

    def __init__(self,
                 genome_graph,
                 preprocessing_layers,
                 optimizer,
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
        super().__init__(*args, **kwargs)
        self.dummy_layer = tf.keras.layers.deserialize(output_layers[0])

    def __call__(self, inputs, *args, **kwargs) -> tf.Tensor:
        return self.dummy_layer(inputs)
