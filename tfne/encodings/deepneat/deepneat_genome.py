import os
import json
import tempfile
import platform
import subprocess
import tensorflow as tf
from tfne.encodings.base_genome import BaseGenome
from .deepneat_model import DeepNEATModel


class DeepNEATGenome(BaseGenome):
    """"""

    def __init__(self,
                 genome_id,
                 parent_mutation,
                 generation,
                 genome_nodes,
                 genome_conns_enabled,
                 genome_conns_disabled,
                 preprocessing_layers,
                 optimizer,
                 input_shape,
                 input_layers,
                 output_layers,
                 recurrent_stateful,
                 recurrent_init,
                 input_scaling,
                 merge_method,
                 dtype):
        """"""
        # Register genome parameters
        self.genome_id = genome_id
        self.parent_mutation = parent_mutation
        self.generation = generation

        # Register genotype
        self.genome_nodes = genome_nodes
        self.genome_conns_enabled = genome_conns_enabled
        self.genome_conns_disabled = genome_conns_disabled
        self.preprocessing_layers = preprocessing_layers
        self.optimizer = optimizer

        # Register immutable phenotype parameters
        self.input_shape = input_shape
        self.input_layers = input_layers
        self.output_layers = output_layers
        self.recurrent_stateful = recurrent_stateful
        self.recurrent_init = recurrent_init
        self.input_scaling = input_scaling
        self.merge_method = merge_method
        self.dtype = dtype

        # Initialize internal variables
        self.fitness = None

        # Preprocess genotype information as required by DeepNEATModel and useful for evolutionary operations on this
        # genome
        self.graph_topology = None
        self.recurrent_conns = None
        self._preprocess_genotype()

        # Create model with genotype
        self.model = DeepNEATModel(self.graph_topology,
                                   self.recurrent_conns,
                                   self.genome_nodes,
                                   preprocessing_layers,
                                   input_shape,
                                   input_layers,
                                   output_layers,
                                   recurrent_stateful,
                                   recurrent_init,
                                   input_scaling,
                                   merge_method,
                                   dtype)

    def __call__(self, inputs) -> tf.Tensor:
        """"""
        return self.model(inputs)

    def __str__(self) -> str:
        """"""
        return "DeepNEAT Genome | ID: {:>6} | Fitness: {:>6} | Nodes: {} | Conns: {} | Optimizer: {}".format(
            self.genome_id,
            self.fitness,
            self.genome_nodes.values(),
            self.genome_conns_enabled.values(),
            self.optimizer)

    def _preprocess_genotype(self):
        """"""
        pass

    def reset_states(self) -> ():
        """"""
        pass

    def visualize(self, show=True, save_dir_path=None, **kwargs) -> str:
        """"""
        raise NotImplementedError("TODO")

    def serialize(self) -> dict:
        """"""
        raise NotImplementedError("TODO")

    def save_genotype(self, save_dir_path) -> str:
        """"""
        raise NotImplementedError("TODO")

    def save_model(self, file_path, **kwargs):
        """"""
        self.model.save(filepath=file_path, **kwargs)

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_genotype(self) -> ({int: (int, str)}, {int: (int, int)}, {int: (int, int)}, [dict], dict):
        """"""
        return self.genome_nodes.copy(), \
               self.genome_conns_enabled.copy(), \
               self.genome_conns_disabled.copy(), \
               self.preprocessing_layers.copy(), \
               self.optimizer.copy()

    def get_graph_topology(self) -> [{int}]:
        """"""
        return self.graph_topology

    def get_recurrent_conns(self) -> {(int, int)}:
        """"""
        return self.recurrent_conns

    def get_model(self) -> tf.keras.Model:
        """"""
        return self.model

    def get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """"""
        return tf.keras.optimizers.deserialize(self.optimizer)

    def get_id(self) -> int:
        return self.genome_id

    def get_fitness(self) -> float:
        return self.fitness
