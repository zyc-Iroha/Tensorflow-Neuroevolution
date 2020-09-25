import os
import json
import tempfile
import platform
import subprocess
from typing import Union

import tensorflow as tf

from .deepneat_model import DeepNEATModel
from tfne.encodings.base_genome import BaseGenome


class DeepNEATNode:
    """"""

    def __init__(self, gene_id, node, layer):
        self.gene_id = gene_id
        self.node = node
        self.layer = layer


class DeepNEATConn:
    """"""

    def __init__(self, gene_id, conn_start, conn_end, enabled=True):
        self.gene_id = gene_id
        self.conn_start = conn_start
        self.conn_end = conn_end
        self.enabled = enabled

    def set_enabled(self, enabled):
        """"""
        self.enabled = enabled


class DeepNEATGenome(BaseGenome,
                     DeepNEATModel):
    """"""

    def __init__(self,
                 genome_id,
                 parent_mutation,
                 generation,
                 input_shape,
                 genome_graph,
                 preprocessing_layers,
                 output_layers,
                 optimizer,
                 dtype):
        """"""
        # Register parameters
        self.genome_id = genome_id
        self.parent_mutation = parent_mutation
        self.generation = generation
        self.dtype = dtype

        # Register genotype
        self.genome_graph = genome_graph
        self.preprocessing_layers = preprocessing_layers
        self.output_layers = output_layers
        self.optimizer = optimizer

        # Initialize internal variables
        self.fitness = None

        # Create model with genotype
        self.model = None
        self._create_model(input_shape)

    def __call__(self, inputs) -> tf.Tensor:
        """"""
        return self.model(inputs)

    def __str__(self) -> str:
        """"""
        raise NotImplementedError("TODO")

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

    def get_genotype(self) -> ({int: Union[DeepNEATNode, DeepNEATConn]}, [dict], [dict], dict):
        """"""
        return self.genome_graph, self.preprocessing_layers, self.output_layers, self.optimizer

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
