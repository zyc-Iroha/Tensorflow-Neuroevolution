import os
import json
import tempfile
import platform
import subprocess
import tensorflow as tf
from typing import Union
from tfne.encodings.base_genome import BaseGenome
from .deepneat_model import DeepNEATModel


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


class DeepNEATGenome(BaseGenome):
    """"""

    def __init__(self,
                 genome_id,
                 parent_mutation,
                 generation,
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
                 dtype):
        """"""
        # Register genome parameters
        self.genome_id = genome_id
        self.parent_mutation = parent_mutation
        self.generation = generation

        # Register genotype
        self.genome_graph = genome_graph
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

        # Create model with genotype
        self.model = DeepNEATModel(genome_graph,
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
        genome_graph_str = list()
        for gene_id, gene in self.genome_graph.items():
            if isinstance(gene, DeepNEATNode):
                genome_graph_str.append((gene.node, gene.layer))
            else:
                genome_graph_str.append((gene.conn_start, gene.conn_end))

        return "DeepNEAT Genome | ID: {:>6} | Fitness: {:>6} | Graph: {} | Optimizer: {}".format(self.genome_id,
                                                                                                 self.fitness,
                                                                                                 genome_graph_str,
                                                                                                 self.optimizer)

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

    def get_genotype(self) -> ({int: Union[DeepNEATNode, DeepNEATConn]}, [dict], dict):
        """"""
        return self.genome_graph, self.preprocessing_layers, self.optimizer

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
