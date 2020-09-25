import os
import json
import tempfile
import platform
import subprocess

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
        self.optimizer = optimizer
        self.dtype = dtype

        # Initialize internal variables
        self.fitness = None

        # Create model with genotype
        self.model = None
        self._create_model(input_shape, genome_graph, preprocessing_layers, output_layers)
