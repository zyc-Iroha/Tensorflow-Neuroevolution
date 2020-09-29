import os
import sys
import json
import random
import statistics
import tensorflow as tf
import tfne
from absl import logging
from tfne.algorithms.base_algorithm import BaseNeuroevolutionAlgorithm
from tfne.encodings.deepneat import DeepNEATGenome
from ._deepneat_config_processing import DeepNEATConfigProcessing


class DeepNEAT(BaseNeuroevolutionAlgorithm,
               DeepNEATConfigProcessing):
    """"""

    def __init__(self, config, input_shape, output_shape, initial_state=None):
        """"""
        # Register and process the supplied configuration
        self.config = config
        self._process_config()
        self._sanity_check_config()

        # Register variables of environment shapes to which the created genomes have to adhere to
        self.input_shape = input_shape
        self.output_shape = output_shape

        # If an initial state of the evolution was supplied, load and recreate this state for the algorithm as well as
        # its dependencies
        if initial_state is not None:
            # Load the backed up state for the algorithm from file
            raise NotImplementedError("TODO")
        else:
            # Initialize a new associated DeepNEAT encoding and population
            self.enc = tfne.encodings.DeepNEATEncoding(self.input_shape,
                                                       self.input_layers,
                                                       self.output_layers,
                                                       self.recurrent_stateful,
                                                       self.recurrent_init,
                                                       self.input_scaling,
                                                       self.merge_method,
                                                       self.dtype)
            self.pop = tfne.populations.DeepNEATPopulation()

    def initialize_population(self):
        """"""
        pass

    def evaluate_population(self, environment) -> (int, float):
        """"""
        pass

    def summarize_population(self):
        """"""
        pass

    def evolve_population(self) -> bool:
        """"""
        pass

    def save_state(self, save_dir_path):
        """"""
        pass

    def get_best_genome(self) -> DeepNEATGenome:
        """"""
        return self.pop.best_genome

    def get_best_consistent_genome(self) -> DeepNEATGenome:
        """"""
        return self.pop.best_consistent_genome

    def get_eval_instance_count(self) -> int:
        """"""
        return 1
