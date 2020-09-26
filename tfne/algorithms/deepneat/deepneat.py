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


class DeepNEAT(BaseNeuroevolutionAlgorithm):
    """"""

    def __init__(self, config, initial_state=None):
        """"""
        # Register and process the supplied configuration
        self.config = config
        self._process_config()
        self._sanity_check_config()

        # Declare variables of environment shapes to which the created genomes have to adhere to
        self.input_shape = None
        self.output_shape = None

        # If an initial state of the evolution was supplied, load and recreate this state for the algorithm as well as
        # its dependencies
        if initial_state is not None:
            # Load the backed up state for the algorithm from file
            raise NotImplementedError("TODO")
        else:
            # Initialize and register a blank associated DeepNEAT encoding and population
            self.enc = tfne.encodings.DeepNEATEncoding()
            self.pop = tfne.populations.DeepNEATPopulation()

    def initialize_population(self, environment):
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
        pass

    def get_best_consistent_genome(self) -> DeepNEATGenome:
        """"""
        pass

    def get_eval_instance_count(self) -> int:
        """"""
        return 1
