import os
import sys
import json
import random
import statistics
import numpy as np
import tensorflow as tf
import tfne
from absl import logging
from tfne.algorithms.base_algorithm import BaseNeuroevolutionAlgorithm
from tfne.encodings.deepneat import DeepNEATGenome
from ._deepneat_config_processing import DeepNEATConfigProcessing
from ._deepneat_initialization import DeepNEATInitialization


class DeepNEAT(BaseNeuroevolutionAlgorithm,
               DeepNEATConfigProcessing,
               DeepNEATInitialization):
    """"""

    def __init__(self, config, input_shape, output_shape, initial_state=None):
        """"""
        # Register and process the supplied configuration
        self.config = config
        self._process_config()
        self._process_available_layers()
        self._sanity_check_config()

        # Register variables of environment shapes to which the created genomes have to adhere to
        self.input_shape = input_shape
        self.output_shape = output_shape

        # Initialize a numpy random number generator
        self.rng = np.random.default_rng()

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
        # COMMENT
        if self.pop.generation_counter is not None:
            raise NotImplementedError("TODO")

        # No pre-evolved population supplied. Initialize it from scratch
        print("Initializing a new population of {} genomes...".format(self.pop_size))

        # Set internal variables of the population to the initialization of a new population
        self.pop.generation_counter = 0
        self.pop.best_fitness = 0
        self.pop.best_consistent_fitness = 0

        # COMMENT
        self.pop.species_counter = 1
        self.pop.species[self.pop.species_counter] = list()

        # COMMENT
        for _ in range(self.pop_size):

            # !FIXME not considering input layers and preprocessing layers but instead only using input shape dim
            input_dim = len(self.input_shape) + 1
            initial_layer = self.rng.choice(self.available_layers[input_dim], p=self.available_layers_p[input_dim])

            genome_id, genome = self._create_initial_genome(initial_layer=initial_layer)

            self.pop.genomes[genome_id] = genome
            self.pop.species[self.pop.species_counter].append(genome)

            if self.spec_type != 'basic' and self.pop.species_counter not in self.pop.species_repr:
                self.pop.species_repr[self.pop.species_counter] = genome_id

    def evaluate_population(self, environment) -> (int, float):
        """"""
        print("EXIT")
        exit()

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
