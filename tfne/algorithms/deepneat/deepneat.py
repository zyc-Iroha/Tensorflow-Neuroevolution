import sys
import statistics
import numpy as np
import tensorflow as tf
import tfne
from copy import deepcopy
from absl import logging
from tfne.algorithms.base_algorithm import BaseNeuroevolutionAlgorithm
from tfne.encodings.deepneat import DeepNEATGenome
from ._deepneat_config_processing import DeepNEATConfigProcessing
from ._deepneat_initialization import DeepNEATInitialization
from ._deepneat_selection import DeepNEATSelection
from ._deepneat_evolution import DeepNEATEvolution
from ._deepneat_speciation import DeepNEATSpeciation
from ._deepneat_distance import DeepNEATDistance


class DeepNEAT(BaseNeuroevolutionAlgorithm,
               DeepNEATConfigProcessing,
               DeepNEATInitialization,
               DeepNEATSelection,
               DeepNEATEvolution,
               DeepNEATSpeciation,
               DeepNEATDistance):
    """"""

    def __init__(self, config, input_shape, output_shape, initial_state=None):
        """"""
        # Register and process the supplied configuration
        self.config = config
        self._read_config()
        self._process_config()
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

            genome_id, genome = self._create_initial_genome()

            self.pop.genomes[genome_id] = genome
            self.pop.species[self.pop.species_counter].append(genome_id)

            if self.spec_type != 'basic' and self.pop.species_counter not in self.pop.species_repr:
                self.pop.species_repr[self.pop.species_counter] = genome_id

    def evaluate_population(self, environment) -> (int, float):
        """"""
        # Initialize population evaluation progress bar. Print notice of evaluation start
        genome_eval_counter = 0
        genome_eval_counter_div = round(self.pop_size / 80.0, 4)
        print("\nEvaluating {} genomes in generation {}...".format(self.pop_size, self.pop.generation_counter))
        print_str = "\r[{:80}] {}/{} Genomes".format("", genome_eval_counter, self.pop_size)
        sys.stdout.write(print_str)
        sys.stdout.flush()

        for genome_id, genome in self.pop.genomes.items():

            genome_fitness = environment.eval_genome_fitness(genome)

            if genome_fitness > self.pop.best_consistent_fitness:
                genome_fitness_list = [genome_fitness]
                for _ in range(self.consistency_evals - 1):
                    # tf.keras.backend.clear_session()
                    genome.reset_states()
                    genome_fitness_list.append(environment.eval_genome_fitness(genome))

                genome_avg_fitness = round(statistics.mean(genome_fitness_list), 4)
                genome.set_fitness(genome_avg_fitness)
                if genome_avg_fitness > self.pop.best_consistent_fitness:
                    self.pop.best_consistent_genome = deepcopy(genome)
                    self.pop.best_consistent_fitness = genome_avg_fitness

                genome_max_fitness = max(genome_fitness_list)
                if genome_max_fitness > self.pop.best_fitness:
                    self.pop.best_genome = deepcopy(genome)
                    self.pop.best_fitness = genome_max_fitness
            else:
                genome.set_fitness(genome_fitness)

            # Print population evaluation progress bar
            genome_eval_counter += 1
            progress_mult = int(round(genome_eval_counter / genome_eval_counter_div, 4))
            print_str = "\r[{:80}] {}/{} Genomes | Genome ID {} achieved fitness of {}".format(
                "=" * progress_mult,
                genome_eval_counter,
                self.pop_size,
                genome_id,
                genome_fitness)
            sys.stdout.write(print_str)
            sys.stdout.flush()

            # Add newline after status update when debugging
            if logging.level_debug():
                print("")

            # Reset models, counters, layers, etc including in the GPU to avoid memory clutter from old models as
            # most likely only limited gpu memory is available
            tf.keras.backend.clear_session()

        for spec_id, spec_genome_ids in self.pop.species.items():
            spec_fitness_list = [self.pop.genomes[genome_id].get_fitness() for genome_id in spec_genome_ids]
            spec_fitness = round(self.spec_fitness_func(spec_fitness_list), 4)
            if spec_id in self.pop.species_fitness_history:
                self.pop.species_fitness_history[spec_id][self.pop.generation_counter] = spec_fitness
            else:
                self.pop.species_fitness_history[spec_id] = {self.pop.generation_counter: spec_fitness}

        return self.pop.generation_counter, self.pop.best_fitness

    def summarize_population(self):
        """"""
        self.pop.summarize_population()

    def evolve_population(self) -> bool:
        """"""
        #### Select Genomes ####
        if self.spec_type == 'basic':
            spec_offspring, spec_parents = self._select_genomes_basic()
        elif self.spec_type == 'fixed':
            spec_offspring, spec_parents = self._select_genomes_fixed()
        elif self.spec_type == 'dynamic':
            spec_offspring, spec_parents = self._select_genomes_dynamic()
        else:
            raise RuntimeError(f"Speciation type '{self.spec_type}' not yet implemented")

        if len(self.pop.species) == 0:
            return True

        #### Evolve Genomes ####
        new_genome_ids = self._evolve_genomes(spec_offspring, spec_parents)

        #### Speciate Genomes ####
        if self.spec_type == 'basic':
            pass
        elif self.spec_type == 'fixed':
            pass
        elif self.spec_type == 'dynamic':
            self._speciate_genomes_dynamic(new_genome_ids, spec_parents)

        #### Return ####
        # Adjust generation counter and return False, signalling that the population has not gone extinct
        self.pop.generation_counter += 1
        return False

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
