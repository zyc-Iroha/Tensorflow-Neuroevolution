import os
import sys
import json
import random
import statistics
import tensorflow as tf
import tfne
from absl import logging
from tfne.algorithms.base_algorithm import BaseNeuroevolutionAlgorithm
from tfne.encodings.neat import NEATGenome
from ._neat_config_processing import NEATConfigProcessing


class NEAT(BaseNeuroevolutionAlgorithm,
           NEATConfigProcessing):
    """"""

    def __init__(self, config, initial_state=None):
        """"""
        pass

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

    def get_best_genome(self) -> NEATGenome:
        """"""
        pass

    def get_eval_instance_count(self) -> int:
        """"""
        pass
