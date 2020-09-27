import os
import sys
import json
import random
import statistics
import tensorflow as tf
import tfne
from absl import logging
from tfne.algorithms.base_algorithm import BaseNeuroevolutionAlgorithm
from tfne.encodings.subgraphdeepneat import SubGraphDeepNEATGenome
from ._subgraphdeepneat_config_processing import SubGraphDeepNEATConfigProcessing


class SubGraphDeepNEAT(BaseNeuroevolutionAlgorithm,
                       SubGraphDeepNEATConfigProcessing):
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

    def get_best_genome(self) -> SubGraphDeepNEATGenome:
        """"""
        pass

    def get_best_consistent_genome(self) -> SubGraphDeepNEATGenome:
        """"""
        pass

    def get_eval_instance_count(self) -> int:
        """"""
        return 1
