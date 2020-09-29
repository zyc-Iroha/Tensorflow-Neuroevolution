import os
import json
import tempfile
import platform
import subprocess
import tensorflow as tf
from typing import Union
from tfne.encodings.base_genome import BaseGenome
from .subgraphdeepneat_model import SubGraphDeepNEATModel


class SubGraphDeepNEATGenome(BaseGenome):
    """"""

    def __init__(self,
                 *args,
                 **kwargs):
        """"""
        raise NotImplementedError("TODO")

    def __call__(self, inputs) -> tf.Tensor:
        """"""
        raise NotImplementedError("TODO")

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
        raise NotImplementedError("TODO")

    def set_fitness(self, fitness):
        raise NotImplementedError("TODO")

    def get_genotype(self) -> ():
        """"""
        raise NotImplementedError("TODO")

    def get_model(self) -> tf.keras.Model:
        """"""
        raise NotImplementedError("TODO")

    def get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """"""
        raise NotImplementedError("TODO")

    def get_id(self) -> int:
        return self.genome_id

    def get_fitness(self) -> float:
        return self.fitness
