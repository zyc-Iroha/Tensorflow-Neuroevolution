import statistics
from tfne.populations.base_population import BasePopulation


class DeepNEATPopulation(BasePopulation):
    """"""

    def __init__(self, initial_state=None):
        """"""
        # Declare general internal variables of the CoDeepNEAT population
        self.generation_counter = None
        self.best_genome = None
        self.best_fitness = None
        self.best_consistent_genome = None
        self.best_consistent_fitness = None

        # Declare and initialize internal variables concerning the genome population of the CoDeepNEAT algorithm
        self.genomes = dict()
        self.species = dict()
        self.species_repr = dict()
        self.species_fitness_history = dict()
        self.species_counter = 0

    def summarize_population(self):
        """"""
        pass

    def serialize(self) -> dict:
        """"""
        pass
