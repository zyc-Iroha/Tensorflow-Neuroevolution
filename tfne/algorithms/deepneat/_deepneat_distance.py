import random


class DeepNEATDistance:
    def _calculate_genome_distance(self, genome_1, genome_2) -> float:
        """"""
        # Stubby code calculating genome distance randomly, though saving that random distance for each genome pair
        if not hasattr(self, 'distance_memory'):
            self.distance_memory = dict()

        genome_id_pair = (genome_1.get_id(), genome_2.get_id())
        if genome_id_pair not in self.distance_memory:
            self.distance_memory[genome_id_pair] = random.uniform(0.01, 0.5)

        return self.distance_memory[genome_id_pair]
