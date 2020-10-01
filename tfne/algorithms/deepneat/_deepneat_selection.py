import math
from typing import Union


class DeepNEATSelection:
    def _select_genomes_basic(self) -> ():
        """"""
        pass

    def _select_genomes_fixed(self) -> ():
        """"""
        pass

    def _select_genomes_dynamic(self) -> ({Union[int, str]: int}, {int: [int]}):
        """"""
        #### Species Extinction ####
        extinct_fitness = 0
        spec_extinct = set()
        species_ordered = sorted(self.pop.species.keys(),
                                 key=lambda x: self.pop.species_fitness_history[x][self.pop.generation_counter])
        for spec_id in species_ordered:
            if len(self.pop.species_fitness_history[spec_id]) < self.spec_max_stagnation + 1:
                continue
            if len(self.pop.species) <= self.spec_species_elitism:
                continue

            # Consider species for extinction and determine if it has been stagnating by checking if the distant fitness
            # is higher than all recent fitnesses
            distant_generation = self.pop.generation_counter - self.spec_max_stagnation
            distant_fitness = self.pop.species_fitness_history[spec_id][distant_generation]
            recent_fitness = list()
            for i in range(self.spec_max_stagnation):
                recent_fitness.append(self.pop.species_fitness_history[spec_id][self.pop.generation_counter - i])
            if distant_fitness >= max(recent_fitness):
                # Species is stagnating. Flag species as extinct, keep track of its fitness and then remove it from the
                # population
                spec_extinct.add(spec_id)
                extinct_fitness += self.pop.species_fitness_history[spec_id][self.pop.generation_counter]
                for genome_id in self.pop.species[spec_id]:
                    del self.pop.genomes[genome_id]
                del self.pop.species[spec_id]
                del self.pop.species_repr[spec_id]
                del self.pop.species_fitness_history[spec_id]

        #### Rebase Species Representative ####
        if self.spec_rebase_repr:
            all_spec_repr_ids = set(self.pop.species_repr.values())
            for spec_id, spec_repr_id in self.pop.species_repr.items():
                other_spec_repr_ids = all_spec_repr_ids - {spec_repr_id}

                spec_genome_ids_sorted = sorted(self.pop.species[spec_id],
                                                key=lambda x: self.pop.genomes[x].get_fitness(),
                                                reverse=True)
                for genome_id in spec_genome_ids_sorted:
                    if genome_id == spec_repr_id:
                        # Best species genome already representative. Abort search.
                        break
                    genome = self.pop.genomes[genome_id]
                    distance_to_other_spec_repr = [
                        self._calculate_genome_distance(genome, self.pop.genomes[other_genome_id])
                        for other_genome_id in other_spec_repr_ids]
                    if all(distance >= self.spec_distance for distance in distance_to_other_spec_repr):
                        # New best species representative found. Set as representative and abort search
                        self.pop.species_repr[spec_id] = genome_id
                        break

        #### Generational Parent Determination ####
        spec_parents = dict()
        for spec_id, spec_genome_ids in self.pop.species.items():
            spec_genome_ids_sorted = sorted(spec_genome_ids, key=lambda x: self.pop.genomes[x].get_fitness())

            # Determine the species elite as the top x members and the species representative
            elites = set(spec_genome_ids_sorted[-self.spec_genome_elitism:])
            elites.add(self.pop.species_repr[spec_id])

            # Determine the species parents as those clearing the reproduction threshold, plus the species elites
            reprod_threshold_index = math.ceil(len(spec_genome_ids) * self.spec_reprod_thres)
            parents = set(spec_genome_ids_sorted[reprod_threshold_index:])
            parents = parents.union(elites)

            # Remove non elite genome from the species list, as they are not part of the species anymore. Remove non
            # parental genomes from the genome container as there is no use for thsoe genomes anymore.
            genome_ids_non_elite = set(spec_genome_ids) - elites
            genome_ids_non_parental = set(spec_genome_ids) - parents
            for genome_id in genome_ids_non_elite:
                self.pop.species[spec_id].remove(genome_id)
            for genome_id in genome_ids_non_parental:
                del self.pop.genomes[genome_id]

            spec_parents[spec_id] = tuple(parents)

        #### Offspring Size Calculation ####
        total_fitness = 0
        for fitness_history in self.pop.species_fitness_history.values():
            total_fitness += fitness_history[self.pop.generation_counter]
        for spec_id in spec_extinct:
            species_ordered.remove(spec_id)

        # Determine the amount of offspring to be reinitialized as the fitness share of the total fitness by the extinct
        # species
        spec_offspring = dict()
        available_pop = self.pop_size
        if self.spec_reinit_extinct and extinct_fitness > 0:
            extinct_fitness_share = extinct_fitness / (total_fitness + extinct_fitness)
            reinit_offspring = int(extinct_fitness_share * available_pop)
            spec_offspring['reinit'] = reinit_offspring
            available_pop -= reinit_offspring

        for spec_id in species_ordered:
            spec_fitness = self.pop.species_fitness_history[spec_id][self.pop.generation_counter]
            spec_fitness_share = spec_fitness / total_fitness
            spec_intended_size = int(round(spec_fitness_share * available_pop))

            if len(self.pop.species[spec_id]) + self.spec_min_offspring > spec_intended_size:
                spec_offspring[spec_id] = self.spec_min_offspring
                available_pop -= len(self.pop.species[spec_id]) + self.spec_min_offspring
            else:
                spec_offspring[spec_id] = spec_intended_size - len(self.pop.species[spec_id])
                available_pop -= spec_intended_size
            total_fitness -= spec_fitness

        #### Return ####
        # spec_offspring {Union[int, str]: int} associating species id with amount of offspring
        # spec_parents {int: [int]} associating species id with list of potential parent ids for species
        return spec_offspring, spec_parents
