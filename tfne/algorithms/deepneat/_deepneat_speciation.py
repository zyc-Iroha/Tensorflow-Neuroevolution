import warnings
import statistics


class DeepNEATSpeciation:
    def _speciate_genomes_dynamic(self, new_genome_ids, spec_parents):
        """"""
        #### Removal of Parental But Not Elite Genomes ####
        for spec_id, spec_parent_ids in spec_parents.items():
            spec_elites = self.pop.species[spec_id]
            for genome_id in spec_parent_ids:
                if genome_id not in spec_elites:
                    del self.pop.genomes[genome_id]

        #### Species Assignment ####
        min_spec_size = self.spec_genome_elitism + self.spec_min_offspring + 1
        for genome_id in new_genome_ids:
            spec_distances = dict()
            for spec_id, spec_repr_id in self.pop.species_repr.items():
                spec_repr_genome = self.pop.genomes[spec_repr_id]
                spec_distances[spec_id] = self._calculate_genome_distance(spec_repr_genome, self.pop.genomes[genome_id])

            min_distance_spec = min(spec_distances, key=spec_distances.get)
            if spec_distances[min_distance_spec] <= self.spec_distance:
                self.pop.species[min_distance_spec].append(genome_id)
            elif spec_distances[min_distance_spec] > self.spec_distance \
                    and min_spec_size * len(self.pop.species) >= self.pop_size:
                warnings.warn(f"Warning: New Genome (#{genome_id}) has sufficient distance to other species "
                              f"representatives in order to form a new species, but it has been assigned to species "
                              f"{min_distance_spec} as the population size does not allow for more species.",
                              UserWarning)
                self.pop.species[min_distance_spec].append(genome_id)
            else:
                # Create a new species with the new genome as the representative
                self.pop.species_counter += 1
                self.pop.species[self.pop.species_counter] = [genome_id]
                self.pop.species_repr[self.pop.species_counter] = genome_id

        #### Dynamic Adjustment of Species Distance ####
        if len(self.pop.species) < self.spec_species_count:
            self.spec_distance = self.spec_distance * (1 - self.spec_distance_dec)
        elif len(self.pop.species) > self.spec_species_count:
            optimal_spec_distance_per_species = list()
            for spec_id, spec_repr_id in self.pop.species_repr.items():
                spec_repr_genome = self.pop.genomes[spec_repr_id]
                # Determine distance of species repr to all other species repr
                other_spec_repr_ids = [genome_id for genome_id in self.pop.species_repr.values()
                                       if genome_id != spec_repr_id]
                distances_to_other_specs = [self._calculate_genome_distance(spec_repr_genome,
                                                                            self.pop.genomes[other_spec_repr_id])
                                            for other_spec_repr_id in other_spec_repr_ids]
                sorted_distances_to_other_specs = sorted(distances_to_other_specs)

                optimal_spec_distance = sorted_distances_to_other_specs[self.spec_species_count - 1]
                optimal_spec_distance_per_species.append(optimal_spec_distance)

            # Average out all optimal distances for each species repr to get the new distance
            self.spec_distance = statistics.mean(optimal_spec_distance_per_species)
