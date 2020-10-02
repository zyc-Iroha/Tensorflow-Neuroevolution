import random
from tfne.encodings.deepneat import DeepNEATGenome


class DeepNEATEvolution:
    def _evolve_genomes(self, spec_offspring, spec_parents) -> [int]:
        """"""
        new_genome_ids = list()

        for spec_id, species_offspring in spec_offspring.items():
            if spec_id == 'reinit':
                for _ in range(species_offspring):
                    new_genome_id, new_genome = self._create_initial_genome()
                    self.pop.genomes[new_genome_id] = new_genome
                    new_genome_ids.append(new_genome_id)
                continue

            for _ in range(species_offspring):
                chosen_mutation = self.rng.choice(self.mutation_methods, p=self.mutation_methods_p)

                if chosen_mutation == 'add_conn':
                    parent_genome = self.pop.genomes[random.choice(spec_parents[spec_id])]
                    new_genome_id, new_genome = self._mutation_add_conn(parent_genome)

                elif chosen_mutation == 'add_node':
                    parent_genome = self.pop.genomes[random.choice(spec_parents[spec_id])]
                    new_genome_id, new_genome = self._mutation_add_node(parent_genome)

                elif chosen_mutation == 'rem_conn':
                    parent_genome = self.pop.genomes[random.choice(spec_parents[spec_id])]
                    new_genome_id, new_genome = self._mutation_rem_conn(parent_genome)

                elif chosen_mutation == 'rem_node':
                    parent_genome = self.pop.genomes[random.choice(spec_parents[spec_id])]
                    new_genome_id, new_genome = self._mutation_rem_node(parent_genome)

                elif chosen_mutation == 'node_layer':
                    parent_genome = self.pop.genomes[random.choice(spec_parents[spec_id])]
                    new_genome_id, new_genome = self._mutation_node_layer(parent_genome)

                elif chosen_mutation == 'hyperparam':
                    parent_genome = self.pop.genomes[random.choice(spec_parents[spec_id])]
                    new_genome_id, new_genome = self._mutation_hyperparam(parent_genome)

                elif chosen_mutation == 'crossover':
                    if len(spec_parents[spec_id]) >= 2:
                        parent_genome_1_id, parent_genome_2_id = random.sample(spec_parents[spec_id], k=2)
                        parent_genome_1 = self.pop.genomes[parent_genome_1_id]
                        parent_genome_2 = self.pop.genomes[parent_genome_2_id]
                        new_genome_id, new_genome = self._crossover(parent_genome_1, parent_genome_2)

                    else:
                        parent_genome = self.pop.genomes[random.choice(spec_parents[spec_id])]
                        new_genome_id, new_genome = self._mutation_node_layer(parent_genome)

                else:
                    raise RuntimeError("COMMENT")

                self.pop.genomes[new_genome_id] = new_genome
                new_genome_ids.append(new_genome_id)

        #### Return ####
        return new_genome_ids

    def _mutation_add_conn(self, parent_genome) -> (int, DeepNEATGenome):
        """"""
        parent_mutation = "stubby"
        parent_genome_graph, parent_preprocessing_layers, parent_optimizer = parent_genome.get_genotype()
        return self.enc.create_genome(parent_mutation=parent_mutation,
                                      generation=self.pop.generation_counter,
                                      genome_graph=parent_genome_graph,
                                      preprocessing_layers=parent_preprocessing_layers,
                                      optimizer=parent_optimizer)

    def _mutation_add_node(self, parent_genome) -> (int, DeepNEATGenome):
        """"""
        parent_mutation = "stubby"
        parent_genome_graph, parent_preprocessing_layers, parent_optimizer = parent_genome.get_genotype()
        return self.enc.create_genome(parent_mutation=parent_mutation,
                                      generation=self.pop.generation_counter,
                                      genome_graph=parent_genome_graph,
                                      preprocessing_layers=parent_preprocessing_layers,
                                      optimizer=parent_optimizer)

    def _mutation_rem_conn(self, parent_genome) -> (int, DeepNEATGenome):
        """"""
        parent_mutation = "stubby"
        parent_genome_graph, parent_preprocessing_layers, parent_optimizer = parent_genome.get_genotype()
        return self.enc.create_genome(parent_mutation=parent_mutation,
                                      generation=self.pop.generation_counter,
                                      genome_graph=parent_genome_graph,
                                      preprocessing_layers=parent_preprocessing_layers,
                                      optimizer=parent_optimizer)

    def _mutation_rem_node(self, parent_genome) -> (int, DeepNEATGenome):
        """"""
        parent_mutation = "stubby"
        parent_genome_graph, parent_preprocessing_layers, parent_optimizer = parent_genome.get_genotype()
        return self.enc.create_genome(parent_mutation=parent_mutation,
                                      generation=self.pop.generation_counter,
                                      genome_graph=parent_genome_graph,
                                      preprocessing_layers=parent_preprocessing_layers,
                                      optimizer=parent_optimizer)

    def _mutation_node_layer(self, parent_genome) -> (int, DeepNEATGenome):
        """"""
        parent_mutation = "stubby"
        parent_genome_graph, parent_preprocessing_layers, parent_optimizer = parent_genome.get_genotype()
        return self.enc.create_genome(parent_mutation=parent_mutation,
                                      generation=self.pop.generation_counter,
                                      genome_graph=parent_genome_graph,
                                      preprocessing_layers=parent_preprocessing_layers,
                                      optimizer=parent_optimizer)

    def _mutation_hyperparam(self, parent_genome) -> (int, DeepNEATGenome):
        """"""
        parent_mutation = "stubby"
        parent_genome_graph, parent_preprocessing_layers, parent_optimizer = parent_genome.get_genotype()
        return self.enc.create_genome(parent_mutation=parent_mutation,
                                      generation=self.pop.generation_counter,
                                      genome_graph=parent_genome_graph,
                                      preprocessing_layers=parent_preprocessing_layers,
                                      optimizer=parent_optimizer)

    def _crossover(self, parent_genome_1, parent_genome_2) -> (int, DeepNEATGenome):
        """"""
        parent_mutation = "stubby"
        parent_genome_graph, parent_preprocessing_layers, parent_optimizer = parent_genome_1.get_genotype()
        return self.enc.create_genome(parent_mutation=parent_mutation,
                                      generation=self.pop.generation_counter,
                                      genome_graph=parent_genome_graph,
                                      preprocessing_layers=parent_preprocessing_layers,
                                      optimizer=parent_optimizer)
