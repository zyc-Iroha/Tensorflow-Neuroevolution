import math
import random
from tfne.encodings.deepneat import DeepNEATGenome
from tfne.helper_functions import round_with_step


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
                    parent_genome_id = random.choice(spec_parents[spec_id])
                    new_genome_id, new_genome = self._mutation_add_conn(parent_genome_id)

                elif chosen_mutation == 'add_node':
                    parent_genome_id = random.choice(spec_parents[spec_id])
                    new_genome_id, new_genome = self._mutation_add_node(parent_genome_id)

                elif chosen_mutation == 'rem_conn':
                    parent_genome_id = random.choice(spec_parents[spec_id])
                    new_genome_id, new_genome = self._mutation_rem_conn(parent_genome_id)

                elif chosen_mutation == 'rem_node':
                    parent_genome_id = random.choice(spec_parents[spec_id])
                    new_genome_id, new_genome = self._mutation_rem_node(parent_genome_id)

                elif chosen_mutation == 'node_layer':
                    parent_genome_id = random.choice(spec_parents[spec_id])
                    new_genome_id, new_genome = self._mutation_node_layer(parent_genome_id)

                elif chosen_mutation == 'hyperparam':
                    parent_genome_id = random.choice(spec_parents[spec_id])
                    new_genome_id, new_genome = self._mutation_hyperparam(parent_genome_id)

                elif chosen_mutation == 'crossover':
                    if len(spec_parents[spec_id]) >= 2:
                        parent_genome_1_id, parent_genome_2_id = random.sample(spec_parents[spec_id], k=2)
                        new_genome_id, new_genome = self._crossover(parent_genome_1_id, parent_genome_2_id)

                    else:
                        parent_genome_id = random.choice(spec_parents[spec_id])
                        new_genome_id, new_genome = self._mutation_node_layer(parent_genome_id)

                else:
                    raise RuntimeError("COMMENT")

                self.pop.genomes[new_genome_id] = new_genome
                new_genome_ids.append(new_genome_id)

        #### Return ####
        return new_genome_ids

    def _mutation_add_conn(self, parent_genome_id) -> (int, DeepNEATGenome):
        """"""
        parent_mutation = {'parent_id': parent_genome_id,
                           'mutation': 'add_conn',
                           'added_genes': list()}

        parent_genome = self.pop.genomes[parent_genome_id]
        g_nodes, g_conns_enabled, g_conns_disabled, preprocessing_layers, optimizer = parent_genome.get_genotype()

        number_of_conns_added = 0
        number_of_conns_to_add = math.ceil(self.mutation_degree * (len(g_conns_enabled) + len(g_conns_disabled)))

        possible_start_nodes = list()
        possible_end_nodes = list()
        for node in g_nodes.values():
            n = node[0]
            if n != 1:
                possible_start_nodes.append(n)
            if n != 0:
                possible_end_nodes.append(n)

        while number_of_conns_added < number_of_conns_to_add:

            random.shuffle(possible_start_nodes)
            random.shuffle(possible_end_nodes)

            conn_added_flag = False
            for start_node in possible_start_nodes:
                for end_node in possible_end_nodes:
                    conn = (start_node, end_node)

                    if conn not in g_conns_enabled.values():
                        gene_id, gene = self.enc.create_conn_gene(conn_start=start_node, conn_end=end_node)
                        g_conns_enabled[gene_id] = gene
                        parent_mutation['added_genes'].append(gene_id)
                        number_of_conns_added += 1
                        if conn in g_conns_disabled.values():
                            del g_conns_disabled[gene_id]

                        conn_added_flag = True

                if conn_added_flag:
                    break

            # Break loop of adding connections as all possible start node and end node combinations have been explored
            # but it was not possible to add a new connection
            if not conn_added_flag:
                break

        return self.enc.create_genome(parent_mutation=parent_mutation,
                                      generation=self.pop.generation_counter,
                                      genome_nodes=g_nodes,
                                      genome_conns_enabled=g_conns_enabled,
                                      genome_conns_disabled=g_conns_disabled,
                                      preprocessing_layers=preprocessing_layers,
                                      optimizer=optimizer)

    def _mutation_add_node(self, parent_genome_id) -> (int, DeepNEATGenome):
        """"""
        parent_mutation = {'parent_id': parent_genome_id,
                           'mutation': 'add_node',
                           'added_genes': list()}

        parent_genome = self.pop.genomes[parent_genome_id]
        g_nodes, g_conns_enabled, g_conns_disabled, preprocessing_layers, optimizer = parent_genome.get_genotype()

        node_output_dims = parent_genome.get_node_output_dims()

        number_of_nodes_to_add = math.ceil(self.mutation_degree * len(g_nodes))
        gene_ids_to_split = random.sample(g_conns_enabled.keys(), k=number_of_nodes_to_add)

        for gene_id_to_split in gene_ids_to_split:
            conn_start = g_conns_enabled[gene_id_to_split][0]
            conn_end = g_conns_enabled[gene_id_to_split][1]

            g_conns_disabled[gene_id_to_split] = g_conns_enabled[gene_id_to_split]
            del g_conns_enabled[gene_id_to_split]

            input_dim = node_output_dims[conn_start]
            chosen_layer = self.rng.choice(self.available_layers[input_dim], p=self.available_layers_p[input_dim])

            layer_config = dict()
            for layer_param, layer_param_val_range in self.layer_params[chosen_layer].items():
                if isinstance(layer_param_val_range, dict):
                    if isinstance(layer_param_val_range['min'], int) \
                            and isinstance(layer_param_val_range['max'], int) \
                            and isinstance(layer_param_val_range['step'], int):
                        layer_param_random = random.randint(layer_param_val_range['min'],
                                                            layer_param_val_range['max'])
                        chosen_layer_param = round_with_step(layer_param_random,
                                                             layer_param_val_range['min'],
                                                             layer_param_val_range['max'],
                                                             layer_param_val_range['step'])
                    elif isinstance(layer_param_val_range['min'], float) \
                            and isinstance(layer_param_val_range['max'], float) \
                            and isinstance(layer_param_val_range['step'], float):
                        layer_param_random = random.uniform(layer_param_val_range['min'],
                                                            layer_param_val_range['max'])
                        chosen_layer_param = round_with_step(layer_param_random,
                                                             layer_param_val_range['min'],
                                                             layer_param_val_range['max'],
                                                             layer_param_val_range['step'])
                    else:
                        raise NotImplementedError(
                            f"Config parameter '{layer_param}' of the {chosen_layer} layer section "
                            f"is of type dict though the dict values are not of type int or float")
                    layer_config[layer_param] = chosen_layer_param
                elif isinstance(layer_param_val_range, list):
                    layer_config[layer_param] = random.choice(layer_param_val_range)
                else:
                    layer_config[layer_param] = layer_param_val_range

            new_node = self.enc.get_node_for_split(conn_start, conn_end)
            new_layer = {'class_name': chosen_layer, 'config': layer_config}

            gene_id, gene = self.enc.create_node_gene(node=new_node, layer=new_layer)
            g_nodes[gene_id] = gene
            parent_mutation['added_genes'].append(gene_id)
            gene_id, gene = self.enc.create_conn_gene(conn_start=conn_start, conn_end=new_node)
            g_conns_enabled[gene_id] = gene
            parent_mutation['added_genes'].append(gene_id)
            gene_id, gene = self.enc.create_conn_gene(conn_start=new_node, conn_end=conn_end)
            g_conns_enabled[gene_id] = gene
            parent_mutation['added_genes'].append(gene_id)

        return self.enc.create_genome(parent_mutation=parent_mutation,
                                      generation=self.pop.generation_counter,
                                      genome_nodes=g_nodes,
                                      genome_conns_enabled=g_conns_enabled,
                                      genome_conns_disabled=g_conns_disabled,
                                      preprocessing_layers=preprocessing_layers,
                                      optimizer=optimizer)

    def _mutation_rem_conn(self, parent_genome_id) -> (int, DeepNEATGenome):
        """"""
        parent_mutation = "stubby"
        parent_genome_nodes, parent_genome_conns_enabled, parent_genome_conns_disabled, parent_preprocessing_layers, \
        parent_optimizer = parent_genome.get_genotype()
        return self.enc.create_genome(parent_mutation=parent_mutation,
                                      generation=self.pop.generation_counter,
                                      genome_nodes=parent_genome_nodes,
                                      genome_conns_enabled=parent_genome_conns_enabled,
                                      genome_conns_disabled=parent_genome_conns_disabled,
                                      preprocessing_layers=parent_preprocessing_layers,
                                      optimizer=parent_optimizer)

    def _mutation_rem_node(self, parent_genome_id) -> (int, DeepNEATGenome):
        """"""
        parent_mutation = "stubby"
        parent_genome_nodes, parent_genome_conns_enabled, parent_genome_conns_disabled, parent_preprocessing_layers, \
        parent_optimizer = parent_genome.get_genotype()
        return self.enc.create_genome(parent_mutation=parent_mutation,
                                      generation=self.pop.generation_counter,
                                      genome_nodes=parent_genome_nodes,
                                      genome_conns_enabled=parent_genome_conns_enabled,
                                      genome_conns_disabled=parent_genome_conns_disabled,
                                      preprocessing_layers=parent_preprocessing_layers,
                                      optimizer=parent_optimizer)

    def _mutation_node_layer(self, parent_genome_id) -> (int, DeepNEATGenome):
        """"""
        parent_mutation = "stubby"
        parent_genome_nodes, parent_genome_conns_enabled, parent_genome_conns_disabled, parent_preprocessing_layers, \
        parent_optimizer = parent_genome.get_genotype()
        return self.enc.create_genome(parent_mutation=parent_mutation,
                                      generation=self.pop.generation_counter,
                                      genome_nodes=parent_genome_nodes,
                                      genome_conns_enabled=parent_genome_conns_enabled,
                                      genome_conns_disabled=parent_genome_conns_disabled,
                                      preprocessing_layers=parent_preprocessing_layers,
                                      optimizer=parent_optimizer)

    def _mutation_hyperparam(self, parent_genome_id) -> (int, DeepNEATGenome):
        """"""
        parent_mutation = "stubby"
        parent_genome_nodes, parent_genome_conns_enabled, parent_genome_conns_disabled, parent_preprocessing_layers, \
        parent_optimizer = parent_genome.get_genotype()
        return self.enc.create_genome(parent_mutation=parent_mutation,
                                      generation=self.pop.generation_counter,
                                      genome_nodes=parent_genome_nodes,
                                      genome_conns_enabled=parent_genome_conns_enabled,
                                      genome_conns_disabled=parent_genome_conns_disabled,
                                      preprocessing_layers=parent_preprocessing_layers,
                                      optimizer=parent_optimizer)

    def _crossover(self, parent_genome_1_id, parent_genome_2_id) -> (int, DeepNEATGenome):
        """"""
        parent_mutation = "stubby"
        parent_genome_nodes, parent_genome_conns_enabled, parent_genome_conns_disabled, parent_preprocessing_layers, \
        parent_optimizer = parent_genome_1.get_genotype()
        return self.enc.create_genome(parent_mutation=parent_mutation,
                                      generation=self.pop.generation_counter,
                                      genome_nodes=parent_genome_nodes,
                                      genome_conns_enabled=parent_genome_conns_enabled,
                                      genome_conns_disabled=parent_genome_conns_disabled,
                                      preprocessing_layers=parent_preprocessing_layers,
                                      optimizer=parent_optimizer)
