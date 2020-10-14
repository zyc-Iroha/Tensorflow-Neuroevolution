import random
from tfne.encodings.deepneat import DeepNEATGenome
from tfne.helper_functions import round_with_step


class DeepNEATInitialization:
    def _create_initial_genome(self) -> (int, DeepNEATGenome):
        """"""
        parent_mutation = {'parent_id': None,
                           'mutation': 'init'}

        # !FIXME not considering input layers and preprocessing layers but instead only using input shape dim
        input_dim = len(self.input_shape) + 1
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
                    raise NotImplementedError(f"Config parameter '{layer_param}' of the {chosen_layer} layer section "
                                              f"is of type dict though the dict values are not of type int or float")
                layer_config[layer_param] = chosen_layer_param
            elif isinstance(layer_param_val_range, list):
                layer_config[layer_param] = random.choice(layer_param_val_range)
            else:
                layer_config[layer_param] = layer_param_val_range

        initial_layer = {'class_name': chosen_layer, 'config': layer_config}

        genome_nodes = dict()
        gene_id, gene = self.enc.create_node_gene(node=1, layer=None)
        genome_nodes[gene_id] = gene
        gene_id, gene = self.enc.create_node_gene(node=2, layer=initial_layer)
        genome_nodes[gene_id] = gene

        genome_conns_enabled = dict()
        gene_id, gene = self.enc.create_conn_gene(conn_start=1, conn_end=2)
        genome_conns_enabled[gene_id] = gene

        genome_conns_disabled = dict()

        # !FIXME No preprocessing layers for this prototype
        preprocessing_layers = []

        chosen_optimizer = self.rng.choice(self.available_optimizers, p=self.available_optimizers_p)

        optimizer_config = dict()
        for opt_param, opt_param_val_range in self.optimizer_params[chosen_optimizer].items():
            if isinstance(opt_param_val_range, dict):
                if isinstance(opt_param_val_range['min'], int) \
                        and isinstance(opt_param_val_range['max'], int) \
                        and isinstance(opt_param_val_range['step'], int):
                    opt_param_random = random.randint(opt_param_val_range['min'],
                                                      opt_param_val_range['max'])
                    chosen_opt_param = round_with_step(opt_param_random,
                                                       opt_param_val_range['min'],
                                                       opt_param_val_range['max'],
                                                       opt_param_val_range['step'])
                elif isinstance(opt_param_val_range['min'], float) \
                        and isinstance(opt_param_val_range['max'], float) \
                        and isinstance(opt_param_val_range['step'], float):
                    opt_param_random = random.uniform(opt_param_val_range['min'],
                                                      opt_param_val_range['max'])
                    chosen_opt_param = round_with_step(opt_param_random,
                                                       opt_param_val_range['min'],
                                                       opt_param_val_range['max'],
                                                       opt_param_val_range['step'])
                else:
                    raise NotImplementedError(f"Config parameter '{opt_param}' of the {chosen_optimizer} optimizer "
                                              f"section is of type dict though the dict values are not of type int or "
                                              f"float")
                optimizer_config[opt_param] = chosen_opt_param
            elif isinstance(opt_param_val_range, list):
                optimizer_config[opt_param] = random.choice(opt_param_val_range)
            else:
                optimizer_config[opt_param] = opt_param_val_range

        optimizer = {'class_name': chosen_optimizer, 'config': optimizer_config}

        return self.enc.create_genome(parent_mutation=parent_mutation,
                                      generation=self.pop.generation_counter,
                                      genome_nodes=genome_nodes,
                                      genome_conns_enabled=genome_conns_enabled,
                                      genome_conns_disabled=genome_conns_disabled,
                                      preprocessing_layers=preprocessing_layers,
                                      optimizer=optimizer)
