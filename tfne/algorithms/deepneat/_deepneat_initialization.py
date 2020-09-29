import random

from tfne.encodings.deepneat import DeepNEATGenome
from tfne.helper_functions import round_with_step


class DeepNEATInitialization:
    def _create_initial_genome(self, initial_layer) -> (int, DeepNEATGenome):
        """"""
        parent_mutation = {'parent_id': None,
                           'mutation': 'init'}

        genome_graph = dict()
        gene_id, gene = self.enc.create_node_gene(node=1, layer=None)
        genome_graph[gene_id] = gene
        gene_id, gene = self.enc.create_node_gene(node=2, layer=initial_layer)
        genome_graph[gene_id] = gene
        gene_id, gene = self.enc.create_conn_gene(conn_start=1, conn_end=2)
        genome_graph[gene_id] = gene

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

        optimizer = {'class_name': chosen_optimizer, 'config': dict()}

        return self.enc.create_genome(parent_mutation=parent_mutation,
                                      generation=self.pop.generation_counter,
                                      genome_graph=genome_graph,
                                      preprocessing_layers=preprocessing_layers,
                                      optimizer=optimizer)
