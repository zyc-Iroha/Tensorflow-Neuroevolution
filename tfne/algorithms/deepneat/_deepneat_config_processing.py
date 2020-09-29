import tensorflow as tf
from tfne.helper_functions import read_option_from_config, create_list_with_probabilities


class DeepNEATConfigProcessing:
    def _process_config(self):
        """"""
        # COMMENT
        self.pop_size = read_option_from_config(self.config, 'GENERAL', 'pop_size')
        self.consistency_evals = read_option_from_config(self.config, 'GENERAL', 'consistency_evals')
        tmp = read_option_from_config(self.config, 'GENERAL', 'available_layers')
        self.available_layers, self.available_layers_p = create_list_with_probabilities(tmp)
        tmp = read_option_from_config(self.config, 'GENERAL', 'available_optimizers')
        self.available_optimizers, self.available_optimizers_p = create_list_with_probabilities(tmp)

        # COMMENT
        self.dtype = read_option_from_config(self.config, 'GENOME', 'dtype')
        self.recurrent_graph = read_option_from_config(self.config, 'GENOME', 'recurrent_graph')
        self.recurrent_stateful = read_option_from_config(self.config, 'GENOME', 'recurrent_stateful')
        self.recurrent_init = read_option_from_config(self.config, 'GENOME', 'recurrent_init')
        self.input_scaling = read_option_from_config(self.config, 'GENOME', 'input_scaling')
        self.merge_method = read_option_from_config(self.config, 'GENOME', 'merge_method')
        self.preprocessing_layers = read_option_from_config(self.config, 'GENOME', 'preprocessing_layers')
        self.input_layers = read_option_from_config(self.config, 'GENOME', 'input_layers')
        self.output_layers = read_option_from_config(self.config, 'GENOME', 'output_layers')

        # COMMENT
        self.spec_type = read_option_from_config(self.config, 'SPECIATION', 'spec_type')
        if self.spec_type == 'dynamic':
            self.spec_species_count = read_option_from_config(self.config, 'SPECIATION', 'spec_species_count')
            self.spec_species_fitness = read_option_from_config(self.config, 'SPECIATION', 'spec_species_fitness')
            self.spec_distance = read_option_from_config(self.config, 'SPECIATION', 'spec_distance')
            self.spec_distance_inc = read_option_from_config(self.config, 'SPECIATION', 'spec_distance_inc')
            self.spec_distance_calc = read_option_from_config(self.config, 'SPECIATION', 'spec_distance_calc')
            self.spec_genome_elitism = read_option_from_config(self.config, 'SPECIATION', 'spec_genome_elitism')
            self.spec_min_offspring = read_option_from_config(self.config, 'SPECIATION', 'spec_min_offspring')
            self.spec_reprod_thres = read_option_from_config(self.config, 'SPECIATION', 'spec_reprod_thres')
            self.spec_max_stagnation = read_option_from_config(self.config, 'SPECIATION', 'spec_max_stagnation')
            self.spec_species_elitism = read_option_from_config(self.config, 'SPECIATION', 'spec_species_elitism')
            self.spec_rebase_repr = read_option_from_config(self.config, 'SPECIATION', 'spec_rebase_repr')
            self.spec_reinit_extinct = read_option_from_config(self.config, 'SPECIATION', 'spec_reinit_extinct')
        else:
            raise NotImplementedError("TODO")

        # COMMENT
        self.max_mutation = read_option_from_config(self.config, 'EVOLUTION', 'max_mutation')
        mutation_methods = dict()
        mutation_methods['add_conn'] = read_option_from_config(self.config, 'EVOLUTION', 'mutation_add_conn_prob')
        mutation_methods['add_node'] = read_option_from_config(self.config, 'EVOLUTION', 'mutation_add_node_prob')
        mutation_methods['rem_conn'] = read_option_from_config(self.config, 'EVOLUTION', 'mutation_rem_conn_prob')
        mutation_methods['rem_node'] = read_option_from_config(self.config, 'EVOLUTION', 'mutation_rem_node_prob')
        mutation_methods['node_layer'] = read_option_from_config(self.config, 'EVOLUTION', 'mutation_node_layer_prob')
        mutation_methods['hyperparam'] = read_option_from_config(self.config, 'EVOLUTION', 'mutation_hyperparam_prob')
        mutation_methods['crossover'] = read_option_from_config(self.config, 'EVOLUTION', 'crossover_prob')
        self.mutation_methods, self.mutation_methods_p = create_list_with_probabilities(mutation_methods)

        # COMMENT
        self.layer_params = dict()
        for available_layer in self.available_layers:
            config_section_str = 'LAYER_' + available_layer.upper()
            if not self.config.has_section(config_section_str):
                raise RuntimeError("COMMENT")

            params = dict()
            for param in self.config.options(config_section_str):
                tmp = read_option_from_config(self.config, config_section_str, param)
                if isinstance(tmp, dict) and 'stddev' not in tmp:
                    tmp['stddev'] = (tmp['max'] - tmp['min']) / 10
                params[param] = tmp

            self.layer_params[available_layer] = params

        # COMMENT
        self.optimizer_params = dict()
        for available_optimizer in self.available_optimizers:
            config_section_str = 'OPTIMIZER_' + available_optimizer.upper()
            if not self.config.has_section(config_section_str):
                raise RuntimeError("COMMENT")

            params = dict()
            for param in self.config.options(config_section_str):
                tmp = read_option_from_config(self.config, config_section_str, param)
                if isinstance(tmp, dict) and 'stddev' not in tmp:
                    tmp['stddev'] = (tmp['max'] - tmp['min']) / 10
                params[param] = tmp

            self.optimizer_params[available_optimizer] = params

        # COMMENT
        self.preprocessing_params = dict()
        for preprocessing_layer in self.preprocessing_layers:
            config_section_str = 'PREPROCESSING_' + preprocessing_layer.upper()
            if not self.config.has_section(config_section_str):
                raise RuntimeError("COMMENT")

            params = dict()
            for param in self.config.options(config_section_str):
                tmp = read_option_from_config(self.config, config_section_str, param)
                if isinstance(tmp, dict) and 'stddev' not in tmp:
                    tmp['stddev'] = (tmp['max'] - tmp['min']) / 10
                params[param] = tmp

            self.preprocessing_params[preprocessing_layer] = params

    def _sanity_check_config(self):
        """"""
        pass
