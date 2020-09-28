import tfne
from absl import app, logging
from datetime import timedelta


def subgraphneat_xor_example_1(_):
    """"""
    # Set configuration specific to TFNE and the example, but not to the neuroevolutionary process. The details of the
    # neuroevolutionary process are listed in the config file.
    logging_level = logging.INFO
    config_file_path = './subgraphneat_xor_example_1_config.cfg'
    backup_dir_path = './tfne_state_backups/'
    max_generations = 128
    max_fitness = 100
    max_time = timedelta(days=0, hours=3, minutes=0, seconds=0)

    # Set logging, parse config
    logging.set_verbosity(logging_level)
    config = tfne.parse_configuration(config_file_path)

    # Initialize the environment and determine the input and output shapes genome phenotypes have to abide by
    environment = tfne.environments.XOREnvironment(weight_training=False,
                                                   verbosity=logging_level)
    env_input_shape = environment.get_input_shape()
    env_output_shape = environment.get_output_shape()

    # Initialize the chosen neuroevolution algorithm
    ne_algorithm = tfne.algorithms.SubGraphNEAT(config=config,
                                                input_shape=env_input_shape,
                                                output_shape=env_output_shape)

    # Initialize evolution engine and supply config as well as initialized NE algorithm and evaluation environment.
    engine = tfne.EvolutionEngine(ne_algorithm=ne_algorithm,
                                  environment=environment,
                                  backup_dir_path=backup_dir_path,
                                  max_generations=max_generations,
                                  max_fitness=max_fitness,
                                  max_time=max_time)

    # Initiate training process
    engine.train()

    # Get the best genome created by the neuroevolution algorithm and print it
    best_genome = ne_algorithm.get_best_genome()
    print("Best genome returned by evolution:\n")
    print(best_genome)

    # Replay best genome on environment to demonstrate it
    print("Replaying best genome:\n")
    environment.replay_genome(best_genome)

    # Serialize and save genotype and Tensorflow model to demonstrate serialization
    best_genome.save_genotype(save_dir_path='./best_genome_genotype/')
    best_genome.save_model(file_path='./best_genome_model/')


if __name__ == '__main__':
    app.run(subgraphneat_xor_example_1)
