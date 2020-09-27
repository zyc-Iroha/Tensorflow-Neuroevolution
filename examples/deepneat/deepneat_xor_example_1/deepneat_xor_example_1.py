import tfne
from absl import app, logging
from datetime import timedelta


def deepneat_xor_example(_):
    """"""
    # Set configuration specific to TFNE and the example, but not to the neuroevolutionary process. The details of the
    # neuroevolutionary process are listed in the config file.
    logging_level = logging.INFO
    config_file_path = './deepneat_xor_dynamic_example_config.cfg'
    backup_dir_path = './tfne_state_backups/'
    max_generations = 64
    max_fitness = 100
    max_time = timedelta(days=0, hours=3, minutes=0, seconds=0)
    training_epochs = 10
    final_epochs = 100
    batch_size = None

    # Set logging, parse config
    logging.set_verbosity(logging_level)
    config = tfne.parse_configuration(config_file_path)

    # Initialize the environment and the specific NE algorithm
    environment = tfne.environments.XOREnvironment(weight_training=True,
                                                   epochs=training_epochs,
                                                   batch_size=batch_size,
                                                   verbosity=logging_level)
    ne_algorithm = tfne.algorithms.DeepNEAT(config=config)

    # Initialize evolution engine and supply config as well as initialized NE algorithm and evaluation environment.
    engine = tfne.EvolutionEngine(ne_algorithm=ne_algorithm,
                                  environment=environment,
                                  backup_dir_path=backup_dir_path,
                                  max_generations=max_generations,
                                  max_fitness=max_fitness,
                                  max_time=max_time)

    # Initiate training process
    engine.train()

    # Get the best genome and best consistent genome created by the neuroevolution algorithm
    best_genome = ne_algorithm.get_best_genome()
    best_consistent_genome = ne_algorithm.get_best_consistent_genome()

    # Increase epoch count in environment for a final training of the best genome. Train the genome and then replay it.
    print("Best genome returned by evolution:\n")
    print(best_genome)
    print(f"Training best genome for {final_epochs} epochs...\n")
    environment = tfne.environments.XOREnvironment(weight_training=True,
                                                   epochs=final_epochs,
                                                   batch_size=batch_size,
                                                   verbosity=logging_level)
    environment.eval_genome_fitness(best_genome)
    environment.replay_genome(best_genome)

    # Serialize and save genotype and Tensorflow model to demonstrate serialization
    best_genome.save_genotype(save_dir_path='./best_genome_genotype/')
    best_genome.save_model(file_path='./best_genome_model/')

    # If best genome and best consistent genome are different, then also evaluate that consistent genome for 100 epochs
    if best_genome.get_id() != best_consistent_genome.get_id():
        # Increase epoch count in environment for a final training of the best consistent genome. Train the genome and
        # then replay it.
        print("Best consistent genome returned by evolution:\n")
        print(best_consistent_genome)
        print(f"Training consistent best genome for {final_epochs} epochs...\n")
        environment.eval_genome_fitness(best_consistent_genome)
        environment.replay_genome(best_consistent_genome)

        # Serialize and save genotype and Tensorflow model to demonstrate serialization
        best_consistent_genome.save_genotype(save_dir_path='./best_consistent_genome_genotype/')
        best_consistent_genome.save_model(file_path='./best_consistent_genome_model/')


if __name__ == '__main__':
    app.run(deepneat_xor_example)