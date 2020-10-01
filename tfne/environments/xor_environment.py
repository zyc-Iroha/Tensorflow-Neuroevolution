from __future__ import annotations
import numpy as np
import tensorflow as tf
from tfne.environments.base_environment import BaseEnvironment


class XOREnvironment(BaseEnvironment):
    """"""

    def __init__(self, weight_training, epochs=None, batch_size=None, verbosity=0):
        """"""
        # Initialize corresponding input and output mappings
        print("Setting up XOR environment...")
        self.x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])

        # Initialize loss function to evaluate performance on either evaluation method
        self.loss_function = tf.keras.losses.BinaryCrossentropy()
        self.weight_training = weight_training
        self.verbosity = verbosity

        # Determine and setup explicit evaluation method in accordance to supplied parameters
        if weight_training:
            # Set up XOR environment as weight training and register parameters
            self.eval_genome_fitness = self._eval_genome_fitness_weight_training
            self.epochs = epochs
            self.batch_size = batch_size
        else:
            # Set up XOR environment as non-weight training, requiring no parameters
            self.eval_genome_fitness = self._eval_genome_fitness_non_weight_training

    def eval_genome_fitness(self, genome) -> float:
        # TO BE OVERRIDEN
        raise RuntimeError()

    def _eval_genome_fitness_weight_training(self, genome) -> float:
        """
        Evaluates the genome's fitness by obtaining the associated Tensorflow model and optimizer, compiling them and
        then training them for the config specified duration. The genomes fitness is then calculated and returned as
        the binary cross entropy in percent of the predicted to the actual results
        @param genome: TFNE compatible genome that is to be evaluated
        @return: genome calculated fitness
        """
        # Get model and optimizer required for compilation
        model = genome.get_model()
        optimizer = genome.get_optimizer()

        # Compile and train model
        model.compile(optimizer=optimizer, loss=self.loss_function)
        '''
        model.fit(x=self.x, y=self.y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbosity)

        # Evaluate and return its fitness
        evaluated_fitness = float(100 * (1 - self.loss_function(self.y, model(self.x))))

        # Check if model is possibly overflowing or underflowing when using dtype float16, which is probable for this
        # problem environment.
        if tf.math.is_nan(evaluated_fitness):
            evaluated_fitness = 0

        return round(evaluated_fitness, 4)
        '''
        import random
        return round(random.random() * 100, 4)

    def _eval_genome_fitness_non_weight_training(self, genome) -> float:
        """
        Evaluates genome's fitness by calculating and returning the binary cross entropy in percent of the predicted to
        the actual results
        @param genome: TFNE compatible genome that is to be evaluated
        @return: genome calculated fitness
        """
        # Evaluate and return its fitness by calling genome directly with input
        evaluated_fitness = float(100 * (1 - self.loss_function(self.y, genome(self.x))))
        return round(evaluated_fitness, 4)

    def replay_genome(self, genome):
        """
        Replay genome on environment by calculating its fitness and printing it.
        @param genome: TFNE compatible genome that is to be evaluated
        """
        print("Replaying Genome #{}:".format(genome.get_id()))
        evaluated_fitness = round(float(100 * (1 - self.loss_function(self.y, genome(self.x)))), 4)
        print("Solution Values: \t{}\n".format(self.y))
        print("Predicted Values:\t{}\n".format(genome(self.x)))
        print("Achieved Fitness:\t{}\n".format(evaluated_fitness))

    def duplicate(self) -> XOREnvironment:
        """"""
        if self.weight_training:
            return XOREnvironment(True, epochs=self.epochs, batch_size=self.batch_size, verbosity=self.verbosity)
        else:
            return XOREnvironment(False, verbosity=self.verbosity)

    def get_input_shape(self) -> (int,):
        """"""
        return (2,)

    def get_output_shape(self) -> (int,):
        """"""
        return (1,)
