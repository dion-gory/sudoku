import copy
import numpy as np


class Organism:
    def __init__(self, dimensions, use_bias=True, output='softmax', mutation_std=0.05):
        self.layers = []
        self.biases = []
        self.use_bias = use_bias
        self.output = output
        self.output_activation = self._activation(output)
        self.mutation_std = mutation_std

        # Initialize weights with improved scaling for better convergence
        for i in range(len(dimensions) - 1):
            shape = (dimensions[i], dimensions[i + 1])

            # He initialization for ReLU activation - better for deep networks
            std = np.sqrt(2.0 / dimensions[i])

            # Create layer with proper initialization
            layer = np.random.normal(0, std, shape)

            # Initialize biases with small positive values to encourage activation
            bias = np.random.normal(0.01, 0.01, (1, dimensions[i + 1])) * use_bias

            self.layers.append(layer)
            self.biases.append(bias)

    def _activation(self, output):
        """Return a specified activation function with improved numerical stability.

        output - a function, or the name of an activation function as a string.
                 String must be one of softmax, sigmoid, linear, tanh."""
        if output == 'softmax':
            # Improved softmax with numerical stability
            return lambda X: np.exp(X - np.max(X, axis=1, keepdims=True)) / \
                             np.sum(np.exp(X - np.max(X, axis=1, keepdims=True)), axis=1, keepdims=True)
        if output == 'sigmoid':
            # Stable sigmoid
            return lambda X: 1.0 / (1.0 + np.exp(-np.clip(X, -88.0, 88.0)))
        if output == 'linear':
            return lambda X: X
        if output == 'tanh':
            return lambda X: np.tanh(X)
        if output == 'relu':
            return lambda X: np.maximum(0, X)
        else:
            return output

    def predict(self, X):
        """Apply the function described by the organism to input X and return the output."""
        if not X.ndim == 2:
            raise ValueError(f'Input has {X.ndim} dimensions, expected 2')
        if not X.shape[1] == self.layers[0].shape[0]:
            raise ValueError(f'Input has {X.shape[1]} features, expected {self.layers[0].shape[0]}')

        activations = X  # Input layer activations

        # Forward pass through the network
        for index, (layer, bias) in enumerate(zip(self.layers, self.biases)):
            # Linear transformation (weights and bias)
            Z = activations @ layer + np.ones((activations.shape[0], 1)) @ bias

            # Apply activation function
            if index == len(self.layers) - 1:
                activations = self.output_activation(Z)  # Output activation
            else:
                # ReLU activation with small leakage to prevent dead neurons
                activations = np.maximum(0.01 * Z, Z)  # Leaky ReLU

        return activations

    def predict_choice(self, X, deterministic=True):
        """Apply `predict` to X and return the organism's "choice".

        if deterministic then return the choice with the highest score.
        if not deterministic then interpret output as probabilities and select
        from them randomly, according to their probabilities.
        """
        probabilities = self.predict(X)

        if deterministic:
            return np.argmax(probabilities, axis=1).reshape((-1, 1))

        # Ensure probabilities are valid for sampling
        if self.output != 'softmax':
            # Normalize if not using softmax
            row_sums = np.sum(probabilities, axis=1, keepdims=True)
            probabilities = probabilities / row_sums

        if np.any(probabilities < 0):
            probabilities = np.maximum(0, probabilities)
            row_sums = np.sum(probabilities, axis=1, keepdims=True)
            probabilities = probabilities / row_sums

        # Sample from the probability distribution
        cumulative_probs = np.cumsum(probabilities, axis=1)
        random_values = np.random.random(size=(X.shape[0], 1))

        # Find the index where random value is less than cumulative probability
        choices = np.sum(random_values > cumulative_probs, axis=1)

        return choices.reshape((-1, 1))

    def mutate(self):
        """Mutate the organism's weights in place using scaled mutations."""
        for i in range(len(self.layers)):
            # Scale mutations by weight magnitude for more stable learning
            layer_scale = np.mean(np.abs(self.layers[i])) + 1e-8
            self.layers[i] += np.random.normal(0, self.mutation_std * layer_scale, self.layers[i].shape)

            if self.use_bias:
                bias_scale = np.mean(np.abs(self.biases[i])) + 1e-8
                self.biases[i] += np.random.normal(0, self.mutation_std * bias_scale, self.biases[i].shape)

    def mate(self, other, mutate=True):
        """Mate two organisms together with improved crossover, create an offspring, mutate it, and return it."""
        if self.use_bias != other.use_bias:
            raise ValueError('Both parents must use bias or not use bias')
        if not len(self.layers) == len(other.layers):
            raise ValueError('Both parents must have same number of layers')
        if not all(self.layers[x].shape == other.layers[x].shape for x in range(len(self.layers))):
            raise ValueError('Both parents must have same shape')

        child = copy.deepcopy(self)

        for i in range(len(child.layers)):
            # Improved crossover: Weighted average of parents rather than just binary selection
            # This allows for more nuanced inheritance of traits
            alpha = np.random.rand(child.layers[i].shape[0], child.layers[i].shape[1])
            child.layers[i] = alpha * self.layers[i] + (1 - alpha) * other.layers[i]

            # Similar for biases
            alpha_bias = np.random.rand(1, child.biases[i].shape[1])
            child.biases[i] = alpha_bias * self.biases[i] + (1 - alpha_bias) * other.biases[i]

        if mutate:
            child.mutate()

        return child

    def organism_like(self):
        """Return a new organism with the same shape and activations but reinitialized weights."""
        dimensions = [x.shape[0] for x in self.layers] + [self.layers[-1].shape[1]]
        return Organism(dimensions, use_bias=self.use_bias, output=self.output, mutation_std=self.mutation_std)