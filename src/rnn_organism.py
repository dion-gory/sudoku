import numpy as np
import copy


class RNNOrganism:
    def __init__(self, dimensions, use_bias=True, output='softmax', mutation_std=0.05):
        """
        Initialize a recurrent neural network organism.

        Parameters:
        - dimensions: List of network dimensions [input_size, hidden_size, output_size]
        - use_bias: Whether to use bias terms
        - output: The activation function for the output layer
        - mutation_std: Standard deviation for mutations
        """
        if len(dimensions) != 3:
            raise ValueError("RNN requires exactly 3 dimensions: [input_size, hidden_size, output_size]")

        self.input_size = dimensions[0]
        self.hidden_size = dimensions[1]
        self.output_size = dimensions[2]
        self.use_bias = use_bias
        self.output = output
        self.output_activation = self._activation(output)
        self.mutation_std = mutation_std

        # Hidden state for stateful prediction
        self.hidden_state = None

        # Initialize weights
        # Weight matrices for input->hidden, hidden->hidden, and hidden->output connections
        std_ih = np.sqrt(2 / (self.input_size + self.hidden_size))
        std_hh = np.sqrt(2 / (self.hidden_size + self.hidden_size))
        std_ho = np.sqrt(2 / (self.hidden_size + self.output_size))

        # Store in a similar format to the original Organism for compatibility
        self.layers = []
        self.biases = []

        # Input to hidden weights
        self.W_ih = np.random.normal(0, std_ih, (self.input_size, self.hidden_size))
        self.layers.append(self.W_ih)

        # Hidden to hidden (recurrent) weights
        self.W_hh = np.random.normal(0, std_hh, (self.hidden_size, self.hidden_size))

        # Hidden to output weights
        self.W_ho = np.random.normal(0, std_ho, (self.hidden_size, self.output_size))
        self.layers.append(self.W_ho)

        # Initialize biases
        if use_bias:
            self.b_h = np.random.normal(0, std_ih, (1, self.hidden_size))
            self.b_o = np.random.normal(0, std_ho, (1, self.output_size))
        else:
            self.b_h = np.zeros((1, self.hidden_size))
            self.b_o = np.zeros((1, self.output_size))

        self.biases.append(self.b_h)
        self.biases.append(self.b_o)

    def _activation(self, output):
        """Return a specified activation function.

        output - a function, or the name of an activation function as a string.
                 String must be one of softmax, sigmoid, linear, tanh."""
        if output == 'softmax':
            return lambda X: np.exp(X) / np.sum(np.exp(X), axis=1).reshape(-1, 1)
        if output == 'sigmoid':
            return lambda X: (1 / (1 + np.exp(-X)))
        if output == 'linear':
            return lambda X: X
        if output == 'tanh':
            return lambda X: np.tanh(X)
        else:
            return output

    def predict(self, X):
        """
        Apply the RNN to input X and return the output.
        This method maintains compatibility with the original Organism class
        by processing a single timestep of input.

        Parameters:
        - X: 2D array of shape (batch_size, input_size)

        Returns:
        - Output of shape (batch_size, output_size)
        """
        if not X.ndim == 2:
            raise ValueError(f'Input has {X.ndim} dimensions, expected 2')
        if not X.shape[1] == self.input_size:
            raise ValueError(f'Input has {X.shape[1]} features, expected {self.input_size}')

        # Initialize hidden state with zeros
        h = np.zeros((X.shape[0], self.hidden_size))

        # Update hidden state: h = tanh(X @ W_ih + h @ W_hh + b_h)
        h_new = X @ self.W_ih + h @ self.W_hh + np.ones((X.shape[0], 1)) @ self.b_h
        h = np.tanh(h_new)  # Use tanh activation for hidden state

        # Compute output
        output = h @ self.W_ho + np.ones((X.shape[0], 1)) @ self.b_o

        # Apply output activation
        output = self.output_activation(output)

        return output

    def predict_sequence(self, X_sequence):
        """
        Process a sequence of inputs and return all outputs.

        Parameters:
        - X_sequence: 3D array of shape (batch_size, sequence_length, input_size)

        Returns:
        - Outputs for each timestep as an array of shape (batch_size, sequence_length, output_size)
        """
        if not X_sequence.ndim == 3:
            raise ValueError(f'Input sequence has {X_sequence.ndim} dimensions, expected 3')
        if not X_sequence.shape[2] == self.input_size:
            raise ValueError(f'Input has {X_sequence.shape[2]} features, expected {self.input_size}')

        batch_size, seq_length, _ = X_sequence.shape

        # Initialize outputs and hidden state
        outputs = np.zeros((batch_size, seq_length, self.output_size))
        h = np.zeros((batch_size, self.hidden_size))

        # Process each timestep
        for t in range(seq_length):
            # Get input at this timestep
            X_t = X_sequence[:, t, :]

            # Update hidden state: h = tanh(X @ W_ih + h @ W_hh + b_h)
            h_new = X_t @ self.W_ih + h @ self.W_hh + np.ones((batch_size, 1)) @ self.b_h
            h = np.tanh(h_new)  # Use tanh activation for hidden state

            # Compute output
            output = h @ self.W_ho + np.ones((batch_size, 1)) @ self.b_o

            # Apply output activation
            output = self.output_activation(output)

            # Store output
            outputs[:, t, :] = output

        return outputs

    def predict_choice(self, X, deterministic=True):
        """Apply `predict` to X and return the organism's "choice".

        if deterministic then return the choice with the highest score.
        if not deterministic then interpret output as probabilities and select
        from them randomly, according to their probabilities.
        """
        probabilities = self.predict(X)
        if deterministic:
            return np.argmax(probabilities, axis=1).reshape((-1, 1))
        if any(np.sum(probabilities, axis=1) != 1):
            raise ValueError(f'Output values must sum to 1 to use deterministic=False')
        if any(probabilities < 0):
            raise ValueError(f'Output values cannot be negative to use deterministic=False')
        choices = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            U = np.random.rand(X.shape[0])
            c = 0
            while U > probabilities[i, c]:
                U -= probabilities[i, c]
                c += 1
            else:
                choices[i] = c
        return choices.reshape((-1, 1))

    def predict_sequence_choice(self, X_sequence, deterministic=True):
        """Apply `predict_sequence` to X_sequence and return choices for each timestep.

        Parameters:
        - X_sequence: 3D array of shape (batch_size, sequence_length, input_size)
        - deterministic: If True, choose highest probability output; if False, sample from distribution

        Returns:
        - Choices for each timestep as array of shape (batch_size, sequence_length, 1)
        """
        probabilities = self.predict_sequence(X_sequence)
        batch_size, seq_length, _ = probabilities.shape
        choices = np.zeros((batch_size, seq_length, 1))

        for t in range(seq_length):
            if deterministic:
                choices[:, t, 0] = np.argmax(probabilities[:, t, :], axis=1)
            else:
                # Check valid probability distribution
                if any(np.sum(probabilities[:, t, :], axis=1) != 1):
                    raise ValueError(f'Output values must sum to 1 to use deterministic=False')
                if any(probabilities[:, t, :] < 0):
                    raise ValueError(f'Output values cannot be negative to use deterministic=False')

                # Sample from distribution
                for i in range(batch_size):
                    U = np.random.rand()
                    c = 0
                    while U > probabilities[i, t, c]:
                        U -= probabilities[i, t, c]
                        c += 1
                    else:
                        choices[i, t, 0] = c

        return choices

    def mutate(self):
        """Mutate the organism's weights in place."""
        # Mutate input->hidden weights
        self.W_ih += np.random.normal(0, self.mutation_std, self.W_ih.shape)

        # Mutate hidden->hidden (recurrent) weights
        self.W_hh += np.random.normal(0, self.mutation_std, self.W_hh.shape)

        # Mutate hidden->output weights
        self.W_ho += np.random.normal(0, self.mutation_std, self.W_ho.shape)

        # Mutate biases if used
        if self.use_bias:
            self.b_h += np.random.normal(0, self.mutation_std, self.b_h.shape)
            self.b_o += np.random.normal(0, self.mutation_std, self.b_o.shape)

        # Update the layer references
        self.layers[0] = self.W_ih
        self.layers[1] = self.W_ho
        self.biases[0] = self.b_h
        self.biases[1] = self.b_o

    def mate(self, other, mutate=True):
        """Mate two organisms together, create an offspring, mutate it, and return it."""
        if not isinstance(other, RNNOrganism):
            raise ValueError("Can only mate with another RNNOrganism")

        if self.use_bias != other.use_bias:
            raise ValueError('Both parents must use bias or not use bias')
        if self.input_size != other.input_size or self.hidden_size != other.hidden_size or self.output_size != other.output_size:
            raise ValueError('Both parents must have same dimensions')

        child = copy.deepcopy(self)

        # Randomly inherit input->hidden weights
        pass_on_ih = np.random.rand(self.input_size, self.hidden_size) < 0.5
        child.W_ih = pass_on_ih * self.W_ih + ~pass_on_ih * other.W_ih

        # Randomly inherit hidden->hidden weights
        pass_on_hh = np.random.rand(self.hidden_size, self.hidden_size) < 0.5
        child.W_hh = pass_on_hh * self.W_hh + ~pass_on_hh * other.W_hh

        # Randomly inherit hidden->output weights
        pass_on_ho = np.random.rand(self.hidden_size, self.output_size) < 0.5
        child.W_ho = pass_on_ho * self.W_ho + ~pass_on_ho * other.W_ho

        # Randomly inherit biases
        if self.use_bias:
            pass_on_bh = np.random.rand(1, self.hidden_size) < 0.5
            pass_on_bo = np.random.rand(1, self.output_size) < 0.5
            child.b_h = pass_on_bh * self.b_h + ~pass_on_bh * other.b_h
            child.b_o = pass_on_bo * self.b_o + ~pass_on_bo * other.b_o

        # Update the layer references
        child.layers[0] = child.W_ih
        child.layers[1] = child.W_ho
        child.biases[0] = child.b_h
        child.biases[1] = child.b_o

        if mutate:
            child.mutate()
        return child

    def organism_like(self):
        """Return a new organism with the same shape and activations but reinitialized weights."""
        dimensions = [self.input_size, self.hidden_size, self.output_size]
        return RNNOrganism(dimensions, use_bias=self.use_bias, output=self.output, mutation_std=self.mutation_std)

    def reset_state(self, batch_size=1):
        """Reset the hidden state to zeros."""
        self.hidden_state = np.zeros((batch_size, self.hidden_size))

    def predict_stateful(self, X):
        """
        Apply the RNN to input X in a stateful manner, preserving hidden state between calls.
        Call reset_state() when starting a new sequence.

        Parameters:
        - X: 2D array of shape (batch_size, input_size)

        Returns:
        - Output of shape (batch_size, output_size)
        """
        if not X.ndim == 2:
            raise ValueError(f'Input has {X.ndim} dimensions, expected 2')
        if not X.shape[1] == self.input_size:
            raise ValueError(f'Input has {X.shape[1]} features, expected {self.input_size}')

        # Initialize hidden state if needed
        if self.hidden_state is None or self.hidden_state.shape[0] != X.shape[0]:
            self.reset_state(X.shape[0])

        # Update hidden state: h = tanh(X @ W_ih + h @ W_hh + b_h)
        h_new = X @ self.W_ih + self.hidden_state @ self.W_hh + np.ones((X.shape[0], 1)) @ self.b_h
        self.hidden_state = np.tanh(h_new)  # Use tanh activation for hidden state

        # Compute output
        output = self.hidden_state @ self.W_ho + np.ones((X.shape[0], 1)) @ self.b_o

        # Apply output activation
        output = self.output_activation(output)

        return output