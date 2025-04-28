#!/usr/bin/env python3
import os
import math
import random
import argparse
import neat
import numpy as np

# Utility to print an N×N grid
def print_grid(grid, N):
    for r in range(N):
        row = grid[r * N:(r + 1) * N]
        print(' '.join(str(v) if v != 0 else '.' for v in row))
    print()

class SudokuNEATSolver:
    def __init__(self, N, block_size=None, config_path='config-neat', generations=100, pop_size=150):
        """
        Initialize the NEAT Sudoku solver.
        - N: grid size (e.g. 4, 9, 16)
        - block_size: subgrid size; defaults to int(sqrt(N))
        - config_path: path to write NEAT config
        - generations: number of generations to evolve
        - pop_size: NEAT population size
        """
        self.N = N
        self.block_size = block_size or int(math.sqrt(N))
        self.CELLS = N * N
        self.DIGITS = N
        self.OUTPUTS = self.CELLS * self.DIGITS
        self.config_path = config_path
        self.generations = generations
        self.pop_size = pop_size
        self.config = None
        self.winner = None

    def _generate_config(self):
        """Auto-generate a NEAT config file for this grid size."""
        tmpl = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = {self.CELLS * 4}.0
pop_size              = {self.pop_size}
reset_on_extinction   = False

[DefaultGenome]
# node gene configuration
num_inputs            = {self.CELLS}
num_hidden            = 0
num_outputs           = {self.OUTPUTS}
feed_forward          = False
initial_connection    = full_direct

# compatibility
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5

# mutation rates
conn_add_prob         = 0.5
conn_delete_prob      = 0.5
node_add_prob         = 0.2
node_delete_prob      = 0.2

# activation options
activation_default    = sigmoid
activation_mutate_rate= 0.1
activation_options    = sigmoid

# aggregation options
aggregation_default   = sum
aggregation_mutate_rate=0.1
aggregation_options   = sum

# bias
bias_init_mean        = 0.0
bias_init_stdev       = 1.0
bias_mutate_rate      = 0.7
bias_replace_rate     = 0.1
bias_mutate_power     = 0.5
bias_max_value        = 30.0
bias_min_value        = -30.0

# response
response_init_mean    = 1.0
response_init_stdev   = 0.1
response_mutate_rate  = 0.1
response_replace_rate = 0.1
response_mutate_power = 0.1
response_max_value    = 30.0
response_min_value    = -30.0

# weight
weight_init_mean      = 0.0
weight_init_stdev     = 1.0
weight_mutate_rate    = 0.8
weight_replace_rate   = 0.1
weight_mutate_power   = 0.5
weight_max_value      = 30.0
weight_min_value      = -30.0

# enabled
enabled_default       = True
enabled_mutate_rate   = 0.01

[DefaultSpeciesSet]
compatibility_threshold = 3.0

[DefaultStagnation]
species_fitness_func  = max
max_stagnation        = 15
species_elitism       = 2

[DefaultReproduction]
elitism               = 2
survival_threshold    = 0.2
"""
        with open(self.config_path, 'w') as f:
            f.write(tmpl.lstrip())
        print(f"[INFO] NEAT config written to '{self.config_path}' for {self.N}×{self.N} grid.")

    def _make_evaluator(self, puzzles):
        """Return a NEAT eval_genomes function for the puzzles."""
        def eval_genomes(genomes, config):
            for _, genome in genomes:
                net = neat.nn.RecurrentNetwork.create(genome, config)
                total_fit = 0.0
                for puzzle in puzzles:
                    grid = puzzle.copy()
                    net.reset()
                    # sequential fill
                    while 0 in grid:
                        inputs = grid / self.DIGITS
                        outputs = np.array(net.activate(inputs)).reshape(self.CELLS, self.DIGITS)
                        empties = np.where(grid == 0)[0]
                        scores = outputs[empties].max(axis=1)
                        pick = empties[np.argmax(scores)]
                        digit = int(outputs[pick].argmax()) + 1
                        grid[pick] = digit
                    total_fit += SudokuNEATSolver._fitness_static(grid, self.N, self.block_size)
                genome.fitness = total_fit / len(puzzles)
        return eval_genomes

    @staticmethod
    def _fitness_static(grid, N, block_size):
        """Static fitness computation for any N."""
        mat = grid.reshape(N, N)
        filled = np.count_nonzero(mat)
        valid = lambda line: line.size == np.unique(line).size
        rows = sum(valid(row[row > 0]) for row in mat)
        cols = sum(valid(col[col > 0]) for col in mat.T)
        blocks = (
            mat[r:r+block_size, c:c+block_size].ravel()
            for r in range(0, N, block_size)
            for c in range(0, N, block_size)
        )
        b_ok = sum(valid(b[b > 0]) for b in blocks)
        bonus = N * 2 if filled == N*N and rows == N and cols == N and b_ok == N else 0
        return filled + rows + cols + b_ok + bonus

    def run(self, puzzles):
        """
        Evolve the population on the given puzzles. Returns the best genome.
        """
        self._generate_config()
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path
        )
        pop = neat.Population(config)
        pop.add_reporter(neat.StdOutReporter(True))
        pop.add_reporter(neat.StatisticsReporter())
        best = pop.run(self._make_evaluator(puzzles), self.generations)
        self.config = config
        self.winner = best
        return best

    def solve(self, puzzle, genome=None, verbose=True):
        """
        Solve a single puzzle with a given genome (or the last evolved one).
        Returns (solved_grid, steps).
        """
        if genome is None:
            if self.winner is None:
                raise ValueError("No genome provided or evolved yet.")
            genome = self.winner
        net = neat.nn.RecurrentNetwork.create(genome, self.config)
        net.reset()
        grid = puzzle.copy()
        steps = 0
        if verbose:
            print("Initial puzzle:")
            print_grid(grid, self.N)
        while 0 in grid:
            steps += 1
            inputs = grid / self.DIGITS
            outputs = np.array(net.activate(inputs)).reshape(self.CELLS, self.DIGITS)
            empties = np.where(grid == 0)[0]
            scores = outputs[empties].max(axis=1)
            pick = empties[np.argmax(scores)]
            digit = int(outputs[pick].argmax()) + 1
            row, col = divmod(pick, self.N)
            if verbose:
                print(f"Step {steps}: filling cell ({row}, {col}) with {digit}")
                grid[pick] = digit
                print_grid(grid, self.N)
            else:
                grid[pick] = digit
        if verbose:
            print(f"Solved in {steps} steps.")
        return grid, steps

# Helper to load puzzles from file
def load_puzzles(path, N):
    """Load puzzles: each line is N*N digits (0 for empty)."""
    puzzles = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        for idx, line in enumerate(f, 1):
            s = line.strip()
            if len(s) != N*N or any(c not in '0123456789ABCDEF'[:N+1] for c in s):
                print(f"[WARNING] skipping invalid puzzle at line {idx}")
                continue
            # map hex digits if needed
            arr = np.fromiter((int(c, 16) for c in s), dtype=int)
            puzzles.append(arr)
    return puzzles

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=4, help='Grid size N (e.g. 4, 9, 16)')
    parser.add_argument('--block', type=int, help='Block size (default sqrt(N))')
    parser.add_argument('--puzzles', default='puzzles.txt', help='Puzzle file path')
    parser.add_argument('--config', default='config-neat', help='Config path')
    parser.add_argument('--gens', type=int, default=100, help='Generations')
    parser.add_argument('--pop', type=int, default=150, help='Population size')
    args = parser.parse_args()

    puzzles = load_puzzles(args.puzzles, args.size)
    solver = SudokuNEATSolver(args.size, args.block, args.config, args.gens, args.pop)
    best = solver.run(puzzles)
    print('=== Testing on loaded puzzles ===')
    for idx, puzz in enumerate(puzzles, 1):
        print(f'Puzzle {idx}:')
        solver.solve(puzz, best, verbose=True)