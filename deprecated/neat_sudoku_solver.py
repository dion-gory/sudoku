#!/usr/bin/env python3
import os
import math
import argparse
import neat
import numpy as np
import multiprocessing
import random
from neat.parallel import ParallelEvaluator
from functools import partial

# Utility to print an N×N grid
def print_grid(grid, N):
    for r in range(N):
        row = grid[r * N:(r + 1) * N]
        print(' '.join(str(v) if v != 0 else '.' for v in row))
    print()

# Check if placing digit at idx is valid in current grid
def is_valid_move(grid, N, block_size, idx, digit):
    row, col = divmod(idx, N)
    # Check row
    row_vals = grid[row*N:(row+1)*N]
    if digit in row_vals:
        return False
    # Check column
    if digit in grid[col::N]:
        return False
    # Check block
    r0 = (row // block_size) * block_size
    c0 = (col // block_size) * block_size
    for rr in range(r0, r0 + block_size):
        for cc in range(c0, c0 + block_size):
            if grid[rr*N + cc] == digit:
                return False
    return True

# Top-level evaluator function to be picklable
def evaluate_genome(genome, config, puzzles, N, block_size, sample_size,
                    reward_valid, penalty_invalid):
    net = neat.nn.RecurrentNetwork.create(genome, config)
    total_fit = 0.0
    # randomly sample puzzles for fitness evaluation
    subset = random.sample(puzzles, sample_size) if sample_size and sample_size < len(puzzles) else puzzles
    for puzzle in subset:
        grid = puzzle.copy()
        net.reset()
        dynamic_score = 0.0
        # sequential fill with rewards & penalties
        while 0 in grid:
            inputs = grid / N
            outputs = np.array(net.activate(inputs)).reshape(N * N, N)
            empties = np.where(grid == 0)[0]
            scores = outputs[empties].max(axis=1)
            pick = empties[np.argmax(scores)]
            digit = int(outputs[pick].argmax()) + 1
            # reward or penalize move
            if is_valid_move(grid, N, block_size, pick, digit):
                dynamic_score += reward_valid
            else:
                dynamic_score -= penalty_invalid
            grid[pick] = digit
        # static fitness for final grid
        static_score = SudokuNEATSolver._fitness_static(grid, N, block_size)
        total_fit += dynamic_score + static_score
    return total_fit / len(subset)

class SudokuNEATSolver:
    def __init__(self, N, block_size=None, config_path='config-neat', generations=100,
                 pop_size=150, workers=None, sample_size=8,
                 reward_valid=1.0, penalty_invalid=1.0):
        """
        Initialize the NEAT Sudoku solver.
        - N: grid size (e.g. 4, 9, 16)
        - block_size: subgrid size; defaults to int(sqrt(N))
        - config_path: path to write NEAT config file
        - generations: number of generations to evolve
        - pop_size: NEAT population size
        - workers: number of parallel workers (defaults to CPU count)
        - sample_size: number of puzzles to sample per genome evaluation
        - reward_valid: fitness reward for a valid move
        - penalty_invalid: fitness penalty for an invalid move
        """
        self.N = N
        self.block_size = block_size or int(math.sqrt(N))
        self.CELLS = N * N
        self.DIGITS = N
        self.OUTPUTS = self.CELLS * self.DIGITS
        self.config_path = config_path
        self.generations = generations
        self.pop_size = pop_size
        self.workers = workers or multiprocessing.cpu_count()
        self.sample_size = sample_size
        self.reward_valid = reward_valid
        self.penalty_invalid = penalty_invalid
        self.config = None
        self.winner = None

    def _generate_config(self):
        tmpl = f"""
[NEAT]
fitness_criterion     = max
fitness_threshold     = {self.CELLS * 2000}.0
pop_size              = {self.pop_size}
reset_on_extinction   = False

[DefaultGenome]
num_inputs            = {self.CELLS}
num_hidden            = 0
num_outputs           = {self.OUTPUTS}
feed_forward          = False
initial_connection    = full_direct
compatibility_disjoint_coefficient = 1.0
compatibility_weight_coefficient   = 0.5
conn_add_prob         = 0.5
conn_delete_prob      = 0.5
node_add_prob         = 0.2
node_delete_prob      = 0.2
activation_default    = sigmoid
activation_mutate_rate= 0.1
activation_options    = sigmoid
aggregation_default   = sum
aggregation_mutate_rate=0.1
aggregation_options   = sum
bias_init_mean        = 0.0
bias_init_stdev       = 1.0
bias_mutate_rate      = 0.7
bias_replace_rate     = 0.1
bias_mutate_power     = 0.5
bias_max_value        = 30.0
bias_min_value        = -30.0
response_init_mean    = 1.0
response_init_stdev   = 0.1
response_mutate_rate  = 0.1
response_replace_rate = 0.1
response_mutate_power = 0.1
response_max_value    = 30.0
response_min_value    = -30.0
weight_init_mean      = 0.0
weight_init_stdev     = 1.0
weight_mutate_rate    = 0.8
weight_replace_rate   = 0.1
weight_mutate_power   = 0.5
weight_max_value      = 30.0
weight_min_value      = -30.0
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
        print(f"[INFO] Configuration written to {self.config_path}")

    @staticmethod
    def _fitness_static(grid, N, block_size):
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
        return filled + rows**3 + cols**3 + b_ok**3 + bonus**3

    def run(self, puzzles):
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

        # Build pickle-friendly evaluator with sampling, rewards, and penalties
        eval_fn = partial(
            evaluate_genome,
            puzzles=puzzles,
            N=self.N,
            block_size=self.block_size,
            sample_size=self.sample_size,
            reward_valid=self.reward_valid,
            penalty_invalid=self.penalty_invalid
        )
        pe = ParallelEvaluator(self.workers, eval_fn)
        self.winner = pop.run(pe.evaluate, self.generations)
        self.config = config
        return self.winner

    def solve(self, puzzle, genome=None, verbose=True):
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


def load_puzzles(path, N):
    puzzles = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        for idx, line in enumerate(f, 1):
            s = line.strip()
            if len(s) != N*N or any(c not in '0123456789ABCDEF'[:N+1] for c in s):
                print(f"[WARNING] skipping invalid puzzle at line {idx}")
                continue
            arr = np.fromiter((int(c, 16) for c in s), dtype=int)
            puzzles.append(arr)
    return puzzles

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=4, help='Grid size N (e.g. 4, 9, 16)')
    parser.add_argument('--block', type=int, help='Block size (default sqrt(N)')
    parser.add_argument('--puzzles', default='puzzles.txt', help='Puzzle file path')
    parser.add_argument('--config', default='config-neat', help='Config path')
    parser.add_argument('--gens', type=int, default=100, help='Generations')
    parser.add_argument('--pop', type=int, default=150, help='Population size')
    parser.add_argument('--workers', type=int, help='Number of parallel workers (defaults to CPU count)')
    parser.add_argument('--sample-size', type=int, default=8, help='Number of puzzles to sample per genome evaluation')
    parser.add_argument('--reward-valid', type=float, default=1.0, help='Fitness reward for each valid move')
    parser.add_argument('--penalty-invalid', type=float, default=1.0, help='Fitness penalty for each invalid move')
    args = parser.parse_args()

    puzzles = load_puzzles(args.puzzles, args.size)
    solver = SudokuNEATSolver(
        args.size,
        args.block,
        args.config,
        args.gens,
        args.pop,
        args.workers,
        args.sample_size,
        args.reward_valid,
        args.penalty_invalid
    )
    best = solver.run(puzzles)
    print('=== Testing on loaded puzzles ===')
    for idx, puzz in enumerate(puzzles, 1):
        print(f'Puzzle {idx}:')
        solver.solve(puzz, best, verbose=True)
