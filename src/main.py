from itertools import product
from concurrent.futures import ThreadPoolExecutor
import random

import numpy as np

from organism import Organism
from rnn_organism import RNNOrganism

N = 4
HIDDEN_LAYERS = [5, 5, 5]
POP_SIZE = 150
NUM_GENERATIONS = 300
SAMPLE_PUZZLES = 2

_BLOCK_SIZE = int(np.sqrt(N))
_CELL = N * N


def load_puzzles(path, N):
    puzzles = []
    with open(path) as f:
        for idx, line in enumerate(f, 1):
            s = line.strip()
            if len(s) != N*N or any(c not in '0123456789ABCDEF'[:N+1] for c in s):
                print(f"[WARNING] skipping invalid puzzle at line {idx}")
                continue
            arr = np.fromiter((int(c, 16) for c in s), dtype=int)
            puzzles.append(arr)
    return puzzles


def is_valid_move(grid, pick, digit):
    row, col = divmod(pick, N)
    # Check row
    row_vals = grid[row*N:(row+1)*N]
    if digit in row_vals:
        return False
    # Check column
    if digit in grid[col::N]:
        return False
    r0 = (row // _BLOCK_SIZE) * _BLOCK_SIZE
    c0 = (col // _BLOCK_SIZE) * _BLOCK_SIZE
    for rr in range(r0, r0 + _BLOCK_SIZE):
        for cc in range(c0, c0 + _BLOCK_SIZE):
            if grid[rr*N + cc] == digit:
                return False
    return True


def solve_puzzle(agent, puzzle):
    grid = puzzle.flatten()
    valid_moves = []
    while np.any(grid == 0):
        inputs = grid / N
        outputs = agent.predict(inputs.reshape(1, -1))[0].reshape(_CELL, N)
        empties = np.where(inputs == 0)[0]
        scores = outputs[empties].max(axis=1)
        pick = empties[np.argmax(scores)]
        digit = int(outputs[pick].argmax()) + 1
        valid_moves.append(is_valid_move(grid, pick, digit))
        grid[pick] = digit
    return grid.reshape(N, N), valid_moves


def create_agent():
    agent = Organism(
        dimensions=[_CELL] + HIDDEN_LAYERS + [_CELL * N],
        use_bias=True,
        output='sigmoid',
        mutation_std=0.05
    )
    return agent


def get_score(valid_moves):
    _valid_moves = np.array(valid_moves)
    scores = np.zeros(len(_valid_moves))
    scores[_valid_moves] = 1
    scores[~_valid_moves] = -10
    score = np.sum(scores)
    return int(score)


def create_new_population(agents, scores):
    fitness = (scores - np.min(scores))
    fitness = fitness**4
    fitness = fitness / np.sum(fitness)
    np.sort(fitness)
    new_population = []
    for _ in range(POP_SIZE):
        parent1, parent2 = np.random.choice(agents, size=2, p=fitness)
        child = parent1.mate(parent2)
        new_population.append(child)
    return new_population


if __name__ == "__main__":
    puzzles = load_puzzles("puzzles/puzzles4x4.txt", N)
    agents = [create_agent() for _ in range(POP_SIZE)]
    champion_score = -np.inf
    for generation in range(NUM_GENERATIONS):
        print(f"Generation {generation + 1}/{NUM_GENERATIONS}...", end=" ")
        sample_puzzles = random.sample(puzzles, SAMPLE_PUZZLES)
        agents_lst, puzzles_lst = list(zip(*product(agents, sample_puzzles)))
        with ThreadPoolExecutor() as e:
            solutions_lst = e.map(lambda agent, puzzle: solve_puzzle(agent, puzzle), agents_lst, puzzles_lst)
        _, valid_moves_lst = list(zip(*list(solutions_lst)))
        scores = np.array(list(map(lambda valid_moves: get_score(valid_moves), valid_moves_lst)))
        scores = scores.reshape(POP_SIZE, len(sample_puzzles))
        scores = scores.sum(axis=1)

        challenger = agents[np.argmax(scores)]
        with ThreadPoolExecutor() as e:
            solutions_lst = e.map(lambda puzzle: solve_puzzle(challenger, puzzle), puzzles)
        _, valid_moves_lst = list(zip(*list(solutions_lst)))
        challenger_scores = np.array(list(map(lambda valid_moves: get_score(valid_moves), valid_moves_lst)))
        challenger_score = challenger_scores.sum()
        if challenger_score > champion_score:
            champion_score = challenger_score
            champion = challenger
        print(f"Champion score: {champion_score}")

        if generation > 0:
            agents = create_new_population(agents, scores)


    puzzle = puzzles[1]
    solution, valid_moves = solve_puzzle(champion, puzzle)