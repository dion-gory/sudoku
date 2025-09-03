from itertools import product
from concurrent.futures import ThreadPoolExecutor
import random
from copy import deepcopy
import time

import numpy as np

from organism import Organism

# Comment out if not using RNN variant
# from rnn_organism import RNNOrganism

N = 4
HIDDEN_LAYERS = [32, 32]  # Wider layers
POP_SIZE = 100  # Increased population size
NUM_GENERATIONS = 50
SAMPLE_PUZZLES = 15  # Increased sample puzzles for better evaluation
MUTATION_RATE = 0.1  # Explicit mutation rate
ELITE_RATIO = 0.1  # Keep top 10% elite organisms

_BLOCK_SIZE = int(np.sqrt(N))
_CELL = N * N


def load_puzzles(path, N):
    puzzles = []
    with open(path) as f:
        for idx, line in enumerate(f, 1):
            s = line.strip()
            if len(s) != N * N or any(c not in '0123456789ABCDEF'[:N + 1] for c in s):
                print(f"[WARNING] skipping invalid puzzle at line {idx}")
                continue
            arr = np.fromiter((int(c, 16) for c in s), dtype=int)
            puzzles.append(arr)
    return puzzles


def is_valid_move(grid, pick, digit):
    row, col = divmod(pick, N)

    # Check row
    row_vals = grid[row * N:(row + 1) * N]
    if digit in row_vals:
        return False

    # Check column
    if digit in grid[col::N]:
        return False

    # Check block
    r0 = (row // _BLOCK_SIZE) * _BLOCK_SIZE
    c0 = (col // _BLOCK_SIZE) * _BLOCK_SIZE
    for rr in range(r0, r0 + _BLOCK_SIZE):
        for cc in range(c0, c0 + _BLOCK_SIZE):
            if grid[rr * N + cc] == digit:
                return False
    return True


def solve_puzzle(agent, puzzle):
    grid = puzzle.copy()
    puzzle_len = np.sum(grid == 0)
    valid_moves = []

    # Keep track of filled positions to avoid infinite loops
    filled_positions = set()
    max_attempts = puzzle_len * 2  # Set maximum attempts
    attempts = 0

    while np.any(grid == 0) and attempts < max_attempts:
        inputs = grid / N  # Normalize input

        # Get network predictions
        outputs = agent.predict(inputs.reshape(1, -1))[0].reshape(_CELL, N)

        # Find empty cells
        empties = np.where(inputs == 0)[0]
        if len(empties) == 0:
            break

        # Add softmax temperature for more deterministic choices
        scores = outputs[empties].max(axis=1)

        # Choose the empty cell with highest confidence
        pick = empties[np.argmax(scores)]

        # If we've already filled this position in this solve attempt,
        # choose another to avoid loops
        attempts_this_pos = 0
        while pick in filled_positions and attempts_this_pos < len(empties):
            scores[np.argmax(scores)] = -np.inf  # Mark as seen
            if np.all(scores == -np.inf):
                break
            pick = empties[np.argmax(scores)]
            attempts_this_pos += 1

        # Choose digit with highest score for that cell
        digit_scores = outputs[pick]
        digit = int(np.argmax(digit_scores)) + 1

        # Check if move is valid
        valid_move = is_valid_move(grid, pick, digit)
        valid_moves.append(valid_move)

        # Make the move regardless of validity - network needs to learn
        grid[pick] = digit
        filled_positions.add(pick)
        attempts += 1

    return grid.reshape(N, N), valid_moves, puzzle_len


def create_agent():
    agent = Organism(
        dimensions=[_CELL] + HIDDEN_LAYERS + [_CELL * N],
        use_bias=True,
        output='sigmoid',  # sigmoid allows more nuanced output representation
        mutation_std=MUTATION_RATE
    )
    return agent


def get_score(solution, original_puzzle, valid_moves):
    """Enhanced scoring function with more nuanced rewards"""
    solution_flat = solution.flatten()
    filled = np.count_nonzero(solution_flat)

    # Calculate proportion of valid moves (important learning signal)
    valid_ratio = sum(valid_moves) / len(valid_moves) if valid_moves else 0

    # Check the rules of Sudoku
    valid = lambda line: line.size == np.unique(line).size

    # Check rows, columns, and blocks
    rows = sum(valid(row[row > 0]) for row in solution)
    cols = sum(valid(col[col > 0]) for col in solution.T)

    blocks = (
        solution[r:r + _BLOCK_SIZE, c:c + _BLOCK_SIZE].ravel()
        for r in range(0, N, _BLOCK_SIZE)
        for c in range(0, N, _BLOCK_SIZE)
    )
    b_ok = sum(valid(b[b > 0]) for b in blocks)

    # Calculate validity percentage
    total_checks = 3 * N  # rows + cols + blocks
    valid_percent = (rows + cols + b_ok) / total_checks

    # Perfect solution bonus
    perfect_bonus = N * 5 if filled == N * N and rows == N and cols == N and b_ok == N else 0

    # Final score combines multiple factors with different weights
    score = (
            filled * 0.5 +  # Reward filling cells
            rows ** 2 * 1.0 +  # Strong reward for valid rows
            cols ** 2 * 1.0 +  # Strong reward for valid columns
            b_ok ** 2 * 1.0 +  # Strong reward for valid blocks
            valid_ratio * N ** 2 * 2.0 +  # Very strong reward for making valid moves
            valid_percent * N ** 2 * 3.0 +  # Very strong reward for overall validity
            perfect_bonus * 10  # Massive bonus for perfect solutions
    )

    return score


def tournament_selection(agents, scores, k=3):
    """Tournament selection for more diverse parent selection"""
    idx = np.random.randint(0, len(agents), k)
    selected_idx = idx[np.argmax(scores[idx])]
    return agents[selected_idx]


def create_new_population(agents, scores):
    # Normalize scores to positive values
    scores = scores - np.min(scores) + 1e-10

    # Calculate fitness with power scaling for emphasizing good solutions
    fitness = (scores / np.max(scores)) ** 1.5  # Increased selection pressure
    fitness = fitness / np.sum(fitness)

    # Elitism: keep best performers
    elite_count = int(POP_SIZE * ELITE_RATIO)
    elite_indices = np.argsort(scores)[-elite_count:]
    elite_agents = [deepcopy(agents[i]) for i in elite_indices]

    # Create rest of the population through tournament selection and crossover
    new_population = elite_agents.copy()

    for _ in range(POP_SIZE - elite_count):
        # Tournament selection instead of fitness proportionate
        parent1 = tournament_selection(agents, scores)
        parent2 = tournament_selection(agents, scores)

        # Create child with controlled mutation
        child = parent1.mate(parent2, mutate=True)
        new_population.append(child)

    return new_population


def visualize_solution(solution, original):
    """Helper function to visualize a solution for debugging"""
    original_grid = original.reshape(N, N)
    solution_grid = solution.copy()

    print("Original puzzle:")
    for i in range(N):
        print(" ".join(str(int(x)) if x > 0 else "." for x in original_grid[i]))

    print("\nSolution:")
    for i in range(N):
        print(" ".join(str(int(x)) for x in solution_grid[i]))


if __name__ == "__main__":
    start_time = time.time()
    all_puzzles = load_puzzles("puzzles/puzzles4x4.txt", N)
    puzzles = random.sample(all_puzzles, 100)  # Use more puzzles for training

    # Create initial population
    agents = [create_agent() for _ in range(POP_SIZE)]
    champion_score = -np.inf
    champion = None

    # Variables to track progress
    progress = []
    best_scores = []

    # Main training loop
    for generation in range(NUM_GENERATIONS):
        gen_start = time.time()
        print(f"Generation {generation + 1}/{NUM_GENERATIONS}...", end=" ")

        # Sample different puzzles each generation for more robust training
        sample_puzzles = random.sample(puzzles, SAMPLE_PUZZLES)
        agents_lst, puzzles_lst = list(zip(*product(agents, sample_puzzles)))

        # Solve puzzles in parallel
        with ThreadPoolExecutor() as e:
            solutions_lst = list(e.map(lambda pair: solve_puzzle(pair[0], pair[1]),
                                       list(zip(agents_lst, puzzles_lst))))

        # Process solutions
        grids = []
        all_valid_moves = []
        puzzle_lengths = []

        for sol in solutions_lst:
            grid, valid_moves, puzzle_len = sol
            grids.append(grid)
            all_valid_moves.append(valid_moves)
            puzzle_lengths.append(puzzle_len)

        # Calculate scores with the original puzzles for context
        scores = []
        for i, grid in enumerate(grids):
            puzzle_idx = i % SAMPLE_PUZZLES
            original_puzzle = sample_puzzles[puzzle_idx].reshape(N, N)
            valid_moves = all_valid_moves[i]
            score = get_score(grid, original_puzzle, valid_moves)
            scores.append(score)

        scores = np.array(scores).reshape(POP_SIZE, SAMPLE_PUZZLES)
        agent_avg_scores = scores.sum(axis=1)

        # Identify current generation champion
        gen_best_idx = np.argmax(agent_avg_scores)
        gen_best_agent = agents[gen_best_idx]
        gen_best_score = agent_avg_scores[gen_best_idx]

        # Track the best scores
        best_scores.append(gen_best_score)

        # Evaluate the current generation's best agent on more puzzles
        challenger = gen_best_agent

        challenger_solutions = []
        for puzzle in puzzles:
            solution, valid_moves, puzzle_len = solve_puzzle(challenger, puzzle)
            challenger_solutions.append((solution, puzzle.reshape(N, N), valid_moves))

        challenger_scores = []
        for solution, puzzle, valid_moves in challenger_solutions:
            score = get_score(solution, puzzle, valid_moves)
            challenger_scores.append(score)

        challenger_score = np.sum(challenger_scores)

        # Update champion if we found a better one
        if challenger_score > champion_score:
            champion_score = challenger_score
            champion = deepcopy(challenger)
            print(f"New champion! Score: {champion_score:.2f}")

            # # Visualize first solution from new champion for debugging
            # if champion_solutions:
            #     solution = challenger_solutions[0][0]
            #     puzzle = challenger_solutions[0][1]
            #     visualize_solution(solution, puzzle)
        else:
            print(f"Best score: {gen_best_score:.2f}, Champion score: {champion_score:.2f}")

        # Create new population through selection and breeding
        if generation < NUM_GENERATIONS - 1:  # Skip last generation's evolution
            agents = create_new_population(agents, agent_avg_scores)

        # Track progress
        avg_score = np.mean(agent_avg_scores)
        valid_move_ratios = [sum(vm) / len(vm) if vm else 0 for vm in all_valid_moves]
        avg_valid_ratio = np.mean(valid_move_ratios)
        progress.append((generation, avg_score, gen_best_score, champion_score, avg_valid_ratio))

        gen_time = time.time() - gen_start
        print(f"Gen time: {gen_time:.2f}s, Valid moves: {avg_valid_ratio:.2%}")

    # Final evaluation
    print("\nFinal Evaluation:")
    num_solved = 0
    test_puzzles = random.sample(all_puzzles, 50)
    for puzzle in test_puzzles:
        solution, valid_moves, puzzle_len = solve_puzzle(champion, puzzle)
        all_valid = sum(valid_moves) == len(valid_moves)
        if all_valid and not np.any(solution == 0):
            num_solved += 1

    print(f"Solved puzzles: {num_solved} out of {len(test_puzzles)} ({num_solved / len(test_puzzles):.1%})")
    print(f"Total runtime: {(time.time() - start_time) / 60:.2f} minutes")

    # Display progress statistics
    print("\nTraining Progress:")
    for i in range(0, len(progress), max(1, len(progress) // 10)):
        gen, avg, best, champ, valid = progress[i]
        print(f"Gen {gen}: Avg={avg:.2f}, Best={best:.2f}, Champion={champ:.2f}, Valid={valid:.2%}")