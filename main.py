from neat_sudoku_solver import SudokuNEATSolver, load_puzzles

N = 9
puzzles = load_puzzles('puzzles9x9.txt', N)
solver = SudokuNEATSolver(N, generations=30, pop_size=3)
winner = solver.run(puzzles)
grid, steps = solver.solve(puzzles[4], verbose=True)