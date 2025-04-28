from neat_sudoku_solver import SudokuNEATSolver, load_puzzles

N = 4
puzzles = load_puzzles('puzzles4x4.txt', N)
solver = SudokuNEATSolver(N, generations=100, pop_size=150)
winner = solver.run(puzzles)
grid, steps = solver.solve(puzzles[0], verbose=True)