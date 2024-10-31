import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
def load_data(file_path):
    df = pd.read_csv(file_path, header=None, names=['x1', 'x2', 'opt_type', 'value'])
    obj_coeffs = df.iloc[0][['x1', 'x2']].astype(float).values
    maximize = df.iloc[0]['opt_type'].strip().lower() == 'max'
    constraints = df.iloc[1:][['x1', 'x2']].astype(float).values
    bounds = df.iloc[1:]['value'].astype(float).values
    return obj_coeffs, constraints, bounds, maximize

def solve_with_ortools(obj_coeffs, constraints, bounds, maximize):
    solver = pywraplp.Solver.CreateSolver("GLOP")
    num_vars = len(obj_coeffs)
    variables = [solver.NumVar(0, solver.infinity(), f'var_{i+1}') for i in range(num_vars)]
    objective = solver.Objective()
    for i, coeff in enumerate(obj_coeffs):
        objective.SetCoefficient(variables[i], coeff)
    objective.SetMaximization() if maximize else objective.SetMinimization()
    for i in range(len(constraints)):
        constraint = solver.RowConstraint(-solver.infinity(), bounds[i], f'constraint_{i+1}')
        for j in range(num_vars):
            constraint.SetCoefficient(variables[j], constraints[i][j])
    solver.Solve()
    return [var.solution_value() for var in variables]

def simplex_method(c, A, b):
    num_vars = len(c)
    num_constraints = len(b)
    tableau = np.zeros((num_constraints + 1, num_vars + num_constraints + 1))
    tableau[:-1, :num_vars] = A
    tableau[:-1, num_vars:num_vars + num_constraints] = np.eye(num_constraints)
    tableau[:-1, -1] = b
    tableau[-1, :num_vars] = -c
    basic_vars = [f'S{i+1}' for i in range(num_constraints)]

    def print_tableau(t):
        df = pd.DataFrame(t, columns=[*basic_vars, *[f'x{i+1}' for i in range(num_vars)], 'RHS'])
        print("Tableau:")
        print(df)

    while True:
        print_tableau(tableau)
        if all(i >= 0 for i in tableau[-1, :-1]):
            break
        pivot_col = np.argmin(tableau[-1, :-1])
        pivot_column = tableau[:-1, pivot_col]
        if np.all(pivot_column <= 0):
            print("The solution is unbounded.")
            return None
        with np.errstate(divide='ignore', invalid='ignore'):
            positive_ratios = np.where(pivot_column > 0, tableau[:-1, -1] / pivot_column, np.inf)
        pivot_row = np.argmin(positive_ratios)
        basic_vars[pivot_row] = f'var_{pivot_col + 1}'
        pivot_value = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_value
        for r in range(tableau.shape[0]):
            if r != pivot_row:
                tableau[r, :] -= tableau[r, pivot_col] * tableau[pivot_row, :]

    print_tableau(tableau)
    return tableau[:-1, -1]

def main(file_path):
    obj_coeffs, constraints, bounds, maximize = load_data(file_path)
    print("Using Google OR-Tools:")
    ortools_solution = solve_with_ortools(obj_coeffs, constraints, bounds, maximize)
    print(ortools_solution)
    print("\nUsing Simplex Method:")
    simplex_solution = simplex_method(np.array(obj_coeffs), np.array(constraints), np.array(bounds))
    print("Optimal solution from Simplex Method:", simplex_solution)

file_path = 'examplebook.csv'
main(file_path)
