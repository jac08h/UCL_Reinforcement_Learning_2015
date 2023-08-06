from copy import deepcopy
from math import inf
from typing import List, Tuple

import numpy as np

TERMINAL_STATE = 1
STEP_PENALTY = -1

MOVE2COORD = {
    "R": (0, 1),
    "L": (0, -1),
    "D": (1, 0),
    "U": (-1, 0)
}
COORD2MOVE = {value: key for key, value in MOVE2COORD.items()}


class CellSingleMovePolicy:
    def __init__(self, name: str, probability: float):
        self.name = name
        self.probability = probability

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


Policy = List[List[List[CellSingleMovePolicy]]]


def initialize_random_policy(grid: np.array) -> Policy:
    width, height = grid.shape
    policy = []
    for row_i in range(height):
        row = []
        for col_i in range(width):
            moves = []
            if grid[row_i][col_i] != TERMINAL_STATE:
                for name, coord_change in MOVE2COORD.items():
                    moves.append(CellSingleMovePolicy(name, 0.25))
            row.append(moves)
        policy.append(row)
    return policy


def initialize_empty_policy(grid: np.array) -> Policy:
    width, height = grid.shape
    policy = []
    for row_i in range(height):
        row = []
        for col_i in range(width):
            moves = []
            row.append(moves)
        policy.append(row)
    return policy


def same_policies(policy_a: Policy, policy_b: Policy) -> bool:
    if len(policy_a) != len(policy_b):
        return False

    for row_a, row_b in zip(policy_a, policy_b):
        if len(row_a) != len(row_b):
            return False

        for cell_a, cell_b in zip(row_a, row_b):
            if len(cell_a) != len(cell_b):
                return False

            for policy_a, policy_b in zip(cell_a, cell_b):
                if policy_a.name != policy_b.name or policy_a.probability != policy_b.probability:
                    return False

    return True


def policy_from_value_function(grid: np.array, value_function: np.array) -> Policy:
    new_policy = initialize_empty_policy(grid)
    num_rows, num_cols = grid.shape
    for row in range(num_rows):
        for col in range(num_cols):
            if grid[row][col] == TERMINAL_STATE:
                continue

            best_moves = []
            best_value = None
            for move_name, move in MOVE2COORD.items():
                n_row, n_col = row + move[0], col + move[1]
                if not on_grid(n_row, n_col, num_rows, num_cols):
                    n_row, n_col = row, col

                value = value_function[n_row][n_col]
                if best_value is None:
                    best_moves.append(move_name)
                    best_value = value_function[n_row][n_col]

                elif value == best_value:
                    best_moves.append(move_name)

                elif value > best_value:
                    best_moves = [move_name]
                    best_value = value

            for best_move in best_moves:
                new_policy[row][col].append(CellSingleMovePolicy(best_move, 1 / len(best_moves)))

    return new_policy


def create_grid_world(width: int,
                      height: int,
                      terminal_squares: List[Tuple[int, int]]
                      ) -> np.array:
    grid_world = np.zeros((height, width), dtype=int)

    for x, y in terminal_squares:
        if 0 <= x < width and 0 <= y < height:
            grid_world[y, x] = TERMINAL_STATE

    return grid_world


def on_grid(row: int, col: int, n_rows: int, n_cols: int) -> bool:
    return 0 <= row < n_rows and 0 <= col < n_cols


def policy_evaluation(grid: np.array, value_function: np.array, policy: Policy) -> np.array:
    num_rows, num_cols = grid.shape
    while True:
        new_value_function = np.copy(value_function)
        for row in range(num_rows):
            for col in range(num_cols):
                if grid[row][col] == TERMINAL_STATE:
                    continue

                cell_value = 0
                for move in policy[row][col]:
                    n_row, n_col = row + MOVE2COORD[move.name][0], col + MOVE2COORD[move.name][1]
                    if not on_grid(n_row, n_col, num_rows, num_cols):
                        n_row, n_col = row, col
                    cell_value += move.probability * (STEP_PENALTY + value_function[n_row][n_col])
                new_value_function[row][col] = cell_value

        if np.allclose(value_function, new_value_function):
            return value_function
        value_function = new_value_function


def policy_iteration(grid: np.array,
                     initial_value_function: np.array,
                     initial_policy: Policy) -> Policy:
    policy = initial_policy
    value_function = initial_value_function
    i = 0
    while True:
        old_policy = deepcopy(policy)
        value_function = policy_evaluation(grid, value_function, policy)
        policy = policy_from_value_function(grid, value_function)
        if same_policies(old_policy, policy):
            return policy

        i += 1


def value_iteration(grid: np.array, initial_value_function: np.array) -> np.array:
    num_rows, num_cols = grid.shape
    value_function = initial_value_function
    while True:
        new_value_function = np.copy(value_function)
        for row in range(num_rows):
            for col in range(num_cols):
                if grid[row][col] == TERMINAL_STATE:
                    continue

                cell_value_on_best_action = -inf
                for move_name, move in MOVE2COORD.items():
                    n_row, n_col = row + move[0], col + move[1]
                    if not on_grid(n_row, n_col, num_rows, num_cols):
                        n_row, n_col = row, col
                    cell_value_on_best_action = max(STEP_PENALTY + value_function[n_row][n_col],
                                                    cell_value_on_best_action)
                new_value_function[row][col] = cell_value_on_best_action

        if np.allclose(value_function, new_value_function):
            return value_function
        value_function = new_value_function


if __name__ == '__main__':
    width = 3
    height = 3
    terminal_squares = [(0, 0), (3, 3), [5, 5]]
    grid_world = create_grid_world(width, height, terminal_squares)

    policy_from_policy_iteration = policy_iteration(grid=grid_world,
                                                    initial_value_function=np.zeros_like(grid_world, dtype=float),
                                                    initial_policy=initialize_random_policy(grid_world),
                                                    )
    value_f = value_iteration(grid_world, initial_value_function=np.zeros_like(grid_world, dtype=float))
    policy_from_value_iteration = policy_from_value_function(grid_world, value_f)

    assert same_policies(policy_from_policy_iteration, policy_from_value_iteration)
