
# Standard Library
from typing import Tuple

# Third party
import numpy as np

# Local
import exceptions


class Life:
    # (self_value, neighbor_count) tuples, apply Life rules to neighbor windows
    # Underpopulation: a live cell with <= 1 live neighbors dies
    # Survival: a live cell with 2 or 3 neighbors survives
    # Overpopulation: a live cell with >4 live neighbors dies
    # Reproduction: a dead cell with 3 neighbors is born
    RULES = {(1, 0): 0,
             (1, 1): 0,
             (1, 2): 1,
             (1, 3): 1,
             (1, 4): 0,
             (0, 3): 1}

    def __init__(self, size: Tuple[int], boundary: str = "fixed") -> None:
        self.size, self.boundary = self.validate_args(size, boundary)
        self.state = np.random.randint(0, 2, self.size)
        self.states = []

    def validate_args(self, size, boundary):
        if type(size) != tuple:
            raise exceptions.SizeError(type(size).__name__)
        if not all(dim > 1 for dim in size):
            raise exceptions.SizeError(f"{size}")
        if not all((types := tuple(type(d) == int for d in size))):
            raise exceptions.SizeError(f"{[t.__name__ for t in types]}")
        if type(boundary) != str:
            raise exceptions.BoundaryError(type(boundary).__name__)
        if boundary not in ("fixed", "periodic"):
            raise exceptions.BoundaryError(boundary)
        return size, boundary

    def determine_state(self,
                        cells: np.ndarray,
                        neighbors: np.ndarray) -> np.ndarray:
        try:
            return Life.RULES[(cells, neighbors)]
        except KeyError:
            print("UH OH")

    def update(self):
        self.states.append(self.state)
        neighbors = self.get_neighbors()
        get_new_state = np.vectorize(self.determine_state)
        self.state = get_new_state(self.state, neighbors)

    # make @property
    def get_game_board(self) -> np.ndarray:
        # fixed boundaries
        if self.boundary == "fixed":
            return np.pad(self.state, 1)

        # periodic boundaries
        # get edges
        top_edge = self.state[0, :]
        bottom_edge = self.state[-1, :]
        left_edge = self.state[:, 0]
        right_edge = self.state[:, -1]

        # get corners
        top_left = self.state[0, 0]
        bottom_left = self.state[-1, 0]
        top_right = self.state[0, -1]
        bottom_right = self.state[-1, -1]

        # pad edges
        left_edge = np.array([bottom_right, *right_edge, top_right]).reshape(-1, 1)
        right_edge = np.array([bottom_left, *left_edge, top_left]).reshape(-1, 1)

        # vertical stack step
        game_board = np.block([[bottom_edge], [self.state], [top_edge]])

        # horizontal stack step
        return np.block([left_edge, game_board, right_edge])

    def get_neighbors(self):
        # Get number of neighboring live cells (subtracting self value)
        windows = np.lib.stride_tricks.sliding_window_view(self.state, (3, 3))

        # Clip to range [0, 4] to simplify rules
        return np.clip(windows.sum(axis=(2, 3)) - 1, 0, 4)

    def start(self):
        # game loop:
        # save "previous" state
        # increment life and update state
        # compare state to previous state
        # exit if no change, continue if change

        # # iteration:
        #
        # # Get 3x3 neighbor subarrays
        # # windows = np.lib.stride_tricks.sliding_window_view(game_board, (3,3))
        #
        # # Get number of neighboring live cells (subtracting self value)
        # # Clip to range [0, 4] to simplify rules
        # neighbor_counts = np.clip(windows.sum(axis=(2,3)) - 1, 0, 4)
        pass
