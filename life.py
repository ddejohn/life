# Standard Library
from typing import Tuple

# Third party
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

# Local
import exceptions
import animation


class Life:
    # (self_value, neighbor_count) tuples, apply Life rules to neighbor windows
    # Underpopulation: a live cell with <= 1 live neighbors dies
    # Survival: a live cell with 2 or 3 neighbors survives
    # Overpopulation: a live cell with >4 live neighbors dies
    # Reproduction: a dead cell with 3 neighbors is reborn
    RULES = {(1, 0): 0,
             (1, 1): 0,
             (1, 2): 1,
             (1, 3): 1,
             (1, 4): 0,
             (0, 3): 1}

    def __init__(self, dims: Tuple[int] = (50, 50), boundary: str = "fixed"):
        self.dims, self.boundary = exceptions.validate_args(dims, boundary)
        self.state = self.initial_state()
        self.states = []
        self.get_new_state = np.vectorize(self.state_transitions)
        self.run()

    def reset(self):
        self.__init__(self.dims, self.boundary)
        self.run()

    def run(self):
        while True:
            self.states.append(self.state)
            self.update()
            if np.array_equal(self.state, self.states[-1]):
                print("steady state")
                break
            elif len(self.states) == 100:
                print("reached max frames")
                break
        animation.make_animation(self.states)

    def update(self):
        self.state = self.get_new_state(self.state, self.neighbors)

    def initial_state(self):
        n = self.dims[0]
        s = np.random.binomial(n, 0.5, self.dims)
        return np.where(np.isin(s, range(n // 2 - int(n*0.05))), 1, 0)
        # return np.random.randint(0, 2, self.dims)

    def state_transitions(self,
                          cells: np.ndarray,
                          neighbors: np.ndarray) -> np.ndarray:
        return Life.RULES.get((cells, neighbors), 0)

    @property
    def game_board(self) -> np.ndarray:
        """Returns state embedded in a 'frame' for sliding window calcs"""
        if self.boundary == "fixed":
            return np.pad(self.state, 1)

        n = len(self.state) - 1
        return np.tile(self.state, (3, 3))[n:-n, n:-n]

    @property
    def neighbors(self) -> np.ndarray:
        """Counts number of neighbors for each cell"""
        # Get number of neighboring live cells
        windows = sliding_window_view(self.game_board, (3, 3))

        # Clip to range [0, 4] to simplify rules calculation
        return np.clip(windows.sum(axis=(2, 3)) - 1, 0, 4)
