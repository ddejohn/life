# Standard Library
import sys
from typing import Tuple
from ast import literal_eval

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
    # Overpopulation: a live cell with >= 4 live neighbors dies
    # Reproduction: a dead cell with 3 neighbors is reborn
    RULES = {(1, 2): 1,
             (1, 3): 1,
             (0, 3): 1}

    def __init__(self, dims: Tuple[int] = (50, 50), boundary: str = "fixed"):
        self.dims, self.boundary = exceptions.validate_args(dims, boundary)
        self.state = self.seed()
        self.generations = []
        self.get_new_state = np.vectorize(self.state_transitions)
        self.run()

    def reset(self):
        self.__init__(self.dims, self.boundary)

    def run(self):
        """Runs Life until either a steady state is reached, or max frames"""
        frames = 0
        while True:
            self.generations.append(self.state)
            frames += 1
            self.update()
            if frames > 3:
                a, _, b = self.generations[-3:]
                if np.array_equal(a, b):
                    self.generations.extend(self.generations[-2:]*20)
                    print("oscillating state period 2")
                    break
            if np.array_equal(self.state, self.generations[-1]):
                print("steady state")
                break
            if frames == 1000:
                print("reached max frames")
                break
        animation.make_animation(self.generations, self.make_filename())

    # TODO: refactor to pure functions
    def update(self):
        """Calculates a new state"""
        self.state = self.get_new_state(self.state, self.neighbors)

    # TODO: add different noise filters for different starting conditions
    def seed(self):
        """Sets an initial seed"""
        n = self.dims[0]
        s = np.random.binomial(n, 0.5, self.dims)
        return np.where(np.isin(s, range(n // 2 - int(n*0.05))), 1, 0)
        # return np.random.randint(0, 2, self.dims)

    def state_transitions(self, x: int, y: int) -> int:
        """Vectorized helper function for whether a cell should live or die"""
        return Life.RULES.get((x, y), 0)

    def make_filename(self):
        """Generates a filename for the resulting gif"""
        x, y = self.dims
        b = self.boundary
        g = len(self.generations)
        return f"./gifs/{x}x{y}_{b}_{g}_frames"

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
        return windows.sum(axis=(2, 3)) - self.state


if __name__ == "__main__":
    dims, boundary = sys.argv[1:]
    dims = literal_eval(dims)
    Life(dims, boundary)
