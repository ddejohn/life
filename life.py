# Standard Library
from typing import Iterator, List

# Third party
import numpy as np

# Local
import exceptions
import animation

# Custom types
LifeGenerator = Iterator[np.ndarray]


class Life:
    def __init__(self, n: int = 50, bounds: str = "fixed"):
        self.n, self.bounds = exceptions.validate_args(n, bounds)
        self.generator = LifeFactory()

    def __str__(self):
        return f"{self.n}x{self.n}_{self.bounds}"

    def __repr__(self):
        return f"Life(n={self.n}, bounds={self.bounds})"

    def seed(self) -> np.ndarray:
        """Generates a uniformly distributed binary array"""
        return np.random.randint(0, 2, (self.n, self.n))

    def animate(self):
        animation.animate(self.generator(self), f"./examples/{self}")


class LifeFactory:
    MAX_GENERATIONS = 1000
    RULES = {(1, 2): 1, (1, 3): 1, (0, 3): 1}
    NEXT_STATE = np.vectorize(lambda x, y: LifeFactory.RULES.get((x, y), 0))

    @staticmethod
    def __call__(life: Life) -> LifeGenerator:
        g = LifeFactory.generator(life.seed(), n=life.n, bounds=life.bounds)
        return [*g]

    @staticmethod
    def generator(seed: np.ndarray, **kwds) -> LifeGenerator:
        """Returns a generator which yields until an exit condition is met"""
        state = seed
        generations = 0
        history = [state]
        while True:
            yield state
            generations += 1
            neighbors = LifeFactory.neighbors(state, **kwds)
            state = LifeFactory.NEXT_STATE(state, neighbors)
            history = history[-2:] + [state]  # only keep the last three states
            exit_code = LifeFactory.check_exit(history, generations)
            if exit_code == 2:  # oscillating state, period 2
                # yields 40 more frames for animation before generator stops
                yield from (h for _ in range(20) for h in history[-2:])
                break
            elif exit_code in (1, 3):
                break

    @staticmethod
    def neighbors(state: np.ndarray, **kwds) -> np.ndarray:
        """
        Counts the number of neighbors in a 3x3 neighborhood around
        each cell in the `state` array
        """
        padded = LifeFactory.pad(state, **kwds)
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
        return windows.sum(axis=(2, 3)) - state

    @staticmethod
    def pad(state: np.ndarray, *, n: int, bounds: str) -> np.ndarray:
        """
        Returns `state` array with a border which makes sliding window
        calculations trivial.

        If `bounds` is 'fixed', then `state` is padded with 0s, otherwise,
        `state` is tiled and sliced such that the right edge is adjacent
        to the left edge, the top edge is adjacent to the bottom edge,
        and catty-corners are adjacent.

        Period bounds example:

        Let `state` be
        ```
        A B C
        D E F
        G H I
        ```
        Then the periodic boundaries would be
        ```
        I G H I G
        C A B C A
        F D E F D
        I G H I G
        C A B C A
        ```
        """
        if bounds == "fixed":
            return np.pad(state, 1)
        return np.tile(state, (3, 3))[(s := n-1):-s, s:-s]

    @staticmethod
    def check_exit(history: List[np.ndarray], generations: int) -> int:
        """
        Checks for three exit conditions and returns a corresponding code

        Exit codes:
            `0`: OK
            `1`: A steady state has been reached
            `2`: An oscillating state of period 2 has been reached
            `3`: The maximum number of generations has been reached
        """
        exit_code = 0
        if np.array_equal(*history[-2:]):
            print("steady state")
            exit_code = 1
        if len(history) == 3:
            a, _, b = history[-3:]
            if np.array_equal(a, b):
                print("oscillating state period 2")
                exit_code = 2
        if generations == LifeFactory.MAX_GENERATIONS:
            print("reached max generations")
            exit_code = 3
        return exit_code
