# Standard Library
import collections
from typing import Iterator

# Third party
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as windows

# Local
import exceptions
import animator
import seeds

# Type aliases
LifeState = np.ndarray
LifeIterator = Iterator[LifeState]

# Constants
MAX_GENERATIONS = 1000


class Life:
    def __init__(self, size=100, bounds="fixed", seed_type="tiles"):
        exceptions.validate_args(size, bounds, seed_type)
        self.size = size
        self.bounds = bounds
        self.seed_type = seed_type
        self.pad_mode = "constant" if self.bounds == "fixed" else "wrap"
        self.seed = self.__seed_generator()
        self.state = self.__state_generator(next(self.seed))
        self.history = collections.deque(maxlen=3)
        self.generations = 0

    def animate(self) -> None:
        """Iterates `Life` until an exit condition is reached, then produces
        an animated gif with a filename auto-generated via `__str__()`.

        An `exit_code` of 2 means that either a steady state or an oscillating
        state of period two was reached in the course of iteration. In this
        case, the last two states are duplicated 15 extra times so that the
        animated gif doesn't `#gifsthatendtoosoon`."""
        frames = []
        try:
            while True:
                frames.append(next(self))
        except StopIteration as exit_code:
            if exit_code.value == 2:
                frames.extend(frames[-2:]*15)
        animator.make_animation(frames, f"./examples/{self}")
        self.reset()

    def reset(self) -> None:
        """Resets this object's seed and state generators"""
        self.seed = self.__seed_generator()
        self.state = self.__state_generator()
        self.history = collections.deque(maxlen=3)
        self.generations = 0

    def __str__(self):
        height, width = (self.size,)*2
        return f"{width}x{height}_{self.bounds}_{self.seed_type}"

    def __repr__(self):
        size, bounds, seed_type = self.size, self.bounds, self.seed_type
        return f"Life({size=}, {bounds=}, {seed_type=})"

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.state)

    def __seed_generator(self) -> seeds.LifeSeedGenerator:
        return seeds.new_seed_generator(self.size, self.seed_type)

    def __state_generator(self, state) -> LifeIterator:
        while True:
            yield state
            self.generations += 1
            self.history.append(state)
            if (exit_code := self.__check_exit()) != 1:
                return exit_code
            state = self.__update(state)

    def __update(self, state: LifeState) -> LifeState:
        """Calculates the next generation of `Life` given the current `state`
        and number of living neighbor `nghbrs`."""
        adj = windows(np.pad(state, 1, self.pad_mode), (3, 3)).sum(axis=(2, 3))
        nghbrs = adj - state
        return (nghbrs < 4) * (1 - state * (nghbrs % 2 - 1) + nghbrs) // 4

    def __check_exit(self) -> int:
        """
        Checks for three exit conditions and returns a corresponding code.

        Exit codes:
            `0`: The maximum number of generations has been reached
            `1`: OK to continue
            `2`: A steady state has been reached
            `2`: An oscillating state of period two has been reached
        """
        exit_code = 1
        if len(self.history) > 2:
            if np.array_equal(self.history[-2], self.history[-1]):
                print("steady state reached")
                exit_code = 2
            elif np.array_equal(self.history[-3], self.history[-1]):
                print("oscillating state period two reached")
                exit_code = 2
        if self.generations == MAX_GENERATIONS:
            print("reached maximum allowed generations")
            exit_code = 0
        return exit_code


if __name__ == "__main__":
    life = Life(100)
    [*life]
