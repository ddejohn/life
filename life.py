# Standard Library
from typing import Callable, Iterator, List, Tuple

# Third party
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as windows

# Local
import exceptions
import animation

# Custom types
ArrayShape = Tuple[int, int]
LifeGenerator = Iterator[np.ndarray]
StateArray = Tuple[np.ndarray, np.ndarray]

# Constants
MAX_GENERATIONS = 1000
RULES = {(1, 2): 1, (1, 3): 1, (0, 3): 1}
NEXT_STATE: Callable[[StateArray], np.ndarray]
NEXT_STATE = np.vectorize(lambda x, y: RULES.get((x, y), 0))


class Life:
    def __init__(self, shape: ArrayShape = (108, 192), bounds: str = "fixed"):
        self.shape, self.bounds = exceptions.validate_args(shape, bounds)
        self.generator = LifeGeneratorFactory()

    def __str__(self):
        height, width = self.shape
        return f"{10*width}x{10*height}_{self.bounds}"

    def __repr__(self):
        return f"Life(shape={self.shape}, bounds={self.bounds})"

    def seed(self) -> np.ndarray:
        """Generates a uniformly distributed binary array"""
        return np.random.randint(0, 2, self.shape, dtype="uint8")

    def animate(self):
        animation.make_animation(self.generator(self), f"./examples/{self}")


class LifeGeneratorFactory:
    @staticmethod
    def __call__(life: Life) -> LifeGenerator:
        mode = "constant" if life.bounds == "fixed" else "wrap"
        params = dict(seed=life.seed(), mode=mode)
        return [*LifeGeneratorFactory.generator(**params)]

    @staticmethod
    def generator(seed: np.ndarray, **kwds) -> LifeGenerator:
        """Returns a generator which yields until an exit condition is met"""
        state = seed
        generations = 0
        history = [state]
        while True:
            yield state
            generations += 1
            neighbors = LifeGeneratorFactory.neighbors(state, **kwds)
            state = NEXT_STATE(state, neighbors)
            history = [*history[-2:], state]  # only keep the last three states
            exit_code = LifeGeneratorFactory.check_exit(history, generations)
            if exit_code == 2:  # oscillating state, period 2
                # yields 40 more frames for animation before generator stops
                yield from (h for _ in range(20) for h in history[-2:])
                break
            elif exit_code in (1, 3):
                break

    @staticmethod
    def neighbors(x: np.ndarray, mode: str) -> np.ndarray:
        """Counts living neighbors around each cell in the `x` array"""
        return windows(np.pad(x, 1, mode=mode), (3, 3)).sum(axis=(2, 3)) - x

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
        if generations == MAX_GENERATIONS:
            print("reached max generations")
            exit_code = 3
        return exit_code
