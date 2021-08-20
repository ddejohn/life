# Standard Library
import random
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
    def __init__(self,
                 size: int = 100,
                 bounds: str = "fixed",
                 pattern_type: str = "tiles"):
        args = exceptions.validate_args(size=size,
                                        bounds=bounds,
                                        pattern_type=pattern_type)
        self.size, self.bounds, self.pattern_type = args
        self.seed_generator = LifeSeedGenerator(self.size, self.pattern_type)

    def __str__(self):
        height, width = (self.size,)*2
        return f"{width}x{height}_{self.bounds}_{self.pattern_type}"

    def __repr__(self):
        return f"Life(size={self.size}, bounds={self.bounds})"

    @property
    def seed(self) -> np.ndarray:
        """Generates a uniformly distributed binary array"""
        return self.seed_generator()

    def animate(self):
        state_generator = LifeGeneratorFactory()
        animation.make_animation(state_generator(self), f"./examples/{self}")


class LifeGeneratorFactory:
    @staticmethod
    def __call__(life: Life) -> LifeGenerator:
        mode = "constant" if life.bounds == "fixed" else "wrap"
        params = dict(seed=life.seed, mode=mode)
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
            if exit_code == 2:
                # yields 40 more frames for animation before generator stops
                yield from (h for _ in range(20) for h in history[-2:])
                break
            elif exit_code == 0:
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
            `1`: OK to continue
            `2`: A steady state has been reached
            `2`: An oscillating state of period 2 has been reached
            `0`: The maximum number of generations has been reached
        """
        exit_code = 1
        if np.array_equal(*history[-2:]):
            print("reached steady state")
            exit_code = 2
        if len(history) == 3:
            a, _, b = history[-3:]
            if np.array_equal(a, b):
                print("oscillating state period 2")
                exit_code = 2
        if generations == MAX_GENERATIONS:
            print("reached max generations")
            exit_code = 0
        return exit_code


class SeedArray:
    @staticmethod
    def __call__(shape: ArrayShape, pattern_type: str) -> np.ndarray:
        if pattern_type == "tiles":
            array_creator = random.choice((SeedArray.diagonal,
                                           SeedArray.inverted_diagonal,
                                           SeedArray.quilt))
            return array_creator(shape)
        return SeedArray.binary_array(shape)

    @staticmethod
    def binary_array(shape: tuple) -> np.ndarray:
        return np.random.randint(0, 2, shape, dtype="uint8")

    @staticmethod
    def diagonal(shape: tuple) -> np.ndarray:
        # x, y = shape
        tri = np.triu(SeedArray.binary_array(shape))
        return np.clip(tri + tri.T, 0, 1)

    @staticmethod
    def inverted_diagonal(shape: tuple) -> np.ndarray:
        # x, y = shape
        tri = np.triu(SeedArray.binary_array(shape))
        return np.triu(np.where(tri, 0, 1)).T + tri

    @staticmethod
    def quilt(shape: tuple) -> np.ndarray:
        array = np.triu(SeedArray.binary_array(shape))
        rotator = random.choice((np.fliplr, np.flipud, np.rot90, None))
        if not rotator:
            return array
        return rotator(array)


class TileMethod:
    @staticmethod
    def __call__(array: np.ndarray, pattern_number: int) -> np.ndarray:
        tiling_method = random.choice((TileMethod.four_corners,
                                       TileMethod.book_match,
                                       TileMethod.hamburger,
                                       TileMethod.repeat))
        return np.tile(tiling_method(array), (pattern_number,)*2)

    @staticmethod
    def four_corners(NW: np.ndarray) -> np.ndarray:
        NE = np.fliplr(NW)
        SW = np.flipud(NW)
        SE = np.flipud(NE)
        return np.block([[NW, NE], [SW, SE]])

    @staticmethod
    def book_match(L: np.ndarray) -> np.ndarray:
        R = np.fliplr(L)
        return np.block([[L, R], [L, R]])

    @staticmethod
    def hamburger(T: np.ndarray) -> np.ndarray:
        B = np.flipud(T)
        return np.block([[T, T], [B, B]])

    @staticmethod
    def repeat(x: np.ndarray) -> np.ndarray:
        return np.tile(x, (2, 2))


class LifeSeedGenerator:
    def __init__(self, n: int, pattern_type: str):
        self.seed = self.generator(n, pattern_type)

    def __call__(self) -> np.ndarray:
        return next(self.seed)

    def generator(self, n: int, pattern_type: str) -> Iterator[np.ndarray]:
        while True:
            pattern_number, initial_size = self.size_and_pattern(n)
            shape = (initial_size,)*2
            array_creator = SeedArray()
            tiling_method = TileMethod()
            array = array_creator(shape, pattern_type)
            yield tiling_method(array, pattern_number)

    @staticmethod
    def size_and_pattern(n: int) -> ArrayShape:
        k = n // 2
        d = [(x, k//x) for x in range(2, k + 1) if k % x == 0]
        return sorted(random.choice(d))
