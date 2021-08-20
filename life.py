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
Generations = List[np.ndarray]
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
        self.bounds = "constant" if self.bounds == "fixed" else "wrap"
        self.seed_generator = SeedGenerator(self)
        self.state_generator = StateGenerator(self)

    def __str__(self):
        height, width = (self.size,)*2
        return f"{width}x{height}_{self.bounds}_{self.pattern_type}"

    def __repr__(self):
        return f"Life(size={self.size}, bounds={self.bounds})"

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.state_generator)

    def animate(self):
        animation.make_animation([*self], f"./examples/{self}")


class StateGenerator:
    """A generator class which yields the next state of a `Life` instance"""
    def __init__(self, life: Life):
        self.state_generator = self.generator(life)

    def __iter__(self):
        return self

    def __next__(self):
        return next(self.state_generator)

    @staticmethod
    def generator(life: Life) -> Iterator[np.ndarray]:
        state = next(life.seed_generator)
        generations = 0
        history = [state]
        while True:
            yield state
            generations += 1
            neighbors = StateGenerator.neighbors(state, life.bounds)
            state = NEXT_STATE(state, neighbors)
            history = [*history[-2:], state]  # only keep the last three states
            exit_code = StateGenerator.check_exit(history, generations)
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
    """Static methods for generating random binary tiles"""
    @staticmethod
    def __call__(shape: ArrayShape, pattern_type: str) -> np.ndarray:
        rotator = random.choice((np.fliplr, np.flipud, np.rot90, None))
        if pattern_type == "tiles":
            array_creator = random.choice((SeedArray.diagonal,
                                           SeedArray.inverted_diagonal,
                                           SeedArray.quilt))
            array = array_creator(shape)
        else:
            array = SeedArray.binary_array(shape)

        if rotator:
            return rotator(array)
        return array

    @staticmethod
    def binary_array(shape: tuple) -> np.ndarray:
        """Binary noise array, the base working unit for the other methods"""
        return np.random.randint(0, 2, shape, dtype="uint8")

    @staticmethod
    def diagonal(shape: tuple) -> np.ndarray:
        """Diagonally symmetric array"""
        tri = np.triu(SeedArray.binary_array(shape))
        return np.clip(tri + tri.T, 0, 1)

    @staticmethod
    def inverted_diagonal(shape: tuple) -> np.ndarray:
        """Diagonally symmetric array with flipped bits in lower triangular"""
        tri = np.triu(SeedArray.binary_array(shape))
        return np.triu(np.where(tri, 0, 1)).T + tri

    @staticmethod
    def quilt(shape: tuple) -> np.ndarray:
        """A triangular array"""
        return np.triu(SeedArray.binary_array(shape))


class TileMethod:
    """Static methods for tiling an array with different symmetries"""
    @staticmethod
    def __call__(array: np.ndarray, pattern_number: int) -> np.ndarray:
        tiling_method = random.choice((TileMethod.four_corners,
                                       TileMethod.book_match,
                                       TileMethod.hamburger,
                                       TileMethod.repeat))
        return np.tile(tiling_method(array), (pattern_number,)*2)

    @staticmethod
    def four_corners(NW: np.ndarray) -> np.ndarray:
        """Radial symmetry"""
        NE = np.fliplr(NW)
        SW = np.flipud(NW)
        SE = np.flipud(NE)
        return np.block([[NW, NE], [SW, SE]])

    @staticmethod
    def book_match(L: np.ndarray) -> np.ndarray:
        """Vertical symmetry"""
        R = np.fliplr(L)
        return np.block([[L, R], [L, R]])

    @staticmethod
    def hamburger(T: np.ndarray) -> np.ndarray:
        """Horizontal symmetry"""
        B = np.flipud(T)
        return np.block([[T, T], [B, B]])

    @staticmethod
    def repeat(x: np.ndarray) -> np.ndarray:
        """Repeating tiles"""
        return np.tile(x, (2, 2))


class SeedGenerator:
    """
    A generator class which yields "Life" seeds.

    Taking `n` as the desired size of the final array, this generator
    first breaks down `n` by half into `k`, and then `k` into a tuple
    randomly chosen from the matched divisors of `k` such that the
    product of the tuple is `k`.

    This tuple gives a pattern number and initial size which are passed
    to the seed array creator and tiling method functions.

    The larger of the two divisors of `k` that were chosen is assigned
    to `tile_size` while the smaller is assigned to `num_tiles`.

    The `tile_size` is used as the shape for the base binary tile unit,
    and after a tiling method is chosen, the pattern number determines
    how many times the binary tile unit should be repeated.

    After the dust settles, the final seed is of the correct shape.
    """
    def __init__(self, life: Life):
        self.seed_generator = self.generator(life)

    def __iter__(self):
        return self

    def __next__(self) -> np.ndarray:
        return next(self.seed_generator)

    def generator(self, life: Life) -> Iterator[np.ndarray]:
        while True:
            array_creator = SeedArray()
            tiling_method = TileMethod()
            num_tiles, tile_size = self.size_and_pattern(life.size)
            array = array_creator((tile_size,)*2, life.pattern_type)
            yield tiling_method(array, num_tiles)

    @staticmethod
    def size_and_pattern(n: int) -> ArrayShape:
        k = n // 2
        d = [(x, k // x) for x in range(2, k + 1) if k % x == 0]
        return sorted(random.choice(d))
