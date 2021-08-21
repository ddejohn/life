# Standard Library
import random
import collections
from typing import Iterator, List, Tuple

# Third party
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as windows

# Local
import exceptions
import animation

# Type aliases
LifeArray = np.ndarray
ArrayShape = Tuple[int, int]
Generations = List[LifeArray]

# Constants
MAX_GENERATIONS = 1000


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
    def generator(life: Life) -> Iterator[LifeArray]:
        num_generations = 0
        state = next(life.seed_generator)
        history = collections.deque(maxlen=3)  # only keeps last three states
        while True:
            yield state
            num_generations += 1
            history.append(state)
            neighbors = StateGenerator.neighbors(state, life.bounds)
            state = StateGenerator.update(state, neighbors)
            exit_code = StateGenerator.check_exit(history, num_generations)
            if exit_code == 2:
                # yield 30 more frames for animation before halting
                _, *last_two_states = history
                yield from (s for _ in range(15) for s in last_two_states)
                break
            elif exit_code == 0:
                break

    @staticmethod
    def update(s: LifeArray, n: LifeArray) -> LifeArray:
        """Calculates the next generation given state `s` and neighbors `n`"""
        return (n < 4) * (1 - s * (n % 2 - 1) + n) // 4

    @staticmethod
    def neighbors(x: LifeArray, mode: str) -> LifeArray:
        """Counts living neighbors around each cell in the `x` array"""
        return windows(np.pad(x, 1, mode=mode), (3, 3)).sum(axis=(2, 3)) - x

    @staticmethod
    def check_exit(history: Generations, num_generations: int) -> int:
        """
        Checks for three exit conditions and returns a corresponding code

        Exit codes:
            `0`: The maximum number of generations has been reached
            `1`: OK to continue
            `2`: A steady state has been reached
            `2`: An oscillating state of period 2 has been reached
        """
        exit_code = 1
        if len(history) > 2:
            if np.array_equal(history[-2], history[-1]):
                print("reached steady state")
                exit_code = 2
            elif np.array_equal(history[-3], history[-1]):
                print("oscillating state period 2")
                exit_code = 2
        if num_generations == MAX_GENERATIONS:
            print("reached max generations")
            exit_code = 0
        return exit_code


class TileMaker:
    """Static methods for generating random binary tiles"""
    @staticmethod
    def __call__(shape: ArrayShape, pattern_type: str) -> LifeArray:
        rotator = random.choice((np.fliplr, np.flipud, np.rot90, None))
        if pattern_type == "tiles":
            tile_maker = random.choice((TileMaker.diagonal,
                                        TileMaker.inverted_diagonal,
                                        TileMaker.quilt))
            array = tile_maker(shape)
        else:
            array = TileMaker.noise(shape)

        if rotator:
            return rotator(array)
        return array

    @staticmethod
    def noise(shape: tuple) -> LifeArray:
        """Binary noise tile, the base working unit for the other methods"""
        return np.random.randint(0, 2, shape, dtype="uint8")

    @staticmethod
    def quilt(shape: tuple) -> LifeArray:
        """A base triangular array"""
        return np.triu(TileMaker.noise(shape))

    @staticmethod
    def diagonal(shape: tuple) -> LifeArray:
        """Diagonally symmetric array"""
        tri = TileMaker.quilt(shape)
        return np.clip(tri + tri.T, 0, 1)

    @staticmethod
    def inverted_diagonal(shape: tuple) -> LifeArray:
        """Diagonally symmetric array with flipped bits in lower triangular"""
        tri = TileMaker.quilt(shape)
        return np.triu(np.where(tri, 0, 1)).T + tri


class TilingMethod:
    """Static methods for tiling an array with different symmetries"""
    @staticmethod
    def __call__(array: LifeArray, pattern_number: int) -> LifeArray:
        tiling_method = random.choice((TilingMethod.four_corners,
                                       TilingMethod.book_match,
                                       TilingMethod.hamburger,
                                       TilingMethod.repeat))
        return np.tile(tiling_method(array), (pattern_number,)*2)

    @staticmethod
    def four_corners(NW: LifeArray) -> LifeArray:
        """
        Radial symmetry, e.g.:
        ```
        A B B A
        C D D C
        C D D C
        A B B A
        ```
        """
        NE = np.fliplr(NW)
        SW = np.flipud(NW)
        SE = np.flipud(NE)
        return np.block([[NW, NE], [SW, SE]])

    @staticmethod
    def book_match(L: LifeArray) -> LifeArray:
        """
        Vertical symmetry, e.g.:
        ```
        A B B A
        A B B A
        A B B A
        A B B A
        ```
        """
        R = np.fliplr(L)
        return np.block([[L, R], [L, R]])

    @staticmethod
    def hamburger(T: LifeArray) -> LifeArray:
        """
        Horizontal symmetry, e.g.:
        ```
        A A A A
        B B B B
        B B B B
        A A A A
        ```
        """
        B = np.flipud(T)
        return np.block([[T, T], [B, B]])

    @staticmethod
    def repeat(x: LifeArray) -> LifeArray:
        """
        Repeating tiles, e.g.:
        ```
        A A
        A A
        ```
        """
        return np.tile(x, (2, 2))


class SeedGenerator:
    """
    A generator class which yields "Life" seeds.

    Taking `n` as the desired size of the final array, this generator
    first breaks down `n` by half into `k`, and then `k` into a tuple
    randomly chosen from the matched divisors of `k` such that the
    product of the tuple is `k`.

    This tuple gives a pattern number and initial size which are passed
    to the seed tile creator and tiling method static classes.

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

    def __next__(self) -> LifeArray:
        return next(self.seed_generator)

    def generator(self, life: Life) -> Iterator[LifeArray]:
        n = life.size
        k = n // 2
        while True:
            tile_maker, tiling_method = TileMaker(), TilingMethod()
            d = [(x, k // x) for x in range(2, k + 1) if k % x == 0]
            num_tiles, tile_size = sorted(random.choice(d))
            tile_shape = (tile_size,)*2
            array = tile_maker(tile_shape, life.pattern_type)
            yield tiling_method(array, num_tiles)


if __name__ == "__main__":
    life = Life(100)
    [*life]
