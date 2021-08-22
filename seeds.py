# Standard Library
import random
from typing import Iterator, Tuple

# Third party
import numpy as np

# Type aliases
Tile = np.ndarray
LifeSeed = np.ndarray
ArrayShape = Tuple[int, int]
LifeSeedGenerator = Iterator[LifeSeed]


class TileMaker:
    """Static methods for generating random binary tiles"""
    @staticmethod
    def __call__(tile_size: int) -> Tile:
        rotator = random.choice((np.fliplr, np.flipud, np.rot90, None))
        tile_maker = random.choice((TileMaker.diagonal,
                                    TileMaker.inverted_diagonal,
                                    TileMaker.quilt))
        array = tile_maker((tile_size,)*2)
        return rotator(array) if rotator else array

    @staticmethod
    def noise(shape: ArrayShape) -> Tile:
        """Binary noise tile, the base working unit for the other methods"""
        return np.random.randint(0, 2, shape, dtype="uint8")

    @staticmethod
    def quilt(shape: ArrayShape) -> Tile:
        """A base triangular array"""
        return np.triu(TileMaker.noise(shape))

    @staticmethod
    def diagonal(shape: ArrayShape) -> Tile:
        """Diagonally symmetric array"""
        tri = TileMaker.quilt(shape)
        return np.clip(tri + tri.T, 0, 1)

    @staticmethod
    def inverted_diagonal(shape: ArrayShape) -> Tile:
        """Diagonally symmetric array with flipped bits in lower triangular"""
        tri = TileMaker.quilt(shape)
        return np.triu(np.where(tri, 0, 1)).T + tri


class TilePattern:
    """Static methods for tiling an array with different symmetries"""
    @staticmethod
    def __call__(array: Tile, pattern_number: int) -> LifeSeed:
        tiling_method = random.choice((TilePattern.four_corners,
                                       TilePattern.book_match,
                                       TilePattern.hamburger,
                                       TilePattern.repeat))
        return np.tile(tiling_method(array), (pattern_number,)*2)

    @staticmethod
    def four_corners(NW: Tile) -> LifeSeed:
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
    def book_match(L: Tile) -> LifeSeed:
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
    def hamburger(T: Tile) -> LifeSeed:
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
    def repeat(x: Tile) -> LifeSeed:
        """
        Repeating tiles, e.g.:
        ```
        A A
        A A
        ```
        """
        return np.tile(x, (2, 2))


class SeedGenerators:
    @staticmethod
    def symmetric(n: int) -> LifeSeedGenerator:
        """
        Taking `n` as the desired dimensions of the final seed, this generator
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
        k = n // 2
        while True:
            tile_maker, tile_pattern = TileMaker(), TilePattern()
            d = [(x, k // x) for x in range(2, k + 1) if k % x == 0]
            num_tiles, tile_size = sorted(random.choice(d))
            yield tile_pattern(tile_maker(tile_size), num_tiles)

    @staticmethod
    def noisy(n: int) -> LifeSeedGenerator:
        """Yields uniformly noisy binary arrays"""
        shape = (n, n)
        while True:
            yield TileMaker.noise(shape)


def new_seed_generator(size: int, seed_type: int) -> LifeSeedGenerator:
    if seed_type == "noise":
        return SeedGenerators.noisy(size)
    return SeedGenerators.symmetric(size)
