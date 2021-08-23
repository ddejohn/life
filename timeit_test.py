import timeit


import numpy as np
from numpy.lib.stride_tricks import sliding_window_view as window


# SHAPE = (100, 100)
# RULES = {(1, 2): 1, (1, 3): 1, (0, 3): 1}
# NEXT_STATE = np.vectorize(lambda x, y: RULES.get((x, y), 0))


# c = np.random.randint(0, 2, SHAPE, dtype="uint8")
# n = np.random.randint(0, 9, SHAPE, dtype="uint8")


# def next_state(s, n):
#     return (n < 4) * (1 - s * (n % 2 - 1) + n) // 4


# t1 = timeit.timeit(
#     "NEXT_STATE(c, n)",
#     number=1000,
#     globals=globals()
# )

# t2 = timeit.timeit(
#     "next_state(c, n)",
#     number=1000,
#     globals=globals()
# )

# print(f"vectorized dict: {t1:.2f} s")
# print(f"formula: {t2:.2f} s")


# c = np.random.randint(0, 2, SHAPE, dtype="uint8")
# n = np.random.randint(0, 9, SHAPE, dtype="uint8")
# print(np.array_equal(NEXT_STATE(c, n), next_state(c, n)))


class PadTest:
    def __init__(self, x: np.ndarray, p: str, b=False):
        self.x = x
        self.z = np.zeros(shape=(102, 102), dtype="uint8")
        self.idxs = [-1, *range(self.x.shape[0]), 0]
        self.mode = "constant" if p == "z" else "wrap"
        self.pad = self.built_in if b else self.zero if p == "z" else self.wrap

    @property
    def zero(self) -> np.ndarray:
        self.z[1:-1, 1:-1] = self.x
        return self.z

    @property
    def wrap(self) -> np.ndarray:
        return self.x[:, self.idxs][self.idxs, :]

    @property
    def built_in(self) -> np.ndarray:
        return np.pad(self.x, 1, mode=self.mode)

    def run(self):
        for _ in range(1000):
            n = window(self.pad, (3, 3)).sum(axis=(2, 3)) - self.x
            self.x = (n < 4) * (1 - self.x * (n % 2 - 1) + n) // 4


x = np.random.randint(2, size=(100, 100), dtype="uint8")
t1 = timeit.timeit(stmt="p.run()",
                   setup="p = PadTest(x, 'z')",
                   number=10,
                   globals=globals())
t2 = timeit.timeit(stmt="p.run()",
                   setup="p = PadTest(x, 'w')",
                   number=10,
                   globals=globals())
t3 = timeit.timeit(stmt="p.run()",
                   setup="p = PadTest(x, 'z', True)",
                   number=10,
                   globals=globals())
t4 = timeit.timeit(stmt="p.run()",
                   setup="p = PadTest(x, 'w', True)",
                   number=10,
                   globals=globals())


print(f"zero: {t1:.4f} s")
print(f"wrap: {t2:.4f} s")
print(f"np.pad zero: {t3:.4f} s")
print(f"np.pad wrap: {t4:.4f} s")
