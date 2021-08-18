# life

Conway's game of life with options for fixed or periodic boundary conditions.

<!-- ![](./gifs/example.gif) -->

## Intro

I had always been very intrigued by [Conway's Game of Life](https://en.m.wikipedia.org/wiki/Conway%27s_Game_of_Life) but had never actually implemented it myself. When I looked up the ruleset, I realized I had no excuse.

For those unfamiliar, Conway's Game of Life (which I'll refer to as **Life** going forward) is what is known as a [cellular automaton](https://en.m.wikipedia.org/wiki/Cellular_automaton). Life is a 2-dimensional array of cells which observe a binary state: either on or off, alive or dead.

Going forward I'm going to be referring to a **game board** (or just **board**) and the state of Life at any given time as **state**.

The state of the game board at time `t` is entirely determined by the state of the board at time `t - 1`, and is governed by this ruleset:

1. **Underpopulation:** a live cell with <= 1 live neighbors dies
2. **Survival:** a live cell with 2 or 3 neighbors survives
3. **Overpopulation:** a live cell with >= 4 live neighbors dies
4. **Reproduction:** a dead cell with exactly 3 neighbors is reborn

Those are the original rules, but they can actually be distilled down to only three rules:

> 1. Any live cell with two or three live neighbors survives.
> 2. Any dead cell with three live neighbors becomes a live cell.
> 3. All other live cells die in the next generation. Similarly, all other dead cells stay dead.

[(source)](https://en.m.wikipedia.org/wiki/Conway%27s_Game_of_Life#Rules)

## The algorithm

The basic algorithm for implementing Life is as follows:

1. Check each cell on the game board, counting how many neighbors it has
2. Determine whether that cell should die, survive, or be reborn
3. Update each cell on the game board

Each cell's neighborhood consists of the eight cells adjacent to it (up, down, left, right, and the diagonals).

## Each cell is a finite automata!

The word **state** in this section will specifically refer only to the state of an individual cell.

Since there are two states in which a cell can be observed (alive or dead), and anywhere from 0 to 8 neighbors, it's entirely possible to enumerate all the possible state transitions as the cartesian product of \(\{0, 1\} \times \{0,1,\dots,8\},\) for a total of 16 possible state transitions. Familiarity with [discrete finite state machines](https://en.m.wikipedia.org/wiki/Finite-state_machine) is helpful here, but not necessary.

Here are a few examples from those 16 state transitions:

```text
(0, 5) -> 0
(1, 3) -> 1
(1, 6) -> 0
(0, 3) -> 1
```

The way to read these state transitions is from left to right: on the left hand side is a 2-tuple containing the current state of any given cell \(c\) and the number of *live* neighbors it has, and the right hand side is the next state of the cell \(c.\) So for instance, the state transition `(0, 5) -> 0` applies to any cell which is dead (the `0` in the 2-tuple) and which has five living neighbors (the `5` in the 2-tuple). As you can see, the next state of that cell is 0, meaning the cell stays dead.

Notice that the vast majority of the state transitions are equivalent based on those three rules from earlier. There's no functional difference between `(0, 5) -> 0` and `(0, 7) -> 0`.

Taking this into account, there are actually only three state transitions which we need to consider:

```text
(1, 2) -> 1
(1, 3) -> 1
(0, 3) -> 1
```

The first two transitions reflect the first rule:

> Any live cell with two or three live neighbors survives.

And the third transition reflects the second rule:

> Any dead cell with three live neighbors becomes a live cell.

Every other possible state (again, a 2-tuple `(alive or dead, number of neighbors)`) results in either a live cell dying or a dead cell staying dead.

## Implementation

Conventionally, most implementations take a looping approach where each cell is checked one-by-one. I decided to take advantage of NumPy's vectorization to simplify the calculations with the tradeoff being that the initial set up was a little bit more complicated to figure out. We'll go through the implementation here step-by-step. It may help to have `life.py` open somewhere to which you can refer, however the relevant code will also be shown here.

The easiest place to start is in encoding the rules of Life. So the three state transitions we care about become

```python
RULES = {(1, 2): 1,
         (1, 3): 1,
         (0, 3): 1}
```

## Planned features

A modal in which a user may draw an initial state.
