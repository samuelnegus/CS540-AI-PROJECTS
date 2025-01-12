# HW8: Funny Puzzle Solver

This project implements a solver for a variant of the 8-tile puzzle using the A* search algorithm. The puzzle operates on a 3x3 grid with fewer than 8 tiles, with empty spaces represented by `0`. The objective is to arrange the tiles in ascending order with the empty spaces at the bottom-right corner of the grid.

## Table of Contents

- [Features](#features)
- [Input and Output](#input-and-output)
- [Functions](#functions)
  - [print_succ(state)](#print_succstate)
  - [solve(state)](#solvestate)
- [Usage](#usage)
- [Examples](#examples)
- [Files](#files)
- [Requirements](#requirements)

---

## Features

- Generates valid successor states for a given puzzle configuration.
- Implements the A* search algorithm to solve the puzzle.
- Calculates the heuristic (Manhattan distance) for each state.
- Prints the solution path, including heuristic values, moves, and maximum queue length.

---

## Input and Output

### Input
- The initial puzzle state is represented as a one-dimensional list of integers with `0` indicating empty spaces.
- Example: `[2, 5, 1, 4, 3, 6, 7, 0, 0]`.

### Output
- For `print_succ(state)`: The function prints all valid successor states and their heuristic values.
- For `solve(state)`: The function prints whether the puzzle is solvable and the solution path with heuristic values, number of moves, and the maximum queue length.

---

## Functions

### `print_succ(state)`
**Description**:  
Prints all valid successors of the given state in sorted order along with their heuristic values.

**Input**:  
- `state` (list): The current puzzle configuration.

**Output**:  
- Prints successor states and their heuristic values.

---

### `solve(state)`
**Description**:  
Uses the A* search algorithm to determine whether the puzzle is solvable and finds the solution path if it exists.

**Input**:  
- `state` (list): The initial puzzle configuration.

**Output**:  
- Prints `True` if solvable and the solution path; `False` if not solvable.
- Solution path includes each state, its heuristic value (`h`), the number of moves, and the maximum queue length.

---

## Usage

Run the `funny_puzzle.py` file to test the functions. Example usage:

1. **Print Successors**:
   ```python
   print_succ([2, 5, 1, 4, 0, 6, 7, 0, 3])


