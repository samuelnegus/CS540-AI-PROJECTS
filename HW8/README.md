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
- Example: `[2, 5, 1

