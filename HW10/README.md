# Reinforcement Learning on CliffWalking-v0

This project implements **Q-Learning** and **SARSA** to train an agent to navigate the **CliffWalking-v0** environment from Farama Gymnasium. The agent learns to avoid falling into the cliff while reaching the goal efficiently.

---

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Setup and Execution](#setup-and-execution)
- [Q-Learning](#q-learning)
  - [Algorithm](#algorithm)
  - [Hyperparameters](#hyperparameters)
- [SARSA (Extra Credit)](#sarsa-extra-credit)
  - [Algorithm](#algorithm-1)
  - [Hyperparameters](#hyperparameters-1)
- [Evaluation](#evaluation)
- [Files](#files)
- [Notes](#notes)

---

## Introduction

The **CliffWalking-v0** environment represents a grid world where:
- The agent starts at the top-left corner.
- The agent must navigate to the bottom-right corner while avoiding the cliff.
- The agent incurs:
  - **-1** reward for every step.
  - **-100** reward for falling into the cliff.
  - A **0** reward for reaching the goal.

The environment is solved using **Q-Learning** and **SARSA** with an ε-greedy policy.

---

## Requirements

- Python 3.8 or later
- Libraries:
  - `gymnasium==0.29.1`
  - `matplotlib==3.9.2`
  - `pygame==2.5.2`
  - `numpy`

---

## Setup and Execution

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-repo/reinforcement-learning-cliffwalking.git
   cd reinforcement-learning-cliffwalking

