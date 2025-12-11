# LunarLander-v3 Reinforcement Learning Comparison

## Overview

This project implements and compares three value-based deep reinforcement learning algorithms for the LunarLander-v3 environment from OpenAI Gymnasium. The implementation is part of coursework for Foundations of Robotics & AI (CIS2719 Coursework 2).

**Author:** Ethan Hulme

## Algorithms Implemented

The project compares the following algorithms:

1. **DQN (Deep Q-Network)** - Standard value-based RL using experience replay and target networks
2. **DDQN (Double DQN)** - Addresses overestimation bias by decoupling action selection from evaluation
3. **PER (Prioritized Experience Replay)** - Samples transitions based on TD-error magnitude combined with DDQN targets

## Environment

- **Task:** LunarLander-v3 (discrete action space)
- **State Space:** 8-dimensional continuous observations
- **Action Space:** 4 discrete actions (do nothing, fire left engine, fire main engine, fire right engine)
- **Solved Threshold:** Average reward ≥ 200 over 100 consecutive episodes

## Architecture

### Q-Network
- 2-layer fully connected neural network
- Hidden dimension: 128 units per layer
- Activation: ReLU
- Output: Q-values for each action

### Hyperparameters
- Learning rate: 1e-3
- Discount factor (γ): 0.99
- Batch size: 64
- Replay buffer capacity: 100,000
- Minimum buffer size before training: 10,000
- Target network update frequency: 1,000 steps
- Epsilon decay: 250,000 frames (1.0 → 0.05)
- Loss function: Smooth L1 (Huber loss)
- Optimizer: Adam

### PER-Specific Parameters
- Alpha (priority exponent): 0.6
- Beta (importance sampling): 0.4 → 1.0 (annealed over 200,000 frames)

## Project Structure

```
LunarLander-V2/
├── LunarLander-V2.py       # Main implementation
├── README.md               # This file
└── Test Results/           # Experimental results
    ├── Test 1 (600 Eps)/
    ├── Test 2 (600 Eps)/
    ├── Test 3 (600 Eps)/
    ├── Test 4 (600 Eps)/
    ├── Test 5 (600 Eps)/
    ├── Test 6 (2000 Eps)/
    ├── Test 7 (2000 Eps)/
    ├── Test 8 (2000 Eps)/
    ├── Test 9 (5000 Eps)/
    └── Test 10 (10000 Eps)/
```

Each test directory contains:
- `learning_curves.png` - Training curves for all three algorithms
- `dqn_agent.gif` - Visualization of DQN policy
- `ddqn_agent.gif` - Visualization of DDQN policy
- `per_agent.gif` - Visualization of PER policy

## Requirements

```bash
gymnasium[box2d]
numpy
torch
matplotlib
imageio
```

## Installation

```bash
pip install gymnasium[box2d] numpy torch matplotlib imageio
```

## Usage

Run the main script to train all three agents and generate results:

```bash
python LunarLander-V2.py
```

The script will:
1. Train DQN, DDQN, and PER agents sequentially
2. Output progress every 20 episodes
3. Generate a comparative learning curve plot (`learning_curves.png`)
4. Create policy visualization GIFs for each algorithm

### Configuring Training Duration

Modify the `NUM_EPISODES` variable in the main block:

```python
NUM_EPISODES = 600  # Adjust as needed
```

## Implementation Details

### DQN
- Uses standard experience replay with uniform sampling
- TD target: `r + γ * max_a' Q_target(s', a')`

### DDQN
- Decouples action selection (online network) from action evaluation (target network)
- TD target: `r + γ * Q_target(s', argmax_a' Q_online(s', a'))`
- Reduces overestimation bias present in DQN

### PER
- Samples transitions proportionally to TD-error magnitude
- Uses importance sampling weights to correct for bias
- Combines prioritized sampling with DDQN-style targets
- Priorities updated after each gradient step

## Results

The project includes 10 experimental runs with varying episode counts:
- Tests 1-5: 600 episodes (initial experiments)
- Tests 6-8: 2,000 episodes (extended training)
- Test 9: 5,000 episodes
- Test 10: 10,000 episodes (maximum training duration)

Results demonstrate the relative performance and sample efficiency of each algorithm across different training horizons.

## Key Features

- Epsilon-greedy exploration with exponential decay
- Target network for stability
- Gradient clipping (max norm: 10.0) for training stability
- Reproducible results via seeding
- Smooth L1 loss for robustness to outliers
- Moving average smoothing for learning curves

## License

This project is submitted as coursework for academic evaluation.

## References

- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*.
- Van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-learning. *AAAI*.
- Schaul, T., et al. (2015). Prioritized Experience Replay. *ICLR*.

