# Reinforcement Learning

[Usage](#1.-usage) | [Agents](#2.-agents) | [Environments](#3.-environments)

A little reinforcement learning project. _Learning by doing._

## 1) Usage:

```
$ python train.py --help
usage: train.py [-h] [--episodes EPISODES] --agent {q_learning,} --environment
                ENVIRONMENT [--hyperparams HYPERPARAMS] [--console_log]
                [--neptune_log]

Reinforcement learning trainer.

optional arguments:
  -h, --help            show this help message and exit
  --episodes EPISODES   (default=5000) The amount of episodes to train for.
  --agent {q_learning,}
                        The name of the agent to be used.
  --environment ENVIRONMENT
                        The environment to run the agent against.
  --hyperparams HYPERPARAMS
                        (default={}) JSON encoded hyperparameters, passed to
                        the agent.
  --console_log         Log training stats to console.
  --neptune_log         Log training stats to neptune.ai.
```
```
$ python train.py --agent q_learning --environment FrozenLake8x8-v0
```

## 2) Agents:

All agents must exent the `Agent` class. Valid parameters to the `Agent` class constructor are:

- `observation_space`: Environment observation space (Discrete or Box).
- `action_space`: Environment action space (Discrete. or Box)
- `hyperparams`: Dictionary containing the agent hyperparameters.

### 2.1) Q-Learning

Tabular `QLearning` agent. Works for discrete state and action spaces.

**Formulas:**

The Q-Learning update function.

<img src="https://render.githubusercontent.com/render/math?math=Q^{new}(s_{t}, a_{t}) = Q(s_{t}, a_{t}) %2B \alpha * [ r_{t} %2B \gamma * maxQ_{a}(s_{t%2B1}, a) - Q(s_t, a_t) ]" />

**Hyperparams:**

- `alpha`: The learning rate. Prevents the algorithm from putting too much weight on random missteps.
- `gamma`: Reward decay over time.
- `epsilon`: Exploratory helper parameter.
- `epsilon_decay`: `epsilon` decay factor: epsilon = epsilon * epsilon_decay.
- `epsilon_min`: Minimum `epsilon` to be reached. Allows for continuous exploration.

### 2.2) Vanilla Policy Gradient

## 3) Environments

Currently the environments are passed into the OpenAI's `gym.make` function.
