<div align="right">
  <a href="dqn.md">English</a> |
  <a href="dqn_CN.md">简体中文</a>
</div>

# DQN

DQN is one of the most basic deep reinforcement learning methods for discrete action spaces. It fits the action-value function with a small neural network and uses experience replay and a target network to stabilize training.

## 1. Mathematical Principles

The optimal Q function satisfies the Bellman equation:

$$
Q^*(s, a) = \mathbb{E}[r + \gamma \max_{a'} Q^*(s', a')]
$$

The implementation uses an online network $Q_\theta$ to predict the current action values and a target network $Q_{\bar\theta}$ to construct the supervision target:

$$
y = r + \gamma (1 - d)\max_{a'} Q_{\bar\theta}(s', a')
$$

The training objective is mean squared error:

$$
L = \frac{1}{N}\sum_i \left(Q_\theta(s_i, a_i) - y_i\right)^2
$$

Action selection uses epsilon-greedy exploration. Epsilon is linearly decayed from `epsilon_start` to `epsilon_end`.

## 2. Pseudocode

```text
initialize online Q network
initialize target Q network = online Q network
initialize replay buffer

for each environment step:
    with probability epsilon choose a random action
    otherwise choose argmax_a Q_online(s, a)

    store transition (s, a, r, s', done)

    if the replay buffer is ready:
        sample a minibatch
        q = Q_online(s, a)
        y = r + gamma * (1 - done) * max_a' Q_target(s', a')
        minimize MSE(q, y)

        decay epsilon linearly
        every K steps copy the online network to the target network
```

## 3. Code Mapping

The implementation is in [`modules/dqn.py`](../modules/dqn.py).

`QNetwork` is a three-layer MLP that outputs the Q value for every discrete action. `ReplayBuffer` only implements basic circular storage and random sampling. In `DQNAgent.update()`, `gather` selects the Q value of the action that was actually taken, and the target network's `max` constructs the target.

The key logic corresponds to these two lines:

```python
q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
next_q_values = self.target_network(next_states).max(dim=1).values
```

The test script is [`tests/test_dqn.py`](../tests/test_dqn.py), using the `CartPole-v1` environment.

## 4. Reference

Mnih et al., *Human-level control through deep reinforcement learning*.
