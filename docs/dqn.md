# DQN

DQN 是离散动作空间里最基础的深度强化学习方法。它把动作价值函数交给一个小型神经网络拟合，再配合经验回放和目标网络稳定训练。

## 1. 数学原理

最优 Q 函数满足 Bellman 方程：

$$
Q^*(s, a) = \mathbb{E}[r + \gamma \max_{a'} Q^*(s', a')]
$$

在实现里，用在线网络 $Q_\theta$ 预测当前动作价值，用目标网络 $Q_{\bar\theta}$ 构造监督信号：

$$
y = r + \gamma (1 - d)\max_{a'} Q_{\bar\theta}(s', a')
$$

训练目标是均方误差：

$$
L = \frac{1}{N}\sum_i \left(Q_\theta(s_i, a_i) - y_i\right)^2
$$

动作选择使用 epsilon-greedy。epsilon 采用线性衰减，从 `epsilon_start` 逐步降到 `epsilon_end`。

## 2. 伪代码

```text
initialize online Q network
initialize target Q network = online Q network
initialize replay buffer

for each environment step:
    with probability epsilon choose random action
    otherwise choose argmax_a Q_online(s, a)

    store transition (s, a, r, s', done)

    if replay buffer is ready:
        sample a minibatch
        q = Q_online(s, a)
        y = r + gamma * (1 - done) * max_a' Q_target(s', a')
        minimize MSE(q, y)

        decay epsilon linearly
        every K steps copy online network to target network
```

## 3. 代码映射

实现位于 [`modules/dqn.py`](../modules/dqn.py)。

`QNetwork` 是一个三层 MLP，输出每个离散动作的 Q 值。`ReplayBuffer` 只做最基本的循环存储和随机采样。`DQNAgent.update()` 里用 `gather` 取出当前动作的 Q 值，再用目标网络的 `max` 构造 target。

关键逻辑对应两句：

```python
q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
next_q_values = self.target_network(next_states).max(dim=1).values
```

测试脚本位于 [`tests/test_dqn.py`](../tests/test_dqn.py)，环境是 `CartPole-v1`。

## 4. 参考

Mnih et al., *Human-level control through deep reinforcement learning*.
