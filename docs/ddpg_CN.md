<div align="right">
  <a href="ddpg.md">English</a> |
  <a href="ddpg_CN.md">简体中文</a>
</div>

# DDPG

DDPG 是面向连续动作空间的确定性 actor-critic 算法。它同时学习一个 actor 产生动作，和一个 critic 评估动作价值。

## 1. 数学原理

actor 记作 $\mu_\theta(s)$，critic 记作 $Q_\phi(s, a)$。

目标值由目标 actor 和目标 critic 给出：

$$
y = r + \gamma (1 - d) Q_{\phi'}(s', \mu_{\theta'}(s'))
$$

critic 用均方误差拟合这个 target：

$$
L_{\text{critic}} = \left(Q_\phi(s, a) - y\right)^2
$$

actor 通过最大化 critic 估计的 Q 值来更新：

$$
L_{\text{actor}} = -Q_\phi(s, \mu_\theta(s))
$$

目标网络采用软更新：

$$
\theta' \leftarrow (1 - \tau)\theta' + \tau \theta
$$

$$
\phi' \leftarrow (1 - \tau)\phi' + \tau \phi
$$

## 2. 伪代码

```text
initialize actor, critic, target actor, target critic
initialize replay buffer

for each environment step:
    a = actor(s) + exploration noise
    store (s, a, r, s', done)

    if replay buffer is ready:
        sample minibatch
        y = r + gamma * (1 - done) * critic_target(s', actor_target(s'))
        minimize critic loss
        minimize actor loss = -critic(s, actor(s))
        softly update target networks
```

## 3. 代码映射

实现位于 [`modules/ddpg.py`](../modules/ddpg.py)。

`Actor` 直接输出连续动作，再乘 `tanh` 和动作上界。`Critic` 拼接 state 和 action 做回归。`DDPGAgent.select_action()` 在 actor 输出上加高斯噪声，测试脚本里把噪声强度从 `0.3` 线性降到 `0.05`。

`DDPGAgent.update()` 的核心是先更新 critic，再更新 actor，最后做软更新。

测试脚本位于 [`tests/test_ddpg.py`](../tests/test_ddpg.py)，环境是 `Pendulum-v1`。

## 4. 参考

Lillicrap et al., *Continuous control with deep reinforcement learning*.
