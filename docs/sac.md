# SAC

SAC 是一种面向连续动作空间的离策略 actor-critic 算法。它的核心是同时优化“高回报”和“高熵”，让策略既能学到奖励，也保留足够探索。

## 1. 数学原理

SAC 使用一个随机策略 $\pi_\theta(a|s)$，本实现里把它写成 squashed Gaussian：actor 输出均值和标准差，在 `SACAgent` 中采样高斯动作，再经过 $\tanh$ 把动作压到范围内。

策略目标可写成：

$$
J_\pi = \mathbb{E}\left[\min(Q_1(s, a), Q_2(s, a)) - \alpha \log \pi_\theta(a|s)\right]
$$

其中 $\alpha$ 是熵系数。为了简洁，这里把它固定为常数，不做自动调节。

双 Q 目标值为：

$$
y = r + \gamma (1 - d)\left(\min(Q_1'(s', a'), Q_2'(s', a')) - \alpha \log \pi_\theta(a'|s')\right)
$$

critic 通过均方误差拟合这个 target：

$$
L_Q = \left(Q_1(s, a) - y\right)^2 + \left(Q_2(s, a) - y\right)^2
$$

目标网络使用软更新：

$$
\theta' \leftarrow (1 - \tau)\theta' + \tau \theta
$$

## 2. 伪代码

```text
initialize actor, two critics, two target critics
initialize replay buffer

for each environment step:
    sample action from stochastic actor
    store transition (s, a, r, s', done)

    if replay buffer is ready:
        sample minibatch
        a' ~ pi(s')
        y = r + gamma * (1 - done) * (min(Q1', Q2') - alpha * log pi(a'|s'))
        minimize critic loss

        a ~ pi(s)
        maximize min(Q1, Q2) - alpha * log pi(a|s)

        softly update target critics
```

## 3. 代码映射

实现位于 [`modules/sac.py`](../modules/sac.py)。

`Actor` 只输出均值和标准差：均值头使用 `tanh`，标准差头使用 `softplus`。`Critic` 和 DDPG 一样，直接把 state 和 action 拼接后回归 Q 值。`SACAgent` 内部维护两套 critic 和两套 target critic。

`SACAgent.update()` 里最关键的部分是：

```python
next_actions, next_log_probs = self._sample_action(next_states)
target_q = torch.min(
    self.target_critic1(next_states, next_actions),
    self.target_critic2(next_states, next_actions),
).squeeze(1) - self.alpha * next_log_probs
```

这正是 SAC 的核心：用双 Q 估计减小过估计，同时把熵项加进目标，保持探索。

测试脚本位于 [`tests/test_sac.py`](../tests/test_sac.py)，环境是 `Pendulum-v1`。

## 4. 参考

Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*.
