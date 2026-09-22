<div align="right">
  <a href="td3.md">English</a> |
  <a href="td3_CN.md">简体中文</a>
</div>

# TD3

TD3 是 DDPG 的改进版本，全称是 Twin Delayed Deep Deterministic Policy Gradient。它仍然是面向连续动作空间的确定性 actor-critic 算法，但加入了三个简单而关键的稳定化技巧：

1. 使用两个 critic，并取两个 Q 值中的较小值；
2. 在目标动作上加入裁剪后的噪声，做 target policy smoothing；
3. critic 每次更新，actor 和目标网络延迟更新。

## 1. 数学原理

actor 记作 $\mu_\theta(s)$，两个 critic 分别记作 $Q_{\phi_1}(s,a)$ 和 $Q_{\phi_2}(s,a)$。

### 1.1 Clipped Double Q

TD3 用两个目标 critic 估计下一个状态动作的价值，并取较小值：

$$
\bar{Q}(s', a')
=
\min
\left(
Q_{\phi_1'}(s', a'),
Q_{\phi_2'}(s', a')
\right)
$$

这样可以减轻 Q 函数过估计问题。

### 1.2 Target Policy Smoothing

目标动作不是直接使用目标 actor 的输出，而是加入一小段裁剪噪声：

$$
a' =
\operatorname{clip}
\left(
\mu_{\theta'}(s') + \epsilon,
-a_{\max},
a_{\max}
\right)
$$

其中：

$$
\epsilon \sim
\operatorname{clip}
\left(
\mathcal{N}(0, \sigma),
-c,
c
\right)
$$

这可以让 critic 不要只依赖非常尖锐的动作点估计。

### 1.3 Critic 目标

TD3 的 Bellman target 为：

$$
y =
r + \gamma(1-d)
\min
\left(
Q_{\phi_1'}(s', a'),
Q_{\phi_2'}(s', a')
\right)
$$

两个 critic 都使用均方误差拟合同一个 target：

$$
L_Q =
\left(Q_{\phi_1}(s,a)-y\right)^2
+
\left(Q_{\phi_2}(s,a)-y\right)^2
$$

### 1.4 Delayed Policy Update

actor 的目标仍然是最大化 critic 估计的 Q 值：

$$
L_\mu =
-Q_{\phi_1}(s, \mu_\theta(s))
$$

但 actor 不会在每次 critic 更新后都更新，而是每隔 `policy_delay` 次 critic 更新才更新一次。目标网络也跟随 actor 的延迟更新一起软更新：

$$
\theta' \leftarrow (1-\tau)\theta' + \tau\theta
$$

$$
\phi_i' \leftarrow (1-\tau)\phi_i' + \tau\phi_i
$$

## 2. 伪代码

```text
initialize actor, actor_target
initialize critic1, critic2, critic1_target, critic2_target
initialize replay buffer

for each environment step:
    action = actor(state) + exploration noise
    store (state, action, reward, next_state, done)

    if replay buffer is ready:
        sample minibatch

        noise = clipped Gaussian noise
        next_action = clip(actor_target(next_state) + noise)
        target_q = min(critic1_target(next_state, next_action),
                       critic2_target(next_state, next_action))
        y = reward + gamma * (1 - done) * target_q

        update critic1 and critic2 with MSE loss

        if update step % policy_delay == 0:
            update actor with -critic1(state, actor(state))
            softly update actor_target and critic targets
```

## 3. 代码映射

实现位于 [`modules/td3.py`](../modules/td3.py)。

`Actor` 与 DDPG 中的结构一致，输入状态，输出经过 `tanh` 限幅后的连续动作：

```python
return self.max_action * torch.tanh(self.net(state))
```

`Critic` 将 state 和 action 拼接后回归一个 Q 值。TD3 中维护两套 critic：

```python
self.critic1 = Critic(...)
self.critic2 = Critic(...)
```

在 `TD3Agent.update()` 中，target policy smoothing 对应：

```python
noise = torch.randn_like(actions) * self.policy_noise
noise = noise.clamp(-self.noise_clip, self.noise_clip)
next_actions = (self.actor_target(next_states) + noise).clamp(
    -self.max_action, self.max_action
)
```

clipped double Q 对应：

```python
target_q = torch.min(target_q1, target_q2)
```

delayed policy update 对应：

```python
if self.total_updates % self.policy_delay == 0:
    actor_loss = -self.critic1(states, self.actor(states)).mean()
```

测试脚本位于 [`tests/test_td3.py`](../tests/test_td3.py)，环境是 `Pendulum-v1`。测试会保存训练奖励曲线和最终策略的 MP4 动态演示。

## 4. 参考

Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods*.
