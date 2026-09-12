# PPO

PPO 是一种稳定的 on-policy actor-critic 方法。它用旧策略采样，再用 clipped objective 限制新旧策略差异，避免更新步子过大。

## 1. 数学原理

策略记作 $\pi_\theta(a|s)$，价值函数记作 $V_\phi(s)$。本实现使用最直接的对角高斯策略：

$$
\mu_\theta(s) = a_{\max}\tanh(f_\mu(s)), \qquad
\sigma_\theta(s) = \operatorname{softplus}(f_\sigma(s))
$$

然后使用

$$
\pi_\theta(a|s) = \mathcal{N}(a;\mu_\theta(s), \sigma_\theta(s))
$$

策略比率为：

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

GAE 计算优势：

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1})(1-d_t) - V_\phi(s_t)
$$

$$
A_t = \delta_t + \gamma \lambda (1-d_t)A_{t+1}
$$

clipped policy loss：

$$
L^{\text{clip}} = \mathbb{E}\left[\min(\rho_tA_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t)\right]
$$

value loss：

$$
L_V = \mathbb{E}\left[(V_\phi(s_t) - R_t)^2\right],
\qquad R_t = A_t + V_\phi(s_t)
$$

## 2. 伪代码

```text
initialize actor_new, actor_old, critic
copy actor_new to actor_old
initialize replay buffer

for each episode:
    use actor_old to collect transitions (s, a, r, s', done)
    store them in replay buffer

    compute values and next values with critic
    compute GAE advantages and returns

    for several epochs:
        sample minibatches from the buffer
        compute ratio = pi_new(a|s) / pi_old(a|s)
        update actor_new with clipped objective
        update critic with MSE loss

    copy actor_new to actor_old
    clear replay buffer
```

## 3. 代码实现

实现位于 [`modules/ppo.py`](../modules/ppo.py)。

`Actor` 用两个 head 分别输出 mean 和 std，其中 mean 使用 `tanh`，std 使用 `softplus`。`ReplayBuffer` 只保存 `(state, action, reward, next_state, done)`，不保存 log prob 和 value。`PPOAgent` 在 `update()` 里现算 GAE、现算 old/new log prob，然后分别优化 actor 和 critic。

核心代码对应这两句：

```python
old_mean, old_std = self.actor_old(states)
old_dist = torch.distributions.Normal(old_mean, old_std)
old_log_probs = old_dist.log_prob(actions).sum(dim=-1)

mean, std = self.actor_new(states[mb])
dist = torch.distributions.Normal(mean, std)
new_log_probs = dist.log_prob(actions[mb]).sum(dim=-1)
```

`actor_old` 只负责采样和提供旧策略概率，`actor_new` 负责学习。更新完成后，`actor_old` 直接同步 `actor_new`。对于有界环境，环境可以在执行动作时进行裁剪，但 PPO 仍然对策略实际采样的动作计算概率。

GAE 得到优势后，本实现对当前 rollout 的优势做标准化，再用于 PPO 损失。优势标准化不是 GAE 定义的一部分，但它是 PPO 连续控制实现中常用的稳定化步骤。

测试脚本位于 [`tests/test_ppo.py`](../tests/test_ppo.py)，环境是 `Pendulum-v1`。

## 4. 参考

Schulman et al., *Proximal Policy Optimization Algorithms*.
