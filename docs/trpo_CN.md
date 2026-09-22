<div align="right">
  <a href="trpo.md">English</a> |
  <a href="trpo_CN.md">简体中文</a>
</div>

# TRPO

TRPO 是一种 on-policy actor-critic 算法，全称是 Trust Region Policy Optimization。它的核心思想是：更新策略时不仅要让新策略带来更高的优势目标，还要限制新旧策略之间的 KL 距离，避免策略一步变化过大。

## 1. 数学原理

策略记作 $\pi_\theta(a|s)$，价值函数记作 $V_\phi(s)$。本实现使用有界的 squashed Gaussian 策略。actor 输出高斯分布的原始均值和标准差：

$$
\mu_\theta(s)=f_\theta(s),
\qquad
\sigma_\theta=\exp(\log\sigma_\theta)
$$

先采样无界动作 $u$，再通过 `tanh` 映射到环境动作范围：

$$
u\sim\mathcal{N}(\mu_\theta(s),\sigma_\theta^2),
\qquad
a=a_{\max}\tanh(u)
$$

计算策略概率时加入 `tanh` 变量变换对应的 Jacobian 修正：

$$
\log\pi_\theta(a|s)
=
\log\mathcal{N}(u;\mu_\theta(s),\sigma_\theta(s))
-
\log\left(
a_{\max}(1-\tanh^2(u))+\epsilon
\right)
$$

这样可以保证实际执行的有界动作与 TRPO 中使用的策略概率一致。

### 1.1 优势估计

本实现使用 GAE 估计优势：

$$
\delta_t =
r_t + \gamma V_\phi(s_{t+1})(1-d_t) - V_\phi(s_t)
$$

$$
A_t =
\delta_t + \gamma\lambda(1-d_t)A_{t+1}
$$

回报目标为：

$$
R_t = A_t + V_\phi(s_t)
$$

critic 使用均方误差拟合 $R_t$。

### 1.2 约束策略优化

TRPO 的策略更新可以写成：

$$
\max_\theta
\mathbb{E}
\left[
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{\text{old}}}(a_t|s_t)}
A_t
\right]
$$

同时要求新旧策略的平均 KL 距离不超过阈值：

$$
\mathbb{E}
\left[
D_{\mathrm{KL}}
\left(
\pi_{\theta_{\text{old}}}(\cdot|s_t)
\|
\pi_\theta(\cdot|s_t)
\right)
\right]
\leq \delta
$$

其中 $\delta$ 对应代码中的 `max_kl`。

### 1.3 自然梯度方向

TRPO 使用二阶近似，把 KL 约束对应的 Hessian 记作 $H$，把策略目标梯度记作 $g$。更新方向近似为：

$$
x = H^{-1}g
$$

代码中不显式构造完整 Hessian，而是通过 Hessian-vector product 和共轭梯度法求解这个方向。

为了满足 KL 约束，步长按下面的形式缩放：

$$
\alpha =
\sqrt{
\frac{2\delta}
{x^THx}
}
$$

最终候选更新为：

$$
\theta_{\text{new}}
=
\theta_{\text{old}} + \alpha x
$$

如果候选更新没有满足 KL 约束或没有提升策略目标，就在线搜索中逐步缩小步长。

## 2. 伪代码

```text
initialize actor and critic
initialize on-policy replay buffer

for each update:
    collect several rollouts with current policy
    store (state, action, reward, next_state, done)

    compute values and next_values with critic
    compute GAE advantages and returns
    update critic with MSE(value, returns)

    compute old log probability and old policy distribution
    compute policy gradient g
    define Hessian-vector product from mean KL
    solve Hx = g with conjugate gradient
    scale x to satisfy max_kl

    line search:
        try theta_old + step
        accept if surrogate improves and KL <= max_kl
        otherwise shrink step
```

## 3. 代码映射

实现位于 [`modules/trpo.py`](../modules/trpo.py)。

`Actor` 输出高斯策略的原始均值和标准差：

```python
mean = self.net(state)
std = self.log_std.exp().expand_as(mean)
```

采样时先从高斯分布得到 `raw_action`，再进行 `tanh` 和动作缩放：

```python
raw_action = torch.distributions.Normal(mean, std).sample()
action = torch.tanh(raw_action) * self.max_action
```

`ReplayBuffer` 只保存当前 on-policy 轨迹中的 `(state, action, reward, next_state, done)`。每次 `update()` 后会清空。

GAE 计算位于：

```python
advantages, returns = self._gae(rewards, dones, values, next_values)
```

TRPO 的 surrogate objective 位于：

```python
log_probs = self._log_prob(states, actions)
ratio = torch.exp(log_probs - old_log_probs)
return (ratio * advantages).mean()
```

`_log_prob()` 先对环境动作除以 `max_action`，再使用 `atanh` 还原高斯空间中的动作，并减去 `tanh` 的 Jacobian 项。

平均 KL 位于：

```python
kl_divergence(old_dist, new_dist).sum(dim=-1).mean()
```

Hessian-vector product 位于 `_hessian_vector_product()`，共轭梯度求解位于 `_conjugate_gradient()`，线搜索直接写在 `update()` 中。

测试脚本位于 [`tests/test_trpo.py`](../tests/test_trpo.py)，环境是 `Pendulum-v1`。测试每次收集 10 个完整回合后更新一次策略，共训练 1000 个回合；critic 使用较多步数拟合当前 batch 的回报。最后使用确定性动作进行 10 个回合的评估，并保存奖励曲线和第一个评估回合的 MP4 动态演示。

## 4. 参考

Schulman et al., *Trust Region Policy Optimization*.
