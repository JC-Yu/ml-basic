<div align="right">
  <a href="trpo.md">English</a> |
  <a href="trpo_CN.md">简体中文</a>
</div>

# TRPO

TRPO, short for Trust Region Policy Optimization, is an on-policy actor-critic algorithm. Its core idea is to improve the policy objective while constraining the KL distance between the old and new policies, preventing the policy from changing too much in one update.

## 1. Mathematical Principles

Let the policy be $\pi_\theta(a|s)$ and the value function be $V_\phi(s)$. This implementation uses a bounded squashed Gaussian policy. The actor outputs the raw Gaussian mean and standard deviation:

$$
\mu_\theta(s)=f_\theta(s),
\qquad
\sigma_\theta=\exp(\log\sigma_\theta)
$$

First sample an unbounded action $u$ and then map it into the environment action range:

$$
u\sim\mathcal{N}(\mu_\theta(s),\sigma_\theta^2),
\qquad
a=a_{\max}\tanh(u)
$$

The policy probability includes the Jacobian correction of the `tanh` transformation:

$$
\log\pi_\theta(a|s)
=
\log\mathcal{N}(u;\mu_\theta(s),\sigma_\theta(s))
-
\log\left(
a_{\max}(1-\tanh^2(u))+\epsilon
\right)
$$

This keeps the executed bounded action consistent with the policy probability used by TRPO.

### 1.1 Advantage Estimation

This implementation uses GAE to estimate advantages:

$$
\delta_t =
r_t + \gamma V_\phi(s_{t+1})(1-d_t) - V_\phi(s_t)
$$

$$
A_t =
\delta_t + \gamma\lambda(1-d_t)A_{t+1}
$$

The return target is:

$$
R_t = A_t + V_\phi(s_t)
$$

The critic fits $R_t$ with mean squared error.

### 1.2 Constrained Policy Optimization

The TRPO policy update can be written as:

$$
\max_\theta
\mathbb{E}
\left[
\frac{\pi_\theta(a_t|s_t)}
{\pi_{\theta_{\text{old}}}(a_t|s_t)}
A_t
\right]
$$

while constraining the average KL divergence between the old and new policies:

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

Here, $\delta$ corresponds to `max_kl` in the code.

### 1.3 Natural Gradient Direction

TRPO uses a second-order approximation. Let $H$ be the Hessian of the KL constraint and $g$ be the policy-objective gradient. The update direction is approximately:

$$
x = H^{-1}g
$$

The implementation does not build the full Hessian explicitly. Instead, it uses Hessian-vector products and solves this linear system with conjugate gradient.

To satisfy the KL constraint, the step is scaled by:

$$
\alpha =
\sqrt{
\frac{2\delta}
{x^THx}
}
$$

The candidate update is:

$$
\theta_{\text{new}}
=
\theta_{\text{old}} + \alpha x
$$

If the candidate update does not satisfy the KL constraint or does not improve the surrogate objective, line search shrinks the step.

## 2. Pseudocode

```text
initialize actor and critic
initialize on-policy replay buffer

for each update:
    collect several rollouts with the current policy
    store (state, action, reward, next_state, done)

    compute values and next_values with the critic
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

## 3. Code Mapping

The implementation is in [`modules/trpo.py`](../modules/trpo.py).

`Actor` outputs the raw mean and standard deviation of a Gaussian policy:

```python
mean = self.net(state)
std = self.log_std.exp().expand_as(mean)
```

During sampling, a Gaussian action is passed through `tanh` and scaled:

```python
raw_action = torch.distributions.Normal(mean, std).sample()
action = torch.tanh(raw_action) * self.max_action
```

`ReplayBuffer` only stores `(state, action, reward, next_state, done)` from the current on-policy trajectory. It is cleared after every `update()`.

GAE is computed by:

```python
advantages, returns = self._gae(rewards, dones, values, next_values)
```

The TRPO surrogate objective is:

```python
log_probs = self._log_prob(states, actions)
ratio = torch.exp(log_probs - old_log_probs)
return (ratio * advantages).mean()
```

`_log_prob()` divides the environment action by `max_action`, maps it back to Gaussian space with `atanh`, and subtracts the Jacobian term of `tanh`.

The mean KL is:

```python
kl_divergence(old_dist, new_dist).sum(dim=-1).mean()
```

The Hessian-vector product is implemented in `_hessian_vector_product()`, conjugate gradient is implemented in `_conjugate_gradient()`, and line search is written directly inside `update()`.

The test script is [`tests/test_trpo.py`](../tests/test_trpo.py), using the `Pendulum-v1` environment. It collects 10 complete episodes per policy update and trains for 1000 episodes. The critic uses more fitting steps for each batch. The final deterministic policy is evaluated for 10 episodes; the reward curve and the first evaluation rollout are saved.

## 4. Reference

Schulman et al., *Trust Region Policy Optimization*.
