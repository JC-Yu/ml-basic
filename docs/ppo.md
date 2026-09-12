<div align="right">
  <a href="ppo.md">English</a> |
  <a href="ppo_CN.md">简体中文</a>
</div>

# PPO

PPO is a stable on-policy actor-critic method. It collects data with an old policy and uses a clipped objective to limit the difference between the new and old policies, preventing excessively large updates.

## 1. Mathematical Principles

Let the policy be $\pi_\theta(a|s)$ and the value function be $V_\phi(s)$. This implementation uses a diagonal Gaussian policy:

$$
\mu_\theta(s) = a_{\max}\tanh(f_\mu(s)), \qquad
\sigma_\theta(s) = \operatorname{softplus}(f_\sigma(s))
$$

The policy distribution is:

$$
\pi_\theta(a|s) = \mathcal{N}(a;\mu_\theta(s), \sigma_\theta(s))
$$

The policy ratio is:

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

The GAE advantage is computed as:

$$
\delta_t = r_t + \gamma V_\phi(s_{t+1})(1-d_t) - V_\phi(s_t)
$$

$$
A_t = \delta_t + \gamma \lambda (1-d_t)A_{t+1}
$$

The clipped policy loss is:

$$
L^{\text{clip}} = \mathbb{E}\left[\min(\rho_tA_t, \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon)A_t)\right]
$$

The value loss is:

$$
L_V = \mathbb{E}\left[(V_\phi(s_t) - R_t)^2\right],
\qquad R_t = A_t + V_\phi(s_t)
$$

## 2. Pseudocode

```text
initialize actor_new, actor_old, and critic
copy actor_new to actor_old
initialize replay buffer

for each episode:
    use actor_old to collect transitions (s, a, r, s', done)
    store them in the replay buffer

    compute values and next values with the critic
    compute GAE advantages and returns

    for several epochs:
        sample minibatches from the buffer
        compute ratio = pi_new(a|s) / pi_old(a|s)
        update actor_new with the clipped objective
        update critic with MSE loss

    copy actor_new to actor_old
    clear the replay buffer
```

## 3. Code Mapping

The implementation is in [`modules/ppo.py`](../modules/ppo.py).

`Actor` uses two heads to output the mean and standard deviation. The mean head uses `tanh`, and the standard deviation head uses `softplus`. `ReplayBuffer` only stores `(state, action, reward, next_state, done)` instead of storing log probabilities and values. `PPOAgent` computes GAE and the old/new log probabilities inside `update()`, then optimizes the actor and critic separately.

The core code corresponds to:

```python
old_mean, old_std = self.actor_old(states)
old_dist = torch.distributions.Normal(old_mean, old_std)
old_log_probs = old_dist.log_prob(actions).sum(dim=-1)

mean, std = self.actor_new(states[mb])
dist = torch.distributions.Normal(mean, std)
new_log_probs = dist.log_prob(actions[mb]).sum(dim=-1)
```

`actor_old` is used for sampling and for evaluating the old policy probability. `actor_new` is optimized during training. After an update, `actor_old` is synchronized directly with `actor_new`. For bounded environments, the environment may clip the executed action, but PPO still computes the policy probability for the action sampled from the policy.

After GAE is computed, this implementation standardizes the advantages for the current rollout before using them in the PPO loss. Advantage standardization is not part of the definition of GAE, but it is a common stabilization step in continuous-control PPO implementations.

The test script is [`tests/test_ppo.py`](../tests/test_ppo.py), using the `Pendulum-v1` environment.

## 4. Reference

Schulman et al., *Proximal Policy Optimization Algorithms*.
