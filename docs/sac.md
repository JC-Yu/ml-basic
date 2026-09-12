<div align="right">
  <a href="sac.md">English</a> |
  <a href="sac_CN.md">简体中文</a>
</div>

# SAC

SAC is an off-policy actor-critic algorithm for continuous action spaces. Its central idea is to optimize both high return and high entropy, allowing the policy to learn useful behavior while maintaining sufficient exploration.

## 1. Mathematical Principles

SAC uses a stochastic policy $\pi_\theta(a|s)$. In this implementation, it is a squashed Gaussian policy: the actor outputs a mean and standard deviation, the `SACAgent` samples a Gaussian action, and `tanh` maps it into the action range.

The policy objective is:

$$
J_\pi = \mathbb{E}\left[\min(Q_1(s, a), Q_2(s, a)) - \alpha \log \pi_\theta(a|s)\right]
$$

Here, $\alpha$ is the entropy coefficient. For simplicity, it is fixed as a constant instead of being automatically tuned.

The double-Q target is:

$$
y = r + \gamma (1 - d)\left(\min(Q_1'(s', a'), Q_2'(s', a')) - \alpha \log \pi_\theta(a'|s')\right)
$$

The critics fit this target with mean squared error:

$$
L_Q = \left(Q_1(s, a) - y\right)^2 + \left(Q_2(s, a) - y\right)^2
$$

The target networks use soft updates:

$$
\theta' \leftarrow (1 - \tau)\theta' + \tau \theta
$$

## 2. Pseudocode

```text
initialize actor, two critics, and two target critics
initialize replay buffer

for each environment step:
    sample an action from the stochastic actor
    store transition (s, a, r, s', done)

    if the replay buffer is ready:
        sample a minibatch
        a' ~ pi(s')
        y = r + gamma * (1 - done) * (min(Q1', Q2') - alpha * log pi(a'|s'))
        minimize the critic loss

        a ~ pi(s)
        maximize min(Q1, Q2) - alpha * log pi(a|s)

        softly update the target critics
```

## 3. Code Mapping

The implementation is in [`modules/sac.py`](../modules/sac.py).

`Actor` only outputs the mean and standard deviation. The mean head uses `tanh`, and the standard deviation head uses `softplus`. Like DDPG, `Critic` directly concatenates the state and action and regresses a Q value. `SACAgent` maintains two critics and two target critics.

The most important part of `SACAgent.update()` is:

```python
next_actions, next_log_probs = self._sample_action(next_states)
target_q = torch.min(
    self.target_critic1(next_states, next_actions),
    self.target_critic2(next_states, next_actions),
).squeeze(1) - self.alpha * next_log_probs
```

This is the core of SAC: double-Q estimation reduces overestimation, while the entropy term encourages exploration.

The test script is [`tests/test_sac.py`](../tests/test_sac.py), using the `Pendulum-v1` environment.

## 4. Reference

Haarnoja et al., *Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor*.
