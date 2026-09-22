<div align="right">
  <a href="td3.md">English</a> |
  <a href="td3_CN.md">简体中文</a>
</div>

# TD3

TD3, short for Twin Delayed Deep Deterministic Policy Gradient, is an improved version of DDPG. It is still a deterministic actor-critic algorithm for continuous action spaces, but it adds three simple and important stabilization ideas:

1. Use two critics and take the smaller Q value;
2. Add clipped noise to the target action for target policy smoothing;
3. Update the critics every step, but update the actor and target networks less frequently.

## 1. Mathematical Principles

Let the actor be $\mu_\theta(s)$, and let the two critics be $Q_{\phi_1}(s,a)$ and $Q_{\phi_2}(s,a)$.

### 1.1 Clipped Double Q

TD3 estimates the next state-action value with two target critics and takes the smaller value:

$$
\bar{Q}(s', a')
=
\min
\left(
Q_{\phi_1'}(s', a'),
Q_{\phi_2'}(s', a')
\right)
$$

This reduces Q-value overestimation.

### 1.2 Target Policy Smoothing

The target action is not only the output of the target actor. TD3 adds clipped noise:

$$
a' =
\operatorname{clip}
\left(
\mu_{\theta'}(s') + \epsilon,
-a_{\max},
a_{\max}
\right)
$$

where:

$$
\epsilon \sim
\operatorname{clip}
\left(
\mathcal{N}(0, \sigma),
-c,
c
\right)
$$

This prevents the critic from depending too strongly on sharp action-value estimates at a single action point.

### 1.3 Critic Target

The TD3 Bellman target is:

$$
y =
r + \gamma(1-d)
\min
\left(
Q_{\phi_1'}(s', a'),
Q_{\phi_2'}(s', a')
\right)
$$

Both critics fit the same target with mean squared error:

$$
L_Q =
\left(Q_{\phi_1}(s,a)-y\right)^2
+
\left(Q_{\phi_2}(s,a)-y\right)^2
$$

### 1.4 Delayed Policy Update

The actor still maximizes the Q value estimated by the critic:

$$
L_\mu =
-Q_{\phi_1}(s, \mu_\theta(s))
$$

But the actor is not updated after every critic update. It is updated only once every `policy_delay` critic updates. The target networks are softly updated together with the delayed actor update:

$$
\theta' \leftarrow (1-\tau)\theta' + \tau\theta
$$

$$
\phi_i' \leftarrow (1-\tau)\phi_i' + \tau\phi_i
$$

## 2. Pseudocode

```text
initialize actor and actor_target
initialize critic1, critic2, critic1_target, and critic2_target
initialize replay buffer

for each environment step:
    action = actor(state) + exploration noise
    store (state, action, reward, next_state, done)

    if the replay buffer is ready:
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

## 3. Code Mapping

The implementation is in [`modules/td3.py`](../modules/td3.py).

`Actor` has the same structure as in DDPG. It receives a state and outputs a continuous action bounded by `tanh`:

```python
return self.max_action * torch.tanh(self.net(state))
```

`Critic` concatenates the state and action and regresses a Q value. TD3 keeps two critics:

```python
self.critic1 = Critic(...)
self.critic2 = Critic(...)
```

In `TD3Agent.update()`, target policy smoothing corresponds to:

```python
noise = torch.randn_like(actions) * self.policy_noise
noise = noise.clamp(-self.noise_clip, self.noise_clip)
next_actions = (self.actor_target(next_states) + noise).clamp(
    -self.max_action, self.max_action
)
```

Clipped double Q corresponds to:

```python
target_q = torch.min(target_q1, target_q2)
```

Delayed policy update corresponds to:

```python
if self.total_updates % self.policy_delay == 0:
    actor_loss = -self.critic1(states, self.actor(states)).mean()
```

The test script is [`tests/test_td3.py`](../tests/test_td3.py), using the `Pendulum-v1` environment. It saves a reward curve and an MP4 rollout of the final policy.

## 4. Reference

Fujimoto et al., *Addressing Function Approximation Error in Actor-Critic Methods*.
