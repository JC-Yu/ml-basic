<div align="right">
  <a href="ddpg.md">English</a> |
  <a href="ddpg_CN.md">简体中文</a>
</div>

# DDPG

DDPG is a deterministic actor-critic algorithm for continuous action spaces. It learns an actor that produces actions and a critic that evaluates their values.

## 1. Mathematical Principles

Let the actor be $\mu_\theta(s)$ and the critic be $Q_\phi(s, a)$.

The target value is computed with the target actor and target critic:

$$
y = r + \gamma (1 - d) Q_{\phi'}(s', \mu_{\theta'}(s'))
$$

The critic is trained to fit this target with mean squared error:

$$
L_{\text{critic}} = \left(Q_\phi(s, a) - y\right)^2
$$

The actor is updated by maximizing the Q value estimated by the critic:

$$
L_{\text{actor}} = -Q_\phi(s, \mu_\theta(s))
$$

The target networks are updated with soft updates:

$$
\theta' \leftarrow (1 - \tau)\theta' + \tau \theta
$$

$$
\phi' \leftarrow (1 - \tau)\phi' + \tau \phi
$$

## 2. Pseudocode

```text
initialize actor, critic, target actor, and target critic
initialize replay buffer

for each environment step:
    a = actor(s) + exploration noise
    store (s, a, r, s', done)

    if the replay buffer is ready:
        sample a minibatch
        y = r + gamma * (1 - done) * critic_target(s', actor_target(s'))
        minimize critic loss
        minimize actor loss = -critic(s, actor(s))
        softly update target networks
```

## 3. Code Mapping

The implementation is in [`modules/ddpg.py`](../modules/ddpg.py).

`Actor` directly outputs continuous actions, followed by `tanh` and scaling by the action limit. `Critic` concatenates the state and action and regresses a Q value. `DDPGAgent.select_action()` adds Gaussian noise to the actor output. In the test script, the noise scale is linearly reduced from `0.3` to `0.05`.

The core of `DDPGAgent.update()` is to update the critic first, then update the actor, and finally perform the soft updates.

The test script is [`tests/test_ddpg.py`](../tests/test_ddpg.py), using the `Pendulum-v1` environment.

## 4. Reference

Lillicrap et al., *Continuous control with deep reinforcement learning*.
