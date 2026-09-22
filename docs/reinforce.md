<div align="right">
  <a href="reinforce.md">English</a> |
  <a href="reinforce_CN.md">简体中文</a>
</div>

# REINFORCE

REINFORCE is one of the most basic policy-gradient algorithms. It directly learns a policy $\pi_\theta(a|s)$ and adjusts the policy according to the return collected from a complete episode. Actions that lead to higher returns become more likely, while actions that lead to lower returns become less likely.

Unlike off-policy algorithms such as DQN and DDPG, REINFORCE does not use experience replay or a target network. It collects a complete episode and then performs one policy update.

## 1. Mathematical Principles

The policy $\pi_\theta(a|s)$ represents the probability of selecting action $a$ in state $s$. The discounted return at time step $t$ is:

$$
G_t = \sum_{k=t}^{T-1}\gamma^{k-t}r_k
$$

Here, $\gamma$ is the discount factor and $T$ is the terminal time step of the episode.

REINFORCE maximizes the expected return:

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T-1}\gamma^t r_t
\right]
$$

Using the policy-gradient theorem:

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t|s_t)G_t
\right]
$$

Therefore, the loss to minimize is:

$$
L(\theta)
=
-\sum_{t=0}^{T-1}
\log\pi_\theta(a_t|s_t)G_t
$$

When an action produces a high future return, $G_t$ is large and the gradient increases the probability of that action in the corresponding state. A lower return has the opposite effect.

## 2. Pseudocode

```text
initialize policy network pi_theta
initialize optimizer

for each episode:
    clear log_probs and rewards
    state = environment.reset()

    while the episode is not finished:
        sample action from pi_theta(. | state)
        save log pi_theta(action | state)
        execute action
        save reward
        state = next_state

    calculate returns G_t from the end of the episode
    loss = -sum(log_probs_t * G_t)
    update the policy network
```

## 3. Code Mapping

The implementation is in [`modules/reinforce.py`](../modules/reinforce.py).

### 3.1 `PolicyNetwork`

`PolicyNetwork` is a simple MLP. It receives a state and outputs one logit for each discrete action:

```python
logits = self.net(state)
```

The logits are passed directly to:

```python
distribution = torch.distributions.Categorical(logits=logits)
```

`Categorical` constructs the discrete action distribution from the logits.

### 3.2 `REINFORCEAgent.select_action()`

During training, `select_action()` samples an action from the current policy and saves the log probability of that action:

```python
action = distribution.sample()
self.log_probs.append(distribution.log_prob(action).squeeze(0))
```

The saved log probabilities are used to construct the policy-gradient loss after the episode ends. During evaluation, `deterministic=True` selects the action with the largest logit and does not save a log probability.

### 3.3 `REINFORCEAgent.update()`

`update()` calculates discounted returns from the end of the episode:

```python
total_return = 0.0
for reward in reversed(self.rewards):
    total_return = reward + gamma * total_return
```

It then builds the REINFORCE loss and updates the policy:

```python
loss = -(log_probs * returns).sum()
```

After the update, the current episode's `log_probs` and `rewards` are cleared so that the next episode can be collected.

This implementation intentionally does not include a baseline, advantage normalization, entropy regularization, experience replay, or gradient clipping. This keeps the original policy-gradient logic visible.

## 4. Reference

Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*.
