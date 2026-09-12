<div align="right">
  <a href="act.md">English</a> |
  <a href="act_CN.md">简体中文</a>
</div>

# ACT: Action Chunking and Temporal Ensembling

This implementation is designed to clearly demonstrate two core mechanisms in robot imitation learning:

1. **Action Chunking**: The policy predicts a sequence of future actions instead of only predicting the action at the current time step.
2. **Temporal Ensembling**: Multiple predictions for the current time step, generated at different time steps, are combined with a weighted average to reduce action jitter.

This is a simplified implementation intended for learning and reading. The policy uses an MLP and does not include the complete Transformer, VAE, or image encoder structure from the original ACT work. The focus is on action chunk prediction and temporal ensembling.

## 1. Mathematical Principles

### 1.1 Action Chunking

Given the current observation $o_t$, the policy predicts an action chunk of length $K$:

$$
\hat{A}_t = \pi_\theta(o_t)
=
\left[
\hat{a}_{t}^{(0)},
\hat{a}_{t}^{(1)},
\dots,
\hat{a}_{t}^{(K-1)}
\right]
$$

Here, $\hat{a}_{t}^{(i)}$ is the prediction for the action at future step $i$, made from the observation at time $t$.

The corresponding target action chunk from demonstrations is:

$$
A_t =
\left[
a_t,
a_{t+1},
\dots,
a_{t+K-1}
\right]
$$

The policy is trained with supervised imitation learning:

$$
\mathcal{L}(\theta)
=
\mathbb{E}
\left[
\left\|
\pi_\theta(o_t)-A_t
\right\|_2^2
\right]
$$

The code applies `tanh` to the policy output and then multiplies it by `action_scale`:

$$
\hat{A}_t
=
s\cdot\tanh\left(\operatorname{MLP}_\theta(o_t)\right)
$$

This limits each action to the range $[-s,s]$.

### 1.2 Temporal Ensembling

During execution, a new action chunk is predicted at every time step. Therefore, multiple historical action chunks may contain predictions for the current time step $t$.

If an action chunk is generated at time $s$, its prediction for time $t$ is:

$$
\hat{a}_{s,t-s}
$$

where $t-s$ is the offset of the current time step within that action chunk. Only action chunks that still cover the current time step are kept. Their weights are defined with exponential decay:

$$
w_{s,t}=\lambda^{t-s}
$$

The executed action is the weighted average:

$$
a_t=
\frac{
\sum_s w_{s,t}\hat{a}_{s,t-s}
}{
\sum_s w_{s,t}
}
$$

Here, $\lambda$ corresponds to `decay` in the code. The newest prediction has weight $1$, while predictions from older chunks receive smaller weights.

This allows predictions from multiple time steps to constrain each other and usually produces smoother actions than executing only the first action of the newest chunk.

## 2. Pseudocode

```text
Build demonstrations:
    for each demonstration trajectory:
        save observation obs_t at every time step
        save future action chunk [a_t, ..., a_{t+K-1}]

Training:
    repeat for several steps:
        sample a batch of observations and target action chunks
        predicted_chunk = policy(obs)
        loss = MSE(predicted_chunk, target_chunk)
        update the policy network

Execution:
    initialize the temporal ensembler
    for each time step t:
        predicted_chunk = policy(obs_t)

        if temporal ensembling is disabled:
            execute predicted_chunk[0]
        else:
            add predicted_chunk to the history
            collect predictions from all valid overlapping chunks
            compute their weighted average using powers of decay
            execute the weighted average action
```

## 3. Code Mapping

### 3.1 `ActionChunkPolicy`

The implementation is in [`modules/act.py`](../modules/act.py).

`ActionChunkPolicy` predicts an action chunk:

- Input: current observation with shape `[B, obs_dim]`
- Output: action chunk with shape `[B, chunk_len, action_dim]`
- Network: an MLP with two hidden layers
- Output processing: `tanh` followed by multiplication by `action_scale`

The main computation is:

```python
chunk = self.net(obs).view(-1, chunk_len, action_dim)
return action_scale * torch.tanh(chunk)
```

### 3.2 `TemporalEnsembler`

`TemporalEnsembler` stores recently generated action chunks in `history` and performs temporal ensembling in `combine()`:

1. Store the action chunk generated at the current time step.
2. Find historical chunks that still cover the current time step.
3. Compute `decay ** offset` for each valid prediction.
4. Compute the weighted average of the action candidates.
5. Remove action chunks that have expired.

It only combines action chunks and does not participate in neural network training.

### 3.3 `ACTAgent`

`ACTAgent` combines the policy network, optimizer, and temporal ensembler:

- `predict_chunk(obs)`: predict an action chunk from an observation;
- `select_action(obs, ensemble=True)`: predict a chunk and return the action to execute;
- `update(obs_batch, chunk_batch)`: perform an MSE imitation learning update;
- `reset()`: clear the temporal ensembler history.

During training, only `update()` is needed. During execution, `ensemble=False` can be used to execute the first action of each chunk, while `ensemble=True` enables temporal ensembling.

### 3.4 Test Data and Visualization

The test script is [`tests/test_act.py`](../tests/test_act.py). It uses a two-dimensional periodic trajectory as a simple robot imitation learning task:

- `reference_path()`: generate the two-dimensional target trajectory;
- `make_obs()`: construct an observation from the robot position, target position, and phase information;
- `build_dataset()`: generate observations and future action chunks;
- `rollout()`: compare Action Chunking alone with Action Chunking plus Temporal Ensembling;
- `save_path_svg()`: save the final trajectory comparison;
- `save_video_mp4()`: save a step-by-step dynamic demonstration.

The main test settings are:

```text
obs_dim = 6
action_dim = 2
chunk_len = 8
hidden_dim = 128
action_scale = 0.08
ensemble_decay = 0.78
train_steps = 1000
batch_size = 128
```

## 4. Reference

Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*.
