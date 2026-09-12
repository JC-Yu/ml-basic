<div align="right">
  <a href="flow_matching.md">English</a> |
  <a href="flow_matching_CN.md">简体中文</a>
</div>

# Flow Matching

Flow Matching does not model the data density directly. Instead, it learns a time-dependent velocity field that transports a simple prior distribution to the target data distribution.

## 1. Mathematical Principles

Let the source sample be $x_0 \sim p_0$ and the target sample be $x_1 \sim p_1$. For any time $t \in [0, 1]$, use linear interpolation:

$$
x_t = (1 - t)x_0 + t x_1
$$

The velocity along this straight path is constant:

$$
u_t = \frac{d x_t}{d t} = x_1 - x_0
$$

The model learns a conditional vector field $v_\theta(x_t, t)$ to fit this velocity:

$$
L = \mathbb{E}\left[\|v_\theta(x_t, t) - u_t\|^2\right]
$$

During sampling, start from a Gaussian prior and integrate the learned velocity field with Euler integration:

$$
x_{k+1} = x_k + v_\theta(x_k, t_k)\Delta t
$$

## 2. Pseudocode

```text
sample source batch x0 and target batch x1
sample t ~ Uniform(0, 1)
x_t = (1 - t) * x0 + t * x1
u_t = x1 - x0
minimize mse(v_theta(x_t, t), u_t)

for sampling:
    x ~ N(0, I)
    for k = 0 ... steps - 1:
        x = x + v_theta(x, t_k) * dt
```

## 3. Code Mapping

The implementation is in [`modules/flow_matching.py`](../modules/flow_matching.py).

`FlowField` is an MLP that concatenates `x` and `t` before processing them. `FlowMatching.update()` directly constructs the linear interpolation point and target velocity. `sample_trajectory()` generates the full trajectory with fixed-step Euler integration, while `sample()` returns only the final step.

The test script is [`tests/test_flow_matching.py`](../tests/test_flow_matching.py). It uses a two-dimensional 8-Gaussians dataset and saves the learned velocity-field samples and a dynamic trajectory after training.

## 4. Reference

Lipman et al., *Flow Matching for Generative Modeling*.
