<div align="right">
  <a href="diffusion.md">English</a> |
  <a href="diffusion_CN.md">简体中文</a>
</div>

# Diffusion

This implementation is a minimal diffusion model: it first adds noise to clean samples, trains a network to predict the noise, and then gradually denoises samples in the reverse process.

## 1. Mathematical Principles

Given a linear noise schedule $\beta_1, \dots, \beta_T$, define:

$$
\alpha_t = 1 - \beta_t
$$

$$
\bar{\alpha}_t = \prod_{i=1}^t \alpha_i
$$

The forward noising process is:

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon
$$

The model learns a noise predictor $\epsilon_\theta(x_t, t)$:

$$
L = \mathbb{E}\left[\|\epsilon_\theta(x_t, t) - \epsilon\|^2\right]
$$

During reverse sampling, the process starts from pure noise and estimates $x_{t-1}$ step by step. This implementation uses a common simplified reverse formula:

$$
x_{t-1} = \frac{x_t - \beta_t \epsilon_\theta(x_t, t)/\sqrt{1-\bar{\alpha}_t}}{\sqrt{\alpha_t}}
$$

Additional random noise is added when $t > 0$.

## 2. Pseudocode

```text
sample clean data x0
sample timestep t
add noise to get xt
predict epsilon with the network
minimize mse(epsilon_theta(xt, t), epsilon)

for sampling:
    x_T ~ N(0, I)
    for t = T - 1 ... 0:
        predict epsilon
        denoise x_t to x_{t-1}
        if t > 0: add fresh noise
```

## 3. Code Mapping

The implementation is in [`modules/diffusion.py`](../modules/diffusion.py).

`NoisePredictor` is an MLP that receives `x` and the normalized time `t`. `Diffusion` precomputes `betas`, `alphas`, and `alpha_bars`. `add_noise()` implements the forward diffusion process, `update()` performs noise regression, and `sample_trajectory()` performs the reverse denoising process.

The test script is [`tests/test_diffusion.py`](../tests/test_diffusion.py), using a two-dimensional 8-Gaussians dataset.

## 4. Reference

Ho et al., *Denoising Diffusion Probabilistic Models*.
