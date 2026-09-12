# Diffusion

这里的实现是一个最小化的扩散模型：先把干净样本加噪，再训练网络去预测噪声，最后反向逐步去噪采样。

## 1. 数学原理

给定线性噪声日程 $\beta_1, \dots, \beta_T$，定义：

$$
\alpha_t = 1 - \beta_t
$$

$$
\bar{\alpha}_t = \prod_{i=1}^t \alpha_i
$$

前向加噪过程写成：

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t}\,\epsilon
$$

模型学习噪声预测器 $\epsilon_\theta(x_t, t)$：

$$
L = \mathbb{E}\left[\|\epsilon_\theta(x_t, t) - \epsilon\|^2\right]
$$

反向采样时，从纯噪声开始，逐步估计 $x_{t-1}$。本实现用的是常见的简化反推公式：

$$
x_{t-1} = \frac{x_t - \beta_t \epsilon_\theta(x_t, t)/\sqrt{1-\bar{\alpha}_t}}{\sqrt{\alpha_t}}
$$

如果 $t > 0$，再补一点随机噪声。

## 2. 伪代码

```text
sample clean data x0
sample timestep t
add noise to get xt
predict epsilon with network
minimize mse(epsilon_theta(xt, t), epsilon)

for sampling:
    x_T ~ N(0, I)
    for t = T - 1 ... 0:
        predict epsilon
        denoise x_t to x_{t-1}
        if t > 0: add fresh noise
```

## 3. 代码映射

实现位于 [`modules/diffusion.py`](../modules/diffusion.py)。

`NoisePredictor` 是一个接收 `x` 和归一化时间 `t` 的 MLP。`Diffusion` 预先构造 `betas`、`alphas` 和 `alpha_bars`。`add_noise()` 负责前向扩散，`update()` 负责噪声回归，`sample_trajectory()` 负责反向去噪过程。

测试脚本位于 [`tests/test_diffusion.py`](../tests/test_diffusion.py)，数据同样是二维 8-Gaussians。

## 4. 参考

Ho et al., *Denoising Diffusion Probabilistic Models*.
