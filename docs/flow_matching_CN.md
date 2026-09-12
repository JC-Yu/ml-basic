<div align="right">
  <a href="flow_matching.md">English</a> |
  <a href="flow_matching_CN.md">简体中文</a>
</div>

# Flow Matching

Flow Matching 的目标不是直接建模密度，而是学习一个随时间变化的速度场，把简单先验分布推到目标数据分布。

## 1. 数学原理

设源样本 $x_0 \sim p_0$，目标样本 $x_1 \sim p_1$。对任意时间 $t \in [0, 1]$，做线性插值：

$$
x_t = (1 - t)x_0 + t x_1
$$

这条直线路径的速度是常数：

$$
u_t = \frac{d x_t}{d t} = x_1 - x_0
$$

模型学习一个条件向量场 $v_\theta(x_t, t)$，让它去拟合这个速度：

$$
L = \mathbb{E}\left[\|v_\theta(x_t, t) - u_t\|^2\right]
$$

采样时，从高斯先验出发，沿着学到的速度场做 Euler 积分：

$$
x_{k+1} = x_k + v_\theta(x_k, t_k)\Delta t
$$

## 2. 伪代码

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

## 3. 代码映射

实现位于 [`modules/flow_matching.py`](../modules/flow_matching.py)。

`FlowField` 是一个把 `x` 和 `t` 拼接后输入的 MLP。`FlowMatching.update()` 直接构造线性插值点和目标速度。`sample_trajectory()` 用固定步长的 Euler 法生成整条轨迹，`sample()` 只取最后一步。

测试脚本位于 [`tests/test_flow_matching.py`](../tests/test_flow_matching.py)，数据是二维 8-Gaussians，训练后会保存速度场采样结果和动态轨迹。

## 4. 参考

Lipman et al., *Flow Matching for Generative Modeling*.
