# ACT：Action Chunking and Temporal Ensembling

这里的实现用于直观展示机器人模仿学习中的两个核心机制：

1. **Action Chunking**：策略一次预测未来一段连续动作，而不是只预测当前时刻的一个动作。
2. **Temporal Ensembling**：将不同时间预测得到、但对应当前时刻的多个动作进行加权平均，从而减小动作抖动。

这是一个面向学习和阅读的简化实现。策略网络使用 MLP，不包含论文 ACT 中更完整的 Transformer、VAE 和图像编码器结构，重点放在动作块预测和时间集成本身。

## 1. 数学原理

### 1.1 Action Chunking

给定当前观测 $o_t$，策略不只输出当前动作，而是输出长度为 $K$ 的动作块：

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

其中 $\hat{a}_{t}^{(i)}$ 表示在时刻 $t$ 进行预测时，对未来第 $i$ 步动作的预测。

演示数据中对应的目标动作块为：

$$
A_t =
\left[
a_t,
a_{t+1},
\dots,
a_{t+K-1}
\right]
$$

策略通过监督学习拟合演示动作块，损失函数为：

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

代码中策略输出先经过 `tanh`，再乘以 `action_scale`：

$$
\hat{A}_t
=
s\cdot\tanh\left(\operatorname{MLP}_\theta(o_t)\right)
$$

这样可以将每个动作限制在 $[-s,s]$ 范围内。

### 1.2 Temporal Ensembling

在实际执行过程中，每个时间步都会重新预测一个动作块。因此，多个历史动作块可能同时包含当前时刻 $t$ 的动作预测。

假设动作块在时刻 $s$ 生成，则它对时刻 $t$ 的预测是：

$$
\hat{a}_{s,t-s}
$$

其中 $t-s$ 是当前时刻在该动作块中的偏移量。只保留仍然覆盖当前时刻的动作块，并使用指数衰减权重：

$$
w_{s,t}=\lambda^{t-s}
$$

最终执行动作是加权平均：

$$
a_t=
\frac{
\sum_s w_{s,t}\hat{a}_{s,t-s}
}{
\sum_s w_{s,t}
}
$$

其中 $\lambda$ 对应代码中的 `decay`。当前时刻新预测的动作权重为 $1$，更早动作块中的预测权重逐步减小。

这种方式可以让多个时间步的预测相互约束，通常比每次只执行最新动作块的第一个动作更加平滑。

## 2. 伪代码

```text
构造演示数据：
    对每个演示轨迹：
        在每个时刻保存观测 obs_t
        保存未来 K 步动作 [a_t, ..., a_{t+K-1}]

训练：
    重复若干次：
        随机采样一批 obs 和动作块 target_chunk
        predicted_chunk = policy(obs)
        loss = MSE(predicted_chunk, target_chunk)
        更新策略网络

执行：
    初始化 temporal ensembler
    对每个时刻 t：
        predicted_chunk = policy(obs_t)

        如果不使用 temporal ensembling：
            执行 predicted_chunk[0]
        否则：
            将 predicted_chunk 放入历史
            取所有仍覆盖当前时刻的动作预测
            按 decay 的幂次进行加权平均
            执行加权平均后的动作
```

## 3. 代码映射

### 3.1 `ActionChunkPolicy`

实现位于 [`modules/act.py`](../modules/act.py)。

`ActionChunkPolicy` 是动作块预测网络：

- 输入：当前观测，形状为 `[B, obs_dim]`
- 输出：动作块，形状为 `[B, chunk_len, action_dim]`
- 网络：两层隐藏层的 MLP
- 输出处理：`tanh` 后乘以 `action_scale`

核心计算可以概括为：

```python
chunk = self.net(obs).view(-1, chunk_len, action_dim)
return action_scale * torch.tanh(chunk)
```

### 3.2 `TemporalEnsembler`

`TemporalEnsembler` 使用 `history` 保存最近产生的动作块，并在 `combine()` 中完成时间集成：

1. 保存当前时刻产生的动作块；
2. 找出仍然覆盖当前时刻的历史动作块；
3. 根据动作块中的偏移量计算 `decay ** offset`；
4. 对动作候选进行加权平均；
5. 删除已经完全过期的动作块。

它只负责动作块的时间组合，不参与神经网络训练。

### 3.3 `ACTAgent`

`ACTAgent` 将策略网络、优化器和时间集成器组合在一起：

- `predict_chunk(obs)`：根据观测预测动作块；
- `select_action(obs, ensemble=True)`：预测动作块并返回当前要执行的动作；
- `update(obs_batch, chunk_batch)`：使用 MSE 进行模仿学习更新；
- `reset()`：清空时间集成器的历史状态。

训练阶段只调用 `update()`。执行阶段可以通过 `ensemble=False` 对比只执行动作块第一个动作的结果，也可以通过 `ensemble=True` 使用 Temporal Ensembling。

### 3.4 测试数据和可视化

测试脚本位于 [`tests/test_act.py`](../tests/test_act.py)，使用一个二维周期轨迹作为简单的机器人模仿学习任务：

- `reference_path()`：生成二维目标轨迹；
- `make_obs()`：将机器人位置、目标位置和周期信息组成观测；
- `build_dataset()`：生成观测和未来动作块组成的训练数据；
- `rollout()`：分别测试只使用 Action Chunking 和同时使用 Temporal Ensembling 的执行效果；
- `save_path_svg()`：保存最终轨迹对比图；
- `save_video_mp4()`：保存逐步执行的动态演示。

测试中使用的主要参数为：

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

## 4. 参考

Zhao et al., *Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware*.
