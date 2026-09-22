<div align="right">
  <a href="reinforce.md">English</a> |
  <a href="reinforce_CN.md">简体中文</a>
</div>

# REINFORCE

REINFORCE 是最基础的策略梯度算法。它直接学习一个策略 $\pi_\theta(a|s)$，通过完整回合得到的实际回报来调整策略，使高回报动作的概率增大，使低回报动作的概率减小。

与 DQN、DDPG 等离策略算法不同，REINFORCE 不使用经验回放和目标网络，而是收集完整回合后进行一次更新。

## 1. 数学原理

策略 $\pi_\theta(a|s)$ 表示在状态 $s$ 下选择动作 $a$ 的概率。一个回合中的折扣回报定义为：

$$
G_t = \sum_{k=t}^{T-1}\gamma^{k-t}r_k
$$

其中 $\gamma$ 是折扣因子，$T$ 是当前回合结束的时间步。

REINFORCE 的目标是最大化期望回报：

$$
J(\theta)=\mathbb{E}_{\tau\sim\pi_\theta}
\left[
\sum_{t=0}^{T-1}\gamma^t r_t
\right]
$$

使用策略梯度定理，可以得到：

$$
\nabla_\theta J(\theta)
=
\mathbb{E}
\left[
\sum_{t=0}^{T-1}
\nabla_\theta\log\pi_\theta(a_t|s_t)G_t
\right]
$$

因此，将需要最小化的损失写成：

$$
L(\theta)
=
-\sum_{t=0}^{T-1}
\log\pi_\theta(a_t|s_t)G_t
$$

当某个动作带来了较高的后续回报时，$G_t$ 较大，梯度会提高该动作在对应状态下的概率。反之，较低的回报会降低该动作的概率。

## 2. 伪代码

```text
initialize policy network pi_theta
initialize optimizer

for each episode:
    clear log_probs and rewards
    state = environment.reset()

    while episode is not finished:
        sample action from pi_theta(. | state)
        save log pi_theta(action | state)
        execute action
        save reward
        state = next_state

    calculate returns G_t from the end of the episode
    loss = -sum(log_probs_t * G_t)
    update policy network
```

## 3. 代码映射

实现位于 [`modules/reinforce.py`](../modules/reinforce.py)。

### 3.1 `PolicyNetwork`

`PolicyNetwork` 是一个简单的 MLP，输入状态，输出每个离散动作对应的 logits：

```python
logits = self.net(state)
```

logits 不需要先手动执行 `softmax`。`REINFORCEAgent` 将其传给：

```python
distribution = torch.distributions.Categorical(logits=logits)
```

`Categorical` 会根据 logits 构造离散动作分布。

### 3.2 `REINFORCEAgent.select_action()`

训练时，`select_action()` 从当前策略分布中采样动作，并保存该动作的对数概率：

```python
action = distribution.sample()
self.log_probs.append(distribution.log_prob(action).squeeze(0))
```

保存对数概率是为了在回合结束后构造策略梯度损失。测试阶段传入 `deterministic=True`，直接选择 logits 最大的动作，不再保存对数概率。

### 3.3 `REINFORCEAgent.update()`

`update()` 首先从回合末尾向前计算折扣回报：

```python
total_return = 0.0
for reward in reversed(self.rewards):
    total_return = reward + gamma * total_return
```

然后按照 REINFORCE 的目标构造损失并更新策略：

```python
loss = -(log_probs * returns).sum()
```

更新完成后清空当前回合的 `log_probs` 和 `rewards`，下一回合重新收集数据。

本实现没有加入 baseline、优势归一化、熵正则、经验回放或梯度裁剪，便于直接观察原始 REINFORCE 的核心逻辑。

## 4. 参考

Williams, *Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning*.
