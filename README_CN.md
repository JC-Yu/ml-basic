<p>
  <a href="https://www.python.org/downloads/release/python-3100/"><img src="https://img.shields.io/badge/Python-3.10-blue.svg" alt="Python 3.10"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License: Apache 2.0"></a>
  <a href="pyproject.toml"><img src="https://img.shields.io/badge/Version-v0.1.0-orange.svg" alt="Version v0.1.0"></a>
  <a href="https://github.com/JC-Yu/ml-basic/stargazers"><img src="https://img.shields.io/github/stars/JC-Yu/ml-basic?style=social" alt="GitHub stars"></a>
  <span style="float: right;"><a href="README.md">English</a> | <a href="README_CN.md">简体中文</a></span>
</p>

# ml-basic

`ml-basic` 是一份面向 Robotics 的机器学习基础模块代码库。项目用尽可能简洁、直观、单文件的方式实现常见算法，直接进入算法核心逻辑。

当前代码覆盖了 Transformer、Deep Reinforcement Learning、Diffusion、Flow Matching、Action Chunking 等基础内容。每个模块尽量保持独立，测试脚本会给出最小可运行示例，并保存训练曲线和动态演示结果。

## 代码结构

```text
ml-basic/
├── modules/                  # 核心算法实现，每个文件尽量独立、简洁
│   ├── attention.py           # Attention 基础实现
│   ├── transformer.py         # Transformer 基础模块
│   ├── dqn.py                 # DQN / Double DQN 风格实现
│   ├── reinforce.py           # REINFORCE 策略梯度实现
│   ├── ddpg.py                # DDPG 连续控制实现
│   ├── td3.py                 # TD3 连续控制实现
│   ├── ppo.py                 # PPO 连续控制实现
│   ├── trpo.py                # TRPO 信赖域策略优化实现
│   ├── sac.py                 # SAC 连续控制实现
│   ├── diffusion.py           # 简易 diffusion 实现
│   ├── flow_matching.py       # 简易 flow matching 实现
│   └── act.py                 # Action Chunking / Temporal Ensembling 示例
├── tests/                    # 可直接运行的测试脚本
├── docs/                     # 算法说明文档
├── assets/                   # README 或文档可引用的静态资源
├── pyproject.toml            # 项目依赖配置
└── uv.lock                   # uv 锁文件
```

## 环境配置

项目使用 Python 3.10，并通过 `uv` 管理依赖。

```bash
cd ml_basic/
uv sync
```

如果已经创建好虚拟环境，也可以直接激活：

```bash
source .venv/bin/activate
```

## 使用方式

所有测试脚本都可以直接从项目根目录运行。

```bash
cd ml-basic
python tests/test_**.py
```

## 算法文档

更详细的算法解释可以参考 `docs/` 目录：

- [DQN](docs/dqn_CN.md)
- [REINFORCE](docs/reinforce_CN.md)
- [DDPG](docs/ddpg_CN.md)
- [TD3](docs/td3_CN.md)
- [PPO](docs/ppo_CN.md)
- [TRPO](docs/trpo_CN.md)
- [SAC](docs/sac_CN.md)
- [Flow Matching](docs/flow_matching_CN.md)
- [Diffusion](docs/diffusion_CN.md)
- [ACT](docs/act_CN.md)

每份文档通常包含：

- 算法背景
- 基本数学原理
- 伪代码
- 代码实现对应关系
- 测试输出说明

## 设计原则

本项目的实现原则是：

- 单文件优先：每个核心算法尽量放在一个 `modules/*.py` 文件中。
- 简洁优先：避免过度封装和复杂工程结构。
- 可读优先：变量命名尽量贴近算法概念。
- 可运行优先：每个主要模块都配有测试脚本。
- 可视化优先：测试结果尽量输出曲线或动态演示，方便直观理解。

## 注意事项

强化学习算法的训练效果会受到随机种子、训练轮数、环境版本和硬件差异影响。当前实现主要用于理解算法流程，不保证达到最优性能。

如果只想阅读算法，建议先看 `modules/` 中对应的单文件实现，再运行 `tests/` 中的最小测试脚本，最后对照 `docs/` 中的数学说明。
