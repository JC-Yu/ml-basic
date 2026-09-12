<div align="right">
  <a href="README.md">English</a> |
  <a href="README_CN.md">简体中文</a>
</div>

# ml-basic

`ml-basic` is a collection of foundational machine learning modules for robotics research. The project implements common algorithms in concise, intuitive, single-file examples, allowing readers to focus directly on the core ideas.

The current codebase covers Transformer, Deep Reinforcement Learning, Diffusion, Flow Matching, Action Chunking, and other foundational topics. Each module is kept as independent as possible. The test scripts provide minimal runnable examples and save training curves and dynamic demonstrations.

## Project Structure

```text
ml-basic/
├── modules/                  # Core algorithm implementations
│   ├── attention.py           # Basic Attention implementation
│   ├── transformer.py         # Basic Transformer module
│   ├── dqn.py                 # DQN / Double DQN-style implementation
│   ├── ddpg.py                # DDPG for continuous control
│   ├── ppo.py                 # PPO for continuous control
│   ├── sac.py                 # SAC for continuous control
│   ├── diffusion.py           # Simple Diffusion implementation
│   ├── flow_matching.py       # Simple Flow Matching implementation
│   └── act.py                 # Action Chunking / Temporal Ensembling example
├── tests/                     # Runnable test scripts
├── docs/                      # Algorithm documentation
├── assets/                    # Static assets for the README and documentation
├── pyproject.toml             # Project dependency configuration
└── uv.lock                   # uv lock file
```

## Environment Setup

The project uses Python 3.10 and `uv` for dependency management.

```bash
cd ml-basic/
uv sync
```

If the virtual environment has already been created, activate it directly:

```bash
source .venv/bin/activate
```

## Usage

All test scripts can be run directly from the project root.

```bash
cd ml-basic
python tests/test_dqn.py
```

Replace `test_dqn.py` with another test script to run a different algorithm.

## Algorithm Documentation

Detailed algorithm explanations are available in the `docs/` directory:

- [DQN](docs/dqn.md)
- [DDPG](docs/ddpg.md)
- [PPO](docs/ppo.md)
- [SAC](docs/sac.md)
- [Flow Matching](docs/flow_matching.md)
- [Diffusion](docs/diffusion.md)
- [ACT](docs/act.md)

Each document usually includes:

- Algorithm background
- Basic mathematical principles
- Pseudocode
- Mapping between the explanation and the implementation
- Test output description

## Design Principles

The project follows these principles:

- **Single-file first:** Keep each core algorithm in one `modules/*.py` file whenever possible.
- **Simplicity first:** Avoid excessive abstraction and complex engineering structures.
- **Readability first:** Use variable names that stay close to the algorithmic concepts.
- **Runnable first:** Provide a test script for each major module.
- **Visualization first:** Save curves or dynamic demonstrations whenever possible, making the algorithms easier to understand intuitively.

## Notes

The training performance of reinforcement learning algorithms depends on random seeds, the number of training episodes, environment versions, and hardware differences. The current implementations are mainly intended for understanding algorithmic workflows and are not guaranteed to achieve optimal performance.

If you only want to study the algorithms, start with the corresponding single-file implementation in `modules/`, run the minimal test script in `tests/`, and then read the mathematical explanation in `docs/`.
