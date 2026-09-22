import os
import random
import sys
from pathlib import Path

os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.reinforce import REINFORCEAgent


def save_reward_svg(rewards, path):
    width, height = 900, 260
    left, right, top, bottom = 40, 20, 20, 40
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min = min(rewards)
    y_max = max(rewards)
    if y_max == y_min:
        y_max = y_min + 1.0

    points = []
    for index, reward in enumerate(rewards):
        x = left + index * plot_w / max(1, len(rewards) - 1)
        y = top + (y_max - reward) * plot_h / (y_max - y_min)
        points.append(f"{x:.1f},{y:.1f}")

    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="18" font-size="16" font-family="monospace" fill="#111">REINFORCE reward curve</text>
<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{" ".join(points)}"/>
</svg>
""",
        encoding="utf-8",
    )


def save_video_mp4(frames, path):
    imageio.mimsave(path, frames, fps=30, macro_block_size=1)


def test_reinforce_cartpole_smoke():
    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    outputs_dir = ROOT_DIR / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    env = gym.make("CartPole-v1")
    eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
    agent = REINFORCEAgent(
        env.observation_space.shape[0],
        env.action_space.n,
        hidden_dim=128,
        gamma=0.99,
        lr=1e-3,
    )

    train_rewards = []
    n_episodes = 1000
    n_steps = 500
    for episode in range(n_episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0

        for _ in range(n_steps):
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            agent.rewards.append(reward)
            state = next_state
            total_reward += reward
            if terminated or truncated:
                break

        loss = agent.update()
        train_rewards.append(total_reward)
        if (episode + 1) % 100 == 0:
            print(
                f"train {episode + 1:04d}/{n_episodes} | "
                f"reward={total_reward:6.1f} | loss={loss:.2f}"
            )

    frames = []
    state, _ = eval_env.reset(seed=seed + 1_000)
    frames.append(np.asarray(eval_env.render()))
    eval_reward = 0.0
    for _ in range(n_steps):
        action = agent.select_action(state, deterministic=True)
        state, reward, terminated, truncated, _ = eval_env.step(action)
        frames.append(np.asarray(eval_env.render()))
        eval_reward += reward
        if terminated or truncated:
            break

    print(f"eval  001/1 | reward={eval_reward:6.1f}")

    reward_path = outputs_dir / "reinforce_rewards.svg"
    video_path = outputs_dir / "reinforce_eval.mp4"
    save_reward_svg(train_rewards, reward_path)
    save_video_mp4(frames, video_path)
    print(f"saved reward plot: {reward_path}")
    print(f"saved video: {video_path}")

    env.close()
    eval_env.close()

    assert reward_path.exists()
    assert video_path.exists()
    assert all(np.isfinite(train_rewards))
    assert np.isfinite(eval_reward)


if __name__ == "__main__":
    test_reinforce_cartpole_smoke()
    print("All tests passed!")
