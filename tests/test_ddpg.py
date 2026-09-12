import os
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import random
import sys
from pathlib import Path

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.ddpg import DDPGAgent


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
    for idx, reward in enumerate(rewards):
        x = left + idx * plot_w / max(1, len(rewards) - 1)
        y = top + (y_max - reward) * plot_h / (y_max - y_min)
        points.append(f"{x:.1f},{y:.1f}")

    path.write_text(
        f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<text x="{left}" y="18" font-size="16" font-family="monospace" fill="#111">DDPG reward curve</text>
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1" />
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1" />
<polyline fill="none" stroke="#1f77b4" stroke-width="2" points="{" ".join(points)}"/>
</svg>
""",
        encoding="utf-8",
    )


def save_video_mp4(frames, path):
    imageio.mimsave(path, frames, fps=30, macro_block_size=1)


def test_ddpg_pendulum_smoke():
    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    outputs_dir = ROOT_DIR / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1", render_mode="rgb_array")
    agent = DDPGAgent(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        float(env.action_space.high[0]),
    )

    train_rewards = []
    train_episodes = 200
    max_steps = 1000
    noise_start = 0.3
    noise_end = 0.05

    for episode in range(train_episodes):
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        loss = None
        ratio = episode / max(1, train_episodes - 1)
        noise_std = noise_start + ratio * (noise_end - noise_start)

        for _ in range(max_steps):
            action = agent.select_action(state, noise_std=noise_std)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            agent.replay_buffer.add(state, action, reward, next_state, done)
            loss = agent.update()
            state = next_state
            total_reward += reward
            if done:
                break

        train_rewards.append(total_reward)
        loss_text = "n/a" if loss is None else f"{loss:.4f}"
        print(
            f"train {episode + 1:03d}/{train_episodes} | reward={total_reward:7.1f} | "
            f"noise={noise_std:.3f} | loss={loss_text}"
        )

    eval_reward = 0.0
    frames = []
    state, _ = eval_env.reset(seed=seed + 1000)
    frames.append(np.asarray(eval_env.render()))
    for _ in range(max_steps):
        action = agent.select_action(state, noise_std=0.0)
        state, reward, terminated, truncated, _ = eval_env.step(action)
        frames.append(np.asarray(eval_env.render()))
        eval_reward += reward
        if terminated or truncated:
            break

    svg_path = outputs_dir / "ddpg_rewards.svg"
    video_path = outputs_dir / "ddpg_pendulum_demo.mp4"
    save_reward_svg(train_rewards, svg_path)
    save_video_mp4(frames, video_path)
    print(f"eval  001/1 | reward={eval_reward:7.1f}")
    print(f"saved reward plot: {svg_path}")
    print(f"saved animation: {video_path}")

    env.close()
    eval_env.close()

    assert len(train_rewards) == train_episodes
    assert svg_path.exists()
    assert video_path.exists()
    assert len(frames) > 1
    assert all(np.isfinite(train_rewards))
    assert np.isfinite(eval_reward)


if __name__ == "__main__":
    test_ddpg_pendulum_smoke()
    print("All tests passed!")

