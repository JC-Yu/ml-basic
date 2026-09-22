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

from modules.trpo import TRPOAgent


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
<text x="{left}" y="18" font-size="16" font-family="monospace" fill="#111">TRPO reward curve</text>
<line x1="{left}" y1="{top + plot_h}" x2="{left + plot_w}" y2="{top + plot_h}" stroke="#222" stroke-width="1" />
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#222" stroke-width="1" />
<polyline fill="none" stroke="#8c564b" stroke-width="2" points="{" ".join(points)}"/>
</svg>
""",
        encoding="utf-8",
    )


def save_video_mp4(frames, path):
    imageio.mimsave(path, frames, fps=30, macro_block_size=1)


def test_trpo_pendulum_smoke():
    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    outputs_dir = ROOT_DIR / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    env = gym.make("Pendulum-v1")
    eval_env = gym.make("Pendulum-v1", render_mode="rgb_array")
    agent = TRPOAgent(
        env.observation_space.shape[0],
        env.action_space.shape[0],
        float(env.action_space.high[0]),
        max_kl=0.01,
        damping=0.1,
        critic_lr=1e-3,
        critic_steps=80,
        cg_steps=10,
    )

    train_rewards = []
    train_episodes = 1000
    episodes_per_update = 10
    max_steps = 200

    for update in range(train_episodes // episodes_per_update):
        batch_reward = 0.0
        for batch_episode in range(episodes_per_update):
            episode = update * episodes_per_update + batch_episode
            state, _ = env.reset(seed=seed + episode)
            total_reward = 0.0

            for _ in range(max_steps):
                action = agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                agent.replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                total_reward += reward
                if done:
                    break
            batch_reward += total_reward

        losses = agent.update()
        mean_reward = batch_reward / episodes_per_update
        train_rewards.append(mean_reward)
        loss_text = "n/a" if losses is None else (
            f"policy={losses[0]:.4f} | value={losses[1]:.4f} | kl={losses[2]:.5f}"
        )
        print(
            f"update {update + 1:03d}/{train_episodes // episodes_per_update} | "
            f"episodes={(update + 1) * episodes_per_update:04d}/{train_episodes} | "
            f"reward={mean_reward:7.1f} | {loss_text}"
        )

    eval_rewards = []
    frames = []
    for eval_episode in range(10):
        state, _ = eval_env.reset(seed=seed + 1000 + eval_episode)
        episode_frames = [np.asarray(eval_env.render())]
        eval_reward = 0.0
        for _ in range(max_steps):
            action = agent.select_action(state, deterministic=True)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            episode_frames.append(np.asarray(eval_env.render()))
            eval_reward += reward
            if terminated or truncated:
                break
        eval_rewards.append(eval_reward)
        if eval_episode == 0:
            frames = episode_frames

    svg_path = outputs_dir / "trpo_rewards.svg"
    video_path = outputs_dir / "trpo_pendulum_demo.mp4"
    save_reward_svg(train_rewards, svg_path)
    save_video_mp4(frames, video_path)
    print(
        f"eval 10 episodes | mean={np.mean(eval_rewards):7.1f} | "
        f"max={np.max(eval_rewards):7.1f}"
    )
    print(f"saved reward plot: {svg_path}")
    print(f"saved animation: {video_path}")

    env.close()
    eval_env.close()

    assert len(train_rewards) == train_episodes // episodes_per_update
    assert svg_path.exists()
    assert video_path.exists()
    assert len(frames) > 1
    assert all(np.isfinite(train_rewards))
    assert all(np.isfinite(eval_rewards))


if __name__ == "__main__":
    test_trpo_pendulum_smoke()
    print("All tests passed!")
