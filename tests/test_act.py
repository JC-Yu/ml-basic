import random
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw
import torch

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from modules.act import ACTAgent, TemporalEnsembler


def reference_path(phase):
    angle = 2.0 * np.pi * phase
    return np.array([0.78 * np.sin(angle), 0.46 * np.sin(2.0 * angle)], dtype=np.float32)


def make_obs(robot, target, phase):
    angle = 2.0 * np.pi * phase
    return np.array(
        [
            robot[0],
            robot[1],
            target[0],
            target[1],
            np.sin(angle),
            np.cos(angle),
        ],
        dtype=np.float32,
    )


def chunk_points(origin, chunk):
    points = [np.asarray(origin, dtype=np.float32)]
    current = np.asarray(origin, dtype=np.float32)
    for action in chunk:
        current = current + action
        points.append(current.copy())
    return np.asarray(points, dtype=np.float32)


def save_path_svg(target, chunk_path, ensemble_path, path):
    width, height = 960, 440
    margin = 34
    all_points = np.concatenate([target, chunk_path, ensemble_path], axis=0)
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    pad_x = (x_max - x_min) * 0.12 + 1e-6
    pad_y = (y_max - y_min) * 0.12 + 1e-6
    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    def map_point(point):
        x = margin + (point[0] - x_min) / (x_max - x_min) * (width - 2 * margin)
        y = height - margin - (point[1] - y_min) / (y_max - y_min) * (height - 2 * margin)
        return x, y

    def polyline(points, color, stroke_width=2):
        mapped = [map_point(point) for point in points]
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in mapped)
        return f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}" points="{pts}" />'

    def circle(point, color, radius=4):
        x, y = map_point(point)
        return f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{radius}" fill="{color}" />'

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="white"/>
<rect x="{margin}" y="{margin}" width="{width - 2 * margin}" height="{height - 2 * margin}" fill="none" stroke="#222" stroke-width="1" />
<text x="{margin}" y="20" font-size="16" font-family="monospace" fill="#111">ACT imitation rollout</text>
<text x="{margin}" y="{height - 12}" font-size="13" font-family="monospace" fill="#1f77b4">target</text>
<text x="{margin + 90}" y="{height - 12}" font-size="13" font-family="monospace" fill="#d62728">chunk only</text>
<text x="{margin + 210}" y="{height - 12}" font-size="13" font-family="monospace" fill="#2ca02c">temporal ensemble</text>
{polyline(target, "#1f77b4", 2)}
{polyline(chunk_path, "#d62728", 2)}
{polyline(ensemble_path, "#2ca02c", 2)}
{circle(target[0], "#1f77b4")}
{circle(chunk_path[0], "#d62728")}
{circle(ensemble_path[0], "#2ca02c")}
{circle(target[-1], "#1f77b4")}
{circle(chunk_path[-1], "#d62728")}
{circle(ensemble_path[-1], "#2ca02c")}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def render_frame(target, chunk_path, ensemble_path, chunk_only_preview, ensemble_preview, step, total_steps):
    width, height = 1080, 480
    margin = 28
    gap = 20
    panel_w = (width - 2 * margin - gap) / 2
    panel_h = height - 2 * margin
    all_points = np.concatenate(
        [target, chunk_path, ensemble_path, chunk_only_preview, ensemble_preview], axis=0
    )
    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    pad_x = (x_max - x_min) * 0.12 + 1e-6
    pad_y = (y_max - y_min) * 0.12 + 1e-6
    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    def map_point(point, ox):
        x = ox + (point[0] - x_min) / (x_max - x_min) * panel_w
        y = margin + (y_max - point[1]) / (y_max - y_min) * panel_h
        return x, y

    def draw_panel(points, path, preview, ox, title, color):
        draw.rectangle((ox, margin, ox + panel_w, margin + panel_h), outline=(35, 35, 35), width=1)
        draw.text((ox + 10, margin + 8), title, fill=(15, 15, 15))

        target_pixels = [map_point(point, ox) for point in points]
        draw.line(target_pixels, fill=(180, 200, 230), width=2)

        path_pixels = [map_point(point, ox) for point in path[: step + 1]]
        if len(path_pixels) > 1:
            draw.line(path_pixels, fill=color, width=3)

        if preview is not None and len(preview) > 0:
            preview_pixels = [map_point(point, ox) for point in preview]
            draw.line(preview_pixels, fill=(255, 165, 0), width=2)

        cur = map_point(path[step], ox)
        tgt = map_point(points[step], ox)
        draw.ellipse((cur[0] - 4, cur[1] - 4, cur[0] + 4, cur[1] + 4), fill=color)
        draw.ellipse((tgt[0] - 3, tgt[1] - 3, tgt[0] + 3, tgt[1] + 3), fill=(40, 90, 200))

    draw_panel(target, chunk_path, chunk_only_preview, margin, "chunk only", (214, 39, 40))
    draw_panel(
        target,
        ensemble_path,
        ensemble_preview,
        margin + panel_w + gap,
        "temporal ensemble",
        (44, 160, 44),
    )
    draw.text((margin, height - 20), f"step {step + 1:03d}/{total_steps}", fill=(20, 20, 20))
    return np.asarray(image)


def save_video_mp4(frames, path):
    imageio.mimsave(path, frames, fps=24, macro_block_size=1)


def build_dataset(n_demos=180, demo_steps=64, chunk_len=8, max_action=0.08):
    obs, chunks = [], []
    for _ in range(n_demos):
        phase0 = random.random()
        robot = reference_path(phase0) + np.random.normal(0.0, 0.05, size=2).astype(np.float32)
        obs_seq = []
        act_seq = []
        for step in range(demo_steps + chunk_len):
            phase = (phase0 + step / demo_steps) % 1.0
            target = reference_path(phase)
            next_target = reference_path((phase0 + (step + 1) / demo_steps) % 1.0)
            obs_seq.append(make_obs(robot, target, phase))
            action = np.clip(next_target - robot, -max_action, max_action).astype(np.float32)
            act_seq.append(action)
            robot = robot + action

        for step in range(demo_steps):
            obs.append(obs_seq[step])
            chunks.append(act_seq[step : step + chunk_len])

    return np.asarray(obs, dtype=np.float32), np.asarray(chunks, dtype=np.float32)


def rollout(agent, ensemble: bool, steps=90, phase0=0.17, max_action=0.08):
    robot = reference_path(phase0) + np.array([-0.08, 0.05], dtype=np.float32)
    path = [robot.copy()]
    targets = []
    chunks = []
    actions = []
    ensembler = TemporalEnsembler(agent.chunk_len, decay=0.78) if ensemble else None

    if ensemble:
        ensembler.reset()

    for step in range(steps):
        phase = (phase0 + step / steps) % 1.0
        target = reference_path(phase)
        obs = make_obs(robot, target, phase)
        chunk = agent.predict_chunk(obs)
        action = ensembler.combine(chunk) if ensemble else chunk[0]
        action = np.clip(action, -max_action, max_action).astype(np.float32)
        robot = robot + action

        path.append(robot.copy())
        targets.append(target)
        chunks.append(chunk)
        actions.append(action)

    target_path = np.asarray(
        [reference_path((phase0 + step / steps) % 1.0) for step in range(steps + 1)],
        dtype=np.float32,
    )
    return np.asarray(path, dtype=np.float32), target_path, np.asarray(chunks, dtype=np.float32), np.asarray(actions, dtype=np.float32)


def test_act_imitation_smoke():
    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    outputs_dir = ROOT_DIR / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    chunk_len = 8
    max_action = 0.08
    agent = ACTAgent(
        obs_dim=6,
        action_dim=2,
        chunk_len=chunk_len,
        hidden_dim=128,
        action_scale=max_action,
        ensemble_decay=0.78,
        lr=1e-3,
    )

    obs_data, chunk_data = build_dataset(chunk_len=chunk_len, max_action=max_action)
    train_steps = 1000
    batch_size = 128
    for step in range(train_steps):
        idx = np.random.randint(0, len(obs_data), size=batch_size)
        loss = agent.update(obs_data[idx], chunk_data[idx])
        if (step + 1) % 100 == 0:
            print(f"step {step + 1:04d}/{train_steps} | loss={loss:.4f}")

    chunk_path, target_path, chunk_pred, _ = rollout(agent, ensemble=False, steps=90)
    ensemble_path, _, ensemble_pred, _ = rollout(agent, ensemble=True, steps=90)

    chunk_mse = float(np.mean((chunk_path - target_path) ** 2))
    ensemble_mse = float(np.mean((ensemble_path - target_path) ** 2))
    print(f"chunk mse={chunk_mse:.4f} | ensemble mse={ensemble_mse:.4f}")

    svg_path = outputs_dir / "act_paths.svg"
    video_path = outputs_dir / "act_rollout.mp4"
    save_path_svg(target_path, chunk_path, ensemble_path, svg_path)

    frames = []
    total_steps = len(target_path) - 1
    for step in range(total_steps):
        chunk_preview = chunk_points(chunk_path[step], chunk_pred[step])
        ensemble_preview = chunk_points(ensemble_path[step], ensemble_pred[step])
        frames.append(
            render_frame(
                target_path,
                chunk_path,
                ensemble_path,
                chunk_preview,
                ensemble_preview,
                step,
                total_steps,
            )
        )
    save_video_mp4(frames, video_path)
    print(f"saved visualization: {svg_path}")
    print(f"saved animation: {video_path}")

    assert svg_path.exists()
    assert video_path.exists()
    assert chunk_path.shape == target_path.shape
    assert ensemble_path.shape == target_path.shape
    assert np.isfinite(chunk_mse)
    assert np.isfinite(ensemble_mse)


if __name__ == "__main__":
    test_act_imitation_smoke()
    print("All tests passed!")
