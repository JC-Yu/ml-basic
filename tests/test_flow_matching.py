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

from modules.flow_matching import FlowMatching


def sample_8_gaussians(batch_size, radius=2.0, std=0.1):
    angles = torch.randint(0, 8, (batch_size,))
    theta = angles.float() * (2.0 * torch.pi / 8.0)
    centers = torch.stack([radius * torch.cos(theta), radius * torch.sin(theta)], dim=1)
    return centers + torch.randn(batch_size, 2) * std


def save_scatter_svg(target, generated, path):
    target = target.detach().cpu().numpy()
    generated = generated.detach().cpu().numpy()
    all_points = np.concatenate([target, generated], axis=0)

    x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
    y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
    pad_x = (x_max - x_min) * 0.12 + 1e-6
    pad_y = (y_max - y_min) * 0.12 + 1e-6
    x_min -= pad_x
    x_max += pad_x
    y_min -= pad_y
    y_max += pad_y

    width, height = 960, 420
    margin = 36
    gap = 24
    panel_w = (width - 2 * margin - gap) / 2
    panel_h = height - 2 * margin

    def map_point(point, ox, oy):
        x = ox + (point[0] - x_min) / (x_max - x_min) * panel_w
        y = oy + (y_max - point[1]) / (y_max - y_min) * panel_h
        return x, y

    def draw_panel(points, ox, title, color):
        circles = []
        for point in points:
            x, y = map_point(point, ox, margin)
            circles.append(
                f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2" fill="{color}" fill-opacity="0.55" />'
            )
        return "".join(
            [
                f'<rect x="{ox}" y="{margin}" width="{panel_w}" height="{panel_h}" fill="#fff" stroke="#222" stroke-width="1" />',
                f'<text x="{ox + 12}" y="{margin + 18}" font-size="16" font-family="monospace" fill="#111">{title}</text>',
                *circles,
            ]
        )

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
{draw_panel(target, margin, "target", "#1f77b4")}
{draw_panel(generated, margin + panel_w + gap, "generated", "#d62728")}
</svg>
"""
    path.write_text(svg, encoding="utf-8")


def map_point(point, bounds, width, height, margin):
    x_min, x_max, y_min, y_max = bounds
    x = margin + (point[0] - x_min) / (x_max - x_min) * (width - 2 * margin)
    y = height - margin - (point[1] - y_min) / (y_max - y_min) * (height - 2 * margin)
    return x, y


def render_frame(current, target, step, total_steps):
    width, height = 640, 640
    margin = 36
    bounds = (-4.0, 4.0, -4.0, 4.0)

    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((margin, margin, width - margin, height - margin), outline=(30, 30, 30), width=1)

    zero_x, zero_y = map_point((0.0, 0.0), bounds, width, height, margin)
    draw.line((zero_x, margin, zero_x, height - margin), fill=(230, 230, 230), width=1)
    draw.line((margin, zero_y, width - margin, zero_y), fill=(230, 230, 230), width=1)

    for point in target:
        x, y = map_point(point, bounds, width, height, margin)
        draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=(135, 180, 230))

    for point in current:
        x, y = map_point(point, bounds, width, height, margin)
        draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(214, 39, 40))

    draw.text((margin, 12), f"flow matching trajectory  step {step:02d}/{total_steps}", fill=(20, 20, 20))
    return np.asarray(image)


def save_trajectory_mp4(trajectory, target, path):
    frames = []
    total_steps = trajectory.shape[0] - 1
    trajectory = trajectory.detach().cpu().numpy()
    target = target.detach().cpu().numpy()

    for step, current in enumerate(trajectory):
        frames.append(render_frame(current, target, step, total_steps))

    imageio.mimsave(path, frames, fps=20, macro_block_size=1)


def test_flow_matching_smoke():
    seed = 7
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    outputs_dir = ROOT_DIR / "outputs"
    outputs_dir.mkdir(exist_ok=True)

    fm = FlowMatching(data_dim=2, hidden_dim=128, lr=1e-3)
    batch_size = 256
    train_steps = 5000

    for step in range(train_steps):
        source = torch.randn(batch_size, 2)
        target = sample_8_gaussians(batch_size)
        loss = fm.update(source, target)
        if (step + 1) % 100 == 0:
            print(f"step {step + 1:04d}/{train_steps} | loss={loss:.4f}")

    target = sample_8_gaussians(512)
    video_target = sample_8_gaussians(256)
    trajectory = fm.sample_trajectory(256, steps=60).cpu()
    generated = trajectory[-1]

    svg_path = outputs_dir / "flow_matching_samples.svg"
    video_path = outputs_dir / "flow_matching_dynamics.mp4"
    save_scatter_svg(target, generated, svg_path)
    save_trajectory_mp4(trajectory, video_target, video_path)
    print(f"saved visualization: {svg_path}")
    print(f"saved animation: {video_path}")

    assert svg_path.exists()
    assert video_path.exists()
    assert trajectory.shape == (61, 256, 2)
    assert generated.shape == (256, 2)
    assert torch.isfinite(generated).all()


if __name__ == "__main__":
    test_flow_matching_smoke()
    print("All tests passed!")
