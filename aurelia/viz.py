"""A simple visualiser: the bowl being stirred, grains coloured by friction heat, diamonds flashing
where they form. Writes <out>.gif and <out>.png (the last frame)."""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

from .bowl import Bowl, BowlConfig


def render_assembly(entry: dict, out: str, every: int = 25) -> None:
    """32 copies of a vault particle assembling (seed 0 of its test), ports coloured by charge,
    bonds drawn at the end."""
    import math

    import torch

    from .particle import Worlds, world_seed

    d = entry["definition"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    w = Worlds([d], [world_seed(entry["provenance"]["run_seed"], d, 0)], dev, record_frames=every)
    w.assemble()
    k, r, R = d["k"], d["radius"], float(w.Rw[0])
    q = np.array(d["charge"])
    cols = plt.cm.twilight((np.arctan2(q[:, 1], q[:, 0]) + math.pi) / (2 * math.pi))
    fig, ax = plt.subplots(figsize=(6, 6), dpi=90)
    fig.patch.set_facecolor("#0d0b08")

    def draw(fi, bonds=False):
        ax.clear()
        ax.set_xlim(-R - 1, R + 1)
        ax.set_ylim(-R - 1, R + 1)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.add_patch(plt.Circle((0, 0), R, fill=False, color="#8a7a5a", lw=1.5))
        pos, th = w.frames[fi] if fi < len(w.frames) else (w.pos[0].cpu().numpy(), w.th[0].cpu().numpy())
        a = th[:, None] + np.array(d["angle"])[None, :]
        pp = pos[:, None, :] + r * np.stack([np.cos(a), np.sin(a)], -1)
        for p in range(len(pos)):
            ax.add_patch(plt.Circle(pos[p], r, color="#3a3226", ec="#c9b98f", lw=0.6))
            ax.scatter(pp[p, :, 0], pp[p, :, 1], s=14, c=cols, zorder=3)
        if bonds:
            from .diamond import K_MAX
            at = lambda i: pp[i // K_MAX, i % K_MAX]  # bond indices count K_MAX slots per particle
            for i, j in w.bonds[0]:
                ax.plot(*zip(at(i), at(j)), color="#9ff3ff", lw=2.2, zorder=4)
                m = (at(i) + at(j)) / 2
                ax.scatter([m[0]], [m[1]], s=70, facecolors="none", edgecolors="#9ff3ff", linewidths=1.6, zorder=5)
            st = w.stats[0]
            ax.set_title(f"{entry['id']}: {st['bonds']} bonds, {st['molecules']} molecules, largest {st['largest']}",
                         color="#e8dcc0", fontsize=9)
        return []

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    anim = FuncAnimation(fig, draw, frames=len(w.frames))
    anim.save(out + ".gif", writer=PillowWriter(fps=12))
    draw(len(w.frames), bonds=True)
    fig.savefig(out + ".png", facecolor=fig.get_facecolor())
    print(f"wrote {out}.gif and {out}.png")


def render(stir: str, seed: int, seconds: float, out: str, every: int = 8) -> None:
    cfg = BowlConfig()
    b = Bowl(seed, stir, cfg)
    frames = []
    steps = int(seconds / cfg.dt)
    for i in range(steps):
        b.step()
        if i % every == 0:
            frames.append((b.x.cpu().numpy().copy(), b.heat.cpu().numpy().copy(),
                           None if b.spoon is None else b.spoon[:2], b.t, len(b.formed)))
    births = [(r["stir"]["time"], r["stir"]["centroid"], len(r["nodes"])) for r in b.formed]

    fig, ax = plt.subplots(figsize=(6, 6), dpi=90)
    fig.patch.set_facecolor("#0d0b08")
    R = b.R
    ax.set_xlim(-R - 2, R + 2)
    ax.set_ylim(-R - 2, R + 2)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(plt.Circle((0, 0), R + 0.5, fill=False, color="#8a7a5a", lw=2))
    x0 = frames[0][0]
    size = (72 * 6 / (2 * R + 4) * cfg.radius * 0.9) ** 2 * 3.1
    sc = ax.scatter(x0[:, 0], x0[:, 1], s=size, c=np.zeros(len(x0)), cmap="inferno", vmin=0, vmax=1.6,
                    linewidths=0)
    spoon = plt.Circle((0, 0), cfg.spoon_radius, color="#c9c2b4", alpha=0.9, visible=False)
    ax.add_patch(spoon)
    stars = ax.scatter([], [], marker="*", s=[], c="#9ff3ff", edgecolors="white", linewidths=0.6)
    label = ax.text(-R, R + 0.5, "", color="#e8dcc0", fontsize=10, family="monospace")

    def draw(fi):
        x, heat, sp, t, nd = frames[fi]
        sc.set_offsets(x)
        sc.set_array(np.log1p(heat) / np.log(10))
        if sp is not None:
            spoon.center = sp
            spoon.set_visible(True)
        recent = [(c, k, t - bt) for bt, c, k in births if 0 <= t - bt < 0.6]
        if recent:
            stars.set_offsets(np.array([c for c, _, _ in recent]))
            stars.set_sizes(np.array([(30 + 18 * k) * (1 - age / 0.6) for _, k, age in recent]))
        else:
            stars.set_offsets(np.zeros((0, 2)))
        label.set_text(f"stir {stir}   t {t:5.1f} s   diamonds {nd}")
        return sc, spoon, stars, label

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    anim = FuncAnimation(fig, draw, frames=len(frames), blit=False)
    anim.save(out + ".gif", writer=PillowWriter(fps=12))
    draw(len(frames) - 1)
    fig.savefig(out + ".png", facecolor=fig.get_facecolor())
    print(f"wrote {out}.gif and {out}.png ({len(frames)} frames, {len(births)} diamonds)")
