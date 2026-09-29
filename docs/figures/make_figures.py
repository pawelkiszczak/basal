"""Figures of the Apple Silicon engine comparison (docs/HARDWARE.md, docs/GGUF.md) from apple_engines.json.

    uv pip install matplotlib && python docs/figures/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.patches import Patch
from matplotlib.text import Text

HERE = Path(__file__).parent
D = json.loads((HERE / "apple_engines.json").read_text())
MODELS = ("4.5B", "1.5B")
COLORS = {"basal": "#1f6fb4", "llama.cpp": "#e07b1f", "mlx-lm based": "#2a9d55", "other": "#8a8a8a"}
TV_FLOOR = 1e-4  # TV below this is drawn at the floor of the log scale
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 160})
# candidate label offsets (points): 8 directions at growing distance; labels further away get a leader line
OFFSETS = [(r * dx, r * dy) for r in (7, 14, 24, 36) for dx, dy in
           ((1, 0.4), (1, -1), (-1, 0.4), (-1, -1), (0.3, 1), (0.3, -1.6), (1, 1), (-1, 1))]


def timed(model):
    return [e for e in D["engines"] if e.get(model)]


def family_legend(ax, **kw):
    ax.legend(handles=[Patch(color=c, label=f) for f, c in COLORS.items()], title="engine family", frameon=False, **kw)


def place_labels(ax, points):
    """Annotate (x, y, text) points, trying several offsets per label so that labels overlap neither each other nor
    the markers."""
    from matplotlib.transforms import Bbox
    renderer = ax.figure.canvas.get_renderer()
    placed = [ax.get_legend().get_window_extent(renderer)] if ax.get_legend() else []
    marks = []
    for x, y, _ in points:
        px, py = ax.transData.transform((x, y))
        marks.append(Bbox.from_bounds(px - 7, py - 7, 14, 14))
    placed += marks
    for j, (x, y, text) in enumerate(points):
        for dx, dy in OFFSETS:
            far = abs(dx) > 10 or abs(dy) > 10
            t = ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=7.5, color="#333",
                            ha="left" if dx > 0 else "right", va="center",
                            arrowprops=dict(arrowstyle="-", color="#aaa", lw=0.6, shrinkA=1, shrinkB=5) if far
                            else None)
            t.update_positions(renderer)
            box = Text.get_window_extent(t, renderer).expanded(1.03, 1.25)  # the text only, not the leader line
            if not any(box.overlaps(b) for b in placed if b is not marks[j]):
                placed.append(box)
                break
            t.remove()
        else:  # no free spot: nearest offset
            ax.annotate(text, (x, y), xytext=OFFSETS[0], textcoords="offset points", fontsize=7.5, color="#333")


def speed_vs_fidelity():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    for ax, model in zip(axes, MODELS):
        pts = []
        for e in sorted(timed(model), key=lambda e: e[model]["tv_mean"]):
            r = e[model]
            y = max(r["tv_mean"], TV_FLOOR)
            ax.scatter(r["ms"], y, s=25 + 18 * r["dec_s"] / (1 if model == "4.5B" else 3),
                       color=COLORS[e["family"]], alpha=0.85, edgecolor="white", linewidth=0.6, zorder=3)
            pts.append((r["ms"], y, e["label"]))
        ax.set_yscale("log")
        ax.margins(x=0.18, y=0.2)
        ax.set_xlabel("ms per decision (both option orders, one request at a time)")
        ax.set_ylabel("mean total-variation distance to fp32 (log)")
        ax.set_title(f"basal-1.0-{model}: lower left is better, marker area ~ decisions/s")
        ax.grid(alpha=0.25, zorder=0)
        if model == "4.5B":
            family_legend(ax, loc="upper right")
    fig.suptitle(f"Speed vs faithfulness on {D['machine']}", fontsize=11)
    fig.tight_layout()
    fig.canvas.draw()
    for ax, model in zip(axes, MODELS):
        place_labels(ax, [(e[model]["ms"], max(e[model]["tv_mean"], TV_FLOOR), e["label"])
                          for e in sorted(timed(model), key=lambda e: e[model]["ms"])])
    fig.savefig(HERE / "apple_speed_vs_fidelity.png")


def latency_throughput():
    order = sorted(timed("4.5B"), key=lambda e: e["4.5B"]["ms"])
    y = np.arange(len(order))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    for ax, key, title in ((axes[0], "ms", "ms per decision (lower is better)"),
                           (axes[1], "dec_s", "decisions per second (higher is better)")):
        for model, off, alpha in (("4.5B", -0.2, 1.0), ("1.5B", 0.2, 0.45)):
            vals = [e[model][key] for e in order]
            ax.barh(y + off, vals, height=0.38, color=[COLORS[e["family"]] for e in order], alpha=alpha)
            for yi, v in zip(y + off, vals):
                ax.text(v, yi, f" {v:.0f}" if key == "ms" else f" {v:.1f}", va="center", fontsize=7)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
    axes[0].set_yticks(y, [e["label"] for e in order])
    axes[0].invert_yaxis()
    family_legend(axes[1], loc="lower right")
    fig.suptitle(f"Latency and throughput on {D['machine']}, cooled GPU, unseen prompts "
                 "(solid bars: basal-1.0-4.5B, light bars: basal-1.0-1.5B)", fontsize=11)
    fig.tight_layout()
    fig.savefig(HERE / "apple_latency_throughput.png")


def fidelity_heatmap():
    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    norm = LogNorm(vmin=TV_FLOOR, vmax=1)
    for ax, model in zip(axes, MODELS):
        tv, flips, items = D["tv"][model], D["flip"][model], D["items"][model]
        rows = sorted(tv, key=lambda k: np.mean(tv[k]))
        cols = sorted(range(len(items)), key=lambda i: (items[i]["type"], items[i]["lang"], items[i]["id"]))
        m = np.array([[max(tv[k][i], TV_FLOOR) for i in cols] for k in rows])
        im = ax.imshow(m, aspect="auto", cmap="magma_r", norm=norm, interpolation="nearest")
        for r, k in enumerate(rows):
            for c, i in enumerate(cols):
                if flips[k][i]:
                    ax.text(c, r, "×", ha="center", va="center", color="#00c2ff", fontsize=9, fontweight="bold")
        types = [items[i]["type"] for i in cols]
        for c in range(1, len(cols)):
            if types[c] != types[c - 1]:
                ax.axvline(c - 0.5, color="white", linewidth=2)
        for t in dict.fromkeys(types):
            idx = [c for c, x in enumerate(types) if x == t]
            ax.text((idx[0] + idx[-1]) / 2, -0.8, f"{t} ({len(idx)})", ha="center", va="bottom", fontsize=8.5)
        ax.set_yticks(range(len(rows)), [f"{D['labels'][k]}  ({np.mean(tv[k]):.4f})" for k in rows], fontsize=8)
        ax.set_xticks(range(len(cols)), [f"{items[i]['id']} {items[i]['lang']}" for i in cols], rotation=90,
                      fontsize=6.5)
        ax.set_title(f"basal-1.0-{model}: per-item total-variation distance to fp32 (row mean in brackets; "
                     f"× = top option differs from fp32)", pad=16)
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="TV (log)")
    fig.tight_layout()
    fig.savefig(HERE / "apple_fidelity_heatmap.png")


def memory_vs_fidelity():
    fig, ax = plt.subplots(figsize=(6.8, 4.3))
    fmt = {"MLX": COLORS["basal"], "GGUF (llama.cpp)": COLORS["llama.cpp"]}
    for p in D["memory_4.5B"]:
        c = fmt["MLX"] if p["label"].startswith("mlx") else fmt["GGUF (llama.cpp)"]
        ax.scatter(p["gb"], max(p["tv_mean"], TV_FLOOR), s=60, color=c, zorder=3)
        ax.annotate(f"{p['label']} · {p['ms']:.0f} ms", (p["gb"], max(p["tv_mean"], TV_FLOOR)), xytext=(6, 2),
                    textcoords="offset points", fontsize=8)
    ax.legend(handles=[Patch(color=c, label=k) for k, c in fmt.items()], frameon=False, loc="upper right")
    ax.set_yscale("log")
    ax.set_xlim(2, 11.5)
    ax.set_xlabel("weights in memory (GB; GGUF: file size)")
    ax.set_ylabel("mean TV distance to fp32 (log)")
    ax.set_title("basal-1.0-4.5B on Apple Silicon: memory vs faithfulness")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(HERE / "apple_memory_vs_fidelity.png")


if __name__ == "__main__":
    speed_vs_fidelity()
    latency_throughput()
    fidelity_heatmap()
    memory_vs_fidelity()
