"""Training-curve figures for the README, drawn from web/data/run_*.json.

Writes docs/img/<name>-light.png and <name>-dark.png (GitHub shows the one matching the viewer's theme).

    .venv/bin/python scripts/make_figures.py
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DATA = Path("web/data")
OUT = Path("docs/img")
THEMES = {
    "light": dict(bg="#fcfcfb", ink="#0b0b0b", muted="#52514e", grid="#e4e3df",
                  series=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark": dict(bg="#1a1a19", ink="#f0efec", muted="#c3c2b7", grid="#34332f",
                 series=["#3987e5", "#d95926", "#199e70"]),
}
SERIES = [("recv_acc", "receiver correct"), ("literal", "colour named"), ("flagged", "monitor flags")]
RUNS = [
    ("en_baseline", "Pilot A: penalties from step 1"),
    ("en_warmup", "Pilot B: 200-step warm-up"),
    ("q3_en_warmup", "Qwen3 English, round 1"),
    ("q3_ar_warmup", "Qwen3 Arabic, round 1"),
    ("r2_en_s1", "Qwen3 English, round 2 seed 1"),
    ("r2_ar_s1", "Qwen3 Arabic, round 2 seed 1"),
    ("r2_en_s2", "Qwen3 English, round 2 seed 2"),
    ("r2_ar_s2", "Qwen3 Arabic, round 2 seed 2"),
]


def rolling(x, w=25):
    x = np.asarray(x, float)
    c = np.cumsum(np.insert(x, 0, 0))
    out = np.empty_like(x)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def panel(ax, run, title, t, labels):
    steps = np.array(run["steps"])
    ax.set_facecolor(t["bg"])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="y", color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlim(1, steps[-1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0%", "50%", "100%"])
    ax.set_xticks([0, 200, 400, 600])
    ax.tick_params(colors=t["muted"], labelsize=8, length=0)
    warm = run["config"].get("penalty_warmup", 0)
    if warm:
        ax.axvline(warm, color=t["muted"], linestyle=(0, (4, 3)), linewidth=1)
        ax.text(warm, 1.06, "full penalty" if not labels else "penalties at full strength", color=t["muted"],
                fontsize=7, ha="center", va="bottom")
    ends = []
    for (key, name), color in zip(SERIES, t["series"]):
        y = rolling(run["metrics"][key])
        ax.plot(steps, y, color=color, linewidth=2, solid_capstyle="round")
        ends.append([y[-1], f"{name} {y[-1]:.0%}"])
    if labels:  # direct labels at the line ends, pushed apart so they don't overlap
        ends.sort()
        for i in range(1, len(ends)):
            ends[i][0] = max(ends[i][0], ends[i - 1][0] + 0.09)
        for y, text in ends:
            ax.annotate(text, (steps[-1], y), xytext=(6, 0), textcoords="offset points", color=t["ink"],
                        fontsize=8, va="center", annotation_clip=False)
    ax.set_title(title, color=t["ink"], fontsize=9.5, loc="left", pad=14)


def legend(fig, t, y):
    handles = [plt.Line2D([], [], color=c, linewidth=2) for c in t["series"]]
    fig.legend(handles, [n for _, n in SERIES], loc="upper left", bbox_to_anchor=(0.01, y), ncol=3, frameon=False,
               fontsize=8.5, labelcolor=t["ink"], handlelength=1.8)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs = {rid: json.loads((DATA / f"run_{rid}.json").read_text()) for rid, _ in RUNS}
    for theme, t in THEMES.items():
        fig, ax = plt.subplots(figsize=(7.6, 3.2), dpi=130)
        fig.patch.set_facecolor(t["bg"])
        panel(ax, runs["en_warmup"], "Pilot B (Qwen2.5-1.5B, English): 25-step rolling mean", t, labels=True)
        ax.set_xlabel("training step", color=t["muted"], fontsize=8)
        fig.subplots_adjust(left=0.07, right=0.76, top=0.78, bottom=0.16)
        legend(fig, t, 0.99)
        fig.savefig(OUT / f"pilot-b-{theme}.png", facecolor=t["bg"])
        plt.close(fig)

        fig, axes = plt.subplots(2, 4, figsize=(11, 4.6), dpi=110, sharey=True)
        fig.patch.set_facecolor(t["bg"])
        for ax, (rid, title) in zip(axes.flat, RUNS):
            panel(ax, runs[rid], title, t, labels=False)
        fig.subplots_adjust(left=0.05, right=0.99, top=0.85, bottom=0.07, hspace=0.45, wspace=0.12)
        legend(fig, t, 0.995)
        fig.savefig(OUT / f"all-runs-{theme}.png", facecolor=t["bg"])
        plt.close(fig)
    for p in sorted(OUT.glob("*.png")):
        print(f"{p}  {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
