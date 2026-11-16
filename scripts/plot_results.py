"""Generate the benchmark figures from checked-in CSV files."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter

COLORS = ["#42526f", "#147a73", "#258194", "#5139b0"]
DARK = "#18253a"


def _read(path):
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _by_variant(rows):
    groups = {}
    for row in rows:
        variant = (row["method"], int(row["interval"]))
        groups.setdefault(variant, []).append(row)
    return groups


def _format_panel(axis, title, xlabel, ylabel):
    axis.set_title(title, loc="left", fontsize=13, fontweight="bold", pad=12)
    axis.set_xlabel(xlabel)
    axis.set_ylabel(ylabel)
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#e2e8f0")
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color("#cbd5e1")


def _endpoint_figure(rows, path):
    modes = tuple(dict.fromkeys(row["sharding"] for row in rows))
    fig, axes = plt.subplots(1, len(modes), figsize=(5.6 * len(modes), 4.5),
                             squeeze=False, layout="constrained", sharey=True)
    fig.suptitle("Model traffic and held-out quality", x=0.02, ha="left",
                 fontweight="bold", fontsize=17)
    for axis, mode in zip(axes[0], modes):
        subset = [row for row in rows if row["sharding"] == mode]
        for index, ((method, interval), group) in enumerate(_by_variant(subset).items()):
            payload = int(group[0]["model_payload_bytes"]) / 2**20
            scores = np.array([float(row["perplexity"]) for row in group])
            average = float(scores.mean())
            color = COLORS[index % len(COLORS)]
            label = "Sync" if method == "synchronous" else f"H={interval}"
            axis.scatter(np.repeat(payload, len(scores)), scores, color=color, alpha=0.32, s=42)
            axis.scatter(payload, average, color=color, s=65, zorder=3, edgecolor="white")
            offset = (-6, 8) if method == "synchronous" else (6, 8)
            axis.annotate(f"{label} · {average:.2f}", (payload, average), xytext=offset,
                          ha="right" if method == "synchronous" else "left",
                          textcoords="offset points", fontsize=9, color=color)
        axis.set_xscale("log")
        if len(subset) >= 4:
            axis.margins(x=0.17, y=0.18)
        _format_panel(axis, "IID shards" if mode == "iid" else "Label-skewed shards",
                      "Model payload exchanged (MiB, log scale)", "Held-out perplexity")
    fig.savefig(path, dpi=155, facecolor="white")
    plt.close(fig)


def _learning_figure(rows, path):
    modes = tuple(dict.fromkeys(row["sharding"] for row in rows))
    fig, axes = plt.subplots(1, len(modes), figsize=(5.6 * len(modes), 4.5),
                             squeeze=False, layout="constrained", sharey=True)
    fig.suptitle("Convergence at the same training budget", x=0.02, ha="left",
                 fontweight="bold", fontsize=17)
    for axis, mode in zip(axes[0], modes):
        subset = [row for row in rows if row["sharding"] == mode]
        for index, ((method, interval), group) in enumerate(_by_variant(subset).items()):
            by_step = {}
            for row in group:
                step = int(row["local_steps"])
                by_step.setdefault(step, []).append(float(row["val_loss"]))
            steps = sorted(by_step)
            average = [np.mean(by_step[step]) for step in steps]
            color = COLORS[index % len(COLORS)]
            label = "Sync AdamW" if method == "synchronous" else f"DiLoCo H={interval}"
            axis.plot(steps, average, marker="o", linewidth=2, markersize=4, color=color, label=label)
        axis.legend(loc="upper right", frameon=False, fontsize=9)
        axis.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{int(val):,}"))
        _format_panel(axis, "IID shards" if mode == "iid" else "Label-skewed shards",
                      "Minibatches processed per worker", "Mean held-out loss (nats)")
    fig.savefig(path, dpi=155, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    args = parser.parse_args()
    with plt.rc_context({"font.family": "DejaVu Sans", "text.color": DARK,
                         "axes.labelcolor": DARK, "xtick.color": DARK, "ytick.color": DARK}):
        _endpoint_figure(_read(args.results / "summary.csv"), args.results / "tradeoff.png")
        _learning_figure(_read(args.results / "learning_curves.csv"), args.results / "convergence.png")


if __name__ == "__main__":
    main()
