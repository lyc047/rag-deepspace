#!/usr/bin/env python
"""Plot frozen Stage-9 confirmation summaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    profiles = list(payload["profiles"])
    policies = ("fixed10", "boot_event", "no_heartbeat", "stateless_exact")
    labels = ("Fixed-10", "Boot event", "No heartbeat", "Stateless exact")
    colors = ("#64748b", "#0f766e", "#d97706", "#7c3aed")
    x = np.arange(len(profiles))
    width = 0.19
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    for index, (policy, label, color) in enumerate(zip(policies, labels, colors)):
        bits = [
            payload["profiles"][name]["policies"][policy]["bits_per_scene"]["mean"]
            for name in profiles
        ]
        clean = [
            100
            * payload["profiles"][name]["policies"][policy]["clean_rate"]["mean"]
            for name in profiles
        ]
        offset = (index - 1.5) * width
        axes[0].bar(x + offset, bits, width, label=label, color=color)
        axes[1].bar(x + offset, clean, width, label=label, color=color)
    display = [name.replace("_", "\n") for name in profiles]
    axes[0].set_title("Total protocol cost")
    axes[0].set_ylabel("Bits per scene")
    axes[1].set_title("Task reliability")
    axes[1].set_ylabel("Clean rate (%)")
    axes[1].set_ylim(55, 101)
    for axis in axes:
        axis.set_xticks(x, display)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)
    fig.suptitle("Stage-9 independent-seed software-in-loop confirmation")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    main()
