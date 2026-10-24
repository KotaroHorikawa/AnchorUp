"""Plot radial spectra from the spectrum.npz written by analysis.anchorup_diagnostics."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

CURVES = {
    "anchor": {"color": "#0173B2", "label": r"Anchor: $F_{HR}^{anchor}$", "linewidth": 1.5},
    "learned": {"color": "#DE8F05", "label": r"Learned: $F_{HR}^{learn}$", "linewidth": 1.6},
    "correction": {
        "color": "#029E73", "label": r"Correction: $\alpha(F_{HR}^{learn} - F_{HR}^{anchor})$",
        "linewidth": 1.6, "linestyle": "--",
    },
    "pred": {"color": "#CC78BC", "label": r"AnchorUp: $\hat{F}_{HR}$", "linewidth": 1.6},
}
STYLE = {
    "axes.edgecolor": "0.25",
    "axes.grid": True,
    "axes.linewidth": 0.8,
    "font.size": 10,
    "grid.color": "0.88",
    "grid.linewidth": 0.7,
    "legend.frameon": True,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.bbox": "tight",
}


def plot_spectrum(input_path: Path, output_path: Path, preview_path: Path) -> None:
    with np.load(input_path) as saved:
        frequencies = saved["frequencies"]
        curves = {name: saved[name] for name in CURVES}

    plt.rcParams.update(STYLE)
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for name, style in CURVES.items():
        ax.semilogy(frequencies, curves[name], **style)
    ax.set_xlabel("Radial spatial frequency [cycles/pixel]")
    ax.set_ylabel("Mean Fourier power")
    ax.set_ylim(1e1, 3e8)
    ax.grid(True, which="major", alpha=0.85)
    ax.grid(True, which="minor", alpha=0.2, linewidth=0.5)
    ax.legend(framealpha=0.92, edgecolor="0.85", fancybox=True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, format="pdf")
    fig.savefig(preview_path, format="png", dpi=200)
    plt.close(fig)
    print(f"Saved {output_path} and {preview_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="spectrum.npz from the diagnostics")
    parser.add_argument("--output", type=Path, required=True, help="Output PDF path; a PNG preview is saved next to it")
    args = parser.parse_args()
    plot_spectrum(args.input, args.output, args.output.with_suffix(".png"))


if __name__ == "__main__":
    main()
