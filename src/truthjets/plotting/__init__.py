"""Shared plotting utilities for truthjets."""
from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages

LABEL_MAP = {
    0: "light", 4: "c-jet", 5: "b-jet", 15: r"$\tau$-jet",
    6: "top", 23: "Z", 24: "W", 25: "Higgs",
}
LABEL_COLORS = {
    0: "C0", 4: "C1", 5: "C2", 15: "C3",
    6: "C3", 23: "C1", 24: "C4", 25: "C2",
}


def load_data(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load jets and constituents from an HDF5 file."""
    with h5py.File(path, "r") as f:
        jets = f["jets"][:]
        constit = f["constituents"][:]
    return jets, constit


def get_label_info(jets: np.ndarray) -> tuple[np.ndarray, list[int]]:
    """Return (labels array, sorted unique labels)."""
    labels = jets["HadronConeExclTruthLabelID"]
    unique_labels = sorted(set(labels))
    return labels, unique_labels


def save_figs(figs: list[plt.Figure], output_path: str | Path) -> None:
    """Save a list of figures to a multi-page PDF."""
    with PdfPages(output_path) as pdf:
        for fig in figs:
            pdf.savefig(fig, bbox_inches="tight")
            plt.close(fig)
    print(f"Saved {output_path}")


def group_by_event(jets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group jets by event_id.

    Returns
    -------
    event_ids : unique event IDs
    splits : indices to split flat jet array by event
    counts : number of jets per event
    """
    event_ids, inverse, counts = np.unique(
        jets["event_id"], return_inverse=True, return_counts=True
    )
    splits = np.cumsum(counts)[:-1]
    return event_ids, splits, counts


def flavor_profile(ax, x, y, bins, labels, unique_labels):
    """Plot mean-y vs x profile, one line per flavor.

    Parameters
    ----------
    ax : matplotlib Axes
    x, y : arrays of same length
    bins : bin edges for x
    labels, unique_labels : flavor labels
    """
    centers = 0.5 * (bins[:-1] + bins[1:])
    for lab in unique_labels:
        mask = labels == lab
        means = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = mask & (x >= lo) & (x < hi)
            means.append(np.mean(y[sel]) if np.any(sel) else np.nan)
        ax.plot(
            centers, means, "o-",
            color=LABEL_COLORS.get(lab, "gray"),
            label=LABEL_MAP.get(lab, str(lab)),
            markersize=3,
        )


def flavor_hist(ax, data, bins, labels, unique_labels, **kwargs):
    """Plot an inclusive + per-flavor stacked histogram."""
    ax.hist(data, bins=bins, histtype="step", color="black", linewidth=1.5, label="All")
    for lab in unique_labels:
        mask = labels == lab
        ax.hist(
            data[mask],
            bins=bins,
            histtype="stepfilled",
            alpha=0.3,
            color=LABEL_COLORS.get(lab, "gray"),
            label=LABEL_MAP.get(lab, str(lab)),
        )
