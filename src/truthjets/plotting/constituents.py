"""Constituent-level distribution plots."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from truthjets.plotting import get_label_info, flavor_hist, load_data, save_figs


def plot(jets: np.ndarray, constit: np.ndarray) -> list[plt.Figure]:
    """Create constituent-level distribution figures.

    Parameters
    ----------
    jets : np.ndarray
        Structured array from /jets dataset.
    constit : np.ndarray
        Structured array from /constituents dataset.

    Returns
    -------
    list[plt.Figure]
        Figures to be saved as pages in the output PDF.
    """
    labels, unique_labels = get_label_info(jets)

    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    fig.suptitle(f"Constituent Distributions ({len(jets)} jets)", fontsize=14)

    # TODO: add constituent-level plots

    plt.tight_layout()
    return [fig]


def main():
    parser = argparse.ArgumentParser(description="Plot constituent-level distributions")
    parser.add_argument("input", help="Input HDF5 file")
    parser.add_argument("-o", "--output", default="constituents.pdf", help="Output PDF path")
    args = parser.parse_args()

    jets, constit = load_data(args.input)
    figs = plot(jets, constit)
    save_figs(figs, args.output)
