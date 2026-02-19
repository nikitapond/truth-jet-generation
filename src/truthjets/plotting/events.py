"""Event-level distribution plots."""

from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from truthjets.plotting import (
    LABEL_COLORS,
    LABEL_MAP,
    group_by_event,
    load_data,
    save_figs,
)


def plot(jets: np.ndarray, constit: np.ndarray) -> list[plt.Figure]:
    """Create event-level distribution figures.

    Parameters
    ----------
    jets : np.ndarray
        Structured array from /jets dataset (must contain event_id).
    constit : np.ndarray
        Structured array from /constituents dataset.

    Returns
    -------
    list[plt.Figure]
        Two figures (pages) with 4 plots each.
    """
    event_ids, splits, n_jets_per_event = group_by_event(jets)
    n_events = len(event_ids)
    labels = jets["HadronConeExclTruthLabelID"]

    # Split arrays by event
    jet_groups = np.split(jets, splits)
    label_groups = np.split(labels, splits)

    # Per-event flavor counts
    flavor_ids = [0, 4, 5, 15]
    flavor_counts = {fid: np.zeros(n_events, dtype=np.int32) for fid in flavor_ids}
    for i, lg in enumerate(label_groups):
        for fid in flavor_ids:
            flavor_counts[fid][i] = np.sum(lg == fid)

    # ---- Page 1: Jets per event (2x2) ----
    fig1, axes1 = plt.subplots(2, 2, figsize=(12, 10))
    fig1.suptitle(f"Jets per Event ({n_events} events, {len(jets)} jets)", fontsize=14)

    # 1. Total jets per event
    ax = axes1[0, 0]
    max_jets = int(n_jets_per_event.max()) + 1
    bins_njets = np.arange(-0.5, max_jets + 1.5, 1)
    ax.hist(n_jets_per_event, bins=bins_njets, histtype="step", color="black", linewidth=1.5)
    ax.set_xlabel("Number of jets per event")
    ax.set_ylabel("Events")
    ax.set_title("Jets per event")

    # 2. Number of each flavor per event
    ax = axes1[0, 1]
    max_flavor = max(fc.max() for fc in flavor_counts.values()) + 1
    bins_flavor = np.arange(-0.5, max_flavor + 1.5, 1)
    for fid in flavor_ids:
        ax.hist(
            flavor_counts[fid],
            bins=bins_flavor,
            histtype="step",
            linewidth=1.5,
            color=LABEL_COLORS.get(fid, "gray"),
            label=LABEL_MAP.get(fid, str(fid)),
        )
    ax.set_xlabel("Number of jets per event")
    ax.set_ylabel("Events")
    ax.set_title("Jets per event by flavor")
    ax.legend()

    # 3. HS vs PU jets per event (only if pileup present)
    ax = axes1[1, 0]
    has_pileup = "pt_frac_pu" in jets.dtype.names and np.any(jets["pt_frac_pu"] > 0)
    if has_pileup:
        is_pu_jet = jets["pt_frac_pu"] > 0.5
        pu_flags = np.split(is_pu_jet, splits)
        n_hs = np.array([np.sum(~pf) for pf in pu_flags])
        n_pu = np.array([np.sum(pf) for pf in pu_flags])
        max_val = max(n_hs.max(), n_pu.max()) + 1
        bins_hspu = np.arange(-0.5, max_val + 1.5, 1)
        ax.hist(n_hs, bins=bins_hspu, histtype="step", linewidth=1.5, color="C0", label="HS jets")
        ax.hist(n_pu, bins=bins_hspu, histtype="step", linewidth=1.5, color="C3", label="PU jets")
        ax.set_xlabel("Number of jets per event")
        ax.set_ylabel("Events")
        ax.set_title("HS vs PU jets per event")
        ax.legend()
    else:
        ax.text(
            0.5,
            0.5,
            "No pileup present",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=14,
            color="gray",
        )
        ax.set_title("HS vs PU jets per event")

    # 4. Flavor composition bar chart
    ax = axes1[1, 1]
    thresholds = [1, 2, 3]
    bar_labels = []
    bar_values = []
    for fid in [5, 4, 15, 0]:  # b, c, tau, light
        fname = LABEL_MAP.get(fid, str(fid))
        for thr in thresholds:
            frac = np.mean(flavor_counts[fid] >= thr)
            bar_labels.append(f"{fname}\n>={thr}")
            bar_values.append(frac)
    x = np.arange(len(bar_labels))
    colors = []
    for fid in [5, 4, 15, 0]:
        colors.extend([LABEL_COLORS.get(fid, "gray")] * len(thresholds))
    ax.bar(x, bar_values, color=colors, alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(bar_labels, fontsize=7)
    ax.set_ylabel("Fraction of events")
    ax.set_title("Flavor composition")
    ax.set_ylim(0, 1.05)

    fig1.tight_layout()

    # ---- Page 2: Event kinematics (2x2) ----
    fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))
    fig2.suptitle(f"Event Kinematics ({n_events} events)", fontsize=14)

    # Pre-compute per-event quantities
    n_constit_per_event = np.zeros(n_events, dtype=np.int64)
    ht_per_event = np.zeros(n_events, dtype=np.float64)
    lead_pt_per_event = np.zeros(n_events, dtype=np.float64)
    for i, jg in enumerate(jet_groups):
        n_constit_per_event[i] = np.sum(jg["n_constituents"])
        ht_per_event[i] = np.sum(jg["pt"])
        lead_pt_per_event[i] = np.max(jg["pt"])

    # 5. Constituents per event
    ax = axes2[0, 0]
    ax.hist(n_constit_per_event, bins=50, histtype="step", color="black", linewidth=1.5)
    ax.set_xlabel("Total constituents per event")
    ax.set_ylabel("Events")
    ax.set_title("Constituents per event")

    # 6. HT per event
    ax = axes2[0, 1]
    ax.hist(ht_per_event / 1e3, bins=50, histtype="step", color="black", linewidth=1.5)
    ax.set_xlabel("HT [TeV]")
    ax.set_ylabel("Events")
    ax.set_title("HT (scalar sum jet pT)")

    # 7. Leading jet pT per event
    ax = axes2[1, 0]
    ax.hist(lead_pt_per_event / 1e3, bins=50, histtype="step", color="black", linewidth=1.5)
    ax.set_xlabel("Leading jet pT [TeV]")
    ax.set_ylabel("Events")
    ax.set_title("Leading jet pT")

    # 8. Leading jet pT vs HT (2D)
    ax = axes2[1, 1]
    h = ax.hist2d(
        ht_per_event / 1e3,
        lead_pt_per_event / 1e3,
        bins=50,
        cmin=1,
    )
    fig2.colorbar(h[3], ax=ax, label="Events")
    ax.set_xlabel("HT [TeV]")
    ax.set_ylabel("Leading jet pT [TeV]")
    ax.set_title("Leading jet pT vs HT")

    fig2.tight_layout()

    return [fig1, fig2]


def main():
    parser = argparse.ArgumentParser(description="Plot event-level distributions")
    parser.add_argument("input", help="Input HDF5 file")
    parser.add_argument("-o", "--output", default="events.pdf", help="Output PDF path")
    args = parser.parse_args()

    jets, constit = load_data(args.input)
    figs = plot(jets, constit)
    save_figs(figs, args.output)
