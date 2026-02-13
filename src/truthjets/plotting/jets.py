"""Jet-level distribution plots."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from truthjets.plotting import (
    LABEL_COLORS,
    LABEL_MAP,
    get_label_info,
    flavor_hist,
    flavor_profile,
    load_data,
    save_figs,
)


def _compute_substructure(jets, constit):
    """Pre-compute per-jet substructure variables from constituents.

    Returns dict with jet_width, constit_dr, constit_z arrays.
    """
    valid = constit["valid"]  # (n_jets, max_c)
    c_pt = constit["pt"]
    c_deta = constit["deta"]
    c_dphi = constit["dphi"]
    c_dr = np.sqrt(c_deta**2 + c_dphi**2)

    # Jet width: pT-weighted mean dR
    pt_dr = c_pt * c_dr
    pt_dr[~valid] = 0.0
    c_pt_masked = c_pt.copy()
    c_pt_masked[~valid] = 0.0
    sum_pt = np.sum(c_pt_masked, axis=1)
    jet_width = np.where(sum_pt > 0, np.sum(pt_dr, axis=1) / sum_pt, 0.0)

    # Flat arrays for all valid constituents (for 1D histograms)
    flat_valid = valid.ravel()
    flat_dr = c_dr.ravel()[flat_valid]

    # Fragmentation z = constit_pT / jet_pT (broadcast jet pT)
    jet_pt_bcast = np.broadcast_to(jets["pt"][:, None], c_pt.shape)
    c_z = np.where(jet_pt_bcast > 0, c_pt / jet_pt_bcast, 0.0)
    flat_z = c_z.ravel()[flat_valid]

    # Per-constituent jet label (for flavor-split histograms)
    lab_bcast = np.broadcast_to(
        jets["HadronConeExclTruthLabelID"][:, None], c_pt.shape
    )
    flat_lab = lab_bcast.ravel()[flat_valid]

    return {
        "jet_width": jet_width,
        "flat_dr": flat_dr,
        "flat_z": flat_z,
        "flat_lab": flat_lab,
    }


def plot(jets: np.ndarray, constit: np.ndarray) -> list[plt.Figure]:
    """Create jet-level distribution figures.

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
    figs = []

    # Pre-compute substructure for later pages
    sub = _compute_substructure(jets, constit)

    # --- Page 1: core jet kinematics ---
    fig, axes = plt.subplots(3, 2, figsize=(12, 14))
    fig.suptitle(f"Jet Distributions ({len(jets)} jets)", fontsize=14, y=0.98)

    # Jet pT
    ax = axes[0, 0]
    flavor_hist(ax, jets["pt"], np.linspace(20, 400, 50), labels, unique_labels)
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel("Jets")
    ax.set_yscale("log")
    ax.legend()

    # Jet eta
    ax = axes[0, 1]
    flavor_hist(ax, jets["eta"], np.linspace(-2.5, 2.5, 50), labels, unique_labels)
    ax.set_xlabel(r"Jet $\eta$")
    ax.set_ylabel("Jets")
    ax.legend()

    # Jet mass
    ax = axes[1, 0]
    flavor_hist(ax, jets["mass"], np.linspace(0, 50, 50), labels, unique_labels)
    ax.set_xlabel("Jet mass [GeV]")
    ax.set_ylabel("Jets")
    ax.legend()

    # Flavor composition
    ax = axes[1, 1]
    counts = [np.sum(labels == lab) for lab in unique_labels]
    pie_labels = [LABEL_MAP.get(lab, str(lab)) for lab in unique_labels]
    colors = [LABEL_COLORS.get(lab, "gray") for lab in unique_labels]
    ax.pie(counts, labels=pie_labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.set_title("Flavor Composition")

    # Number of constituents
    ax = axes[2, 0]
    nc = jets["n_constituents"]
    flavor_hist(ax, nc, np.arange(0, nc.max() + 2) - 0.5, labels, unique_labels)
    ax.set_xlabel("Number of constituents")
    ax.set_ylabel("Jets")
    ax.legend()

    # Leading constituent pT fraction
    ax = axes[2, 1]
    lead_pt = constit["pt"][:, 0]
    valid_mask = jets["pt"] > 0
    frac = lead_pt / np.where(valid_mask, jets["pt"], 1.0)
    bins = np.linspace(0, 1, 50)
    ax.hist(frac[valid_mask], bins=bins, histtype="step", color="black", linewidth=1.5, label="All")
    for lab in unique_labels:
        mask = (labels == lab) & valid_mask
        ax.hist(
            lead_pt[mask] / jets["pt"][mask],
            bins=bins,
            histtype="stepfilled",
            alpha=0.3,
            color=LABEL_COLORS.get(lab, "gray"),
            label=LABEL_MAP.get(lab, str(lab)),
        )
    ax.set_xlabel("Leading constituent $p_T$ / Jet $p_T$")
    ax.set_ylabel("Jets")
    ax.legend()

    plt.tight_layout()
    figs.append(fig)

    # --- Page 2: kinematics II ---
    figs.append(_plot_kinematics_ii(jets, labels, unique_labels))

    # --- Page 3: substructure ---
    figs.append(_plot_substructure(jets, constit, labels, unique_labels, sub))

    # --- Page 4: 2D correlations ---
    figs.append(_plot_2d_correlations(jets, constit, labels, unique_labels, sub))

    # --- Page 5: flavor-comparison profiles ---
    figs.append(_plot_flavor_profiles(jets, labels, unique_labels, sub))

    # --- Jet display pages (one per flavor) ---
    figs.extend(_plot_jet_displays(jets, constit, labels, unique_labels))

    # --- Pileup page (if present) ---
    has_pu = "pt_frac_pu" in jets.dtype.names and np.any(jets["pt_frac_pu"] > 0)
    if has_pu:
        figs.append(_plot_pileup(jets, constit, labels, unique_labels))

    return figs


def _plot_kinematics_ii(jets, labels, unique_labels):
    """Page 2: additional jet kinematics."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Jet Kinematics II", fontsize=14)

    # Jet phi
    ax = axes[0, 0]
    flavor_hist(ax, jets["phi"], np.linspace(-np.pi, np.pi, 50), labels, unique_labels)
    ax.set_xlabel(r"Jet $\phi$")
    ax.set_ylabel("Jets")
    ax.legend()

    # Jet energy
    ax = axes[0, 1]
    flavor_hist(ax, jets["energy"], np.linspace(0, 1000, 50), labels, unique_labels)
    ax.set_xlabel("Jet energy [GeV]")
    ax.set_ylabel("Jets")
    ax.set_yscale("log")
    ax.legend()

    # Jet pT (log-spaced bins)
    ax = axes[1, 0]
    log_bins = np.geomspace(20, max(jets["pt"].max(), 21), 50)
    flavor_hist(ax, jets["pt"], log_bins, labels, unique_labels)
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel("Jets")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    # Jet mass vs pT (2D)
    ax = axes[1, 1]
    h = ax.hist2d(
        jets["pt"], jets["mass"],
        bins=[np.linspace(20, 400, 50), np.linspace(0, 50, 50)],
        cmin=1,
    )
    fig.colorbar(h[3], ax=ax, label="Jets")
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel("Jet mass [GeV]")
    ax.set_title("Mass vs $p_T$")

    fig.tight_layout()
    return fig


def _plot_substructure(jets, constit, labels, unique_labels, sub):
    """Page 3: substructure / constituent summary."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Jet Substructure", fontsize=14)

    # Jet width by flavor
    ax = axes[0, 0]
    flavor_hist(ax, sub["jet_width"], np.linspace(0, 0.5, 50), labels, unique_labels)
    ax.set_xlabel(r"Jet width $\sum p_T^i \Delta R_i / \sum p_T^i$")
    ax.set_ylabel("Jets")
    ax.legend()

    # N constituents vs pT (2D)
    ax = axes[0, 1]
    nc = jets["n_constituents"]
    h = ax.hist2d(
        jets["pt"], nc.astype(float),
        bins=[np.linspace(20, 400, 50), np.arange(0, nc.max() + 2) - 0.5],
        cmin=1,
    )
    fig.colorbar(h[3], ax=ax, label="Jets")
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel("Number of constituents")
    ax.set_title("N constituents vs $p_T$")

    # Constituent dR distribution by flavor
    ax = axes[1, 0]
    flat_dr = sub["flat_dr"]
    flat_lab = sub["flat_lab"]
    flat_unique = sorted(set(flat_lab))
    dr_bins = np.linspace(0, 0.5, 50)
    ax.hist(flat_dr, bins=dr_bins, histtype="step", color="black", linewidth=1.5, label="All")
    for lab in flat_unique:
        mask = flat_lab == lab
        ax.hist(
            flat_dr[mask], bins=dr_bins, histtype="stepfilled", alpha=0.3,
            color=LABEL_COLORS.get(lab, "gray"), label=LABEL_MAP.get(lab, str(lab)),
        )
    ax.set_xlabel(r"Constituent $\Delta R$ from jet axis")
    ax.set_ylabel("Constituents")
    ax.set_yscale("log")
    ax.legend()

    # Fragmentation z by flavor
    ax = axes[1, 1]
    flat_z = sub["flat_z"]
    z_bins = np.linspace(0, 1, 50)
    ax.hist(flat_z, bins=z_bins, histtype="step", color="black", linewidth=1.5, label="All")
    for lab in flat_unique:
        mask = flat_lab == lab
        ax.hist(
            flat_z[mask], bins=z_bins, histtype="stepfilled", alpha=0.3,
            color=LABEL_COLORS.get(lab, "gray"), label=LABEL_MAP.get(lab, str(lab)),
        )
    ax.set_xlabel(r"$z = p_T^{\mathrm{constit}} / p_T^{\mathrm{jet}}$")
    ax.set_ylabel("Constituents")
    ax.set_yscale("log")
    ax.legend()

    fig.tight_layout()
    return fig


def _plot_2d_correlations(jets, constit, labels, unique_labels, sub):
    """Page 4: 2D correlations."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("2D Correlations", fontsize=14)

    # eta vs phi occupancy
    ax = axes[0, 0]
    h = ax.hist2d(
        jets["eta"], jets["phi"],
        bins=[np.linspace(-2.5, 2.5, 50), np.linspace(-np.pi, np.pi, 50)],
        cmin=1,
    )
    fig.colorbar(h[3], ax=ax, label="Jets")
    ax.set_xlabel(r"Jet $\eta$")
    ax.set_ylabel(r"Jet $\phi$")
    ax.set_title(r"$\eta$-$\phi$ occupancy")

    # Mean N constituents vs pT (profile by flavor)
    ax = axes[0, 1]
    pt_bins = np.linspace(20, 400, 25)
    nc = jets["n_constituents"].astype(float)
    flavor_profile(ax, jets["pt"], nc, pt_bins, labels, unique_labels)
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel(r"$\langle$ N constituents $\rangle$")
    ax.set_title("Mean N constituents vs $p_T$")
    ax.legend()

    # Mass vs N constituents (2D)
    ax = axes[1, 0]
    h = ax.hist2d(
        nc, jets["mass"],
        bins=[np.arange(0, nc.max() + 2) - 0.5, np.linspace(0, 50, 50)],
        cmin=1,
    )
    fig.colorbar(h[3], ax=ax, label="Jets")
    ax.set_xlabel("Number of constituents")
    ax.set_ylabel("Jet mass [GeV]")
    ax.set_title("Mass vs N constituents")

    # Leading constit pT fraction vs jet pT (profile by flavor)
    ax = axes[1, 1]
    lead_pt = constit["pt"][:, 0]
    valid_mask = jets["pt"] > 0
    lead_frac = np.where(valid_mask, lead_pt / jets["pt"], np.nan)
    flavor_profile(ax, jets["pt"], lead_frac, pt_bins, labels, unique_labels)
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel(r"$\langle$ Leading constit $p_T$ / jet $p_T$ $\rangle$")
    ax.set_title("Leading constituent fraction vs $p_T$")
    ax.legend()

    fig.tight_layout()
    return fig


def _plot_flavor_profiles(jets, labels, unique_labels, sub):
    """Page 5: flavor-comparison profile plots."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Flavor Profiles vs Jet $p_T$", fontsize=14)

    pt_bins = np.linspace(20, 400, 25)

    # Mean mass vs pT by flavor
    ax = axes[0, 0]
    flavor_profile(ax, jets["pt"], jets["mass"], pt_bins, labels, unique_labels)
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel(r"$\langle$ Jet mass $\rangle$ [GeV]")
    ax.set_title("Mean jet mass vs $p_T$")
    ax.legend()

    # Mean width vs pT by flavor
    ax = axes[0, 1]
    flavor_profile(ax, jets["pt"], sub["jet_width"], pt_bins, labels, unique_labels)
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel(r"$\langle$ Jet width $\rangle$")
    ax.set_title("Mean jet width vs $p_T$")
    ax.legend()

    # Mean N constituents vs pT by flavor
    ax = axes[1, 0]
    nc = jets["n_constituents"].astype(float)
    flavor_profile(ax, jets["pt"], nc, pt_bins, labels, unique_labels)
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel(r"$\langle$ N constituents $\rangle$")
    ax.set_title("Mean N constituents vs $p_T$")
    ax.legend()

    # b-jet purity vs pT
    ax = axes[1, 1]
    pt_centers = 0.5 * (pt_bins[:-1] + pt_bins[1:])
    for fid in [5, 4, 15]:
        if fid not in unique_labels:
            continue
        fracs = []
        for lo, hi in zip(pt_bins[:-1], pt_bins[1:]):
            sel = (jets["pt"] >= lo) & (jets["pt"] < hi)
            n_total = np.sum(sel)
            fracs.append(np.sum(labels[sel] == fid) / n_total if n_total > 0 else np.nan)
        ax.plot(
            pt_centers, fracs, "o-",
            color=LABEL_COLORS.get(fid, "gray"),
            label=LABEL_MAP.get(fid, str(fid)),
            markersize=3,
        )
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel("Fraction of jets")
    ax.set_title("Flavor fraction vs $p_T$")
    ax.set_ylim(0, None)
    ax.legend()

    fig.tight_layout()
    return fig


# Particle type -> marker mapping for jet displays
_PARTICLE_MARKERS = {
    "charged hadron": ("o", "C0"),   # pi+-, K+-, p
    "neutral hadron": ("s", "C4"),   # K0L, n
    "photon": ("*", "C1"),           # gamma
    "electron": ("^", "C2"),         # e+-
    "muon": ("v", "C3"),             # mu+-
    "other": ("D", "C7"),
}

# Map |pdgId| to particle category
_PDG_TO_CATEGORY = {
    211: "charged hadron",
    321: "charged hadron",
    2212: "charged hadron",
    130: "neutral hadron",
    310: "neutral hadron",
    2112: "neutral hadron",
    311: "neutral hadron",
    22: "photon",
    11: "electron",
    13: "muon",
}

N_DISPLAY_JETS = 5


def _classify_pdgid(pdgids):
    """Map array of pdgId to category strings."""
    abs_ids = np.abs(pdgids)
    cats = np.full(len(abs_ids), "other", dtype=object)
    for pid, cat in _PDG_TO_CATEGORY.items():
        cats[abs_ids == pid] = cat
    return cats


def _plot_single_jet(ax, jet, c_row, jet_R):
    """Draw one jet's constituents in the deta-dphi plane."""
    valid = c_row["valid"]
    deta = c_row["deta"][valid]
    dphi = c_row["dphi"][valid]
    pt = c_row["pt"][valid]
    pdgid = c_row["pdgId"][valid]

    cats = _classify_pdgid(pdgid)

    # Size proportional to 1/pT, normalised so the range is visible
    inv_pt = 1.0 / np.clip(pt, 0.1, None)
    # Scale to marker area [20, 400]
    s_min, s_max = 20, 400
    if inv_pt.max() > inv_pt.min():
        sizes = s_min + (s_max - s_min) * (inv_pt - inv_pt.min()) / (inv_pt.max() - inv_pt.min())
    else:
        sizes = np.full_like(inv_pt, (s_min + s_max) / 2)

    # Draw each particle category with its own marker
    for cat, (marker, color) in _PARTICLE_MARKERS.items():
        mask = cats == cat
        if not np.any(mask):
            continue
        ax.scatter(
            deta[mask], dphi[mask],
            s=sizes[mask], marker=marker, color=color,
            edgecolors="black", linewidths=0.5, alpha=0.85,
            label=cat, zorder=3,
        )

    # Draw jet cone
    circle = plt.Circle((0, 0), jet_R, fill=False, linestyle="--",
                         color="gray", linewidth=1.0, zorder=2)
    ax.add_patch(circle)

    # Axis formatting
    lim = jet_R * 1.4
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.axhline(0, color="gray", linewidth=0.3)
    ax.axvline(0, color="gray", linewidth=0.3)
    ax.set_xlabel(r"$\Delta\eta$")
    ax.set_ylabel(r"$\Delta\phi$")

    ax.set_title(
        f"$p_T$={jet['pt']:.0f} GeV, m={jet['mass']:.1f} GeV, "
        f"$n_{{\\mathrm{{constit}}}}$={jet['n_constituents']}",
        fontsize=9,
    )


def _plot_jet_displays(jets, constit, labels, unique_labels, n_jets=N_DISPLAY_JETS, seed=42):
    """Create one page per flavor showing individual jet constituent displays."""
    rng = np.random.default_rng(seed)
    figs = []

    # Infer jet R from max constituent dR (approximate)
    all_valid = constit["valid"]
    all_dr = np.sqrt(constit["deta"]**2 + constit["dphi"]**2)
    all_dr[~all_valid] = 0.0
    max_dr = np.max(all_dr)
    # Round to nearest standard R value
    jet_R = min([0.2, 0.4, 0.6, 0.8, 1.0, 1.2], key=lambda r: abs(r - max_dr))

    for fid in unique_labels:
        flavor_mask = labels == fid
        flavor_indices = np.where(flavor_mask)[0]
        if len(flavor_indices) == 0:
            continue

        n_pick = min(n_jets, len(flavor_indices))
        chosen = rng.choice(flavor_indices, size=n_pick, replace=False)
        chosen.sort()

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fname = LABEL_MAP.get(fid, str(fid))
        fig.suptitle(
            f"Jet Displays: {fname} ({n_pick} random jets, "
            f"marker size $\\propto\\; 1/p_T$)",
            fontsize=14,
        )

        for j, idx in enumerate(chosen):
            ax = axes[j // 3, j % 3]
            _plot_single_jet(ax, jets[idx], constit[idx], jet_R)

        # Use the last panel for the legend
        ax_leg = axes[1, 2]
        if n_pick <= 5:
            ax_leg.set_axis_off()
            for cat, (marker, color) in _PARTICLE_MARKERS.items():
                ax_leg.scatter([], [], s=80, marker=marker, color=color,
                               edgecolors="black", linewidths=0.5, label=cat)
            ax_leg.legend(loc="center", fontsize=11, frameon=True,
                          title="Particle type", title_fontsize=12)

        fig.tight_layout()
        figs.append(fig)

    return figs


def _plot_pileup(jets, constit, labels, unique_labels):
    """Create pileup-specific distribution page."""
    pt_frac_pu = jets["pt_frac_pu"]
    is_pu = constit["is_pu"]
    valid = constit["valid"]
    n_pu_per_jet = np.sum(is_pu & valid, axis=1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Pileup Distributions", fontsize=14, y=0.98)

    # pt_frac_pu by flavor
    ax = axes[0, 0]
    flavor_hist(ax, pt_frac_pu, np.linspace(0, 1, 50), labels, unique_labels)
    ax.set_xlabel("Pileup $p_T$ fraction per jet")
    ax.set_ylabel("Jets")
    ax.set_yscale("log")
    ax.legend()

    # N PU constituents by flavor
    ax = axes[0, 1]
    max_npu = max(n_pu_per_jet.max(), 1)
    flavor_hist(ax, n_pu_per_jet, np.arange(0, max_npu + 2) - 0.5, labels, unique_labels)
    ax.set_xlabel("Number of PU constituents per jet")
    ax.set_ylabel("Jets")
    ax.legend()

    # pt_frac_pu vs jet pT (2D)
    ax = axes[1, 0]
    h = ax.hist2d(
        jets["pt"], pt_frac_pu,
        bins=[np.linspace(20, 400, 50), np.linspace(0, 1, 50)],
        cmin=1,
    )
    fig.colorbar(h[3], ax=ax, label="Jets")
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel("Pileup $p_T$ fraction")

    # Mean pt_frac_pu vs jet pT profile
    ax = axes[1, 1]
    pt_bins = np.linspace(20, 400, 25)
    flavor_profile(ax, jets["pt"], pt_frac_pu, pt_bins, labels, unique_labels)
    ax.set_xlabel("Jet $p_T$ [GeV]")
    ax.set_ylabel(r"$\langle$ Pileup $p_T$ fraction $\rangle$")
    ax.set_ylim(0, None)
    ax.legend()

    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot jet-level distributions")
    parser.add_argument("input", help="Input HDF5 file")
    parser.add_argument("-o", "--output", default="jets.pdf", help="Output PDF path")
    args = parser.parse_args()

    jets, constit = load_data(args.input)
    figs = plot(jets, constit)
    save_figs(figs, args.output)
