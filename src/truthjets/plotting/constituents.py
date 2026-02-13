"""Constituent-level distribution plots."""
from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np

from truthjets.plotting import (
    LABEL_COLORS,
    LABEL_MAP,
    get_label_info,
    flavor_hist,
    load_data,
    save_figs,
)

# Particle type classification (matches jets.py)
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

_PARTICLE_CATEGORIES = [
    "charged hadron", "neutral hadron", "photon", "electron", "muon", "other",
]
_PARTICLE_COLORS = {
    "charged hadron": "C0",
    "neutral hadron": "C4",
    "photon": "C1",
    "electron": "C2",
    "muon": "C3",
    "other": "C7",
}


def _classify_pdgid(pdgids):
    """Map array of pdgId to category strings."""
    abs_ids = np.abs(pdgids)
    cats = np.full(len(abs_ids), "other", dtype=object)
    for pid, cat in _PDG_TO_CATEGORY.items():
        cats[abs_ids == pid] = cat
    return cats


def _flatten_valid(constit, jets):
    """Extract flat arrays for all valid constituents, plus per-constituent jet label.

    Returns dict with keys: pt, energy, deta, dphi, dr, pdgId, jet_label, jet_pt.
    """
    valid = constit["valid"]
    flat_valid = valid.ravel()

    # Broadcast jet-level quantities to (n_jets, max_c) then flatten
    n_jets, max_c = valid.shape
    lab_bcast = np.broadcast_to(
        jets["HadronConeExclTruthLabelID"][:, None], (n_jets, max_c)
    )
    jpt_bcast = np.broadcast_to(jets["pt"][:, None], (n_jets, max_c))

    deta = constit["deta"]
    dphi = constit["dphi"]

    return {
        "pt": constit["pt"].ravel()[flat_valid],
        "energy": constit["energy"].ravel()[flat_valid],
        "deta": deta.ravel()[flat_valid],
        "dphi": dphi.ravel()[flat_valid],
        "dr": np.sqrt(deta**2 + dphi**2).ravel()[flat_valid],
        "pdgId": constit["pdgId"].ravel()[flat_valid],
        "jet_label": lab_bcast.ravel()[flat_valid],
        "jet_pt": jpt_bcast.ravel()[flat_valid],
    }


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
    flat = _flatten_valid(constit, jets)
    n_constit = len(flat["pt"])

    figs = []
    figs.append(_plot_core_kinematics(flat, unique_labels, n_constit))
    figs.append(_plot_ordering(jets, constit, labels, unique_labels))
    figs.append(_plot_particle_composition(jets, constit, labels, unique_labels, flat))
    figs.append(_plot_2d_distributions(jets, constit, flat, unique_labels, n_constit))
    return figs


# ---------------------------------------------------------------------------
# Page 1: core constituent kinematics
# ---------------------------------------------------------------------------

def _plot_core_kinematics(flat, unique_labels, n_constit):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Constituent Kinematics ({n_constit:,} constituents)", fontsize=14)

    jlab = flat["jet_label"]
    jlab_unique = unique_labels

    # 1. pT spectrum
    ax = axes[0, 0]
    pt_bins = np.geomspace(0.1, max(flat["pt"].max(), 1), 60)
    ax.hist(flat["pt"], bins=pt_bins, histtype="step", color="black", linewidth=1.5, label="All")
    for lab in jlab_unique:
        m = jlab == lab
        ax.hist(flat["pt"][m], bins=pt_bins, histtype="stepfilled", alpha=0.3,
                color=LABEL_COLORS.get(lab, "gray"), label=LABEL_MAP.get(lab, str(lab)))
    ax.set_xlabel("Constituent $p_T$ [GeV]")
    ax.set_ylabel("Constituents")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    # 2. Energy spectrum
    ax = axes[0, 1]
    e_bins = np.geomspace(0.1, max(flat["energy"].max(), 1), 60)
    ax.hist(flat["energy"], bins=e_bins, histtype="step", color="black", linewidth=1.5, label="All")
    for lab in jlab_unique:
        m = jlab == lab
        ax.hist(flat["energy"][m], bins=e_bins, histtype="stepfilled", alpha=0.3,
                color=LABEL_COLORS.get(lab, "gray"), label=LABEL_MAP.get(lab, str(lab)))
    ax.set_xlabel("Constituent energy [GeV]")
    ax.set_ylabel("Constituents")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.legend()

    # 3. deta
    ax = axes[1, 0]
    deta_bins = np.linspace(-0.5, 0.5, 60)
    flavor_hist(ax, flat["deta"], deta_bins, jlab, jlab_unique)
    ax.set_xlabel(r"Constituent $\Delta\eta$")
    ax.set_ylabel("Constituents")
    ax.legend()

    # 4. dphi
    ax = axes[1, 1]
    dphi_bins = np.linspace(-0.5, 0.5, 60)
    flavor_hist(ax, flat["dphi"], dphi_bins, jlab, jlab_unique)
    ax.set_xlabel(r"Constituent $\Delta\phi$")
    ax.set_ylabel("Constituents")
    ax.legend()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page 2: constituent ordering & multiplicity
# ---------------------------------------------------------------------------

def _plot_ordering(jets, constit, labels, unique_labels):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Constituent Ordering by $p_T$ Rank", fontsize=14)

    valid = constit["valid"]  # (n_jets, max_c)
    c_pt = constit["pt"].copy()
    c_pt[~valid] = 0.0
    n_jets, max_c = valid.shape

    # dR per constituent
    c_dr = np.sqrt(constit["deta"]**2 + constit["dphi"]**2)
    c_dr[~valid] = np.nan

    # z = constit_pt / jet_pt
    jet_pt_bc = np.broadcast_to(jets["pt"][:, None], (n_jets, max_c))
    c_z = np.where((jet_pt_bc > 0) & valid, c_pt / jet_pt_bc, np.nan)

    max_rank = min(max_c, 40)  # show up to rank 40
    ranks = np.arange(max_rank)

    # 5. Mean constituent pT vs rank
    ax = axes[0, 0]
    for lab in unique_labels:
        m = labels == lab
        means = np.nanmean(np.where(valid[m, :max_rank], c_pt[m, :max_rank], np.nan), axis=0)
        ax.plot(ranks, means, "o-", markersize=3,
                color=LABEL_COLORS.get(lab, "gray"), label=LABEL_MAP.get(lab, str(lab)))
    ax.set_xlabel("Constituent rank (pT-ordered)")
    ax.set_ylabel(r"$\langle p_T \rangle$ [GeV]")
    ax.set_title("Mean constituent $p_T$ vs rank")
    ax.set_yscale("log")
    ax.legend()

    # 6. Mean z vs rank
    ax = axes[0, 1]
    for lab in unique_labels:
        m = labels == lab
        means = np.nanmean(np.where(valid[m, :max_rank], c_z[m, :max_rank], np.nan), axis=0)
        ax.plot(ranks, means, "o-", markersize=3,
                color=LABEL_COLORS.get(lab, "gray"), label=LABEL_MAP.get(lab, str(lab)))
    ax.set_xlabel("Constituent rank (pT-ordered)")
    ax.set_ylabel(r"$\langle z \rangle = \langle p_T^{\mathrm{c}} / p_T^{\mathrm{jet}} \rangle$")
    ax.set_title("Mean fragmentation $z$ vs rank")
    ax.set_yscale("log")
    ax.legend()

    # 7. Mean dR vs rank
    ax = axes[1, 0]
    for lab in unique_labels:
        m = labels == lab
        means = np.nanmean(np.where(valid[m, :max_rank], c_dr[m, :max_rank], np.nan), axis=0)
        ax.plot(ranks, means, "o-", markersize=3,
                color=LABEL_COLORS.get(lab, "gray"), label=LABEL_MAP.get(lab, str(lab)))
    ax.set_xlabel("Constituent rank (pT-ordered)")
    ax.set_ylabel(r"$\langle \Delta R \rangle$")
    ax.set_title(r"Mean $\Delta R$ vs rank")
    ax.legend()

    # 8. Cumulative pT fraction vs top-N
    ax = axes[1, 1]
    # For each jet, cumsum of sorted pT / jet pT
    cumsum_z = np.cumsum(np.where(valid[:, :max_rank], c_pt[:, :max_rank], 0.0), axis=1)
    # Normalise by jet pT
    cum_frac = np.where(jet_pt_bc[:, :max_rank] > 0, cumsum_z / jet_pt_bc[:, :max_rank], 0.0)
    top_n = np.arange(1, max_rank + 1)
    for lab in unique_labels:
        m = labels == lab
        means = np.mean(cum_frac[m], axis=0)
        ax.plot(top_n, means, "o-", markersize=3,
                color=LABEL_COLORS.get(lab, "gray"), label=LABEL_MAP.get(lab, str(lab)))
    ax.set_xlabel("Number of leading constituents (top-N)")
    ax.set_ylabel(r"$\langle \sum_{i=1}^{N} p_T^i \; / \; p_T^{\mathrm{jet}} \rangle$")
    ax.set_title("Cumulative $p_T$ fraction vs top-N")
    ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_ylim(0, 1.1)
    ax.legend()

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page 3: particle composition
# ---------------------------------------------------------------------------

def _plot_particle_composition(jets, constit, labels, unique_labels, flat):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Particle Composition", fontsize=14)

    valid = constit["valid"]
    n_jets, max_c = valid.shape
    abs_pdg = np.abs(constit["pdgId"])

    # Classify every constituent slot (invalid ones don't matter, we mask them)
    cats_2d = np.full((n_jets, max_c), "other", dtype=object)
    for pid, cat in _PDG_TO_CATEGORY.items():
        cats_2d[abs_pdg == pid] = cat

    # Per-jet counts of each category
    per_jet_counts = {}
    for cat in _PARTICLE_CATEGORIES:
        per_jet_counts[cat] = np.sum((cats_2d == cat) & valid, axis=1).astype(float)

    # Per-jet total valid
    n_valid = np.sum(valid, axis=1).astype(float)
    n_valid_safe = np.where(n_valid > 0, n_valid, 1.0)

    # 9. Particle type composition per flavor (stacked bar)
    ax = axes[0, 0]
    flavor_labels_str = [LABEL_MAP.get(f, str(f)) for f in unique_labels]
    x = np.arange(len(unique_labels))
    width = 0.6
    bottom = np.zeros(len(unique_labels))
    for cat in _PARTICLE_CATEGORIES:
        fracs = []
        for fid in unique_labels:
            m = labels == fid
            fracs.append(np.mean(per_jet_counts[cat][m] / n_valid_safe[m]) if np.any(m) else 0)
        fracs = np.array(fracs)
        ax.bar(x, fracs, width, bottom=bottom, label=cat,
               color=_PARTICLE_COLORS.get(cat, "gray"), alpha=0.8)
        bottom += fracs
    ax.set_xticks(x)
    ax.set_xticklabels(flavor_labels_str)
    ax.set_ylabel("Mean fraction of constituents")
    ax.set_title("Particle type composition by flavor")
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8, loc="upper right")

    # 10. Mean number of each particle type per jet, by flavor (grouped bar)
    ax = axes[0, 1]
    n_cats = len(_PARTICLE_CATEGORIES)
    n_flavors = len(unique_labels)
    bar_width = 0.8 / n_flavors
    for i, fid in enumerate(unique_labels):
        m = labels == fid
        means = [np.mean(per_jet_counts[cat][m]) for cat in _PARTICLE_CATEGORIES]
        offsets = np.arange(n_cats) + i * bar_width - 0.4 + bar_width / 2
        ax.bar(offsets, means, bar_width, label=LABEL_MAP.get(fid, str(fid)),
               color=LABEL_COLORS.get(fid, "gray"), alpha=0.8)
    ax.set_xticks(np.arange(n_cats))
    ax.set_xticklabels(_PARTICLE_CATEGORIES, fontsize=8, rotation=30, ha="right")
    ax.set_ylabel("Mean count per jet")
    ax.set_title("Mean particle count by type and flavor")
    ax.legend(fontsize=8)

    # 11. Charged vs neutral pT fraction by flavor
    ax = axes[1, 0]
    # Charged = charged hadron + electron + muon; neutral = neutral hadron + photon
    c_pt_arr = constit["pt"].copy()
    c_pt_arr[~valid] = 0.0
    charged_mask = np.isin(cats_2d, ["charged hadron", "electron", "muon"]) & valid
    neutral_mask = np.isin(cats_2d, ["neutral hadron", "photon"]) & valid
    charged_pt = np.sum(np.where(charged_mask, c_pt_arr, 0.0), axis=1)
    neutral_pt = np.sum(np.where(neutral_mask, c_pt_arr, 0.0), axis=1)
    total_pt = charged_pt + neutral_pt
    total_pt_safe = np.where(total_pt > 0, total_pt, 1.0)

    x = np.arange(len(unique_labels))
    charged_fracs = [np.mean(charged_pt[labels == fid] / total_pt_safe[labels == fid])
                     for fid in unique_labels]
    neutral_fracs = [np.mean(neutral_pt[labels == fid] / total_pt_safe[labels == fid])
                     for fid in unique_labels]
    ax.bar(x, charged_fracs, 0.6, label="Charged", color="C0", alpha=0.8)
    ax.bar(x, neutral_fracs, 0.6, bottom=charged_fracs, label="Neutral", color="C1", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(flavor_labels_str)
    ax.set_ylabel("Mean $p_T$ fraction")
    ax.set_title("Charged vs neutral $p_T$ fraction")
    ax.set_ylim(0, 1.1)
    ax.legend()

    # 12. Particle type pT spectrum (split by particle type)
    ax = axes[1, 1]
    flat_cats = _classify_pdgid(flat["pdgId"])
    pt_bins = np.geomspace(0.1, max(flat["pt"].max(), 1), 60)
    ax.hist(flat["pt"], bins=pt_bins, histtype="step", color="black", linewidth=1.5, label="All")
    for cat in _PARTICLE_CATEGORIES:
        m = flat_cats == cat
        if not np.any(m):
            continue
        ax.hist(flat["pt"][m], bins=pt_bins, histtype="stepfilled", alpha=0.3,
                color=_PARTICLE_COLORS.get(cat, "gray"), label=cat)
    ax.set_xlabel("Constituent $p_T$ [GeV]")
    ax.set_ylabel("Constituents")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("$p_T$ spectrum by particle type")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Page 4: 2D constituent distributions
# ---------------------------------------------------------------------------

def _plot_2d_distributions(jets, constit, flat, unique_labels, n_constit):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"2D Constituent Distributions ({n_constit:,} constituents)", fontsize=14)

    jlab = flat["jet_label"]

    # 13. deta vs dphi (all)
    ax = axes[0, 0]
    h = ax.hist2d(
        flat["deta"], flat["dphi"],
        bins=[np.linspace(-0.5, 0.5, 60), np.linspace(-0.5, 0.5, 60)],
        cmin=1,
    )
    fig.colorbar(h[3], ax=ax, label="Constituents")
    ax.set_xlabel(r"$\Delta\eta$")
    ax.set_ylabel(r"$\Delta\phi$")
    ax.set_title(r"$\Delta\eta$ vs $\Delta\phi$ (all)")
    ax.set_aspect("equal")

    # 14. deta vs dphi contours by flavor
    ax = axes[0, 1]
    bins_2d = [np.linspace(-0.5, 0.5, 40), np.linspace(-0.5, 0.5, 40)]
    for lab in unique_labels:
        m = jlab == lab
        if np.sum(m) < 10:
            continue
        H, xedges, yedges = np.histogram2d(flat["deta"][m], flat["dphi"][m], bins=bins_2d)
        if H.max() > 0:
            H = H / H.max()
        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        ax.contour(
            xc, yc, H.T, levels=[0.1, 0.3, 0.5, 0.7, 0.9],
            colors=LABEL_COLORS.get(lab, "gray"), linewidths=1.2,
        )
        ax.plot([], [], color=LABEL_COLORS.get(lab, "gray"),
                label=LABEL_MAP.get(lab, str(lab)))
    ax.set_xlabel(r"$\Delta\eta$")
    ax.set_ylabel(r"$\Delta\phi$")
    ax.set_title(r"$\Delta\eta$-$\Delta\phi$ contours by flavor")
    ax.set_aspect("equal")
    ax.legend(fontsize=9)

    # 15. Constituent pT vs dR
    ax = axes[1, 0]
    h = ax.hist2d(
        flat["dr"], flat["pt"],
        bins=[np.linspace(0, 0.5, 50), np.geomspace(0.1, max(flat["pt"].max(), 1), 50)],
        cmin=1,
    )
    fig.colorbar(h[3], ax=ax, label="Constituents")
    ax.set_xlabel(r"$\Delta R$ from jet axis")
    ax.set_ylabel("Constituent $p_T$ [GeV]")
    ax.set_yscale("log")
    ax.set_title(r"$p_T$ vs $\Delta R$")

    # 16. Constituent log(pT) vs rank
    ax = axes[1, 1]
    valid = constit["valid"]
    n_jets_2d, max_c = valid.shape
    rank_2d = np.broadcast_to(np.arange(max_c)[None, :], (n_jets_2d, max_c))
    flat_rank = rank_2d.ravel()[valid.ravel()]
    max_rank = min(max_c, 40)
    rank_mask = flat_rank < max_rank
    h = ax.hist2d(
        flat_rank[rank_mask].astype(float), flat["pt"][rank_mask],
        bins=[np.arange(-0.5, max_rank + 0.5, 1),
              np.geomspace(0.1, max(flat["pt"].max(), 1), 50)],
        cmin=1,
    )
    fig.colorbar(h[3], ax=ax, label="Constituents")
    ax.set_xlabel("Constituent rank ($p_T$-ordered)")
    ax.set_ylabel("Constituent $p_T$ [GeV]")
    ax.set_yscale("log")
    ax.set_title("Constituent $p_T$ vs rank")

    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description="Plot constituent-level distributions")
    parser.add_argument("input", help="Input HDF5 file")
    parser.add_argument("-o", "--output", default="constituents.pdf", help="Output PDF path")
    args = parser.parse_args()

    jets, constit = load_data(args.input)
    figs = plot(jets, constit)
    save_figs(figs, args.output)
