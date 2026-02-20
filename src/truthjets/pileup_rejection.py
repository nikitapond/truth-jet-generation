from __future__ import annotations

import awkward as ak
import numpy as np


def softkiller(particles, grid_size: float = 0.4, rapidity_max: float = 5.0):
    """Apply SoftKiller pileup mitigation (arXiv:1407.0408) before clustering.

    Divides rapidity-phi space into patches, finds the median of the
    per-patch maximum pT, and removes all particles below that threshold.

    Parameters
    ----------
    particles : ak.Array
        Particle records (events x particles) with fields {px, py, pz, E, ...}.
    grid_size : float
        Patch size in rapidity and phi (default 0.4).
    rapidity_max : float
        Maximum |rapidity| coverage (default 5.0).

    Returns
    -------
    filtered : ak.Array
        Particles surviving the SoftKiller cut.
    """
    px = particles.px
    py = particles.py
    pz = particles.pz
    E = particles.E

    pt = np.sqrt(px**2 + py**2)
    # Rapidity y = 0.5 * ln((E + pz) / (E - pz))
    ep = E + pz
    em = E - pz
    # Protect against division by zero / log of non-positive
    safe_em = ak.where(em > 0, em, 1e-10)
    safe_ep = ak.where(ep > 0, ep, 1e-10)
    rapidity = 0.5 * np.log(safe_ep / safe_em)
    phi = np.arctan2(py, px)

    # Define grid bins
    n_y_bins = max(1, int(np.ceil(2 * rapidity_max / grid_size)))
    n_phi_bins = max(1, int(np.ceil(2 * np.pi / grid_size)))
    n_patches = n_y_bins * n_phi_bins

    y_edges = np.linspace(-rapidity_max, rapidity_max, n_y_bins + 1)
    phi_edges = np.linspace(-np.pi, np.pi, n_phi_bins + 1)

    # Vectorized SoftKiller: flatten all particles, compute patch indices
    # with a per-event offset, then scatter-max into a (n_events x n_patches) array.
    n_events = len(particles)
    counts = ak.num(pt, axis=1)
    counts_np = ak.to_numpy(counts)

    flat_pt = ak.to_numpy(ak.flatten(pt)).astype(np.float32)
    flat_y = ak.to_numpy(ak.flatten(rapidity)).astype(np.float32)
    flat_phi = ak.to_numpy(ak.flatten(phi)).astype(np.float32)

    # Digitize all particles at once
    y_bin = np.clip(np.digitize(flat_y, y_edges) - 1, 0, n_y_bins - 1)
    phi_bin = np.clip(np.digitize(flat_phi, phi_edges) - 1, 0, n_phi_bins - 1)
    patch_idx = y_bin * n_phi_bins + phi_bin

    # Event index per particle
    event_idx = np.repeat(np.arange(n_events), counts_np)

    # Global index into (n_events x n_patches) flattened array
    global_patch_idx = event_idx * n_patches + patch_idx

    # Scatter max pT into per-event patches
    max_pt_grid = np.zeros(n_events * n_patches, dtype=np.float32)
    np.maximum.at(max_pt_grid, global_patch_idx, flat_pt)
    max_pt_grid = max_pt_grid.reshape(n_events, n_patches)

    # Median of per-patch max pT gives the SoftKiller threshold per event
    pt_cuts = np.median(max_pt_grid, axis=1).astype(np.float32)

    # Broadcast pt_cut to particle level and apply mask
    pt_cut_bcast = ak.Array(pt_cuts)[:, np.newaxis] * ak.ones_like(pt)
    keep = pt >= pt_cut_bcast

    return particles[keep]


def vertex_z_filter(constits, jet_kin, max_dz: float):
    """Reject jets based on pT-weighted mean vertex z of constituents.

    Parameters
    ----------
    constits : ak.Array
        Constituent particles per jet (events x jets x constituents)
        with fields including {px, py, vz}.
    jet_kin : ak.Array
        Jet kinematics (events x jets) with field {pt, ...}.
    max_dz : float
        Maximum allowed |<vz>_jet| in mm. Jets exceeding this are rejected.

    Returns
    -------
    mask : ak.Array
        Boolean mask (events x jets), True for jets to keep.
    """
    c_px = constits.px
    c_py = constits.py
    c_pt = np.sqrt(c_px**2 + c_py**2)
    c_vz = constits.vz

    # pT-weighted mean vz per jet
    pt_sum = ak.sum(c_pt, axis=-1)
    vz_weighted = ak.sum(c_pt * c_vz, axis=-1)
    mean_vz = ak.where(pt_sum > 0, vz_weighted / pt_sum, 0.0)

    # HS primary vertex is at vz = 0
    return np.abs(mean_vz) <= max_dz
