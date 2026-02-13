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

    # Process each event to find its pT cut
    # Flatten to per-event processing
    n_events = len(particles)
    pt_cuts = np.zeros(n_events, dtype=np.float32)

    flat_pt = ak.to_list(pt)
    flat_rapidity = ak.to_list(rapidity)
    flat_phi = ak.to_list(phi)

    for ievt in range(n_events):
        evt_pt = np.asarray(flat_pt[ievt], dtype=np.float32)
        evt_y = np.asarray(flat_rapidity[ievt], dtype=np.float32)
        evt_phi = np.asarray(flat_phi[ievt], dtype=np.float32)

        if len(evt_pt) == 0:
            pt_cuts[ievt] = 0.0
            continue

        # Digitize particles into grid bins
        y_bin = np.clip(np.digitize(evt_y, y_edges) - 1, 0, n_y_bins - 1)
        phi_bin = np.clip(np.digitize(evt_phi, phi_edges) - 1, 0, n_phi_bins - 1)
        patch_idx = y_bin * n_phi_bins + phi_bin

        # Find max pT per patch (patches with no particles contribute 0)
        max_pt_per_patch = np.zeros(n_patches, dtype=np.float32)
        np.maximum.at(max_pt_per_patch, patch_idx, evt_pt)

        pt_cuts[ievt] = np.median(max_pt_per_patch)

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
