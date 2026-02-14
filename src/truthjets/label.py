from __future__ import annotations

import awkward as ak
import numpy as np


def is_b_hadron(pdg_id):
    """Vectorized check for b-hadrons based on PDG ID convention.

    B hadrons have a quark content digit of 5 in the hundreds, thousands,
    or ten-thousands place.
    """
    aid = np.abs(pdg_id)
    return (
        ((aid // 100) % 10 == 5)
        | ((aid // 1000) % 10 == 5)
        | ((aid // 10000) % 10 == 5)
    )


def is_c_hadron(pdg_id):
    """Vectorized check for c-hadrons based on PDG ID convention.

    C hadrons have a quark content digit of 4 in the hundreds, thousands,
    or ten-thousands place (and are NOT b-hadrons).
    """
    aid = np.abs(pdg_id)
    has_c = (
        ((aid // 100) % 10 == 4)
        | ((aid // 1000) % 10 == 4)
        | ((aid // 10000) % 10 == 4)
    )
    return has_c & ~is_b_hadron(pdg_id)


def is_tau_lepton(pdg_id):
    """Vectorized check for tau leptons (|pdg_id| == 15)."""
    return np.abs(pdg_id) == 15


def final_b_hadron_mask(particles):
    """Return a boolean mask selecting only weakly-decaying b-hadrons.

    A b-hadron is "final" (weakly decaying) if none of its daughters
    in the Pythia event record are also b-hadrons. This filters out
    excited states (B**, B*) that decay via strong/EM interactions
    to ground-state B mesons.

    Parameters
    ----------
    particles : ak.Array
        Pythia particle record (events x particles) with fields
        ``id``, ``daughter1``, ``daughter2``.

    Returns
    -------
    mask : ak.Array
        Boolean mask (events x particles), True for weakly-decaying b-hadrons.
    """
    prt_id = particles.id
    d1 = particles.daughter1
    d2 = particles.daughter2

    b_flag = is_b_hadron(prt_id)

    # Flatten to numpy for efficient prefix-sum computation
    counts = ak.num(prt_id)
    counts_np = ak.to_numpy(counts)
    global_offsets = np.concatenate([[0], np.cumsum(counts_np)])

    flat_b = ak.to_numpy(ak.flatten(b_flag)).astype(np.int32)
    flat_cumsum = np.concatenate([[0], np.cumsum(flat_b)])

    flat_d1 = ak.to_numpy(ak.flatten(d1))
    flat_d2 = ak.to_numpy(ak.flatten(d2))

    # Convert local (per-event) indices to global flat indices
    event_idx = np.repeat(np.arange(len(counts_np)), counts_np)
    global_d1 = flat_d1 + global_offsets[event_idx]
    global_d2 = flat_d2 + global_offsets[event_idx]

    # Particles with d1==0 have no daughters in Pythia convention
    has_daughters = flat_d1 > 0

    # Count b-hadron daughters via prefix sum: sum(b_flag[d1:d2+1])
    safe_d1 = np.where(has_daughters, global_d1, 0)
    safe_d2 = np.where(has_daughters, global_d2, 0)
    n_b_daughters = flat_cumsum[safe_d2 + 1] - flat_cumsum[safe_d1]
    n_b_daughters = np.where(has_daughters, n_b_daughters, 0)

    flat_is_final_b = ak.to_numpy(ak.flatten(b_flag)) & (n_b_daughters == 0)

    return ak.unflatten(flat_is_final_b, counts)


def _delta_phi(phi1, phi2):
    """Compute delta-phi, wrapped to [-pi, pi]."""
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def _delta_r(eta1, phi1, eta2, phi2):
    """Compute delta-R between two sets of (eta, phi)."""
    deta = eta1 - eta2
    dphi = _delta_phi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


def label_jets(events, jet_eta, jet_phi, R):
    """Label jets by dR-matching to b/c hadrons and taus in the event record.

    Parameters
    ----------
    events : ak.Array
        Full Pythia event batch (with events.prt containing all particles).
    jet_eta : ak.Array
        Jet eta values (events x jets).
    jet_phi : ak.Array
        Jet phi values (events x jets).
    R : float
        Matching cone radius (typically same as jet R).

    Returns
    -------
    labels : ak.Array
        Integer labels per jet following HadronConeExclTruthLabelID convention:
        b=5, c=4, tau=15, light=0. Priority: b > c > tau > light.
    """
    particles = events.prt

    # Compute particle eta/phi from momentum
    px = particles.p.px
    py = particles.p.py
    pz = particles.p.pz
    p_mag = np.sqrt(px**2 + py**2 + pz**2)
    prt_eta = np.arctanh(
        ak.where(p_mag > 0, pz / ak.where(p_mag > 0, p_mag, 1.0), 0.0)
    )
    prt_phi = np.arctan2(py, px)
    prt_id = particles.id

    # Identify b-hadrons, c-hadrons, and taus in the event record
    b_mask = is_b_hadron(prt_id)
    c_mask = is_c_hadron(prt_id)
    tau_mask = is_tau_lepton(prt_id)

    # Start with all-light labels (zeros matching jet shape)
    labels = ak.zeros_like(jet_eta, dtype=np.int32)

    # For each flavor, check if any matching particle is within dR < R
    # Process in priority order: tau first, then c, then b (last write wins)
    for mask, label_val in [(tau_mask, 15), (c_mask, 4), (b_mask, 5)]:
        flavor_eta = ak.drop_none(ak.mask(prt_eta, mask))
        flavor_phi = ak.drop_none(ak.mask(prt_phi, mask))

        # Skip if no particles of this type in any event
        if ak.sum(ak.num(flavor_eta)) == 0:
            continue

        # Cartesian product: (events x jets) x (events x flavor_particles)
        jet_eta_bcast, flav_eta_bcast = ak.unzip(
            ak.cartesian([jet_eta, flavor_eta], nested=True)
        )
        jet_phi_bcast, flav_phi_bcast = ak.unzip(
            ak.cartesian([jet_phi, flavor_phi], nested=True)
        )

        dr = _delta_r(jet_eta_bcast, jet_phi_bcast, flav_eta_bcast, flav_phi_bcast)

        # Check if any flavor particle is within cone
        matched = ak.any(dr < R, axis=-1)

        labels = ak.where(matched, label_val, labels)

    return labels
