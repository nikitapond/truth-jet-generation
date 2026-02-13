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
