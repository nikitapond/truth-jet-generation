from __future__ import annotations

import awkward as ak
import numpy as np

from truthjets.config import JetConfig
from truthjets.label import label_jets
from truthjets.modules import ModuleResult, TruthJetModule


class HadronConeExclLabelModule(TruthJetModule):
    """Labels jets using dR-matched HadronConeExclTruthLabelID.

    Matches b-hadrons, c-hadrons, and tau leptons to jets within
    a cone of radius R. Priority: b > c > tau > light.

    This module is automatically loaded for R=0.4 jets.
    """

    def init(self, jet_config: JetConfig) -> None:
        self.R = jet_config.R

    def post_clustering(self, events, jets, constituents, jet_kin, labels):
        new_labels = label_jets(events, jet_kin.eta, jet_kin.phi, self.R)
        return ModuleResult(labels=new_labels)


# PDG IDs for large-R resonances
_RESONANCE_PDGIDS = {
    "W": 24,
    "Z": 23,
    "H": 25,
    "top": 6,
}

# Labels use PDG IDs; priority order (last write wins): W -> Z -> H -> top
_LARGE_R_PRIORITY = [
    (24, 24),   # W
    (23, 23),   # Z
    (25, 25),   # H
    (6, 6),     # top
]


def _delta_phi(phi1, phi2):
    """Compute delta-phi, wrapped to [-pi, pi]."""
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def _delta_r(eta1, phi1, eta2, phi2):
    """Compute delta-R between two sets of (eta, phi)."""
    deta = eta1 - eta2
    dphi = _delta_phi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


def label_large_r_jets(events, jet_eta, jet_phi, R):
    """Label large-R jets by dR-matching to truth resonances.

    Matches W, Z, H, and top quarks in the event record to jets
    within a cone of radius R.

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
        Integer labels per jet: 0=QCD, 6=top, 23=Z, 24=W, 25=Higgs.
        Priority: top > H > Z > W.
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

    # Start with all-QCD labels (zeros matching jet shape)
    labels = ak.zeros_like(jet_eta, dtype=np.int32)

    # Match each resonance type; last write wins so process in priority order
    for abs_pdgid, label_val in _LARGE_R_PRIORITY:
        mask = np.abs(prt_id) == abs_pdgid
        res_eta = ak.drop_none(ak.mask(prt_eta, mask))
        res_phi = ak.drop_none(ak.mask(prt_phi, mask))

        # Skip if no particles of this type in any event
        if ak.sum(ak.num(res_eta)) == 0:
            continue

        # Cartesian product: (events x jets) x (events x resonance_particles)
        jet_eta_bcast, res_eta_bcast = ak.unzip(
            ak.cartesian([jet_eta, res_eta], nested=True)
        )
        jet_phi_bcast, res_phi_bcast = ak.unzip(
            ak.cartesian([jet_phi, res_phi], nested=True)
        )

        dr = _delta_r(jet_eta_bcast, jet_phi_bcast, res_eta_bcast, res_phi_bcast)

        # Check if any resonance particle is within cone
        matched = ak.any(dr < R, axis=-1)

        labels = ak.where(matched, label_val, labels)

    return labels


class LargeRLabelModule(TruthJetModule):
    """Labels large-R jets by dR-matching to truth W/Z/H/top resonances.

    Labels: 0=QCD, 6=top, 23=Z, 24=W, 25=Higgs (PDG IDs).
    Priority: top > H > Z > W.

    This module is automatically loaded for jets with R > 0.4.
    """

    def init(self, jet_config: JetConfig) -> None:
        self.R = jet_config.R

    def post_clustering(self, events, jets, constituents, jet_kin, labels):
        new_labels = label_large_r_jets(events, jet_kin.eta, jet_kin.phi, self.R)
        return ModuleResult(labels=new_labels)
