from __future__ import annotations

import awkward as ak
import numpy as np

from truthjets.cluster import safe_eta
from truthjets.config import JetConfig
from truthjets.label import label_jets
from truthjets.modules import ModuleResult, TruthJetModule
from truthjets.utils import NEUTRINO_PDGIDS, delta_r


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


# Labels use PDG IDs; priority order (last write wins): W -> Z -> H -> top
_LARGE_R_PRIORITY = [
    (24, 24),  # W
    (23, 23),  # Z
    (25, 25),  # H
    (6, 6),  # top
]


def _has_visible_decay(particles, res_mask):
    """Check which resonances have at least one visible (non-neutrino) daughter.

    Parameters
    ----------
    particles : ak.Array
        Full particle record with fields ``id``, ``daughter1``, ``daughter2``.
    res_mask : ak.Array
        Boolean mask selecting the resonance particles.

    Returns
    -------
    visible_mask : ak.Array
        Boolean mask (same shape as res_mask's True entries) indicating
        which resonances have visible decays.
    """
    # If daughter fields aren't present, assume all resonances are visible
    if "daughter1" not in particles.fields:
        n_res = ak.num(ak.drop_none(ak.mask(particles.id, res_mask)))
        return ak.unflatten(np.ones(ak.sum(n_res), dtype=bool), n_res)

    prt_id = particles.id
    d1 = particles.daughter1
    d2 = particles.daughter2

    # Get daughter ranges and the resonance's own PDG ID
    res_d1 = ak.drop_none(ak.mask(d1, res_mask))
    res_d2 = ak.drop_none(ak.mask(d2, res_mask))
    res_id = ak.drop_none(ak.mask(np.abs(prt_id), res_mask))

    # For each resonance, check daughters
    counts = ak.num(prt_id)
    counts_np = ak.to_numpy(counts)
    offsets = np.concatenate([[0], np.cumsum(counts_np)])
    flat_id = ak.to_numpy(ak.flatten(np.abs(prt_id)))

    flat_d1 = ak.to_numpy(ak.flatten(res_d1))
    flat_d2 = ak.to_numpy(ak.flatten(res_d2))
    flat_res_id = ak.to_numpy(ak.flatten(res_id))
    res_counts = ak.num(res_d1)
    event_idx = np.repeat(np.arange(len(counts_np)), ak.to_numpy(res_counts))

    # Invisible PDG IDs: neutrinos + copies of the resonance itself
    has_visible = np.zeros(len(flat_d1), dtype=bool)
    for i in range(len(flat_d1)):
        if flat_d1[i] == 0:
            continue
        own_id = flat_res_id[i]
        global_start = flat_d1[i] + offsets[event_idx[i]]
        global_end = flat_d2[i] + offsets[event_idx[i]]
        for j in range(global_start, global_end + 1):
            if j < len(flat_id):
                did = flat_id[j]
                # Skip self-copies and neutrinos
                if did != own_id and did not in NEUTRINO_PDGIDS:
                    has_visible[i] = True
                    break

    return ak.unflatten(has_visible, res_counts)


def label_large_r_jets(events, jet_eta, jet_phi, R):
    """Label large-R jets by dR-matching to truth resonances.

    Matches W, Z, H, and top quarks in the event record to jets
    within a cone of radius R. Only resonances with at least one
    visible (non-neutrino) daughter are considered.

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
    prt_eta = safe_eta(pz, p_mag)
    prt_phi = np.arctan2(py, px)
    prt_id = particles.id

    # Start with all-QCD labels (zeros matching jet shape)
    labels = ak.zeros_like(jet_eta, dtype=np.int32)

    # Match each resonance type; last write wins so process in priority order
    for abs_pdgid, label_val in _LARGE_R_PRIORITY:
        mask = np.abs(prt_id) == abs_pdgid

        # Filter to only resonances with visible decays
        visible = _has_visible_decay(particles, mask)
        res_eta_all = ak.drop_none(ak.mask(prt_eta, mask))
        res_phi_all = ak.drop_none(ak.mask(prt_phi, mask))
        res_eta = ak.drop_none(ak.mask(res_eta_all, visible))
        res_phi = ak.drop_none(ak.mask(res_phi_all, visible))

        # Skip if no visible resonances of this type in any event
        if ak.sum(ak.num(res_eta)) == 0:
            continue

        # Cartesian product: (events x jets) x (events x resonance_particles)
        jet_eta_bcast, res_eta_bcast = ak.unzip(ak.cartesian([jet_eta, res_eta], nested=True))
        jet_phi_bcast, res_phi_bcast = ak.unzip(ak.cartesian([jet_phi, res_phi], nested=True))

        dr = delta_r(jet_eta_bcast, jet_phi_bcast, res_eta_bcast, res_phi_bcast)

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
