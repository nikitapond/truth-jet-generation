from __future__ import annotations

import awkward as ak
import numpy as np

from truthjets.cluster import safe_eta
from truthjets.config import JetConfig
from truthjets.label import final_b_hadron_mask
from truthjets.modules import ModuleResult, TruthJetModule


def _delta_phi(phi1, phi2):
    """Compute delta-phi, wrapped to [-pi, pi]."""
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def _delta_r(eta1, phi1, eta2, phi2):
    """Compute delta-R between two sets of (eta, phi)."""
    deta = eta1 - eta2
    dphi = _delta_phi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)


class BBOpeningAngleModule(TruthJetModule):
    """Computes the dR opening angle between exactly 2 b-hadrons matched to a jet.

    For large-R jets only (R > 0.4). Jets with != 2 matched b-hadrons
    get NaN.

    Extra jet field: ``bb_dR`` (float32).
    """

    def init(self, jet_config: JetConfig) -> None:
        if jet_config.R <= 0.4:
            raise ValueError(
                "BBOpeningAngleModule requires R > 0.4 (large-R jets)"
            )
        self.R = jet_config.R

    def extra_jet_fields(self):
        return [("bb_dR", np.float32)]

    def post_clustering(self, events, jets, constituents, jet_kin, labels):
        particles = events.prt

        # Compute particle eta/phi from momentum
        px = particles.p.px
        py = particles.p.py
        pz = particles.p.pz
        p_mag = np.sqrt(px**2 + py**2 + pz**2)
        prt_eta = safe_eta(pz, p_mag)
        prt_phi = np.arctan2(py, px)
        prt_id = particles.id

        # Filter to weakly-decaying b-hadrons only (no excited states)
        b_mask = final_b_hadron_mask(particles)
        b_eta = ak.drop_none(ak.mask(prt_eta, b_mask))
        b_phi = ak.drop_none(ak.mask(prt_phi, b_mask))

        # Match b-hadrons to jets via dR < R using cartesian product
        # jet_kin fields are (events x jets), b_eta/b_phi are (events x b_hadrons)
        jet_b_eta = ak.cartesian([jet_kin.eta, b_eta], nested=True)
        jet_b_phi = ak.cartesian([jet_kin.phi, b_phi], nested=True)
        j_eta, bh_eta = ak.unzip(jet_b_eta)
        j_phi, bh_phi = ak.unzip(jet_b_phi)

        dr_to_jet = _delta_r(j_eta, j_phi, bh_eta, bh_phi)

        # Boolean mask: which b-hadrons are within the jet cone
        # Shape: (events x jets x b_hadrons)
        in_cone = dr_to_jet < self.R

        # Get matched b-hadron eta/phi per jet (mask out non-matched)
        matched_eta = ak.where(in_cone, bh_eta, np.nan)
        matched_phi = ak.where(in_cone, bh_phi, np.nan)

        # Drop NaN entries to get only matched b-hadrons per jet
        # We need to filter: keep only entries where in_cone is True
        matched_eta = ak.drop_none(ak.mask(bh_eta, in_cone))
        matched_phi = ak.drop_none(ak.mask(bh_phi, in_cone))

        # Count matched b-hadrons per jet: (events x jets)
        n_matched = ak.num(matched_eta, axis=-1)

        # Pad to at least 2 entries so indexing [0] and [1] is always safe
        padded_eta = ak.pad_none(matched_eta, 2, axis=-1)
        padded_phi = ak.pad_none(matched_phi, 2, axis=-1)

        # For jets with exactly 2 matched b-hadrons, compute dR between the pair
        has_two = n_matched == 2
        eta1 = ak.where(has_two, ak.fill_none(padded_eta[..., 0], 0.0), 0.0)
        eta2 = ak.where(has_two, ak.fill_none(padded_eta[..., 1], 0.0), 0.0)
        phi1 = ak.where(has_two, ak.fill_none(padded_phi[..., 0], 0.0), 0.0)
        phi2 = ak.where(has_two, ak.fill_none(padded_phi[..., 1], 0.0), 0.0)

        bb_dr = _delta_r(eta1, phi1, eta2, phi2)

        # Set NaN for jets without exactly 2 matched b-hadrons
        bb_dr = ak.where(has_two, bb_dr, np.nan)

        return ModuleResult(extra_jet_data={"bb_dR": bb_dr})
