from __future__ import annotations

import awkward as ak
import fastjet
import numpy as np

from truthjets.config import JetConfig

_ALGORITHMS = {
    "antikt": fastjet.antikt_algorithm,
    "kt": fastjet.kt_algorithm,
    "cambridge": fastjet.cambridge_algorithm,
}


def cluster_jets(events, jet_config: JetConfig, particles=None):
    """Cluster jets from a Pythia event batch.

    Parameters
    ----------
    events : ak.Array
        Batch from pythia.nextBatch() with events.prt fields.
    jet_config : JetConfig
        Jet clustering configuration.
    particles : ak.Array, optional
        Pre-merged particle arrays (events x particles) with fields
        {px, py, pz, E, pdgId, is_pu}. When provided, used directly
        instead of extracting from events. Used for pileup overlay.

    Returns
    -------
    jets : ak.Array
        Clustered jets (nested: events x jets) with {px, py, pz, E}.
    constituents : ak.Array
        Constituent particles per jet (events x jets x constituents)
        with {px, py, pz, E, pdgId, is_pu}.
    """
    if particles is not None:
        # Use pre-merged particles (e.g. from pileup overlay)
        fj_particles = particles
    else:
        # Extract final-state from events and add is_pu=False
        prt = events.prt
        final = prt[prt.status > 0]
        fj_particles = ak.zip(
            {
                "px": final.p.px,
                "py": final.p.py,
                "pz": final.p.pz,
                "E": final.p.e,
                "pdgId": final.id,
                "is_pu": ak.zeros_like(final.id, dtype=np.bool_),
            }
        )

    # Jet definition
    alg = _ALGORITHMS.get(jet_config.algorithm)
    if alg is None:
        raise ValueError(
            f"Unknown algorithm '{jet_config.algorithm}'. "
            f"Available: {list(_ALGORITHMS)}"
        )
    jetdef = fastjet.JetDefinition(alg, jet_config.R)

    # Cluster
    cluster = fastjet.ClusterSequence(fj_particles, jetdef)

    # Get jets and their constituents
    jets = cluster.inclusive_jets(min_pt=jet_config.pt_min)
    constits = cluster.constituents(min_pt=jet_config.pt_min)

    # Apply constituent pT cut
    if jet_config.constituent_pt_min > 0:
        c_pt = np.sqrt(constits.px**2 + constits.py**2)
        constits = constits[c_pt >= jet_config.constituent_pt_min]

    # Compute jet kinematics for eta cut
    jet_kin = compute_jet_kinematics(jets)

    # Apply eta cut
    eta_mask = np.abs(jet_kin.eta) < jet_config.eta_max
    jets = jets[eta_mask]
    constits = constits[eta_mask]
    jet_kin = jet_kin[eta_mask]

    return jets, constits, jet_kin


def compute_jet_kinematics(jets):
    """Derive pt, eta, phi, mass from jet px/py/pz/E.

    Returns an ak.Array with fields {pt, eta, phi, mass, energy}.
    """
    px = jets.px
    py = jets.py
    pz = jets.pz
    E = jets.E

    pt = np.sqrt(px**2 + py**2)
    p = np.sqrt(px**2 + py**2 + pz**2)
    eta = np.arctanh(ak.where(p > 0, pz / ak.where(p > 0, p, 1.0), 0.0))
    phi = np.arctan2(py, px)
    mass_sq = E**2 - px**2 - py**2 - pz**2
    mass = np.sqrt(ak.where(mass_sq > 0, mass_sq, 0.0))

    return ak.zip(
        {"pt": pt, "eta": eta, "phi": phi, "mass": mass, "energy": E}
    )
