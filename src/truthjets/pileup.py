from __future__ import annotations

import awkward as ak
import numpy as np


def sample_n_pileup(mu: float, n_events: int, rng: np.random.Generator) -> np.ndarray:
    """Sample the number of pileup interactions per event from Poisson(mu).

    Returns an array of shape (n_events,) with integer counts.
    """
    return rng.poisson(mu, size=n_events)


def overlay_pileup(
    hs_events,
    pu_events,
    n_pu_per_event: np.ndarray,
    sigma_z: float = 53.0,
    rng: np.random.Generator | None = None,
):
    """Merge hard-scatter and pileup final-state particles for jet clustering.

    Parameters
    ----------
    hs_events : ak.Array
        Hard-scatter events from Pythia (with .prt field).
    pu_events : ak.Array
        Pileup (min-bias) events from Pythia, total count = sum(n_pu_per_event).
    n_pu_per_event : np.ndarray
        Number of PU events overlaid on each HS event.
    sigma_z : float
        Gaussian sigma for PU vertex z spread in mm (default 53.0).
    rng : np.random.Generator, optional
        Random number generator for vertex z sampling.

    Returns
    -------
    merged : ak.Array
        Merged particle arrays (events x particles) with fields
        {px, py, pz, E, pdgId, is_pu, vz}, ready for jet clustering.
    """
    # Extract HS final-state particles
    hs_prt = hs_events.prt
    hs_final = hs_prt[hs_prt.status > 0]

    hs_records = ak.zip(
        {
            "px": hs_final.p.px,
            "py": hs_final.p.py,
            "pz": hs_final.p.pz,
            "E": hs_final.p.e,
            "pdgId": hs_final.id,
            "is_pu": ak.zeros_like(hs_final.id, dtype=np.bool_),
            "vz": ak.zeros_like(hs_final.p.px, dtype=np.float32),
        }
    )

    # If no PU events at all, return HS with is_pu=False
    total_pu = int(np.sum(n_pu_per_event))
    if total_pu == 0 or pu_events is None:
        return hs_records

    if rng is None:
        rng = np.random.default_rng()

    # Sample one vertex z per PU interaction
    pu_vz_per_interaction = rng.normal(0.0, sigma_z, size=total_pu).astype(np.float32)

    # Extract PU final-state particles
    pu_prt = pu_events.prt
    pu_final = pu_prt[pu_prt.status > 0]

    # Broadcast interaction-level vz to each particle in that interaction
    n_particles_per_pu_event = ak.num(pu_final, axis=1)
    vz_flat = np.repeat(pu_vz_per_interaction, ak.to_numpy(n_particles_per_pu_event))
    vz_per_particle = ak.unflatten(vz_flat, n_particles_per_pu_event)

    pu_records = ak.zip(
        {
            "px": pu_final.p.px,
            "py": pu_final.p.py,
            "pz": pu_final.p.pz,
            "E": pu_final.p.e,
            "pdgId": pu_final.id,
            "is_pu": ak.ones_like(pu_final.id, dtype=np.bool_),
            "vz": vz_per_particle,
        }
    )

    # Group PU events by HS event: unflatten by n_pu_per_event
    # This gives (n_hs_events x var_pu_events x var_particles)
    pu_grouped = ak.unflatten(pu_records, n_pu_per_event, axis=0)

    # Flatten the PU event level to get (n_hs_events x all_pu_particles)
    pu_per_hs = ak.flatten(pu_grouped, axis=2)

    # Concatenate HS and PU particles within each event
    merged = ak.concatenate([hs_records, pu_per_hs], axis=1)

    return merged
