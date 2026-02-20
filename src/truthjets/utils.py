"""Shared utilities for truthjets."""

from __future__ import annotations

import numpy as np

# Neutrino PDG IDs (invisible particles excluded from jet clustering)
NEUTRINO_PDGIDS = {12, 14, 16}


def is_neutrino(pdg_id):
    """Vectorized check for neutrinos (|pdgId| in {12, 14, 16})."""
    abs_id = np.abs(pdg_id)
    return (abs_id == 12) | (abs_id == 14) | (abs_id == 16)


def filter_neutrinos(particles, id_field="id"):
    """Remove neutrinos from a particle collection.

    Parameters
    ----------
    particles : ak.Array
        Particle records with a PDG ID field.
    id_field : str
        Name of the PDG ID field (default: "id").

    Returns
    -------
    ak.Array
        Particles with neutrinos removed.
    """
    return particles[~is_neutrino(particles[id_field])]


def delta_phi(phi1, phi2):
    """Compute delta-phi, wrapped to [-pi, pi]."""
    dphi = phi1 - phi2
    return (dphi + np.pi) % (2 * np.pi) - np.pi


def delta_r(eta1, phi1, eta2, phi2):
    """Compute delta-R between two sets of (eta, phi)."""
    deta = eta1 - eta2
    dphi = delta_phi(phi1, phi2)
    return np.sqrt(deta**2 + dphi**2)
