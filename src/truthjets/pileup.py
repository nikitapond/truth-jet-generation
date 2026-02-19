from __future__ import annotations

from pathlib import Path

import awkward as ak
import h5py
import numpy as np

from truthjets.h5utils import H5_COMPRESSION


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
    pu_particles: ak.Array | None = None,
):
    """Merge hard-scatter and pileup final-state particles for jet clustering.

    Parameters
    ----------
    hs_events : ak.Array
        Hard-scatter events from Pythia (with .prt field).
    pu_events : ak.Array
        Pileup (min-bias) events from Pythia, total count = sum(n_pu_per_event).
        Can be None if pu_particles is provided.
    n_pu_per_event : np.ndarray
        Number of PU events overlaid on each HS event.
    sigma_z : float
        Gaussian sigma for PU vertex z spread in mm (default 53.0).
    rng : np.random.Generator, optional
        Random number generator for vertex z sampling.
    pu_particles : ak.Array, optional
        Pre-processed PU particles with fields {px, py, pz, E, pdgId}.
        When provided, skip Pythia event extraction and use these directly.

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
    if total_pu == 0 or (pu_events is None and pu_particles is None):
        return hs_records

    if rng is None:
        rng = np.random.default_rng()

    # Sample one vertex z per PU interaction
    pu_vz_per_interaction = rng.normal(0.0, sigma_z, size=total_pu).astype(np.float32)

    if pu_particles is not None:
        # Use pre-processed particles from pool
        pu_final = pu_particles
        n_particles_per_pu_event = ak.num(pu_final, axis=1)
        vz_flat = np.repeat(pu_vz_per_interaction, ak.to_numpy(n_particles_per_pu_event))
        vz_per_particle = ak.unflatten(vz_flat, n_particles_per_pu_event)

        pu_records = ak.zip(
            {
                "px": pu_final.px,
                "py": pu_final.py,
                "pz": pu_final.pz,
                "E": pu_final.E,
                "pdgId": pu_final.pdgId,
                "is_pu": ak.ones_like(pu_final.pdgId, dtype=np.bool_),
                "vz": vz_per_particle,
            }
        )
    else:
        # Extract PU final-state particles from Pythia events
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


def generate_pileup_pool(pythia_pu, n_events: int, batch_size: int = 10_000) -> ak.Array:
    """Generate a pool of min-bias PU events, extracting final-state particles.

    Parameters
    ----------
    pythia_pu : pythia8mc.Pythia
        Initialized Pythia instance for min-bias generation.
    n_events : int
        Total number of PU events to generate.
    batch_size : int
        Number of events per Pythia batch call.

    Returns
    -------
    pool : ak.Array
        Ragged array of shape (n_events, var) with fields {px, py, pz, E, pdgId}.
    """
    all_px, all_py, all_pz, all_e, all_pdgid = [], [], [], [], []

    generated = 0
    while generated < n_events:
        this_batch = min(batch_size, n_events - generated)
        events = pythia_pu.nextBatch(this_batch)

        prt = events.prt
        final = prt[prt.status > 0]

        all_px.append(final.p.px)
        all_py.append(final.p.py)
        all_pz.append(final.p.pz)
        all_e.append(final.p.e)
        all_pdgid.append(final.id)

        generated += this_batch

    pool = ak.zip(
        {
            "px": ak.concatenate(all_px, axis=0),
            "py": ak.concatenate(all_py, axis=0),
            "pz": ak.concatenate(all_pz, axis=0),
            "E": ak.concatenate(all_e, axis=0),
            "pdgId": ak.concatenate(all_pdgid, axis=0),
        }
    )
    return pool


def save_pileup_pool(pool: ak.Array, path: str | Path) -> None:
    """Save a pileup pool to HDF5 using flat arrays + offsets.

    Format:
        /n_particles — (n_events,) int32
        /px, /py, /pz, /E — (total_particles,) float32
        /pdgId — (total_particles,) int32
    """
    n_particles = ak.to_numpy(ak.num(pool, axis=1)).astype(np.int32)
    px_flat = ak.to_numpy(ak.flatten(pool.px)).astype(np.float32)
    py_flat = ak.to_numpy(ak.flatten(pool.py)).astype(np.float32)
    pz_flat = ak.to_numpy(ak.flatten(pool.pz)).astype(np.float32)
    e_flat = ak.to_numpy(ak.flatten(pool.E)).astype(np.float32)
    pdgid_flat = ak.to_numpy(ak.flatten(pool.pdgId)).astype(np.int32)

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        f.create_dataset("n_particles", data=n_particles, **H5_COMPRESSION)
        f.create_dataset("px", data=px_flat, **H5_COMPRESSION)
        f.create_dataset("py", data=py_flat, **H5_COMPRESSION)
        f.create_dataset("pz", data=pz_flat, **H5_COMPRESSION)
        f.create_dataset("E", data=e_flat, **H5_COMPRESSION)
        f.create_dataset("pdgId", data=pdgid_flat, **H5_COMPRESSION)


def load_pileup_pool(path: str | Path) -> ak.Array:
    """Load a pileup pool from HDF5, reconstructing the ragged awkward array.

    If *path* is a directory, all ``*.h5`` files inside it are loaded and
    concatenated (sorted by filename) into a single pool.
    """
    path = Path(path)

    if path.is_dir():
        pool_files = sorted(path.glob("*.h5"))
        if not pool_files:
            raise FileNotFoundError(f"No .h5 files found in {path}")
        pools = [_load_single_pool(pf) for pf in pool_files]
        return ak.concatenate(pools, axis=0)

    return _load_single_pool(path)


def _load_single_pool(path: Path) -> ak.Array:
    """Load a single pileup pool HDF5 file."""
    with h5py.File(path, "r") as f:
        n_particles = f["n_particles"][:]
        px_flat = f["px"][:]
        py_flat = f["py"][:]
        pz_flat = f["pz"][:]
        e_flat = f["E"][:]
        pdgid_flat = f["pdgId"][:]

    pool = ak.zip(
        {
            "px": ak.unflatten(px_flat, n_particles),
            "py": ak.unflatten(py_flat, n_particles),
            "pz": ak.unflatten(pz_flat, n_particles),
            "E": ak.unflatten(e_flat, n_particles),
            "pdgId": ak.unflatten(pdgid_flat, n_particles),
        }
    )
    return pool


def sample_from_pool(
    pool: ak.Array,
    n_events: int,
    rng: np.random.Generator,
) -> ak.Array:
    """Sample events from pool with replacement and apply random phi rotation.

    Each sampled event gets a random phi rotation to maintain physical randomness:
        px' = px*cos(theta) - py*sin(theta)
        py' = px*sin(theta) + py*cos(theta)
        pz, E unchanged.

    Parameters
    ----------
    pool : ak.Array
        Pileup pool with fields {px, py, pz, E, pdgId}.
    n_events : int
        Number of events to sample.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    sampled : ak.Array
        Ragged array with fields {px, py, pz, E, pdgId}.
    """
    pool_size = len(pool)
    indices = rng.integers(0, pool_size, size=n_events)
    sampled = pool[indices]

    # Random phi rotation per event
    theta = rng.uniform(0, 2 * np.pi, size=n_events).astype(np.float32)

    # Broadcast theta to particle level
    n_per_event = ak.num(sampled, axis=1)
    theta_flat = np.repeat(theta, ak.to_numpy(n_per_event))
    cos_t = np.cos(theta_flat)
    sin_t = np.sin(theta_flat)

    px_flat = ak.to_numpy(ak.flatten(sampled.px))
    py_flat = ak.to_numpy(ak.flatten(sampled.py))

    px_rot = px_flat * cos_t - py_flat * sin_t
    py_rot = px_flat * sin_t + py_flat * cos_t

    rotated = ak.zip(
        {
            "px": ak.unflatten(px_rot, n_per_event),
            "py": ak.unflatten(py_rot, n_per_event),
            "pz": sampled.pz,
            "E": sampled.E,
            "pdgId": sampled.pdgId,
        }
    )
    return rotated
