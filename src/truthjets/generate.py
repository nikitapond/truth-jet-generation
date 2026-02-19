from __future__ import annotations

from collections.abc import Iterator

import pythia8mc

from truthjets.config import PythiaConfig


def init_pythia(config: PythiaConfig) -> pythia8mc.Pythia:
    """Create and initialize a Pythia instance from config."""
    pythia = pythia8mc.Pythia("", False)

    # Apply process settings from card file
    if config.pythia_card is None:
        raise ValueError("pythia_card must be set (use resolve_card() to resolve short names)")
    pythia.readFile(config.pythia_card)

    # Beam settings
    pythia.readString(f"Beams:eCM = {config.ecm}")

    # pTHat cuts
    if config.pt_hat_min is not None:
        pythia.readString(f"PhaseSpace:pTHatMin = {config.pt_hat_min}")
    if config.pt_hat_max is not None:
        pythia.readString(f"PhaseSpace:pTHatMax = {config.pt_hat_max}")

    # Vertex smearing (realistic spatial origins for particles)
    pythia.readString("Beams:allowVertexSpread = on")
    pythia.readString("Beams:sigmaVertexX = 0.015")
    pythia.readString("Beams:sigmaVertexY = 0.015")
    pythia.readString("Beams:sigmaVertexZ = 53.0")

    # Random seed
    pythia.readString("Random:setSeed = on")
    pythia.readString(f"Random:seed = {config.seed}")

    # Suppress banner output
    pythia.readString("Print:quiet = on")

    # Extra user settings
    for setting in config.extra_settings:
        pythia.readString(setting)

    pythia.init()
    return pythia


def init_pileup_pythia(config: PythiaConfig) -> pythia8mc.Pythia:
    """Create a Pythia instance for minimum-bias pileup generation."""
    pythia = pythia8mc.Pythia("", False)

    pythia.readString("SoftQCD:nonDiffractive = on")
    pythia.readString(f"Beams:eCM = {config.ecm}")

    # Vertex smearing (realistic spatial origins for particles)
    pythia.readString("Beams:allowVertexSpread = on")
    pythia.readString("Beams:sigmaVertexX = 0.015")
    pythia.readString("Beams:sigmaVertexY = 0.015")
    pythia.readString("Beams:sigmaVertexZ = 53.0")

    # Use seed offset to avoid correlation with hard-scatter
    pythia.readString("Random:setSeed = on")
    pythia.readString(f"Random:seed = {config.seed + 1}")

    pythia.readString("Print:quiet = on")

    pythia.init()
    return pythia


def generate_pileup_batch(pythia_pu: pythia8mc.Pythia, n_events: int):
    """Generate a batch of minimum-bias events for pileup overlay.

    Returns None if n_events is 0.
    """
    if n_events <= 0:
        return None
    return pythia_pu.nextBatch(n_events)


def generate_events(pythia: pythia8mc.Pythia, n_events: int, batch_size: int) -> Iterator:
    """Yield batches of events as Awkward Arrays."""
    generated = 0
    while generated < n_events:
        this_batch = min(batch_size, n_events - generated)
        events = pythia.nextBatch(this_batch)
        generated += this_batch
        yield events
