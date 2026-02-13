# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TruthJets generates truth-level jet training data for ATLAS flavor-tagging ML models (Salt/Umami frameworks) without full detector simulation. It uses Pythia8 for event generation, FastJet for jet clustering, and outputs HDF5 files with structured dtypes.

## Commands

```bash
# Setup environment
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"

# Generate events (exactly one of --process or --pythia-card is required)
truthjets --process ttbar -n 100000 -o ttbar.h5
truthjets --process qcd -n 1000000 -o qcd.h5
truthjets --process zprime_tt -n 100000 -o zprime.h5
truthjets --pythia-card my_process.cmnd -n 100000 -o custom.h5

# Run all tests
pytest tests/

# Run a single test file
pytest tests/test_label.py

# Plot output
python scripts/plot_jets.py output.h5 -o plots.pdf
```

## Architecture

The pipeline flows: **Pythia8 event generation → jet clustering → flavor labeling → HDF5 writing**, orchestrated by the CLI in batches for memory efficiency.

All source lives in `src/truthjets/`:

- **`cli.py`** — Entry point (`truthjets` command). Parses args, supports YAML config or CLI flags, runs the batch processing loop.
- **`config.py`** — Three dataclasses (`PythiaConfig`, `JetConfig`, `OutputConfig`) plus `PROCESS_PRESETS` mapping process names to Pythia `readString` commands. `PythiaConfig.pythia_card` allows pointing to a `.cmnd` file instead of a preset.
- **`generate.py`** — `init_pythia()` and `generate_events()` iterator yielding batches as Awkward Arrays.
- **`cluster.py`** — `cluster_jets()` using FastJet (antikt/kt/cambridge algorithms) with eta cuts. `compute_jet_kinematics()` derives pt/eta/phi/mass.
- **`label.py`** — PDG ID classification (`is_b_hadron`, `is_c_hadron`, `is_tau_lepton`) and `label_jets()` via dR-matching with priority: b > c > tau > light. Labels: 0=light, 4=c, 5=b, 15=tau.
- **`writer.py`** — `HDF5Writer` with resizable/chunked datasets. Pads constituents to `max_constituents` (default 80), sorts by pT descending, computes relative coordinates (deta, dphi).

## HDF5 Output Format

- `/jets` — structured array: `pt`, `eta`, `phi`, `mass`, `energy` (float32), `HadronConeExclTruthLabelID` (int32), `n_constituents` (int32)
- `/constituents` — shape [n_jets, max_constituents]: `pt`, `deta`, `dphi`, `energy` (float32), `pdgId` (int32), `valid` (bool)

## Key Dependencies

Core physics: `pythia8mc`, `fastjet`, `awkward>=2.0`, `vector`. Data I/O: `h5py`, `numpy`. Config: `pyyaml`.
