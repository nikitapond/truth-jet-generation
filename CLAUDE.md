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

# Large-R jets (auto-loads LargeRLabelModule for W/Z/H/top labeling)
truthjets --pythia-card cards/z_qq.cmnd -R 1.0 -n 100000 -o z_jets.h5
truthjets --pythia-card cards/zh_llbb.cmnd -R 1.0 -n 100000 -o zh_jets.h5

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
- **`cluster.py`** — `extract_particles()` pulls final-state particles from events. `cluster_jets()` uses FastJet (antikt/kt/cambridge algorithms) with eta cuts. `compute_jet_kinematics()` derives pt/eta/phi/mass.
- **`label.py`** — PDG ID classification (`is_b_hadron`, `is_c_hadron`, `is_tau_lepton`) and `label_jets()` via dR-matching with priority: b > c > tau > light. Labels: 0=light, 4=c, 5=b, 15=tau.
- **`modules.py`** — Pipeline module system. `TruthJetModule` base class with `pre_clustering`/`post_clustering` hooks. `ModuleResult` and `DatasetSchema` dataclasses. `load_module()` and `validate_modules()`.
- **`label_module.py`** — Built-in `HadronConeExclLabelModule` that wraps `label_jets()`. Auto-loaded for R=0.4 jets.
- **`writer.py`** — `HDF5Writer` with resizable/chunked datasets. Pads constituents to `max_constituents` (default 80), sorts by pT descending, computes relative coordinates (deta, dphi). Supports extra jet fields and datasets from modules.

## Pipeline Modules

Modules hook into the event processing pipeline at two points: before jet clustering (to inspect/modify particles) and after clustering (to inspect/modify jets, add custom labels, compute derived quantities).

### Built-in modules

- **`HadronConeExclLabelModule`** — dR-matched b/c/tau labeling (`label_module.py`). Auto-loaded for R=0.4 jets.
- **`LargeRLabelModule`** — dR-matched W/Z/H/top labeling (`label_module.py`). Auto-loaded for R > 0.4 jets. Labels: 0=QCD, 6=top, 23=Z, 24=W, 25=Higgs (PDG IDs). Priority: top > H > Z > W.

### Using modules

```bash
# Custom module (auto-discovers single TruthJetModule subclass)
truthjets --process ttbar -n 100000 -o out.h5 --module my_package.my_module

# Explicit class name
truthjets --process ttbar -n 100000 -o out.h5 --module my_package.my_module:ClassName

# Multiple modules (repeatable)
truthjets --process ttbar -n 100000 -o out.h5 --module mod_a --module mod_b
```

### Writing a module

Subclass `TruthJetModule` and override the methods you need:

```python
from truthjets.modules import TruthJetModule, ModuleResult
import numpy as np

class MyModule(TruthJetModule):
    def init(self, jet_config):
        self.R = jet_config.R

    def extra_jet_fields(self):
        return [("my_score", np.float32)]

    def post_clustering(self, events, jets, constituents, jet_kin, labels):
        scores = compute_something(events, jet_kin)
        return ModuleResult(extra_jet_data={"my_score": scores})
```

**Available methods:**
- `init(jet_config)` — called once before processing
- `pre_clustering(events, particles)` — return modified particles or None
- `post_clustering(events, jets, constituents, jet_kin, labels)` — return `ModuleResult` or None
- `extra_jet_fields()` — declare extra `/jets` columns as `[(name, dtype), ...]`
- `extra_datasets()` — declare extra HDF5 datasets as `{name: DatasetSchema(dtype, shape_suffix)}`

## Pythia Cards

Pre-built Pythia configuration cards live in `cards/`:

- **`cards/z_qq.cmnd`** — Boosted Z+jets with Z → qq (hadronic). `pTHatMin = 200`.
- **`cards/zh_llbb.cmnd`** — ZH associated production with H → bb, Z → ll. `pTHatMin = 150`.

## HDF5 Output Format

- `/jets` — structured array: `pt`, `eta`, `phi`, `mass`, `energy` (float32), `HadronConeExclTruthLabelID` (int32), `n_constituents` (int32)
- `/constituents` — shape [n_jets, max_constituents]: `pt`, `deta`, `dphi`, `energy` (float32), `pdgId` (int32), `valid` (bool)

## Key Dependencies

Core physics: `pythia8mc`, `fastjet`, `awkward>=2.0`, `vector`. Data I/O: `h5py`, `numpy`. Config: `pyyaml`.
