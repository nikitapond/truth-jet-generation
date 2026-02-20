# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

TruthJets generates truth-level jet training data for ATLAS flavor-tagging ML models (Salt/Umami frameworks) without full detector simulation. It uses Pythia8 for event generation, FastJet for jet clustering, and outputs HDF5 files with structured dtypes.

## Commands

```bash
# Setup environment (ALWAYS use [dev] to include pytest and other dev tools)
uv venv && source .venv/bin/activate && uv pip install -e ".[dev]"

# Generate events (--pythia-card accepts a built-in name or path to .cmnd file)
truthjets --pythia-card ttbar -n 100000 -o ttbar.h5
truthjets --pythia-card qcd -n 1000000 -o qcd.h5
truthjets --pythia-card zprime_tt -n 100000 -o zprime.h5
truthjets --pythia-card my_process.cmnd -n 100000 -o custom.h5

# Large-R jets (auto-loads LargeRLabelModule for W/Z/H/top labeling)
truthjets --pythia-card z_qq -R 1.0 -n 100000 -o z_jets.h5
truthjets --pythia-card zh_llbb -R 1.0 -n 100000 -o zh_jets.h5

# Generate a pileup pool for later reuse
generate-pu-pool -n 100000 -o pu_pool.h5

# Use it with hard-scatter generation (accepts a file or directory of pool chunks)
truthjets --pythia-card ttbar -n 100000 --pu 50 --pu-file pu_pool.h5 -o ttbar_pu.h5
truthjets --pythia-card ttbar -n 100000 --pu 50 --pu-file /path/to/pool_chunks/ -o ttbar_pu.h5

# Create a Virtual Dataset from multiple HDF5 files
create-vds part_000.h5 part_001.h5 part_002.h5 -o combined.h5
create-vds /path/to/parts/ -o combined.h5

# Run all tests (use .venv/bin/pytest if venv is not activated)
.venv/bin/pytest tests/

# Run a single test file
.venv/bin/pytest tests/test_label.py

# Run tests with coverage
.venv/bin/pytest tests/ --cov=truthjets --cov-report=term-missing

# NOTE: If pytest is missing, you likely installed without [dev].
# Fix with: uv pip install -e ".[dev]"

# Benchmark pipeline stages
truthjets --pythia-card ttbar -n 10000 -o ttbar.h5 --benchmark

# Plot output (installed entry points)
plot-jets output.h5 -o jets.pdf
plot-events output.h5 -o events.pdf
plot-constituents output.h5 -o constituents.pdf
plot-all output.h5 -o plots/
```

## CI

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on push to `main` and all PRs:

- **lint** — `ruff check` and `ruff format --check` on `src/` and `tests/`
- **test** — `pytest` across Python 3.10/3.11/3.12 (skips `test_pu_closure.py`)

Run linting locally: `ruff check src/ tests/` and `ruff format --check src/ tests/`. Auto-fix with `ruff check --fix` and `ruff format`.

## Architecture

The pipeline flows: **Pythia8 event generation → jet clustering → flavor labeling → HDF5 writing**, orchestrated by the CLI in batches for memory efficiency.

All source lives in `src/truthjets/`:

- **`cli.py`** — Entry point (`truthjets` command). Parses args, supports YAML config or CLI flags, runs the batch processing loop.
- **`config.py`** — Three dataclasses (`PythiaConfig`, `JetConfig`, `OutputConfig`) plus `CARDS_DIR` and `resolve_card()` for resolving built-in card names (e.g. `ttbar`) or file paths to absolute `.cmnd` paths.
- **`generate.py`** — `init_pythia()` and `generate_events()` iterator yielding batches as Awkward Arrays.
- **`cluster.py`** — `extract_particles()` pulls final-state particles from events. `cluster_jets()` uses FastJet (antikt/kt/cambridge algorithms) with eta cuts. `compute_jet_kinematics()` derives pt/eta/phi/mass.
- **`label.py`** — PDG ID classification (`is_b_hadron`, `is_c_hadron`, `is_tau_lepton`) and `label_jets()` via dR-matching with priority: b > c > tau > light. Labels: 0=light, 4=c, 5=b, 15=tau.
- **`modules/`** — Pipeline module package. `TruthJetModule` base class with `pre_clustering`/`post_clustering` hooks. `ModuleResult` and `DatasetSchema` dataclasses. `load_module()` and `validate_modules()`.
  - **`modules/label.py`** — Built-in `HadronConeExclLabelModule` (b/c/tau labeling) and `LargeRLabelModule` (W/Z/H/top labeling).
  - **`modules/bb_opening_angle.py`** — `BBOpeningAngleModule` for computing dR between b-hadron pairs in large-R jets.
  - **`modules/pileup_rejection.py`** — `SoftKillerModule` (pre-clustering SoftKiller) and `VertexZFilterModule` (post-clustering vertex z cut). Auto-loaded when pileup is active and `--softkiller`/`--max-dz` are set.
- **`benchmark.py`** — `Benchmark` class for optional per-stage pipeline timing (`--benchmark` flag).
- **`h5utils.py`** — Shared HDF5 utilities. `H5_COMPRESSION` dict (gzip-7 + shuffle) used by all dataset creation. `create_vds()` builds HDF5 Virtual Datasets from part files. Also provides the `create-vds` CLI entry point.
- **`writer.py`** — `HDF5Writer` with resizable/chunked datasets. Pads constituents to `max_constituents` (default 80), sorts by pT descending, computes relative coordinates (deta, dphi). Supports extra jet fields and datasets from modules.

## Pipeline Modules

Modules hook into the event processing pipeline at two points: before jet clustering (to inspect/modify particles) and after clustering (to inspect/modify jets, add custom labels, compute derived quantities).

### Built-in modules

- **`HadronConeExclLabelModule`** — dR-matched b/c/tau labeling (`modules/label.py`). Auto-loaded for R=0.4 jets.
- **`LargeRLabelModule`** — dR-matched W/Z/H/top labeling (`modules/label.py`). Auto-loaded for R > 0.4 jets. Labels: 0=QCD, 6=top, 23=Z, 24=W, 25=Higgs (PDG IDs). Priority: top > H > Z > W.
- **`BBOpeningAngleModule`** — Computes `bb_dR` opening angle between exactly 2 b-hadrons matched to a large-R jet (`modules/bb_opening_angle.py`). Jets with != 2 matched b-hadrons get NaN. Requires R > 0.4.
- **`SoftKillerModule`** — Applies SoftKiller pileup mitigation before clustering (`modules/pileup_rejection.py`). Auto-loaded when `--softkiller` and `--pu` are both set.
- **`VertexZFilterModule`** — Rejects jets with large pT-weighted mean vertex z after clustering (`modules/pileup_rejection.py`). Auto-loaded when `--max-dz` and `--pu` are both set.

### Using modules

The `--modules` flag accepts a mix of built-in short names, YAML config files, and import paths:

```bash
# Built-in short names (case-insensitive)
truthjets --pythia-card ttbar -n 100000 -o out.h5 --modules softkiller vertexzfilter

# YAML config file (with init_args for custom parameters)
truthjets --pythia-card ttbar -n 100000 -o out.h5 --modules pipeline.yaml

# Import path (existing behavior)
truthjets --pythia-card ttbar -n 100000 -o out.h5 --modules my_package.my_module:ClassName

# Mix all three
truthjets --pythia-card ttbar -n 100000 -o out.h5 --modules softkiller pipeline.yaml my_package:MyModule
```

**Built-in short names:** `softkiller`, `vertexzfilter`, `hadronconelabel`, `largerlabel`

**YAML config format:**
```yaml
- class_path: truthjets.modules.pileup_rejection:SoftKillerModule
  init_args:
    grid_size: 0.6

- class_path: truthjets.modules.pileup_rejection:VertexZFilterModule
  init_args:
    max_dz: 3.0
```

**Module ordering:** auto-loaded (config flags) → `--modules` → legacy `--module`, deduplicated by class (first wins).

The legacy `--module` flag (repeatable import path) still works:

```bash
# Legacy: explicit class name (repeatable)
truthjets --pythia-card ttbar -n 100000 -o out.h5 --module my_package.my_module:ClassName
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

All physics processes are configured via `.cmnd` card files shipped in `src/truthjets/cards/`. Use `--pythia-card <name>` with a built-in short name or a path to a custom `.cmnd` file.

**Built-in cards:**
- **`ttbar`** — Top pair production (gg and qqbar channels).
- **`qcd`** — Generic QCD hard processes.
- **`zprime_tt`** — Z' → ttbar at 3 TeV.
- **`z_qq`** — Boosted Z+jets with Z → qq (hadronic). `pTHatMin = 200`.
- **`zh_llbb`** — ZH associated production with H → bb, Z → ll. `pTHatMin = 150`.
- **`zh_vvbb`** — ZH associated production with H → bb, Z → vv (invisible). `pTHatMin = 150`.

`--process` and `--pythia-card` are aliases — both accept a built-in card name or a path to a `.cmnd` file.

## HDF5 Output Format

- `/jets` — structured array: `pt`, `eta`, `phi`, `mass`, `energy` (float32), `HadronConeExclTruthLabelID` (int32), `n_constituents` (int32)
- `/constituents` — shape [n_jets, max_constituents]: `pt`, `deta`, `dphi`, `energy` (float32), `pdgId` (int32), `valid` (bool)

## Bulk Generation Best Practices

When using `bulk-generate`, always use `--vds` to get a clean output structure with a virtual dataset:

```bash
bulk-generate --pythia-card ttbar -n 12500 --num-files 8 --parallel 4 --vds \
    -o /path/to/output/
```

This creates `parts/ttbar_*.h5` plus a `vds.h5` that presents all parts as a single concatenated dataset — much easier to work with downstream.

For pileup runs, pre-generate the PU pool to save memory and time (allows more parallel workers):

```bash
# 1. Generate PU pool once
truthjets --pythia-card ttbar -n 1 --pu 60 --pu-pre-gen 20000 --batch-size 1 -o /tmp/dummy.h5
# Pool saved as dummy_20000_pu_events.h5 in cwd

# 2. Use pool for bulk generation
bulk-generate --pythia-card ttbar --pu 60 --softkiller \
    --pu-file pu_pool_20k.h5 \
    -n 12500 --num-files 8 --parallel 4 --vds \
    -o /path/to/output/
```

Memory note: without a pre-generated pool, each worker generates its own PU events in memory (~7 GB at mu=60). On a 16 GB machine, limit `--parallel` to 1-2 without a pool, or 4+ with a pool.

## Benchmarking

Use `--benchmark` to print per-batch and summary timing for each pipeline stage: `event_generation`, `pileup_overlay`, `pre_clustering`, `jet_clustering`, `post_clustering`, `h5_writing`. Stages that don't fire (e.g. pileup when `--pu` is not set) are omitted from output.

## HTCondor Scripts

The `condor/` directory contains submit scripts for batch generation on CERN lxplus:

- **`condor/generate_pu_pool.sh`** / **`condor/generate_pu_pool.sub`** — Pileup pool generation. Each job runs `generate-pu-pool` with a unique seed, producing one pool chunk.
- **`condor/generate_jets.sh`** / **`condor/generate_jets.sub`** — Hard-scatter jet generation. Each job runs `truthjets` with a unique seed. Uses `--pythia-card` (accepts built-in names or paths).
- **`condor/logs/`** — HTCondor stdout/stderr/log files (gitignored via `.gitkeep`).

All config is passed at `condor_submit` time (no need to edit `.sub` files). Both wrappers handle the lxplus environment: source LCG_106, unset `PYTHIA8DATA`, activate venv.

## Key Dependencies

Core physics: `pythia8mc`, `fastjet`, `awkward>=2.0`, `vector`. Data I/O: `h5py`, `numpy`. Config: `pyyaml`.
