# Truth Jet Generation

Generate truth-level jet training data for ML (per-jet models) without full ATLAS simulation. Uses Pythia8 for event generation, FastJet for jet clustering, and writes HDF5 output compatible with ATLAS ftag training tools (Salt/Umami).

## Setup

Requires a C++ compiler (`g++`) for building `pythia8mc`.

```bash
# Install uv if you don't have it
pip install uv

# Create venv and install
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Generating jets

After activating the venv (`source .venv/bin/activate`):

```bash
# QCD jets (1M events)
truthjets --process qcd -n 1000000 -o qcd_jets.h5

# ttbar (good source of b-jets)
truthjets --process ttbar -n 100000 -o ttbar_jets.h5

# Z' -> ttbar BSM process
truthjets --process zprime_tt -n 100000 -o zprime_jets.h5
```

### Options

**Config files** (override individual flags):

| Flag | Default | Description |
|------|---------|-------------|
| `--pythia-config` | — | Path to Pythia YAML config file |
| `--jet-config` | — | Path to jet/output YAML config file |

**Pythia settings** (exactly one of `--process` or `--pythia-card` is required):

| Flag | Default | Description |
|------|---------|-------------|
| `--process` | — | Physics process preset (`qcd`, `ttbar`, `zprime_tt`) |
| `--pythia-card` | — | Path to a Pythia command file (`.cmnd`) |
| `--ecm` | `13600` | Centre-of-mass energy [GeV] |
| `--pt-hat-min` | — | Minimum pTHat cut [GeV] |
| `--pt-hat-max` | — | Maximum pTHat cut [GeV] |
| `--seed` | `42` | Random seed |

**Pileup:**

| Flag | Default | Description |
|------|---------|-------------|
| `--pu MU` | — | Mean number of pileup interactions (Poisson mu). Disabled by default |
| `--pu-pre-gen N` | — | Pre-generate N PU events upfront, save to file, then sample from pool |
| `--pu-file PATH` | — | Load pre-generated PU pool from file (skip Pythia PU generation) |

**Jet clustering:**

| Flag | Default | Description |
|------|---------|-------------|
| `-R` | `0.4` | Jet radius |
| `--jet-pt-min` | `20` | Minimum jet pT [GeV] |
| `--jet-eta-max` | `2.5` | Maximum jet \|eta\| |
| `--constituent-pt-min` | `0.5` | Minimum constituent pT [GeV] |
| `--max-constituents` | `80` | Constituents per jet (zero-padded) |

**Pileup rejection:**

| Flag | Default | Description |
|------|---------|-------------|
| `--softkiller` | off | Enable SoftKiller pileup mitigation before clustering |
| `--softkiller-grid` | `0.4` | SoftKiller grid size in rapidity-phi |
| `--max-dz` | — | Vertex z cut in mm — reject jets with \|&lt;vz&gt;\| > max_dz |

**Output:**

| Flag | Default | Description |
|------|---------|-------------|
| `-o`, `--output` | `jets.h5` | Output HDF5 path |
| `-n`, `--n-events` | `100000` | Number of events |
| `--batch-size` | `10000` | Events per batch |

## Plotting

Four plotting scripts are provided in `scripts/`:

### `plot_jets.py` — Jet-level distributions

```bash
python scripts/plot_jets.py <input.h5> [-o jets.pdf]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | (required) | Input HDF5 file |
| `-o`, `--output` | `jets.pdf` | Output PDF path |

### `plot_events.py` — Event-level distributions

```bash
python scripts/plot_events.py <input.h5> [-o events.pdf]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | (required) | Input HDF5 file |
| `-o`, `--output` | `events.pdf` | Output PDF path |

### `plot_constituents.py` — Constituent-level distributions

```bash
python scripts/plot_constituents.py <input.h5> [-o constituents.pdf]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | (required) | Input HDF5 file |
| `-o`, `--output` | `constituents.pdf` | Output PDF path |

### `plot_all.py` — Run all plot scripts at once

```bash
python scripts/plot_all.py <input.h5> [-o plots/]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `input` | (required) | Input HDF5 file |
| `-o`, `--output-dir` | `plots/` | Output directory for PDFs |

Produces `events.pdf`, `jets.pdf`, and `constituents.pdf` in the output directory.

## Pileup pool generation

Pre-generate a pool of min-bias events for reuse across multiple hard-scatter runs. This avoids re-running Pythia for pileup each time. Pool files are saved with gzip-7 compression.

```bash
# Generate a pool of 100k min-bias events
generate-pu-pool -n 100000 -o pu_pool.h5

# Use it with hard-scatter generation
truthjets --process ttbar -n 100000 --pu 50 --pu-file pu_pool.h5 -o ttbar_pu.h5
```

| Flag | Default | Description |
|------|---------|-------------|
| `-n`, `--n-events` | (required) | Number of min-bias events to generate |
| `-o`, `--output` | (required) | Output HDF5 file path |
| `--ecm` | `13600` | Centre-of-mass energy [GeV] |
| `--seed` | `42` | Random seed |
| `--batch-size` | `10000` | Events per Pythia batch |

### Loading pool chunks from a directory

`--pu-file` accepts either a single HDF5 file or a directory of pool chunks. When given a directory, all `*.h5` files inside are loaded (sorted by filename) and concatenated into one pool. This is useful for large-scale production where pool generation is split across batch jobs (e.g. HTCondor on lxplus):

```bash
# Each batch job produces a chunk
generate-pu-pool -n 1000000 --seed 1 -o pool_chunks/chunk_000.h5
generate-pu-pool -n 1000000 --seed 2 -o pool_chunks/chunk_001.h5
# ...

# Point at the directory — all chunks are loaded and merged
truthjets --process ttbar -n 100000 --pu 50 --pu-file pool_chunks/ -o ttbar_pu.h5
```

## Bulk generation

Generate multiple HDF5 files in parallel with unique seeds:

```bash
# Using a process preset
bulk-generate --process ttbar -n 100000 --num-files 10 \
    --parallel 4 -o output/ttbar/

# Using a Pythia card
bulk-generate --pythia-card cards/z_qq.cmnd -n 100000 --num-files 10 \
    --parallel 4 -R 1.0 -o output/z_qq/
```

This produces `output/ttbar/ttbar_000.h5` through `ttbar_009.h5`, each with 100k events. With `--pythia-card`, the card filename stem is used as the prefix (e.g. `z_qq_000.h5`).

| Flag | Default | Description |
|------|---------|-------------|
| `--process` | — | Physics process preset (exactly one of `--process` or `--pythia-card` required) |
| `--pythia-card` | — | Path to a Pythia command file (`.cmnd`) |
| `--prefix` | — | File name prefix (default: process name or card stem) |
| `-o`, `--output-dir` | (required) | Output directory for HDF5 files (staging dir if `--final-dir` is set) |
| `-n`, `--events-per-file` | (required) | Number of events per file |
| `--num-files` | (required) | Number of files to generate |
| `--parallel` | `1` | Number of parallel processes |
| `--seed-start` | `1` | Starting seed; file *i* gets seed = seed_start + *i* |
| `--final-dir` | — | Final directory to move completed files to (e.g. HDD) |
| `--vds` | off | Create an HDF5 Virtual Dataset concatenating all part files |

Any extra flags are forwarded to `truthjets` (e.g. `--ecm 14000 -R 1.0 --pu 60`).

### Virtual Dataset (`--vds`)

The `--vds` flag creates a single `vds.h5` that virtually concatenates all part files without copying data, so you can treat the output as one file for plotting and analysis.

```bash
# Parts and VDS in the same directory
bulk-generate --process ttbar -n 100000 --num-files 10 --vds -o output/ttbar/

# With staging on SSD, final output on HDD
bulk-generate --process ttbar -n 100000 --num-files 10 --vds \
    -o /fast-ssd/staging/ --final-dir /large-hdd/ttbar/
```

Output structure:
```
output/ttbar/          # or --final-dir if set
├── parts/
│   ├── ttbar_000.h5
│   ├── ttbar_001.h5
│   └── ...
└── vds.h5             ← virtual dataset (references parts/ via relative paths)
```

The VDS uses relative paths, so the entire output directory can be moved or renamed. Only successful jobs are included.

## Creating Virtual Datasets (`create-vds`)

Standalone CLI for creating HDF5 Virtual Datasets from any set of HDF5 files (pool chunks, jet files, etc.):

```bash
# From explicit files
create-vds part_000.h5 part_001.h5 part_002.h5 -o combined.h5

# From a directory (all *.h5 files inside, sorted by name)
create-vds /path/to/parts/ -o combined.h5
```

| Flag | Default | Description |
|------|---------|-------------|
| `inputs` | (required) | Input HDF5 files, or a single directory containing `*.h5` files |
| `-o`, `--output` | (required) | Output VDS file path |

The VDS uses relative paths for portability. All datasets found in the input files are concatenated along axis 0, with per-dataset size tracking (so files where different datasets have different axis-0 sizes, like pileup pools, are handled correctly).

## Batch generation on lxplus (HTCondor)

The `condor/` directory contains submit scripts for running on CERN's lxplus batch system.

### One-time setup on lxplus

```bash
# Source LCG view for Python 3.11+ and system packages (fastjet, ROOT, etc.)
source /cvmfs/sft.cern.ch/lcg/views/LCG_106/x86_64-el9-gcc13-opt/setup.sh

# Unset PYTHIA8DATA to avoid version mismatch (LCG has 8.312, pip has 8.317)
unset PYTHIA8DATA

# Create venv with system-site-packages (inherits fastjet, etc. from LCG)
cd ~/truth-jet-generation
python -m venv --system-site-packages .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Generating pileup pools

```bash
# Create a seeds file (one seed per job)
seq 1 100 > seeds.txt

# Submit 100 pool-generation jobs (100k events each)
condor_submit condor/generate_pu_pool.sub \
    project_dir=/afs/cern.ch/user/j/jsmith/truth-jet-generation \
    output_dir=/eos/user/j/jsmith/truthjets/pu_pool \
    n_events=100000
```

### Generating jets

```bash
# ttbar with a process preset
condor_submit condor/generate_jets.sub \
    project_dir=/afs/cern.ch/user/j/jsmith/truth-jet-generation \
    output_dir=/eos/user/j/jsmith/truthjets/ttbar \
    process_flag=--process process_val=ttbar \
    prefix=ttbar n_events=100000

# Large-R jets with a Pythia card
condor_submit condor/generate_jets.sub \
    project_dir=/afs/cern.ch/user/j/jsmith/truth-jet-generation \
    output_dir=/eos/user/j/jsmith/truthjets/z_qq \
    process_flag=--pythia-card \
    process_val=/afs/cern.ch/user/j/jsmith/truth-jet-generation/cards/z_qq.cmnd \
    prefix=z_qq n_events=100000 \
    extra_args="-R 1.0"

# With pileup (uses pre-generated pool)
condor_submit condor/generate_jets.sub \
    project_dir=/afs/cern.ch/user/j/jsmith/truth-jet-generation \
    output_dir=/eos/user/j/jsmith/truthjets/ttbar_pu \
    process_flag=--process process_val=ttbar \
    prefix=ttbar n_events=100000 \
    extra_args="--pu 50 --pu-file /eos/user/j/jsmith/truthjets/pu_pool/"
```

### Post-processing

After jobs complete, create a Virtual Dataset for easy downstream use:

```bash
create-vds /eos/user/j/jsmith/truthjets/ttbar/ -o /eos/user/j/jsmith/truthjets/ttbar/vds.h5
```

## Output format

The HDF5 file contains two datasets:

**`/jets`** — one entry per jet:
- `pt`, `eta`, `phi`, `mass`, `energy` (float32)
- `HadronConeExclTruthLabelID` (int32): `0`=light, `4`=c, `5`=b, `15`=tau
- `n_constituents` (int32)

**`/constituents`** — shape `[n_jets, max_constituents]`, zero-padded:
- `pt`, `deta`, `dphi`, `energy` (float32): deta/dphi are relative to the jet axis
- `pdgId` (int32)
- `valid` (bool): `True` for real constituents, `False` for padding

Constituents are sorted by pT descending.

## Tests

```bash
uv pip install -e ".[dev]"
pytest tests/
```
