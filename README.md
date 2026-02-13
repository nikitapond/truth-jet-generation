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

| Flag | Default | Description |
|------|---------|-------------|
| `--process` | `qcd` | Physics process (`qcd`, `ttbar`, `zprime_tt`) |
| `--ecm` | `13600` | Centre-of-mass energy [GeV] |
| `--pt-hat-min` | — | Minimum pTHat cut [GeV] |
| `--pt-hat-max` | — | Maximum pTHat cut [GeV] |
| `--seed` | `42` | Random seed |
| `-R` | `0.4` | Jet radius |
| `--jet-pt-min` | `20` | Minimum jet pT [GeV] |
| `--jet-eta-max` | `2.5` | Maximum jet \|eta\| |
| `--max-constituents` | `80` | Constituents per jet (zero-padded) |
| `-n` | `100000` | Number of events |
| `--batch-size` | `10000` | Events per batch |
| `-o` | `jets.h5` | Output HDF5 path |

## Plotting

```bash
python scripts/plot_jets.py ttbar_jets.h5 -o ttbar_plots.pdf
```

Produces a 6-panel PDF with jet pT, eta, mass, flavor composition, constituent multiplicity, and leading constituent pT fraction — all broken down by truth flavor label.

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
