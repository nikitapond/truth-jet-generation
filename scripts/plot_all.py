#!/usr/bin/env python
"""Run all plot scripts and save outputs to a directory."""
from __future__ import annotations

import argparse
from pathlib import Path

from truthjets.plotting import load_data, save_figs
from truthjets.plotting.events import plot as plot_events
from truthjets.plotting.jets import plot as plot_jets
from truthjets.plotting.constituents import plot as plot_constituents

PLOTTERS = {
    "events.pdf": plot_events,
    "jets.pdf": plot_jets,
    "constituents.pdf": plot_constituents,
}


def main():
    parser = argparse.ArgumentParser(description="Run all truth jet plot scripts")
    parser.add_argument("input", help="Input HDF5 file")
    parser.add_argument(
        "-o", "--output-dir", default="plots",
        help="Output directory for PDFs (default: plots/)",
    )
    args = parser.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Load data once, reuse for all plotters
    jets, constit = load_data(args.input)

    for filename, plot_fn in PLOTTERS.items():
        figs = plot_fn(jets, constit)
        save_figs(figs, outdir / filename)


if __name__ == "__main__":
    main()
