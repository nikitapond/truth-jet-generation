#!/usr/bin/env python
"""Plot jet-level distributions from HDF5 output."""
from __future__ import annotations

import argparse

from truthjets.plotting import load_data, save_figs
from truthjets.plotting.jets import plot


def main():
    parser = argparse.ArgumentParser(description="Plot jet-level distributions")
    parser.add_argument("input", help="Input HDF5 file")
    parser.add_argument("-o", "--output", default="jets.pdf", help="Output PDF path")
    args = parser.parse_args()

    jets, constit = load_data(args.input)
    figs = plot(jets, constit)
    save_figs(figs, args.output)


if __name__ == "__main__":
    main()
