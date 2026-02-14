#!/usr/bin/env python
"""Generate a reusable pileup pool HDF5 file.

Example:
    generate-pu-pool -n 100000 -o pu_pool.h5
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from truthjets.config import PythiaConfig
from truthjets.generate import init_pileup_pythia
from truthjets.pileup import generate_pileup_pool, save_pileup_pool


def main():
    parser = argparse.ArgumentParser(
        description="Generate a pileup pool HDF5 file for reuse across hard-scatter runs"
    )
    parser.add_argument(
        "-n", "--n-events", type=int, required=True,
        help="Number of min-bias events to generate",
    )
    parser.add_argument(
        "-o", "--output", type=Path, required=True,
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--ecm", type=float, default=13600.0,
        help="Centre-of-mass energy in GeV (default: 13600.0)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=10000,
        help="Events per Pythia batch (default: 10000)",
    )

    args = parser.parse_args()

    config = PythiaConfig(ecm=args.ecm, seed=args.seed)

    print(f"Generating pileup pool: {args.n_events} min-bias events")
    print(f"  ecm={config.ecm}, seed={config.seed}, batch_size={args.batch_size}")

    t0 = time.time()
    pythia_pu = init_pileup_pythia(config)
    pool = generate_pileup_pool(pythia_pu, args.n_events, batch_size=args.batch_size)
    save_pileup_pool(pool, args.output)
    elapsed = time.time() - t0

    import awkward as ak
    total_particles = int(ak.sum(ak.num(pool, axis=1)))
    file_size_mb = args.output.stat().st_size / (1024 * 1024)

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Events:     {args.n_events}")
    print(f"  Particles:  {total_particles}")
    print(f"  Output:     {args.output} ({file_size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
