from __future__ import annotations

import argparse
import time

import numpy as np

from truthjets.cluster import cluster_jets, compute_jet_kinematics
from truthjets.config import (
    JetConfig,
    OutputConfig,
    PythiaConfig,
    load_jet_and_output_config,
    load_pythia_config,
)
from truthjets.generate import generate_events, generate_pileup_batch, init_pileup_pythia, init_pythia
from truthjets.label import label_jets
from truthjets.pileup import overlay_pileup, sample_n_pileup
from truthjets.writer import HDF5Writer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate truth-level jet training data"
    )

    # Config files (override individual flags)
    parser.add_argument(
        "--pythia-config", default=None,
        help="Path to Pythia YAML config file",
    )
    parser.add_argument(
        "--jet-config", default=None,
        help="Path to jet/output YAML config file",
    )

    # Pythia settings
    parser.add_argument(
        "--process",
        default="qcd",
        help="Physics process preset (default: qcd)",
    )
    parser.add_argument(
        "--ecm", type=float, default=13600.0, help="Center-of-mass energy in GeV"
    )
    parser.add_argument("--pt-hat-min", type=float, default=None)
    parser.add_argument("--pt-hat-max", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pu", type=float, default=None, metavar="MU",
        help="Mean number of pileup interactions (Poisson mu). Disabled by default.",
    )

    # Jet settings
    parser.add_argument("-R", type=float, default=0.4, help="Jet radius")
    parser.add_argument(
        "--jet-pt-min", type=float, default=20.0, help="Minimum jet pT in GeV"
    )
    parser.add_argument(
        "--jet-eta-max", type=float, default=2.5, help="Maximum jet |eta|"
    )
    parser.add_argument(
        "--max-constituents",
        type=int,
        default=80,
        help="Max constituents per jet (zero-padded)",
    )

    # Output settings
    parser.add_argument(
        "-o", "--output", default="jets.h5", help="Output HDF5 path"
    )
    parser.add_argument(
        "-n", "--n-events", type=int, default=100_000, help="Number of events"
    )
    parser.add_argument(
        "--batch-size", type=int, default=10_000, help="Events per batch"
    )

    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    # Load from config files if provided, then override with CLI flags
    if args.pythia_config:
        pythia_config = load_pythia_config(args.pythia_config)
    else:
        pythia_config = PythiaConfig()

    if args.jet_config:
        jet_config, output_config = load_jet_and_output_config(args.jet_config)
    else:
        jet_config = JetConfig()
        output_config = OutputConfig()

    # CLI flags override config file values
    pythia_config.process = args.process
    pythia_config.ecm = args.ecm
    pythia_config.seed = args.seed
    if args.pt_hat_min is not None:
        pythia_config.pt_hat_min = args.pt_hat_min
    if args.pt_hat_max is not None:
        pythia_config.pt_hat_max = args.pt_hat_max

    jet_config.R = args.R
    jet_config.pt_min = args.jet_pt_min
    jet_config.eta_max = args.jet_eta_max
    jet_config.max_constituents = args.max_constituents
    output_config.output_path = args.output
    output_config.n_events = args.n_events
    output_config.batch_size = args.batch_size

    # Set pileup mu from CLI
    if args.pu is not None:
        pythia_config.mu = args.pu

    print(f"Process: {pythia_config.process}")
    print(f"ECM: {pythia_config.ecm} GeV")
    if pythia_config.mu is not None:
        print(f"Pileup: <mu> = {pythia_config.mu}")
    print(f"Jet R={jet_config.R}, pT>{jet_config.pt_min} GeV, |eta|<{jet_config.eta_max}")
    print(f"Events: {output_config.n_events}, batch size: {output_config.batch_size}")
    print(f"Output: {output_config.output_path}")
    print()

    # Initialize Pythia
    pythia = init_pythia(pythia_config)

    # Initialize pileup Pythia if mu is set
    pythia_pu = None
    pu_rng = None
    if pythia_config.mu is not None:
        pythia_pu = init_pileup_pythia(pythia_config)
        pu_rng = np.random.default_rng(pythia_config.seed + 100)

    t0 = time.time()

    event_offset = 0
    with HDF5Writer(output_config.output_path, jet_config) as writer:
        for i, events in enumerate(
            generate_events(pythia, output_config.n_events, output_config.batch_size)
        ):
            t_batch = time.time()

            # Pileup overlay
            merged_particles = None
            if pythia_pu is not None:
                n_events_in_batch = len(events.prt)
                n_pu = sample_n_pileup(pythia_config.mu, n_events_in_batch, pu_rng)
                total_pu = int(np.sum(n_pu))
                pu_events = generate_pileup_batch(pythia_pu, total_pu)
                merged_particles = overlay_pileup(events, pu_events, n_pu)

            # Cluster jets (with merged particles if pileup is active)
            jets, constits, jet_kin = cluster_jets(
                events, jet_config, particles=merged_particles
            )

            # Label jets using only HS events (not PU)
            labels = label_jets(events, jet_kin.eta, jet_kin.phi, jet_config.R)

            # Write to HDF5
            writer.write_batch(
                jet_kin, labels, constits, jet_kin.eta, jet_kin.phi,
                event_offset=event_offset,
            )
            event_offset += len(events.prt)

            elapsed = time.time() - t_batch
            print(
                f"Batch {i + 1}: {writer.n_jets} jets total "
                f"({elapsed:.1f}s this batch)"
            )

    total_time = time.time() - t0
    print(f"\nDone. {writer.n_jets} jets written to {output_config.output_path}")
    print(f"Total time: {total_time:.1f}s")

    # Print Pythia statistics
    pythia.stat()


if __name__ == "__main__":
    main()
