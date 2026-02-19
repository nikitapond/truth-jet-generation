from __future__ import annotations

import argparse
import time
from pathlib import Path

import awkward as ak
import numpy as np

from truthjets.benchmark import Benchmark
from truthjets.cluster import cluster_jets, extract_particles
from truthjets.config import (
    JetConfig,
    OutputConfig,
    PythiaConfig,
    load_jet_and_output_config,
    load_pythia_config,
    resolve_card,
)
from truthjets.generate import generate_events, generate_pileup_batch, init_pileup_pythia, init_pythia
from truthjets.modules import (
    HadronConeExclLabelModule,
    LargeRLabelModule,
    SoftKillerModule,
    VertexZFilterModule,
    deduplicate_modules,
    load_module,
    resolve_module_specs,
    validate_modules,
)
from truthjets.pileup import (
    generate_pileup_pool,
    load_pileup_pool,
    overlay_pileup,
    sample_from_pool,
    sample_n_pileup,
    save_pileup_pool,
)
from truthjets.writer import HDF5Writer


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Generate truth-level jet training data")

    # Config files (override individual flags)
    parser.add_argument(
        "--pythia-config",
        default=None,
        help="Path to Pythia YAML config file",
    )
    parser.add_argument(
        "--jet-config",
        default=None,
        help="Path to jet/output YAML config file",
    )

    # Pythia settings
    parser.add_argument(
        "--process",
        "--pythia-card",
        dest="pythia_card",
        default=None,
        help="Pythia card: a built-in name (e.g. ttbar, qcd, zprime_tt) or path to a .cmnd file",
    )
    parser.add_argument("--ecm", type=float, default=13600.0, help="Center-of-mass energy in GeV")
    parser.add_argument("--pt-hat-min", type=float, default=None)
    parser.add_argument("--pt-hat-max", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--pu",
        type=float,
        default=None,
        metavar="MU",
        help="Mean number of pileup interactions (Poisson mu). Disabled by default.",
    )
    parser.add_argument(
        "--pu-pre-gen",
        type=int,
        default=None,
        metavar="N",
        help="Pre-generate N PU events upfront, save to file, then sample from pool.",
    )
    parser.add_argument(
        "--pu-file",
        type=str,
        default=None,
        metavar="PATH",
        help="Load pre-generated PU pool from file (skip Pythia PU generation).",
    )

    # Jet settings
    parser.add_argument("-R", type=float, default=0.4, help="Jet radius")
    parser.add_argument("--jet-pt-min", type=float, default=20.0, help="Minimum jet pT in GeV")
    parser.add_argument("--jet-eta-max", type=float, default=2.5, help="Maximum jet |eta|")
    parser.add_argument(
        "--constituent-pt-min",
        type=float,
        default=0.5,
        help="Minimum constituent pT in GeV (default: 0.5)",
    )
    parser.add_argument(
        "--max-constituents",
        type=int,
        default=80,
        help="Max constituents per jet (zero-padded)",
    )

    # Pileup rejection
    parser.add_argument(
        "--softkiller",
        action="store_true",
        help="Enable SoftKiller pileup mitigation before clustering",
    )
    parser.add_argument(
        "--softkiller-grid",
        type=float,
        default=0.4,
        help="SoftKiller grid size in rapidity-phi (default: 0.4)",
    )
    parser.add_argument(
        "--max-dz",
        type=float,
        default=None,
        help="Vertex z cut in mm — reject jets with |<vz>| > max_dz",
    )

    # Output settings
    parser.add_argument("-o", "--output", default="jets.h5", help="Output HDF5 path")
    parser.add_argument("-n", "--n-events", type=int, default=100_000, help="Number of events")
    parser.add_argument("--batch-size", type=int, default=10_000, help="Events per batch")

    # Pipeline modules
    parser.add_argument(
        "--module",
        action="append",
        default=[],
        metavar="IMPORT_PATH",
        help=(
            "Pipeline module to load (repeatable). Format: 'my_package.my_module' or 'my_package.my_module:ClassName'"
        ),
    )
    parser.add_argument(
        "--modules",
        nargs="*",
        default=[],
        metavar="SPEC",
        help=(
            "Pipeline modules to load. Each SPEC can be: a built-in name "
            "(e.g. 'softkiller', 'vertexzfilter'), a YAML config file "
            "(*.yaml/*.yml), or an import path."
        ),
    )

    # Benchmarking
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Print per-batch and summary timing for each pipeline stage",
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
    if args.pythia_card is not None:
        pythia_config.pythia_card = resolve_card(args.pythia_card)

    # Validate: pythia_card must be set (via --pythia-card/--process or config file)
    if pythia_config.pythia_card is None:
        raise SystemExit("error: must specify --process or --pythia-card")

    pythia_config.ecm = args.ecm
    pythia_config.seed = args.seed
    if args.pt_hat_min is not None:
        pythia_config.pt_hat_min = args.pt_hat_min
    if args.pt_hat_max is not None:
        pythia_config.pt_hat_max = args.pt_hat_max

    jet_config.R = args.R
    jet_config.pt_min = args.jet_pt_min
    jet_config.eta_max = args.jet_eta_max
    jet_config.constituent_pt_min = args.constituent_pt_min
    jet_config.max_constituents = args.max_constituents
    if args.softkiller:
        jet_config.softkiller = True
    jet_config.softkiller_grid = args.softkiller_grid
    if args.max_dz is not None:
        jet_config.max_dz = args.max_dz
    output_config.output_path = args.output
    output_config.n_events = args.n_events
    output_config.batch_size = args.batch_size

    # Set pileup mu from CLI
    if args.pu is not None:
        pythia_config.mu = args.pu

    # Pileup pool settings
    if args.pu_pre_gen is not None:
        pythia_config.pu_pre_gen = args.pu_pre_gen
    if args.pu_file is not None:
        pythia_config.pu_file = args.pu_file

    # Validate pileup pool flags
    if pythia_config.pu_pre_gen is not None and pythia_config.pu_file is not None:
        raise SystemExit("error: cannot use both --pu-pre-gen and --pu-file")
    if (pythia_config.pu_pre_gen is not None or pythia_config.pu_file is not None) and pythia_config.mu is None:
        raise SystemExit("error: --pu-pre-gen and --pu-file require --pu to be set")

    print(f"Pythia card: {pythia_config.pythia_card}")
    print(f"ECM: {pythia_config.ecm} GeV")
    if pythia_config.mu is not None:
        print(f"Pileup: <mu> = {pythia_config.mu}")
    if jet_config.softkiller:
        print(f"SoftKiller: grid_size={jet_config.softkiller_grid}")
    if jet_config.max_dz is not None:
        print(f"Vertex z cut: |<vz>| < {jet_config.max_dz} mm")
    print(f"Jet R={jet_config.R}, pT>{jet_config.pt_min} GeV, |eta|<{jet_config.eta_max}")
    print(f"Events: {output_config.n_events}, batch size: {output_config.batch_size}")
    print(f"Output: {output_config.output_path}")
    print()

    # Initialize Pythia
    pythia = init_pythia(pythia_config)

    # Initialize pileup Pythia if mu is set (skip if using pre-generated file)
    pythia_pu = None
    pu_rng = None
    if pythia_config.mu is not None:
        pu_rng = np.random.default_rng(pythia_config.seed + 100)
        if pythia_config.pu_file is None:
            pythia_pu = init_pileup_pythia(pythia_config)

    # Load or generate pileup pool
    pu_pool = None
    if pythia_config.pu_file:
        print(f"Loading PU pool from {pythia_config.pu_file}")
        pu_pool = load_pileup_pool(pythia_config.pu_file)
        print(f"  Pool size: {len(pu_pool)} events")
    elif pythia_config.pu_pre_gen:
        print(f"Pre-generating {pythia_config.pu_pre_gen} PU events...")
        pu_pool = generate_pileup_pool(pythia_pu, pythia_config.pu_pre_gen)
        pool_path = f"{Path(output_config.output_path).stem}_{pythia_config.pu_pre_gen}_pu_events.h5"
        save_pileup_pool(pu_pool, pool_path)
        print(f"  Saved pool to {pool_path}")

    # Load and initialize pipeline modules
    modules = []

    # Auto-add labeling module based on jet radius
    if jet_config.R == 0.4:
        modules.append(HadronConeExclLabelModule())
    elif jet_config.R > 0.4:
        modules.append(LargeRLabelModule())

    # Auto-add pileup rejection modules when configured
    if jet_config.softkiller and pythia_config.mu is not None:
        modules.append(SoftKillerModule())
    if jet_config.max_dz is not None and pythia_config.mu is not None:
        modules.append(VertexZFilterModule())

    # Resolve --modules specs (built-in names, YAML files, import paths)
    if args.modules:
        modules.extend(resolve_module_specs(args.modules))

    # Legacy --module import paths
    for module_path in args.module:
        mod = load_module(module_path)
        modules.append(mod)

    # Deduplicate (first occurrence wins — auto-loaded takes priority)
    modules = deduplicate_modules(modules)

    for mod in modules:
        mod.init(jet_config)

    if modules:
        validate_modules(modules)
        print(f"Loaded {len(modules)} module(s): {[type(m).__name__ for m in modules]}")

    # Collect extra schemas from modules
    extra_jet_fields = []
    extra_datasets = {}
    for mod in modules:
        extra_jet_fields.extend(mod.extra_jet_fields())
        extra_datasets.update(mod.extra_datasets())

    bench = Benchmark(enabled=args.benchmark)
    t0 = time.time()

    event_offset = 0
    with HDF5Writer(
        output_config.output_path,
        jet_config,
        extra_jet_fields=extra_jet_fields or None,
        extra_datasets=extra_datasets or None,
    ) as writer:
        event_iter = generate_events(pythia, output_config.n_events, output_config.batch_size)
        n_batches = (output_config.n_events + output_config.batch_size - 1) // output_config.batch_size
        for batch_i in range(n_batches):
            t_batch = time.time()
            bench.start("event_generation")
            events = next(event_iter)
            bench.stop("event_generation")

            # Pileup overlay
            bench.start("pileup_overlay")
            merged_particles = None
            if pythia_config.mu is not None:
                n_events_in_batch = len(events.prt)
                n_pu = sample_n_pileup(pythia_config.mu, n_events_in_batch, pu_rng)
                total_pu = int(np.sum(n_pu))
                if pu_pool is not None:
                    pu_particles = sample_from_pool(pu_pool, total_pu, rng=pu_rng)
                    merged_particles = overlay_pileup(
                        events,
                        None,
                        n_pu,
                        rng=pu_rng,
                        pu_particles=pu_particles,
                    )
                else:
                    pu_events = generate_pileup_batch(pythia_pu, total_pu)
                    merged_particles = overlay_pileup(
                        events,
                        pu_events,
                        n_pu,
                        rng=pu_rng,
                    )
            bench.stop("pileup_overlay")

            # Extract particles for pre_clustering hooks (when no pileup)
            if modules and merged_particles is None:
                merged_particles = extract_particles(events)

            # Pre-clustering hooks
            bench.start("pre_clustering")
            for mod in modules:
                mod_name = type(mod).__name__
                bench.start(f"pre_clustering/{mod_name}")
                result = mod.pre_clustering(events, merged_particles)
                bench.stop(f"pre_clustering/{mod_name}")
                if result is not None:
                    merged_particles = result
            bench.stop("pre_clustering")

            # Cluster jets (with merged particles if pileup is active)
            bench.start("jet_clustering")
            jets, constits, jet_kin = cluster_jets(events, jet_config, particles=merged_particles)
            bench.stop("jet_clustering")

            # Default labels (all light); labeling modules override via post_clustering
            labels = ak.zeros_like(jet_kin.pt, dtype=np.int32)

            # Post-clustering hooks
            bench.start("post_clustering")
            batch_extra_jet_data = {}
            batch_extra_dataset_data = {}
            for mod in modules:
                mod_name = type(mod).__name__
                bench.start(f"post_clustering/{mod_name}")
                result = mod.post_clustering(events, jets, constits, jet_kin, labels)
                bench.stop(f"post_clustering/{mod_name}")
                if result is not None:
                    if result.jet_kin is not None:
                        jet_kin = result.jet_kin
                    if result.labels is not None:
                        labels = result.labels
                    if result.jets is not None:
                        jets = result.jets
                    if result.constituents is not None:
                        constits = result.constituents
                    batch_extra_jet_data.update(result.extra_jet_data)
                    batch_extra_dataset_data.update(result.extra_dataset_data)
            bench.stop("post_clustering")

            # Write to HDF5
            bench.start("h5_writing")
            writer.write_batch(
                jet_kin,
                labels,
                constits,
                jet_kin.eta,
                jet_kin.phi,
                event_offset=event_offset,
                extra_jet_data=batch_extra_jet_data or None,
                extra_dataset_data=batch_extra_dataset_data or None,
            )
            bench.stop("h5_writing")

            event_offset += len(events.prt)
            bench.end_batch()

            elapsed = time.time() - t_batch
            print(f"Batch {batch_i + 1}: {writer.n_jets} jets total ({elapsed:.1f}s this batch)")

    total_time = time.time() - t0
    print(f"\nDone. {writer.n_jets} jets written to {output_config.output_path}")
    print(f"Total time: {total_time:.1f}s")
    bench.report()

    # Print Pythia statistics
    pythia.stat()


if __name__ == "__main__":
    main()
