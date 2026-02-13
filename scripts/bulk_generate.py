#!/usr/bin/env python
"""Bulk-generate truth jet files in parallel with unique seeds.

Example:
    python scripts/bulk_generate.py --process ttbar -n 100000 --num-files 10 \
        --parallel 4 -o output/ttbar/

This will produce output/ttbar/ttbar_000.h5 through ttbar_009.h5, each with
100k events and a unique Pythia seed.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from multiprocessing import Pool
from pathlib import Path


def run_job(job: dict) -> dict:
    """Run a single truthjets generation job as a subprocess."""
    cmd = [
        sys.executable, "-m", "truthjets.cli",
        "--process", job["process"],
        "-n", str(job["n_events"]),
        "--seed", str(job["seed"]),
        "-o", str(job["output"]),
    ]
    # Forward any extra CLI flags
    cmd.extend(job.get("extra_args", []))

    print(f"[{job['index'] + 1}/{job['total']}] Starting: {job['output']} (seed={job['seed']})")
    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0

    status = "OK" if result.returncode == 0 else "FAILED"
    print(f"[{job['index'] + 1}/{job['total']}] {status}: {job['output']} ({elapsed:.1f}s)")

    if result.returncode != 0:
        print(f"  stderr: {result.stderr.strip()}")

    # Move file from staging to final directory
    final_path = None
    if result.returncode == 0 and job.get("final_dir"):
        src = Path(job["output"])
        dst = Path(job["final_dir"]) / src.name
        shutil.move(str(src), str(dst))
        final_path = str(dst)
        print(f"[{job['index'] + 1}/{job['total']}] Moved -> {dst}")

    return {
        "output": final_path or str(job["output"]),
        "seed": job["seed"],
        "returncode": result.returncode,
        "stderr": result.stderr.strip(),
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Bulk-generate truth jet HDF5 files in parallel"
    )
    parser.add_argument(
        "--process", required=True,
        help="Physics process preset (e.g. ttbar, qcd, zprime_tt)",
    )
    parser.add_argument(
        "-o", "--output-dir", required=True, type=Path,
        help="Output directory for HDF5 files (used as staging dir if --final-dir is set)",
    )
    parser.add_argument(
        "--final-dir", type=Path, default=None,
        help="Final directory to move completed files to (e.g. HDD). "
             "Files are generated in --output-dir (e.g. SSD) then moved.",
    )
    parser.add_argument(
        "-n", "--events-per-file", type=int, required=True,
        help="Number of events per file",
    )
    parser.add_argument(
        "--num-files", type=int, required=True,
        help="Number of files to generate",
    )
    parser.add_argument(
        "--parallel", type=int, default=1,
        help="Number of parallel processes (default: 1)",
    )
    parser.add_argument(
        "--seed-start", type=int, default=1,
        help="Starting seed; file i gets seed = seed_start + i (default: 1)",
    )

    args, extra = parser.parse_known_args()

    # Create directories
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.final_dir:
        args.final_dir.mkdir(parents=True, exist_ok=True)

    # Build job list
    jobs = []
    for i in range(args.num_files):
        output_path = args.output_dir / f"{args.process}_{i:03d}.h5"
        jobs.append({
            "index": i,
            "total": args.num_files,
            "process": args.process,
            "n_events": args.events_per_file,
            "seed": args.seed_start + i,
            "output": output_path,
            "final_dir": str(args.final_dir) if args.final_dir else None,
            "extra_args": extra,
        })

    print(f"Generating {args.num_files} files, {args.events_per_file} events each")
    print(f"Process: {args.process}")
    print(f"Seeds: {args.seed_start} .. {args.seed_start + args.num_files - 1}")
    print(f"Parallel workers: {args.parallel}")
    print(f"Staging: {args.output_dir}/")
    if args.final_dir:
        print(f"Final:   {args.final_dir}/")
    if extra:
        print(f"Extra args forwarded to truthjets: {extra}")
    print()

    # Run jobs
    t_total = time.time()
    if args.parallel == 1:
        results = [run_job(job) for job in jobs]
    else:
        with Pool(processes=args.parallel) as pool:
            results = pool.map(run_job, jobs)
    total_elapsed = time.time() - t_total

    # Summary
    n_ok = sum(1 for r in results if r["returncode"] == 0)
    n_fail = len(results) - n_ok
    print(f"\nDone: {n_ok}/{len(results)} succeeded in {total_elapsed:.1f}s", end="")
    if n_fail:
        print(f", {n_fail} failed:")
        for r in results:
            if r["returncode"] != 0:
                print(f"  {r['output']} (seed={r['seed']}): {r['stderr']}")
    else:
        print()


if __name__ == "__main__":
    main()
