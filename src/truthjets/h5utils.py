"""Shared HDF5 utilities for truthjets."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py

# Standard compression kwargs for all HDF5 dataset creation
H5_COMPRESSION = {"compression": "gzip", "compression_opts": 7, "shuffle": True}


def create_vds(part_files: list[Path], output_path: Path) -> None:
    """Create an HDF5 Virtual Dataset that concatenates all part files.

    Discovers datasets from the first part file, so it works with any
    modules/schemas. Uses relative paths so the output directory is portable.
    """
    # Discover datasets from the first file
    with h5py.File(part_files[0], "r") as f:
        dataset_info = {}
        for name in f:
            ds = f[name]
            dataset_info[name] = {
                "dtype": ds.dtype,
                "shape_suffix": ds.shape[1:],
            }

    # Collect per-dataset axis-0 sizes from each part
    # (datasets may have different axis-0 sizes, e.g. pool files)
    ds_sizes = {name: [] for name in dataset_info}
    for pf in part_files:
        with h5py.File(pf, "r") as f:
            for name in dataset_info:
                ds_sizes[name].append(f[name].shape[0])

    # Build virtual layouts and write
    rel_parts = [pf.relative_to(output_path.parent) for pf in part_files]

    with h5py.File(output_path, "w") as out:
        for ds_name, info in dataset_info.items():
            sizes = ds_sizes[ds_name]
            total = sum(sizes)
            full_shape = (total, *info["shape_suffix"])
            layout = h5py.VirtualLayout(shape=full_shape, dtype=info["dtype"])

            offset = 0
            for rel_path, size in zip(rel_parts, sizes):
                src = h5py.VirtualSource(
                    str(rel_path),
                    ds_name,
                    shape=(size, *info["shape_suffix"]),
                    dtype=info["dtype"],
                )
                layout[offset : offset + size] = src
                offset += size

            out.create_virtual_dataset(ds_name, layout)

    # Report using the first dataset's total for the summary
    first_ds = next(iter(dataset_info))
    total_entries = sum(ds_sizes[first_ds])

    print(f"Created VDS: {output_path} ({len(part_files)} files, {total_entries} total entries)")


def main():
    """CLI entry point for creating HDF5 Virtual Datasets."""
    parser = argparse.ArgumentParser(description="Create an HDF5 Virtual Dataset concatenating part files")
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Input HDF5 files or a single directory containing *.h5 files",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output VDS file path",
    )
    args = parser.parse_args()

    # Resolve input files
    inputs = [Path(p) for p in args.inputs]
    if len(inputs) == 1 and inputs[0].is_dir():
        part_files = sorted(inputs[0].glob("*.h5"))
        if not part_files:
            print(f"No .h5 files found in {inputs[0]}", file=sys.stderr)
            sys.exit(1)
    else:
        part_files = inputs
        for pf in part_files:
            if not pf.exists():
                print(f"File not found: {pf}", file=sys.stderr)
                sys.exit(1)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    create_vds(part_files, args.output)


if __name__ == "__main__":
    main()
