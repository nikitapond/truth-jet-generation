"""Tests for pileup pool loading and VDS creation."""

from __future__ import annotations

import subprocess
import sys

import awkward as ak
import h5py
import numpy as np

from truthjets.pileup import load_pileup_pool, save_pileup_pool


class TestLoadPileupPoolDirectory:
    def _make_pool(self, n_events, rng):
        """Create a small synthetic pileup pool awkward array."""
        counts = rng.integers(5, 15, size=n_events)
        total = int(counts.sum())
        return ak.zip(
            {
                "px": ak.unflatten(rng.standard_normal(total).astype(np.float32), counts),
                "py": ak.unflatten(rng.standard_normal(total).astype(np.float32), counts),
                "pz": ak.unflatten(rng.standard_normal(total).astype(np.float32), counts),
                "E": ak.unflatten(np.abs(rng.standard_normal(total)).astype(np.float32) + 1, counts),
                "pdgId": ak.unflatten(rng.integers(-300, 300, size=total).astype(np.int32), counts),
            }
        )

    def test_load_from_directory(self, tmp_path):
        """load_pileup_pool on a directory concatenates all *.h5 files."""
        rng = np.random.default_rng(123)
        pool_dir = tmp_path / "chunks"
        pool_dir.mkdir()

        total_events = 0
        for i in range(3):
            n = 10 + i * 5  # 10, 15, 20
            pool = self._make_pool(n, rng)
            save_pileup_pool(pool, pool_dir / f"chunk_{i:03d}.h5")
            total_events += n

        loaded = load_pileup_pool(pool_dir)
        assert len(loaded) == total_events
        for field in ["px", "py", "pz", "E", "pdgId"]:
            assert field in loaded.fields

    def test_load_empty_directory_raises(self, tmp_path):
        """load_pileup_pool on an empty directory should raise."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        try:
            load_pileup_pool(empty_dir)
            assert False, "Should have raised FileNotFoundError"
        except FileNotFoundError:
            pass

    def test_directory_order_is_sorted(self, tmp_path):
        """Files in a directory are loaded in sorted order."""
        rng = np.random.default_rng(999)
        pool_dir = tmp_path / "ordered"
        pool_dir.mkdir()

        # Create files with known sizes in reverse alphabetical order
        sizes = [5, 10, 15]
        names = ["c.h5", "b.h5", "a.h5"]
        for name, n in zip(names, sizes):
            save_pileup_pool(self._make_pool(n, rng), pool_dir / name)

        loaded = load_pileup_pool(pool_dir)
        # sorted order: a.h5(15), b.h5(10), c.h5(5) => total 30
        assert len(loaded) == 30


class TestCreateVdsCLI:
    def _make_pool_file(self, path, n_events, rng):
        """Create a pool file with known data."""
        counts = rng.integers(3, 8, size=n_events)
        total = int(counts.sum())
        pool = ak.zip(
            {
                "px": ak.unflatten(rng.standard_normal(total).astype(np.float32), counts),
                "py": ak.unflatten(rng.standard_normal(total).astype(np.float32), counts),
                "pz": ak.unflatten(rng.standard_normal(total).astype(np.float32), counts),
                "E": ak.unflatten(np.abs(rng.standard_normal(total)).astype(np.float32) + 1, counts),
                "pdgId": ak.unflatten(rng.integers(-300, 300, size=total).astype(np.int32), counts),
            }
        )
        save_pileup_pool(pool, path)
        return n_events, int(counts.sum())

    def test_create_vds_from_files(self, tmp_path):
        """create-vds CLI produces a loadable VDS from pool files."""
        rng = np.random.default_rng(42)
        parts_dir = tmp_path / "parts"
        parts_dir.mkdir()

        files = []
        for i in range(3):
            f = parts_dir / f"part_{i:03d}.h5"
            self._make_pool_file(f, 10, rng)
            files.append(str(f))

        vds_path = tmp_path / "vds.h5"
        result = subprocess.run(
            [sys.executable, "-m", "truthjets.h5utils", *files, "-o", str(vds_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert vds_path.exists()

        # VDS should be loadable and contain concatenated data
        with h5py.File(vds_path, "r") as f:
            assert "n_particles" in f
            assert len(f["n_particles"]) == 30  # 3 files x 10 events
            # px is flattened particles, should equal sum of n_particles
            assert f["px"].shape[0] == int(np.sum(f["n_particles"][:]))

    def test_create_vds_from_directory(self, tmp_path):
        """create-vds CLI accepts a directory as input."""
        rng = np.random.default_rng(77)
        parts_dir = tmp_path / "pool_chunks"
        parts_dir.mkdir()

        for i in range(2):
            self._make_pool_file(parts_dir / f"chunk_{i:03d}.h5", 8, rng)

        vds_path = tmp_path / "vds.h5"
        result = subprocess.run(
            [sys.executable, "-m", "truthjets.h5utils", str(parts_dir), "-o", str(vds_path)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert vds_path.exists()

        with h5py.File(vds_path, "r") as f:
            assert len(f["n_particles"]) == 16  # 2 files x 8 events
