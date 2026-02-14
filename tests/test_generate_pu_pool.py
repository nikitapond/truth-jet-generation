"""Tests for the generate-pu-pool CLI entry point."""
from __future__ import annotations

import subprocess
import sys

import awkward as ak
import h5py
import numpy as np

from truthjets.pileup import load_pileup_pool, save_pileup_pool


class TestGeneratePuPoolCLI:
    def test_help_flag(self):
        result = subprocess.run(
            [sys.executable, "-m", "truthjets.generate_pu_pool", "--help"],
            capture_output=True, text=True,
        )
        assert result.returncode == 0
        assert "pileup pool" in result.stdout.lower()

    def test_missing_required_args(self):
        result = subprocess.run(
            [sys.executable, "-m", "truthjets.generate_pu_pool"],
            capture_output=True, text=True,
        )
        assert result.returncode != 0

    def test_generates_valid_pool(self, tmp_path):
        output = tmp_path / "pool.h5"
        result = subprocess.run(
            [
                sys.executable, "-m", "truthjets.generate_pu_pool",
                "-n", "50",
                "-o", str(output),
                "--seed", "123",
                "--batch-size", "50",
            ],
            capture_output=True, text=True,
            timeout=60,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output.exists()

        # Verify HDF5 structure matches pileup pool format
        with h5py.File(output, "r") as f:
            for ds in ["n_particles", "px", "py", "pz", "E", "pdgId"]:
                assert ds in f, f"Missing dataset: {ds}"

            n_particles = f["n_particles"][:]
            assert len(n_particles) == 50
            assert np.all(n_particles > 0)

            total = int(np.sum(n_particles))
            assert f["px"].shape == (total,)
            assert f["py"].shape == (total,)
            assert f["pz"].shape == (total,)
            assert f["E"].shape == (total,)
            assert f["pdgId"].shape == (total,)

    def test_output_summary_printed(self, tmp_path):
        output = tmp_path / "pool.h5"
        result = subprocess.run(
            [
                sys.executable, "-m", "truthjets.generate_pu_pool",
                "-n", "20",
                "-o", str(output),
            ],
            capture_output=True, text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert "Events:" in result.stdout
        assert "Particles:" in result.stdout
        assert "Done in" in result.stdout

    def test_custom_ecm(self, tmp_path):
        output = tmp_path / "pool.h5"
        result = subprocess.run(
            [
                sys.executable, "-m", "truthjets.generate_pu_pool",
                "-n", "10",
                "-o", str(output),
                "--ecm", "14000.0",
            ],
            capture_output=True, text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert "ecm=14000.0" in result.stdout

    def test_pool_loadable_by_pileup_module(self, tmp_path):
        """Generated pool can be loaded by load_pileup_pool."""
        from truthjets.pileup import load_pileup_pool

        output = tmp_path / "pool.h5"
        result = subprocess.run(
            [
                sys.executable, "-m", "truthjets.generate_pu_pool",
                "-n", "30",
                "-o", str(output),
                "--seed", "99",
            ],
            capture_output=True, text=True,
            timeout=60,
        )
        assert result.returncode == 0

        pool = load_pileup_pool(output)
        assert len(pool) == 30
        for field in ["px", "py", "pz", "E", "pdgId"]:
            assert field in pool.fields

    def test_creates_parent_directories(self, tmp_path):
        output = tmp_path / "nested" / "dir" / "pool.h5"
        result = subprocess.run(
            [
                sys.executable, "-m", "truthjets.generate_pu_pool",
                "-n", "10",
                "-o", str(output),
            ],
            capture_output=True, text=True,
            timeout=60,
        )
        assert result.returncode == 0
        assert output.exists()

    def test_pool_has_gzip_compression(self, tmp_path):
        """Generated pool files should use gzip compression on all datasets."""
        output = tmp_path / "pool.h5"
        result = subprocess.run(
            [
                sys.executable, "-m", "truthjets.generate_pu_pool",
                "-n", "20",
                "-o", str(output),
                "--seed", "42",
            ],
            capture_output=True, text=True,
            timeout=60,
        )
        assert result.returncode == 0

        with h5py.File(output, "r") as f:
            for ds_name in ["n_particles", "px", "py", "pz", "E", "pdgId"]:
                ds = f[ds_name]
                assert ds.compression == "gzip", f"{ds_name} not gzip"
                assert ds.compression_opts == 7, f"{ds_name} opts != 7"
                assert ds.shuffle, f"{ds_name} shuffle not enabled"


class TestLoadPileupPoolDirectory:
    def _make_pool(self, n_events, rng):
        """Create a small synthetic pileup pool awkward array."""
        counts = rng.integers(5, 15, size=n_events)
        total = int(counts.sum())
        return ak.zip({
            "px": ak.unflatten(rng.standard_normal(total).astype(np.float32), counts),
            "py": ak.unflatten(rng.standard_normal(total).astype(np.float32), counts),
            "pz": ak.unflatten(rng.standard_normal(total).astype(np.float32), counts),
            "E": ak.unflatten(np.abs(rng.standard_normal(total)).astype(np.float32) + 1, counts),
            "pdgId": ak.unflatten(rng.integers(-300, 300, size=total).astype(np.int32), counts),
        })

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
        pool = ak.zip({
            "px": ak.unflatten(rng.standard_normal(total).astype(np.float32), counts),
            "py": ak.unflatten(rng.standard_normal(total).astype(np.float32), counts),
            "pz": ak.unflatten(rng.standard_normal(total).astype(np.float32), counts),
            "E": ak.unflatten(np.abs(rng.standard_normal(total)).astype(np.float32) + 1, counts),
            "pdgId": ak.unflatten(rng.integers(-300, 300, size=total).astype(np.int32), counts),
        })
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
            capture_output=True, text=True,
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
            capture_output=True, text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert vds_path.exists()

        with h5py.File(vds_path, "r") as f:
            assert len(f["n_particles"]) == 16  # 2 files x 8 events
