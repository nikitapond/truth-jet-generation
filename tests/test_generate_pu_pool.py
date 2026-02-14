"""Tests for the generate-pu-pool CLI entry point."""
from __future__ import annotations

import subprocess
import sys

import h5py
import numpy as np


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
