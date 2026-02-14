"""Closure test: inline PU vs pool-file PU produce equivalent jet distributions."""
from __future__ import annotations

import subprocess
import sys

import h5py
import numpy as np
import pytest
from scipy.stats import ks_2samp


def run_cli(args: list[str], timeout: int = 300) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, "-m", "truthjets.cli", *args],
        capture_output=True, text=True, timeout=timeout,
    )
    assert result.returncode == 0, f"CLI failed: {result.stderr}"
    return result


def run_pu_pool(args: list[str], timeout: int = 120) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, "-m", "truthjets.generate_pu_pool", *args],
        capture_output=True, text=True, timeout=timeout,
    )
    assert result.returncode == 0, f"generate-pu-pool failed: {result.stderr}"
    return result


@pytest.fixture(scope="module")
def closure_outputs(tmp_path_factory):
    """Generate outputs from both PU methods once for the whole module."""
    tmp = tmp_path_factory.mktemp("pu_closure")
    method1 = tmp / "method1.h5"
    method2 = tmp / "method2.h5"
    pool = tmp / "pool.h5"

    # Method 1: inline PU (pre-generate pool on the fly)
    run_cli([
        "--process", "ttbar", "-n", "500",
        "--pu", "20", "--pu-pre-gen", "5000", "--softkiller",
        "--seed", "42", "-o", str(method1),
    ])

    # Method 2: separate pool file
    run_pu_pool(["-n", "5000", "-o", str(pool), "--seed", "99"])
    run_cli([
        "--process", "ttbar", "-n", "500",
        "--pu", "20", "--pu-file", str(pool), "--softkiller",
        "--seed", "42", "-o", str(method2),
    ])

    with h5py.File(method1, "r") as f1, h5py.File(method2, "r") as f2:
        data = {
            "pt_1": f1["jets"]["pt"][:],
            "eta_1": f1["jets"]["eta"][:],
            "njets_1": f1["jets"]["pt"].shape[0],
            "pt_2": f2["jets"]["pt"][:],
            "eta_2": f2["jets"]["eta"][:],
            "njets_2": f2["jets"]["pt"].shape[0],
        }

    # Compute jets-per-event from the outputs
    # Each method processes the same number of events (500)
    # We can approximate multiplicity from total jets
    return data


class TestPUClosure:
    """Both PU methods should yield statistically equivalent jet distributions."""

    def test_both_produce_jets(self, closure_outputs):
        assert closure_outputs["njets_1"] > 0
        assert closure_outputs["njets_2"] > 0

    def test_jet_count_ratio(self, closure_outputs):
        ratio = closure_outputs["njets_1"] / closure_outputs["njets_2"]
        assert 0.8 < ratio < 1.2, f"Jet count ratio {ratio:.3f} outside [0.8, 1.2]"

    def test_pt_distribution(self, closure_outputs):
        stat, pval = ks_2samp(closure_outputs["pt_1"], closure_outputs["pt_2"])
        assert pval > 0.05, f"pT KS p-value {pval:.4f} < 0.05 (stat={stat:.4f})"

    def test_eta_distribution(self, closure_outputs):
        stat, pval = ks_2samp(closure_outputs["eta_1"], closure_outputs["eta_2"])
        assert pval > 0.05, f"eta KS p-value {pval:.4f} < 0.05 (stat={stat:.4f})"
