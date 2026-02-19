"""End-to-end integration test: generate events, cluster, label, write."""

from __future__ import annotations

import os
import tempfile

import h5py
import numpy as np
import pytest

from truthjets.cluster import cluster_jets
from truthjets.config import JetConfig, PythiaConfig, resolve_card
from truthjets.generate import generate_events, generate_pileup_batch, init_pileup_pythia, init_pythia
from truthjets.label import label_jets
from truthjets.pileup import overlay_pileup, sample_n_pileup
from truthjets.writer import HDF5Writer


@pytest.fixture(scope="module")
def output_file():
    """Run a small end-to-end generation and return the HDF5 path."""
    pythia_config = PythiaConfig(
        pythia_card=resolve_card("ttbar"),
        ecm=13600.0,
        seed=123,
    )
    jet_config = JetConfig(
        R=0.4,
        pt_min=20.0,
        eta_max=2.5,
        max_constituents=40,
    )

    pythia = init_pythia(pythia_config)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name

    with HDF5Writer(path, jet_config) as writer:
        for events in generate_events(pythia, n_events=100, batch_size=50):
            jets, constits, jet_kin = cluster_jets(events, jet_config)
            labels = label_jets(events, jet_kin.eta, jet_kin.phi, jet_config.R)
            writer.write_batch(jet_kin, labels, constits, jet_kin.eta, jet_kin.phi)

    yield path
    os.unlink(path)


class TestIntegration:
    def test_output_has_jets(self, output_file):
        with h5py.File(output_file, "r") as f:
            n_jets = f["jets"].shape[0]
            assert n_jets > 0, "Should have produced at least some jets"

    def test_jet_pt_above_cut(self, output_file):
        with h5py.File(output_file, "r") as f:
            pts = f["jets"]["pt"][:]
            assert np.all(pts >= 20.0), "All jets should be above pT cut"

    def test_jet_eta_within_cut(self, output_file):
        with h5py.File(output_file, "r") as f:
            etas = f["jets"]["eta"][:]
            assert np.all(np.abs(etas) < 2.5), "All jets should be within eta cut"

    def test_valid_labels(self, output_file):
        with h5py.File(output_file, "r") as f:
            labels = f["jets"]["HadronConeExclTruthLabelID"][:]
            valid_labels = {0, 4, 5, 15}
            unique = set(np.unique(labels))
            assert unique.issubset(valid_labels), f"Unexpected labels: {unique}"

    def test_ttbar_has_b_jets(self, output_file):
        """ttbar events should produce b-jets."""
        with h5py.File(output_file, "r") as f:
            labels = f["jets"]["HadronConeExclTruthLabelID"][:]
            assert 5 in labels, "ttbar should produce b-jets"

    def test_constituent_shapes(self, output_file):
        with h5py.File(output_file, "r") as f:
            n_jets = f["jets"].shape[0]
            assert f["constituents"].shape == (n_jets, 40)

    def test_constituent_padding(self, output_file):
        with h5py.File(output_file, "r") as f:
            for i in range(min(10, f["jets"].shape[0])):
                nc = f["jets"]["n_constituents"][i]
                valid = f["constituents"]["valid"][i]
                assert np.sum(valid) == nc

    def test_constituent_deta_dphi_relative(self, output_file):
        """deta and dphi should be small (relative to jet axis)."""
        with h5py.File(output_file, "r") as f:
            deta = f["constituents"]["deta"][:]
            dphi = f["constituents"]["dphi"][:]
            valid = f["constituents"]["valid"][:]

            # Valid constituents should have reasonable deta/dphi values
            # (can exceed jet R due to clustering merging, but should be bounded)
            assert np.all(np.abs(deta[valid]) < 5.0)
            assert np.all(np.abs(dphi[valid]) < np.pi + 0.01)

    def test_n_constituents_positive(self, output_file):
        with h5py.File(output_file, "r") as f:
            nc = f["jets"]["n_constituents"][:]
            assert np.all(nc > 0), "All jets should have at least one constituent"

    def test_jet_mass_nonnegative(self, output_file):
        with h5py.File(output_file, "r") as f:
            mass = f["jets"]["mass"][:]
            assert np.all(mass >= 0), "Jet mass should be non-negative"

    def test_pileup_fields_present(self, output_file):
        """Output should always have pt_frac_pu and is_pu (even without PU)."""
        with h5py.File(output_file, "r") as f:
            assert "pt_frac_pu" in f["jets"].dtype.names
            assert "is_pu" in f["constituents"].dtype.names
            # Without PU, all should be zero/False
            assert np.all(f["jets"]["pt_frac_pu"][:] == 0.0)
            assert not np.any(f["constituents"]["is_pu"][:])


@pytest.fixture(scope="module")
def output_file_pu():
    """Run an end-to-end generation with pileup and return the HDF5 path."""
    pythia_config = PythiaConfig(
        pythia_card=resolve_card("ttbar"),
        ecm=13600.0,
        seed=456,
        mu=20.0,
    )
    jet_config = JetConfig(
        R=0.4,
        pt_min=20.0,
        eta_max=2.5,
        max_constituents=40,
    )

    pythia = init_pythia(pythia_config)
    pythia_pu = init_pileup_pythia(pythia_config)
    pu_rng = np.random.default_rng(pythia_config.seed + 100)

    with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
        path = f.name

    with HDF5Writer(path, jet_config) as writer:
        for events in generate_events(pythia, n_events=100, batch_size=50):
            n_events_in_batch = len(events.prt)
            n_pu = sample_n_pileup(pythia_config.mu, n_events_in_batch, pu_rng)
            total_pu = int(np.sum(n_pu))
            pu_events = generate_pileup_batch(pythia_pu, total_pu)
            merged = overlay_pileup(events, pu_events, n_pu)

            jets, constits, jet_kin = cluster_jets(events, jet_config, particles=merged)
            labels = label_jets(events, jet_kin.eta, jet_kin.phi, jet_config.R)
            writer.write_batch(jet_kin, labels, constits, jet_kin.eta, jet_kin.phi)

    yield path
    os.unlink(path)


class TestIntegrationPileup:
    def test_output_has_jets(self, output_file_pu):
        with h5py.File(output_file_pu, "r") as f:
            assert f["jets"].shape[0] > 0

    def test_some_pu_constituents(self, output_file_pu):
        """With mu=20, some constituents should be marked as PU."""
        with h5py.File(output_file_pu, "r") as f:
            is_pu = f["constituents"]["is_pu"][:]
            valid = f["constituents"]["valid"][:]
            assert np.sum(is_pu & valid) > 0, "Should have PU constituents"

    def test_pt_frac_pu_nonzero_for_some(self, output_file_pu):
        """Some jets should have nonzero pileup fraction."""
        with h5py.File(output_file_pu, "r") as f:
            pt_frac = f["jets"]["pt_frac_pu"][:]
            assert np.any(pt_frac > 0), "Some jets should have PU contamination"

    def test_labels_still_valid(self, output_file_pu):
        """Labels should still be in the valid set (PU should not affect labeling)."""
        with h5py.File(output_file_pu, "r") as f:
            labels = f["jets"]["HadronConeExclTruthLabelID"][:]
            valid_labels = {0, 4, 5, 15}
            unique = set(np.unique(labels))
            assert unique.issubset(valid_labels)

    def test_ttbar_has_b_jets_with_pu(self, output_file_pu):
        """ttbar should still produce b-jets even with PU overlay."""
        with h5py.File(output_file_pu, "r") as f:
            labels = f["jets"]["HadronConeExclTruthLabelID"][:]
            assert 5 in labels
