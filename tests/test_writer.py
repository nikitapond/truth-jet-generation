from __future__ import annotations

import os
import tempfile

import awkward as ak
import h5py
import numpy as np

from truthjets.config import JetConfig
from truthjets.writer import CONSTITUENT_DTYPE, JET_DTYPE, HDF5Writer


class TestHDF5Writer:
    def _make_test_data(self, n_jets_per_event=2, n_events=3, n_constit=5):
        """Create minimal test data for writing."""
        # Jet kinematics (events x jets)
        rng = np.random.default_rng(42)
        jet_kin = ak.zip(
            {
                "pt": ak.Array(
                    rng.uniform(20, 200, (n_events, n_jets_per_event)).tolist()
                ),
                "eta": ak.Array(
                    rng.uniform(-2.5, 2.5, (n_events, n_jets_per_event)).tolist()
                ),
                "phi": ak.Array(
                    rng.uniform(-np.pi, np.pi, (n_events, n_jets_per_event)).tolist()
                ),
                "mass": ak.Array(
                    rng.uniform(0, 20, (n_events, n_jets_per_event)).tolist()
                ),
                "energy": ak.Array(
                    rng.uniform(50, 500, (n_events, n_jets_per_event)).tolist()
                ),
            }
        )

        # Labels
        labels = ak.Array(
            rng.choice([0, 4, 5, 15], (n_events, n_jets_per_event)).tolist()
        )

        # Constituents (events x jets x variable constituents)
        # Use list-of-records format matching fastjet output
        constit_list = []
        for i in range(n_events):
            event_constits = []
            for j in range(n_jets_per_event):
                nc = rng.integers(1, n_constit + 1)
                jet_constit = [
                    {
                        "px": float(rng.uniform(-10, 10)),
                        "py": float(rng.uniform(-10, 10)),
                        "pz": float(rng.uniform(-10, 10)),
                        "E": float(rng.uniform(1, 50)),
                        "pdgId": int(rng.choice([211, -211, 321, 22, 11])),
                        "is_pu": False,
                    }
                    for _ in range(nc)
                ]
                event_constits.append(jet_constit)
            constit_list.append(event_constits)

        constituents = ak.Array(constit_list)

        return jet_kin, labels, constituents

    def test_write_and_read_back(self):
        """Write data and verify shapes and dtypes on read-back."""
        jet_config = JetConfig(max_constituents=10)
        jet_kin, labels, constituents = self._make_test_data()

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        try:
            with HDF5Writer(path, jet_config) as writer:
                writer.write_batch(
                    jet_kin, labels, constituents, jet_kin.eta, jet_kin.phi
                )

            n_total_jets = int(ak.sum(ak.num(jet_kin.pt)))

            with h5py.File(path, "r") as f:
                assert "jets" in f
                assert "constituents" in f

                jets_ds = f["jets"]
                constit_ds = f["constituents"]

                assert jets_ds.shape == (n_total_jets,)
                assert jets_ds.dtype == JET_DTYPE
                assert constit_ds.shape == (n_total_jets, 10)
                assert constit_ds.dtype == CONSTITUENT_DTYPE

                # Check jet values are reasonable
                assert np.all(jets_ds["pt"] > 0)
                assert np.all(np.abs(jets_ds["eta"]) < 5.0)
                assert np.all(jets_ds["n_constituents"] > 0)

                # Check constituent valid flags
                for i in range(n_total_jets):
                    nc = jets_ds["n_constituents"][i]
                    assert np.all(constit_ds["valid"][i, :nc])
                    if nc < 10:
                        assert not np.any(constit_ds["valid"][i, nc:])
        finally:
            os.unlink(path)

    def test_streaming_multiple_batches(self):
        """Multiple write_batch calls should accumulate."""
        jet_config = JetConfig(max_constituents=10)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        try:
            with HDF5Writer(path, jet_config) as writer:
                for _ in range(3):
                    jet_kin, labels, constits = self._make_test_data(
                        n_jets_per_event=2, n_events=2
                    )
                    writer.write_batch(
                        jet_kin, labels, constits, jet_kin.eta, jet_kin.phi
                    )
                assert writer.n_jets == 12  # 3 batches * 2 events * 2 jets

            with h5py.File(path, "r") as f:
                assert f["jets"].shape[0] == 12
                assert f["constituents"].shape[0] == 12
        finally:
            os.unlink(path)

    def test_empty_batch(self):
        """Writing an empty batch should be a no-op."""
        jet_config = JetConfig(max_constituents=10)

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        try:
            with HDF5Writer(path, jet_config) as writer:
                # Empty arrays
                jet_kin = ak.zip(
                    {
                        "pt": ak.Array([[]]),
                        "eta": ak.Array([[]]),
                        "phi": ak.Array([[]]),
                        "mass": ak.Array([[]]),
                        "energy": ak.Array([[]]),
                    }
                )
                labels = ak.Array([[]])
                constits = ak.Array([[]])
                writer.write_batch(
                    jet_kin, labels, constits, jet_kin.eta, jet_kin.phi
                )
                assert writer.n_jets == 0
        finally:
            os.unlink(path)

    def test_pileup_fields_exist(self):
        """Output should contain is_pu and pt_frac_pu fields."""
        jet_config = JetConfig(max_constituents=10)
        jet_kin, labels, constituents = self._make_test_data()

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        try:
            with HDF5Writer(path, jet_config) as writer:
                writer.write_batch(
                    jet_kin, labels, constituents, jet_kin.eta, jet_kin.phi
                )

            with h5py.File(path, "r") as f:
                # pt_frac_pu in jets
                assert "pt_frac_pu" in f["jets"].dtype.names
                # All zero since no PU constituents
                assert np.all(f["jets"]["pt_frac_pu"][:] == 0.0)

                # is_pu in constituents
                assert "is_pu" in f["constituents"].dtype.names
                # All False since no PU constituents
                assert not np.any(f["constituents"]["is_pu"][:])
        finally:
            os.unlink(path)

    def test_pt_frac_pu_computed(self):
        """pt_frac_pu should be nonzero when PU constituents are present."""
        jet_config = JetConfig(max_constituents=10)
        jet_kin = ak.zip(
            {
                "pt": ak.Array([[50.0]]),
                "eta": ak.Array([[0.0]]),
                "phi": ak.Array([[0.0]]),
                "mass": ak.Array([[5.0]]),
                "energy": ak.Array([[50.0]]),
            }
        )
        labels = ak.Array([[0]])
        # 2 HS + 1 PU constituent
        constituents = ak.Array(
            [
                [
                    [
                        {"px": 30.0, "py": 0.0, "pz": 0.0, "E": 30.0, "pdgId": 211, "is_pu": False},
                        {"px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "pdgId": 211, "is_pu": False},
                        {"px": 10.0, "py": 0.0, "pz": 0.0, "E": 10.0, "pdgId": 211, "is_pu": True},
                    ]
                ]
            ]
        )

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        try:
            with HDF5Writer(path, jet_config) as writer:
                writer.write_batch(
                    jet_kin, labels, constituents, jet_kin.eta, jet_kin.phi
                )

            with h5py.File(path, "r") as f:
                pt_frac = f["jets"]["pt_frac_pu"][0]
                # PU pT = 10, total pT = 50, fraction = 0.2
                np.testing.assert_allclose(pt_frac, 0.2, atol=0.01)

                is_pu = f["constituents"]["is_pu"][0]
                valid = f["constituents"]["valid"][0]
                # Sorted by pT desc: 30(HS), 10(HS), 10(PU)
                assert np.sum(is_pu & valid) == 1
        finally:
            os.unlink(path)

    def test_constituent_pt_sorting(self):
        """Constituents should be sorted by pT descending."""
        jet_config = JetConfig(max_constituents=10)

        # One event, one jet, 3 constituents with known pT ordering
        jet_kin = ak.zip(
            {
                "pt": ak.Array([[50.0]]),
                "eta": ak.Array([[0.0]]),
                "phi": ak.Array([[0.0]]),
                "mass": ak.Array([[5.0]]),
                "energy": ak.Array([[50.0]]),
            }
        )
        labels = ak.Array([[0]])
        # Constituents with px values 5, 30, 15 -> pT = 5, 30, 15
        constituents = ak.Array(
            [
                [
                    [
                        {"px": 5.0, "py": 0.0, "pz": 0.0, "E": 5.0, "pdgId": 211, "is_pu": False},
                        {"px": 30.0, "py": 0.0, "pz": 0.0, "E": 30.0, "pdgId": 211, "is_pu": False},
                        {"px": 15.0, "py": 0.0, "pz": 0.0, "E": 15.0, "pdgId": 211, "is_pu": False},
                    ]
                ]
            ]
        )

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        try:
            with HDF5Writer(path, jet_config) as writer:
                writer.write_batch(
                    jet_kin, labels, constituents, jet_kin.eta, jet_kin.phi
                )

            with h5py.File(path, "r") as f:
                pt_vals = f["constituents"]["pt"][0]
                valid = f["constituents"]["valid"][0]
                valid_pts = pt_vals[valid]
                # Should be sorted descending: 30, 15, 5
                assert valid_pts[0] == 30.0
                assert valid_pts[1] == 15.0
                assert valid_pts[2] == 5.0
        finally:
            os.unlink(path)
