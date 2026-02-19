from __future__ import annotations

import awkward as ak
import numpy as np

from truthjets.label import (
    _delta_r,
    is_b_hadron,
    is_c_hadron,
    is_tau_lepton,
    label_jets,
)


class TestPDGClassification:
    """Test PDG ID classification functions."""

    def test_b_mesons(self):
        # B+, B0, Bs, Bc
        b_mesons = np.array([521, 511, 531, 541])
        assert np.all(is_b_hadron(b_mesons))

    def test_b_baryons(self):
        # Lambda_b, Sigma_b, Xi_b
        b_baryons = np.array([5122, 5112, 5232])
        assert np.all(is_b_hadron(b_baryons))

    def test_negative_b_hadrons(self):
        # Anti-B mesons
        assert np.all(is_b_hadron(np.array([-521, -511])))

    def test_not_b_hadrons(self):
        # Pion, kaon, proton, D meson
        not_b = np.array([211, 321, 2212, 411])
        assert not np.any(is_b_hadron(not_b))

    def test_c_mesons(self):
        # D+, D0, Ds
        c_mesons = np.array([411, 421, 431])
        assert np.all(is_c_hadron(c_mesons))

    def test_c_baryons(self):
        # Lambda_c
        c_baryons = np.array([4122])
        assert np.all(is_c_hadron(c_baryons))

    def test_c_hadron_excludes_b(self):
        # Bc meson has both b and c, should NOT be classified as c-hadron
        assert not is_c_hadron(np.array([541]))[0]

    def test_not_c_hadrons(self):
        not_c = np.array([211, 321, 2212])
        assert not np.any(is_c_hadron(not_c))

    def test_tau_lepton(self):
        assert is_tau_lepton(np.array([15]))[0]
        assert is_tau_lepton(np.array([-15]))[0]

    def test_not_tau(self):
        assert not is_tau_lepton(np.array([13]))[0]  # muon
        assert not is_tau_lepton(np.array([11]))[0]  # electron


class TestDeltaR:
    def test_same_point(self):
        dr = _delta_r(np.array([1.0]), np.array([1.0]), np.array([1.0]), np.array([1.0]))
        np.testing.assert_allclose(dr, 0.0, atol=1e-10)

    def test_known_distance(self):
        dr = _delta_r(np.array([0.0]), np.array([0.0]), np.array([0.3]), np.array([0.4]))
        np.testing.assert_allclose(dr, 0.5, atol=1e-10)

    def test_phi_wrapping(self):
        # Points near +pi and -pi should be close
        dr = _delta_r(np.array([0.0]), np.array([3.1]), np.array([0.0]), np.array([-3.1]))
        assert dr[0] < 0.1  # Should be ~0.083, not ~6.2


class TestLabelJets:
    def _make_mock_events(self, particle_ids, particle_eta, particle_phi):
        """Create a minimal mock events structure."""
        # Single event with given particles
        px = np.cos(particle_phi) * 10.0  # arbitrary pT=10
        py = np.sin(particle_phi) * 10.0
        theta = 2 * np.arctan(np.exp(-np.array(particle_eta)))
        pz = 10.0 / np.tan(theta)
        E = np.sqrt(px**2 + py**2 + pz**2 + 0.14**2)

        prt = ak.zip(
            {
                "p": ak.zip({"px": [px], "py": [py], "pz": [pz], "e": [E]}),
                "id": [particle_ids],
                "status": [np.ones(len(particle_ids), dtype=np.int32)],
            }
        )
        return ak.zip({"prt": prt})

    def test_b_jet_label(self):
        # B meson at eta=0.5, phi=0.5
        events = self._make_mock_events([521, 211, 211], [0.5, 1.0, -1.0], [0.5, 1.0, -1.0])
        jet_eta = ak.Array([[0.5]])
        jet_phi = ak.Array([[0.5]])
        labels = label_jets(events, jet_eta, jet_phi, R=0.4)
        assert ak.to_numpy(ak.flatten(labels))[0] == 5

    def test_c_jet_label(self):
        # D meson at eta=0.5, phi=0.5
        events = self._make_mock_events([411, 211], [0.5, -1.0], [0.5, -1.0])
        jet_eta = ak.Array([[0.5]])
        jet_phi = ak.Array([[0.5]])
        labels = label_jets(events, jet_eta, jet_phi, R=0.4)
        assert ak.to_numpy(ak.flatten(labels))[0] == 4

    def test_light_jet_label(self):
        # Only pions, far from jet
        events = self._make_mock_events([211, 211], [2.0, -2.0], [2.0, -2.0])
        jet_eta = ak.Array([[0.0]])
        jet_phi = ak.Array([[0.0]])
        labels = label_jets(events, jet_eta, jet_phi, R=0.4)
        assert ak.to_numpy(ak.flatten(labels))[0] == 0

    def test_b_over_c_priority(self):
        # Both B and D meson near jet -> should be labeled b
        events = self._make_mock_events([521, 411], [0.5, 0.5], [0.5, 0.5])
        jet_eta = ak.Array([[0.5]])
        jet_phi = ak.Array([[0.5]])
        labels = label_jets(events, jet_eta, jet_phi, R=0.4)
        assert ak.to_numpy(ak.flatten(labels))[0] == 5

    def test_tau_label(self):
        events = self._make_mock_events([15], [0.5], [0.5])
        jet_eta = ak.Array([[0.5]])
        jet_phi = ak.Array([[0.5]])
        labels = label_jets(events, jet_eta, jet_phi, R=0.4)
        assert ak.to_numpy(ak.flatten(labels))[0] == 15

    def test_b_over_tau_priority(self):
        # B hadron and tau near jet -> b wins
        events = self._make_mock_events([521, 15], [0.5, 0.5], [0.5, 0.5])
        jet_eta = ak.Array([[0.5]])
        jet_phi = ak.Array([[0.5]])
        labels = label_jets(events, jet_eta, jet_phi, R=0.4)
        assert ak.to_numpy(ak.flatten(labels))[0] == 5
