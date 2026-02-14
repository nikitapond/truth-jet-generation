from __future__ import annotations

import awkward as ak
import numpy as np

from truthjets.modules.label import label_large_r_jets


class TestLargeRLabelJets:
    def _make_mock_events(self, particle_ids, particle_eta, particle_phi):
        """Create a minimal mock events structure with given particles."""
        px = np.cos(particle_phi) * 100.0  # high pT for boosted regime
        py = np.sin(particle_phi) * 100.0
        theta = 2 * np.arctan(np.exp(-np.array(particle_eta)))
        pz = 100.0 / np.tan(theta)
        E = np.sqrt(px**2 + py**2 + pz**2 + 80.0**2)

        prt = ak.zip(
            {
                "p": ak.zip({"px": [px], "py": [py], "pz": [pz], "e": [E]}),
                "id": [particle_ids],
                "status": [np.ones(len(particle_ids), dtype=np.int32)],
            }
        )
        return ak.zip({"prt": prt})

    # ---------------------------------------------------------------
    # Individual resonance labels
    # ---------------------------------------------------------------

    def test_top_label(self):
        events = self._make_mock_events([6, 211], [0.5, -2.0], [0.5, -2.0])
        jet_eta = ak.Array([[0.5]])
        jet_phi = ak.Array([[0.5]])
        labels = label_large_r_jets(events, jet_eta, jet_phi, R=1.0)
        assert ak.to_numpy(ak.flatten(labels))[0] == 6

    def test_higgs_label(self):
        events = self._make_mock_events([25, 211], [0.5, -2.0], [0.5, -2.0])
        jet_eta = ak.Array([[0.5]])
        jet_phi = ak.Array([[0.5]])
        labels = label_large_r_jets(events, jet_eta, jet_phi, R=1.0)
        assert ak.to_numpy(ak.flatten(labels))[0] == 25

    def test_z_label(self):
        events = self._make_mock_events([23, 211], [0.5, -2.0], [0.5, -2.0])
        jet_eta = ak.Array([[0.5]])
        jet_phi = ak.Array([[0.5]])
        labels = label_large_r_jets(events, jet_eta, jet_phi, R=1.0)
        assert ak.to_numpy(ak.flatten(labels))[0] == 23

    def test_w_label(self):
        events = self._make_mock_events([24, 211], [0.5, -2.0], [0.5, -2.0])
        jet_eta = ak.Array([[0.5]])
        jet_phi = ak.Array([[0.5]])
        labels = label_large_r_jets(events, jet_eta, jet_phi, R=1.0)
        assert ak.to_numpy(ak.flatten(labels))[0] == 24

    # ---------------------------------------------------------------
    # QCD (no match)
    # ---------------------------------------------------------------

    def test_qcd_no_resonances(self):
        """Only light particles, no resonances at all -> QCD."""
        events = self._make_mock_events([211, 321], [0.5, -0.5], [0.5, -0.5])
        jet_eta = ak.Array([[0.0]])
        jet_phi = ak.Array([[0.0]])
        labels = label_large_r_jets(events, jet_eta, jet_phi, R=1.0)
        assert ak.to_numpy(ak.flatten(labels))[0] == 0

    def test_qcd_resonance_far_away(self):
        """Top quark exists but is far from the jet -> QCD."""
        events = self._make_mock_events([6], [2.5], [2.5])
        jet_eta = ak.Array([[0.0]])
        jet_phi = ak.Array([[0.0]])
        labels = label_large_r_jets(events, jet_eta, jet_phi, R=1.0)
        assert ak.to_numpy(ak.flatten(labels))[0] == 0

    # ---------------------------------------------------------------
    # Priority tests (last write wins: W -> Z -> H -> top)
    # ---------------------------------------------------------------

    def test_top_over_higgs(self):
        events = self._make_mock_events([6, 25], [0.5, 0.5], [0.5, 0.5])
        jet_eta = ak.Array([[0.5]])
        jet_phi = ak.Array([[0.5]])
        labels = label_large_r_jets(events, jet_eta, jet_phi, R=1.0)
        assert ak.to_numpy(ak.flatten(labels))[0] == 6

    def test_higgs_over_z(self):
        events = self._make_mock_events([25, 23], [0.5, 0.5], [0.5, 0.5])
        jet_eta = ak.Array([[0.5]])
        jet_phi = ak.Array([[0.5]])
        labels = label_large_r_jets(events, jet_eta, jet_phi, R=1.0)
        assert ak.to_numpy(ak.flatten(labels))[0] == 25

    def test_z_over_w(self):
        events = self._make_mock_events([23, 24], [0.5, 0.5], [0.5, 0.5])
        jet_eta = ak.Array([[0.5]])
        jet_phi = ak.Array([[0.5]])
        labels = label_large_r_jets(events, jet_eta, jet_phi, R=1.0)
        assert ak.to_numpy(ak.flatten(labels))[0] == 23

    def test_top_over_w(self):
        events = self._make_mock_events([6, 24], [0.5, 0.5], [0.5, 0.5])
        jet_eta = ak.Array([[0.5]])
        jet_phi = ak.Array([[0.5]])
        labels = label_large_r_jets(events, jet_eta, jet_phi, R=1.0)
        assert ak.to_numpy(ak.flatten(labels))[0] == 6

    # ---------------------------------------------------------------
    # Anti-particles
    # ---------------------------------------------------------------

    def test_anti_top(self):
        events = self._make_mock_events([-6], [0.5], [0.5])
        jet_eta = ak.Array([[0.5]])
        jet_phi = ak.Array([[0.5]])
        labels = label_large_r_jets(events, jet_eta, jet_phi, R=1.0)
        assert ak.to_numpy(ak.flatten(labels))[0] == 6

    def test_anti_w(self):
        events = self._make_mock_events([-24], [0.5], [0.5])
        jet_eta = ak.Array([[0.5]])
        jet_phi = ak.Array([[0.5]])
        labels = label_large_r_jets(events, jet_eta, jet_phi, R=1.0)
        assert ak.to_numpy(ak.flatten(labels))[0] == 24

    # ---------------------------------------------------------------
    # Multiple jets
    # ---------------------------------------------------------------

    def test_multiple_jets_different_labels(self):
        """Two jets: one matched to top, one to nothing."""
        events = self._make_mock_events([6], [0.5], [0.5])
        jet_eta = ak.Array([[0.5, -2.0]])
        jet_phi = ak.Array([[0.5, -2.0]])
        labels = label_large_r_jets(events, jet_eta, jet_phi, R=1.0)
        flat = ak.to_numpy(ak.flatten(labels))
        assert flat[0] == 6
        assert flat[1] == 0

    # ---------------------------------------------------------------
    # R-dependence
    # ---------------------------------------------------------------

    def test_r_dependence_match(self):
        """Resonance at dR=0.7 from jet: matched with R=1.0, not with R=0.5."""
        # Place top at (eta=0.0, phi=0.0), jet at (eta=0.5, phi=0.5) -> dR ~= 0.71
        events = self._make_mock_events([6], [0.0], [0.0])
        jet_eta = ak.Array([[0.5]])
        jet_phi = ak.Array([[0.5]])

        labels_large = label_large_r_jets(events, jet_eta, jet_phi, R=1.0)
        assert ak.to_numpy(ak.flatten(labels_large))[0] == 6

        labels_small = label_large_r_jets(events, jet_eta, jet_phi, R=0.5)
        assert ak.to_numpy(ak.flatten(labels_small))[0] == 0
