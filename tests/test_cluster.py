from __future__ import annotations

import awkward as ak
import numpy as np

from truthjets.cluster import cluster_jets, compute_jet_kinematics
from truthjets.config import JetConfig


def _make_mock_events(particles_per_event):
    """Create mock Pythia-like events from a list of particle lists.

    Each particle is (px, py, pz, E, pdg_id).
    """
    all_px, all_py, all_pz, all_E, all_id, all_status = [], [], [], [], [], []
    for event_particles in particles_per_event:
        px, py, pz, E, pdg = zip(*event_particles)
        all_px.append(list(px))
        all_py.append(list(py))
        all_pz.append(list(pz))
        all_E.append(list(E))
        all_id.append(list(pdg))
        all_status.append([1] * len(px))  # all final state

    prt = ak.zip(
        {
            "p": ak.zip(
                {
                    "px": ak.Array(all_px),
                    "py": ak.Array(all_py),
                    "pz": ak.Array(all_pz),
                    "e": ak.Array(all_E),
                }
            ),
            "id": ak.Array(all_id),
            "status": ak.Array(all_status),
        }
    )
    return ak.zip({"prt": prt})


class TestClusterJets:
    def test_single_hard_particle_makes_one_jet(self):
        """A single high-pT particle should produce exactly one jet."""
        # Particle along +x with pT=50 GeV, eta~0
        events = _make_mock_events([[(50.0, 0.0, 0.0, 50.0, 211)]])
        jet_config = JetConfig(pt_min=20.0, eta_max=2.5)
        jets, constits, jet_kin = cluster_jets(events, jet_config)

        flat_jets = ak.flatten(jet_kin)
        assert len(flat_jets) == 1
        np.testing.assert_allclose(ak.to_numpy(flat_jets.pt)[0], 50.0, atol=0.1)

    def test_collinear_particles_merge(self):
        """Two collinear particles should merge into one jet."""
        events = _make_mock_events(
            [
                [
                    (30.0, 0.0, 0.0, 30.0, 211),
                    (20.0, 0.1, 0.0, 20.0, 211),
                ]
            ]
        )
        jet_config = JetConfig(R=0.4, pt_min=20.0, eta_max=2.5)
        jets, constits, jet_kin = cluster_jets(events, jet_config)

        flat_jets = ak.flatten(jet_kin)
        assert len(flat_jets) == 1
        np.testing.assert_allclose(ak.to_numpy(flat_jets.pt)[0], 50.0, atol=0.5)

    def test_separated_particles_make_two_jets(self):
        """Two well-separated hard particles should make two jets."""
        events = _make_mock_events(
            [
                [
                    (50.0, 0.0, 0.0, 50.0, 211),  # along +x
                    (0.0, 50.0, 0.0, 50.0, -211),  # along +y
                ]
            ]
        )
        jet_config = JetConfig(R=0.4, pt_min=20.0, eta_max=2.5)
        jets, constits, jet_kin = cluster_jets(events, jet_config)

        flat_jets = ak.flatten(jet_kin)
        assert len(flat_jets) == 2

    def test_pt_cut(self):
        """Jets below pT cut should be removed."""
        events = _make_mock_events(
            [
                [
                    (50.0, 0.0, 0.0, 50.0, 211),
                    (5.0, 0.0, 5.0, 7.1, 211),  # low pT
                ]
            ]
        )
        jet_config = JetConfig(pt_min=20.0, eta_max=2.5)
        jets, constits, jet_kin = cluster_jets(events, jet_config)

        flat_pts = ak.to_numpy(ak.flatten(jet_kin.pt))
        assert np.all(flat_pts > 20.0)

    def test_eta_cut(self):
        """Jets beyond eta cut should be removed."""
        # Particle at very high eta (along beam)
        events = _make_mock_events(
            [
                [
                    (50.0, 0.0, 0.0, 50.0, 211),
                    (1.0, 0.0, 500.0, 500.0, 211),  # very forward
                ]
            ]
        )
        jet_config = JetConfig(pt_min=0.5, eta_max=2.5)
        jets, constits, jet_kin = cluster_jets(events, jet_config)

        flat_etas = ak.to_numpy(ak.flatten(jet_kin.eta))
        assert np.all(np.abs(flat_etas) < 2.5)

    def test_constituents_have_pdgid(self):
        """Constituents should carry pdgId field."""
        events = _make_mock_events([[(50.0, 0.0, 0.0, 50.0, 211)]])
        jet_config = JetConfig(pt_min=20.0, eta_max=2.5)
        jets, constits, jet_kin = cluster_jets(events, jet_config)

        flat_constits = ak.flatten(constits, axis=1)
        assert "pdgId" in ak.fields(flat_constits[0])


class TestComputeJetKinematics:
    def test_massless_particle(self):
        """A massless particle along +x should have eta=0, phi=0."""
        jets = ak.Array([{"px": 50.0, "py": 0.0, "pz": 0.0, "E": 50.0}])
        kin = compute_jet_kinematics(jets)
        np.testing.assert_allclose(ak.to_numpy(kin.pt), [50.0], atol=0.01)
        np.testing.assert_allclose(ak.to_numpy(kin.eta), [0.0], atol=0.01)
        np.testing.assert_allclose(ak.to_numpy(kin.phi), [0.0], atol=0.01)
        np.testing.assert_allclose(ak.to_numpy(kin.mass), [0.0], atol=0.01)
