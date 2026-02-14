from __future__ import annotations

import awkward as ak
import numpy as np
import pytest

from truthjets.config import JetConfig
from truthjets.modules.bb_opening_angle import BBOpeningAngleModule


class TestBBOpeningAngleModule:
    def _make_mock_events(self, particle_ids, particle_eta, particle_phi,
                          daughter1=None, daughter2=None):
        """Create a minimal mock events structure with given particles.

        By default all particles have no daughters (d1=d2=0), meaning
        final_b_hadron_mask treats them as weakly-decaying.
        """
        particle_ids = np.array(particle_ids)
        particle_eta = np.array(particle_eta, dtype=np.float64)
        particle_phi = np.array(particle_phi, dtype=np.float64)
        n = len(particle_ids)

        if daughter1 is None:
            daughter1 = np.zeros(n, dtype=np.int32)
        else:
            daughter1 = np.array(daughter1, dtype=np.int32)
        if daughter2 is None:
            daughter2 = np.zeros(n, dtype=np.int32)
        else:
            daughter2 = np.array(daughter2, dtype=np.int32)

        px = np.cos(particle_phi) * 100.0
        py = np.sin(particle_phi) * 100.0
        theta = 2 * np.arctan(np.exp(-particle_eta))
        pz = 100.0 / np.tan(theta)
        E = np.sqrt(px**2 + py**2 + pz**2 + 5.0**2)

        prt = ak.zip(
            {
                "p": ak.zip({"px": [px], "py": [py], "pz": [pz], "e": [E]}),
                "id": [particle_ids],
                "status": [np.ones(n, dtype=np.int32)],
                "daughter1": [daughter1],
                "daughter2": [daughter2],
            }
        )
        return ak.zip({"prt": prt})

    def _run_module(self, events, jet_eta, jet_phi, R=1.0):
        """Initialize module and run post_clustering, return bb_dR array."""
        mod = BBOpeningAngleModule()
        mod.init(JetConfig(R=R))

        jet_kin = ak.zip({"eta": jet_eta, "phi": jet_phi, "pt": jet_eta, "mass": jet_eta, "energy": jet_eta})
        labels = ak.zeros_like(jet_eta, dtype=np.int32)

        result = mod.post_clustering(events, None, None, jet_kin, labels)
        return result.extra_jet_data["bb_dR"]

    # ---------------------------------------------------------------
    # Exactly 2 b-hadrons matched -> correct dR
    # ---------------------------------------------------------------

    def test_two_b_hadrons_gives_correct_dr(self):
        """Two B mesons (511, 521) near the jet -> dR computed correctly."""
        # Place two b-hadrons at known positions
        b1_eta, b1_phi = 0.5, 0.3
        b2_eta, b2_phi = 0.8, 0.6
        events = self._make_mock_events(
            [511, 521], [b1_eta, b2_eta], [b1_phi, b2_phi]
        )
        jet_eta = ak.Array([[0.6]])
        jet_phi = ak.Array([[0.4]])

        bb_dr = self._run_module(events, jet_eta, jet_phi, R=1.0)
        flat = ak.to_numpy(ak.flatten(bb_dr))

        # Expected dR between the two b-hadrons
        deta = b1_eta - b2_eta
        dphi = b1_phi - b2_phi
        expected_dr = np.sqrt(deta**2 + dphi**2)

        assert flat.shape == (1,)
        np.testing.assert_allclose(flat[0], expected_dr, atol=1e-5)

    # ---------------------------------------------------------------
    # 0 or 1 b-hadrons -> NaN
    # ---------------------------------------------------------------

    def test_zero_b_hadrons_gives_nan(self):
        """No b-hadrons in event -> NaN."""
        events = self._make_mock_events([211, 321], [0.5, -0.5], [0.5, -0.5])
        jet_eta = ak.Array([[0.0]])
        jet_phi = ak.Array([[0.0]])

        bb_dr = self._run_module(events, jet_eta, jet_phi, R=1.0)
        flat = ak.to_numpy(ak.flatten(bb_dr))
        assert np.isnan(flat[0])

    def test_one_b_hadron_gives_nan(self):
        """Only one b-hadron matched -> NaN."""
        events = self._make_mock_events([511, 211], [0.5, -2.0], [0.5, -2.0])
        jet_eta = ak.Array([[0.5]])
        jet_phi = ak.Array([[0.5]])

        bb_dr = self._run_module(events, jet_eta, jet_phi, R=1.0)
        flat = ak.to_numpy(ak.flatten(bb_dr))
        assert np.isnan(flat[0])

    # ---------------------------------------------------------------
    # 3+ b-hadrons -> NaN
    # ---------------------------------------------------------------

    def test_three_b_hadrons_gives_nan(self):
        """Three b-hadrons in cone -> NaN (ambiguous)."""
        events = self._make_mock_events(
            [511, 521, 531], [0.5, 0.6, 0.4], [0.5, 0.6, 0.4]
        )
        jet_eta = ak.Array([[0.5]])
        jet_phi = ak.Array([[0.5]])

        bb_dr = self._run_module(events, jet_eta, jet_phi, R=1.0)
        flat = ak.to_numpy(ak.flatten(bb_dr))
        assert np.isnan(flat[0])

    # ---------------------------------------------------------------
    # R <= 0.4 raises ValueError
    # ---------------------------------------------------------------

    def test_small_r_raises(self):
        """R=0.4 should raise ValueError on init."""
        mod = BBOpeningAngleModule()
        with pytest.raises(ValueError, match="R > 0.4"):
            mod.init(JetConfig(R=0.4))

    def test_small_r_raises_below(self):
        """R=0.2 should also raise ValueError on init."""
        mod = BBOpeningAngleModule()
        with pytest.raises(ValueError, match="R > 0.4"):
            mod.init(JetConfig(R=0.2))

    # ---------------------------------------------------------------
    # Extra jet fields declaration
    # ---------------------------------------------------------------

    def test_extra_jet_fields(self):
        mod = BBOpeningAngleModule()
        fields = mod.extra_jet_fields()
        assert len(fields) == 1
        assert fields[0][0] == "bb_dR"
        assert fields[0][1] == np.float32

    # ---------------------------------------------------------------
    # Excited state filtering (B** -> B* -> B chain)
    # ---------------------------------------------------------------

    def test_excited_states_filtered_out(self):
        """B** and B at same position should give dR from the 2 final B's, not 4."""
        # Simulate H->bb: two decay chains at different positions
        # Chain 1: B**(523) at idx=0 -> B(521) at idx=2 + gamma(22) at idx=3
        # Chain 2: B**(513) at idx=1 -> B(511) at idx=4 + gamma(22) at idx=5
        # B**(523) has daughter B(521), so it's NOT final
        # B(521) has no b-hadron daughters, so it IS final
        b_star_eta, b_star_phi = 0.5, 0.3
        b_final1_eta, b_final1_phi = 0.5, 0.3  # same position as parent
        gamma1_eta, gamma1_phi = 0.5, 0.3
        b_star2_eta, b_star2_phi = 0.8, 0.6
        b_final2_eta, b_final2_phi = 0.8, 0.6  # same position as parent
        gamma2_eta, gamma2_phi = 0.8, 0.6

        events = self._make_mock_events(
            particle_ids=[523, 513, 521, 22, 511, 22],
            particle_eta=[b_star_eta, b_star2_eta, b_final1_eta,
                          gamma1_eta, b_final2_eta, gamma2_eta],
            particle_phi=[b_star_phi, b_star2_phi, b_final1_phi,
                          gamma1_phi, b_final2_phi, gamma2_phi],
            # B**(523) at idx=0 -> daughters at idx=2,3
            # B**(513) at idx=1 -> daughters at idx=4,5
            # B(521) at idx=2 -> no daughters (d1=d2=0)
            # B(511) at idx=4 -> no daughters (d1=d2=0)
            daughter1=[2, 4, 0, 0, 0, 0],
            daughter2=[3, 5, 0, 0, 0, 0],
        )

        jet_eta = ak.Array([[0.6]])
        jet_phi = ak.Array([[0.4]])

        bb_dr = self._run_module(events, jet_eta, jet_phi, R=1.0)
        flat = ak.to_numpy(ak.flatten(bb_dr))

        # Should get dR between the two FINAL b-hadrons (idx 2 and 4)
        deta = b_final1_eta - b_final2_eta
        dphi = b_final1_phi - b_final2_phi
        expected_dr = np.sqrt(deta**2 + dphi**2)

        assert flat.shape == (1,)
        np.testing.assert_allclose(flat[0], expected_dr, atol=1e-5)

    # ---------------------------------------------------------------
    # Multiple jets: one with 2 b-hadrons, one without
    # ---------------------------------------------------------------

    def test_multiple_jets_mixed(self):
        """Two jets: first has 2 b-hadrons in cone, second has none."""
        b1_eta, b1_phi = 0.5, 0.3
        b2_eta, b2_phi = 0.8, 0.6
        events = self._make_mock_events(
            [511, 521], [b1_eta, b2_eta], [b1_phi, b2_phi]
        )
        # First jet near the b-hadrons, second jet far away
        jet_eta = ak.Array([[0.6, -2.0]])
        jet_phi = ak.Array([[0.4, -2.0]])

        bb_dr = self._run_module(events, jet_eta, jet_phi, R=1.0)
        flat = ak.to_numpy(ak.flatten(bb_dr))

        # First jet should have a valid dR
        deta = b1_eta - b2_eta
        dphi = b1_phi - b2_phi
        expected_dr = np.sqrt(deta**2 + dphi**2)
        np.testing.assert_allclose(flat[0], expected_dr, atol=1e-5)

        # Second jet should be NaN
        assert np.isnan(flat[1])
