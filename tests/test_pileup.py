from __future__ import annotations

import awkward as ak
import numpy as np

from truthjets.pileup import overlay_pileup, sample_n_pileup


def _make_mock_events(n_events, particles_per_event=3):
    """Create mock Pythia-like events with final-state particles."""
    rng = np.random.default_rng(42)
    all_px, all_py, all_pz, all_e, all_id, all_status = [], [], [], [], [], []
    for _ in range(n_events):
        n_prt = particles_per_event
        px = rng.uniform(-10, 10, n_prt).tolist()
        py = rng.uniform(-10, 10, n_prt).tolist()
        pz = rng.uniform(-10, 10, n_prt).tolist()
        e = rng.uniform(10, 50, n_prt).tolist()
        ids = rng.choice([211, -211, 321, 22], n_prt).tolist()
        all_px.append(px)
        all_py.append(py)
        all_pz.append(pz)
        all_e.append(e)
        all_id.append(ids)
        all_status.append([1] * n_prt)

    prt = ak.zip(
        {
            "p": ak.zip(
                {
                    "px": ak.Array(all_px),
                    "py": ak.Array(all_py),
                    "pz": ak.Array(all_pz),
                    "e": ak.Array(all_e),
                }
            ),
            "id": ak.Array(all_id),
            "status": ak.Array(all_status),
        }
    )
    return ak.zip({"prt": prt})


class TestSampleNPileup:
    def test_returns_correct_shape(self):
        rng = np.random.default_rng(42)
        result = sample_n_pileup(40.0, 100, rng)
        assert result.shape == (100,)

    def test_mean_close_to_mu(self):
        rng = np.random.default_rng(42)
        result = sample_n_pileup(40.0, 100_000, rng)
        assert abs(np.mean(result) - 40.0) < 1.0

    def test_all_nonnegative(self):
        rng = np.random.default_rng(42)
        result = sample_n_pileup(20.0, 1000, rng)
        assert np.all(result >= 0)

    def test_zero_mu(self):
        rng = np.random.default_rng(42)
        result = sample_n_pileup(0.0, 100, rng)
        assert np.all(result == 0)


class TestOverlayPileup:
    def test_no_pileup_returns_hs_with_is_pu(self):
        hs = _make_mock_events(3, particles_per_event=5)
        n_pu = np.array([0, 0, 0])
        merged = overlay_pileup(hs, None, n_pu)

        # Should have is_pu field, all False
        assert "is_pu" in ak.fields(merged)
        assert not ak.any(ak.flatten(merged.is_pu))

        # Particle count should match HS
        assert ak.sum(ak.num(merged)) == ak.sum(ak.num(hs.prt[hs.prt.status > 0]))

    def test_pileup_adds_particles(self):
        hs = _make_mock_events(2, particles_per_event=3)
        pu = _make_mock_events(5, particles_per_event=4)  # 5 PU events total
        n_pu = np.array([2, 3])  # 2 PU events for HS event 0, 3 for event 1

        merged = overlay_pileup(hs, pu, n_pu)

        # Event 0: 3 HS + 2*4 PU = 11
        assert ak.num(merged, axis=1)[0] == 3 + 2 * 4
        # Event 1: 3 HS + 3*4 PU = 15
        assert ak.num(merged, axis=1)[1] == 3 + 3 * 4

    def test_is_pu_flags_correct(self):
        hs = _make_mock_events(1, particles_per_event=3)
        pu = _make_mock_events(2, particles_per_event=4)
        n_pu = np.array([2])

        merged = overlay_pileup(hs, pu, n_pu)

        is_pu = ak.to_numpy(merged.is_pu[0])
        # First 3 should be HS (False), last 8 should be PU (True)
        assert not np.any(is_pu[:3])
        assert np.all(is_pu[3:])

    def test_merged_has_required_fields(self):
        hs = _make_mock_events(2, particles_per_event=3)
        pu = _make_mock_events(3, particles_per_event=2)
        n_pu = np.array([1, 2])

        merged = overlay_pileup(hs, pu, n_pu)

        for field in ["px", "py", "pz", "E", "pdgId", "is_pu"]:
            assert field in ak.fields(merged), f"Missing field: {field}"
