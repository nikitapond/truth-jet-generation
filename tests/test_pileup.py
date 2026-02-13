from __future__ import annotations

import awkward as ak
import numpy as np

from truthjets.pileup import (
    load_pileup_pool,
    overlay_pileup,
    sample_from_pool,
    sample_n_pileup,
    save_pileup_pool,
)


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


def _make_mock_pool(n_events, particles_per_event=5, seed=42):
    """Create a mock pileup pool with fields {px, py, pz, E, pdgId}."""
    rng = np.random.default_rng(seed)
    all_px, all_py, all_pz, all_e, all_pdgid = [], [], [], [], []
    for _ in range(n_events):
        n = particles_per_event
        all_px.append(rng.uniform(-10, 10, n).tolist())
        all_py.append(rng.uniform(-10, 10, n).tolist())
        all_pz.append(rng.uniform(-10, 10, n).tolist())
        all_e.append(rng.uniform(10, 50, n).tolist())
        all_pdgid.append(rng.choice([211, -211, 321, 22], n).tolist())

    return ak.zip(
        {
            "px": ak.Array(all_px),
            "py": ak.Array(all_py),
            "pz": ak.Array(all_pz),
            "E": ak.Array(all_e),
            "pdgId": ak.Array(all_pdgid),
        }
    )


class TestSaveLoadRoundtrip:
    def test_roundtrip_preserves_data(self, tmp_path):
        pool = _make_mock_pool(20, particles_per_event=5)
        path = tmp_path / "pool.h5"
        save_pileup_pool(pool, path)
        loaded = load_pileup_pool(path)

        assert len(loaded) == len(pool)
        for field in ["px", "py", "pz", "E", "pdgId"]:
            orig = ak.to_numpy(ak.flatten(getattr(pool, field)))
            back = ak.to_numpy(ak.flatten(getattr(loaded, field)))
            np.testing.assert_allclose(orig, back, atol=1e-6)

    def test_roundtrip_preserves_ragged_structure(self, tmp_path):
        """Pool with variable-length events round-trips correctly."""
        rng = np.random.default_rng(99)
        # Build ragged pool manually
        all_px, all_py, all_pz, all_e, all_pdgid = [], [], [], [], []
        for _ in range(10):
            n = rng.integers(2, 8)
            all_px.append(rng.uniform(-5, 5, n).tolist())
            all_py.append(rng.uniform(-5, 5, n).tolist())
            all_pz.append(rng.uniform(-5, 5, n).tolist())
            all_e.append(rng.uniform(5, 30, n).tolist())
            all_pdgid.append(rng.choice([211, -211], n).tolist())

        pool = ak.zip(
            {
                "px": ak.Array(all_px),
                "py": ak.Array(all_py),
                "pz": ak.Array(all_pz),
                "E": ak.Array(all_e),
                "pdgId": ak.Array(all_pdgid),
            }
        )
        path = tmp_path / "ragged_pool.h5"
        save_pileup_pool(pool, path)
        loaded = load_pileup_pool(path)

        # Check per-event particle counts match
        orig_counts = ak.to_numpy(ak.num(pool, axis=1))
        loaded_counts = ak.to_numpy(ak.num(loaded, axis=1))
        np.testing.assert_array_equal(orig_counts, loaded_counts)


class TestSampleFromPool:
    def test_correct_count(self):
        pool = _make_mock_pool(100, particles_per_event=5)
        rng = np.random.default_rng(42)
        sampled = sample_from_pool(pool, 50, rng)
        assert len(sampled) == 50

    def test_phi_rotation_changes_px_py(self):
        pool = _make_mock_pool(100, particles_per_event=5)
        rng = np.random.default_rng(42)
        sampled = sample_from_pool(pool, 50, rng)

        # Reconstruct what the unrotated sample would be (re-sample with same seed)
        rng2 = np.random.default_rng(42)
        pool_size = len(pool)
        indices = rng2.integers(0, pool_size, size=50)
        unrotated = pool[indices]

        # px/py should differ (rotation applied)
        px_orig = ak.to_numpy(ak.flatten(unrotated.px))
        px_rot = ak.to_numpy(ak.flatten(sampled.px))
        assert not np.allclose(px_orig, px_rot)

    def test_phi_rotation_preserves_pt(self):
        pool = _make_mock_pool(100, particles_per_event=5)
        rng = np.random.default_rng(42)
        sampled = sample_from_pool(pool, 50, rng)

        # Reconstruct unrotated sample
        rng2 = np.random.default_rng(42)
        indices = rng2.integers(0, len(pool), size=50)
        unrotated = pool[indices]

        # pT = sqrt(px^2 + py^2) should be preserved
        pt_orig = np.sqrt(
            ak.to_numpy(ak.flatten(unrotated.px)) ** 2
            + ak.to_numpy(ak.flatten(unrotated.py)) ** 2
        )
        pt_rot = np.sqrt(
            ak.to_numpy(ak.flatten(sampled.px)) ** 2
            + ak.to_numpy(ak.flatten(sampled.py)) ** 2
        )
        np.testing.assert_allclose(pt_orig, pt_rot, atol=1e-5)

    def test_phi_rotation_preserves_pz(self):
        pool = _make_mock_pool(100, particles_per_event=5)
        rng = np.random.default_rng(42)
        sampled = sample_from_pool(pool, 50, rng)

        # Reconstruct unrotated sample
        rng2 = np.random.default_rng(42)
        indices = rng2.integers(0, len(pool), size=50)
        unrotated = pool[indices]

        pz_orig = ak.to_numpy(ak.flatten(unrotated.pz))
        pz_rot = ak.to_numpy(ak.flatten(sampled.pz))
        np.testing.assert_allclose(pz_orig, pz_rot, atol=1e-6)

    def test_has_required_fields(self):
        pool = _make_mock_pool(50, particles_per_event=5)
        rng = np.random.default_rng(42)
        sampled = sample_from_pool(pool, 20, rng)
        for field in ["px", "py", "pz", "E", "pdgId"]:
            assert field in ak.fields(sampled), f"Missing field: {field}"


class TestSamplePhiRotationUniformity:
    def test_phi_angles_are_uniform(self):
        """Rotation angles should be roughly uniform in [0, 2pi)."""
        # Use a pool with known px, py so we can recover theta
        n_pool = 1
        pool = ak.zip(
            {
                "px": ak.Array([[1.0]]),
                "py": ak.Array([[0.0]]),
                "pz": ak.Array([[0.0]]),
                "E": ak.Array([[1.0]]),
                "pdgId": ak.Array([[211]]),
            }
        )
        rng = np.random.default_rng(42)
        n_samples = 10_000
        sampled = sample_from_pool(pool, n_samples, rng)

        # For initial (1, 0), after rotation by theta: px'=cos(theta), py'=sin(theta)
        px_vals = ak.to_numpy(ak.flatten(sampled.px))
        py_vals = ak.to_numpy(ak.flatten(sampled.py))
        recovered_theta = np.arctan2(py_vals, px_vals) % (2 * np.pi)

        # Check that thetas are roughly uniform using histogram
        hist, _ = np.histogram(recovered_theta, bins=8, range=(0, 2 * np.pi))
        expected = n_samples / 8
        # Chi-squared-like: each bin within 20% of expected
        for count in hist:
            assert abs(count - expected) / expected < 0.2, (
                f"Bin count {count} too far from expected {expected}"
            )


class TestOverlayWithPuParticles:
    def test_overlay_with_pre_processed_particles(self):
        """overlay_pileup works with pu_particles instead of pu_events."""
        hs = _make_mock_events(2, particles_per_event=3)
        pu_particles = _make_mock_pool(5, particles_per_event=4)
        n_pu = np.array([2, 3])

        merged = overlay_pileup(hs, None, n_pu, pu_particles=pu_particles)

        # Event 0: 3 HS + 2*4 PU = 11
        assert ak.num(merged, axis=1)[0] == 3 + 2 * 4
        # Event 1: 3 HS + 3*4 PU = 15
        assert ak.num(merged, axis=1)[1] == 3 + 3 * 4

    def test_overlay_pu_particles_has_all_fields(self):
        hs = _make_mock_events(1, particles_per_event=3)
        pu_particles = _make_mock_pool(2, particles_per_event=4)
        n_pu = np.array([2])

        merged = overlay_pileup(hs, None, n_pu, pu_particles=pu_particles)

        for field in ["px", "py", "pz", "E", "pdgId", "is_pu", "vz"]:
            assert field in ak.fields(merged), f"Missing field: {field}"

    def test_overlay_pu_particles_is_pu_flags(self):
        hs = _make_mock_events(1, particles_per_event=3)
        pu_particles = _make_mock_pool(2, particles_per_event=4)
        n_pu = np.array([2])

        merged = overlay_pileup(hs, None, n_pu, pu_particles=pu_particles)

        is_pu = ak.to_numpy(merged.is_pu[0])
        # First 3 are HS (False), last 8 are PU (True)
        assert not np.any(is_pu[:3])
        assert np.all(is_pu[3:])
