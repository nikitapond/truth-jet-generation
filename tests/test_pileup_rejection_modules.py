from __future__ import annotations

import awkward as ak
import numpy as np

from truthjets.config import JetConfig
from truthjets.modules.pileup_rejection import SoftKillerModule, VertexZFilterModule
from truthjets.pileup_rejection import softkiller

# ---------------------------------------------------------------------------
# SoftKillerModule
# ---------------------------------------------------------------------------


class TestSoftKillerModule:
    def _make_particles(self, n_events=2, n_particles=50, seed=42):
        """Create random particles with px, py, pz, E, pdgId, is_pu fields."""
        rng = np.random.default_rng(seed)
        records = []
        for _ in range(n_events):
            n = rng.integers(n_particles // 2, n_particles + 1)
            evt = {
                "px": rng.uniform(-10, 10, n).tolist(),
                "py": rng.uniform(-10, 10, n).tolist(),
                "pz": rng.uniform(-50, 50, n).tolist(),
                "E": rng.uniform(5, 100, n).tolist(),
                "pdgId": rng.choice([211, -211, 321, 22], n).tolist(),
                "is_pu": [False] * n,
            }
            records.append(evt)
        return ak.Array(records)

    def test_init_reads_grid_size(self):
        mod = SoftKillerModule()
        cfg = JetConfig(softkiller_grid=0.6)
        mod.init(cfg)
        assert mod.grid_size == 0.6

    def test_pre_clustering_matches_direct_call(self):
        """Module output should match direct softkiller() call."""
        particles = self._make_particles()
        grid_size = 0.5

        mod = SoftKillerModule()
        mod.init(JetConfig(softkiller_grid=grid_size))
        result = mod.pre_clustering(None, particles)

        expected = softkiller(particles, grid_size=grid_size)

        # Compare particle counts per event
        assert ak.all(ak.num(result.px) == ak.num(expected.px))

    def test_pre_clustering_none_particles_returns_none(self):
        """If particles is None, module should return None."""
        mod = SoftKillerModule()
        mod.init(JetConfig())
        assert mod.pre_clustering(None, None) is None

    def test_post_clustering_is_noop(self):
        mod = SoftKillerModule()
        mod.init(JetConfig())
        assert mod.post_clustering(None, None, None, None, None) is None

    def test_no_extra_fields(self):
        mod = SoftKillerModule()
        assert mod.extra_jet_fields() == []
        assert mod.extra_datasets() == {}


# ---------------------------------------------------------------------------
# VertexZFilterModule
# ---------------------------------------------------------------------------


class TestVertexZFilterModule:
    def _make_data(self, vz_values, n_constit=3, seed=42):
        """Create jet data with specified per-jet mean vz values.

        Each jet gets n_constit constituents with uniform pT and vz set
        so that the pT-weighted mean vz equals the target value.
        """
        rng = np.random.default_rng(seed)
        n_jets = len(vz_values)

        # Single event with n_jets jets
        constit_list = []
        for target_vz in vz_values:
            jet_constit = {
                "px": [10.0] * n_constit,
                "py": [0.0] * n_constit,
                "pz": [0.0] * n_constit,
                "E": [10.0] * n_constit,
                "vz": [float(target_vz)] * n_constit,
            }
            constit_list.append(jet_constit)

        constituents = ak.Array([constit_list])
        jet_kin = ak.zip(
            {
                "pt": ak.Array([rng.uniform(20, 200, n_jets).tolist()]),
                "eta": ak.Array([rng.uniform(-2.5, 2.5, n_jets).tolist()]),
                "phi": ak.Array([rng.uniform(-np.pi, np.pi, n_jets).tolist()]),
                "mass": ak.Array([rng.uniform(0, 20, n_jets).tolist()]),
                "energy": ak.Array([rng.uniform(50, 500, n_jets).tolist()]),
            }
        )
        jets = jet_kin  # placeholder
        labels = ak.zeros_like(jet_kin.pt, dtype=np.int32)
        return jets, constituents, jet_kin, labels

    def test_init_reads_max_dz(self):
        mod = VertexZFilterModule()
        mod.init(JetConfig(max_dz=5.0))
        assert mod.max_dz == 5.0

    def test_all_pass(self):
        """All jets within max_dz should survive."""
        jets, constits, jet_kin, labels = self._make_data([0.0, 1.0, -2.0])
        mod = VertexZFilterModule()
        mod.init(JetConfig(max_dz=5.0))

        result = mod.post_clustering(None, jets, constits, jet_kin, labels)
        assert ak.sum(ak.num(result.jet_kin.pt)) == 3

    def test_all_fail(self):
        """All jets outside max_dz should be removed."""
        jets, constits, jet_kin, labels = self._make_data([100.0, -100.0])
        mod = VertexZFilterModule()
        mod.init(JetConfig(max_dz=5.0))

        result = mod.post_clustering(None, jets, constits, jet_kin, labels)
        assert ak.sum(ak.num(result.jet_kin.pt)) == 0

    def test_mixed(self):
        """Only jets within max_dz should survive."""
        jets, constits, jet_kin, labels = self._make_data([0.0, 100.0, 2.0, -50.0])
        mod = VertexZFilterModule()
        mod.init(JetConfig(max_dz=5.0))

        result = mod.post_clustering(None, jets, constits, jet_kin, labels)
        # Jets at vz=0.0 and vz=2.0 survive; vz=100 and vz=-50 rejected
        assert ak.sum(ak.num(result.jet_kin.pt)) == 2

    def test_all_arrays_filtered_consistently(self):
        """jets, constituents, jet_kin, and labels should all be filtered."""
        jets, constits, jet_kin, labels = self._make_data([0.0, 100.0, 2.0])
        mod = VertexZFilterModule()
        mod.init(JetConfig(max_dz=5.0))

        result = mod.post_clustering(None, jets, constits, jet_kin, labels)
        n = ak.sum(ak.num(result.jet_kin.pt))
        assert ak.sum(ak.num(result.jets.pt)) == n
        assert ak.sum(ak.num(result.constituents.px)) == n
        assert ak.sum(ak.num(result.labels)) == n

    def test_pre_clustering_is_noop(self):
        mod = VertexZFilterModule()
        mod.init(JetConfig(max_dz=5.0))
        assert mod.pre_clustering(None, None) is None

    def test_no_extra_fields(self):
        mod = VertexZFilterModule()
        assert mod.extra_jet_fields() == []
        assert mod.extra_datasets() == {}
