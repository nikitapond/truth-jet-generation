from __future__ import annotations

import os
import sys
import tempfile
import textwrap

import awkward as ak
import h5py
import numpy as np
import pytest

from truthjets.config import JetConfig
from truthjets.modules import (
    DatasetSchema,
    ModuleResult,
    TruthJetModule,
    load_module,
    validate_modules,
)
from truthjets.writer import JET_DTYPE, HDF5Writer

# ---------------------------------------------------------------------------
# Helper modules for tests
# ---------------------------------------------------------------------------


class DummyModule(TruthJetModule):
    """No-op module for baseline tests."""


class ExtraFieldModule(TruthJetModule):
    """Adds a single extra float32 jet field."""

    def extra_jet_fields(self):
        return [("my_score", np.float32)]

    def post_clustering(self, events, jets, constituents, jet_kin, labels):
        flat_pt = ak.to_numpy(ak.flatten(jet_kin.pt))
        return ModuleResult(extra_jet_data={"my_score": flat_pt * 0.5})


class ExtraDatasetModule(TruthJetModule):
    """Adds an extra per-jet dataset."""

    def extra_datasets(self):
        return {
            "custom_info": DatasetSchema(
                dtype=np.dtype([("value", np.float32)]),
                shape_suffix=(),
            )
        }

    def post_clustering(self, events, jets, constituents, jet_kin, labels):
        flat_pt = ak.to_numpy(ak.flatten(jet_kin.pt))
        n = len(flat_pt)
        data = np.zeros(n, dtype=np.dtype([("value", np.float32)]))
        data["value"] = flat_pt * 2.0
        return ModuleResult(extra_dataset_data={"custom_info": data})


class ConflictFieldModule(TruthJetModule):
    """Tries to shadow built-in 'pt' field."""

    def extra_jet_fields(self):
        return [("pt", np.float32)]


class ConflictDatasetModule(TruthJetModule):
    """Tries to shadow built-in 'jets' dataset."""

    def extra_datasets(self):
        return {"jets": DatasetSchema(dtype=np.float32)}


class DuplicateFieldModuleA(TruthJetModule):
    def extra_jet_fields(self):
        return [("shared_field", np.float32)]


class DuplicateFieldModuleB(TruthJetModule):
    def extra_jet_fields(self):
        return [("shared_field", np.float32)]


# ---------------------------------------------------------------------------
# Test data helper (reused from test_writer.py pattern)
# ---------------------------------------------------------------------------


def _make_test_data(n_jets_per_event=2, n_events=3, n_constit=5):
    rng = np.random.default_rng(42)
    jet_kin = ak.zip(
        {
            "pt": ak.Array(rng.uniform(20, 200, (n_events, n_jets_per_event)).tolist()),
            "eta": ak.Array(rng.uniform(-2.5, 2.5, (n_events, n_jets_per_event)).tolist()),
            "phi": ak.Array(rng.uniform(-np.pi, np.pi, (n_events, n_jets_per_event)).tolist()),
            "mass": ak.Array(rng.uniform(0, 20, (n_events, n_jets_per_event)).tolist()),
            "energy": ak.Array(rng.uniform(50, 500, (n_events, n_jets_per_event)).tolist()),
        }
    )
    labels = ak.Array(rng.choice([0, 4, 5, 15], (n_events, n_jets_per_event)).tolist())
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


# ---------------------------------------------------------------------------
# Module loading tests
# ---------------------------------------------------------------------------


class TestLoadModule:
    def test_load_explicit_class(self, tmp_path):
        """Load a module with explicit :ClassName syntax."""
        mod_file = tmp_path / "my_mod.py"
        mod_file.write_text(
            textwrap.dedent("""\
            import numpy as np
            from truthjets.modules import TruthJetModule

            class MyModule(TruthJetModule):
                def extra_jet_fields(self):
                    return [("test_field", np.float32)]
        """)
        )
        sys.path.insert(0, str(tmp_path))
        try:
            mod = load_module("my_mod:MyModule")
            assert isinstance(mod, TruthJetModule)
            assert mod.extra_jet_fields() == [("test_field", np.float32)]
        finally:
            sys.path.pop(0)
            sys.modules.pop("my_mod", None)

    def test_load_auto_discover(self, tmp_path):
        """Load a module with auto-discovery (no :ClassName)."""
        mod_file = tmp_path / "auto_mod.py"
        mod_file.write_text(
            textwrap.dedent("""\
            from truthjets.modules import TruthJetModule

            class AutoModule(TruthJetModule):
                pass
        """)
        )
        sys.path.insert(0, str(tmp_path))
        try:
            mod = load_module("auto_mod")
            assert type(mod).__name__ == "AutoModule"
        finally:
            sys.path.pop(0)
            sys.modules.pop("auto_mod", None)

    def test_load_auto_discover_multiple_raises(self, tmp_path):
        """Multiple subclasses without :ClassName should raise."""
        mod_file = tmp_path / "multi_mod.py"
        mod_file.write_text(
            textwrap.dedent("""\
            from truthjets.modules import TruthJetModule

            class ModA(TruthJetModule):
                pass

            class ModB(TruthJetModule):
                pass
        """)
        )
        sys.path.insert(0, str(tmp_path))
        try:
            with pytest.raises(ImportError, match="Multiple"):
                load_module("multi_mod")
        finally:
            sys.path.pop(0)
            sys.modules.pop("multi_mod", None)

    def test_load_no_subclass_raises(self, tmp_path):
        """Module with no TruthJetModule subclass should raise."""
        mod_file = tmp_path / "empty_mod.py"
        mod_file.write_text("x = 1\n")
        sys.path.insert(0, str(tmp_path))
        try:
            with pytest.raises(ImportError, match="No TruthJetModule"):
                load_module("empty_mod")
        finally:
            sys.path.pop(0)
            sys.modules.pop("empty_mod", None)

    def test_load_explicit_not_found_raises(self, tmp_path):
        """Explicit class name that doesn't exist should raise."""
        mod_file = tmp_path / "some_mod.py"
        mod_file.write_text("x = 1\n")
        sys.path.insert(0, str(tmp_path))
        try:
            with pytest.raises(ImportError, match="not found"):
                load_module("some_mod:NoSuchClass")
        finally:
            sys.path.pop(0)
            sys.modules.pop("some_mod", None)

    def test_load_explicit_not_subclass_raises(self, tmp_path):
        """Explicit class that isn't a TruthJetModule subclass should raise."""
        mod_file = tmp_path / "bad_mod.py"
        mod_file.write_text("class NotAModule:\n    pass\n")
        sys.path.insert(0, str(tmp_path))
        try:
            with pytest.raises(TypeError, match="not a TruthJetModule"):
                load_module("bad_mod:NotAModule")
        finally:
            sys.path.pop(0)
            sys.modules.pop("bad_mod", None)


# ---------------------------------------------------------------------------
# Validation tests
# ---------------------------------------------------------------------------


class TestValidateModules:
    def test_no_modules_ok(self):
        validate_modules([])

    def test_single_module_ok(self):
        validate_modules([ExtraFieldModule()])

    def test_conflict_with_builtin_field(self):
        with pytest.raises(ValueError, match="built-in field"):
            validate_modules([ConflictFieldModule()])

    def test_conflict_with_builtin_dataset(self):
        with pytest.raises(ValueError, match="built-in dataset"):
            validate_modules([ConflictDatasetModule()])

    def test_duplicate_field_across_modules(self):
        with pytest.raises(ValueError, match="already declared"):
            validate_modules([DuplicateFieldModuleA(), DuplicateFieldModuleB()])

    def test_non_conflicting_modules_ok(self):
        validate_modules([ExtraFieldModule(), ExtraDatasetModule()])


# ---------------------------------------------------------------------------
# TruthJetModule base class tests
# ---------------------------------------------------------------------------


class TestTruthJetModuleBase:
    def test_default_methods_are_noop(self):
        mod = TruthJetModule()
        mod.init(JetConfig())
        assert mod.pre_clustering(None, None) is None
        assert mod.post_clustering(None, None, None, None, None) is None
        assert mod.extra_jet_fields() == []
        assert mod.extra_datasets() == {}


# ---------------------------------------------------------------------------
# Writer integration tests
# ---------------------------------------------------------------------------


class TestWriterWithExtraFields:
    def test_extra_jet_field_written(self):
        """Extra jet fields should appear in the HDF5 /jets dataset."""
        jet_config = JetConfig(max_constituents=10)
        jet_kin, labels, constituents = _make_test_data()
        extra_fields = [("my_score", np.float32)]

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        try:
            with HDF5Writer(path, jet_config, extra_jet_fields=extra_fields) as writer:
                flat_pt = ak.to_numpy(ak.flatten(jet_kin.pt)).astype(np.float32)
                extra_data = {"my_score": flat_pt * 0.5}
                writer.write_batch(
                    jet_kin,
                    labels,
                    constituents,
                    jet_kin.eta,
                    jet_kin.phi,
                    extra_jet_data=extra_data,
                )

            with h5py.File(path, "r") as f:
                assert "my_score" in f["jets"].dtype.names
                scores = f["jets"]["my_score"][:]
                expected_pt = ak.to_numpy(ak.flatten(jet_kin.pt)).astype(np.float32)
                np.testing.assert_allclose(scores, expected_pt * 0.5, rtol=1e-5)

                # Standard fields still present
                assert "pt" in f["jets"].dtype.names
                assert "eta" in f["jets"].dtype.names
        finally:
            os.unlink(path)

    def test_extra_dataset_written(self):
        """Extra datasets should appear as top-level HDF5 datasets."""
        jet_config = JetConfig(max_constituents=10)
        jet_kin, labels, constituents = _make_test_data()
        ds_dtype = np.dtype([("value", np.float32)])
        extra_datasets = {
            "custom_info": DatasetSchema(dtype=ds_dtype, shape_suffix=()),
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        try:
            with HDF5Writer(path, jet_config, extra_datasets=extra_datasets) as writer:
                n_total = int(ak.sum(ak.num(jet_kin.pt)))
                flat_pt = ak.to_numpy(ak.flatten(jet_kin.pt)).astype(np.float32)
                ds_data = np.zeros(n_total, dtype=ds_dtype)
                ds_data["value"] = flat_pt * 2.0
                writer.write_batch(
                    jet_kin,
                    labels,
                    constituents,
                    jet_kin.eta,
                    jet_kin.phi,
                    extra_dataset_data={"custom_info": ds_data},
                )

            with h5py.File(path, "r") as f:
                assert "custom_info" in f
                assert f["custom_info"].shape == (n_total,)
                vals = f["custom_info"]["value"][:]
                expected = ak.to_numpy(ak.flatten(jet_kin.pt)).astype(np.float32) * 2.0
                np.testing.assert_allclose(vals, expected, rtol=1e-5)
        finally:
            os.unlink(path)

    def test_extra_dataset_2d(self):
        """Extra datasets with shape_suffix should create multi-dim datasets."""
        jet_config = JetConfig(max_constituents=10)
        jet_kin, labels, constituents = _make_test_data()
        extra_datasets = {
            "vectors": DatasetSchema(dtype=np.float32, shape_suffix=(3,)),
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        try:
            with HDF5Writer(path, jet_config, extra_datasets=extra_datasets) as writer:
                n_total = int(ak.sum(ak.num(jet_kin.pt)))
                vec_data = np.ones((n_total, 3), dtype=np.float32)
                writer.write_batch(
                    jet_kin,
                    labels,
                    constituents,
                    jet_kin.eta,
                    jet_kin.phi,
                    extra_dataset_data={"vectors": vec_data},
                )

            with h5py.File(path, "r") as f:
                assert f["vectors"].shape == (n_total, 3)
                np.testing.assert_allclose(f["vectors"][:], 1.0)
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------


class TestBackwardCompatibility:
    def test_writer_without_extras_unchanged(self):
        """Writer with no extra fields/datasets should produce identical output."""
        jet_config = JetConfig(max_constituents=10)
        jet_kin, labels, constituents = _make_test_data()

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        try:
            with HDF5Writer(path, jet_config) as writer:
                writer.write_batch(jet_kin, labels, constituents, jet_kin.eta, jet_kin.phi)

            with h5py.File(path, "r") as f:
                assert f["jets"].dtype == JET_DTYPE
                assert set(f.keys()) == {"jets", "constituents"}
        finally:
            os.unlink(path)

    def test_writer_streaming_with_extras(self):
        """Multiple batches with extra fields should accumulate correctly."""
        jet_config = JetConfig(max_constituents=10)
        extra_fields = [("batch_id", np.int32)]

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            path = f.name

        try:
            with HDF5Writer(path, jet_config, extra_jet_fields=extra_fields) as writer:
                for batch_idx in range(3):
                    jet_kin, labels, constits = _make_test_data(n_jets_per_event=2, n_events=2)
                    n = int(ak.sum(ak.num(jet_kin.pt)))
                    extra_data = {"batch_id": np.full(n, batch_idx, dtype=np.int32)}
                    writer.write_batch(
                        jet_kin,
                        labels,
                        constits,
                        jet_kin.eta,
                        jet_kin.phi,
                        extra_jet_data=extra_data,
                    )
                assert writer.n_jets == 12

            with h5py.File(path, "r") as f:
                assert f["jets"].shape[0] == 12
                batch_ids = f["jets"]["batch_id"][:]
                assert np.all(batch_ids[:4] == 0)
                assert np.all(batch_ids[4:8] == 1)
                assert np.all(batch_ids[8:12] == 2)
        finally:
            os.unlink(path)
