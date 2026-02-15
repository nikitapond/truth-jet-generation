from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from truthjets.config import JetConfig
from truthjets.modules import (
    BUILTIN_MODULES,
    TruthJetModule,
    deduplicate_modules,
    load_module,
    load_modules_from_yaml,
    resolve_module_specs,
)
from truthjets.modules.pileup_rejection import SoftKillerModule, VertexZFilterModule


class TestBuiltinModules:
    """Test the BUILTIN_MODULES registry."""

    def test_has_expected_entries(self):
        expected = {"softkiller", "vertexzfilter", "hadronconelabel", "largerlabel"}
        assert set(BUILTIN_MODULES.keys()) == expected

    def test_all_values_are_importable(self):
        for name, path in BUILTIN_MODULES.items():
            mod = load_module(path)
            assert isinstance(mod, TruthJetModule), (
                f"Built-in '{name}' did not produce a TruthJetModule"
            )


class TestLoadModuleInitArgs:
    """Test load_module() with init_args parameter."""

    def test_without_init_args(self):
        mod = load_module("truthjets.modules.pileup_rejection:SoftKillerModule")
        assert isinstance(mod, SoftKillerModule)
        assert mod._grid_size is None

    def test_with_init_args(self):
        mod = load_module(
            "truthjets.modules.pileup_rejection:SoftKillerModule",
            init_args={"grid_size": 0.6},
        )
        assert isinstance(mod, SoftKillerModule)
        assert mod._grid_size == 0.6

    def test_auto_discover_with_init_args(self):
        # VertexZFilterModule is in a module with multiple subclasses,
        # so use explicit class_path
        mod = load_module(
            "truthjets.modules.pileup_rejection:VertexZFilterModule",
            init_args={"max_dz": 3.0},
        )
        assert isinstance(mod, VertexZFilterModule)
        assert mod._max_dz == 3.0


class TestLoadModulesFromYaml:
    """Test YAML module config loading."""

    def test_valid_yaml(self, tmp_path):
        yaml_file = tmp_path / "modules.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            - class_path: truthjets.modules.pileup_rejection:SoftKillerModule
              init_args:
                grid_size: 0.6

            - class_path: truthjets.modules.pileup_rejection:VertexZFilterModule
              init_args:
                max_dz: 3.0
        """))

        modules = load_modules_from_yaml(yaml_file)
        assert len(modules) == 2
        assert isinstance(modules[0], SoftKillerModule)
        assert modules[0]._grid_size == 0.6
        assert isinstance(modules[1], VertexZFilterModule)
        assert modules[1]._max_dz == 3.0

    def test_valid_yaml_no_init_args(self, tmp_path):
        yaml_file = tmp_path / "modules.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            - class_path: truthjets.modules.pileup_rejection:SoftKillerModule
        """))

        modules = load_modules_from_yaml(yaml_file)
        assert len(modules) == 1
        assert isinstance(modules[0], SoftKillerModule)
        assert modules[0]._grid_size is None

    def test_not_a_list(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("class_path: something\n")

        with pytest.raises(ValueError, match="must contain a list"):
            load_modules_from_yaml(yaml_file)

    def test_missing_class_path(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            - init_args:
                grid_size: 0.6
        """))

        with pytest.raises(ValueError, match="'class_path' key"):
            load_modules_from_yaml(yaml_file)

    def test_entry_not_a_dict(self, tmp_path):
        yaml_file = tmp_path / "bad.yaml"
        yaml_file.write_text("- just_a_string\n")

        with pytest.raises(ValueError, match="'class_path' key"):
            load_modules_from_yaml(yaml_file)

    def test_yml_extension(self, tmp_path):
        yaml_file = tmp_path / "modules.yml"
        yaml_file.write_text(textwrap.dedent("""\
            - class_path: truthjets.modules.pileup_rejection:SoftKillerModule
        """))

        modules = load_modules_from_yaml(yaml_file)
        assert len(modules) == 1


class TestResolveModuleSpecs:
    """Test resolve_module_specs() with mixed spec types."""

    def test_builtin_name(self):
        modules = resolve_module_specs(["softkiller"])
        assert len(modules) == 1
        assert isinstance(modules[0], SoftKillerModule)

    def test_builtin_name_case_insensitive(self):
        modules = resolve_module_specs(["SoftKiller"])
        assert len(modules) == 1
        assert isinstance(modules[0], SoftKillerModule)

    def test_import_path(self):
        modules = resolve_module_specs(
            ["truthjets.modules.pileup_rejection:VertexZFilterModule"]
        )
        assert len(modules) == 1
        assert isinstance(modules[0], VertexZFilterModule)

    def test_yaml_file(self, tmp_path):
        yaml_file = tmp_path / "mods.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            - class_path: truthjets.modules.pileup_rejection:SoftKillerModule
              init_args:
                grid_size: 0.8
        """))

        modules = resolve_module_specs([str(yaml_file)])
        assert len(modules) == 1
        assert isinstance(modules[0], SoftKillerModule)
        assert modules[0]._grid_size == 0.8

    def test_mixed_specs(self, tmp_path):
        yaml_file = tmp_path / "mods.yaml"
        yaml_file.write_text(textwrap.dedent("""\
            - class_path: truthjets.modules.pileup_rejection:VertexZFilterModule
              init_args:
                max_dz: 5.0
        """))

        modules = resolve_module_specs([
            "softkiller",
            str(yaml_file),
            "truthjets.modules.label:HadronConeExclLabelModule",
        ])
        assert len(modules) == 3
        assert isinstance(modules[0], SoftKillerModule)
        assert isinstance(modules[1], VertexZFilterModule)
        assert modules[1]._max_dz == 5.0

    def test_empty_specs(self):
        modules = resolve_module_specs([])
        assert modules == []


class TestDeduplicateModules:
    """Test deduplicate_modules()."""

    def test_no_duplicates(self):
        mods = [SoftKillerModule(), VertexZFilterModule()]
        result = deduplicate_modules(mods)
        assert len(result) == 2

    def test_removes_duplicates(self):
        mods = [SoftKillerModule(), VertexZFilterModule(), SoftKillerModule()]
        result = deduplicate_modules(mods)
        assert len(result) == 2
        assert isinstance(result[0], SoftKillerModule)
        assert isinstance(result[1], VertexZFilterModule)

    def test_first_occurrence_wins(self):
        first = SoftKillerModule(grid_size=0.4)
        second = SoftKillerModule(grid_size=0.8)
        result = deduplicate_modules([first, second])
        assert len(result) == 1
        assert result[0] is first
        assert result[0]._grid_size == 0.4

    def test_empty_list(self):
        assert deduplicate_modules([]) == []


class TestPileupRejectionOverrides:
    """Test that SoftKillerModule and VertexZFilterModule respect init overrides."""

    def _make_jet_config(self, **kwargs):
        config = JetConfig()
        for k, v in kwargs.items():
            setattr(config, k, v)
        return config

    def test_softkiller_default(self):
        mod = SoftKillerModule()
        config = self._make_jet_config(softkiller_grid=0.4)
        mod.init(config)
        assert mod.grid_size == 0.4

    def test_softkiller_override(self):
        mod = SoftKillerModule(grid_size=0.6)
        config = self._make_jet_config(softkiller_grid=0.4)
        mod.init(config)
        assert mod.grid_size == 0.6

    def test_vertexzfilter_default(self):
        mod = VertexZFilterModule()
        config = self._make_jet_config(max_dz=2.0)
        mod.init(config)
        assert mod.max_dz == 2.0

    def test_vertexzfilter_override(self):
        mod = VertexZFilterModule(max_dz=3.0)
        config = self._make_jet_config(max_dz=2.0)
        mod.init(config)
        assert mod.max_dz == 3.0
