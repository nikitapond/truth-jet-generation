from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from truthjets.config import JetConfig

# Registry of built-in module short names -> import paths
BUILTIN_MODULES: dict[str, str] = {
    "softkiller": "truthjets.modules.pileup_rejection:SoftKillerModule",
    "vertexzfilter": "truthjets.modules.pileup_rejection:VertexZFilterModule",
    "hadronconelabel": "truthjets.modules.label:HadronConeExclLabelModule",
    "largerlabel": "truthjets.modules.label:LargeRLabelModule",
}


@dataclass
class DatasetSchema:
    """Schema for an extra HDF5 dataset created by a module.

    Parameters
    ----------
    dtype : np.dtype
        The numpy dtype for the dataset.
    shape_suffix : tuple[int, ...]
        Dimensions appended after the leading (n_jets,) axis.
        For a flat per-jet array, use () (the default).
        For e.g. a (n_jets, 10) array, use (10,).
    """

    dtype: np.dtype
    shape_suffix: tuple[int, ...] = ()


@dataclass
class ModuleResult:
    """Return type from TruthJetModule.post_clustering.

    Any non-None field replaces the corresponding pipeline variable.
    Extra data dicts are merged across modules.
    """

    jet_kin: object = None
    labels: object = None
    jets: object = None
    constituents: object = None
    extra_jet_data: dict = field(default_factory=dict)
    extra_dataset_data: dict = field(default_factory=dict)


class TruthJetModule:
    """Base class for pipeline modules.

    Subclass this and override the methods you need. All methods have
    default no-op implementations so you only need to implement what
    you use.
    """

    def init(self, jet_config: JetConfig) -> None:
        """Called once after loading, before any events are processed."""

    def pre_clustering(self, events, particles):
        """Called before jet clustering.

        Parameters
        ----------
        events : ak.Array
            Raw Pythia event batch.
        particles : ak.Array
            Particle arrays (events x particles) with fields
            {px, py, pz, E, pdgId, is_pu}.

        Returns
        -------
        ak.Array or None
            Modified particles, or None to keep unchanged.
        """
        return None

    def post_clustering(self, events, jets, constituents, jet_kin, labels):
        """Called after jet clustering and labeling.

        Parameters
        ----------
        events : ak.Array
            Raw Pythia event batch.
        jets : ak.Array
            Clustered jets (events x jets).
        constituents : ak.Array
            Jet constituents (events x jets x constituents).
        jet_kin : ak.Array
            Jet kinematics with fields {pt, eta, phi, mass, energy}.
        labels : ak.Array
            Integer truth labels per jet (events x jets).

        Returns
        -------
        ModuleResult or None
            Result with mutations and/or extra data, or None.
        """
        return None

    def extra_jet_fields(self) -> list[tuple[str, np.dtype]]:
        """Declare extra columns to add to the /jets dataset.

        Returns
        -------
        list of (name, dtype) tuples
            e.g. [("bb_angle", np.float32)]
        """
        return []

    def extra_datasets(self) -> dict[str, DatasetSchema]:
        """Declare extra HDF5 datasets to create.

        Returns
        -------
        dict mapping dataset name to DatasetSchema
        """
        return {}


def load_module(
    import_path: str,
    init_args: dict | None = None,
) -> TruthJetModule:
    """Load a TruthJetModule from a dotted import path.

    Supports two forms:
    - ``my_package.my_module`` — auto-discovers the single TruthJetModule
      subclass defined in that module.
    - ``my_package.my_module:ClassName`` — loads a specific class.

    Parameters
    ----------
    import_path : str
        Dotted Python import path, optionally with ``:ClassName`` suffix.
    init_args : dict or None
        If provided, passed as kwargs to the module constructor.

    Returns
    -------
    TruthJetModule
        An instantiated module.
    """
    kwargs = init_args or {}

    if ":" in import_path:
        module_path, class_name = import_path.rsplit(":", 1)
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name, None)
        if cls is None:
            raise ImportError(
                f"Class '{class_name}' not found in '{module_path}'"
            )
        if not (inspect.isclass(cls) and issubclass(cls, TruthJetModule)):
            raise TypeError(
                f"'{class_name}' is not a TruthJetModule subclass"
            )
        return cls(**kwargs)

    mod = importlib.import_module(import_path)
    subclasses = [
        obj
        for _, obj in inspect.getmembers(mod, inspect.isclass)
        if issubclass(obj, TruthJetModule)
        and obj is not TruthJetModule
        and obj.__module__ == mod.__name__
    ]

    if len(subclasses) == 0:
        raise ImportError(
            f"No TruthJetModule subclass found in '{import_path}'"
        )
    if len(subclasses) > 1:
        names = [c.__name__ for c in subclasses]
        raise ImportError(
            f"Multiple TruthJetModule subclasses found in '{import_path}': "
            f"{names}. Use '{import_path}:ClassName' to specify one."
        )
    return subclasses[0](**kwargs)


def load_modules_from_yaml(path: str | Path) -> list[TruthJetModule]:
    """Load modules from a YAML config file.

    The YAML file must contain a list of entries, each with a ``class_path``
    key and an optional ``init_args`` dict.

    Parameters
    ----------
    path : str or Path
        Path to the YAML file.

    Returns
    -------
    list[TruthJetModule]
        Instantiated modules.

    Raises
    ------
    ValueError
        If the YAML content is not a list or entries are missing ``class_path``.
    """
    path = Path(path)
    with open(path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, list):
        raise ValueError(
            f"YAML module config '{path}' must contain a list of module "
            f"entries, got {type(data).__name__}"
        )

    modules = []
    for i, entry in enumerate(data):
        if not isinstance(entry, dict) or "class_path" not in entry:
            raise ValueError(
                f"Entry {i} in '{path}' must be a dict with a 'class_path' key, "
                f"got: {entry!r}"
            )
        class_path = entry["class_path"]
        init_args = entry.get("init_args")
        modules.append(load_module(class_path, init_args=init_args))

    return modules


def resolve_module_specs(specs: list[str]) -> list[TruthJetModule]:
    """Resolve a list of module specs to instantiated modules.

    Each spec is resolved in order:
    1. If it matches a key in ``BUILTIN_MODULES`` (case-insensitive),
       load that class with default args.
    2. If it ends with ``.yaml`` or ``.yml``, parse as a YAML config.
    3. Otherwise, treat as an import path (``load_module()`` behavior).

    Parameters
    ----------
    specs : list[str]
        Mixed list of built-in names, YAML file paths, or import paths.

    Returns
    -------
    list[TruthJetModule]
        Instantiated modules.
    """
    modules = []
    for spec in specs:
        builtin_key = spec.lower()
        if builtin_key in BUILTIN_MODULES:
            modules.append(load_module(BUILTIN_MODULES[builtin_key]))
        elif spec.endswith((".yaml", ".yml")):
            modules.extend(load_modules_from_yaml(spec))
        else:
            modules.append(load_module(spec))
    return modules


def deduplicate_modules(modules: list[TruthJetModule]) -> list[TruthJetModule]:
    """Remove duplicate modules by class identity (first occurrence wins).

    Parameters
    ----------
    modules : list[TruthJetModule]
        List of module instances, possibly with duplicates.

    Returns
    -------
    list[TruthJetModule]
        Deduplicated list preserving order.
    """
    seen: set[type] = set()
    result = []
    for mod in modules:
        cls = type(mod)
        if cls not in seen:
            seen.add(cls)
            result.append(mod)
    return result


def validate_modules(modules: list[TruthJetModule]) -> None:
    """Validate that modules don't conflict with each other or built-ins.

    Raises ValueError if duplicate field/dataset names are found or if
    a module tries to shadow built-in names.
    """
    from truthjets.writer import JET_DTYPE

    builtin_jet_fields = set(JET_DTYPE.names)
    builtin_datasets = {"jets", "constituents"}

    seen_fields: dict[str, str] = {}  # field_name -> module class name
    seen_datasets: dict[str, str] = {}  # dataset_name -> module class name

    for mod in modules:
        mod_name = type(mod).__name__

        for field_name, _ in mod.extra_jet_fields():
            if field_name in builtin_jet_fields:
                raise ValueError(
                    f"Module '{mod_name}' declares extra jet field "
                    f"'{field_name}' which conflicts with a built-in field"
                )
            if field_name in seen_fields:
                raise ValueError(
                    f"Module '{mod_name}' declares extra jet field "
                    f"'{field_name}' which is already declared by "
                    f"'{seen_fields[field_name]}'"
                )
            seen_fields[field_name] = mod_name

        for ds_name in mod.extra_datasets():
            if ds_name in builtin_datasets:
                raise ValueError(
                    f"Module '{mod_name}' declares extra dataset "
                    f"'{ds_name}' which conflicts with a built-in dataset"
                )
            if ds_name in seen_datasets:
                raise ValueError(
                    f"Module '{mod_name}' declares extra dataset "
                    f"'{ds_name}' which is already declared by "
                    f"'{seen_datasets[ds_name]}'"
                )
            seen_datasets[ds_name] = mod_name


# Re-export label module classes for convenience
from truthjets.modules.label import (  # noqa: E402, F401
    HadronConeExclLabelModule,
    LargeRLabelModule,
    label_large_r_jets,
)
from truthjets.modules.pileup_rejection import (  # noqa: E402, F401
    SoftKillerModule,
    VertexZFilterModule,
)
