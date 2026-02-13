from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class PythiaConfig:
    process: str | None = None
    pythia_card: str | None = None
    ecm: float = 13600.0
    pt_hat_min: float | None = None
    pt_hat_max: float | None = None
    seed: int = 42
    mu: float | None = None
    pu_pre_gen: int | None = None
    pu_file: str | None = None
    extra_settings: list[str] = field(default_factory=list)


@dataclass
class JetConfig:
    algorithm: str = "antikt"
    R: float = 0.4
    pt_min: float = 20.0
    eta_max: float = 2.5
    max_constituents: int = 80
    softkiller: bool = False
    softkiller_grid: float = 0.4
    constituent_pt_min: float = 0.5  # GeV, min pT for constituents
    max_dz: float | None = None  # mm, vertex z cut for PU rejection


@dataclass
class OutputConfig:
    output_path: str = "jets.h5"
    n_events: int = 100_000
    batch_size: int = 10_000


# Maps process name -> list of Pythia readString commands
PROCESS_PRESETS: dict[str, list[str]] = {
    "qcd": [
        "HardQCD:all = on",
    ],
    "ttbar": [
        "Top:gg2ttbar = on",
        "Top:qqbar2ttbar = on",
        "6:m0 = 172.5",
    ],
    "zprime_tt": [
        "NewGaugeBoson:ffbar2gmZZprime = on",
        "Zprime:gmZmode = 3",
        "32:m0 = 3000",
        "32:onIfAny = 6",
    ],
}


def load_pythia_config(path: str | Path) -> PythiaConfig:
    """Load PythiaConfig from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return PythiaConfig(
        process=data.get("process"),
        pythia_card=data.get("pythia_card"),
        ecm=data.get("ecm", 13600.0),
        pt_hat_min=data.get("pt_hat_min"),
        pt_hat_max=data.get("pt_hat_max"),
        seed=data.get("seed", 42),
        mu=data.get("mu"),
        pu_pre_gen=data.get("pu_pre_gen"),
        pu_file=data.get("pu_file"),
        extra_settings=data.get("extra_settings", []),
    )


def load_jet_and_output_config(
    path: str | Path,
) -> tuple[JetConfig, OutputConfig]:
    """Load JetConfig and OutputConfig from a YAML file."""
    with open(path) as f:
        data = yaml.safe_load(f)
    jet_data = data.get("jet", {})
    out_data = data.get("output", {})
    jet_config = JetConfig(
        algorithm=jet_data.get("algorithm", "antikt"),
        R=jet_data.get("R", 0.4),
        pt_min=jet_data.get("pt_min", 20.0),
        eta_max=jet_data.get("eta_max", 2.5),
        max_constituents=jet_data.get("max_constituents", 80),
        constituent_pt_min=jet_data.get("constituent_pt_min", 0.5),
        softkiller=jet_data.get("softkiller", False),
        softkiller_grid=jet_data.get("softkiller_grid", 0.4),
        max_dz=jet_data.get("max_dz"),
    )
    output_config = OutputConfig(
        output_path=out_data.get("path", "jets.h5"),
        n_events=out_data.get("n_events", 100_000),
        batch_size=out_data.get("batch_size", 10_000),
    )
    return jet_config, output_config
