from __future__ import annotations

from truthjets.config import JetConfig
from truthjets.modules import ModuleResult, TruthJetModule
from truthjets.pileup_rejection import softkiller, vertex_z_filter


class SoftKillerModule(TruthJetModule):
    """Applies SoftKiller pileup mitigation before jet clustering.

    Wraps :func:`truthjets.pileup_rejection.softkiller`.
    Auto-loaded when ``--softkiller`` is set and pileup is active.

    Parameters
    ----------
    grid_size : float or None
        If set, overrides ``jet_config.softkiller_grid`` in ``init()``.
    """

    def __init__(self, grid_size: float | None = None) -> None:
        self._grid_size = grid_size

    def init(self, jet_config: JetConfig) -> None:
        self.grid_size = self._grid_size if self._grid_size is not None else jet_config.softkiller_grid

    def pre_clustering(self, events, particles):
        if particles is None:
            return None
        return softkiller(particles, grid_size=self.grid_size)


class VertexZFilterModule(TruthJetModule):
    """Rejects jets with large pT-weighted mean vertex z after clustering.

    Wraps :func:`truthjets.pileup_rejection.vertex_z_filter`.
    Auto-loaded when ``--max-dz`` is set and pileup is active.

    Parameters
    ----------
    max_dz : float or None
        If set, overrides ``jet_config.max_dz`` in ``init()``.
    """

    def __init__(self, max_dz: float | None = None) -> None:
        self._max_dz = max_dz

    def init(self, jet_config: JetConfig) -> None:
        self.max_dz = self._max_dz if self._max_dz is not None else jet_config.max_dz

    def post_clustering(self, events, jets, constituents, jet_kin, labels):
        mask = vertex_z_filter(constituents, jet_kin, self.max_dz)
        return ModuleResult(
            jets=jets[mask],
            constituents=constituents[mask],
            jet_kin=jet_kin[mask],
            labels=labels[mask],
        )
