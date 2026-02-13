from __future__ import annotations

from truthjets.config import JetConfig
from truthjets.label import label_jets
from truthjets.modules import ModuleResult, TruthJetModule


class HadronConeExclLabelModule(TruthJetModule):
    """Labels jets using dR-matched HadronConeExclTruthLabelID.

    Matches b-hadrons, c-hadrons, and tau leptons to jets within
    a cone of radius R. Priority: b > c > tau > light.

    This module is automatically loaded for R=0.4 jets.
    """

    def init(self, jet_config: JetConfig) -> None:
        self.R = jet_config.R

    def post_clustering(self, events, jets, constituents, jet_kin, labels):
        new_labels = label_jets(events, jet_kin.eta, jet_kin.phi, self.R)
        return ModuleResult(labels=new_labels)
