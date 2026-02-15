from __future__ import annotations

from pathlib import Path

import awkward as ak
import h5py
import numpy as np

from truthjets.cluster import safe_eta
from truthjets.config import JetConfig
from truthjets.h5utils import H5_COMPRESSION

# Structured dtype for the /jets dataset
JET_DTYPE = np.dtype(
    [
        ("event_id", np.int64),
        ("pt", np.float32),
        ("eta", np.float32),
        ("phi", np.float32),
        ("mass", np.float32),
        ("energy", np.float32),
        ("HadronConeExclTruthLabelID", np.int32),
        ("n_constituents", np.int32),
        ("pt_frac_pu", np.float32),
    ]
)

# Structured dtype for the /constituents dataset
CONSTITUENT_DTYPE = np.dtype(
    [
        ("pt", np.float32),
        ("deta", np.float32),
        ("dphi", np.float32),
        ("energy", np.float32),
        ("pdgId", np.int32),
        ("is_pu", np.bool_),
        ("valid", np.bool_),
    ]
)


class HDF5Writer:
    """Streaming HDF5 writer for ftag-compatible jet output."""

    def __init__(
        self,
        path: str,
        jet_config: JetConfig,
        extra_jet_fields: list[tuple[str, np.dtype]] | None = None,
        extra_datasets: dict | None = None,
    ):
        self.path = path
        self.max_constituents = jet_config.max_constituents
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.file = h5py.File(path, "w")

        # Build jet dtype with any extra fields
        if extra_jet_fields:
            self.jet_dtype = np.dtype(
                JET_DTYPE.descr + [(name, dt) for name, dt in extra_jet_fields]
            )
        else:
            self.jet_dtype = JET_DTYPE

        # Create resizable datasets
        self.jets_ds = self.file.create_dataset(
            "jets",
            shape=(0,),
            maxshape=(None,),
            dtype=self.jet_dtype,
            chunks=(1000,),
            **H5_COMPRESSION,
        )
        self.constit_ds = self.file.create_dataset(
            "constituents",
            shape=(0, self.max_constituents),
            maxshape=(None, self.max_constituents),
            dtype=CONSTITUENT_DTYPE,
            chunks=(1000, self.max_constituents),
            **H5_COMPRESSION,
        )

        # Create extra datasets from modules
        self._extra_ds = {}
        if extra_datasets:
            for ds_name, schema in extra_datasets.items():
                shape = (0,) + schema.shape_suffix
                maxshape = (None,) + schema.shape_suffix
                chunks = (1000,) + schema.shape_suffix
                self._extra_ds[ds_name] = self.file.create_dataset(
                    ds_name,
                    shape=shape,
                    maxshape=maxshape,
                    dtype=schema.dtype,
                    chunks=chunks,
                    **H5_COMPRESSION,
                )

        self._n_jets = 0

    def write_batch(
        self,
        jet_kin,
        labels,
        constituents,
        jet_eta,
        jet_phi,
        event_offset: int = 0,
        extra_jet_data: dict | None = None,
        extra_dataset_data: dict | None = None,
    ):
        """Write a batch of jets and constituents to HDF5.

        Parameters
        ----------
        jet_kin : ak.Array
            Jet kinematics with fields {pt, eta, phi, mass, energy}.
        labels : ak.Array
            Integer truth labels per jet (events x jets).
        constituents : ak.Array
            Constituent particles per jet (events x jets x constituents)
            with fields {px, py, pz, E, pdgId}.
        jet_eta : ak.Array
            Jet eta (events x jets), used for deta computation.
        jet_phi : ak.Array
            Jet phi (events x jets), used for dphi computation.
        """
        # Compute event IDs: broadcast event index to each jet
        n_jets_per_event = ak.num(jet_kin.pt)
        event_indices = ak.broadcast_arrays(
            ak.local_index(n_jets_per_event) + event_offset,
            jet_kin.pt,
        )[0]
        flat_event_id = ak.to_numpy(ak.flatten(event_indices)).astype(np.int64)

        # Flatten from (events x jets) to flat jet list
        flat_pt = ak.to_numpy(ak.flatten(jet_kin.pt)).astype(np.float32)
        flat_eta = ak.to_numpy(ak.flatten(jet_kin.eta)).astype(np.float32)
        flat_phi = ak.to_numpy(ak.flatten(jet_kin.phi)).astype(np.float32)
        flat_mass = ak.to_numpy(ak.flatten(jet_kin.mass)).astype(np.float32)
        flat_energy = ak.to_numpy(ak.flatten(jet_kin.energy)).astype(np.float32)
        flat_labels = ak.to_numpy(ak.flatten(labels)).astype(np.int32)

        n_new = len(flat_pt)
        if n_new == 0:
            return

        # Build padded constituent arrays
        constit_array = self._pad_constituents(
            constituents, jet_eta, jet_phi
        )

        # Count constituents per jet (use .px field to count list entries)
        flat_nconstit = ak.to_numpy(
            ak.flatten(ak.num(constituents.px, axis=-1))
        ).astype(np.int32)
        flat_nconstit = np.minimum(flat_nconstit, self.max_constituents)

        # Compute pt_frac_pu per jet
        flat_constit_ak = ak.flatten(constituents, axis=1)
        c_px = flat_constit_ak.px
        c_py = flat_constit_ak.py
        c_pt_all = np.sqrt(c_px**2 + c_py**2)

        # Check if is_pu field exists on constituents
        if "is_pu" in ak.fields(flat_constit_ak):
            c_is_pu = flat_constit_ak.is_pu
            pt_pu = ak.sum(ak.where(c_is_pu, c_pt_all, 0.0), axis=-1)
        else:
            pt_pu = ak.zeros_like(ak.sum(c_pt_all, axis=-1))
        pt_total = ak.sum(c_pt_all, axis=-1)
        pt_frac_pu = ak.where(pt_total > 0, pt_pu / pt_total, 0.0)
        flat_pt_frac_pu = ak.to_numpy(pt_frac_pu).astype(np.float32)

        # Build structured jet array
        jet_array = np.zeros(n_new, dtype=self.jet_dtype)
        jet_array["event_id"] = flat_event_id
        jet_array["pt"] = flat_pt
        jet_array["eta"] = flat_eta
        jet_array["phi"] = flat_phi
        jet_array["mass"] = flat_mass
        jet_array["energy"] = flat_energy
        jet_array["HadronConeExclTruthLabelID"] = flat_labels
        jet_array["n_constituents"] = flat_nconstit
        jet_array["pt_frac_pu"] = flat_pt_frac_pu

        # Fill extra jet columns from modules
        if extra_jet_data:
            for field_name, values in extra_jet_data.items():
                if isinstance(values, np.ndarray):
                    flat_values = values
                else:
                    flat_values = ak.to_numpy(ak.flatten(values))
                jet_array[field_name] = flat_values.astype(
                    jet_array[field_name].dtype
                )

        # Extend datasets
        self.jets_ds.resize(self._n_jets + n_new, axis=0)
        self.constit_ds.resize(self._n_jets + n_new, axis=0)
        self.jets_ds[self._n_jets : self._n_jets + n_new] = jet_array
        self.constit_ds[self._n_jets : self._n_jets + n_new] = constit_array

        # Write extra datasets from modules
        if extra_dataset_data:
            for ds_name, data in extra_dataset_data.items():
                ds = self._extra_ds[ds_name]
                ds.resize(self._n_jets + n_new, axis=0)
                if isinstance(data, np.ndarray):
                    flat_data = data
                else:
                    flat_data = ak.to_numpy(ak.flatten(data))
                ds[self._n_jets : self._n_jets + n_new] = flat_data

        self._n_jets += n_new

    def _pad_constituents(self, constituents, jet_eta, jet_phi):
        """Pad and format constituents to fixed-size structured array.

        Constituents are sorted by pT descending, truncated/padded to
        max_constituents, with relative deta/dphi coordinates.
        """
        max_c = self.max_constituents

        # Flatten to (flat_jets x var_constituents)
        flat_constit = ak.flatten(constituents, axis=1)
        flat_jet_eta = ak.flatten(jet_eta)
        flat_jet_phi = ak.flatten(jet_phi)

        # Compute constituent pT for sorting
        c_px = flat_constit.px
        c_py = flat_constit.py
        c_pz = flat_constit.pz
        c_E = flat_constit.E
        c_pt = np.sqrt(c_px**2 + c_py**2)

        # Sort by pT descending
        sort_idx = ak.argsort(c_pt, ascending=False)
        flat_constit = flat_constit[sort_idx]
        c_pt = c_pt[sort_idx]

        # Recompute after sorting
        c_px = flat_constit.px
        c_py = flat_constit.py
        c_pz = flat_constit.pz
        c_E = flat_constit.E
        c_pdgid = flat_constit.pdgId
        c_is_pu = (
            flat_constit.is_pu
            if "is_pu" in ak.fields(flat_constit)
            else ak.zeros_like(c_pdgid, dtype=np.bool_)
        )
        c_pt = np.sqrt(c_px**2 + c_py**2)

        # Compute eta, phi for constituents
        c_p = np.sqrt(c_px**2 + c_py**2 + c_pz**2)
        c_eta = safe_eta(c_pz, c_p)
        c_phi = np.arctan2(c_py, c_px)

        # Compute deta, dphi relative to jet axis
        # Broadcast jet-level values to constituent level
        j_eta_bcast = flat_jet_eta * ak.ones_like(c_eta)
        j_phi_bcast = flat_jet_phi * ak.ones_like(c_phi)
        deta = c_eta - j_eta_bcast
        dphi = (c_phi - j_phi_bcast + np.pi) % (2 * np.pi) - np.pi

        # Pad to max_constituents
        c_pt_pad = ak.fill_none(ak.pad_none(c_pt, max_c, clip=True), 0.0)
        deta_pad = ak.fill_none(ak.pad_none(deta, max_c, clip=True), 0.0)
        dphi_pad = ak.fill_none(ak.pad_none(dphi, max_c, clip=True), 0.0)
        c_E_pad = ak.fill_none(ak.pad_none(c_E, max_c, clip=True), 0.0)
        c_pdgid_pad = ak.fill_none(
            ak.pad_none(c_pdgid, max_c, clip=True), 0
        )
        c_is_pu_pad = ak.fill_none(
            ak.pad_none(c_is_pu, max_c, clip=True), False
        )

        # Valid mask: True where original constituent existed
        valid = ak.fill_none(
            ak.pad_none(ak.ones_like(c_pt, dtype=np.bool_), max_c, clip=True),
            False,
        )

        # Convert to numpy
        n_jets = len(flat_constit)
        out = np.zeros((n_jets, max_c), dtype=CONSTITUENT_DTYPE)
        out["pt"] = ak.to_numpy(c_pt_pad).astype(np.float32)
        out["deta"] = ak.to_numpy(deta_pad).astype(np.float32)
        out["dphi"] = ak.to_numpy(dphi_pad).astype(np.float32)
        out["energy"] = ak.to_numpy(c_E_pad).astype(np.float32)
        out["pdgId"] = ak.to_numpy(c_pdgid_pad).astype(np.int32)
        out["is_pu"] = ak.to_numpy(c_is_pu_pad)
        out["valid"] = ak.to_numpy(valid)

        return out

    def close(self):
        self.file.close()

    @property
    def n_jets(self):
        return self._n_jets

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
