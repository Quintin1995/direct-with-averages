# coding=utf-8
# Copyright (c) DIRECT Contributors
import logging
import pathlib
import re
from typing import Any, Dict, List, Optional, Tuple, Union
import sqlite3

from pathlib import Path
import pandas as pd

import h5py
import numpy as np
from torch.utils.data import Dataset

from direct.types import PathOrString
from direct.utils import cast_as_path
from direct.utils.dataset import get_filenames_for_datasets

logger = logging.getLogger(__name__)


class H5WithAvgsSliceData(Dataset):
    """A PyTorch Dataset class which outputs k-space slices based on the h5 dataformat.
       However, this class assumes that the averages are at the first axis of the kspace.
       Like the prostate NYU dataset with 3 averages as the first dimension.
       
       Combine the three averages by taking the average of avg1 and avg3 (odd lines) and adding avg2 (even lines)
       full_kspace = (avg1 + avg3) / 2 + avg2
    """

    def __init__(
        self,
        root: pathlib.Path,
        filenames_filter: Union[List[PathOrString], None]      = None,
        filenames_lists: Union[List[PathOrString], None]       = None,
        filenames_lists_root: Union[PathOrString, None]        = None,
        regex_filter: Optional[str]                            = None,
        dataset_description: Optional[Dict[PathOrString, Any]] = None,
        metadata: Optional[Dict[PathOrString, Dict]]           = None,
        sensitivity_maps: Optional[PathOrString]               = None,
        extra_keys: Optional[Tuple]                            = None,
        pass_attrs: bool                                       = False,
        text_description: Optional[str]                        = None,
        kspace_context: Optional[int]                          = None,
        pass_dictionaries: Optional[Dict[str, Dict]]           = None,
        pass_h5s: Optional[Dict[str, List]]                    = None,
        slice_data: Optional[slice]                            = None,
        compute_mask: bool                                     = False,
        store_applied_acs_mask: bool                           = True,
        avg_collapse_strat: str                                = None,
        avg_acceleration: int                                  = None,
        add_gaussian_noise: bool                               = False,
        noise_mult: float                                      = 2.5,
        db_path: Optional[str]                                 = None,
        tablename: Optional[str]                               = None,
        do_lxo_for_uq: bool                                    = False,         # If True, apply fold_idx dropout for Uncertainty Quantification
        echo_train_acceleration: int                           = 1,            # Acceleration factor; 1 means no acceleration
        echo_train_fold_idx: int                               = 0,            # Index of ET(s) to leave out from retained set
    ) -> None:
        """Initialize the dataset.

        Parameters
        ----------
        root: pathlib.Path
            Root directory to data.
        filenames_filter: Union[List[PathOrString], None]
            List of filenames to include in the dataset, should be the same as the ones that can be derived from a glob
            on the root. If set, will skip searching for files in the root. Default: None.
        filenames_lists: Union[List[PathOrString], None]
            List of paths pointing to `.lst` file(s) that contain file-names in `root` to filter.
            Should be the same as the ones that can be derived from a glob on the root. If this is set,
            this will override the `filenames_filter` option if not None. Defualt: None.
        filenames_lists_root: Union[PathOrString, None]
            Root of `filenames_lists`. Ignored if `filename_lists` is None. Default: None.
        regex_filter: str
            Regular expression filter on the absolute filename. Will be applied after any filenames filter.
        metadata: dict
            If given, this dictionary will be passed to the output transform.
        sensitivity_maps: [pathlib.Path, None]
            Path to sensitivity maps, or None.
        extra_keys: Tuple
            Add extra keys in h5 file to output.
        pass_attrs: bool
            Pass the attributes saved in the h5 file.
        text_description: str
            Description of dataset, can be useful for logging.
        pass_dictionaries: dict
            Pass a dictionary of dictionaries, e.g. if {"name": {"filename_0": val}}, then to `filename_0`s sample dict,
            a key with name `name` and value `val` will be added.
        pass_h5s: dict
            Pass a dictionary of paths. If {"name": path} is given then to the sample of `filename` the same slice
            of path / filename will be added to the sample dictionary and will be asigned key `name`. This can first
            instance be convenient when you want to pass sensitivity maps as well. So for instance:
        avg_collapse_strat: str
            DIRECT does not include any strategy to collapse the averages and remove the average dimension. So we must implement a strategy for this
            if we want to use the dataset with a model that does not support the average dimension. This can be done by setting the `avg_collapse_strat`
            parameter to a string that describes the strategy to collapse the averages. Default: None.     # options: avg1, avg2, avg3, allavg       #[allavg=(avg1+avg3)/2+avg2)]
            avg1: simply takes the first average only
            avg2: takes the second average only
            avg3: takes the third average only
            allavg: takes the average of avg1 and avg3 and adds avg2, since avg1 and avg3 measure odd lines and avg2 measures even lines we geta  full kspace
            full_kspace = (avg1 + avg3) / 2 + avg2
        avg_acceleration: int
            The acceleration factor when considering the number of averages and the parallel imaging factor. For example can be 6. Then 1 out of 3 averages is measured and the acceleration factor is 2.
        add_gaussian_noise: bool
            Add gaussian noise to the k-space data. Default: False.
        compute_mask: bool
            Compute the mask for the ACS region. Default: False.
        store_applied_acs_mask: bool
            >>> pass_h5s = {"sensitivity_map": "/data/sensitivity_maps"}
            will add to each output sample a key `sensitivity_map` with value a numpy array containing the same slice
            of /data/sensitivity_maps/filename.h5 as the one of the original filename filename.h5.
        slice_data : Optional[slice]
            If set, for instance to slice(50,-50) only data within this slide will be added to the dataset. This
            is for instance convenient in the validation set of the public Calgary-Campinas dataset as the first 50
            and last 50 slices are excluded in the evaluation.
        """
        self.logger                                    = logging.getLogger(type(self).__name__)
        self.root                                      = pathlib.Path(root)
        self.filenames_filter                          = filenames_filter
        self.metadata                                  = metadata
        self.dataset_description                       = dataset_description
        self.text_description                          = text_description
        self.data: List[Tuple]                         = []
        self.volume_indices: Dict[pathlib.Path, range] = {}

        # QVL Class specific parameters
        self.compute_mask                              = compute_mask              # collapse avg - Compute the mask for the ACS region
        self.store_applied_acs_mask                    = store_applied_acs_mask    # collapse avg - Store the applied ACS mask in the dataset 
        self.average_collapse_strat                    = avg_collapse_strat        # collapse avg - Strategy to collapse the averages and remove the average dimension
        self.avg_acceleration                          = avg_acceleration          # collapse avg - Acceleration factor when considering the number of averages and the parallel imaging factor. For example can be 6. Then 1 out of 3 averages is measured and the acceleration factor is 2.
        self.add_gaussian_noise                        = add_gaussian_noise        # Gaus - Add gaussian noise to the k-space data, now add noise_mult
        self.noise_mult                                = noise_mult                # Gaus - Fraction of noise to add to the k-space data
        self.db_path                                   = db_path                   # Gaus - Path to the database to use for the uncertainty quantification
        self.tablename_uq                              = tablename                 # Gaus - The tablename to use for the uncertainty quantification
        self.tablename_noise                           = "noise_estimation"        # Gaus - Table in the DB where the noise levels for each patient for each coil are stored.
        self.do_lxo_for_uq                             = do_lxo_for_uq             # LXO - If True, apply fold_idx dropout for Uncertainty Quantification
        self.echo_train_acceleration                   = echo_train_acceleration   # LXO - Acceleration factor when dropping a factor of the echo trains per average. 2 means that 1/2 echo trains are dropped entirely (masked out)
        self.echo_train_fold_idx                       = echo_train_fold_idx       # LXO - Index of ET(s) to leave out from retained set
        self.et_mapping_tablename                      = "echo_train_mapping"  # Table in the DB where the echo train mapping for each patient for each coil are stored.
        self.db_path = Path("/home1/p290820/repos/Uncertainty-Quantification-Prostate-MRI/databases/master_habrok_20231106_v2.db")   # sorry this is hardcoded for now

        self.logger.info(f"Extra QvL settings: compute_mask (bool): {self.compute_mask}")
        self.logger.info(f"Extra QvL settings: store_applied_acs_mask (bool): {self.store_applied_acs_mask}")
        self.logger.info(f"Extra QvL settings: average_collapse_strat (str): {self.average_collapse_strat}")
        self.logger.info(f"Extra QvL settings: add_gaussian_noise (bool): {self.add_gaussian_noise}")
        self.logger.info(f"Kspace context: {kspace_context}. If 0 it means we only consider the corrent slice. if not we consider the slices around the slice.")
        self.logger.info(f"Extra QvL settings: do_lxo_for_uq (bool): {self.do_lxo_for_uq}")
        self.logger.info(f"Extra QvL settings: echo_train_acceleration (int): {self.echo_train_acceleration}")
        self.logger.info(f"Extra QvL settings: echo_train_fold_idx (int): {self.echo_train_fold_idx}")

        assert not (self.do_lxo_for_uq and self.add_gaussian_noise), "do_lxo_for_uq and add_gaussian_noise are not compatible. Please set one of them to False."

        if True:
            # If filenames_filter and filenames_lists are given, it will load files in filenames_filter
            # and filenames_lists will be ignored.
            if filenames_filter is None:
                if filenames_lists is not None:
                    if filenames_lists_root is None:
                        e = "`filenames_lists` is passed but `filenames_lists_root` is None."
                        self.logger.error(e)
                        raise ValueError(e)
                    filenames = get_filenames_for_datasets(
                        lists=filenames_lists, files_root=filenames_lists_root, data_root=root
                    )
                    self.logger.info("Attempting to load %s filenames from list(s).", len(filenames))
                else:
                    self.logger.info("Parsing directory %s for h5 files.", self.root)
                    filenames = list(self.root.glob("*.h5"))
            else:
                self.logger.info("Attempting to load %s filenames.", len(filenames_filter))
                filenames = filenames_filter

            filenames = [pathlib.Path(_) for _ in filenames]

            if regex_filter:
                filenames = [_ for _ in filenames if re.match(regex_filter, str(_))]

            if len(filenames) == 0:
                warn = (
                    f"Found 0 h5 files in directory {self.root}."
                    if not self.text_description
                    else f"Found 0 h5 files in directory {self.root} for dataset {self.text_description}."
                )
                self.logger.warning(warn)
            else:
                self.logger.info("Using %s h5 files in %s.", len(filenames), self.root)

            self.sensitivity_maps = cast_as_path(sensitivity_maps)

            self.parse_filenames_data(
                filenames, extra_h5s=pass_h5s, filter_slice=slice_data
            )  # Collect information on the image masks_dict.
            self.pass_h5s = pass_h5s

            self.pass_attrs = pass_attrs
            self.extra_keys = extra_keys
            self.pass_dictionaries = pass_dictionaries

            self.kspace_context = kspace_context if kspace_context else 0
            self.ndim = 2 if self.kspace_context == 0 else 3

            if self.text_description:
                self.logger.info("Dataset description: %s.", self.text_description)


    def parse_filenames_data(self, filenames, extra_h5s=None, filter_slice=None):
        current_slice_number = 0  # This is required to keep track of where a volume is in the dataset

        for idx, filename in enumerate(filenames):
            if len(filenames) < 5 or idx % (len(filenames) // 5) == 0 or len(filenames) == (idx + 1):
                self.logger.info(f"Parsing: {(idx + 1) / len(filenames) * 100:.2f}%.")
            try:
                kspace_shape = h5py.File(filename, "r")["kspace"].shape  # pylint: disable = E1101
                self.verify_extra_h5_integrity(filename, kspace_shape, extra_h5s=extra_h5s)  # pylint: disable = E1101
            except FileNotFoundError as exc:
                self.logger.warning("%s not found. Failed with: %s. Skipping...", filename, exc)
                continue
            except OSError as exc:
                self.logger.warning("%s failed with OSError: %s. Skipping...", filename, exc)
                continue

            if self.sensitivity_maps:
                try:
                    _ = h5py.File(self.sensitivity_maps / filename.name, "r")
                except FileNotFoundError as exc:
                    self.logger.warning(
                        "Sensitivity map %s not found. Failed with: %s. Skipping %s...",
                        self.sensitivity_maps / filename.name,
                        exc,
                        filename,
                    )
                    continue
                except OSError as exc:
                    self.logger.warning(
                        "Sensitivity map %s failed with OSError: %s. Skipping %s...",
                        self.sensitivity_maps / filename.name,
                        exc,
                        filename,
                    )
                    continue

            # In this case we assume the averages of kspace are at the first axis.
            num_slices = kspace_shape[1]
            
            if not filter_slice:
                self.data += [(filename, _) for _ in range(num_slices)]

            elif isinstance(filter_slice, slice):
                admissible_indices = range(*filter_slice.indices(num_slices))
                self.data += [(filename, _) for _ in range(num_slices) if _ in admissible_indices]
                num_slices = len(admissible_indices)

            else:
                raise NotImplementedError

            self.volume_indices[filename] = range(current_slice_number, current_slice_number + num_slices)

            current_slice_number += num_slices

    @staticmethod
    def verify_extra_h5_integrity(image_fn, _, extra_h5s):
        # TODO: This function is not doing much right now, and can be removed or should be refactored to something else
        # TODO: For instance a `direct verify-dataset`?
        if not extra_h5s:
            return

        for key in extra_h5s:
            h5_key, path = extra_h5s[key]
            extra_fn = path / image_fn.name
            try:
                with h5py.File(extra_fn, "r") as file:
                    _ = file[h5_key].shape
            except (OSError, TypeError) as exc:
                raise ValueError(f"Reading of {extra_fn} for key {h5_key} failed: {exc}.") from exc

            # TODO: This is not so trivial to do it this way, as the shape depends on context
            # if image_shape != shape:
            #     raise ValueError(f"{extra_fn} and {image_fn} has different shape. "
            #                      f"Got {shape} and {image_shape}")

    def __len__(self):
        return len(self.data)


    def _fetch_et_summary(self, pat_id: str, slice_idx: int, avg_idx=0) -> pd.DataFrame:
        """
        Fetch per‑echo‑train column‑index summaries for a given patient & slice.

        Parameters
        ----------
        patient_id : str
            The seq_id of the patient.
        db_path : str or Path
            Path to the SQLite database file.
        table_name : str
            Name of the table containing acquisitions.
        slice_index : int, default=0
            Which slice to filter on (usually 0 for mapping).
        debug : bool, default=False
            If True, prints the resulting DataFrame.

        Returns
        -------
        pd.DataFrame
            Columns: ['id','avg_idx','echo_train_idx','col_indexes','sample_count'].
            - col_indexes is a comma‑separated string of the sampled column indices.
            - sample_count should equal the echo‑train length (e.g. 25).

        Raises
        ------
        ValueError
            If no rows are returned for the given patient & slice.
        RuntimeError
            On any database I/O error.
        """
        db_file = str(self.db_path)
        sql = f"""
        SELECT
            id,
            avg_idx,
            echo_train_idx,
            GROUP_CONCAT(col_idx, ',') AS col_indexes,
            COUNT(col_idx)             AS sample_count
        FROM {self.et_mapping_tablename}
        WHERE 
            id = ?
            AND avg_idx = ?
            AND slice_idx = ?
        GROUP BY avg_idx, echo_train_idx
        ORDER BY avg_idx, echo_train_idx
        ;
        """
        try:
            with sqlite3.connect(db_file) as conn:
                df = pd.read_sql_query(sql, conn, params=(pat_id, avg_idx, slice_idx))
        except sqlite3.Error as e:
            raise RuntimeError(f"Unable to query echo‑train mapping: {e}")

        if df.empty:
            raise ValueError(
                f"No echo‑train summary found for patient={pat_id!r}, slice_idx={slice_idx}"
            )

        self.logger(f"Echo‑train summary for {pat_id!r}, slice {slice_idx}:\n", df)

        return df


    def _count_et(self, summary_df: pd.DataFrame) -> int:
        """Return the number of distinct echo trains in the summary DataFrame."""
        self.logger(f"Counting echo trains in summary DataFrame:\n{summary_df}")
        return summary_df["echo_train_idx"].nunique()


    def _find_center_et(self, ncols: int, summary_df: pd.DataFrame) -> int:
        """
        Find which echo‑train contains the center k-space column.

        Parameters
        ----------
        ncols : int
            Total number of k‑space columns.
        summary_df : pd.DataFrame
            Output of _fetch_et_summary() with 'col_indexes' per echo train.

        Returns
        -------
        int
            Echo-train index containing the center column.

        Raises
        ------
        ValueError
            If no echo train contains the center column.
        """
        middle_col = ncols // 2
        self.logger(f"Center column index is {middle_col}")
        
        for _, row in summary_df.iterrows():
            cols = list(map(int, row["col_indexes"].split(",")))
            if middle_col in cols:
                self.logger(f"Found center column in echo train {row['echo_train_idx']}")
                return int(row["echo_train_idx"])

        raise ValueError(f"No echo train found containing center column {middle_col}")


    def _select_retained_ets(self, total_ets: int, center_et: int, acc: int) -> List[int]:
        """
        Select a centered block of echo trains to retain, protecting the center ET.

        Parameters
        ----------
        total_ets : int
            Total number of echo trains.
        center_et : int
            The index of the echo train containing the center column.
        acc : int
            Acceleration factor.

        Returns
        -------
        List[int]
            List of retained echo-train indices.
        """
        if acc == 1:
            return list(range(total_ets))

        n_keep = (total_ets + 1) // acc
        half = n_keep // 2
        self.logger(f"Retaining {n_keep} echo trains centered around {center_et} with half: {half}.")

        start = max(0, center_et - half)
        end = start + n_keep
        if end > total_ets:
            start = max(0, total_ets - n_keep)
            end = total_ets

        retained = list(range(start, end))
        if center_et not in retained:
            raise ValueError(f"Center ET {center_et} not in retained ETs: {retained}")

        self.logger(f"Retained echo trains: {retained}")
        return retained
    

    def _select_et_fold(self, retained_ets: List[int], center_et: int, fold_idx: int) -> List[int]:
        """
        Select which echo train(s) to drop for a given fold.

        For acc=1: adjecent pairs (excluding center ET).
        For acc>1: single leave-one-out drop from retained ETs (excluding center ET).

        Parameters
        ----------
        retained_ets : List[int]
            List of retained echo train indices.
        center_et : int
            Echo train containing the center column.
        fold_idx : int
            Index of the fold variant to select (wrapped via modulo).

        Returns
        -------
        List[int]
            Echo train indices to drop for this fold.
        """
        if self.echo_train_acceleration == 1:
            # Adjecent dropout pairs
            pairs = []
            i = 0
            while i < len(retained_ets) - 1:
                et1, et2 = retained_ets[i], retained_ets[i + 1]
                if center_et in (et1, et2):
                    i += 1
                    continue
                pairs.append([et1, et2])
                i += 2
            if not pairs:
                raise ValueError("No valid echo-train dropout pairs (acc=1) excluding center ET.")
            self.logger(f"Adjecent dropout pairs: {pairs}")
            return pairs[fold_idx % len(pairs)]
        else:
            # Leave-one-out dropout
            candidates = [et for et in retained_ets if et != center_et]
            if not candidates:
                raise ValueError("No ETs left to drop for LOO (acc>1) after excluding center ET.")
            self.logger(f"Candidates for dropout: {candidates}")
            self.logger(f"Fold idx modulo: {fold_idx % len(candidates)}^: fold_idx: {fold_idx} % len(candidates): {len(candidates)}")
            return [candidates[fold_idx % len(candidates)]]


    def _build_col_mask(self, summary_df: pd.DataFrame, ets_to_drop: List[int], ncols: int) -> np.ndarray:
        """
        Build a binary mask over columns: 1=retain, 0=drop.

        Parameters
        ----------
        summary_df : pd.DataFrame
            Echo-train summary with 'echo_train_idx' and 'col_indexes' columns.
        ets_to_drop : List[int]
            Echo-train indices to drop.
        ncols : int
            Number of k-space columns (from kspace.shape[-1]).

        Returns
        -------
        np.ndarray
            Boolean mask over k-space columns, shape (ncols,), dtype=bool.
        """
        col_mask = np.ones(ncols, dtype=bool)

        for _, row in summary_df.iterrows():
            if row["echo_train_idx"] in ets_to_drop:
                cols = list(map(int, row["col_indexes"].split(",")))
                col_mask[cols] = False

        if not np.any(col_mask):
            raise RuntimeError("All columns masked out — this would zero all k-space.")

        return col_mask


    def _mask_kspace(self, kspace: np.ndarray, col_mask: np.ndarray) -> np.ndarray:
        """
        Zero out columns in k-space not marked as retained.

        Parameters
        ----------
        kspace : np.ndarray
            The 3D k-space array (coils, rows, cols).
        col_mask : np.ndarray
            Boolean array of shape (cols,), where False = drop.

        Returns
        -------
        np.ndarray
            K-space with selected columns zeroed out.
        """
        kspace[..., ~col_mask] = 0
        return kspace


    def perform_lxo_for_uq(self, kspace: np.ndarray, pat_id: str, slice_no: int, avg_idx: int=0) -> np.ndarray:

        # 1. Fetch ET metadata from SQL for this slice (avg=0 assumed)
        et_summary = self._fetch_et_summary(pat_id, slice_no, avg_idx=avg_idx)

        # 2. Count total ETs and find the center ET
        total_ets = self._count_et(et_summary)
        center_et = self._find_center_et(ncols=kspace.shape[-1], et_summary=et_summary)

        # 3. Select subset of retained ETs (acceleration logic)
        retained_ets = self._select_retained_ets(total_ets, center_et, self.echo_train_acceleration)

        # 4. Select ET(s) to drop for this fold
        ets_to_drop = self._select_et_fold(retained_ets, fold_idx=self.echo_train_fold_idx)

        # 5. Build a binary column mask (1=keep, 0=drop)
        col_mask = self._build_col_mask(et_summary, ets_to_drop, kspace.shape[-1])

        # 6. Apply the mask to k-space (in-place or copy)
        kspace = self._mask_kspace(kspace, col_mask)
        return kspace


    def __getitem__(self, idx: int) -> Dict[str, Any]:
        filename, slice_no = self.data[idx]
        filename = pathlib.Path(filename)
        metadata = None if not self.metadata else self.metadata[filename.name]
        # EXAMPLE:
        # filename = /scratch/p290820/datasets/003_umcg_pst_ksps/data/0007_ANON1586301/h5s/meas_MID00670_FID710360_T2_TSE_tra_obl-out_2.h5'
        # extract only patient_id = 0007_ANON1586301
        pat_id = filename.parts[-3]

        kspace, extra_data = self.get_slice_data(
            filename, slice_no, pass_attrs=self.pass_attrs, extra_keys=self.extra_keys
        )

        if kspace.ndim == 2:  # Singlecoil data does not always have coils at the first axis.
            kspace = kspace[np.newaxis, ...]

        self.logger.info(f"QVL - Patient ID: {pat_id}")
        self.logger.info(f"QVL - kspace.shape = {kspace.shape}")
        self.logger.info(f"QVL - kspace.dtype = {kspace.dtype}")
        self.logger.info(f"QVL - coil dimension is the first dimension.")

        if self.add_gaussian_noise: # kspace is 3D here, with dimensions (coils, rows, columns)
            gaussian_id     = self.get_gaussian_id(pat_id, debug = True)
            empirical_noise = self.get_noise_coil_list_from_db(pat_id, debug = True)
            kspace          = self.apply_mult_gaussian_noise_to_measured_lines(kspace=kspace, empirical_noise=empirical_noise, noise_multiplier=self.noise_mult, seed=gaussian_id, debug=True)

        if self.do_lxo_for_uq:
            kspace = self.perform_lxo_for_uq(kspace, pat_id, slice_no)

        # TODO: Write a custom collate function which disables batching for certain keys
        sample = {
            "kspace": kspace,
            "filename": str(filename),
            "slice_no": slice_no,
            "pat_id": pat_id,
            "gaussian_id": gaussian_id if self.add_gaussian_noise else None,
        }

        # If the sensitivity maps exist, load these
        if self.sensitivity_maps:
            sensitivity_map, _ = self.get_slice_data(
                self.sensitivity_maps / filename.name, slice_no, key="sensitivity_map"
            )
            sample["sensitivity_map"] = sensitivity_map

        if metadata is not None:
            sample["metadata"] = metadata

        sample.update(extra_data)

        if self.pass_dictionaries:
            for key in self.pass_dictionaries:
                if key in sample:
                    raise ValueError(f"Trying to add key {key} to sample dict, but this key already exists.")
                sample[key] = self.pass_dictionaries[key][filename.name]

        if self.pass_h5s:
            self.logger.info(f"QVL - self.pass_h5s = {self.pass_h5s}")
            for key, (h5_key, path) in self.pass_h5s.items():
                curr_slice, _ = self.get_slice_data(path / filename.name, slice_no, key=h5_key)
                if key in sample:
                    raise ValueError(f"Trying to add key {key} to sample dict, but this key already exists.")
                sample[key] = curr_slice

        return sample
    

    def get_gaussian_id(self, pat_id: str, debug: bool = False) -> int:
        """
        Retrieve the gaussian_id for a patient from the database.
        
        Connects to the database, fetches the latest row for the patient,
        and returns the existing gaussian_id if recon_path is NULL,
        or increments it by one otherwise.
        
        Args:
            pat_id (str): The patient identifier.
            
        Returns:
            int: The gaussian_id to use.
        """
        self.logger.info(f"QVL - self.acc_avg = {self.avg_acceleration}")

        conn = sqlite3.connect(self.db_path)  # Assumes self.db_path is defined in the class.
        cursor = conn.cursor()
        query = f"""
            SELECT recon_path, gaussian_id, avg_acceleration
            FROM {self.tablename_uq}
            WHERE id = ? and avg_acceleration = ?
            ORDER BY gaussian_id DESC
            LIMIT 1;
        """
        cursor.execute(query, (pat_id, self.avg_acceleration))
        row = cursor.fetchone()
        if row is None:
            conn.close()
            raise ValueError(f"No row found for patient ID: {pat_id}")
        
        recon_path, gaussian_id, avg_acceleration = row

        assert avg_acceleration == self.avg_acceleration, f"Expected avg_acceleration {self.avg_acceleration}, but got {avg_acceleration}"
        
        # If recon_path is NULL (None), return the existing gaussian_id; otherwise, increment it.
        result = gaussian_id if recon_path is None else gaussian_id + 1
        conn.close()

        if debug:
            self.logger.info(f"QVL - Returning gaussian_id: {result} for patient ID: {pat_id}. Gaussian_id was found to be {gaussian_id} and recon_path was found to be {recon_path}.")

        return result
        

    def apply_gaussian_noise_to_measured_lines(self, kspace: np.ndarray, sigma: np.ndarray, seed: int = None, debug=False) -> np.ndarray:
        """
        Apply Gaussian noise only to the measured (non-zero) lines in the k-space data for a single slice.
        
        Assumes:
            - kspace has shape (coils, rows, cols) for one slice.
            - sigma is a 1D array of per-coil sigma values.
        
        Args:
            kspace (np.ndarray): K-space data with shape (coils, rows, cols).
            sigma (np.ndarray): 1D array of noise standard deviations for each coil.
            seed (int, optional): Local random seed for reproducibility.
        
        Returns:
            np.ndarray: K-space data with added Gaussian noise on measured lines.
        """
        # Check inputs
        assert isinstance(kspace, np.ndarray), "kspace must be a numpy array"
        assert kspace.ndim == 3, "kspace must have 3 dimensions (coils, rows, cols)"
        assert isinstance(sigma, np.ndarray), "sigma must be a numpy array"
        assert sigma.ndim == 1, "sigma must be a 1D array (one value per coil)"
        assert sigma.shape[0] == kspace.shape[0], "Length of sigma must equal number of coils"
        assert np.all(sigma > 0), "All sigma values must be positive"
        
        # Use a local random generator for reproducibility without affecting global state.
        rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
        
        # Process each coil individually to save memory.
        for c in range(kspace.shape[0]):
            coil_data = kspace[c]  # shape: (rows, cols)
            # Find indices where the coil has a measured (nonzero) value.
            mask = np.nonzero(np.abs(coil_data) > 0)
            if mask[0].size > 0:
                # Number of measured elements in this coil.
                n_measured = mask[0].size
                # Generate noise only for these measured entries.
                noise = sigma[c] * (rng.standard_normal(n_measured) + 1j * rng.standard_normal(n_measured))
                # Add noise in-place.
                coil_data[mask] += noise
                kspace[c] = coil_data  # Update the coil's data.

        if debug:
            self.logger.info(f"QVL - Added Gaussian noise to k-space data with sigma values: {sigma}")
        
        return kspace
    
        
    def compute_noise_sigma(self, kspace: np.ndarray, fraction: float = 0.01, debug: bool = False) -> np.ndarray:
        """
        Compute noise sigma for each coil as a fraction of the maximum absolute value of the k-space data.

        Args:
            kspace (np.ndarray): Collapsed k-space data for a single slice, shape (coils, rows, columns).
            fraction (float): Fraction of the maximum amplitude to use as noise sigma.
            debug (bool): If True, print debug information.

        Returns:
            np.ndarray: Array of computed noise sigma values per coil, shape (coils,).
        """
        assert isinstance(kspace, np.ndarray), "kspace must be a numpy array"
        # Now we expect kspace to have 3 dimensions: (coils, rows, columns)
        assert kspace.ndim == 3, "kspace must have 3 dimensions (coils, rows, columns)"
        assert isinstance(fraction, float) and 0 < fraction < 1, "fraction must be a float between 0 and 1"
        
        # Compute the maximum absolute value for each coil (over rows and columns)
        max_vals = np.max(np.abs(kspace), axis=(1, 2))
        sigma = fraction * max_vals
        
        if debug:
            print(f"Per-coil maximum values: {max_vals}")
            print(f"Computed noise sigma per coil: {sigma}")

        assert sigma.shape[0] == kspace.shape[0], "sigma must have the same length as the number of coils"
        
        return sigma


    def parse_noise_coils_list(self, noise_coils_list: str) -> np.ndarray:
        """
        Parse the noise_coils_list string into a numpy array of floats.
        
        Args:
            noise_coils_list (str): Comma-separated string of floats.
        
        Returns:
            np.ndarray: Array of floats.
        """
        # Split the string by commas and convert each element to a float.
        noise_list = [float(val) for val in noise_coils_list.split(',')]
        return np.array(noise_list)


    def get_noise_coil_list_from_db(self, pat_id: str, debug = False) -> np.ndarray:

        conn = sqlite3.connect(str(self.db_path))
        c = conn.cursor()
        query = "SELECT * FROM noise_estimation WHERE id = ?"
        c.execute(query, (pat_id.strip(),))
        _, _, _, _, noise_coils_list, _, _ = c.fetchone()
        conn.close()

        # The found noise is a  list of floats as a string, we need to parse it into a numpy array.
        emperical_noise = self.parse_noise_coils_list(noise_coils_list)

        # if debug: 
        #     self.logger.info(f"\nemperical_noise = {emperical_noise}")

        return emperical_noise


    def apply_mult_gaussian_noise_to_measured_lines(self, kspace: np.ndarray, empirical_noise: np.ndarray, noise_multiplier: float, seed: int = None, debug=False) -> np.ndarray:
        # Scale the baseline noise by the multiplier.
        scaled_sigma = noise_multiplier * empirical_noise
        if debug:
            self.logger.info(f"Applying increased emperical noise per coil ... Noise multiplier {self.noise_mult}")
        rng = np.random.default_rng(seed)

        # Create a copy to hold the noisy k-space data.
        noisy_kspace = kspace.copy()
        coils, rows, cols = kspace.shape

        for c in range(coils):
            coil_data = noisy_kspace[c].copy()  # shape: (rows, cols)
            # Find indices where the coil has measured (nonzero) data.
            mask = np.nonzero(np.abs(coil_data) > 0)
            if mask[0].size > 0:
                n_measured = mask[0].size
                # Generate complex Gaussian noise for these measured entries,
                # scaled by the increased sigma for the current coil.
                coil_data[mask] += scaled_sigma[c] * (rng.standard_normal(n_measured) + 1j * rng.standard_normal(n_measured))
            noisy_kspace[c] = coil_data

        return noisy_kspace
    

    def get_slice_data(self, filename, slice_no, key="kspace", pass_attrs=False, extra_keys=None):
        avg1, avg2, avg3 = 0, 1, 2
        
        extra_data = {}
        if not filename.exists():
            raise OSError(f"{filename} does not exist.")
        try:
            data = h5py.File(filename, "r")
        except Exception as e:
            raise Exception(f"Reading filename {filename} caused exception: {e}")

        if self.kspace_context == 0:
            # We must collapse the averages first, then take the slice.
            if self.average_collapse_strat == "allavg" or self.average_collapse_strat == None: # ALLAVG - We will combine the 3 averages with full_kspace = (avg1+avg3)/2 + avg2
                curr_data = (data[key][avg1][slice_no] + data[key][avg3][slice_no]) / 2 + data[key][avg2][slice_no]
            elif self.average_collapse_strat == "avg1":   # AVG1 - We will only take the first average, that will be equal to R3 already relative to the full protocol with 3 averages
                curr_data = data[key][avg1][slice_no]
            elif self.average_collapse_strat == "avg2":   # AVG2 - We will only take the second average, that will be equal to R3 already relative to the full protocol with 3 averages
                curr_data = data[key][avg2][slice_no]
            elif self.average_collapse_strat == "avg3":   # AVG3 - We will only take the third average, that will be equal to R3 already relative to the full protocol with 3 averages
                curr_data = data[key][avg3][slice_no]
            else:
                raise ValueError(f"Invalid average_collapse_strat: {self.average_collapse_strat}")
            
            if self.store_applied_acs_mask:
                # Get the ACS region from avg1+avg2
                curr_data_avg12 = data[key][avg1][slice_no] + data[key][avg2][slice_no]
                ncoils = curr_data_avg12.shape[0]
                nx, ny = curr_data_avg12.shape[-2:]
                acs_mask = np.zeros((ncoils, nx, ny, 2), dtype=bool)
                acs_mask[:, :, ny // 2 - 51 : ny // 2 + 51, :] = True
                curr_data_avg12 = np.stack((curr_data_avg12.real, curr_data_avg12.imag), axis=-1)
                # print(f"QVL - curr_data_avg12.shape = {curr_data_avg12.shape}")
                # print(f"QVL - curr_data_avg12.dtype = {curr_data_avg12.dtype}")
                # print(f"QVL - acs_mask.shape = {acs_mask.shape}")
                applied_mask = curr_data_avg12 * acs_mask
                del curr_data_avg12
                extra_data["applied_acs_mask"] = applied_mask
            
        else:
            # This can be useful for getting stacks of slices.
            num_slices = self.get_num_slices(filename)
            
            # We must collapse the averages first, then take the slice.
            if self.average_collapse_strat == "allavg" or self.average_collapse_strat == None:
                avg_collapsed = (data[key][avg1] + data[key][avg3]) / 2 + data[key][avg2]
            elif self.average_collapse_strat == "avg1":
                avg_collapsed = data[key][avg1]
            elif self.average_collapse_strat == "avg2":
                avg_collapsed = data[key][avg2]
            elif self.average_collapse_strat == "avg3":
                avg_collapsed = data[key][avg3]
            
            # Continue as normal. Essentially if kspace_context = 0, we will always take the slice_no slice as the current slice. So nothing changes.
            curr_data = avg_collapsed[
                max(0, slice_no - self.kspace_context) : min(slice_no + self.kspace_context + 1, num_slices),
            ]
            curr_shape = curr_data.shape
            self.logger.info(f"QVL - curr_data.shape (v2) = {curr_data.shape}")
            if curr_shape[0] < num_slices - 1:
                if slice_no - self.kspace_context < 0:
                    new_shape = list(curr_shape).copy()
                    new_shape[0] = self.kspace_context - slice_no
                    curr_data = np.concatenate(
                        [np.zeros(new_shape, dtype=curr_data.dtype), curr_data],
                        axis=0,
                    )
                if self.kspace_context + slice_no > num_slices - 1:
                    new_shape = list(curr_shape).copy()
                    new_shape[0] = slice_no + self.kspace_context - num_slices + 1
                    curr_data = np.concatenate(
                        [curr_data, np.zeros(new_shape, dtype=curr_data.dtype)],
                        axis=0,
                    )
            # Move the depth axis to the second spot.
            curr_data = np.swapaxes(curr_data, 0, 1)
            self.logger.info(f"QVL - curr_data.shape (v3) = {curr_data.shape}")

        if pass_attrs:
            extra_data["attrs"] = dict(data.attrs)

        if extra_keys:
            for extra_key in self.extra_keys:
                if extra_key == "attrs":
                    raise ValueError("attrs need to be passed by setting `pass_attrs = True`.")
                extra_data[extra_key] = data[extra_key][()]
        data.close()

        # just some logging
        self.logger.info(f"QVL - curr_data.shape = {curr_data.shape}")
        self.logger.info(f"QVL - curr_data.dtype = {curr_data.dtype}")

        return curr_data, extra_data


    def get_num_slices(self, filename):
        num_slices = self.volume_indices[filename].stop - self.volume_indices[filename].start
        return num_slices
