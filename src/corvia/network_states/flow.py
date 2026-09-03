"""
flow.py
==================
Time-series storage matrices for traffic volumes and algorithmic reconstructions.
"""

from __future__ import annotations
from operator import sub
from typing import List, Optional, Set, TYPE_CHECKING, Tuple
import numpy as np
import pandas as pd

from corvia.framework.network import Network
import corvia.utils as utils

from corvia.logger import get_logger
logger = get_logger(__name__)


class FlowStore():
    """
    Consolidated memory-optimized data container holding time-series measurements,
    reconstructions, and algorithm weights across the entire highway network.

    Parameters
    ----------
    vehicle_types : tuple of str or None, optional
        Restricts the ``vehicle_type`` category allowed in this store.
        Defaults to ``("PW", "VR", "TOTAL", "PAE")``; pass ``None`` for no
        restriction.

    Attributes
    ----------
    SOURCE_OBS, SOURCE_REC, SOURCE_RES : str
        Allowed values for the ``source_type`` column: direct sensor
        observation, algorithmic reconstruction, and resolver-published
        baseline, respectively.
    COLUMNS : list of str
        Full column schema of the underlying matrix.
    VTYPE_DEFAULTS : tuple of str
        Default ``vehicle_types`` restriction used by :meth:`__init__`.
    SCREENING_STATES : set of str
        Allowed values for the ``screening`` column.
    VALIDATION_STATES : set of str
        Allowed values for the ``validation`` column.
    """

    SOURCE_OBS = "direct_sensor"
    SOURCE_REC = "reconstructed"
    SOURCE_RES = "resolved"

    COLUMNS = ["road_section_id", "timestamp", "vehicle_type", 
               "volume", "volume_err", "source_type", "source_id", 
               "weight", "screening", "validation"]

    VTYPE_DEFAULTS = ("PW", "VR", "TOTAL", "PAE")
    
    SCREENING_STATES = {"unknown", "accepted", "disputed", "held", "void", "NA"}
    VALIDATION_STATES = {"pending", "verified", "conforming", "dismissed", "rejected", "unresolved", "NA"}

    def __init__(self, vehicle_types: Optional[Tuple[str, ...]] = VTYPE_DEFAULTS) -> None:
        self._vehicle_types = tuple(vehicle_types) if vehicle_types is not None else None
        self._df = self._create_empty_matrix()
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")
        self.logger.debug(f"FlowStore initialized (vehicle_types={self._vehicle_types}).")

    @classmethod
    def from_dataframe(
        cls, 
        df: pd.DataFrame,
        vehicle_types: Optional[Tuple[str, ...]] = VTYPE_DEFAULTS,
    ) -> FlowStore:
        """
        Build a FlowStore directly from a prepared dataframe.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain every column in :attr:`COLUMNS`; see
            :meth:`append_estimates` for what happens to anything else.
        vehicle_types : tuple of str or None, optional
            Forwarded to the new store's constructor. Defaults to
            ``("PW", "VR", "TOTAL", "PAE")``; pass ``None`` for no
            restriction.

        Returns
        -------
        FlowStore
        """
        store = cls(vehicle_types=vehicle_types)
        store.append_estimates(df)
        return store

    def _create_empty_matrix(self) -> pd.DataFrame:
        """Initializes schema using tight data types to optimize memory footprint."""
        screening_type = pd.CategoricalDtype(categories=list(self.SCREENING_STATES))
        validation_type = pd.CategoricalDtype(categories=list(self.VALIDATION_STATES))
        vehicle_type_dtype = (
            pd.CategoricalDtype(categories=list(self._vehicle_types))
            if self._vehicle_types is not None
            else "category"
        )

        df = pd.DataFrame(columns=self.COLUMNS)
        df = pd.DataFrame({
            "road_section_id": pd.Series(dtype="string"),
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "vehicle_type": pd.Series(dtype=vehicle_type_dtype),
            "volume": pd.Series(dtype="float32"),
            "volume_err": pd.Series(dtype="float32"),
            "source_type": pd.Series(dtype="category"),
            "source_id": pd.Series(dtype="string"),
            "weight": pd.Series(dtype="float32"),
            "screening": pd.Series(dtype=screening_type),
            "validation": pd.Series(dtype=validation_type)
        })
        # Explicit categories restrict memory expansion over large timelines
        df["source_type"] = df["source_type"].cat.set_categories([self.SOURCE_OBS, self.SOURCE_REC])
        return df

    @property
    def dataframe(self) -> pd.DataFrame:
        """pd.DataFrame: Access the underlying pandas matrix for analysis."""
        return self._df.copy()

    @property
    def vehicle_types(self) -> Optional[Tuple[str, ...]]:
        """
        tuple of str or None : Allowed ``vehicle_type`` categories for
        this store *(read-only)*. ``None`` means unrestricted — no
        ``vehicle_type`` validation is performed by :meth:`append_estimates`.
        """
        return self._vehicle_types

    def set_validation_state(self, indices: pd.Index | list | np.array, state: str) -> None:
        """
        Sets the dynamic validation state for given indices. See
        FlowStore.VALIDATION_STATES for the allowed values.

        Parameters
        ----------
        indices : pd.Index or list or np.array
            Row indices (into :attr:`dataframe`) to update.
        state : str
            Must be one of :attr:`VALIDATION_STATES`.

        Raises
        ------
        ValueError
            If *state* is not a recognized validation state.
        """
        if state not in self.VALIDATION_STATES:
            error_msg = f"Invalid validation state '{state}' requested; must be one of {self.VALIDATION_STATES}."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        indices = pd.Index(indices)
        if not indices.dropna().empty:
            self._df.loc[indices, "validation"] = state

    def clear_reconstructions(
        self,
        vehicle_types: Optional[List[str]] = None,
        periods: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        """Purges reconstruction rows.

        Parameters
        ----------
        vehicle_types : list of str, optional
            If given, only purges reconstruction rows for these vehicle types.
        periods : list of Timestamp, optional
            If given, only purges reconstruction rows at these timestamps.
        """
        mask = self._df["source_type"] == self.SOURCE_REC
        if vehicle_types is not None:
            mask &= self._df["vehicle_type"].isin(vehicle_types)
        if periods is not None:
            mask &= self._df["timestamp"].isin(periods)
        self.logger.debug(
            f"clear_reconstructions() purging {int(mask.sum())} row(s) "
            f"(vehicle_types={vehicle_types}, periods={periods})."
        )
        self._df = self._df[~mask].reset_index(drop=True)

    def clear_resolved(
        self,
        vehicle_types: Optional[List[str]] = None,
        periods: Optional[List[pd.Timestamp]] = None,
        source_id: Optional[str] = None,
    ) -> None:
        """
        Purges existing resolved (SOURCE_RES) rows. 

        Parameters
        ----------
        vehicle_types : list of str, optional
            If given, only purges resolved rows for these vehicle types.
        periods : list of Timestamp, optional
            If given, only purges resolved rows at these timestamps.
        source_id : str, optional
            If given, only purges resolved rows with this source_id
        """
        mask = self._df["source_type"] == self.SOURCE_RES
        if vehicle_types is not None:
            mask &= self._df["vehicle_type"].isin(vehicle_types)
        if periods is not None:
            mask &= self._df["timestamp"].isin(periods)
        if source_id is not None:
            mask &= self._df["source_id"] == source_id
        self.logger.debug(
            f"clear_resolved() purging {int(mask.sum())} row(s) "
            f"(vehicle_types={vehicle_types}, periods={periods}, source_id={source_id})."
        )
        self._df = self._df[~mask].reset_index(drop=True)

    def append_estimates(self, df_to_append: pd.DataFrame) -> None:
        """
        Safely bulk-appends rows, casting columns back to expected categories.

        Only :attr:`COLUMNS` ever end up in the stored matrix: any other
        column on *df_to_append* is dropped rather than silently added
        as a new, mostly-empty column via ``pd.concat``.

        Parameters
        ----------
        df_to_append : pd.DataFrame
            Must contain every column in :attr:`COLUMNS`, except
            ``screening`` and ``validation``, which fall back to
            ``"unknown"`` and ``"pending"`` respectively when absent.

        Raises
        ------
        ValueError
            If a required column is missing, or if ``screening``,
            ``validation``, or (when this store restricts it via
            :attr:`vehicle_types`) ``vehicle_type`` contains a value
            outside its allowed set.
        """
        df_copy = df_to_append.copy()

        # 0. Every column the matrix needs must be present, aside from
        #    the two that fall back to a default below.
        optional_with_default = {"screening", "validation"}
        required = [c for c in self.COLUMNS if c not in optional_with_default]
        missing = [c for c in required if c not in df_copy.columns]
        if missing:
            error_msg = f"append_estimates() cannot append, missing required columns: {missing}."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        # 1. Check and validate "screening" values if present in incoming data
        if "screening" in df_copy.columns:
            incoming_screening = set(df_copy["screening"].dropna().unique())
            invalid_screen = incoming_screening - self.SCREENING_STATES
            if invalid_screen:
                error_msg = (
                    f"append_estimates() cannot append, invalid 'screening' values: {invalid_screen}."
                    f"Must be one of {self.SCREENING_STATES}"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            # Fallback default if not supplied
            df_copy["screening"] = "unknown"

        # 2. Check and validate "validation" values if present in incoming data
        if "validation" in df_copy.columns:
            incoming_validation = set(df_copy["validation"].dropna().unique())
            invalid_valid = incoming_validation - self.VALIDATION_STATES
            if invalid_valid:
                error_msg = (
                    f"append_estimates() cannot append, invalid 'validation' values: {invalid_valid}."
                    f"Must be one of {self.VALIDATION_STATES}"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            # Default state
            df_copy["validation"] = "pending"

        # 3. Check "vehicle_type" values against this store's allowed
        #    set, when it has one (None means unrestricted).
        if self._vehicle_types is not None:
            incoming_vehicle_type = set(df_copy["vehicle_type"].dropna().unique())
            invalid_vt = incoming_vehicle_type - set(self._vehicle_types)
            if invalid_vt:
                error_msg = (
                    f"append_estimates() cannot append, invalid 'vehicle_type' values: {invalid_vt}."
                    f"Must be one of {self._vehicle_types}"
                )
                self.logger.error(error_msg)
                raise ValueError(error_msg)

        # 4. Cast to the right types to avoid memory bloat and ensure consistency
        df_copy["road_section_id"] = df_copy["road_section_id"].astype("string")
        df_copy["timestamp"] = pd.to_datetime(df_copy["timestamp"], errors="coerce")
        df_copy["source_id"] = df_copy["source_id"].astype("string")
        for col in ["volume", "volume_err", "weight"]:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce").astype("float32")
        for col in ["source_type", "vehicle_type", "screening", "validation"]:
            df_copy[col] = df_copy[col].astype("category")

        # 5. Restrict to exactly COLUMNS — concatenating mismatched
        #    columns would otherwise silently add new, mostly-empty
        #    columns to the matrix instead of raising.
        df_copy = df_copy[self.COLUMNS]
        
        self._df = pd.concat([self._df, df_copy], ignore_index=True)
        self.logger.info(f"append_estimates () appended {len(df_copy)} row(s); store now holds {len(self._df)} row(s).")

    def load_raw_observations(self, network: Network, counts_df: pd.DataFrame) -> None:
        """
        Maps a user's raw time-series count DataFrame to compiled network RoadSections.

        Parameters
        ----------
        network : Network
            The compiled network infrastructure map.
        counts_df : pd.DataFrame
            Must contain columns: ['timestamp', 'location_id', 'vehicle_type', 'volume']
            Optional column: ['volume_err'] for error estimates, 
            if not provided, defaults to NaN.

        Raises
        ------
        ValueError
            If none of *counts_df*'s ``location_id`` values match a sensor in
            *network*.
        """
        # Map location_id -> road_section_id using the network topology graph
        sensor_to_section = {}
        for sec_id, section in network.sections.items():
            for sensor in section.iter_sensors():
                sensor_to_section[sensor.location_id] = sec_id

        if not sensor_to_section:
            error_msg = "load_raw_observations() found no matching sensors within the provided network topology."
            self.logger.error(error_msg)
            raise ValueError(error_msg)

        records = counts_df.copy()
        records["road_section_id"] = records["location_id"].map(sensor_to_section)

        if not "volume_err" in records.columns:
            records["volume_err"] = np.nan  # Default to NaN if no error column is provided
        if not "screening" in records.columns:
            records["screening"] = "unknown"
        
        # Strip measurements from any location that couldn't be mapped
        unmapped = int(records["road_section_id"].isna().sum())
        if unmapped:
            self.logger.warning(
                f"load_raw_observations: dropping {unmapped} row(s) whose "
                f"location_id could not be mapped to a road section."
            )
        records = records.dropna(subset=["road_section_id"])

        # Format columns for the master matrix
        records["source_type"] = self.SOURCE_OBS
        records["source_id"] = "sensor:" + records["location_id"].astype(str)
        records["weight"] = 1.0
        records["validation"] = "pending"

        self.logger.info(f"load_raw_observations() loaded {len(records)} row(s) from {len(counts_df)} raw record(s).")
        self.append_estimates(records[self.COLUMNS])

    def section_consensus(
        self,
        road_section_id: str,
        timestamp: pd.Timestamp,
        vehicle_type: str,
        source_types: tuple = (SOURCE_OBS, SOURCE_REC),
    ) -> Tuple[float, float]:
        """
        Return the weighted mean volume for a (section, timestamp, vehicle_type).

        Parameters
        ----------
        road_section_id : str
        timestamp : pd.Timestamp
        vehicle_type : str
        source_types : tuple of str, optional
            Only rows with these source types contribute.
            Defaults to ``(SOURCE_OBS, SOURCE_REC)`` so both readings
            and reconstructions are used. Pass ``(SOURCE_REC,)`` to 
            only include reconstructions in the consensus.

        Returns
        -------
        volume, volume_err : float, float
            Weighted mean volume and its error. ``(np.nan, np.nan)`` when no
            matching rows exist or all volumes are NaN.
        """
        mask = (
            (self._df["road_section_id"] == road_section_id) &
            (self._df["timestamp"]       == timestamp) &
            (self._df["vehicle_type"]    == vehicle_type) &
            (self._df["source_type"].isin(source_types)) &
            (self._df["validation"] != "rejected")
        )

        if self.SOURCE_OBS in source_types:
            mask = mask & (
                (self._df["source_type"] != self.SOURCE_OBS) |
                (self._df["screening"].isin(["accepted", "disputed", "unknown"]))
            )

        sub = self._df[mask].dropna(subset=["volume"])
        if sub.empty:
            return np.nan, np.nan
        
        v = sub["volume"].to_numpy(dtype="float64")
        e = sub["volume_err"].to_numpy(dtype="float64")
        w = sub["weight"].to_numpy(dtype="float64")

        return utils.weighted_mean_and_error(v, e, w, label="consensus"+road_section_id)

    def lookup_resolved_baseline(
        self,
        target_rows: pd.DataFrame,
        source_id: Optional[str] = None,
    ) -> pd.DataFrame:
        """
        Looks up the already-published SOURCE_RES value for each target row's
        (road_section_id, timestamp, vehicle_type) coordinate.

        Unlike section_consensus() (which blends multiple contributing rows via
        a weighted mean), this returns the exact SOURCE_RES row already sitting
        in the store — useful when a caller wants to test against the specific
        value that was actually published, rather than recomputing an aggregate.

        Parameters
        ----------
        target_rows : pd.DataFrame
            Must contain columns ["road_section_id", "timestamp", "vehicle_type"].
        source_id : str, optional
            If given, only matches SOURCE_RES rows with this source_id — so a
            lookup doesn't accidentally pick up another resolver's published
            output for the same coordinate. If None, matches any SOURCE_RES row.

        Returns
        -------
        pd.DataFrame
            Columns ["baseline", "baseline_error"], aligned to target_rows.index.
            NaN where no SOURCE_RES row exists yet for that coordinate.
        """
        mask = self._df["source_type"] == self.SOURCE_RES
        if source_id is not None:
            mask &= self._df["source_id"] == source_id
        resolved = self._df[mask]
        if resolved.empty:
            self.logger.debug(
                f"lookup_resolved_baseline: no SOURCE_RES rows available (source_id={source_id}); "
                f"returning all-NaN for {len(target_rows)} target row(s)."
            )
            return pd.DataFrame(np.nan, index=target_rows.index, columns=["baseline", "baseline_error"])

        lookup_map = (
            resolved[["road_section_id", "timestamp", "vehicle_type", "volume", "volume_err"]]
            .rename(columns={"volume": "baseline", "volume_err": "baseline_error"})
        )
        obs_coords = target_rows[["road_section_id", "timestamp", "vehicle_type"]].reset_index()
        baseline = obs_coords.merge(
            lookup_map,
            on=["road_section_id", "timestamp", "vehicle_type"],
            how="left"
        ).set_index("index")[["baseline", "baseline_error"]]

        unmatched = int(baseline["baseline"].isna().sum())
        if unmatched:
            self.logger.debug(
                f"lookup_resolved_baseline: {unmatched}/{len(target_rows)} target row(s) "
                f"had no matching resolved baseline (source_id={source_id})."
            )
        return baseline