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


class FlowStore():
    """
    Consolidated memory-optimized data container holding time-series measurements,
    reconstructions, and algorithm weights across the entire highway network.
    """

    SOURCE_OBS = "direct_sensor"
    SOURCE_REC = "reconstructed"
    SOURCE_RES = "resolved"

    COLUMNS = ["road_section_id", "timestamp", "vehicle_type", 
               "volume", "volume_err", "source_type", "source_id", 
               "weight", "screening", "validation"]
    
    SCREENING_STATES = {"unknown", "accepted", "disputed", "held", "void", "NA"}
    VALIDATION_STATES = {"pending", "verified", "unresolved", "rejected", "NA"}

    def __init__(self) -> None:
        self._df = self._create_empty_matrix()

    def _create_empty_matrix(self) -> pd.DataFrame:
        """Initializes schema using tight data types to optimize memory footprint."""
        screening_type = pd.CategoricalDtype(categories=list(self.SCREENING_STATES))
        validation_type = pd.CategoricalDtype(categories=list(self.VALIDATION_STATES))

        df = pd.DataFrame(columns=self.COLUMNS)
        df = pd.DataFrame({
            "road_section_id": pd.Series(dtype="string"),
            "timestamp": pd.Series(dtype="datetime64[ns]"),
            "vehicle_type": pd.Series(dtype="category"),
            "volume": pd.Series(dtype="float32"),
            "volume_err": pd.Series(dtype="float32"),
            "source_type": pd.Series(dtype="category"),
            "source_id": pd.Series(dtype="string"),
            "weight": pd.Series(dtype="float32"),
            "screening": pd.Series(dtype=screening_type),
            "validation": pd.Series(dtype=validation_type)
        })
        # Explicit categories restrict memory expansion over large timelines
        df["vehicle_type"] = df["vehicle_type"].cat.set_categories(["PW", "VR"])
        df["source_type"] = df["source_type"].cat.set_categories([self.SOURCE_OBS, self.SOURCE_REC])
        return df

    @property
    def dataframe(self) -> pd.DataFrame:
        """pd.DataFrame: Access the underlying pandas matrix for analysis."""
        return self._df.copy()

    def set_validation_state(self, indices: pd.Index | list | np.array, state: str) -> None:
        """Sets the dynamic validation state ('verified', 'unresolved', 'rejected', 'pending') for given indices."""
        if state not in self.VALIDATION_STATES:
            raise ValueError(f"Invalid validation state: '{state}'. Must be one of {self.VALIDATION_STATES}")
        indices = pd.Index(indices)
        if not indices.dropna().empty:
            self._df.loc[indices, "validation"] = state

    def clear_reconstructions(self) -> None:
        """Purges old reconstruction rows to clean up memory between iterative passes."""
        self._df = self._df[self._df["source_type"] != self.SOURCE_REC].reset_index(drop=True)

    def append_estimates(self, df_to_append: pd.DataFrame) -> None:
        """Safely bulk-appends rows, casting columns back to expected categories."""
        df_copy = df_to_append.copy()

        # 1. Check and validate "screening" values if present in incoming data
        if "screening" in df_copy.columns:
            incoming_screening = set(df_copy["screening"].dropna().unique())
            invalid_screen = incoming_screening - self.SCREENING_STATES
            if invalid_screen:
                raise ValueError(
                    f"Cannot append: 'screening' column contains invalid values: {invalid_screen}. "
                    f"Must be one of {self.SCREENING_STATES}"
                )
        else:
            # Fallback default if not supplied
            df_copy["screening"] = "unknown"

        # 2. Check and validate "validation" values if present in incoming data
        if "validation" in df_copy.columns:
            incoming_validation = set(df_copy["validation"].dropna().unique())
            invalid_valid = incoming_validation - self.VALIDATION_STATES
            if invalid_valid:
                raise ValueError(
                    f"Cannot append: 'validation' column contains invalid values: {invalid_valid}. "
                    f"Must be one of {self.VALIDATION_STATES}"
                )
        else:
            # Default state as requested
            df_copy["validation"] = "pending"

        # 3. Cast to Categories for memory optimization
        for col in ["source_type", "vehicle_type", "screening", "validation"]:
            if col in df_copy.columns:
                df_copy[col] = df_copy[col].astype("category")
        
        self._df = pd.concat([self._df, df_copy], ignore_index=True)

    def load_raw_observations(self, network: Network, counts_df: pd.DataFrame) -> None:
        """
        Maps a user's raw time-series count DataFrame to compiled network RoadSections.

        Parameters
        ----------
        network : Network
            The compiled network infrastructure map.
        counts_df : pd.DataFrame
            Must contain columns: ['timestamp', 'location_id', 'vehicle_type', 'volume']
            Optional column: ['volume_err'] for error estimates. If not provided, defaults to NaN.
        """
        # Map location_id -> road_section_id using the network topology graph
        sensor_to_section = {}
        for sec_id, section in network.sections.items():
            for sensor in section.iter_sensors():
                sensor_to_section[sensor.location_id] = sec_id

        if not sensor_to_section:
            raise ValueError("No matching sensors found within the provided network topology.")

        records = counts_df.copy()
        records["road_section_id"] = records["location_id"].map(sensor_to_section)

        if not "volume_err" in records.columns:
            records["volume_err"] = np.nan  # Default to NaN if no error column is provided
        if not "screening" in records.columns:
            records["screening"] = "unknown"
        
        # Strip measurements from any location that couldn't be mapped
        records = records.dropna(subset=["road_section_id"])

        # Format columns for the master matrix
        records["source_type"] = self.SOURCE_OBS
        records["source_id"] = "sensor:" + records["location_id"].astype(str)
        records["weight"] = 1.0
        records["validation"] = "pending"

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
        float or None
            None when no matching rows exist or all volumes are NaN.
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

        return utils.weighted_mean_and_error(v, e, w, label=road_section_id)