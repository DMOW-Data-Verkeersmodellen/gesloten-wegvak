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

    COLUMNS = ["road_section_id", "timestamp", "vehicle_type", 
               "volume", "volume_err", "source_type", "source_id", 
               "weight", "outlier"]


    def __init__(self) -> None:
        self._df = self._create_empty_matrix()

    def _create_empty_matrix(self) -> pd.DataFrame:
        """Initializes schema using tight data types to optimize memory footprint."""
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
            "outlier": pd.Series(dtype="boolean")
        })
        # Explicit categories restrict memory expansion over large timelines
        df["vehicle_type"] = df["vehicle_type"].cat.set_categories(["PW", "VR"])
        df["source_type"] = df["source_type"].cat.set_categories([self.SOURCE_OBS, self.SOURCE_REC])
        return df

    @property
    def dataframe(self) -> pd.DataFrame:
        """pd.DataFrame: Access the underlying pandas matrix for analysis."""
        return self._df
    
    def flag_outliers(self, indices: pd.Index | list | np.array) -> None:
        """Flags rows as outliers."""
        self._df.loc[indices, "outlier"] = True

    def unflag_outliers(self, indices: pd.Index | list | np.array) -> None:
        """Unflags rows as outliers."""
        self._df.loc[indices, "outlier"] = False

    def clear_reconstructions(self) -> None:
        """Purges old reconstruction rows to clean up memory between iterative passes."""
        self._df = self._df[self._df["source_type"] != self.SOURCE_REC].reset_index(drop=True)

    def append_estimates(self, df_to_append: pd.DataFrame) -> None:
        """Safely bulk-appends rows, casting columns back to expected categories."""
        df_copy = df_to_append.copy()
        for col in ["source_type", "vehicle_type"]:
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
        
        # Strip measurements from any location that couldn't be mapped
        records = records.dropna(subset=["road_section_id"])

        # Format columns for the master matrix
        records["source_type"] = self.SOURCE_OBS
        records["source_id"] = "sensor:" + records["location_id"].astype(str)
        records["weight"] = 1.0
        records["outlier"] = False

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
            (self._df["outlier"] == False) &
            (self._df["source_type"].isin(source_types))
        )
        sub = self._df[mask].dropna(subset=["volume"])
        if sub.empty:
            return np.nan, np.nan
        
        v = sub["volume"].to_numpy(dtype="float64")
        e = sub["volume_err"].to_numpy(dtype="float64")
        w = sub["weight"].to_numpy(dtype="float64")

        return utils.weighted_mean_and_error(v, e, w, label=road_section_id)