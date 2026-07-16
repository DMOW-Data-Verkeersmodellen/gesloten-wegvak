# corvia/engine/validators.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd

from corvia import utils
from corvia.network_states.flow import FlowStore

class BaseValidator(ABC):
    """Abstract Base Class for all network baseline validation strategies."""
    
    @abstractmethod
    def validate(self, store: FlowStore, target_indices: pd.Index) -> Tuple[pd.Index, pd.Index]:
        """
        Executes validation on the specified target indices.
        
        Returns
        -------
        conforming_indices : pd.Index
            Indices of observations that conform (verified).
        anomalies_indices : pd.Index
            Indices of observations that do not conform (rejected).
        """
        pass

    @staticmethod
    def _calc_weighted_mean_and_std(flow_df: pd.DataFrame) -> pd.Series:
        v = flow_df["volume"].to_numpy(dtype="float64")
        e = flow_df["volume_err"].to_numpy(dtype="float64")
        w = flow_df["weight"].to_numpy(dtype="float64")
        mean, std = utils.weighted_mean_and_error(v, e, w)
        return pd.Series({"baseline_mean": mean, "baseline_std": std})
    
    @staticmethod
    def _calc_median_and_mad(flow_df: pd.DataFrame) -> pd.Series:
        volumes = flow_df["volume"].to_numpy(dtype="float64")
        median = np.median(volumes)
        mad = np.median(np.abs(volumes - median))
        return pd.Series({"baseline_median": median, "baseline_mad": mad})


class ZScoreValidator(BaseValidator):
    """
    Standard Z-Score validation comparing observations to a weighted baseline mean.
    """
    def __init__(self, z_threshold: float = 1.96, use_errors: bool = True):
        self.z_threshold = z_threshold
        self.use_errors = use_errors

    def validate(self, store: FlowStore, obs_indices: pd.Index) -> Tuple[pd.Index, pd.Index]:
        df = store.dataframe
        obs_rows = df.loc[obs_indices]
        
        # 1. Isolate valid active reconstruction traces
        recon_mask = (df["source_type"] == FlowStore.SOURCE_REC) & (df["validation"] != "rejected")
        recon_rows = df[recon_mask].dropna(subset=["volume"])
        
        if recon_rows.empty:
            empty_series = pd.Series(np.nan, index=obs_indices, dtype="float32")
            return empty_series, empty_series

        # 2. Compute baseline EXACTLY ONCE per unique spatial-temporal coordinate
        lookup_map = (
            recon_rows.groupby(["road_section_id", "timestamp", "vehicle_type"], observed=True)
            .apply(self._calc_weighted_mean_and_std, include_groups=False)
        )

        # 3. Broadcast the unique baselines out to the investigated indices
        obs_coords = obs_rows[["road_section_id", "timestamp", "vehicle_type"]].reset_index()
        baseline = obs_coords.merge(
            lookup_map, 
            on=["road_section_id", "timestamp", "vehicle_type"], 
            how="left"
        ).set_index('index')[["baseline_mean", "baseline_std"]]
        baseline = baseline.rename(columns={"baseline_mean": "mean", "baseline_std": "std"})
        
        # 4. Calculate Z Scores
        if self.use_errors and "volume_err" in obs_rows.columns:
            # Mathematical variant that integrates observation-level uncertainties
            obs_err = obs_rows["volume_err"]
        else:
            #obs_err = pd.Series(np.nan, index=baseline.index)
            obs_err = obs_rows["volume"] * 0.001  # Set observation error to small noise floor

        z_scores = utils.compute_z_score(
            observed_val=obs_rows["volume"],
            observed_err=obs_err,
            baseline_val=baseline["mean"],
            baseline_err=baseline["std"]
        )

        # 5. Determine validation status
        anomalies = z_scores[z_scores.abs() > self.z_threshold].index
        conforming = z_scores[z_scores.abs() <= self.z_threshold].index
        return conforming, anomalies


class ZScoreModifiedValidator(BaseValidator):
    """
    Modified Z-Score validation using Median and Median Absolute Deviation (MAD).
    """
    def __init__(self, z_threshold: float = 1.96, use_errors: bool = True):
        # 3.5 is the standard recommended threshold for modified Z-score outliers
        self.z_threshold = z_threshold
        self.use_errors = use_errors

    def validate(self, store: FlowStore, obs_indices: pd.Index) -> Tuple[pd.Index, pd.Index]:
        df = store.dataframe
        obs_rows = df.loc[obs_indices]
        
        # 1. Isolate valid active reconstruction traces
        recon_mask = (df["source_type"] == FlowStore.SOURCE_REC) & (df["validation"] != "rejected")
        recon_rows = df[recon_mask].dropna(subset=["volume"])
        
        if recon_rows.empty:
            empty_series = pd.Series(np.nan, index=obs_indices, dtype="float32")
            return empty_series, empty_series

        # 2. Compute baseline EXACTLY ONCE per unique spatial-temporal coordinate
        lookup_map = (
            recon_rows.groupby(["road_section_id", "timestamp", "vehicle_type"], observed=True)
            .apply(self._calc_median_and_mad, include_groups=False)
        )

        # 3. Broadcast the unique baselines out to the investigated indices
        obs_coords = obs_rows[["road_section_id", "timestamp", "vehicle_type"]].reset_index()
        baseline = obs_coords.merge(
            lookup_map, 
            on=["road_section_id", "timestamp", "vehicle_type"], 
            how="left"
        ).set_index('index')[["baseline_median", "baseline_mad"]]
        baseline = baseline.rename(columns={"baseline_median": "median", "baseline_mad": "mad"})
        

        # 4. Calculate Modified Z Scores
        if self.use_errors and "volume_err" in obs_rows.columns:
            # Mathematical variant that integrates observation-level uncertainties
            obs_err = obs_rows["volume_err"]
        else:
            #obs_err = pd.Series(np.nan, index=baseline.index)
            obs_err = obs_rows["volume"] * 0.001  # Set observation error to small noise floor

        modified_z_scores = 0.6745 * utils.compute_z_score(
            observed_val=obs_rows["volume"],
            observed_err=obs_err,
            baseline_val=baseline["median"],
            baseline_err=baseline["mad"]
        )

        # 5. Determine validation status
        anomalies = modified_z_scores[modified_z_scores.abs() > self.z_threshold].index
        conforming = modified_z_scores[modified_z_scores.abs() <= self.z_threshold].index
        return conforming, anomalies


class RelativeErrorValidator(BaseValidator):
    """
    Validation based on relative percentage difference from consensus baseline mean.
    """
    def __init__(self, max_percent_deviation: float = 2.0, use_errors: bool = True):
        # Max allowed deviation (e.g. 30%)
        self.limit = max_percent_deviation / 100.0
        self.use_errors = use_errors

    def validate(self, store: FlowStore, obs_indices: pd.Index) -> Tuple[pd.Index, pd.Index]:
        df = store.dataframe
        obs_rows = df.loc[obs_indices]
        
        # --- Computing mean baseline --- 
        # 1. Isolate valid active reconstruction traces
        recon_mask = (df["source_type"] == FlowStore.SOURCE_REC) & (df["validation"] != "rejected")
        recon_rows = df[recon_mask].dropna(subset=["volume"])
        
        if recon_rows.empty:
            empty_series = pd.Series(np.nan, index=obs_indices, dtype="float32")
            return empty_series, empty_series

        # 2. Compute baseline EXACTLY ONCE per unique spatial-temporal coordinate
        lookup_map = (
            recon_rows.groupby(["road_section_id", "timestamp", "vehicle_type"], observed=True)
            .apply(self._calc_median_and_mad, include_groups=False)
        )

        # 3. Broadcast the unique baselines out to the investigated indices
        obs_coords = obs_rows[["road_section_id", "timestamp", "vehicle_type"]].reset_index()
        baseline_median = obs_coords.merge(
            lookup_map, 
            on=["road_section_id", "timestamp", "vehicle_type"], 
            how="left"
        ).set_index('index')["baseline_median"]
        

        # 4. Check relative differences
        diff_abs = (obs_rows["volume"] - baseline_median).abs()  
        if self.use_errors and "volume_err" in obs_rows.columns:
            safe_err = obs_rows["volume_err"].fillna(0)
            diff_abs = diff_abs - safe_err

        # 5. Determine validation status
        anomalies = diff_abs[diff_abs > baseline_median*self.limit].index
        conforming = diff_abs[diff_abs <= baseline_median*self.limit].index
        return conforming, anomalies