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

    # Small-sample bias-correction factors for MAD → consistent scale estimator
    # (Rousseeuw & Croux, 1993). Converges to the standard 1.4826 asymptotic factor.
    _MAD_CORRECTION_FACTORS = {
        1: 1.196, 2: 1.495, 3: 1.363, 4: 1.206, 5: 1.200,
        6: 1.140, 7: 1.129, 8: 1.107, 9: 1.093,
    }
    _MAD_ASYMPTOTIC_FACTOR = 1.4826
    
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
        return pd.Series({"baseline": mean, "baseline_error": std})
    
    @staticmethod
    def _calc_median_and_mad(flow_df: pd.DataFrame, poisson_floor_factor: float = 1.0) -> pd.Series:
        volumes = flow_df["volume"].to_numpy(dtype="float64")
        n = len(volumes)

        median = np.median(volumes)
        raw_mad = np.median(np.abs(volumes - median))

        # Bias-correct so MAD is a consistent estimator of std, especially at small n
        c_n = BaseValidator._MAD_CORRECTION_FACTORS.get(n, BaseValidator._MAD_ASYMPTOTIC_FACTOR)
        mad = raw_mad * c_n

        # Poisson-like counting-noise floor, scaled by volume magnitude.
        poisson_floor = poisson_floor_factor * np.sqrt(max(median, 1.0))
        mad = max(mad, poisson_floor)

        return pd.Series({"baseline" : median, "baseline_error": mad})


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
        ).set_index('index')[["baseline", "baseline_error"]]
        baseline = baseline.rename(columns={"baseline": "mean", "baseline_error": "std"})
        
        # 4. Calculate Z Scores
        if self.use_errors and "volume_err" in obs_rows.columns:
            # Mathematical variant that integrates observation-level uncertainties
            obs_err = obs_rows["volume_err"]
        else:
            #obs_err = pd.Series(np.nan, index=baseline.index)
            obs_err = obs_rows["volume"] * 0.  # Set observation error to small noise floor

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
    def __init__(self, z_threshold: float = 3.5, use_errors: bool = True):
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
        ).set_index('index')[["baseline", "baseline_error"]]
        baseline = baseline.rename(columns={"baseline": "median", "baseline_error": "mad"})
        

        # 4. Calculate Modified Z Scores
        if self.use_errors and "volume_err" in obs_rows.columns:
            # Mathematical variant that integrates observation-level uncertainties
            obs_err = obs_rows["volume_err"]
        else:
            #obs_err = pd.Series(np.nan, index=baseline.index)
            obs_err = obs_rows["volume"] * 0.  # Set observation error to small noise floor

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
    def __init__(self, max_percent_deviation: float = 2.0, use_median: bool = True, use_errors: bool = True):
        # Max allowed deviation (e.g. 30%)
        self.limit = max_percent_deviation / 100.0
        self.use_median = use_median
        self.use_errors = use_errors
        self._noise_floor = 10.

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
            .apply(self._calc_median_and_mad if self.use_median else self._calc_weighted_mean_and_std, include_groups=False)
        )

        # 3. Broadcast the unique baselines out to the investigated indices
        obs_coords = obs_rows[["road_section_id", "timestamp", "vehicle_type"]].reset_index()
        baseline = obs_coords.merge(
            lookup_map, 
            on=["road_section_id", "timestamp", "vehicle_type"], 
            how="left"
        ).set_index('index')["baseline"]
        

        # 4. Check relative differences
        diff_abs = (obs_rows["volume"] - baseline).abs()  
        if self.use_errors and "volume_err" in obs_rows.columns:
            safe_err = obs_rows["volume_err"].fillna(0)
            diff_abs = diff_abs - safe_err

        # 5. Determine validation status
        anomalies = diff_abs[diff_abs > (baseline*self.limit).clip(lower=self._noise_floor)].index
        conforming = diff_abs[diff_abs <= (baseline*self.limit).clip(lower=self._noise_floor)].index
        return conforming, anomalies