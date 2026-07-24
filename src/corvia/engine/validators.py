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
    def score(self, store: FlowStore, obs_indices: pd.Index) -> pd.Series:
        """
        Computes a normalized severity score for each observation.

        The score is the underlying test statistic divided by the validator's
        own rejection threshold, so |score| <= 1.0 means "conforming" and
        |score| > 1.0 means "anomalous", regardless of which statistic or
        threshold a particular subclass uses internally. This makes severity
        directly comparable *across* validator types, which greedy/stepwise
        elimination strategies rely on to rank observations by "how bad" they
        are rather than just whether they crossed the line.

        NaN indicates no baseline could be established for that observation
        (e.g. no surviving reconstruction traces at that coordinate) and is
        therefore neither conforming nor anomalous.

        Returns
        -------
        pd.Series
            Float severity scores aligned to obs_indices.
        """
        pass

    @abstractmethod
    def _aggregate_group(self, group: pd.DataFrame) -> pd.Series:
        """
        Reduces the surviving reconstruction rows at ONE
        (road_section_id, timestamp, vehicle_type) coordinate to this
        validator's own ("baseline", "baseline_error") pair.
        """
        pass

    def compute_baseline(self, flow_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes this validator's baseline EXACTLY ONCE per unique
        (road_section_id, timestamp, vehicle_type) coordinate, from the
        currently-surviving reconstruction traces in `flow_df`. This is 
        the single source of truth for "what does this validator consider 
        the consensus to be".

        Returns
        -------
        pd.DataFrame
            Columns ["baseline", "baseline_error"], indexed by
            (road_section_id, timestamp, vehicle_type). Empty (but with the
            right columns) if no valid reconstruction traces exist anywhere
            in `flow_df`.
        """
        # 1. Isolate valid active reconstruction traces
        recon_mask = (flow_df["source_type"] == FlowStore.SOURCE_REC) & (flow_df["validation"] != "rejected")
        recon_rows = flow_df[recon_mask].dropna(subset=["volume"])

        if recon_rows.empty:
            return pd.DataFrame(columns=["baseline", "baseline_error"])

        # 2. Compute baseline EXACTLY ONCE per unique spatial-temporal coordinate
        return (
            recon_rows.groupby(["road_section_id", "timestamp", "vehicle_type"], observed=True)
            .apply(self._aggregate_group, include_groups=False)
        )
    
    @staticmethod
    def _broadcast_baseline(lookup_map: pd.DataFrame, target_rows: pd.DataFrame) -> pd.DataFrame:
        """
        Aligns a per-coordinate baseline table (as returned by
        `compute_baseline`) onto arbitrary target rows — the merge-by-
        coordinate step every subclass's score() previously duplicated.

        Returns
        -------
        pd.DataFrame
            Columns ["baseline", "baseline_error"], aligned to
            target_rows.index. NaN where no baseline exists for that row's
            coordinate (e.g. no surviving reconstruction there).
        """
        if lookup_map.empty:
            return pd.DataFrame(np.nan, index=target_rows.index, columns=["baseline", "baseline_error"])

        obs_coords = target_rows[["road_section_id", "timestamp", "vehicle_type"]].reset_index()
        baseline = obs_coords.merge(
            lookup_map, 
            on=["road_section_id", "timestamp", "vehicle_type"], 
            how="left"
        ).set_index("index")[["baseline", "baseline_error"]]
        return baseline

    def validate(self, store: FlowStore, obs_indices: pd.Index) -> Tuple[pd.Index, pd.Index]:
        """
        Executes validation on the specified target indices by thresholding
        score() at |severity| == 1.0. Subclasses generally shouldn't need to
        override this — implement score() instead.

        Returns
        -------
        conforming_indices : pd.Index
            Indices of observations that conform (verified).
        anomalies_indices : pd.Index
            Indices of observations that do not conform (rejected).
        """
        severity = self.score(store, obs_indices)
        conforming = severity[severity.abs() <= 1.0].index
        anomalies = severity[severity.abs() > 1.0].index
        return conforming, anomalies

    @staticmethod
    def _calc_weighted_mean_and_std(flow_df: pd.DataFrame, median_mixing_fraction: float = 0.) -> pd.Series:
        v = flow_df["volume"].to_numpy(dtype="float64")
        e = flow_df["volume_err"].to_numpy(dtype="float64")
        w = flow_df["weight"].to_numpy(dtype="float64")
        mean, std = utils.weighted_mean_and_error(v, e, w)
        median = np.median(v)
        baseline = mean*(1.-median_mixing_fraction) + median*median_mixing_fraction
        return pd.Series({"baseline": baseline, "baseline_error": std})
    
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
    def __init__(self, z_threshold: float = 1.96, use_errors: bool = True, median_mixing_fraction: float = 0.):
        self.z_threshold = z_threshold
        self.use_errors = use_errors
        if not (0. <= median_mixing_fraction <= 1. ):
            raise ValueError("Median mixing fraction should be a value of the interval [0,1]")
        self.median_mixing_fraction = median_mixing_fraction

    def score(self, store: FlowStore, obs_indices: pd.Index) -> Tuple[pd.Index, pd.Index]:
        df = store.dataframe.copy()
        obs_rows = df.loc[obs_indices]
        
        # A. Compute baselines
        lookup_map = self.compute_baseline(df)
        if lookup_map.empty:
            return pd.Series(np.nan, index=obs_indices, dtype="float32")
        baseline = self._broadcast_baseline(lookup_map, obs_rows)
        
        # B. Calculate Z Scores
        if self.use_errors and "volume_err" in obs_rows.columns:
            # Mathematical variant that integrates observation-level uncertainties
            obs_err = obs_rows["volume_err"]
        else:
            #obs_err = pd.Series(np.nan, index=baseline.index)
            obs_err = obs_rows["volume"] * 0.  # Set observation error to small noise floor

        z_scores = utils.compute_z_score(
            observed_val=obs_rows["volume"],
            observed_err=obs_err,
            baseline_val=baseline["baseline"],
            baseline_err=baseline["baseline_error"]
        )

        # C. Normalize so |severity| > 1.0 <=> anomalous at this validator's threshold
        return z_scores / self.z_threshold

    def _aggregate_group(self, group: pd.DataFrame) -> pd.Series:
        return self._calc_weighted_mean_and_std(group, median_mixing_fraction=self.median_mixing_fraction)

class ZScoreModifiedValidator(BaseValidator):
    """
    Modified Z-Score validation using Median and Median Absolute Deviation (MAD).
    """
    def __init__(self, z_threshold: float = 3.5, use_errors: bool = True, poisson_floor_factor: float = 1.):
        # 3.5 is the standard recommended threshold for modified Z-score outliers
        self.z_threshold = z_threshold
        self.use_errors = use_errors
        self.poisson_floor_factor = poisson_floor_factor

    def score(self, store: FlowStore, obs_indices: pd.Index) -> Tuple[pd.Index, pd.Index]:
        df = store.dataframe.copy()
        obs_rows = df.loc[obs_indices]
        
        # A. Compute baselines
        lookup_map = self.compute_baseline(df)
        if lookup_map.empty:
            return pd.Series(np.nan, index=obs_indices, dtype="float32")
        baseline = self._broadcast_baseline(lookup_map, obs_rows)
        
        # B. Calculate Z Scores
        if self.use_errors and "volume_err" in obs_rows.columns:
            # Mathematical variant that integrates observation-level uncertainties
            obs_err = obs_rows["volume_err"]
        else:
            #obs_err = pd.Series(np.nan, index=baseline.index)
            obs_err = obs_rows["volume"] * 0.  # Set observation error to small noise floor

        modified_z_scores = 0.6745 * utils.compute_z_score(
            observed_val=obs_rows["volume"],
            observed_err=obs_err,
            baseline_val=baseline["baseline"],
            baseline_err=baseline["baseline_error"]
        )

        # C. Normalize so |severity| > 1.0 <=> anomalous at this validator's threshold
        return modified_z_scores / self.z_threshold
    
    def _aggregate_group(self, group: pd.DataFrame) -> pd.Series:
        return self._calc_median_and_mad(group, poisson_floor_factor=self.poisson_floor_factor)


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

    def score(self, store: FlowStore, obs_indices: pd.Index) -> Tuple[pd.Index, pd.Index]:
        df = store.dataframe
        obs_rows = df.loc[obs_indices]
        
        # A. Compute baselines
        lookup_map = self.compute_baseline(df)
        if lookup_map.empty:
            return pd.Series(np.nan, index=obs_indices, dtype="float32")
        baseline = self._broadcast_baseline(lookup_map, obs_rows)
        
        # B. Check relative differences
        diff_abs = (obs_rows["volume"] - baseline).abs()  
        if self.use_errors and "volume_err" in obs_rows.columns:
            safe_err = obs_rows["volume_err"].fillna(0)
            diff_abs = (diff_abs - safe_err).clip(lower=0.)

        # C. Normalize so |severity| > 1.0 <=> anomalous at this validator's threshold.
        # diff_abs is already non-negative, so this ratio is an unsigned severity —
        # fine for magnitude-based ranking, it just carries no over/under direction.
        limit_val = (baseline * self.limit).clip(lower=self._noise_floor)
        return diff_abs / limit_val
    
    def _aggregate_group(self, group: pd.DataFrame) -> pd.Series:
        return self._calc_median_and_mad(group) if self.use_median else self._calc_weighted_mean_and_std(group)