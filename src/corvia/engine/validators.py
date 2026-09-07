"""
validators.py
==============
Statistical validation strategies for network baseline consensus.

Every validator implements the :class:`BaseValidator` interface and can be
configured to run in one of two roles via :meth:`~BaseValidator.run_as_rejector`
/ :meth:`~BaseValidator.run_as_acceptor`:

- **Rejector** — flags statistical outliers against a self-computed baseline.
  Runs iteratively inside :class:`~corvia.engine.flow_resolver.FlowResolver`.
- **Acceptor** — verifies or dismisses observations against an already
  *published* (resolved) baseline. Runs once, after the Rejector converges.

All validators expose severity on the same normalized scale
(``severity <= 1.0`` conforming, ``severity > 1.0`` anomalous), so results
are comparable across validator types — see :meth:`BaseValidator.severity_score`.

Dependency chain::

    validators.py  →  corvia.network_states.flow.FlowStore
                   →  corvia.utils
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Tuple, Optional, TYPE_CHECKING
from functools import cached_property
import copy

import numpy as np
import pandas as pd
from scipy import stats

from corvia import utils
from corvia.network_states.flow import FlowStore

from corvia.logger import get_logger
logger = get_logger(__name__)

if TYPE_CHECKING:
    import logging

class BaseValidator(ABC):
    """
    Abstract base class for all network baseline validation strategies.

    Concrete subclasses implement :meth:`severity_score` (how "bad" an
    observation is, on the shared severity scale) and :meth:`_aggregate_group`
    (how a group of surviving reconstruction traces collapses into a single
    ``("baseline", "baseline_error")`` pair). Everything else — baseline
    caching, coordinate broadcasting, pass/fail thresholding — is provided here
    so subclasses only need to supply the statistics.

    Notes
    -----
    Most validators are Rejector-shaped by default; only a subclass that
    overrides :meth:`run_as_acceptor` supports the Acceptor role (see
    :meth:`run_as_acceptor`).
    """

    # Small-sample bias-correction factors for MAD → consistent scale estimator
    # (Rousseeuw & Croux, 1993). Converges to the standard 1.4826 asymptotic factor.
    _MAD_CORRECTION_FACTORS = {
        1: 1.196, 2: 1.495, 3: 1.363, 4: 1.206, 5: 1.200,
        6: 1.140, 7: 1.129, 8: 1.107, 9: 1.093,
    }
    _MAD_ASYMPTOTIC_FACTOR = 1.4826
    _SE_MEDIAN_FACTOR = np.sqrt(np.pi / 2)

    @cached_property
    def logger(self) -> logging.Logger:
        """
        logging.Logger : Instance logger, named after the concrete subclass.
        Lazily created and cached on first access.
        """
        return get_logger(f"{self.__class__.__module__}.{self.__class__.__name__}")


    def run_as_rejector(self) -> BaseValidator:
        """
        Configures this validator for use as a Rejector. Most validators are
        Rejector-shaped by default (self-computed baseline, no resolved-output
        dependency), so the base implementation is a no-op. Override to raise
        if a subclass's statistics are only valid as an Acceptor.

        Returns
        -------
        BaseValidator
            A copy of self, for inline chaining at construction/wiring time.
        """
        return copy.copy(self)

    def run_as_acceptor(self, baseline_source_id: Optional[str] = None) -> BaseValidator:
        """
        Configures this validator for use as an Acceptor. Unlike
        run_as_rejector, this raises by default — most validators are 
        NOT statistically sound as Acceptors; only classes that override 
        this method to reconfigure themselves appropriately support the role.

        Parameters
        ----------
        baseline_source_id : str, optional
            Identifier of the published resolved baseline to validate against.
            Defaults to ``None``.

        Returns
        -------
        BaseValidator
            A copy of self, configured as an Acceptor.

        Raises
        ------
        TypeError
            Always, unless overridden by a subclass.
        """
        error_msg = (
            f"{type(self).__name__} cannot be used as an Acceptor. Its test "
            f"structure is only statistically valid for rejection (failing to "
            f"reject 'obs = baseline' is absence of evidence against the "
            f"observation, not positive evidence it's trustworthy)."
        )
        self.logger.error(error_msg)
        raise TypeError(error_msg)

    @abstractmethod
    def severity_score(self, store: FlowStore, obs_indices: pd.Index) -> pd.Series:
        """
        Computes a severity for each observation.

        The score from the underlying test statistic compared to the validator's
        own rejection critarion; severity <= 1.0 means "conforming" and
        severity > 1.0 means "anomalous", regardless of which test statistic 
        a particular subclass uses internally. This makes severity directly 
        comparable *across* validator types, which greedy/stepwise elimination 
        strategies rely on to rank observations by "how bad" they are rather 
        than just whether they crossed the line.

        NaN indicates no baseline could be established for that observation
        (e.g. no surviving reconstruction traces at that coordinate) and is
        therefore neither conforming nor anomalous.

        Parameters
        ----------
        store : FlowStore
            The flow data store to validate against.
        obs_indices : pd.Index
            Row indices (into ``store.dataframe``) of the observations to score.

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

        Parameters
        ----------
        group : pd.DataFrame
            Rows sharing one (road_section_id, timestamp, vehicle_type)
            coordinate, already filtered to surviving reconstruction traces.

        Returns
        -------
        pd.Series
            Index: ``"baseline"``, ``"baseline_error"``, ``"n_effective"``.
        """
        pass

    def compute_baseline(self, flow_df: pd.DataFrame) -> pd.DataFrame:
        """
        Computes this validator's baseline EXACTLY ONCE per unique
        (road_section_id, timestamp, vehicle_type) coordinate, from the
        currently-surviving reconstruction traces in `flow_df`. This is 
        the single source of truth for "what does this validator consider 
        the consensus to be".

        Parameters
        ----------
        flow_df : pd.DataFrame
            Full flow dataframe (typically ``store.dataframe``).
        
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
            self.logger.debug(f"{type(self).__name__}.compute_baseline: no surviving reconstruction traces to aggregate.")
            return pd.DataFrame(columns=["baseline", "baseline_error", "n_effective"])

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
        coordinate step every subclass's severity_score() previously duplicated.

        Parameters
        ----------
        lookup_map : pd.DataFrame
            Per-coordinate baseline table, as returned by :meth:`compute_baseline`.
        target_rows : pd.DataFrame
            Observation rows to align the baseline onto. Must contain
            ``road_section_id``, ``timestamp``, ``vehicle_type`` columns.
        
        Returns
        -------
        pd.DataFrame
            Columns ["baseline", "baseline_error", "n_effective"], aligned to
            target_rows.index. NaN where no baseline exists for that row's
            coordinate (e.g. no surviving reconstruction there).
        """
        if lookup_map.empty:
            return pd.DataFrame(np.nan, index=target_rows.index, columns=["baseline", "baseline_error", "n_effective"])

        obs_coords = target_rows[["road_section_id", "timestamp", "vehicle_type"]].reset_index()
        baseline = obs_coords.merge(
            lookup_map, 
            on=["road_section_id", "timestamp", "vehicle_type"], 
            how="left"
        ).set_index("index")[["baseline", "baseline_error", "n_effective"]]
        return baseline

    def validate(self, store: FlowStore, obs_indices: pd.Index) -> Tuple[pd.Index, pd.Index]:
        """
        Executes validation on the specified target indices by thresholding
        severity_score() at |severity| == 1.0. Subclasses generally shouldn't need to
        override this — implement severity_score() instead.

        Parameters
        ----------
        store : FlowStore
            The flow data store to validate against.
        obs_indices : pd.Index
            Row indices of the observations to validate.
        
        Returns
        -------
        conforming_indices : pd.Index
            Indices of observations that conform (verified).
        anomalies_indices : pd.Index
            Indices of observations that do not conform (rejected).
        """
        severity = self.severity_score(store, obs_indices)
        idx_pass = severity[severity.abs() <= 1.0].index
        idx_fail = severity[severity.abs() > 1.0].index
        self.logger.debug(f"{type(self).__name__}.validate: {len(idx_pass)} conforming, {len(idx_fail)} anomalous, out of {len(obs_indices)}.")
        return idx_pass, idx_fail

    @staticmethod
    def _calc_weighted_mean_and_std(flow_df: pd.DataFrame, median_mixing_fraction: float = 0.) -> pd.Series:
        """
        Weighted mean/std baseline, optionally mixed with the plain median.

        Parameters
        ----------
        flow_df : pd.DataFrame
            Must contain ``volume``, ``volume_err``, ``weight`` columns.
        median_mixing_fraction : float, default = 0.
            Blend factor between the weighted mean (``0.``) and the plain
            median (``1.``) for the returned baseline.

        Returns
        -------
        pd.Series
            Index: ``"baseline"``, ``"baseline_error"``, ``"n_effective"``.
        """
        v = flow_df["volume"].to_numpy(dtype="float64")
        e = flow_df["volume_err"].to_numpy(dtype="float64")
        w = flow_df["weight"].to_numpy(dtype="float64")
        mean, std, n_eff = utils.weighted_mean_and_error(v, e, w, return_n_effective=True, label="validator")
        median = np.median(v)
        baseline = mean*(1.-median_mixing_fraction) + median*median_mixing_fraction
        return pd.Series({"baseline": baseline, "baseline_error": std, "n_effective": n_eff})
    
    @staticmethod
    def _calc_median_and_mad(flow_df: pd.DataFrame, poisson_floor_factor: float = 1.0) -> pd.Series:
        """
        Median/MAD baseline, bias-corrected and Poisson-floored.

        Parameters
        ----------
        flow_df : pd.DataFrame
            Must contain a ``volume`` column.
        poisson_floor_factor : float, default = 1.0
            Scales the counting-noise floor applied to the MAD-based std
            estimate, to avoid underestimating error at low volumes.

        Returns
        -------
        pd.Series
            Index: ``"baseline"``, ``"baseline_error"``, ``"n_effective"``.

        Notes
        -----
        The MAD → std conversion is bias-corrected via
        :attr:`BaseValidator._MAD_CORRECTION_FACTORS` at small ``n``, converging
        to the standard 1.4826 asymptotic factor.
        """
        volumes = flow_df["volume"].to_numpy(dtype="float64")
        n = len(volumes)

        median = np.median(volumes)
        mad = np.median(np.abs(volumes - median))

        # Bias-correct so MAD is a consistent estimator of std, especially at small n
        c_n = BaseValidator._MAD_CORRECTION_FACTORS.get(n, 1.) * BaseValidator._MAD_ASYMPTOTIC_FACTOR
        std_estimate = mad * c_n

        # Poisson-like counting-noise floor, scaled by volume magnitude.
        poisson_floor = poisson_floor_factor * np.sqrt(max(median, 1.0))
        std_estimate = max(std_estimate, poisson_floor)
        median_error = BaseValidator._SE_MEDIAN_FACTOR * std_estimate / np.sqrt(max(n, 1))

        return pd.Series({"baseline" : median, "baseline_error": median_error, "n_effective": n})


class StatisticalRejector(BaseValidator):
    """
    Abstract base class for validators that use a statistical test to compare
    observations against a baseline. Subclasses should implement the p-value 
    computation for the specific test statistic they use, with the null hypothesis
    being "observation is consistent with the baseline". The severity score is then
    derived from the p-value and the validator's significance level.

    Parameters
    ----------
    significance_level : float, default = 0.05
        Significance threshold; severity == 1.0 corresponds to
        ``p_value == significance_level``.
    two_tailed : bool, default = True
        Whether the underlying test is two-tailed. When ``True``, severity
        carries a sign from :meth:`p_value`'s ``z_sign`` (over/under
        baseline); when ``False``, severity is unsigned.
    """

    def __init__(self, significance_level: float = 0.05, two_tailed: bool = True):
        self._significance_level = significance_level
        self._confidence_level = 1.0 - significance_level
        self._two_tailed = two_tailed

    def severity_score(self, store: FlowStore, obs_indices: pd.Index) -> pd.Series:
        """
        Converts the subclass's p-value into a signed, normalized severity.

        Parameters
        ----------
        store : FlowStore
        obs_indices : pd.Index

        Returns
        -------
        pd.Series
            ``-log10(p) / -log10(significance_level)``, signed by ``z_sign``
            when :attr:`_two_tailed`, unsigned otherwise.
        """
        p_values, z_sign = self.p_value(store, obs_indices)
        if self._two_tailed:
            directions = np.where(z_sign < 0, -1, 1)
        else:
            directions = np.ones_like(p_values, dtype="float64")

        # Compute log10 safely only where p > 0
        with np.errstate(divide='ignore'):
            return directions * -np.log10(p_values) / -np.log10(self._significance_level)

    @abstractmethod
    def p_value(self, store: FlowStore, obs_indices: pd.Index) -> Tuple[pd.Series, pd.series]:
        """
        Computes the test's p-value and directional sign for each observation.

        Parameters
        ----------
        store : FlowStore
        obs_indices : pd.Index

        Returns
        -------
        p_values : pd.Series
            P-value of the test statistic, aligned to obs_indices.
        z_sign : pd.Series
            Sign of the standardized difference (obs vs. baseline); used to
            direct severity when the test is two-tailed.
        """
        pass

class ZScoreRejector(StatisticalRejector):
    """
    Standard Z-Score rejection test comparing observations to a weighted baseline mean.

    Parameters
    ----------
    significance_level : float, default = 0.1
    two_tailed : bool, default = True
    use_errors : bool, default = True
        Whether to fold ``volume_err`` into the standardized difference,
        or treat observations as exact.
    median_mixing_fraction : float, default = 0.
        Forwarded to :meth:`~BaseValidator._calc_weighted_mean_and_std`.
        Must lie in [0, 1].

    Raises
    ------
    ValueError
        If *median_mixing_fraction* is outside [0, 1].
    """
    def __init__(
        self, 
        significance_level: float = 0.1,
        two_tailed: bool = True,
        use_errors: bool = True, 
        median_mixing_fraction: float = 0.
    ):
        super().__init__(significance_level=significance_level, two_tailed=two_tailed)
        self._use_errors = use_errors
        if not (0. <= median_mixing_fraction <= 1. ):
            error_msg = f"Invalid median_mixing_fraction={median_mixing_fraction}: should be a value of the interval [0,1]."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        self._median_mixing_fraction = median_mixing_fraction
    
    def compute_z_scores(self, store: FlowStore, obs_indices: pd.Index) -> pd.Series:
        """
        Computes the standardized difference of each observation from its
        weighted-mean baseline.

        Parameters
        ----------
        store : FlowStore
        obs_indices : pd.Index

        Returns
        -------
        pd.Series
            Z-scores aligned to obs_indices. NaN where no baseline exists.

        Notes
        -----
        When :attr:`_use_errors` is ``False`` (or ``volume_err`` is absent),
        observation error is treated as exactly 0 rather than NaN, so the
        baseline's own error dominates the standardized difference.
        """
        df = store.dataframe.copy()
        obs_rows = df.loc[obs_indices]
        
        # A. Compute baselines
        lookup_map = self.compute_baseline(df)
        if lookup_map.empty:
            return pd.Series(np.nan, index=obs_indices, dtype="float32")
        baseline = self._broadcast_baseline(lookup_map, obs_rows)
        
        # B. Calculate Z Scores
        if self._use_errors and "volume_err" in obs_rows.columns:
            # Mathematical variant that integrates observation-level uncertainties
            obs_err = obs_rows["volume_err"]
        else:
            #obs_err = pd.Series(np.nan, index=baseline.index)
            obs_err = obs_rows["volume"] * 0.  # Set observation error to small noise floor

        z_scores = utils.standardized_difference(
            observed_val=obs_rows["volume"],
            observed_err=obs_err,
            baseline_val=baseline["baseline"],
            baseline_err=baseline["baseline_error"]
        )

        return z_scores

    def p_value(self, store: FlowStore, obs_indices: pd.Index) -> Tuple[pd.Series, pd.Series]:
        """
        Two- or one-tailed normal p-value from the Z-score.

        Parameters
        ----------
        store : FlowStore
        obs_indices : pd.Index

        Returns
        -------
        p_values, z_sign : pd.Series, pd.Series
            See :meth:`StatisticalRejector.p_value`.
        """
        z_scores = self.compute_z_scores(store, obs_indices)
        p_values = 2 * stats.norm.sf(z_scores.abs()) if self._two_tailed else stats.norm.sf(z_scores)
        return pd.Series(p_values, index=z_scores.index), pd.Series(z_scores/z_scores.abs(), index=z_scores.index)

    def _aggregate_group(self, group: pd.DataFrame) -> pd.Series:
        return self._calc_weighted_mean_and_std(group, median_mixing_fraction=self._median_mixing_fraction)


class ZScoreModifiedRejector(ZScoreRejector):
    """
    Modified Z-Score rejection test using Median and Median Absolute Deviation (MAD).

    Parameters
    ----------
    significance_level : float, default = 0.05
    two_tailed : bool, default = True
    use_errors : bool, default = True
    poisson_floor_factor : float, default = 1.
        Forwarded to :meth:`~BaseValidator._calc_median_and_mad`.
    """
    def __init__(
        self, 
        significance_level: float = .05,
        two_tailed: bool = True,
        use_errors: bool = True, 
        poisson_floor_factor: float = 1.
    ):
        super().__init__(significance_level=significance_level, two_tailed=two_tailed)
        self._use_errors = use_errors
        self._poisson_floor_factor = poisson_floor_factor

    def p_value(self, store: FlowStore, obs_indices: pd.Index) -> Tuple[pd.Series, pd.Series]:
        """
        Two- or one-tailed normal p-value from the modified (MAD-scaled) Z-score.

        Parameters
        ----------
        store : FlowStore
        obs_indices : pd.Index

        Returns
        -------
        p_values, z_sign : pd.Series, pd.Series

        Notes
        -----
        Applies the standard 0.6745 MAD → std scale factor to the underlying
        Z-score before computing the p-value.
        """
        mod_z_scores = 0.6745 * self.compute_z_scores(store, obs_indices)  # Scale factor for MAD → std conversion
        p_values = 2 * stats.norm.sf(mod_z_scores.abs()) if self._two_tailed else stats.norm.sf(mod_z_scores)
        return pd.Series(p_values, index=mod_z_scores.index), pd.Series(mod_z_scores/mod_z_scores.abs(), index=mod_z_scores.index)
    
    def _aggregate_group(self, group: pd.DataFrame) -> pd.Series:
        return self._calc_median_and_mad(group, poisson_floor_factor=self._poisson_floor_factor)


class TScoreRejector(ZScoreRejector):
    """
    T-distribution rejection test, using the effective sample size of the
    surviving reconstruction pool for the degrees of freedom.

    Notes
    -----
    Constructor arguments are inherited unchanged from :class:`ZScoreRejector`.
    """  

    def p_value(self, store: FlowStore, obs_indices: pd.Index) -> pd.Series:
        """
        T-distribution p-value, using ``n_effective - 1`` degrees of freedom.

        Parameters
        ----------
        store : FlowStore
        obs_indices : pd.Index

        Returns
        -------
        p_values, z_sign : pd.Series, pd.Series
        """
        df = store.dataframe.copy()
        obs_rows = df.loc[obs_indices]
        
        # A. Compute baselines
        lookup_map = self.compute_baseline(df)
        if lookup_map.empty:
            return pd.Series(np.nan, index=obs_indices, dtype="float32")
        baseline = self._broadcast_baseline(lookup_map, obs_rows)
        
        # B. Calculate Z Scores
        if self._use_errors and "volume_err" in obs_rows.columns:
            # Mathematical variant that integrates observation-level uncertainties
            obs_err = obs_rows["volume_err"]
        else:
            #obs_err = pd.Series(np.nan, index=baseline.index)
            obs_err = obs_rows["volume"] * 0.  # Set observation error to small noise floor

        t_scores = utils.standardized_difference(
            observed_val=obs_rows["volume"],
            observed_err=obs_err,
            baseline_val=baseline["baseline"],
            baseline_err=baseline["baseline_error"]
        )

        dof = baseline["n_effective"] - 1

        p_values = 2 * stats.t.sf(t_scores.abs(), df=dof) if self._two_tailed else stats.t.sf(t_scores, df=dof)
        return pd.Series(p_values, index=t_scores.index), pd.Series(t_scores/t_scores.abs(), index=t_scores.index)


class RelativeErrorValidator(BaseValidator):
    """
    Validation based on relative percentage difference from consensus baseline mean.

    Role-agnostic: this is a symmetric margin test (|obs - baseline| vs. a
    relative threshold), which is a sound basis for either a Rejector
    (loose margin, catches gross outliers) or an Acceptor (tight margin,
    gates calibration-quality data) — unlike the z/t-score sameness tests,
    it does not need a separate equivalence-test formulation for acceptance use.

    Parameters
    ----------
    max_percent_deviation : float, default = 2.0
        Allowed deviation from baseline, as a percentage (e.g. ``2.0`` = 2%).
    noise_floor : float, default = 10.
        Minimum absolute deviation threshold, regardless of baseline
        magnitude — prevents division-by-near-zero at very low volumes.
    use_median : bool, default = False
        Use :meth:`~BaseValidator._calc_median_and_mad` instead of the
        weighted mean/std for the baseline.
    use_errors : bool, default = True
        Whether to subtract combined observation/baseline error from the
        raw difference before comparing to the margin.
    """
    def __init__(self, 
        max_percent_deviation: float = 2.0, 
        noise_floor: float = 10., 
        use_median: bool = False, 
        use_errors: bool = True
    ):
        # Max allowed deviation (e.g. 30%)
        self._limit = max_percent_deviation / 100.0
        self._use_median = use_median
        self._use_errors = use_errors
        self._noise_floor = noise_floor
        self._use_resolved_baseline = False
        self._baseline_source_id = None

    def run_as_rejector(self) -> RelativeErrorValidator:
        new = copy.copy(self)
        new._use_resolved_baseline = False
        new._baseline_source_id = None
        return new

    def run_as_acceptor(self, baseline_source_id: Optional[str] = None) -> RelativeErrorValidator:
        new = copy.copy(self)
        new._use_resolved_baseline = True
        new._baseline_source_id = baseline_source_id
        return new

    def severity_score(self, store: FlowStore, obs_indices: pd.Index) -> pd.Series:
        """
        Severity as the error-adjusted absolute deviation over the allowed margin.

        Parameters
        ----------
        store : FlowStore
        obs_indices : pd.Index

        Returns
        -------
        pd.Series
            ``diff_abs / limit_val``. NaN where no baseline exists.

        Notes
        -----
        When :attr:`_use_errors`, combined observation/baseline error is
        subtracted from the raw absolute difference (clipped at 0) before
        comparing to the margin, so a difference fully explained by measurement
        uncertainty scores as conforming.
        """
        obs_rows = store.dataframe.loc[obs_indices]
        
        # A. Compute baselines
        baseline = None
        if self._use_resolved_baseline:
            baseline = store.lookup_resolved_baseline(obs_rows, source_id=self._baseline_source_id)
            if baseline["baseline"].isna().any():
                n_missing = int(baseline["baseline"].isna().sum())
                self.logger.warning(
                    f"{n_missing}/{len(obs_indices)} observations have no published "
                    f"resolved baseline yet — falling back to a self-computed baseline for the "
                    f"ENTIRE batch of {len(obs_indices)} observations this call, not just the "
                    f"missing ones, to keep severities comparable within this call."
                )
                baseline = None

        if baseline is None:
            lookup_map = self.compute_baseline(store.dataframe)
            if lookup_map.empty:
                return pd.Series(np.nan, index=obs_indices, dtype="float32")
            baseline = self._broadcast_baseline(lookup_map, obs_rows)
        
        # B. Check relative differences
        diff_abs_raw = (obs_rows["volume"] - baseline["baseline"]).abs()
        if self._use_errors and "volume_err" in obs_rows.columns:
            safe_err2 = (obs_rows["volume_err"].fillna(0))**2 + (baseline["baseline_error"].fillna(0))**2
        else:
            safe_err2 = pd.Series(0., index=obs_rows.index)
        diff_abs = (diff_abs_raw - np.sqrt(safe_err2)).clip(lower=0.)

        # C. Normalize so |severity| > 1.0 <=> anomalous at this validator's threshold.
        # diff_abs is already non-negative, so this ratio is an unsigned severity —
        # fine for magnitude-based ranking, it just carries no over/under direction.
        limit_val = (baseline["baseline"].abs() * self._limit).clip(lower=self._noise_floor)
        return diff_abs / limit_val
    
    def _aggregate_group(self, group: pd.DataFrame) -> pd.Series:
        return self._calc_median_and_mad(group) if self._use_median else self._calc_weighted_mean_and_std(group)

class RelativeMarginAcceptor(BaseValidator):
    """
    TOST (Two One-Sided Tests) equivalence test against a relative margin.
 
    Tests two one-sided hypotheses:
        H0_1: true (obs - baseline) <= -margin   (obs meaningfully lower)
        H0_2: true (obs - baseline) >=  margin   (obs meaningfully higher)

    Both must be rejected to declare equivalence — unlike a two-tailed
    sameness test, this requires positive statistical evidence that the
    true difference lies inside (-margin, +margin), not merely an absence
    of evidence against equality. This is the statistically correct
    structure for an Acceptor (as opposed to reusing a sameness test like
    ZScoreRejector with a tighter significance level).

    Parameters
    ----------
    significance_level : float, default = 0.05
    max_percent_deviation : float, default = 2.0
        Equivalence margin, as a percentage of baseline (e.g. ``2.0`` = 2%).
    noise_floor : float, default = 10.0
    use_errors : bool, default = True
    """
 
    def __init__(
        self,
        significance_level: float = 0.05,
        max_percent_deviation: float = 2.0,
        noise_floor: float = 10.0,
        use_errors: bool = True,
    ):
        self._significance_level = significance_level
        self._limit = max_percent_deviation / 100.0
        self._noise_floor = noise_floor
        self._use_errors = use_errors
        self._baseline_source_id = None

    def run_as_rejector(self) -> "RelativeMarginAcceptor":
        error_msg = (
            "RelativeMarginAcceptor cannot be used as a Rejector: Its TOST "
            "equivalence test answers 'is this close enough to trust', not "
            "'is this a statistical outlier'."
        )
        self.logger.error(error_msg)
        raise TypeError(error_msg)

    def run_as_acceptor(self, baseline_source_id: Optional[str] = None) -> "RelativeMarginAcceptor":
        new = copy.copy(self)
        new._baseline_source_id = baseline_source_id
        return new

    def severity_score(self, store: FlowStore, obs_indices: pd.Index) -> pd.Series:
        """
        Severity from the TOST equivalence test's max p-value.

        Parameters
        ----------
        store : FlowStore
        obs_indices : pd.Index

        Returns
        -------
        pd.Series
            Log-transformed so severity shrinks toward 0 as equivalence
            strengthens and exceeds 1.0 once equivalence fails to be
            established at :attr:`_significance_level`. See
            :meth:`equivalence_p_value` for the two underlying p-values.
        """
        p1, p2 = self.equivalence_p_value(store, obs_indices)
        p_max = np.maximum(p1, p2)
    
        # Log-transform the complementary quantity so severity shrinks
        # toward 0 as equivalence gets stronger (p_max -> 0) and grows past
        # 1 as equivalence fails to be established (p_max -> 1), matching
        # the framework-wide -log10(x)/-log10(y) severity convention while
        # preserving severity <= 1.0 <=> p_max <= significance_level.
        with np.errstate(divide='ignore'):
            severity = (
                -np.log10(1.0 - p_max) / -np.log10(1.0 - self._significance_level)
            )
        return severity.astype("float32")
 
    def equivalence_p_value(self, store: FlowStore, obs_indices: pd.Index) -> Tuple[pd.Series, pd.Series]:
        """
        Computes the two one-sided p-values for the TOST equivalence test.
 
        Parameters
        ----------
        store : FlowStore
        obs_indices : pd.Index

        Returns
        -------
        p1, p2 : pd.Series, pd.Series
            p1 tests H0_1 (true diff <= -margin, obs meaningfully lower).
            p2 tests H0_2 (true diff >=  margin, obs meaningfully higher).
            Both aligned to obs_indices. NaN where no baseline exists.
        """
        df = store.dataframe
        obs_rows = df.loc[obs_indices]
 
        # A. Compute baselines
        baseline = store.lookup_resolved_baseline(obs_rows, source_id=self._baseline_source_id)
        if baseline["baseline"].isna().any():
            n_missing = int(baseline["baseline"].isna().sum())
            self.logger.warning(
                f"{n_missing}/{len(obs_indices)} observations have no published "
                f"resolved baseline yet — falling back to a self-computed baseline for the "
                f"ENTIRE batch of {len(obs_indices)} observations this call, not just the "
                f"missing ones, to keep severities comparable within this call."
            )
            baseline = None

        if baseline is None:
            lookup_map = self.compute_baseline(store.dataframe)
            if lookup_map.empty:
                return pd.Series(np.nan, index=obs_indices, dtype="float32")
            baseline = self._broadcast_baseline(lookup_map, obs_rows)
 
        # B. Calculate the p-value of both hypotheses
        diff = obs_rows["volume"] - baseline["baseline"]
        if self._use_errors and "volume_err" in obs_rows.columns:
            obs_err = obs_rows["volume_err"].fillna(0.)
        else:
            obs_err = obs_rows["volume"] * 0.
 
        se = np.sqrt(obs_err**2 + baseline["baseline_error"]**2)
        se = se.replace(0, np.nan).fillna(1e-15)
 
        margin = (baseline["baseline"].abs() * self._limit).clip(lower=self._noise_floor)
 
        # H0_1: true diff <= -margin  (rejected when obs is convincingly ABOVE -margin)
        t1 = (diff + margin) / se
        p1 = pd.Series(stats.norm.sf(t1), index=t1.index)
 
        # H0_2: true diff >= +margin  (rejected when obs is convincingly BELOW +margin)
        t2 = (diff - margin) / se
        p2 = pd.Series(stats.norm.cdf(t2), index=t2.index)
 
        return p1, p2
 
    def _aggregate_group(self, group: pd.DataFrame) -> pd.Series:
        return self._calc_weighted_mean_and_std(group)