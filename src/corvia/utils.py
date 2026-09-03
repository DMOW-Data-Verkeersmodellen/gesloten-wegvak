"""
utils.py
========
Shared statistical helpers for weighted aggregation and standardized
comparison, used across the validation and reconstruction engines.
"""
from typing import Tuple, Union, Optional, TYPE_CHECKING
import sys

import numpy as np
import pandas as pd

from corvia.logger import get_logger
logger = get_logger(__name__)

@staticmethod
def weighted_mean_and_error(
    values_arr, 
    errors_arr=None, 
    weights_arr=None, 
    label: Optional[str] = None,
    return_n_effective: bool = False
) -> Union[Tuple[float, float], Tuple[float, float, float]]:
    """
    Compute the weighted mean and its associated error.

    Parameters
    ----------
    values_arr : array-like
        The values to average.
    errors_arr : array-like, optional
        The errors associated with each value. If ``None``, treated as
        entirely missing (equivalent to passing an all-NaN array).
    weights_arr : array-like, optional
        The weights for each value.
        If None, uniform weights are assumed.
    label : str, optional
        Included in the console warning printed for the zero/single
        non-zero weight edge cases, to identify which call site triggered it.
    return_n_effective : bool, default = False
        If ``True``, also return the effective sample size (Kish's
        effective n) as a third value.

    Returns
    -------
    mean, error : float, float
        The weighted mean and its standard error. When *return_n_effective*
        is ``True``, a third value ``n_effective`` (float) is also returned.
        All-NaN when there are no valid (non-NaN) values.

    Raises
    ------
    ValueError
        If *values_arr*, *errors_arr*, and *weights_arr* don't share the
        same length; if errors are a mix of positive and non-positive/NaN
        (must be all-positive or all-NaN); or if any weight is negative.

    Notes
    -----
    Falls back to simpler formulas at the edges: a single valid value
    returns that value and its own error directly; zero non-zero combined
    weights returns NaN with a console warning; exactly one non-zero
    combined weight returns that value/error directly with a warning.
    Otherwise uses the standard weighted mean with the Cochran (1977)
    weight correction for the variance estimate.
    """
    values = np.asarray(values_arr, dtype=np.float64)
    errors = np.asarray(errors_arr, dtype=np.float64) if errors_arr is not None else np.full_like(values, np.nan)
    weights = np.asarray(weights_arr, dtype=np.float64) if weights_arr is not None else np.ones_like(values)

    if not (len(values) == len(errors) == len(weights)):
        error_msg = "Input arrays 'values_arr', 'errors_arr' and 'weights_arr' must have the same length."
        logger.error(f"[{sys._getframe().f_code.co_name}] " + error_msg)
        raise ValueError(error_msg)
    
    # A. Filter out invalid value entries
    valid_mask = (~np.isnan(values))
    if valid_mask.sum() == 0:
        return (np.nan, np.nan, np.nan) if return_n_effective else (np.nan, np.nan)

    values = values[valid_mask]
    errors = errors[valid_mask]
    weights = weights[valid_mask]

    # B. Validate remaining input
    if not np.isnan(errors).all() and (np.isnan(errors).any() or np.any(errors <= 0)):
        error_msg = "Incorrect volume errors detected: errors should all be positive (> 0) or all be missing (np.nan)."
        logger.error(f"[{sys._getframe().f_code.co_name} - {label}] " + error_msg)
        raise ValueError(error_msg)
    
    if np.isnan(weights).any() or np.any(weights < 0):
        error_msg = "Incorrect weights detected: weights should all be non-negative (>= 0)."
        logger.error(f"[{sys._getframe().f_code.co_name} - {label}] " + error_msg)
        raise ValueError(error_msg)

    # C. Calculate the combined weights considering both the provided weights and the inverse square of errors
    combined_weights = weights.copy()
    if not np.isnan(errors).any():
        combined_weights /= (errors ** 2)

    # SPECIAL CASE: no or one data point
    if len(values) == 1:
        err = errors[0]
        err = np.nan if (err <= 0) else err
        return (values[0], err, 1) if return_n_effective else (values[0], err)
    
    if (combined_weights > 0.).sum() == 0:
        logger.warning(f"[{sys._getframe().f_code.co_name} - {label}] All weights are zero. Returning NaN.")   
        return (np.nan, np.nan, np.nan) if return_n_effective else (np.nan, np.nan)
    if (combined_weights > 0.).sum() == 1:
        logger.warning(f"[{sys._getframe().f_code.co_name} - {label}] Only one non-zero weight. Returning that volume and its error.")
        idx = np.argmax(combined_weights)
        err = errors[idx]
        err = np.nan if (err <= 0) else err
        return (values[idx], err, 1.0) if return_n_effective else (values[idx], err)

    # D. Calculate the weighted mean
    V1 = combined_weights.sum()
    V2 = (combined_weights**2).sum()
    weighted_mean = np.sum(combined_weights * values) / V1

    # E. Calculate the error of the weighted mean
    cochran_weight_correction = (V1**2 - V2) / V1
    weighted_estimated_variance = (combined_weights*(values - weighted_mean)**2).sum() / cochran_weight_correction

    n_effective = V1**2 / V2
    SEM_estimated = np.sqrt(weighted_estimated_variance / n_effective)
    SEM_computed = np.sqrt(1./combined_weights.sum())

    if return_n_effective:
        return weighted_mean, SEM_estimated, n_effective
    return weighted_mean, SEM_estimated

@staticmethod
def standardized_difference(
    observed_val: pd.Series,
    observed_err: pd.Series,
    baseline_val: pd.Series,
    baseline_err: pd.Series
) -> pd.Series:
    """
    Computes point-by-point the difference between observation and baseline,
    normalized by the combined standard deviation.

    Parameters
    ----------
    observed_val : pd.Series
        The raw measurements being tested.
    observed_err : pd.Series
        The absolute standard error associated with the raw measurements.
    baseline_val : pd.Series
        The expected consensus baseline mean values.
    baseline_err : pd.Series
        The standard deviation/uncertainty trail of the baseline consensus.

    Returns
    -------
    pd.Series
        A continuous series of standardized differences aligned with the input series index.
        Contains np.nan where inputs are missing or total variance is non-positive.

    Raises
    ------
    AssertionError
        If the four input series don't all share an identical index.
    """
    # Ensure all series share identical index mapping for perfect vectorization
    index_mismatch = False
    error_msg = "Index mismatch between: "
    if not observed_val.index.equals(observed_err.index): 
        index_mismatch = True
        error_msg += "observation values and errors, "
    if not observed_val.index.equals(baseline_val.index):
        index_mismatch = True 
        error_msg += "observation and baseline values, "
    if not observed_val.index.equals(baseline_err.index): 
        index_mismatch = True
        error_msg += "baseline values and errors."
    if index_mismatch:
        logger.error(f"[{sys._getframe().f_code.co_name}] " + error_msg)
        raise IndexError(error_msg)

    # Compute combined pool variance: σ²_total = σ²_obs + σ²_baseline
    total_variance = ((observed_err ** 2) + (baseline_err ** 2)).replace(0, np.nan).fillna(1e-15)

    # Calculate vectorized score
    sd = (observed_val - baseline_val).abs() / total_variance.pow(0.5)

    # Mask out records where the initial variance was zero or invalid
    valid_variance_mask = total_variance > 0
    sd = sd.where(valid_variance_mask, np.nan)
    
    return sd.astype("float32")