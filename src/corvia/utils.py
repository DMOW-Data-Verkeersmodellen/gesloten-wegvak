from typing import Tuple, TYPE_CHECKING

import numpy as np
import pandas as pd

@staticmethod
def weighted_mean_and_error(values_arr, errors_arr, weights_arr=None, label=None) -> Tuple[float, float]:
    """
    Compute the weighted mean and its associated error.

    Parameters
    ----------
    values_arr : array-like
        The values to average.
    errors_arr : array-like
        The errors associated with each value.
    weights_arr : array-like [optional]
        The weights for each value.
        If None, uniform weights are assumed.

    Returns
    -------
    tuple
        A tuple containing the weighted mean and its error.
    """
    values = np.asarray(values_arr, dtype=np.float64)
    errors = np.asarray(errors_arr, dtype=np.float64)
    if weights_arr is None:
        weights = np.ones_like(values)
    else:
        weights = np.asarray(weights_arr, dtype=np.float64)

    # Ensure that all arrays have the same length
    if not (len(values) == len(errors) == len(weights)):
        raise ValueError("All input arrays must have the same length.")
    
    #Filter out invalid value entries
    valid_mask = (~np.isnan(values))
    
    if not np.any(valid_mask):
        return np.nan, np.nan

    values = values[valid_mask]
    errors = errors[valid_mask]
    weights = weights[valid_mask]

    if len(values) == 1:
        # If there's only one valid value, return it and its error
        err = errors[0]
        err = err if (~np.isnan(err) and err > 0) else np.nan
        return values[0], err
    
    # Check for invalid entries and fall back to the uniform weights/errors if needed
    has_all_errors = np.all(~np.isnan(errors) & (errors > 0))
    has_all_weights = np.all(~np.isnan(weights) & (weights >= 0))
    if not has_all_weights:
        weights = np.ones_like(values)
        print(f"Warning ({label}): Missing or negative weights detected. Defaulting to uniform weights.")
    raw_errors = np.copy(errors)
    if not has_all_errors:
        errors = np.ones_like(values)
        print(f"Warning ({label}): Missing, zero, or negative volume errors detected. Defaulting to uniform errors.")

    # Calculate the combined weights considering both the provided weights and the inverse square of errors
    combined_weights = weights / (errors ** 2)
    if (combined_weights > 0.).sum() == 0:
        print(f"Warning ({label}): All weights are zero. Returning NaN.")   
        return np.nan, np.nan
    if (combined_weights > 0.).sum() == 1:
        print(f"Warning ({label}): Only one non-zero weight. Returning that volume and its error.")
        idx = np.argmax(combined_weights)
        err = raw_errors[idx]
        err = err if (~np.isnan(err) and err > 0) else np.nan
        return values[idx], err

    # Calculate the weighted mean
    weighted_mean = np.sum(combined_weights * values) / np.sum(combined_weights)

    # Calculate the error of the weighted mean
    cochran_weight_correction = ((combined_weights.sum())**2 - (combined_weights**2).sum()) / combined_weights.sum()
    weighted_estimated_variance = (combined_weights*(values - weighted_mean)**2).sum() / cochran_weight_correction

    n_effective = (combined_weights.sum())**2 / (combined_weights**2).sum()
    SEM_estimated = np.sqrt(weighted_estimated_variance / n_effective)
    SEM_computed = np.sqrt(1./combined_weights.sum())

    return weighted_mean, SEM_estimated

@staticmethod
def compute_z_score(
    observed_val: pd.Series,
    observed_err: pd.Series,
    baseline_mean: pd.Series,
    baseline_std: pd.Series
) -> pd.Series:
    """
    Computes point-by-point 1-DoF Z-scores combining observation and baseline variances.

    Parameters
    ----------
    observed_val : pd.Series
        The raw measurements being tested.
    observed_err : pd.Series
        The absolute standard error associated with the raw measurements.
    baseline_mean : pd.Series
        The expected consensus baseline mean values.
    baseline_std : pd.Series
        The standard deviation/uncertainty trail of the baseline consensus.

    Returns
    -------
    pd.Series
        A continuous series of absolute Z-scores aligned with the input series index.
        Contains np.nan where inputs are missing or total variance is non-positive.
    """
    # Ensure all series share identical index mapping for perfect vectorization
    assert observed_val.index.equals(observed_err.index), "Index mismatch on observations."
    assert observed_val.index.equals(baseline_mean.index), "Index mismatch between observations and baseline."
    assert observed_val.index.equals(baseline_std.index), "Index mismatch on baseline variances."

    # Compute combined pool variance: σ²_total = σ²_obs + σ²_baseline
    total_variance = (observed_err ** 2) + (baseline_std ** 2)

    # Calculate vectorized absolute Z-score
    z_scores = (observed_val - baseline_mean).abs() / total_variance.pow(0.5)

    # Mask out records where the initial variance was zero or invalid
    valid_variance_mask = total_variance > 0
    z_scores = z_scores.where(valid_variance_mask, np.nan)
    
    return z_scores.astype("float32")