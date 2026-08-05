"""
loader.py
=========
Turn raw high-frequency sensor CSVs into FlowStore-ready records, in
two steps you can run separately or together:

    aggregate()  — raw per-5-min CSVs -> aggregated report (no network needed)
    prepare()    — aggregated report   -> FlowStore-ready long-format table
                    (one row per sensor, period, and vehicle type)

Dependency chain::

    loader.py  →  corvia.framework.network
               →  corvia.network_states.flow
               →  pandas / numpy
"""

from __future__ import annotations

import logging
from typing import Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pandas.tseries.frequencies import to_offset

from corvia.framework.network import Network
from corvia.network_states.flow import FlowStore


def relative_volume_error(rel_err: float) -> Callable[[pd.DataFrame], pd.Series]:
    """
    Build a volume-error function using a fixed relative fraction.

    Parameters
    ----------
    rel_err : float
        Fraction of ``volume`` to use as the error

    Returns
    -------
    callable
        A function ``(df) -> pandas.Series`` suitable for
        :paramref:`FlowDataLoader.volume_err_fn`.
    """
    return lambda df: df["volume"] * rel_err


class VCFlowDataLoader:
    """
    Turn raw sensor exports into FlowStore-ready records.

    Two independent steps:

    :meth:`aggregate`
        Reads :attr:`csv_paths`, computes per-row volumes and
        reconstruction quality, and aggregates to :attr:`agg_freq`
        buckets per sensor. Doesn't touch a network at all.
    :meth:`prepare`
        Takes an aggregated report (freshly computed via
        :meth:`aggregate`, or a previously saved one), maps sensors onto
        road sections via a :class:`~corvia.framework.network.Network`,
        melts ``volume_{vt}`` into one row per vehicle type in
        :attr:`vehicle_types`, and attaches the remaining
        :class:`~corvia.network_states.flow.FlowStore` columns.
        ``screening`` defaults to ``"unknown"`` everywhere.

    Parameters
    ----------
    csv_filenames : list of str, optional
        Paths to the raw measurement CSVs. Only required for :meth:`aggregate` 
    agg_freq : str, optional
        Any pandas offset alias — ``"15min"``, ``"1h"``, ``"6h"``,
        ``"1D"``, ``"1W"``, … Defaults to ``"1D"``.
    pae_factor : float, optional
        Passenger-car-equivalent factor applied to truck volumes when
        computing ``volume_PAE``. Defaults to ``2.0``.
    start, end : str or pandas.Timestamp, optional
        Restrict aggregation to ``[start, end]`` (inclusive). ``None``
        (default, for either bound) keeps everything on that side.
    delimiter : str, optional
        CSV delimiter for the *raw* files read by :meth:`aggregate`.
        Defaults to ``","``.
    vehicle_types : sequence of str, optional
        Which ``volume_{vehicle_type}`` columns :meth:`prepare` melts
        into rows. Defaults to ``("PW", "VR")``. Pass e.g.
        ``("PW", "VR", "TOTAL", "PAE")`` to load composites too — as
        long as the report has those ``volume_*`` columns, which
        :meth:`aggregate` always computes.
    volume_err_fn : callable, optional
        Return a ``pandas.Series`` of per-row volume errors aligned to
        it. Defaults to :func:`relative_volume_error` with a 5%
        relative error.

    Attributes
    ----------
    _csv_filenames : list of str or None
    _agg_freq : str
    _pae_factor : float
    _start : pandas.Timestamp or None
    _end : pandas.Timestamp or None
    _delimiter : str
    _vehicle_types : tuple of str
    _volume_err_fn : callable

    Examples
    --------
    One-shot, no intermediate file::

    >>> loader = FlowDataLoader(csv_paths=["raw.csv"], agg_freq="1h")
    >>> df = loader.prepare(network)
    >>> store = FlowStore.from_dataframe(df)

    Save the aggregated report once, reuse it while iterating on
    ``vehicle_types`` / ``volume_err_fn`` without re-reading raw CSVs::

    >>> loader.aggregate(output_path="data/R2_sensor_report.csv")
    >>> df = loader.prepare(network, report="data/R2_sensor_report.csv")
    """

    #: Columns pulled from the raw export; anything else is dropped
    #: before concatenation to save memory.
    RAW_COLUMNS = [
        "LOCPOST", "TIME_MEASURED", "I2", "I3", "I4", "I5",
        "ingevuld", "rec_kl_gaten", "rec_geslvak", "rec_typed", "BESCH",
    ]
    VOLUME_COLUMNS = ["I2", "I3", "I4", "I5"]
    RECONSTRUCTION_COLUMNS = ["rec_kl_gaten", "rec_geslvak", "rec_typed"]

    def __init__(
        self,
        csv_filenames: Optional[List[str]] = None,
        agg_freq: str = "1D",
        pae_factor: float = 2.0,
        start: Optional[str] = None,
        end: Optional[str] = None,
        delimiter: str = ",",
        vehicle_types: Sequence[str] = ("PW", "VR"),
        volume_err_fn: Optional[Callable[[pd.DataFrame], pd.Series]] = None,
    ) -> None:
        self._csv_filenames: Optional[List[str]] = list(csv_filenames) if csv_filenames is not None else None
        self._agg_freq: str = agg_freq
        self._pae_factor: float = pae_factor
        self._start: Optional[pd.Timestamp] = pd.Timestamp(start) if start else None
        self._end: Optional[pd.Timestamp] = pd.Timestamp(end) if end else None
        self._delimiter: str = delimiter
        self._vehicle_types: Tuple[str, ...] = tuple(vehicle_types)
        self._volume_err_fn: Callable[[pd.DataFrame], pd.Series] = (
            volume_err_fn if volume_err_fn is not None else relative_volume_error(0.05)
        )

    # ------------------------------------------------------------------
    # Interface
    # ------------------------------------------------------------------

    def aggregate(self, output_path: Optional[str] = None, output_delimiter: str = ";") -> pd.DataFrame:
        """
        Read, clean and aggregate :attr:`csv_filenames`.

        Parameters
        ----------
        output_path : str, optional
            When given, also saves the resulting report there. 
            Defaults to ``None`` (no file written).
        output_delimiter : str, optional
            CSV delimiter used when *output_path* is given. 
            Defaults to ``";"``.

        Returns
        -------
        pandas.DataFrame
            One row per sensor (``LOCPOST``) per :attr:`agg_freq`
            bucket. Columns include ``period_start`` (bucket start),
            ``volume_TOTAL`` / ``volume_PW`` / ``volume_VR`` / ``volume_PAE``,
            and the ``pct_*`` reconstruction-quality ratios.
        """
        if not self._csv_filenames:
            raise ValueError("aggregate() needs csv_filenames — none were given to FlowDataLoader().")

        df = self._read_and_clean()
        df = self._filter_period(df, time_column="TIME_MEASURED")
        if df.empty:
            raise ValueError(
                "No records found in the requested start/end range. "
                "Check your CSV timestamps and the start/end arguments."
            )
        df = self._compute_volumes(df)
        grouped = self._group_by_period(df)
        grouped = self._compute_ratios(grouped)

        if output_path:
            grouped.to_csv(output_path, index=False, sep=output_delimiter)
            print(f"> Saved aggregated report to '{output_path}'.")

        return grouped

    def prepare(
        self,
        network: Network,
        report: Optional[Union[str, pd.DataFrame]] = None,
        delimiter: str = ";",
    ) -> pd.DataFrame:
        """
        Turn an aggregated report into a FlowStore-ready, long-format table.

        Parameters
        ----------
        network : Network
            Built network, used to map ``LOCPOST`` -> ``road_section_id``
            via each section's sensors.
        report : str or pandas.DataFrame, optional
            An already-aggregated dataframe or a path to a saved report
            CSV. ``None`` (default) calls :meth:`aggregate` internally.
        delimiter : str, optional
            CSV delimiter, used only when *report* is a path. 
            Defaults to ``";"``.

        Returns
        -------
        pandas.DataFrame
            One row per sensor, period, and vehicle type, with every
            :attr:`~corvia.network_states.flow.FlowStore.COLUMNS` field
            present (``screening`` defaulted to ``"unknown"``) plus the
            ``pct_*`` quality columns — ready to review/edit and hand to
            :meth:`~corvia.network_states.flow.FlowStore.from_dataframe`.
        """
        if report is None:
            df = self.aggregate()
        else:
            try:
                df = self._read_report(report, delimiter)
            except FileNotFoundError:
                print(f"> WARNING: {report} not found. Aggregating raw data and saving it to requested file.")
                df = self.aggregate(output_path=report, output_delimiter=delimiter)

        #df = self.aggregate() if report is None else self._read_report(report, delimiter)
        df = self._map_sections(df, network)
        df = self._melt_vehicle_types(df)
        return self._finalize(df)

    # ------------------------------------------------------------------
    # aggregate() steps
    # ------------------------------------------------------------------

    def _read_and_clean(self) -> pd.DataFrame:
        """
        Read every configured CSV and concatenate them into one frame.

        Returns
        -------
        pandas.DataFrame
            Only :attr:`RAW_COLUMNS` are kept, with ``LOCPOST`` cleaned
            to a stripped string.
        """
        loaded = []
        for filename in self._csv_filenames:
            print(f"> Reading sensor data from: '{filename}'...")
            raw = pd.read_csv(filename, delimiter=self._delimiter)
            raw["TIME_MEASURED"] = pd.to_datetime(raw["TIME_MEASURED"], format="mixed")
            existing_cols = [c for c in self.RAW_COLUMNS if c in raw.columns]
            loaded.append(raw[existing_cols])

        print("> Merging raw datasets...")
        df = pd.concat(loaded, ignore_index=True)

        df["LOCPOST"] = (
            df["LOCPOST"]
            .dropna()
            .astype(float, errors="ignore")
            .astype(int, errors="ignore")
            .astype(str)
            .str.strip()
        )
        return df

    @staticmethod
    def _find_common_timeinterval(df: pd.DataFrame, time_column: str) -> Optional[pd.Timedelta]:
        """
        Find the common time interval between consecutive rows.

        Parameters
        ----------
        df : pandas.DataFrame
        time_column : str
            Name of the column containing timestamps.

        Returns
        -------
        pandas.Timedelta or None
            ``None`` if the dataframe has fewer than 2 rows, 
            or if no common interval can be found.
        """
        if len(df) < 2:
            return None
        df = df.sort_values(by=['LOCPOST', time_column])
        diffs = df.groupby('LOCPOST')[time_column].diff().dropna().unique()
        return diffs[0] if len(diffs) == 1 else None

    def _filter_period(self, df: pd.DataFrame, time_column: str, dT: Optional[pd.Timedelta] = None) -> pd.DataFrame:
        """
        Restrict to ``[start, end]``.

        Parameters
        ----------
        df : pandas.DataFrame
        time_column : str
            Name of the column containing timestamps.
        dT : pandas.Timedelta, optional
            The common time interval between consecutive rows.

        Returns
        -------
        pandas.DataFrame
        """
        dT = self._find_common_timeinterval(df, time_column) if dT is None else dT
        if self._start is not None:
            df = df[df[time_column] >= self._start]
        if self._end is not None:
            df = df[df[time_column] < self._end] if dT is None else df[df[time_column] <= self._end + dT]
        return df.copy()

    def _compute_volumes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute per-row volumes and reconstruction weights.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        df[self.VOLUME_COLUMNS] = df[self.VOLUME_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0)
        df["total_volume"] = df[self.VOLUME_COLUMNS].sum(axis=1)
        df["PW_volume"] = df[["I2", "I3"]].sum(axis=1)
        df["VR_volume"] = df[["I4", "I5"]].sum(axis=1)
        df["PAE_volume"] = df["PW_volume"] + self._pae_factor * df["VR_volume"]

        df["ingevuld"] = df["ingevuld"].astype(bool)

        for col in self.RECONSTRUCTION_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0) / 100.0

        df["reconstruction_ratio"] = df[self.RECONSTRUCTION_COLUMNS].sum(axis=1).clip(upper=1.0)
        df["vol_reconstructed"] = df["total_volume"] * df["reconstruction_ratio"]
        for col in self.RECONSTRUCTION_COLUMNS:
            df[f"vol_{col}"] = df["total_volume"] * df[col]

        return df

    def _group_by_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Group by sensor and :attr:`agg_freq` bucket, summing volumes.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
            Grouped frame with a ``timestamp`` column (renamed from
            the ``TIME_MEASURED`` bucket key).
        """
        print(f"> Aggregating data at '{self._agg_freq}' resolution per sensor...")
        grouped = df.groupby(
            ["LOCPOST", pd.Grouper(key="TIME_MEASURED", freq=self._agg_freq)]
        ).agg(
            total_rows=("TIME_MEASURED", "count"),
            reconstructed_rows=("ingevuld", "sum"),
            volume_TOTAL=("total_volume", "sum"),
            volume_PW=("PW_volume", "sum"),
            volume_VR=("VR_volume", "sum"),
            volume_PAE=("PAE_volume", "sum"),
            vol_rec_total=("vol_reconstructed", "sum"),
            vol_rec_kl_gaten=("vol_rec_kl_gaten", "sum"),
            vol_rec_geslvak=("vol_rec_geslvak", "sum"),
            vol_rec_typed=("vol_rec_typed", "sum"),
            avg_availability=("BESCH", "mean"),
        ).reset_index()
        return grouped.rename(columns={"TIME_MEASURED": "timestamp"})

    def _compute_ratios(self, grouped: pd.DataFrame) -> pd.DataFrame:
        """
        Derive the ``pct_*`` reconstruction-quality ratios.

        Parameters
        ----------
        grouped : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
        """
        grouped["pct_rows_reconstructed"] = (
            grouped["reconstructed_rows"] / grouped["total_rows"]
        ) * 100.0
        grouped["pct_vol_reconstructed"] = np.where(
            grouped["volume_TOTAL"] > 0,
            (grouped["vol_rec_total"] / grouped["volume_TOTAL"]) * 100.0,
            0.0,
        )
        for method in ["kl_gaten", "geslvak", "typed"]:
            grouped[f"pct_vol_reco_{method}"] = np.where(
                grouped["volume_TOTAL"] > 0,
                (grouped[f"vol_rec_{method}"] / grouped["volume_TOTAL"]) * 100.0,
                0.0,
            )

        pct_cols = [c for c in grouped.columns if "pct" in c]
        grouped[pct_cols] = grouped[pct_cols].round(1)
        return grouped

    # ------------------------------------------------------------------
    # prepare() steps
    # ------------------------------------------------------------------

    def _read_report(self, report: Union[str, pd.DataFrame], delimiter: str) -> pd.DataFrame:
        """
        Normalise an already-aggregated report into a working dataframe.

        Parameters
        ----------
        report : str or pandas.DataFrame
        delimiter : str

        Returns
        -------
        pandas.DataFrame
        """
        df = report.copy() if isinstance(report, pd.DataFrame) else pd.read_csv(report, delimiter=delimiter)
        df["LOCPOST"] = df["LOCPOST"].astype(str).str.strip()
        return df

    def _map_sections(self, df: pd.DataFrame, network: Network) -> pd.DataFrame:
        """
        Map each row's ``LOCPOST`` to a ``road_section_id`` via the network.

        Parameters
        ----------
        df : pandas.DataFrame
        network : Network

        Returns
        -------
        pandas.DataFrame
            Rows whose sensor isn't attached to any section in
            *network* are dropped and logged via
            ``logging.getLogger(__name__).warning``.
        """
        print("> Mapping sensors to road sections...")
        sensor_to_section = {
            sensor.location_id: sec_id
            for sec_id, section in network.sections.items()
            for sensor in section.iter_sensors()
        }
        df["road_section_id"] = df["LOCPOST"].map(sensor_to_section)

        initial_count = len(df)
        df = df.dropna(subset=["road_section_id"])
        skipped = initial_count - len(df)
        if skipped > 0:
            print(f"> WARNING: Skipped {skipped} sensors from report (not in network).")
        return df

    def _melt_vehicle_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Turn one row per sensor-period into one row per sensor-period-vehicle_type.

        Parameters
        ----------
        df : pandas.DataFrame
            Must contain ``volume_{vt}`` for every *vt* in
            :attr:`vehicle_types`.

        Returns
        -------
        pandas.DataFrame
            All other columns (``pct_*``, ``road_section_id``,
            ``timestamp``, …) are duplicated onto every vehicle-type row
            of the same sensor-period, since those diagnostics describe
            the interval, not a specific vehicle type.

        Raises
        ------
        ValueError
            If a required ``volume_{vt}`` column is missing.
        """
        print("> Melting vehicle types...")
        missing = [vt for vt in self._vehicle_types if f"volume_{vt}" not in df.columns]
        if missing:
            raise ValueError(f"Report has no volume_{{vt}} column for: {missing}")

        melted = [
            df.assign(vehicle_type=vt, volume=df[f"volume_{vt}"])
            for vt in self._vehicle_types
        ]
        return pd.concat(melted, ignore_index=True)

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Attach the remaining FlowStore columns.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.DataFrame
            ``screening`` is set to ``"unknown"`` for every row —
            deliberately not a judgement call this class makes; see the
            class docstring.
        """
        print("> Finalizing flow data format...")
        df["volume_err"] = self._volume_err_fn(df)
        df["timestamp"] = self._resolve_timestamp(df)
        df["source_type"] = FlowStore.SOURCE_OBS
        df["source_id"] = "sensor:" + df["LOCPOST"]
        df["weight"] = 1.0
        df["validation"] = "pending"
        df["screening"] = "unknown"
        return df

    @staticmethod
    def _resolve_timestamp(df: pd.DataFrame) -> pd.Series:
        """
        Pick a representative timestamp for each row.

        Parameters
        ----------
        df : pandas.DataFrame

        Returns
        -------
        pandas.Series

        Notes
        -----
        Prefers ``timestamp`` (already computed by :meth:`aggregate`),
        falls back to ``period_start``, and finally to midnight of 
        ``DATE`` for reports produced by the original, daily-only script.

        Raises
        ------
        ValueError
            If the report has none of those columns.
        """
        if "timestamp" in df.columns:
            return pd.to_datetime(df["timestamp"])
        if "period_start" in df.columns:
            return pd.to_datetime(df["period_start"])
        if "DATE" in df.columns:
            return pd.to_datetime(df["DATE"])
        raise ValueError(
            "Report has none of 'timestamp', 'period_start', or 'DATE' — "
            "can't determine a timestamp for these records."
        )