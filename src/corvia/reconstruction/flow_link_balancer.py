"""
reconstruction/flow_link_balancer.py
==============================
Flow-conservation reconstruction using direct and higher-degree neighbours.

A single class handles both upstream and downstream directions and any degree
of neighbour expansion.  The recursion works by tracking a running sign that
flips each time a "competing" (negative-contribution) branch is followed.
The traversal direction also flips on competing branches to keep the
expansion moving away from the target section and avoid circular dependencies.

Flow balance equations
----------------------
Upstream (entry node):
    S = sum(upstream_in) - sum(upstream_out)

Downstream (exit node):
    S = sum(downstream_out) - sum(downstream_in)

At each additional degree, every neighbour term is expanded one level further
using the same logic.  The sign carried into the recursive call determines
whether that term adds or subtracts from the total.
"""

from __future__ import annotations

from typing import List, Optional, Set, Tuple, TYPE_CHECKING
import numpy as np
import pandas as pd

from corvia.network_states.flow import FlowStore as FlowStore

from corvia.logger import get_logger
logger = get_logger(__name__)

if TYPE_CHECKING:
    from corvia.framework.network import Network
    from corvia.framework.topology import RoadSection


class FlowLinkBalancer():
    """
    Reconstruct a section's volume from neighbouring section volumes using
    flow conservation, extended to arbitrary neighbour depth.

    Recursively expands into upstream/downstream neighbour sections, applying
    the flow balance equations described in the module docstring, until an
    observation is found on every required branch or the expansion is
    abandoned as infeasible (missing data, a cycle, or the network edge).

    Parameters
    ----------
    direction : str
        ``"upstream"`` to reconstruct from the entry node,
        ``"downstream"`` to reconstruct from the exit node.
    min_obs_degree : int, default = 1
        Minimum number of observations to bypass before accepting one as
        the base case — forces the recursion past the nearest sensor(s) to
        average out local effects. Must be >= 1.
    max_link_degree : int or float, default = 1
        Maximum neighbour-expansion depth before giving up. Must be >=
        *min_obs_degree*. ``np.inf`` is coerced to
        ``PRACTICAL_MAX_DEGREE + 3`` with a warning, to prevent unbounded
        graph traversal.
    weight : float, optional
        Override the computed default weight. When not supplied, weight
        decays as ``_base_weight ** degree_effective`` so that
        higher-degree reconstructions are automatically trusted less.
    allow_partial_fallback : bool, default = False
        If a branch is infeasible partway through expansion but this
        section's own observation was available, fall back to that
        observation instead of failing the whole reconstruction.

    Raises
    ------
    ValueError
        If *min_obs_degree* < 1, or *max_link_degree* < *min_obs_degree*.

    Examples
    --------
    >>> m1 = FlowLinkBalancer("upstream", min_obs_degree=1, max_link_degree=1)
    >>> m2 = FlowLinkBalancer("downstream", min_obs_degree=1, max_link_degree=2)
    >>> pipeline.add_method(m1)
    >>> pipeline.add_method(m2)
    """

    # Base weight before degree decay
    _base_weight: float = 0.9
    PRACTICAL_MAX_DEGREE: int = 5

    def __init__(
        self,
        direction: str,
        min_obs_degree: int = 1,
        max_link_degree: int | float = 1,
        weight: Optional[float] = None,
        allow_partial_fallback: bool = False,
    ) -> None:
        self.logger = get_logger(f"{__name__}.{self.__class__.__name__}")

        if np.isinf(max_link_degree):
            max_link_degree = self.PRACTICAL_MAX_DEGREE + 3
            self.logger.warning(
                f"max_link_degree was set to infinity. Coercing to a maximum safe "
                f"computational depth of {max_link_degree} to prevent network traversal explosions."
            )
        max_link_degree = int(max_link_degree)

        if min_obs_degree < 1:
            error_msg = f"Invalid min_obs_degree={min_obs_degree}, must be >= 1."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        if max_link_degree < min_obs_degree:
            error_msg = f"Invalid max_link_degree={max_link_degree} < min_obs_degree={min_obs_degree}."
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        # 2. Issue a warning if the limit is computationally dangerous but allow the user to proceed
        if max_link_degree > self.PRACTICAL_MAX_DEGREE:
            self.logger.warning(
                f"Configured max_link_degree={max_link_degree} exceeds the recommended practical limit "
                f"of {self.PRACTICAL_MAX_DEGREE}. This may result in exponential graph expansion "
                f"slowdowns and a high risk of total reconstruction failure."
            )

        self.direction: str = direction
        self.min_obs_degree: int = min_obs_degree
        self.max_link_degree: int = max_link_degree
        self.weight = weight #if weight is not None else self._base_weight ** min_obs_degree
        self.allow_partial_fallback = allow_partial_fallback

    @property
    def name(self) -> str:
        """str : Unique source label written to the observation store."""
        return f"FlowLinkBalancer_{self.direction}_d{self.min_obs_degree}-{self.max_link_degree}"

    # ------------------------------------------------------------------
    # ReconstructionMethod interface
    # ------------------------------------------------------------------

    def reconstruct(
        self,
        network: Network,
        flow_data: FlowStore,
        periods: List[pd.Timestamp],
        vehicle_types: List[str],
    ) -> pd.DataFrame:
        """
        Produce reconstruction rows for all applicable sections.

        A reconstruction is produced only when the recursive expansion can
        reach a data value for every required branch.  Sections where any
        branch returns ``None`` (missing data or cycle detected) are skipped.

        Parameters
        ----------
        network : Network
            The network on which flows need to be balanced
        flow_data : FlowStore
            Data store holding the flowvolumes of different
            sensors connected to links in the network
        periods : list of Timestamp
            The starting times of the time periods to reconstruct
        vehicle_types : list of str

        Returns
        -------
        pd.DataFrame
            Reconstructed values for sensors in the same
            format as the dataframe of the FlowStore.
        """
        rows = []
        attempted = 0

        for section in network.sections.values():
            for period in periods:
                for vtype in vehicle_types:
                    attempted += 1
                    volume, volume_err, degree_eff = self._estimate(
                        section=section,
                        direction=self.direction,
                        min_obs_degree=self.min_obs_degree,
                        max_link_degree=self.max_link_degree,
                        flow_data=flow_data,
                        period=period,
                        vtype=vtype,
                        visited=frozenset(),
                    )
                    if not np.isnan(volume):
                        weight = self.weight if self.weight is not None else self._base_weight ** degree_eff
                        rows.append(
                            self._make_row(
                                section.section_id, period, vtype, volume, volume_err=volume_err, weight=weight,
                            )
                        )
        self.logger.info(f"{self.name}: reconstructed {len(rows)}/{attempted} (section, period, vtype) combination(s).")
        if not rows:
            self.logger.warning(f"{self.name}: produced no reconstructions this pass.")

        return (
            pd.DataFrame(rows, columns=FlowStore.COLUMNS) if rows else self._empty_result()
        )

    # ------------------------------------------------------------------
    # Recursive core
    # ------------------------------------------------------------------

    def _estimate(
        self,
        section: RoadSection,
        direction: str,
        min_obs_degree: int,
        max_link_degree: int,
        flow_data: FlowStore,
        period: pd.Timestamp,
        vtype: str,
        visited: frozenset,
        numb_obs_skipped: int = 0,
    ) -> Tuple[float, float, float]:
        """
        Recursively estimate a section's signed volume contribution.

        Parameters
        ----------
        section : RoadSection
            The section to estimate.
        direction : str
            ``"upstream"`` or ``"downstream"`` — which node to balance at.
        min_obs_degree : int
            Remaining number of observations to bypass before accepting one
            as a base case. Decremented only when an observation is actually
            skipped at this section.
        max_link_degree : int
            Remaining neighbour-expansion depth. When <= 0, the expansion is
            abandoned as infeasible.
        flow_data : FlowStore
        period : pd.Timestamp
        vtype : str
        visited : frozenset of str
            Section IDs already on the current call path.  Used for cycle
            detection.  A frozenset is used so each recursive branch carries
            its own independent copy without explicit copying.
        numb_obs_skipped : int, default = 0
            Running count of observations bypassed so far on this call path;
            carried through to compute the effective degree of the eventual
            base case.

        Returns
        -------
        tuple of float : volume, volume_err, effective_degree
            The estimated volume and its error.
            ``(np.nan, np.nan, np.nan)`` when the expansion is infeasible 
            (missing data, cycle, or no neighbours to expand from).
        """
        # 1. Cycle guard
        if section.section_id in visited:
            return np.nan, np.nan, np.nan
        # 2. Stop traversal if we have fully exhausted our topological max_link_degree depth[cite: 1]
        if max_link_degree <= 0:
            return np.nan, np.nan, np.nan
        next_max_link_degree = max_link_degree - 1
        
        # 3. Check if an actual observation exists
        volume, volume_err = flow_data.section_consensus(
            section.section_id, period, vtype, source_types=(FlowStore.SOURCE_OBS,)
        )
        own_data_available = not np.isnan(volume)

        def fallback():
            if own_data_available and self.allow_partial_fallback:
                return volume, volume_err, numb_obs_skipped + 1
            return np.nan, np.nan, np.nan

        # 4. Decide if further expansion is needed
        if own_data_available:
            if min_obs_degree <= 0:
                # We have bypassed the required number of observations; return this one!
                return volume, volume_err, numb_obs_skipped + 1
            else:
                # An observation exists, but we must bypass it.
                # Decrement min_obs_degree because we are actively skipping a valid data point.
                next_min_obs_degree = min_obs_degree - 1
                next_numb_obs_skipped = numb_obs_skipped + 1
        else:
            # No observation exists on this link.
            # Do NOT decrement min_obs_degree (we didn't bypass any actual data).
            next_min_obs_degree = min_obs_degree
            next_numb_obs_skipped = numb_obs_skipped

        # 5. Determine which neighbour sets to expand and in which direction
        if direction == "upstream":
            in_neighbours = section.upstream_in 
            out_neighbours = section.upstream_out
            at_network_edge = not in_neighbours  # is this a source node?
        elif direction == "downstream":
            in_neighbours = section.downstream_in
            out_neighbours = section.downstream_out
            at_network_edge = not out_neighbours  # is this sink node?

        # If no neighbours exist at this level, expansion is infeasible
        if at_network_edge:
            return fallback()

        visited = visited | {section.section_id}

        total_in = 0.0
        total_out = 0.0
        total_err2 = 0.0
        degrees_effective = []

        for s in in_neighbours:
            v, v_err, d_eff = self._estimate(
                s, 'upstream', next_min_obs_degree, next_max_link_degree,
                flow_data, period, vtype, visited, numb_obs_skipped=next_numb_obs_skipped,
            )
            if np.isnan(v):
                return fallback()
            total_in += v
            total_err2 += v_err ** 2
            degrees_effective.append(d_eff)

        for s in out_neighbours:
            v, v_err, d_eff = self._estimate(
                s, 'downstream', next_min_obs_degree, next_max_link_degree,
                flow_data, period, vtype, visited, numb_obs_skipped=next_numb_obs_skipped,
            )
            if np.isnan(v):
                return fallback()
            total_out += v
            total_err2 += v_err ** 2
            degrees_effective.append(d_eff)

        if direction == 'upstream':
            total = total_in - total_out
        elif direction == 'downstream':
            total = total_out - total_in
        return total, np.sqrt(total_err2), np.mean(degrees_effective)

    def _make_row( 
        self,
        section_id: str,
        period_start: pd.Timestamp,
        vehicle_type: str,
        volume: float,
        volume_err: float = np.nan,
        weight: float = np.nan,
    ) -> dict: 
        """
        Build a single reconstruction row dictionary.

        Parameters
        ----------
        section_id : str
        period_start : pd.Timestamp
        vehicle_type : str
        volume : float
        volume_err : float, default = np.nan
        weight : float, default = np.nan

        Returns
        -------
        dict
            Ready to be collected into a DataFrame and passed to
            :meth:`~network_states.FlowStore.Append_estimates`.
        """
        return {
            "road_section_id":   section_id,
            "timestamp":        period_start,
            "vehicle_type":     vehicle_type,
            "volume":           volume,
            "volume_err":       volume_err,
            "source_type":      FlowStore.SOURCE_REC,
            "source_id":        self.name,
            "screening":        "NA",
            "validation":       "NA",
            "weight":           weight,
        }

    def _empty_result(self) -> pd.DataFrame:
        """Return an empty DataFrame with the correct schema."""
        return pd.DataFrame(columns=FlowStore.COLUMNS)

    def __repr__(self) -> str:
        return (
            f"FlowBalanceReconstruction("
            f"{self.direction}, "
            f"degree=[{self.min_obs_degree},{self.max_link_degree}])"
        )