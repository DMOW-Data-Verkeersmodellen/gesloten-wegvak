# corvia/engine/flow_resolver.py
from __future__ import annotations
import os
from typing import List, Tuple, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from corvia.network_states.flow import FlowStore
from corvia.utils import weighted_mean_and_error

if TYPE_CHECKING:
    from corvia.framework.network import Network
    from corvia.engine.validators import BaseValidator

class FlowResolver:
    """Manages iterative adaptive multi-pass algorithms over network state models."""
    
    def __init__(
        self, 
        reconstructors: List, 
        validator: BaseValidator,
        max_iterations: int = 3,
        name: Optional[str] = None,
        debug_plot: bool = False,
        plot_dir: str = 'diagnostics',
        greedy_elimination: bool = False,
        greedy_adaptive_batch: bool = False,
        greedy_k_fraction: float = 0.10,
        greedy_k_pool_fraction: float = 0.05,
        greedy_k_max: int = 20,
    ) -> None:
        
        self.reconstructors = reconstructors
        self.validator = validator
        self.max_iterations = max_iterations
        self.name = name
        self.debug_plot = debug_plot
        self.plot_dir = plot_dir
        self.set_plotting_restrictions()
        self.greedy_elimination = greedy_elimination
        self.greedy_adaptive_batch = greedy_adaptive_batch
        if greedy_elimination:
            if not (0 < greedy_k_fraction <= 1):
                raise ValueError(f"greedy_k_fraction must be in (0, 1], got {greedy_k_fraction}")
            if not (0 < greedy_k_pool_fraction <= 1):
                raise ValueError(f"greedy_k_pool_fraction must be in (0, 1], got {greedy_k_pool_fraction}")
            if greedy_k_max < 1:
                raise ValueError(f"greedy_k_max must be >= 1, got {greedy_k_max}")
        self.greedy_k_fraction = greedy_k_fraction
        self.greedy_k_pool_fraction = greedy_k_pool_fraction
        self.greedy_k_max = greedy_k_max

        if self.debug_plot:
            os.makedirs(self.plot_dir, exist_ok=True)

    def _select_rejection_batch_size(self, n_anomalies: int, n_pool: int) -> int:
        """
        Chooses how many of the current worst offenders to permanently
        reject this pass, when running in greedy-elimination mode.

        k scales with how many anomalies are currently flagged (so a
        clean pool converges in one pass, a messy one takes proportionally
        bigger bites) but is capped as a fraction of the active pool and by
        an absolute ceiling, so no single pass can over-reject.
        """
        if n_anomalies == 0:
            return 0
        k = int(np.ceil(self.greedy_k_fraction * n_anomalies))
        k = min(k, int(np.ceil(self.greedy_k_pool_fraction * n_pool)))
        k = min(k, self.greedy_k_max)
        return min(k, n_anomalies)
    
    def run(
        self, 
        network: Network, 
        store: FlowStore, 
        periods: List[pd.Timestamp], 
        vehicle_types: List[str]
    ) -> FlowStore:
        """Runs adaptive iterations until the network data clears or stabilizes."""
        
        resolver_converged = {vt: False for vt in vehicle_types}
        resolver_done = {vt: False for vt in vehicle_types}
        prev_deferred_indices = {vt: None for vt in vehicle_types}  # over-threshold survivors left unrejected at the end of the last pass
        adaptive_multiplier = {vt: 1 for vt in vehicle_types}
        snapshots = {vt: [] for vt in vehicle_types}
        final_iteration = {vt: 0 for vt in vehicle_types}

        if self.greedy_elimination:
            print("Running FlowResolver GREEDILY")

        for iteration in range(1, self.max_iterations + 1):
            active_vtypes = [vt for vt in vehicle_types if not resolver_done[vt]]
            if not active_vtypes:
                break

            print(f"[Pass {iteration}] Generating network estimations...")
            store.clear_reconstructions()
            
            # 1. Execute reconstruction engines
            for recon in self.reconstructors:
                recon_df = recon.reconstruct(network, store, periods, vehicle_types)
                if recon_df is not None and not recon_df.empty:
                    store.append_estimates(recon_df)

            df = store.dataframe

            for vt in active_vtypes:
                final_iteration[vt] = iteration
                done, converged = self._run_vtype_pass(
                    store, df, vt, iteration,
                    prev_deferred_indices, adaptive_multiplier, snapshots,
                )
                resolver_done[vt] = done
                resolver_converged[vt] = converged

            if all(resolver_done.values()):
                break

        else:
            still_active = [vt for vt, done in resolver_done.items() if not done]
            if still_active:
                print(
                    f"WARNING: FlowResolver reached maximum iterations ({self.max_iterations}) "
                    f"without convergence for vehicle types: {still_active}"
                )
    
        # Only publish resolved baselines for vtypes that actually converged.
        converged_vtypes = {vt for vt, ok in resolver_converged.items() if ok}
        if converged_vtypes:
            lookup_map = self.validator.compute_baseline(store.dataframe.copy())
            if not lookup_map.empty:
                resolved_df = (
                    lookup_map.reset_index()
                    .rename(columns={"baseline": "volume", "baseline_error": "volume_err"})
                )
                resolved_df = resolved_df[resolved_df["vehicle_type"].isin(converged_vtypes)]
                if not resolved_df.empty:
                    resolved_df["source_type"] = FlowStore.SOURCE_RES
                    resolved_df["source_id"] = f"{self.__class__.__name__}-baseline"
                    resolved_df["validation"] = "NA"
                    resolved_df["weight"] = 1.0
                    store.append_estimates(resolved_df)
    
        if self.debug_plot:
            for vt in vehicle_types:
                self.plot_iteration_snapshot(final_iteration[vt] or 1, store, periods[0], vt)
            #self.plot_iteration_summary(snapshots)  # would need a per-vtype rework too, see note below
    
        return store

    def _run_vtype_pass(
        self,
        store: FlowStore,
        df: pd.DataFrame,
        vt: str,
        iteration: int,
        prev_deferred_indices: dict,
        adaptive_multiplier: dict,
        snapshots: dict,
    ) -> Tuple[bool, bool]:
        """
        Runs validation/rejection bookkeeping for a single vehicle type within
        one pass.

        Returns
        -------
        (done, converged) : tuple of bool
            done : True if this vehicle type should stop being processed in
            subsequent passes (either because it converged, or because it
            failed critically and further passes won't help).
            converged : True if a resolved baseline should be published for
            this vehicle type once the whole run finishes.
        """
        

        # 1. Extract active raw observations to test
        obs_mask = (
            (df["source_type"] == FlowStore.SOURCE_OBS)
            & (df["vehicle_type"] == vt)
            & (~df["screening"].isin(["void", "NA"]))     
        )
        if self.greedy_elimination:
            obs_mask = (obs_mask & (df["validation"] != "rejected"))
        obs_rows = df[obs_mask]
        if obs_rows.empty:
            return True, False

        store.set_validation_state(obs_rows.index, "pending")       
            
        # 2 Execute validation and classification choices
        severity = self.validator.severity_score(store, obs_rows.index)
        conforming_indices = severity[severity.abs() <= 1.0].index
        anomalies_indices = severity[severity.abs() > 1.0].index
        anomalies_ranked = severity.loc[anomalies_indices].abs().sort_values(ascending=False).index

        if self.greedy_elimination:
            base_k = self._select_rejection_batch_size(len(anomalies_ranked), len(obs_rows))

            if self.greedy_adaptive_batch:
                # If last pass's rejection left the flagged set completely unchanged —
                # nobody got rescued, nobody new got exposed — that's empirical proof
                # those observations weren't coupled to anything we removed, so it's
                # safe to move faster. Anything else (a rescue, a newly-exposed
                # anomaly, or no history yet) resets us back to the cautious baseline.
                prev_deferred_idx = prev_deferred_indices[vt]
                unchanged = (
                    prev_deferred_idx is not None
                    and pd.Index(anomalies_ranked).symmetric_difference(prev_deferred_idx).empty
                )
                adaptive_multiplier[vt] = adaptive_multiplier[vt] * 2 if unchanged else 1
                k = min(base_k * adaptive_multiplier[vt], len(anomalies_ranked))
            else:
                k = base_k

            rejected_this_pass = anomalies_ranked[:k]
            deferred_indices = anomalies_ranked[k:]  # over threshold, but not yet rejected
            prev_deferred_indices[vt] = deferred_indices
        else:
            rejected_this_pass = anomalies_ranked
            deferred_indices = pd.Index([])
            
        store.set_validation_state(obs_rows.index, "unresolved")
        store.set_validation_state(conforming_indices, "verified")
        store.set_validation_state(rejected_this_pass, "rejected")
        # deferred_indices are left "unresolved" — they're over threshold this pass,
        # but the baseline is still contaminated by worse offenders, so we hold off
        # judgement until those are gone and the baseline is recomputed next pass.

        validated_idx = pd.Index(conforming_indices).union(pd.Index(anomalies_indices)) 
        failed_indices = obs_rows.index.difference(validated_idx)
        if len(failed_indices) == len(obs_rows) and len(obs_rows) > 0:
            print(
                f"CRITICAL: Reconstruction {vt} failed completely for all {len(obs_rows)} observations. "
                f"No baseline consensus could be computed. Breaking resolving loop."
            )
            return True, False
        elif len(failed_indices) > 0:
            print(
                f"WARNING: Reconstruction {vt} failed to generate consensus baselines for {len(failed_indices)} "
                f"observations. These values remain in a 'unresolved' state."
            )

        # 6. Track the history and look for convergence
        snapshots[vt].append((iteration,store.dataframe.copy()))
        rejected_mask = (
            (store.dataframe["source_type"] == FlowStore.SOURCE_OBS)
            & (store.dataframe["vehicle_type"] == vt)
            & (store.dataframe["validation"] == "rejected")
        )

        if len(anomalies_indices) == 0:
            rejected_mask = (
                (store.dataframe["source_type"] == FlowStore.SOURCE_OBS)
                & (store.dataframe["validation"] == "rejected")
            )
            print(f"--> Architecture converged cleanly for {vt} at iteration {iteration}")
            print(f"--> {int(rejected_mask.sum())} {vt} observations rejected in total: {store.dataframe.loc[rejected_mask].index.to_list()}.")
            return True, True

        if self.greedy_elimination:
            print(
                f"--> Pool {vt} shrinking: {len(anomalies_ranked)} remain over threshold, "
                f"rejected top {len(rejected_this_pass)}, deferred {len(deferred_indices)} for re-scoring."
            )
            print(f"--> {int(rejected_mask.sum())} {vt} observations rejected in total: {store.dataframe.loc[rejected_mask].index.to_list()}.")
            return False, False

        current_validation = (
            store.dataframe[
                (store.dataframe["source_type"] == FlowStore.SOURCE_OBS)
                & (store.dataframe["vehicle_type"] == vt)
            ]
            .set_index(["source_id", "timestamp", "vehicle_type"])["validation"]
            .sort_index()
        )

        detected_duplicate = False
        duplicate_iteration = None
        for prev_idx, prev_df_copy in snapshots[vt][:-1]:
            prev_validation = (
                prev_df_copy[
                    (prev_df_copy["source_type"] == FlowStore.SOURCE_OBS)
                    & (prev_df_copy["vehicle_type"] == vt)
                ]
                .set_index(["source_id", "timestamp", "vehicle_type"])["validation"]
                .sort_index()
            )
            # Compare the validation categories row-by-row
            if prev_validation.equals(current_validation):
                detected_duplicate = True
                duplicate_iteration = prev_idx
                break

        if detected_duplicate:
            if duplicate_iteration == iteration - 1:
                print(f"--> Convergence for {vt} reached at iteration {iteration}: Outlier validation states have stabilized.")
            else:
                print(f"--> Convergence for {vt} stopped at iteration {iteration}: Detected an oscillation cycle (matches iteration {duplicate_iteration}).")
            return True, True
        
        print(f"--> No convergence found for {vt}, outliers over threshold: {list(anomalies_ranked)}")
        return False, False
    
    def set_plotting_restrictions(self, 
            sections_to_plot: Optional[List[str]] = None,
            sections_per_plot: int = 20
        ) -> None:
        self._sections_to_plot = sections_to_plot
        self._sections_per_plot = sections_per_plot

    def plot_iteration_snapshot(
        self, 
        iteration: int, 
        store: FlowStore, 
        timestamp: pd.Timestamp, 
        vehicle_type: str
    ) -> None:
        """Generates a standalone single-iteration snapshot figure with a horizontal bottom legend."""
        df = store.dataframe
        snap_mask = (df["timestamp"] == timestamp) & (df["vehicle_type"] == vehicle_type)
        snap_data = df[snap_mask]
        
        if snap_data.empty:
            return
            
        all_sections = snap_data["road_section_id"].unique()
        if self._sections_to_plot is not None:
            section_ids = sorted([sec for sec in all_sections if sec in self._sections_to_plot])
        else:
            section_ids = sorted(all_sections)

        total_parts = int(np.ceil(len(section_ids)/self._sections_per_plot))
        for chunk_idx, start_idx in enumerate(range(0, len(section_ids), self._sections_per_plot)):
            chunk_section_ids = section_ids[start_idx : start_idx + self._sections_per_plot]
            part_num = chunk_idx + 1
            
            x_positions = np.arange(len(chunk_section_ids))
            sec_to_x = {sec: x for x, sec in enumerate(chunk_section_ids)}
            
            target_sec_mask = snap_data["road_section_id"].isin(chunk_section_ids)
            obs = snap_data[(snap_data["source_type"] == FlowStore.SOURCE_OBS) & target_sec_mask]
            recons = snap_data[(snap_data["source_type"] == FlowStore.SOURCE_REC) & target_sec_mask]
            
            # 1. Initialize Figure Canvas with a 2:1 height ratio matrix
            fig, (ax1, ax2) = plt.subplots(
                2, 1, 
                figsize=(7.5, 6.5), 
                sharex=True, 
                gridspec_kw={'height_ratios': [2, 1]}
            )
            time_str = timestamp.strftime('%Y-%m-%d %H:%M')
            title_suffix = f"({part_num}/{total_parts})"
            title_prefix = f" '{self.name}'" if self.name is not None else ""
            fig.suptitle(f"Flow Link Balancer{title_prefix} - {time_str} | {vehicle_type} {title_suffix}", fontsize=14, fontweight="bold")
            
            RECO_DATA_OFFSET = -0.15
            RECO_MEAN_OFFSET = 0.15
            
            # Calculate consensus baseline parameters
            reco_means, reco_stds = [], []
            for sec_id in chunk_section_ids:
                mask = (
                    (df["road_section_id"] == sec_id) &
                    (df["timestamp"] == timestamp) &
                    (df["vehicle_type"] == vehicle_type) &
                    (df["validation"] != "rejected") &
                    (df["source_type"] == FlowStore.SOURCE_REC)
                )
                sub = df[mask].dropna(subset=["volume"])
                if sub.empty:
                    reco_means.append(np.nan)
                    reco_stds.append(np.nan)
                else:
                    m, s = weighted_mean_and_error(
                        sub["volume"].to_numpy(), sub["volume_err"].to_numpy(), sub["weight"].to_numpy(), label="plot"+sec_id
                    )
                    reco_means.append(m)
                    reco_stds.append(s)
                    
            reco_means = np.array(reco_means, dtype=np.float32)
            reco_stds = np.array(reco_stds, dtype=np.float32)
            
            # --- ROW 1: VOLUMES PROFILE ---
            ax1.grid(True, linestyle="--", alpha=0.4)
            
            # Fully filled Consensus Range square markers
            ax1.errorbar(
                x_positions + RECO_MEAN_OFFSET, reco_means, yerr=reco_stds, fmt='s', 
                color="#1f77b4", capsize=4, elinewidth=2, 
                label=r"Reconstruction Consensus ($\pm\sigma$)"
            )
            
            # Method Reconstructions
            for _, row in recons.iterrows():
                x_loc = sec_to_x[row["road_section_id"]] + RECO_DATA_OFFSET
                ax1.scatter(
                    x_loc, row["volume"], color="#5c8cb3", marker="d", alpha=0.5, s=25, zorder=2,
                    label="Reconstructed Values"
                )
                
            # Sensor Observations
            for _, row in obs.iterrows():
                x_loc = sec_to_x[row["road_section_id"]]

                if row["validation"] == "rejected":
                    colour = "#d62728"  # Red
                    fmt = "x"
                    label_str = "Observation - Rejected (Outlier)"
                elif row["validation"] == "unresolved":
                    colour = "#ff7f0e"  # Orange
                    fmt = "^"          # Triangle
                    label_str = "Observation - Unresolved"
                else:
                    colour = "#2ca02c"  # Green
                    fmt = "o"          # Circle
                    label_str = "Observation - Verified"

                if row["screening"] == "held":
                    colour = "#8320f3"
                    label_str = "Observation - HELD"
                
                ax1.errorbar(x_loc, row["volume"], yerr=row["volume_err"], fmt=fmt,
                    color=colour, capsize=4, elinewidth=2, zorder=5, label=label_str)
                    
            ax1.set_ylabel("Flow Volume (Veh/Period)", fontsize=11)
            ax1.set_title(f"Pass {iteration}", fontweight="bold")
            
            # --- ROW 2: RESIDUAL DELTAS ---
            #ax2.grid(True, linestyle="--", alpha=0.4)
            ax2.axhline(0, color="#1f77b4", linestyle="-", alpha=0.3)
            
            for _, row in obs.iterrows():
                x_loc = sec_to_x[row["road_section_id"]]
                base_m = reco_means[x_loc]
                base_std = reco_stds[x_loc]
                delta = (row["volume"] - base_m) * 100/abs(base_m) if not pd.isna(base_m) else np.nan
                delta_std = np.sqrt(row["volume_err"]**2 + base_std**2) * 100/abs(base_m) if not pd.isna(base_std) else np.nan

                if row["validation"] == "rejected":
                    colour = "#d62728"  # Red
                    fmt = "x"
                    label_str = "Observation - Rejected (Outlier)"
                elif row["validation"] == "unresolved":
                    colour = "#ff7f0e"  # Orange
                    fmt = "^"          # Triangle
                    label_str = "Observation - Unresolved"
                else:
                    colour = "#2ca02c"  # Green
                    fmt = "o"          # Circle
                    label_str = "Observation - Verified"

                if row["screening"] == "held":
                    colour = "#8320f3"
                    label_str = "Observation - HELD"
                
                if not pd.isna(delta):
                    ax2.vlines(x_loc, 0, delta, colors=colour, alpha=0.4, linewidth=1, ls=':')
                    ax2.errorbar(x_loc, delta, yerr=delta_std, color=colour, fmt=fmt, elinewidth=2, capsize=5, zorder=3)
                    
            ax2.set_ylabel("Residual Delta (%)", fontsize=11)
            ax2.set_xticks(x_positions)
            ax2.set_xticklabels(chunk_section_ids, rotation=45, ha="right")
            ax2.set_yscale('symlog', linthresh=3, linscale=1.5)
            ax2.axhspan(-2,2, color="#BBBBBB", alpha=0.2, zorder=0, label="Acceptance band (2%)")
            ax2.axhspan(-.5,.5, color="#BBBBBB", alpha=0.2, zorder=1, label="Acceptance band (0.5%)")
            ax2.set_ylim((-100,100))
            ax2.yaxis.set_major_locator(ticker.SymmetricalLogLocator(base=10, linthresh=3))
            ax2.yaxis.set_minor_locator(ticker.SymmetricalLogLocator(base=10, linthresh=3, subs=np.arange(2, 9)))
            ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda val, pos: f"{int(val)}%"))
            
            # --- BUILD SINGLE ROW LEGEND UNDER THE AXES ---
            # Harvest handles from the top plotting layer and remove duplicates
            h, l = ax1.get_legend_handles_labels()
            by_label = dict(zip(l, h))
            
            fig.legend(
                by_label.values(), 
                by_label.keys(), 
                loc="lower center", 
                ncol=4, 
                bbox_to_anchor=(0.5, -0.06),
                framealpha=0.9
            )
            
            plt.tight_layout()
            time_str = timestamp.strftime('%Y%m%d_%H%M%S')
            prefix = f"{self.name}_" if self.name is not None else ""
            suffix = f"_part{part_num}" if total_parts > 1 else ""
            filename = f"{prefix}snapshot_pass_{iteration}_{time_str}_{vehicle_type}{suffix}.png"
            plt.savefig(os.path.join(self.plot_dir, filename), dpi=150, bbox_inches='tight')
            plt.close()

    def plot_iteration_summary(
        self,
        snapshots: dict,
    ) -> None:
        """
        Generates comparative dashboard grids across multiple iterations,
        one dashboard set per vehicle type.

        Vehicle types no longer run in lockstep (each converges/stops on its
        own schedule), so `snapshots` is keyed by vehicle_type, and each
        vtype gets its own set of dashboards sized to its own iteration count.

        Parameters
        ----------
        snapshots : dict
            Mapping of vehicle_type -> list of (iteration, store.dataframe
            copy) tuples, as accumulated in `run()`.
        """
        for vt, historical_states in snapshots.items():
            if not historical_states:
                continue
            self._plot_iteration_summary_for_vtype(vt, historical_states)

    def _plot_iteration_summary_for_vtype(
        self, 
        vehicle_type: str,
        historical_states: List[Tuple[int, pd.DataFrame]]
    ) -> None:
        """
        Generates comparative dashboard grids across multiple iterations, 
        ensuring proper row height dimensions and de-duplicated external legends.
        """
        first_df = historical_states[0][1]
        unique_timestamps = (
            first_df.loc[first_df["vehicle_type"] == vehicle_type, "timestamp"]
            .drop_duplicates()
            .tolist()
        )

        RECO_DATA_OFFSET = -0.15
        RECO_MEAN_OFFSET = 0.15
        num_iters = len(historical_states)

        for timestamp_np in unique_timestamps:
            timestamp = pd.Timestamp(timestamp_np)

            # Determine unique section across all iterations
            all_sections = set()
            for _, df in historical_states:
                snap_mask = (df["timestamp"] == timestamp) & (df["vehicle_type"] == vehicle_type)
                all_sections.update(df[snap_mask]["road_section_id"].unique())
            
            if self._sections_to_plot is not None:
                section_ids = sorted([sec for sec in all_sections if sec in self._sections_to_plot])
            else:
                section_ids = sorted(list(all_sections))

            total_parts = int(np.ceil(len(section_ids) / self._sections_per_plot))
            for chunk_idx, start_idx in enumerate(range(0, len(section_ids), self._sections_per_plot)):
                chunk_section_ids = section_ids[start_idx : start_idx + self._sections_per_plot]
                part_num = chunk_idx + 1

                fig, axes = plt.subplots(
                    2, num_iters, 
                    figsize=(4.8 * num_iters + 2.5, 7.5), 
                    sharey='row', 
                    sharex=True,
                    gridspec_kw={'height_ratios': [2, 1]}
                )
                time_str = timestamp.strftime('%Y-%m-%d %H:%M')
                title_suffix = f"({part_num}/{total_parts})"
                title_prefix = f" '{self.name}'" if self.name is not None else ""
                fig.suptitle(f"Flow Link Balancer{title_prefix} - {time_str} | {vehicle_type} {title_suffix}", fontsize=14, fontweight="bold")

                if num_iters == 1:
                    axes = np.expand_dims(axes, axis=1)

                # Create containers to capture master legend handles globally across all columns
                global_handles = []
                global_labels = []

                for col_idx, (iteration, df) in enumerate(historical_states):
                    ax_vol = axes[0, col_idx]
                    ax_res = axes[1, col_idx]
                    
                    ax_vol.set_title(f"Pass {iteration}", fontsize=11, fontweight="bold")
                    
                    snap_mask = (df["timestamp"] == timestamp) & (df["vehicle_type"] == vehicle_type)
                    snap_data = df[snap_mask]
                    
                    if snap_data.empty:
                        continue
                        
                    x_positions = np.arange(len(chunk_section_ids))
                    sec_to_x = {sec: x for x, sec in enumerate(chunk_section_ids)}
                    
                    target_sec_mask = snap_data["road_section_id"].isin(chunk_section_ids)
                    obs = snap_data[(snap_data["source_type"] == FlowStore.SOURCE_OBS) & target_sec_mask]
                    recons = snap_data[(snap_data["source_type"] == FlowStore.SOURCE_REC) & target_sec_mask]
                    
                    # Baseline metrics extraction
                    reco_means, reco_stds = [], []
                    for sec_id in chunk_section_ids:
                        mask = (
                            (df["road_section_id"] == sec_id) &
                            (df["timestamp"] == timestamp) &
                            (df["vehicle_type"] == vehicle_type) &
                            (df["validation"] != "rejected") &
                            (df["source_type"] == FlowStore.SOURCE_REC)
                        )
                        sub = df[mask].dropna(subset=["volume"])
                        if sub.empty:
                            reco_means.append(np.nan)
                            reco_stds.append(np.nan)
                        else:
                            m, s = weighted_mean_and_error(
                                sub["volume"].to_numpy(), sub["volume_err"].to_numpy(), sub["weight"].to_numpy(), label="plot"+sec_id
                            )
                            reco_means.append(m)
                            reco_stds.append(s)
                            
                    reco_means = np.array(reco_means, dtype=np.float32)
                    reco_stds = np.array(reco_stds, dtype=np.float32)
                    
                    # --- ROW 1: VOLUMES (Height Ratio: 2) ---
                    ax_vol.grid(True, linestyle="--", alpha=0.4)
                    
                    ax_vol.errorbar(
                        x_positions + RECO_MEAN_OFFSET, reco_means, yerr=reco_stds, fmt='s', 
                        color="#1f77b4", capsize=4, elinewidth=2, 
                        label=r"Reconstruction Consensus ($\pm\sigma$)"
                    )
                    
                    for _, row in recons.iterrows():
                        x_loc = sec_to_x[row["road_section_id"]] + RECO_DATA_OFFSET
                        ax_vol.scatter(
                            x_loc, row["volume"], color="#5c8cb3", marker="d", alpha=0.5, s=25, zorder=2,
                            label="Reconstructed Values"
                        )
                        
                    for _, row in obs.iterrows():
                        x_loc = sec_to_x[row["road_section_id"]]

                        if row["validation"] == "rejected":
                            colour = "#d62728"  # Red
                            fmt = "x"
                            label_str = "Observation - Rejected (Outlier)"
                        elif row["validation"] == "unresolved":
                            colour = "#ff7f0e"  # Orange
                            fmt = "^"          # Triangle
                            label_str = "Observation - Unresolved"
                        else:
                            colour = "#2ca02c"  # Green
                            fmt = "o"          # Circle
                            label_str = "Observation - Verified"

                        if row["screening"] == "held":
                            colour = "#8320f3"
                            label_str = "Observation - HELD"

                        ax_vol.errorbar(x_loc, row["volume"], yerr=row["volume_err"], fmt=fmt,
                            color=colour, capsize=4, elinewidth=2, zorder=5, label=label_str)
                    
                    if col_idx == 0:
                        ax_vol.set_ylabel("Flow Volume (Veh/Period)", fontsize=11)
                    
                    # Accumulate handles from this axis to generate a clean unified legend layout later
                    h, l = ax_vol.get_legend_handles_labels()
                    global_handles.extend(h)
                    global_labels.extend(l)
                        
                    # --- ROW 2: RESIDUAL DELTAS (Height Ratio: 1) ---
                    ax_res.grid(True, linestyle="--", alpha=0.4)
                    ax_res.axhline(0, color="#1f77b4", linestyle="-", alpha=0.3)
                    
                    for _, row in obs.iterrows():
                        x_loc = sec_to_x[row["road_section_id"]]
                        base_m = reco_means[x_loc]
                        base_std = reco_stds[x_loc]
                        delta = (row["volume"] - base_m) * 100/abs(base_m) if not pd.isna(base_m) else np.nan
                        delta_std = np.sqrt(row["volume_err"]**2 + base_std**2) * 100/abs(base_m) if not pd.isna(base_std) else np.nan
                        
                        if row["validation"] == "rejected":
                            colour = "#d62728"  # Red
                            fmt = "x"
                            label_str = "Observation - Rejected (Outlier)"
                        elif row["validation"] == "unresolved":
                            colour = "#ff7f0e"  # Orange
                            fmt = "^"          # Triangle
                            label_str = "Observation - Unresolved"
                        else:
                            colour = "#2ca02c"  # Green
                            fmt = "o"          # Circle
                            label_str = "Observation - Verified"

                        if row["screening"] == "held":
                            colour = "#8320f3"
                            label_str = "Observation - HELD"
                        
                        if not pd.isna(delta):
                            ax_res.vlines(x_loc, 0, delta, colors=colour, alpha=0.4, linewidth=1, ls=':')
                            ax_res.errorbar(x_loc, delta, yerr=delta_std, color=colour, fmt=fmt, elinewidth=2, capsize=5, zorder=3)
                    
                    if col_idx == 0:
                        ax_res.set_ylabel("Residual Delta (%)", fontsize=11)
                        
                    ax_res.set_xticks(x_positions)
                    ax_res.set_xticklabels(chunk_section_ids, rotation=45, ha="right")
                    ax_res.set_yscale('symlog', linthresh=3)
                    ax_res.axhspan(-2,2, color="#BBBBBB", alpha=0.2, zorder=0, label="Acceptance band (2%)")
                    ax_res.axhspan(-.5,.5, color="#BBBBBB", alpha=0.2, zorder=1, label="Acceptance band (0.5%)")
                    ax_res.set_ylim((-100,100))
                    ax_res.yaxis.set_major_locator(ticker.SymmetricalLogLocator(base=10, linthresh=3))
                    ax_res.yaxis.set_minor_locator(ticker.SymmetricalLogLocator(base=10, linthresh=3, subs=np.arange(2, 10)))

                # --- RENDER UNIQUE EXTERNAL LEGEND ON THE RIGHT-MOST AXIS ---
                # Use dictionary conversion to filter out duplicate names cleanly
                by_label = dict(zip(global_labels, global_handles))
                
                # Place on the last top volume column plot, pushed outside the bounding box
                fig.legend(
                    by_label.values(), 
                    by_label.keys(), 
                    loc="lower center", 
                    ncol=4, 
                    bbox_to_anchor=(0.5, -0.05),
                    framealpha=0.9
                )

                plt.tight_layout()  

                time_str = timestamp.strftime('%Y%m%d_%H%M%S')
                prefix = f"{self.name}_" if self.name is not None else ""
                suffix = f"_part{part_num}" if total_parts > 1 else ""
                filename = f"{prefix}dashboard_summary_{time_str}_{vehicle_type}{suffix}.png"
                plt.savefig(os.path.join(self.plot_dir, filename), dpi=150, bbox_inches='tight')
                plt.close()