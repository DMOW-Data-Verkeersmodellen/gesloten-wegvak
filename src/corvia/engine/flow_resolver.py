# corvia/engine/flow_resolver.py
from __future__ import annotations
import os
from typing import List, Tuple, Optional, TYPE_CHECKING

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from corvia.network_states.flow import FlowStore
from corvia.utils import compute_z_score, weighted_mean_and_error

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
        plot_dir: str = 'diagnostics'
    ) -> None:
        
        self.reconstructors = reconstructors
        self.validator = validator
        self.max_iterations = max_iterations
        self.name = name
        self.debug_plot = debug_plot
        self.plot_dir = plot_dir
        self.set_plotting_restrictions()

        if self.debug_plot:
            os.makedirs(self.plot_dir, exist_ok=True)

    def compute_baselines(self, store: FlowStore, obs_indices: pd.Index) -> Tuple[pd.Series, pd.Series]:
        """
        Extracts independent reconstruction consensus baselines for each road section.

        Parameters
        ----------
        store : FlowStore
            The active state store container.
        target_indices : pd.Index
            Indices from the store representing observations to benchmark.

        Returns
        -------
        Tuple[pd.Series, pd.Series]
            Two series containing (baseline_means, baseline_stds) perfectly aligned 
            with the input dataframe's index.
        """
    
        df = store.dataframe
        
        # 1. Isolate valid active reconstruction traces
        recon_mask = (df["source_type"] == FlowStore.SOURCE_REC) & (df["validation"] != "rejected")
        recon_rows = df[recon_mask].dropna(subset=["volume"])
        
        if recon_rows.empty:
            empty_series = pd.Series(np.nan, index=obs_indices, dtype="float32")
            return empty_series, empty_series

        # 2. Compute consensus EXACTLY ONCE per unique spatial-temporal coordinate
        def _calc_unique_slice(group: pd.DataFrame) -> pd.Series:
            v = group["volume"].to_numpy(dtype="float64")
            e = group["volume_err"].to_numpy(dtype="float64")
            w = group["weight"].to_numpy(dtype="float64")
            
            mean, std = weighted_mean_and_error(v, e, w)
            return pd.Series({"baseline_mean": mean, "baseline_std": std})

        lookup_map = (
            recon_rows.groupby(["road_section_id", "timestamp", "vehicle_type"], observed=True)
            .apply(_calc_unique_slice, include_groups=False)
        )

        # 3. Broadcast the unique baselines out to the investigated indices
        obs_coords = df.loc[obs_indices, ["road_section_id", "timestamp", "vehicle_type"]].reset_index()
        aligned = obs_coords.merge(
            lookup_map, 
            on=["road_section_id", "timestamp", "vehicle_type"], 
            how="left"
        )
        aligned = aligned.set_index('index')
        
        return aligned["baseline_mean"], aligned["baseline_std"]

    def run(
        self, 
        network: Network, 
        store: FlowStore, 
        periods: List[pd.Timestamp], 
        vehicle_types: List[str]
    ) -> FlowStore:
        """Runs adaptive iterations until the network data clears or stabilizes."""
        
        snapshots = []

        for iteration in range(1, self.max_iterations + 1):
            print(f"[Pass {iteration}] Generating network estimations...")
            store.clear_reconstructions()
            
            # 1. Execute reconstruction engines
            for recon in self.reconstructors:
                recon_df = recon.reconstruct(network, store, periods, vehicle_types)
                if recon_df is not None and not recon_df.empty:
                    store.append_estimates(recon_df)

            df = store.dataframe
            
            # 2. Extract active raw observations to test
            obs_mask = (df["source_type"] == FlowStore.SOURCE_OBS) & (~df["screening"].isin(["void", "NA"]))
            obs_rows = df[obs_mask]
            if obs_rows.empty:
                break

            store.set_validation_state(obs_rows.index, "pending")       
            
            # 5. Execute validation and classification choices
            conforming_indices, anomalies_indices = self.validator.validate(store, obs_rows.index)
            store.set_validation_state(obs_rows.index, "unresolved") # change this to intersect or something?
            store.set_validation_state(conforming_indices, "verified")
            store.set_validation_state(anomalies_indices, "rejected")

            validated_idx = pd.Index(conforming_indices).union(pd.Index(anomalies_indices)) 
            failed_indices = obs_rows.index.difference(validated_idx)
            if len(failed_indices) == len(obs_rows) and len(obs_rows) > 0:
                print(
                    f"CRITICAL: Reconstruction failed completely for all {len(obs_rows)} observations. "
                    f"No baseline consensus could be computed. Breaking resolving loop."
                )
                break
            elif len(failed_indices) > 0:
                print(
                    f"WARNING: Reconstruction failed to generate consensus baselines for {len(failed_indices)} "
                    f"observations. These values remain in a 'unresolved' state."
                )

            # 6. Track the history and look for convergence
            snapshots.append((iteration,store.dataframe.copy()))
    
            if len(anomalies_indices) == 0:
                print(f"-> Architecture converged cleanly at iteration {iteration} (0 anomalies).")
                break

            current_validation = (
                store.dataframe[store.dataframe["source_type"] == FlowStore.SOURCE_OBS]
                .set_index(["source_id", "timestamp", "vehicle_type"])["validation"]
                .sort_index()
            )

            detected_duplicate = False
            duplicate_iteration = None
            for prev_idx, prev_df_copy in snapshots[:-1]:
                prev_validation = (
                    prev_df_copy[prev_df_copy["source_type"] == FlowStore.SOURCE_OBS]
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
                    print(f"-> Convergence reached at iteration {iteration}: Outlier validation states have stabilized.")
                else:
                    print(f"-> Convergence stopped at iteration {iteration}: Detected an oscillation cycle (matches iteration {duplicate_iteration}).")
                break
            else:
                print(f"--> No convergence found, new outliers found: {list(anomalies_indices)}")
        
        else:
            # Executes ONLY if the loop ran max_iterations and did not execute a 'break'
            print(f"WARNING: FlowResolver reached maximum iterations ({self.max_iterations}) without reaching convergence.")
        
        if self.debug_plot:
            self.plot_iteration_summary(snapshots)
        return store
    
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
                        sub["volume"].to_numpy(), sub["volume_err"].to_numpy(), sub["weight"].to_numpy()
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
                    
            ax1.set_ylabel("Flow Volume (Veh/hr)", fontsize=11)
            ax1.set_title(f"Pass {iteration}", fontweight="bold")
            
            # --- ROW 2: RESIDUAL DELTAS ---
            ax2.grid(True, linestyle="--", alpha=0.4)
            ax2.axhline(0, color="#1f77b4", linestyle="-", alpha=0.3)
            
            for _, row in obs.iterrows():
                x_loc = sec_to_x[row["road_section_id"]]
                base_m = reco_means[x_loc]
                base_std = reco_stds[x_loc]
                delta = row["volume"] - base_m if not pd.isna(base_m) else np.nan
                delta_std = np.sqrt(row["volume_err"]**2 + base_std**2) if not pd.isna(base_std) else np.nan

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
                    
            ax2.set_ylabel("Residual Delta (Veh/h)", fontsize=11)
            ax2.set_xticks(x_positions)
            ax2.set_xticklabels(chunk_section_ids, rotation=45, ha="right")
            
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
        historical_states: List[Tuple[int, pd.DataFrame]]
    ) -> None:
        """
        Generates comparative dashboard grids across multiple iterations, 
        ensuring proper row height dimensions and de-duplicated external legends.
        """
        if not historical_states:
            return

        first_df = historical_states[0][1]
        unique_combinations = (
            first_df[["timestamp", "vehicle_type"]]
            .drop_duplicates()
            .to_records(index=False)
        )

        RECO_DATA_OFFSET = -0.15
        RECO_MEAN_OFFSET = 0.15
        num_iters = len(historical_states)

        for timestamp_np, vehicle_type in unique_combinations:
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
                                sub["volume"].to_numpy(), sub["volume_err"].to_numpy(), sub["weight"].to_numpy()
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
                        ax_vol.set_ylabel("Flow Volume (Veh/hr)", fontsize=11)
                    
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
                        delta = row["volume"] - base_m if not pd.isna(base_m) else np.nan
                        delta_std = np.sqrt(row["volume_err"]**2 + base_std**2) if not pd.isna(base_std) else np.nan
                        
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
                        ax_res.set_ylabel("Residual Delta (Veh/h)", fontsize=11)
                        
                    ax_res.set_xticks(x_positions)
                    ax_res.set_xticklabels(chunk_section_ids, rotation=45, ha="right")

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