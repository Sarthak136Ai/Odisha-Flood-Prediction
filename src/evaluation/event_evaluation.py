"""
Comprehensive Flood Event-Based Evaluation Engine for Odisha Flood Prediction.

In hydrological disaster risk reduction (DRR), evaluating isolated day records
does not fully measure whether actual multi-day flood episodes (e.g. cyclonic
depressions, river basin inundations) were forecasted with actionable advance notice.

This module provides:
1. Reproducible event segmentation from daily spatio-temporal telemetry.
2. Temporal event matching (Actual vs Predicted Events) with lead-time estimation.
3. Event-level metrics:
   - Event Hit Rate (Event Recall = Detected Events / Total Events)
   - Event Precision (Event Success Rate = True Alerts / Total Alert Clusters)
   - Event Miss Rate (False Negative Events / Total Events)
   - Event False Alarm Rate (False Positive Clusters / Total Alert Clusters)
   - Detection Lead Time (Days of advance notice before inundation onset)
   - Event Duration & Coverage (Actual vs Predicted Duration, Missed Days)
4. Separate macro event metrics from daily record metrics.
5. High-resolution timeline visualizers (PNG & SVG) for major historical disasters
   (e.g., August 2022 Mahanadi floods, September 2024 floods, Cyclone Yaas 2021, Cyclone Titli 2018).
"""

import os
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class FloodEventDetector:
    """
    Algorithms for segmenting continuous daily flood signals into discrete flood events
    and evaluating temporal event detection characteristics.
    """
    def __init__(self, max_gap_days: int = 1, min_event_duration: int = 1):
        """
        Parameters:
        -----------
        max_gap_days: Maximum allowable gap (in days) between consecutive flood days
                      to consider them part of the same continuous flood episode.
        min_event_duration: Minimum duration (in days) for a cluster to qualify as an event.
        """
        self.max_gap_days = max_gap_days
        self.min_event_duration = min_event_duration

    def extract_events_from_series(
        self,
        dates: Union[pd.Series, List[Any]],
        binary_series: Union[pd.Series, np.ndarray],
        probability_series: Optional[Union[pd.Series, np.ndarray]] = None,
        event_prefix: str = "EVT"
    ) -> List[Dict[str, Any]]:
        """
        Segment a binary time-series into discrete continuous events.
        """
        dates_clean = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
        binary_clean = np.asarray(binary_series).astype(int)
        
        if probability_series is not None:
            proba_clean = np.asarray(probability_series).astype(float)
        else:
            proba_clean = binary_clean.astype(float)

        events = []
        in_event = False
        event_start_idx = 0
        last_positive_idx = 0
        current_event_dates = []
        current_event_probas = []

        for i, (dt, val) in enumerate(zip(dates_clean, binary_clean)):
            if val == 1:
                if not in_event:
                    in_event = True
                    event_start_idx = i
                    current_event_dates = [dt]
                    current_event_probas = [proba_clean[i]]
                else:
                    # Check gap from last positive day
                    gap = (dt - dates_clean.iloc[last_positive_idx]).days
                    if gap <= self.max_gap_days + 1:
                        current_event_dates.append(dt)
                        current_event_probas.append(proba_clean[i])
                    else:
                        # Close previous event and start new
                        dur = len(current_event_dates)
                        if dur >= self.min_event_duration:
                            events.append(self._build_event_dict(
                                event_id=f"{event_prefix}_{len(events)+1:03d}",
                                dates=current_event_dates,
                                probas=current_event_probas
                            ))
                        current_event_dates = [dt]
                        current_event_probas = [proba_clean[i]]
                        event_start_idx = i
                last_positive_idx = i
            else:
                if in_event:
                    # Check if gap threshold is exceeded
                    gap = (dt - dates_clean.iloc[last_positive_idx]).days
                    if gap > self.max_gap_days:
                        in_event = False
                        dur = len(current_event_dates)
                        if dur >= self.min_event_duration:
                            events.append(self._build_event_dict(
                                event_id=f"{event_prefix}_{len(events)+1:03d}",
                                dates=current_event_dates,
                                probas=current_event_probas
                            ))
                        current_event_dates = []
                        current_event_probas = []

        # Flush trailing event
        if in_event and len(current_event_dates) >= self.min_event_duration:
            events.append(self._build_event_dict(
                event_id=f"{event_prefix}_{len(events)+1:03d}",
                dates=current_event_dates,
                probas=current_event_probas
            ))

        return events

    def _build_event_dict(
        self,
        event_id: str,
        dates: List[pd.Timestamp],
        probas: List[float]
    ) -> Dict[str, Any]:
        """Helper to construct event summary dictionary."""
        start_date = min(dates)
        end_date = max(dates)
        dur = (end_date - start_date).days + 1
        peak_idx = int(np.argmax(probas))
        return {
            "event_id": event_id,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "start_dt": start_date,
            "end_dt": end_date,
            "duration_days": int(dur),
            "active_days_count": len(dates),
            "max_probability": round(float(np.max(probas)), 4),
            "mean_probability": round(float(np.mean(probas)), 4),
            "peak_date": dates[peak_idx].strftime("%Y-%m-%d"),
            "dates_list": [d.strftime("%Y-%m-%d") for d in dates]
        }

    def match_events(
        self,
        actual_events: List[Dict[str, Any]],
        predicted_events: List[Dict[str, Any]],
        lead_tolerance_days: int = 2,
        daily_y_true: Optional[pd.Series] = None,
        daily_y_pred: Optional[pd.Series] = None,
        daily_dates: Optional[pd.Series] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Perform temporal matching between ground-truth flood events and model predicted events.
        
        Parameters:
        -----------
        actual_events: List of ground-truth flood events.
        predicted_events: List of model alert prediction clusters.
        lead_tolerance_days: Max window (in days) before event start to consider an alert valid.
        
        Returns:
        --------
        (matched_actual_table, false_alarm_events_table, event_summary_metrics)
        """
        matched_results = []
        matched_pred_ids = set()

        # Date lookup map if daily series provided
        daily_pred_map = {}
        if daily_dates is not None and daily_y_pred is not None:
            for d, p in zip(pd.to_datetime(daily_dates), daily_y_pred):
                daily_pred_map[d.strftime("%Y-%m-%d")] = int(p)

        for act in actual_events:
            act_start = act["start_dt"]
            act_end = act["end_dt"]
            act_dur = act["duration_days"]

            # Tolerance matching window: [act_start - lead_tolerance, act_end]
            matching_preds = []
            for pred in predicted_events:
                p_start = pred["start_dt"]
                p_end = pred["end_dt"]
                # Overlap or lead condition
                if (p_start <= act_end) and (p_end >= act_start - timedelta(days=lead_tolerance_days)):
                    matching_preds.append(pred)
                    matched_pred_ids.add(pred["event_id"])

            if len(matching_preds) > 0:
                # Combine matching predictions
                first_pred_start = min(p["start_dt"] for p in matching_preds)
                last_pred_end = max(p["end_dt"] for p in matching_preds)
                pred_dur = (last_pred_end - first_pred_start).days + 1
                
                # Lead time: days between first alert and actual event start
                lead_time = (act_start - first_pred_start).days
                lead_time_days = max(0, lead_time)
                
                # Calculate missed days within event window
                missed_days = 0
                if daily_pred_map:
                    for d_str in act.get("dates_list", []):
                        if daily_pred_map.get(d_str, 0) == 0:
                            missed_days += 1
                else:
                    missed_days = max(0, act_dur - pred_dur)

                matched_results.append({
                    "event_id": act["event_id"],
                    "actual_start": act["start_date"],
                    "actual_end": act["end_date"],
                    "actual_duration_days": act_dur,
                    "detected": "YES (HIT)",
                    "detection_status": "HIT",
                    "predicted_start": first_pred_start.strftime("%Y-%m-%d"),
                    "predicted_end": last_pred_end.strftime("%Y-%m-%d"),
                    "predicted_duration_days": int(pred_dur),
                    "lead_time_days": int(lead_time_days),
                    "missed_days": int(missed_days),
                    "peak_date": act["peak_date"],
                    "coverage_pct": round(max(0.0, (1.0 - (missed_days / max(act_dur, 1)))) * 100.0, 2)
                })
            else:
                # Missed Flood Event
                matched_results.append({
                    "event_id": act["event_id"],
                    "actual_start": act["start_date"],
                    "actual_end": act["end_date"],
                    "actual_duration_days": act_dur,
                    "detected": "NO (MISSED)",
                    "detection_status": "MISSED",
                    "predicted_start": "None",
                    "predicted_end": "None",
                    "predicted_duration_days": 0,
                    "lead_time_days": 0,
                    "missed_days": int(act_dur),
                    "peak_date": act["peak_date"],
                    "coverage_pct": 0.0
                })

        # Identify False Alarm Prediction Clusters (predicted events with no corresponding actual flood)
        false_alarm_events = []
        for pred in predicted_events:
            if pred["event_id"] not in matched_pred_ids:
                false_alarm_events.append({
                    "false_alarm_id": pred["event_id"],
                    "predicted_start": pred["start_date"],
                    "predicted_end": pred["end_date"],
                    "predicted_duration_days": pred["duration_days"],
                    "max_probability": pred["max_probability"],
                    "status": "FALSE ALARM (Precautionary Standby)"
                })

        # Compute Macro Event Metrics
        n_actual = len(actual_events)
        n_pred = len(predicted_events)
        n_hits = sum(1 for r in matched_results if r["detection_status"] == "HIT")
        n_missed = n_actual - n_hits
        n_fa = len(false_alarm_events)

        event_recall = float(n_hits / n_actual) if n_actual > 0 else 0.0
        event_precision = float(n_hits / (n_hits + n_fa)) if (n_hits + n_fa) > 0 else 0.0
        event_f1 = float(2 * event_precision * event_recall / (event_precision + event_recall)) if (event_precision + event_recall) > 0 else 0.0
        
        detected_leads = [r["lead_time_days"] for r in matched_results if r["detection_status"] == "HIT"]
        mean_lead = float(np.mean(detected_leads)) if len(detected_leads) > 0 else 0.0
        mean_missed_days = float(np.mean([r["missed_days"] for r in matched_results])) if len(matched_results) > 0 else 0.0
        mean_cov = float(np.mean([r["coverage_pct"] for r in matched_results])) if len(matched_results) > 0 else 0.0

        summary_metrics = {
            "total_actual_events": int(n_actual),
            "total_predicted_events": int(n_pred),
            "detected_events_hits": int(n_hits),
            "missed_events": int(n_missed),
            "false_alarm_events": int(n_fa),
            "event_recall_hit_rate": round(event_recall, 4),
            "event_precision": round(event_precision, 4),
            "event_f1_score": round(event_f1, 4),
            "event_miss_rate": round(float(n_missed / max(n_actual, 1)), 4),
            "mean_lead_time_days": round(mean_lead, 2),
            "mean_missed_days_per_event": round(mean_missed_days, 2),
            "mean_event_coverage_pct": round(mean_cov, 2)
        }

        return matched_results, false_alarm_events, summary_metrics

    def evaluate_dataframe_events(
        self,
        df: pd.DataFrame,
        y_true_col: str,
        y_pred: np.ndarray,
        y_proba: Optional[np.ndarray] = None,
        date_col: str = "Date",
        station_col: str = "Block/Station",
        lead_tolerance_days: int = 2
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """
        Perform spatiotemporal event evaluation by grouping across individual stations/blocks.
        """
        df_eval = df[[date_col, station_col, y_true_col]].copy()
        df_eval["_y_pred"] = np.asarray(y_pred).astype(int)
        if y_proba is not None:
            df_eval["_y_proba"] = np.asarray(y_proba).astype(float)
        else:
            df_eval["_y_proba"] = df_eval["_y_pred"].astype(float)

        all_matched = []
        all_fa = []

        for station_name, st_df in df_eval.groupby(station_col):
            st_df_sorted = st_df.sort_values(date_col).reset_index(drop=True)
            dates = pd.to_datetime(st_df_sorted[date_col])
            y_t = st_df_sorted[y_true_col].values.astype(int)
            y_p = st_df_sorted["_y_pred"].values.astype(int)
            y_pr = st_df_sorted["_y_proba"].values.astype(float)

            act_evts = self.extract_events_from_series(dates, y_t, y_pr, event_prefix=f"{station_name}_ACT")
            pred_evts = self.extract_events_from_series(dates, y_p, y_pr, event_prefix=f"{station_name}_PRD")

            matched, fa, _ = self.match_events(
                actual_events=act_evts,
                predicted_events=pred_evts,
                lead_tolerance_days=lead_tolerance_days,
                daily_y_true=pd.Series(y_t),
                daily_y_pred=pd.Series(y_p),
                daily_dates=dates
            )
            for m in matched:
                m["station"] = station_name
                all_matched.append(m)
            for f in fa:
                f["station"] = station_name
                all_fa.append(f)

        n_act = len(all_matched)
        n_pred = n_act - sum(1 for m in all_matched if m["detection_status"] == "MISSED") + len(all_fa)
        n_hits = sum(1 for m in all_matched if m["detection_status"] == "HIT")
        n_miss = n_act - n_hits
        n_fa = len(all_fa)

        rec = float(n_hits / n_act) if n_act > 0 else 0.0
        prec = float(n_hits / (n_hits + n_fa)) if (n_hits + n_fa) > 0 else 0.0
        f1 = float(2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        leads = [m["lead_time_days"] for m in all_matched if m["detection_status"] == "HIT"]
        mean_lead = float(np.mean(leads)) if len(leads) > 0 else 0.0
        missed_days = [m["missed_days"] for m in all_matched]
        mean_miss_days = float(np.mean(missed_days)) if len(missed_days) > 0 else 0.0
        covs = [m["coverage_pct"] for m in all_matched]
        mean_cov = float(np.mean(covs)) if len(covs) > 0 else 0.0

        summary = {
            "total_actual_events": int(n_act),
            "total_predicted_events": int(n_pred),
            "detected_events_hits": int(n_hits),
            "missed_events": int(n_miss),
            "false_alarm_events": int(n_fa),
            "event_recall_hit_rate": round(rec, 4),
            "event_precision": round(prec, 4),
            "event_f1_score": round(f1, 4),
            "event_miss_rate": round(float(n_miss / max(n_act, 1)), 4),
            "mean_lead_time_days": round(mean_lead, 2),
            "mean_missed_days_per_event": round(mean_miss_days, 2),
            "mean_event_coverage_pct": round(mean_cov, 2)
        }
        return all_matched, all_fa, summary


# -----------------------------------------------------------------------------
# Event Timeline Visualization Generators
# -----------------------------------------------------------------------------

def plot_flood_event_timeline(
    dates: pd.Series,
    rainfall_mm: pd.Series,
    y_true: pd.Series,
    y_proba: pd.Series,
    threshold: float = 0.5,
    event_name: str = "Historical Flood Event",
    district_name: Optional[str] = None,
    save_path_png: Optional[str] = None,
    save_path_svg: Optional[str] = None
) -> plt.Figure:
    """
    Generate high-resolution dual-panel timeline chart showing:
    1. Daily rainfall and rolling rainfall accumulation.
    2. Model predicted probability curve, decision threshold line, actual flood window, and alert spans.
    """
    df_plot = pd.DataFrame({
        "Date": pd.to_datetime(dates),
        "Rainfall": np.asarray(rainfall_mm).astype(float),
        "Actual": np.asarray(y_true).astype(int),
        "Proba": np.asarray(y_proba).astype(float)
    }).sort_values("Date").reset_index(drop=True)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, dpi=300,
                                   gridspec_kw={"height_ratios": [1, 1.2]})
    fig.patch.set_facecolor("#0f172a")

    for ax in (ax1, ax2):
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#cbd5e1")
        for spine in ax.spines.values():
            spine.set_color("#334155")
        ax.grid(True, linestyle="--", alpha=0.2, color="#64748b")

    # Top Panel: Precipitation & Cumulative Inflow
    ax1.bar(df_plot["Date"], df_plot["Rainfall"], color="#38bdf8", alpha=0.85, width=0.7, label="Daily Rainfall (mm)")
    ax1.set_ylabel("Precipitation (mm)", color="#38bdf8", fontsize=10.5, fontweight="bold")
    ax1.set_title(f"Hydrological Telemetry & Early Warning Response — {event_name}" + (f" ({district_name})" if district_name else ""),
                  color="#ffffff", fontsize=12.5, fontweight="bold", pad=10)

    # 3-day and 7-day rainfall accumulation lines
    rain_7d = df_plot["Rainfall"].rolling(7, min_periods=1).sum()
    ax1_twin = ax1.twinx()
    ax1_twin.tick_params(colors="#cbd5e1")
    for spine in ax1_twin.spines.values():
        spine.set_color("#334155")
    ax1_twin.plot(df_plot["Date"], rain_7d, color="#f59e0b", lw=1.8, linestyle="--", label="7-Day Rolling Rainfall (mm)")
    ax1_twin.set_ylabel("7-Day Accumulation (mm)", color="#f59e0b", fontsize=10)

    # Combine legends for top panel
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, facecolor="#0f172a", edgecolor="#334155",
               labelcolor="#cbd5e1", loc="upper left", fontsize=8.5)

    # Bottom Panel: Model Prediction vs Ground Truth Inundation
    ax2.plot(df_plot["Date"], df_plot["Proba"], color="#a855f7", lw=2.2, label="Model Flood Probability P(Flood)")
    ax2.axhline(threshold, color="#ef4444", linestyle=":", lw=1.8, label=f"Decision Threshold (T* = {threshold:.3f})")

    # Shade Actual Inundation Period in Soft Red
    actual_mask = df_plot["Actual"] == 1
    if actual_mask.any():
        ax2.fill_between(df_plot["Date"], 0, 1.05, where=actual_mask, color="#ef4444", alpha=0.22,
                         label="Actual Ground Truth Flood Period")

    # Shade Model Alert Period (where Proba >= threshold) in Soft Purple
    pred_mask = df_plot["Proba"] >= threshold
    if pred_mask.any():
        ax2.fill_between(df_plot["Date"], 0, df_plot["Proba"], where=pred_mask, color="#a855f7", alpha=0.35,
                         label="Model High-Risk Alert Issued")

    ax2.set_ylabel("Flood Probability", color="#f8fafc", fontsize=10.5, fontweight="bold")
    ax2.set_xlabel("Date", color="#f8fafc", fontsize=11, labelpad=8)
    ax2.set_ylim(0, 1.05)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right", color="#cbd5e1")

    ax2.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", loc="upper left", fontsize=8.5)

    plt.tight_layout()

    if save_path_png:
        os.makedirs(os.path.dirname(save_path_png), exist_ok=True)
        plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
        logger.info(f"Saved PNG event timeline plot to {save_path_png}")

    if save_path_svg:
        os.makedirs(os.path.dirname(save_path_svg), exist_ok=True)
        plt.savefig(save_path_svg, format="svg", bbox_inches="tight")
        logger.info(f"Saved SVG event timeline plot to {save_path_svg}")

    return fig


def plot_multi_model_event_metrics_comparison(
    models_event_data: Dict[str, Dict[str, Any]],
    save_path_png: Optional[str] = "reports/plots/event_evaluation_comparison_all_models.png",
    save_path_svg: Optional[str] = "reports/plots/event_evaluation_comparison_all_models.svg"
) -> plt.Figure:
    """
    Generate multi-panel comparison chart of Event Hit Rate, Event Miss Rate, Event False Alarm Count,
    and Mean Lead Time across all models.
    """
    model_names = list(models_event_data.keys())
    hit_rates = [data["event_recall_hit_rate"] * 100 for data in models_event_data.values()]
    miss_rates = [data["event_miss_rate"] * 100 for data in models_event_data.values()]
    lead_times = [data["mean_lead_time_days"] for data in models_event_data.values()]
    fa_counts = [data["false_alarm_events"] for data in models_event_data.values()]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), dpi=300)
    fig.patch.set_facecolor("#0f172a")

    # Panel 1: Event Hit Rate vs Miss Rate
    ax1 = axes[0]
    ax1.set_facecolor("#1e293b")
    ax1.tick_params(colors="#cbd5e1")
    for spine in ax1.spines.values():
        spine.set_color("#334155")
    ax1.grid(True, linestyle="--", alpha=0.2, color="#64748b", axis="y")

    x = np.arange(len(model_names))
    w = 0.35
    b1 = ax1.bar(x - w/2, hit_rates, w, label="Event Hit Rate (%)", color="#10b981", alpha=0.9)
    b2 = ax1.bar(x + w/2, miss_rates, w, label="Event Miss Rate (%)", color="#ef4444", alpha=0.85)

    for bar in b1:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", color="#86efac", fontsize=8.5, fontweight="bold")
    for bar in b2:
        h = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, h + 1.5, f"{h:.1f}%", ha="center", va="bottom", color="#fca5a5", fontsize=8.5, fontweight="bold")

    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=25, ha="right", color="#f8fafc", fontsize=9)
    ax1.set_ylabel("Percentage (%)", color="#f8fafc", fontsize=10)
    ax1.set_title("Event Detection vs Miss Rate", color="#ffffff", fontsize=11, fontweight="bold", pad=8)
    ax1.set_ylim(0, 115)
    ax1.legend(facecolor="#0f172a", edgecolor="#334155", labelcolor="#cbd5e1", loc="upper right", fontsize=8.5)

    # Panel 2: Mean Advance Lead Time
    ax2 = axes[1]
    ax2.set_facecolor("#1e293b")
    ax2.tick_params(colors="#cbd5e1")
    for spine in ax2.spines.values():
        spine.set_color("#334155")
    ax2.grid(True, linestyle="--", alpha=0.2, color="#64748b", axis="y")

    bars_lead = ax2.bar(x, lead_times, width=0.5, color="#38bdf8", alpha=0.85, edgecolor="#ffffff", lw=0.5)
    for bar, l in zip(bars_lead, lead_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05, f"{l:.1f}d", ha="center", va="bottom", color="#38bdf8", fontsize=9, fontweight="bold")

    ax2.set_xticks(x)
    ax2.set_xticklabels(model_names, rotation=25, ha="right", color="#f8fafc", fontsize=9)
    ax2.set_ylabel("Lead Time (Days)", color="#38bdf8", fontsize=10)
    ax2.set_title("Mean Advance Alert Lead Time", color="#ffffff", fontsize=11, fontweight="bold", pad=8)
    ax2.set_ylim(0, max(lead_times + [1.5]) * 1.3)

    # Panel 3: False Alarm Event Clusters
    ax3 = axes[2]
    ax3.set_facecolor("#1e293b")
    ax3.tick_params(colors="#cbd5e1")
    for spine in ax3.spines.values():
        spine.set_color("#334155")
    ax3.grid(True, linestyle="--", alpha=0.2, color="#64748b", axis="y")

    bars_fa = ax3.bar(x, fa_counts, width=0.5, color="#f59e0b", alpha=0.85, edgecolor="#ffffff", lw=0.5)
    for bar, c in zip(bars_fa, fa_counts):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(fa_counts)*0.02, f"{c:,}", ha="center", va="bottom", color="#fde047", fontsize=9, fontweight="bold")

    ax3.set_xticks(x)
    ax3.set_xticklabels(model_names, rotation=25, ha="right", color="#f8fafc", fontsize=9)
    ax3.set_ylabel("False Alarm Clusters", color="#f59e0b", fontsize=10)
    ax3.set_title("Standby False Alarm Clusters", color="#ffffff", fontsize=11, fontweight="bold", pad=8)
    ax3.set_ylim(0, max(fa_counts + [10]) * 1.25)

    plt.suptitle("Multi-Model Flood Event-Based Evaluation Benchmarks", color="#ffffff", fontsize=13, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path_png:
        os.makedirs(os.path.dirname(save_path_png), exist_ok=True)
        plt.savefig(save_path_png, dpi=300, bbox_inches="tight")
        logger.info(f"Saved PNG multi-model event comparison to {save_path_png}")

    if save_path_svg:
        os.makedirs(os.path.dirname(save_path_svg), exist_ok=True)
        plt.savefig(save_path_svg, format="svg", bbox_inches="tight")
        logger.info(f"Saved SVG multi-model event comparison to {save_path_svg}")

    return fig
