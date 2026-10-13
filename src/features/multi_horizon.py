"""
Multi-Horizon Flood Target Generation and Temporal Alignment Engine.

Provides configurable generation of multi-day forecast horizon targets:
- Flood_T+1: Next-day flood occurrence (identical to Flood_Next_Day, lead time: 24h)
- Flood_T+2: Two-day ahead flood occurrence (lead time: 48h)
- Flood_T+3: Three-day ahead flood occurrence (lead time: 72h)
- Flood_T+k: Arbitrary k-day ahead forecast horizon

Strict Methodological Rules:
1. Zero Data Leakage: At forecast time T, strictly past and current observations
   (rainfall up to time T) are used. No future rainfall is ever used in features.
2. Calendar-Day Alignment: Verifies Date(T+k) - Date(T) == k calendar days.
   If stations contain temporal gaps, targets are masked with NaN rather than
   shifting across discontinuous records.
3. Backward Compatibility: Preserves and aliases existing 'Flood_Next_Day' pipeline.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

HORIZON_NAME_MAP = {
    1: "Flood_T+1",
    2: "Flood_T+2",
    3: "Flood_T+3"
}


def get_horizon_target_name(horizon: int) -> str:
    """Return standard column name for a given forecast horizon k."""
    if horizon in HORIZON_NAME_MAP:
        return HORIZON_NAME_MAP[horizon]
    return f"Flood_T+{horizon}"


def generate_horizon_target(
    df: pd.DataFrame,
    horizon: int = 1,
    source_col: str = "Flood_Occurred",
    target_col: Optional[str] = None,
    align_calendar_days: bool = True
) -> pd.DataFrame:
    """
    Generate a single forecast horizon target Flood_T+{k}.

    Parameters
    ----------
    df : pd.DataFrame
        Historical or operational dataset with District, Block/Station, Date, and source_col.
    horizon : int, default 1
        Lead time in days (e.g. 1 for T+1, 2 for T+2, 3 for T+3).
    source_col : str, default 'Flood_Occurred'
        The same-day flood binary ground truth column.
    target_col : str, optional
        Target column name. If None, defaults to 'Flood_T+{horizon}'.
    align_calendar_days : bool, default True
        If True, validates that the shifted record is exactly horizon days ahead in calendar time.
        Discontinuous date shifts are assigned NaN to prevent temporal misalignment.

    Returns
    -------
    pd.DataFrame
        DataFrame with new target column added.
    """
    if horizon < 1:
        raise ValueError(f"Forecast horizon must be a positive integer >= 1, got {horizon}")

    if source_col not in df.columns:
        raise KeyError(f"Source flood ground truth column '{source_col}' not found in DataFrame.")

    if target_col is None:
        target_col = get_horizon_target_name(horizon)

    df_out = df.copy()

    # Parse and sort chronologically per station
    has_parsed = "_date_dt_internal" in df_out.columns
    if not has_parsed:
        df_out["_date_dt_internal"] = pd.to_datetime(df_out["Date"])

    # Ensure chronological order
    df_out = df_out.sort_values(by=["District", "Block/Station", "_date_dt_internal"]).reset_index(drop=True)

    grouped = df_out.groupby(["District", "Block/Station"])

    shifted_target = grouped[source_col].shift(-horizon)

    if align_calendar_days:
        shifted_date = grouped["_date_dt_internal"].shift(-horizon)
        expected_delta = pd.Timedelta(days=horizon)
        valid_alignment = (shifted_date - df_out["_date_dt_internal"]) == expected_delta
        # Set invalid or gap-shifted targets to NaN
        df_out[target_col] = shifted_target.where(valid_alignment, np.nan)
    else:
        df_out[target_col] = shifted_target

    # Clean internal helper column if we created it
    if not has_parsed:
        df_out.drop(columns=["_date_dt_internal"], inplace=True)

    return df_out


def generate_multi_horizon_targets(
    df: pd.DataFrame,
    horizons: Optional[List[int]] = None,
    source_col: str = "Flood_Occurred",
    preserve_next_day: bool = True,
    align_calendar_days: bool = True
) -> pd.DataFrame:
    """
    Generate targets across multiple forecast horizons (default: [1, 2, 3]).

    Parameters
    ----------
    df : pd.DataFrame
        Historical dataset.
    horizons : list of int, optional
        Forecast horizons to generate. Defaults to [1, 2, 3].
    source_col : str, default 'Flood_Occurred'
        Same-day flood observation column.
    preserve_next_day : bool, default True
        If True, ensures 'Flood_Next_Day' is present and synchronized with 'Flood_T+1'.
    align_calendar_days : bool, default True
        Strict calendar delta validation.

    Returns
    -------
    pd.DataFrame
        DataFrame augmented with multi-horizon targets.
    """
    if horizons is None:
        horizons = [1, 2, 3]

    df_out = df.copy()
    df_out["_date_dt_internal"] = pd.to_datetime(df_out["Date"])
    df_out = df_out.sort_values(by=["District", "Block/Station", "_date_dt_internal"]).reset_index(drop=True)

    grouped = df_out.groupby(["District", "Block/Station"])
    expected_deltas = {h: pd.Timedelta(days=h) for h in horizons}

    for h in horizons:
        col_name = get_horizon_target_name(h)
        shifted_target = grouped[source_col].shift(-h)

        if align_calendar_days:
            shifted_date = grouped["_date_dt_internal"].shift(-h)
            valid = (shifted_date - df_out["_date_dt_internal"]) == expected_deltas[h]
            df_out[col_name] = shifted_target.where(valid, np.nan)
        else:
            df_out[col_name] = shifted_target

    # Backward compatibility: preserve Flood_Next_Day
    if preserve_next_day:
        if "Flood_Next_Day" not in df_out.columns:
            df_out["Flood_Next_Day"] = df_out[get_horizon_target_name(1)]

    df_out.drop(columns=["_date_dt_internal"], inplace=True)
    return df_out


def verify_target_alignment(
    df: pd.DataFrame,
    horizon: int = 1,
    source_col: str = "Flood_Occurred",
    target_col: Optional[str] = None
) -> Dict[str, Any]:
    """
    Mathematically audit target alignment against future ground-truth dates.

    Verifies:
    1. For every row with valid target at date D, the value matches ground truth at Date(D + h days).
    2. Boundary conditions: Last h rows per station correctly have NaN targets.
    3. Cross-station isolation: No targets leak across district/station boundaries.
    """
    if target_col is None:
        target_col = get_horizon_target_name(horizon)

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    df_audit = df.copy()
    df_audit["_dt"] = pd.to_datetime(df_audit["Date"])
    df_audit = df_audit.sort_values(by=["District", "Block/Station", "_dt"]).reset_index(drop=True)

    grouped = df_audit.groupby(["District", "Block/Station"])
    shifted_dt = grouped["_dt"].shift(-horizon)
    shifted_source = grouped[source_col].shift(-horizon)

    # Check date arithmetic
    expected_delta = pd.Timedelta(days=horizon)
    date_matches = (shifted_dt - df_audit["_dt"]) == expected_delta

    # Valid targets where date matches
    valid_mask = df_audit[target_col].notna()
    mismatch_values = (df_audit.loc[valid_mask, target_col] != shifted_source.loc[valid_mask]).sum()
    mismatch_dates = (~date_matches & valid_mask).sum()

    total_valid = int(valid_mask.sum())
    total_rows = len(df_audit)
    num_positives = int((df_audit[target_col] == 1).sum())

    audit_result = {
        "horizon": horizon,
        "target_col": target_col,
        "total_rows": total_rows,
        "valid_target_rows": total_valid,
        "null_target_rows": total_rows - total_valid,
        "positive_flood_count": num_positives,
        "positive_flood_rate": round(float(num_positives / max(total_valid, 1)), 5),
        "value_mismatches": int(mismatch_values),
        "date_delta_mismatches": int(mismatch_dates),
        "is_perfectly_aligned": bool(mismatch_values == 0 and mismatch_dates == 0)
    }

    return audit_result
