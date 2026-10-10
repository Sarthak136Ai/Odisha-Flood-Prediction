"""
Unit tests for the event-based evaluation engine and visualization module.
"""

import os
import shutil
import pytest
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")

from src.evaluation.event_evaluation import (
    FloodEventDetector,
    plot_flood_event_timeline,
    plot_multi_model_event_metrics_comparison
)


@pytest.fixture
def sample_detector():
    return FloodEventDetector(max_gap_days=1, min_event_duration=1)


@pytest.fixture
def tmp_test_dir():
    test_dir = "tests/tmp_event_eval"
    os.makedirs(test_dir, exist_ok=True)
    yield test_dir
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)


def test_extract_events_from_series_single(sample_detector):
    dates = pd.date_range("2023-08-01", periods=10, freq="D")
    binary = np.array([0, 0, 1, 1, 1, 0, 0, 0, 0, 0])
    probas = np.array([0.1, 0.2, 0.8, 0.9, 0.7, 0.1, 0.1, 0.1, 0.1, 0.1])

    events = sample_detector.extract_events_from_series(dates, binary, probas, event_prefix="TEST")
    assert len(events) == 1
    evt = events[0]
    assert evt["event_id"] == "TEST_001"
    assert evt["start_date"] == "2023-08-03"
    assert evt["end_date"] == "2023-08-05"
    assert evt["duration_days"] == 3
    assert evt["active_days_count"] == 3
    assert evt["peak_date"] == "2023-08-04"
    assert evt["max_probability"] == 0.9


def test_extract_events_gap_tolerance(sample_detector):
    # Gap of 1 day (max_gap_days=1) should be merged into a single event
    dates = pd.date_range("2023-08-01", periods=10, freq="D")
    binary = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 0])
    events = sample_detector.extract_events_from_series(dates, binary, event_prefix="GAP")
    assert len(events) == 1
    assert events[0]["duration_days"] == 5
    assert events[0]["start_date"] == "2023-08-02"
    assert events[0]["end_date"] == "2023-08-06"


def test_extract_events_empty_series(sample_detector):
    dates = pd.date_range("2023-08-01", periods=5, freq="D")
    binary = np.array([0, 0, 0, 0, 0])
    events = sample_detector.extract_events_from_series(dates, binary)
    assert len(events) == 0


def test_match_events_hit_and_lead_time(sample_detector):
    dates = pd.date_range("2023-08-01", periods=15, freq="D")
    # Actual flood from Aug 5 to Aug 8
    y_true = np.zeros(15, dtype=int)
    y_true[4:8] = 1

    # Predicted alert from Aug 3 to Aug 8 (2 days lead time)
    y_pred = np.zeros(15, dtype=int)
    y_pred[2:8] = 1

    act_evts = sample_detector.extract_events_from_series(dates, y_true, event_prefix="ACT")
    pred_evts = sample_detector.extract_events_from_series(dates, y_pred, event_prefix="PRD")

    matched, fa, summary = sample_detector.match_events(
        actual_events=act_evts,
        predicted_events=pred_evts,
        lead_tolerance_days=2,
        daily_y_true=pd.Series(y_true),
        daily_y_pred=pd.Series(y_pred),
        daily_dates=dates
    )

    assert len(matched) == 1
    assert matched[0]["detection_status"] == "HIT"
    assert matched[0]["lead_time_days"] == 2
    assert matched[0]["missed_days"] == 0
    assert matched[0]["coverage_pct"] == 100.0
    assert len(fa) == 0
    assert summary["detected_events_hits"] == 1
    assert summary["event_recall_hit_rate"] == 1.0


def test_match_events_missed(sample_detector):
    dates = pd.date_range("2023-08-01", periods=10, freq="D")
    y_true = np.zeros(10, dtype=int)
    y_true[2:5] = 1  # Aug 3 to Aug 5

    y_pred = np.zeros(10, dtype=int)  # No alerts

    act_evts = sample_detector.extract_events_from_series(dates, y_true, event_prefix="ACT")
    pred_evts = sample_detector.extract_events_from_series(dates, y_pred, event_prefix="PRD")

    matched, fa, summary = sample_detector.match_events(
        actual_events=act_evts,
        predicted_events=pred_evts,
        lead_tolerance_days=2,
        daily_y_true=pd.Series(y_true),
        daily_y_pred=pd.Series(y_pred),
        daily_dates=dates
    )

    assert len(matched) == 1
    assert matched[0]["detection_status"] == "MISSED"
    assert matched[0]["missed_days"] == 3
    assert matched[0]["coverage_pct"] == 0.0
    assert summary["missed_events"] == 1
    assert summary["event_recall_hit_rate"] == 0.0


def test_match_events_false_alarm(sample_detector):
    dates = pd.date_range("2023-08-01", periods=10, freq="D")
    y_true = np.zeros(10, dtype=int)
    y_pred = np.zeros(10, dtype=int)
    y_pred[3:6] = 1  # False alarm alert

    act_evts = sample_detector.extract_events_from_series(dates, y_true, event_prefix="ACT")
    pred_evts = sample_detector.extract_events_from_series(dates, y_pred, event_prefix="PRD")

    matched, fa, summary = sample_detector.match_events(
        actual_events=act_evts,
        predicted_events=pred_evts,
        lead_tolerance_days=2
    )

    assert len(matched) == 0
    assert len(fa) == 1
    assert fa[0]["status"] == "FALSE ALARM (Precautionary Standby)"
    assert summary["false_alarm_events"] == 1


def test_evaluate_dataframe_events_spatiotemporal(sample_detector):
    dates = pd.date_range("2023-08-01", periods=10, freq="D")
    df_st1 = pd.DataFrame({
        "Date": dates,
        "Block/Station": "Station_A",
        "Flood_Next_Day": [0, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    })
    df_st2 = pd.DataFrame({
        "Date": dates,
        "Block/Station": "Station_B",
        "Flood_Next_Day": [0, 0, 0, 0, 0, 1, 1, 0, 0, 0]
    })
    df_comb = pd.concat([df_st1, df_st2], ignore_index=True)

    y_pred = np.array([
        0, 1, 1, 1, 0, 0, 0, 0, 0, 0,  # Station A: HIT
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0   # Station B: MISSED
    ])
    y_proba = np.array([
        0.1, 0.9, 0.9, 0.8, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1,
        0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1
    ])

    matched, fa, summary = sample_detector.evaluate_dataframe_events(
        df=df_comb,
        y_true_col="Flood_Next_Day",
        y_pred=y_pred,
        y_proba=y_proba,
        date_col="Date",
        station_col="Block/Station",
        lead_tolerance_days=2
    )

    assert summary["total_actual_events"] == 2
    assert summary["detected_events_hits"] == 1
    assert summary["missed_events"] == 1
    assert summary["event_recall_hit_rate"] == 0.5


def test_plot_flood_event_timeline(tmp_test_dir):
    dates = pd.date_range("2023-08-01", periods=15, freq="D")
    rainfall = pd.Series(np.random.exponential(scale=20.0, size=15))
    y_true = pd.Series([0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    y_proba = pd.Series([0.1, 0.2, 0.6, 0.85, 0.92, 0.75, 0.3, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])

    png_path = os.path.join(tmp_test_dir, "test_timeline.png")
    svg_path = os.path.join(tmp_test_dir, "test_timeline.svg")

    fig = plot_flood_event_timeline(
        dates=dates,
        rainfall_mm=rainfall,
        y_true=y_true,
        y_proba=y_proba,
        threshold=0.5,
        event_name="Test Flood Inundation",
        district_name="Cuttack",
        save_path_png=png_path,
        save_path_svg=svg_path
    )

    assert fig is not None
    assert os.path.exists(png_path)
    assert os.path.exists(svg_path)
    assert os.path.getsize(png_path) > 1000


def test_plot_multi_model_event_metrics_comparison(tmp_test_dir):
    sample_data = {
        "Logistic Regression": {
            "event_recall_hit_rate": 0.58,
            "event_miss_rate": 0.42,
            "mean_lead_time_days": 0.65,
            "false_alarm_events": 2871
        },
        "XGBoost": {
            "event_recall_hit_rate": 0.64,
            "event_miss_rate": 0.36,
            "mean_lead_time_days": 1.22,
            "false_alarm_events": 3249
        }
    }

    png_path = os.path.join(tmp_test_dir, "test_multi_model.png")
    svg_path = os.path.join(tmp_test_dir, "test_multi_model.svg")

    fig = plot_multi_model_event_metrics_comparison(
        models_event_data=sample_data,
        save_path_png=png_path,
        save_path_svg=svg_path
    )

    assert fig is not None
    assert os.path.exists(png_path)
    assert os.path.exists(svg_path)
