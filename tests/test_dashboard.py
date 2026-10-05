"""
Unit test for dashboard loading and resource initialization.
"""

import os
import pytest
from app.app import get_system_resources


def test_dashboard_resources_loading():
    config, df, bot, pred_tool, comp_df, yw_df, down_df, imp_df = get_system_resources()
    
    assert config is not None
    assert len(df) == 2752252, f"Expected 2,752,252 rows in dashboard dataset, got {len(df)}"
    assert bot is not None
    assert pred_tool is not None
    assert not comp_df.empty, "Model comparison metrics table empty in dashboard"
    assert not yw_df.empty, "Year-wise metrics table empty in dashboard"
    assert not down_df.empty, "Downscaling metrics table empty in dashboard"
    assert not imp_df.empty, "Feature importance table empty in dashboard"
