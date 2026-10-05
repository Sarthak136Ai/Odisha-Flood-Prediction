"""
Year-wise historical performance and temporal stability evaluation module.
"""

import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support, accuracy_score, brier_score_loss
from typing import Dict, List, Any


def evaluate_year_wise_performance(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float = 0.5,
    year_col: str = "Year"
) -> pd.DataFrame:
    """
    Compute performance breakdown for each individual year to assess temporal drift and stability.
    """
    df_eval = pd.DataFrame({
        "Year": df[year_col].values,
        "y_true": y_true,
        "y_proba": y_proba,
        "y_pred": (y_proba >= threshold).astype(int)
    })
    
    rows = []
    for yr, group in df_eval.groupby("Year"):
        y_t = group["y_true"].values
        y_p = group["y_proba"].values
        y_hat = group["y_pred"].values
        
        pos_count = int(np.sum(y_t == 1))
        total_count = len(y_t)
        
        if pos_count > 0 and pos_count < total_count:
            roc_auc = float(roc_auc_score(y_t, y_p))
            pr_auc = float(average_precision_score(y_t, y_p))
        else:
            roc_auc = np.nan
            pr_auc = np.nan
            
        prec, rec, f1, _ = precision_recall_fscore_support(y_t, y_hat, average="binary", zero_division=0)
        acc = accuracy_score(y_t, y_hat)
        brier = brier_score_loss(y_t, y_p)
        
        rows.append({
            "Year": int(yr),
            "Observations": total_count,
            "Flood_Days": pos_count,
            "Flood_Rate_Pct": round(pos_count / total_count * 100, 2),
            "Accuracy": round(acc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1_Score": round(f1, 4),
            "ROC_AUC": round(roc_auc, 4) if not np.isnan(roc_auc) else None,
            "PR_AUC": round(pr_auc, 4) if not np.isnan(pr_auc) else None,
            "Brier_Score": round(brier, 4)
        })
        
    res_df = pd.DataFrame(rows).sort_values(by="Year").reset_index(drop=True)
    return res_df
