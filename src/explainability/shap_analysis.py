"""
SHAP (SHapley Additive exPlanations) analysis module.
Computes global SHAP importance and local prediction explanations.
"""

import os
import joblib
import json
import shap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Any, Optional

from src.data.load_data import load_combined_data, load_config


def compute_shap_analysis(
    model,
    X_train_sample: pd.DataFrame,
    X_test_sample: pd.DataFrame,
    save_plot_path: Optional[str] = "results/plots/shap_summary.png",
    save_csv_path: Optional[str] = "results/metrics/shap_feature_importance.csv"
) -> Tuple[np.ndarray, shap.Explanation, pd.DataFrame]:
    """
    Compute SHAP values using exact LinearExplainer or TreeExplainer.
    """
    clf = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
    scaler = model.named_steps["scaler"] if hasattr(model, "named_steps") else None
    
    if scaler is not None:
        X_train_transformed = pd.DataFrame(
            scaler.transform(X_train_sample),
            columns=X_train_sample.columns,
            index=X_train_sample.index
        )
        X_test_transformed = pd.DataFrame(
            scaler.transform(X_test_sample),
            columns=X_test_sample.columns,
            index=X_test_sample.index
        )
    else:
        X_train_transformed = X_train_sample
        X_test_transformed = X_test_sample
        
    # Choose explainer
    if hasattr(clf, "coef_"):
        explainer = shap.LinearExplainer(clf, X_train_transformed)
        shap_values = explainer(X_test_transformed)
        values = shap_values.values if hasattr(shap_values, "values") else shap_values
    else:
        try:
            explainer = shap.TreeExplainer(clf)
            shap_values = explainer(X_test_transformed)
            values = shap_values.values if hasattr(shap_values, "values") else shap_values
        except Exception:
            explainer = shap.Explainer(clf.predict_proba, X_train_transformed.iloc[:100])
            shap_values = explainer(X_test_transformed)
            values = shap_values.values if hasattr(shap_values, "values") else shap_values
        
    # Mean absolute SHAP values per feature
    if len(values.shape) == 3: # multi-class/output
        mean_abs_shap = np.mean(np.abs(values[:, :, 1]), axis=0)
    else:
        mean_abs_shap = np.mean(np.abs(values), axis=0)
        
    df_shap_imp = pd.DataFrame({
        "Feature": X_test_sample.columns,
        "Mean_Absolute_SHAP": mean_abs_shap
    }).sort_values(by="Mean_Absolute_SHAP", ascending=False).reset_index(drop=True)
    
    if save_csv_path:
        os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
        df_shap_imp.to_csv(save_csv_path, index=False)
        
    # Plot SHAP summary
    if save_plot_path:
        os.makedirs(os.path.dirname(save_plot_path), exist_ok=True)
        plt.figure(figsize=(10, 7), dpi=300)
        shap.summary_plot(
            shap_values if hasattr(shap_values, "values") else values,
            X_test_sample,
            show=False,
            max_display=15
        )
        plt.title("SHAP Global Feature Importance (Impact on Flood Risk)", fontsize=14, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(save_plot_path, bbox_inches="tight")
        plt.close()
        
    return values, shap_values, df_shap_imp


def explain_instance(
    model,
    feature_dict: Dict[str, Any],
    feature_names: List[str]
) -> Dict[str, Any]:
    """
    Produce structured natural-language and quantitative explanation for a single prediction.
    """
    clf = model.named_steps["classifier"] if hasattr(model, "named_steps") else model
    scaler = model.named_steps["scaler"] if hasattr(model, "named_steps") else None
    
    df_single = pd.DataFrame([feature_dict])[feature_names]
    
    if scaler is not None:
        X_sc = scaler.transform(df_single)[0]
    else:
        X_sc = df_single.values[0]
        
    if hasattr(clf, "coef_"):
        coefs = clf.coef_[0]
        intercept = clf.intercept_[0]
        contributions = X_sc * coefs
        
        df_contrib = pd.DataFrame({
            "feature": feature_names,
            "value": df_single.values[0],
            "contribution": contributions,
            "abs_contribution": np.abs(contributions)
        }).sort_values(by="abs_contribution", ascending=False)
        
        top_risk_drivers = df_contrib[df_contrib["contribution"] > 0].head(5).to_dict(orient="records")
        top_mitigators = df_contrib[df_contrib["contribution"] < 0].head(5).to_dict(orient="records")
        
        return {
            "top_risk_drivers": top_risk_drivers,
            "top_mitigators": top_mitigators,
            "base_log_odds": float(intercept)
        }
    else:
        # Default top features from input
        return {
            "top_risk_drivers": [],
            "top_mitigators": [],
            "base_log_odds": 0.0
        }
