# Temporal Validation Strategy & Walk-Forward Cross-Validation

## 🌊 Executive Summary & Scientific Motivation

Hydrological disaster forecasting is governed by non-stationary meteorological processes, seasonal cyclonic regimes, and severe spatio-temporal autocorrelation. Standard randomly shuffled $k$-fold cross-validation is **scientifically invalid** for flood prediction systems for three critical reasons:

1. **Future-to-Past Target Leakage**: Shuffling allows observations from future monsoon seasons (e.g. 2021) to inform model parameters tested on past events (e.g. 2017).
2. **Autoregressive Temporal Spillover**: Antecedent rolling accumulations (such as `Rainfall_Prev_7d_Sum` or `Rainfall_Prev_30d_Sum`) create dependencies across consecutive days; random splitting places adjacent days across train and test partitions, artificially inflating ROC-AUC and PR-AUC.
3. **Distribution Shift Blindness**: Static random splits fail to evaluate model resilience against inter-annual climate shifts, El Niño / La Niña oscillation cycles, and shifting monsoon intensity.

To ensure realistic, deployable early-warning performance, this repository implements a **Causal Walk-Forward Temporal Cross-Validation Strategy** with an expanding training window across 2001–2021, while maintaining a strictly isolated final test evaluation on **2022–2024**.

---

## 📅 Walk-Forward Fold Architecture

The temporal validation protocol partitions 24 years (2001–2024, 2,752,252 records) into 5 non-overlapping expanding folds followed by a held-out operational test horizon:

```
Historical Dataset (2001–2024)
├─ Walk-Forward Validation Horizon (2001–2021)
│  ├─ Fold 1: [Train: 2001–2016 (16 yrs, 1,834,769 rows)] ──► [Val: 2017 (1 yr, 114,610 rows)]
│  ├─ Fold 2: [Train: 2001–2017 (17 yrs, 1,949,379 rows)] ──► [Val: 2018 (1 yr, 114,610 rows)]
│  ├─ Fold 3: [Train: 2001–2018 (18 yrs, 2,063,989 rows)] ──► [Val: 2019 (1 yr, 114,610 rows)]
│  ├─ Fold 4: [Train: 2001–2019 (19 yrs, 2,178,599 rows)] ──► [Val: 2020 (1 yr, 114,610 rows)]
│  └─ Fold 5: [Train: 2001–2020 (20 yrs, 2,293,209 rows)] ──► [Val: 2021 (1 yr, 114,610 rows)]
│
└─ Strictly Held-Out Test Set (2022–2024, 343,758 rows)
   └─ Reserved exclusively for final unbiased generalization audit.
```

### Mathematical Formulation
For each fold $k \in \{1, 2, 3, 4, 5\}$:
$$\mathcal{D}_{\text{train}}^{(k)} = \left\{ (x_{i,t}, y_{i,t+1}) \mid t \in [2001, 2016 + k - 1], \, i \in \text{Districts} \right\}$$
$$\mathcal{D}_{\text{val}}^{(k)} = \left\{ (x_{i,t}, y_{i,t+1}) \mid t = 2016 + k, \, i \in \text{Districts} \right\}$$

$$\max\left(\text{Years}\left(\mathcal{D}_{\text{train}}^{(k)}\right)\right) < \min\left(\text{Years}\left(\mathcal{D}_{\text{val}}^{(k)}\right)\right) < \min(\text{Test Years})$$

---

## 🛡️ Strict Isolation & Zero-Leakage Guarantees

1. **Chronological Causal Order**:
   - The validation year is always strictly ahead of the maximum training year.
   - Zero future observations enter model training or parameter estimation.

2. **Preprocessor & Scaler Isolation**:
   - For every fold $k$, preprocessors (such as `StandardScaler`) are instantiated fresh and **fitted exclusively on $\mathcal{D}_{\text{train}}^{(k)}$**.
   - Feature means $\mu_{\text{train}}^{(k)}$ and variances $\sigma_{\text{train}}^{(k)}$ are never informed by validation or test distributions:
     $$\hat{x}_{\text{val}}^{(k)} = \frac{x_{\text{val}}^{(k)} - \mu_{\text{train}}^{(k)}}{\sigma_{\text{train}}^{(k)}}$$

3. **Class-Imbalance Weight Isolation**:
   - Imbalance weighting parameters (such as `scale_pos_weight` in XGBoost and class weights in Logistic Regression) are calculated strictly from the empirical positive rate of $\mathcal{D}_{\text{train}}^{(k)}$.

4. **Independent Threshold Optimization**:
   - Because the historical flood observation rate is $\approx 2.638\%$ (severe class imbalance), standard default threshold $0.50$ is suboptimal for disaster management.
   - For each fold, an optimal threshold $T_k^*$ is computed on the validation set to maximize the $F_1$-score (harmonic mean of Precision and Recall).

5. **Held-Out Test Set Decoupling**:
   - The final test years (2022–2024) are completely absent from all walk-forward folds.
   - No model selection, hyperparameter tuning, or threshold setting utilizes test data.

---

## 📊 Tracked Evaluation Metrics

For each fold and model architecture, the engine computes:

| Metric | Formula / Description | Hydrological Relevance |
|---|---|---|
| **PR-AUC** | Area under Precision-Recall Curve ($\int P(R) dR$) | **Primary metric** under extreme class imbalance ($2.6\%$ positive rate) |
| **ROC-AUC** | Area under Receiver Operating Characteristic | Discrimination power across all potential decision thresholds |
| **Optimal $F_1$** | $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ at $T^*$ | Balanced early-warning performance |
| **Recall (Sensitivity)** | $\frac{TP}{TP + FN}$ | Inundation detection rate (minimizing missed disasters) |
| **Precision** | $\frac{TP}{TP + FP}$ | False alarm suppression for civil defense logistics |
| **Specificity** | $\frac{TN}{TN + FP}$ | Correct identification of safe, non-inundated days |
| **Brier Score** | $\frac{1}{N}\sum (p_i - y_i)^2$ | Probability calibration and forecast confidence reliability |
| **ECE** | Expected Calibration Error | Binned reliability deviation from perfect calibration |

---

## 🚀 Running the Temporal Validation Engine

The temporal validation pipeline is fully automated and CLI-executable:

```bash
# Run walk-forward validation across all 5 model architectures
python scripts/run_temporal_validation.py

# Run for specific candidate models
python scripts/run_temporal_validation.py --models "Logistic Regression" "XGBoost"
```

### Generated Artifacts
- `reports/temporal_validation_results.csv`: Complete fold-by-fold results table.
- `reports/temporal_validation_summary.csv`: Aggregated $\text{Mean} \pm \text{Std}$ table across all folds.
- `reports/temporal_validation_summary.json`: Structured machine-readable schema for automated CI/CD.
- `reports/plots/temporal_validation_folds.png`: Multi-panel comparison of PR-AUC, ROC-AUC, F1, and Brier scores across folds.
- `reports/plots/temporal_validation_pr_curves.png`: Precision-Recall curves across all chronological folds.

---

## 🧪 Automated Verification Suite

Validation constraints are enforced via automated tests in `tests/test_temporal_validation.py`:
- `test_temporal_split_chronological_ordering`: Asserts causal direction $\max(\text{train}) < \min(\text{val})$.
- `test_zero_train_validation_index_overlap`: Asserts 0 overlapping row indices between splits.
- `test_expanding_window_growth`: Asserts strictly monotonic training sample growth.
- `test_held_out_test_years_isolation`: Asserts complete exclusion of 2022–2024 from validation folds.
- `test_scaler_preprocessor_training_isolation`: Asserts scaler parameters match training data exclusively.
- `test_walk_forward_evaluation_metrics_validity`: Asserts all calculated metrics remain within $[0, 1]$ bounds.
