# Predicting Short-Term Drawdowns of US Large-Cap Stocks

An imbalanced classification problem.
Machine Learning Project, ESILV A4 Financial Engineering, 2026-2027.

**Team:** Chloé Choueiry, Hugo Clauzon, Mathis Henry

**Research question:** Using only information available at day *t*, can a model identify the stocks that will suffer a drawdown of at least X% within the next 5 trading days, better than simple baselines, while handling class imbalance without data leakage?

---

## 1. How to Run

Python version: `TODO`

```bash
pip install -r requirements.txt
# TODO: add the commands, in execution order
```

## 2. Repository Structure

| Path | Content |
|---|---|
| `src/` | Source code |
| `notebooks/` | Exploratory data analysis |
| `tests/` | Leakage tests |
| `data/raw/` | Frozen data snapshot |
| `data/processed/` | Clean dataset |
| `results/` | Metrics and figures |
| `ai_governance/` | Contributions, prompt log, AI audit |
| `report/` | Final report |

## 3. Data

| Source | Content |
|---|---|
| Yahoo Finance (`yfinance`) | Daily adjusted OHLCV |
| FRED `VIXCLS` | VIX index |
| FRED `T10Y2Y` | 10Y minus 2Y Treasury spread |

- **Snapshot date:** TODO
- **Download period:** TODO
- **Stock universe and selection criteria:** TODO
- **Final size (rows, columns, tickers):** TODO

## 4. Data Cleaning

| Issue found | Number of cases | Treatment | Justification |
|---|---|---|---|
| TODO | | | |

## 5. Features and Target

| Feature | Definition |
|---|---|
| TODO | |

**Target:** TODO

## 6. Exploratory Data Analysis (training set only)

- **Class distribution for each candidate X:** TODO
- **Prevalence over time and across tickers:** TODO
- **Missing values:** TODO
- **Outliers:** TODO
- **Correlations and VIF:** TODO
- **Chosen threshold X and justification:** TODO

## 7. Data Split and Leakage Control

| Set | Period | Rows | Positive rate | Role |
|---|---|---|---|---|
| Train | 2010-2018 | TODO | TODO | Fit models |
| Validation | 2019-2021 | TODO | TODO | Tuning and threshold choice |
| Test | 2022-2025 | not reported yet | not reported yet | Final evaluation, used once |

- **Embargo:** TODO
- **Walk-forward folds:** TODO
- **Preprocessing fitted on training data only:** TODO
- **Leakage tests:** TODO

## 8. Baselines

| Model | Description | Features | Why this baseline |
|---|---|---|---|
| B0 | TODO | | |
| B1 | TODO | | |
| B2 | TODO | | |

## 9. Evaluation Protocol

- **Cost assumption and justification:** A missed drawdown (FN) costs 5 and a false alarm (FP) costs 1. We consider a missed loss more harmful than an unnecessary alert. This is a modelling assumption, not an estimate from data. Ratios of 3:1 and 10:1 are tested as a sensitivity analysis; 5:1 remains the main setting.
- **Threshold selection rule:** For B1 and B2, the threshold minimises 5·FN + FP on the validation set. In case of a tie, the highest threshold is kept. The sensitivity analysis applies the same rule with each ratio.
- **Target precision rule:** Twice the training prevalence, computed once X is chosen and before running the models. If a model never reaches it, recall is reported as 0 (target not reached). We must be twice as good as random alerts.
- **Metrics:** PR-AUC (primary), recall at target precision, F2, ROC-AUC (secondary). Precision, recall, accuracy and cost are also reported. Models are compared with the "never alert" (B0) and "always alert" costs.

## 10. Initial Results (validation set)

No-skill PR-AUC (validation prevalence): TODO

| Model | Threshold | PR-AUC | Recall at target precision | F2 | ROC-AUC | Precision | Recall | Accuracy | Cost |
|---|---|---|---|---|---|---|---|---|---|
| B0 | | | | | | | | | |
| B1 | | | | | | | | | |
| B2 | | | | | | | | | |
| Always alert | | | | | | | | | |

*Metrics computed at a threshold selected on the validation set are optimistic.*

**Sensitivity to the cost ratio:** TODO

**Interpretation:** TODO

## 11. Limitations

- TODO

## 12. Changes Since Kick-off

| Topic | Kick-off document | Current choice | Reason |
|---|---|---|---|
| B1 threshold | Chosen on the training set | Chosen on the validation set | Same selection rule as B2 |
| Cost ratio | Cost-aware threshold, no ratio given | 5:1 (3:1 and 10:1 as sensitivity) | Make the decision rule explicit |
| Target precision | Recall at fixed precision, no value given | Twice the training prevalence | Set the rule before seeing model results |

## 13. Next Steps

- TODO