# Predicting Short-Term Drawdowns of US Large-Cap Stocks

An imbalanced classification problem.
Machine Learning Project, ESILV A4 Financial Engineering, 2026-2027.

**Team:** Chloé Choueiry, Hugo Clauzon, Mathis Henry

**Research question:** Using only information available at day *t*, can a model identify the stocks that will suffer a drawdown of at least X% within the next 5 trading days, better than simple baselines, while handling class imbalance without data leakage?

---

## 1. How to Run

Python version: 3.14.8

```bash
pip install -r requirements.txt
python3 -m src.data_cleaning
python3 -m src.features
python3 -m pytest
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

- **Snapshot date:** 2026-11-27
- **Download period:** TODO
- **Stock universe and selection criteria:** TODO
- **Final size (rows, columns, tickers):** TODO

## 4. Data Cleaning

| Issue found | Number of cases | Treatment | Justification |
|---|---|---|---|
| Duplicates, missing values, missing stock-days | 0 | None | Missing days are checked against the dates observed across all stocks |
| Non-positive prices, negative or zero volumes | 0 | None | No issue detected |
| Incoherent high/low prices | 0 | None | Relative tolerance of 1e-6 |
| Daily moves above 25% | 14 | Kept | No correction without evidence of a data error; most occur in early 2009 |
| Missing T10Y2Y on trading days | 32 | Forward-fill | Uses earlier observations only |
| Missing VIX on trading days | 0 | None | No issue detected |

## 5. Features and Target

| Feature | Definition |
|---|---|
| `ret_1d`, `ret_5d`, `ret_22d` | close(t) / close(t-n) - 1 |
| `park_vol_5d`, `park_vol_22d` | Square root of the n-day mean of ln(high/low)² / (4 ln 2), daily scale |
| `dist_high_252d` | close(t) / highest close over the last 252 days (t included) - 1 |
| `volume_ratio_20d` | volume(t) / mean volume over the last 20 days (t included) |
| `vix` | VIX value dated t, forward-filled if missing |
| `vix_chg_5d` | vix(t) - vix(t-5), in points |
| `t10y2y` | 10Y-2Y Treasury spread dated t, forward-filled if missing |

**Target:** y = 1 if the lowest close from t+1 to t+5 is at most (1 - X) × close(t), else y = 0. Rows without 5 future trading days are removed. X is selected using the training set only.


## 6. Exploratory Data Analysis (training set only)

- **Class distribution for each candidate X:** On the training set, a 5% threshold yielded 5,332 positive observations out of 90,360 (5.90%), compared with 2,165 (2.40%) at 7%.
- **Prevalence over time and across tickers:** In 2017, the year with the fewest positives, the 5% threshold yielded 220 positives compared with 57 at 7%. For Coca-Cola (KO), the counts were 36 and 4, respectively.
- **Missing values:** None of the ten features had missing values in the 90,360 training observations.
- **Outliers:** Using the 1.5 × IQR rule, 7.39% of observations were flagged for the five-day VIX change, 5.55% for the one-day return, and 5.49% for the VIX level. No observations were flagged for the 10Y–2Y Treasury spread. All flagged values were retained at this stage.
- **Correlations and VIF:** The strongest pairwise correlation was between the 5-day and 22-day Parkinson volatility estimates (0.81). Their VIFs were 3.56 and 4.16, respectively. All VIFs were below 5, so we retained all ten features for the initial logistic regression baseline.
- **Chosen threshold X and justification:** We selected X = 5% using only the 2010–2018 training set. It provided about 2.5 times as many positive observations as the 7% threshold while keeping the positive class rare (5.90%).

## 7. Data Split and Leakage Control

| Set | Period | Rows | Positive rate | Role |
|---|---|---|---|---|
| Train | 2010-01-04 to 2018-12-21 | 90,360 | TODO | Fit models |
| Validation | 2019-01-02 to 2021-12-23 | 30,080 | TODO | Tuning and threshold choice |
| Test | 2022-01-03 to 2025-12-23 | 39,920 | not reported yet | Final evaluation, used once |

- **Embargo:** The last 5 trading dates of the train and validation sets are removed, because their labels use prices from the next period.
- **Walk-forward folds:** 4 expanding-window folds inside the training set, each validated on one year (2015 to 2018), with a 5-trading-day embargo.
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