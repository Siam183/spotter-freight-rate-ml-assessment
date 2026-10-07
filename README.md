# Freight Rate Prediction Challenge - Machine Learning Assessment

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![CatBoost](https://img.shields.io/badge/model-CatBoost-green.svg)](https://catboost.ai/)
[![Status](https://img.shields.io/badge/status-complete-success.svg)](#)

This repository contains the end-to-end Machine Learning pipeline and submission artifacts for the Spotter Freight Rate ML Assessment[cite: 6]. The objective is to predict spot freight rates for 12,000 unlabeled validation loads (Nov–Dec 2025) and score a 31-day December 2025 fixed-lane scenario[cite: 6].

---

## 📌 Executive Summary

- **Development Data**: 48,000 labeled loads (January 1 – October 31, 2025)[cite: 6].
- **Validation Task**: 12,000 unlabeled loads (November 1 – December 31, 2025)[cite: 6].
- **Validation Strategy**: Out-of-time chronological split (Train: Jan–Aug 2025 | Internal Val: Sep–Oct 2025) to prevent optimistic temporal leakage[cite: 6].
- **Selected Model**: **CatBoost Regressor** (Internal Val MAE: **$122.40**, RMSE: **$636.02** vs XGBoost MAE: $208.78)[cite: 6].
- **December Scenario**: Generated 31 daily predictions for Lexington $\rightarrow$ Fort Wayne lane and generated `candidate_december.png` using `score.py`[cite: 6].

---

## 📁 Repository Layout

```text
spotter-freight-rate-ml-assessment/
│
├── README.md                           ← Project setup, execution, and strategy guide[cite: 6]
├── requirements.txt                    ← Environment dependencies
├── .gitignore                          ← Git rules
├── validation_predictions.csv          ← Final 12,000 validation rate predictions[cite: 6]
│
├── src/                                ← Source pipeline and model code
│   ├── pipeline.py                     ← Cleaning, imputation, and feature engineering[cite: 6]
│   ├── train.py                        ← Model training & artifact builder[cite: 6]
│   ├── predict.py                      ← Validation set prediction runner[cite: 6]
│   └── december_predict.py             ← December 31-day scenario predictor[cite: 6]
│
├── data/                               ← Input datasets
│   └── README.md                       ← Dataset dictionary & schema guide
│
├── models/                             ← Trained models and preprocessing statistics[cite: 6]
│   ├── catboost_freight_rate_final.cbm
│   ├── final_preprocessing_artifacts.pkl
│   └── catboost_final_parameters.json
│
├── predictions/                        ← Prediction output exports[cite: 6]
│   ├── validation_predictions.csv
│   └── december-chart-inputs-scored.csv
│
├── figures/                            ← Generated evaluation charts[cite: 6]
│   └── candidate_december.png
│
└── reports/                            ← Written final analysis report[cite: 6]
    └── Siam_Al_Qureshi_Spotter_ML_Assessment_Report.docx

⚙️ Environment Setup & Installation
Clone the repository:

Bash
git clone <your-repository-url>
cd spotter-freight-rate-ml-assessment
Create and activate a virtual environment:

Bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
# venv\Scripts\activate       # Windows PowerShell
Install dependencies:

Bash
pip install -r requirements.txt
🚀 Execution & Reproducibility Pipeline
Execute the pipeline step-by-step from the root directory[cite: 6]:

1. Train Model & Persist Preprocessing Artifacts
Fits imputation rules on data/train_test.csv[cite: 6], generates spatial/temporal engineered features[cite: 6], and trains the final CatBoost Regressor model[cite: 6]:

Bash
python src/train.py
2. Generate Validation Predictions
Applies saved preprocessing statistics and model to data/validation.csv[cite: 6], exporting validation_predictions.csv to the root folder[cite: 6]:

Bash
python src/predict.py
3. Generate December Scenario Predictions
Scores data/december_chart_inputs.csv for the 31-day December evaluation scenario[cite: 6]:

Bash
python src/december_predict.py
4. Run Official Assessment Scorer
Validates prediction schemas and outputs the official chart scorer_results/candidate_december.png[cite: 6]:

Bash
python score.py --predictions validation_predictions.csv --december-predictions predictions/december-chart-inputs-scored.csv
📈 Key Findings & Feature Importance
Distance Dominance: Operational road distance (distance) and Haversine straight-line distance (geo_distance) are the strongest drivers of freight rate (Pearson r≈0.909)[cite: 6].

Top 5 Features:

geo_distance (32.07%)[cite: 6]

distance (19.20%)[cite: 6]

distance_quote (12.55%)[cite: 6]

distance_market (10.58%)[cite: 6]

abs_lon_diff (7.95%)[cite: 6]
(Combined feature importance: ~82%[cite: 6])

👤 Author & Submission Information
Author: Siam Al Qureshi[cite: 6]

Role Target: Machine Learning Engineer Assessment[cite: 6]

Report Document: reports/Siam_Al_Qureshi_Spotter_ML_Assessment_Report.docx

[cite: 6]
