# Freight Rate Prediction Challenge

Machine Learning Engineer assessment solution for freight-rate prediction.

## Project Overview

This project develops a machine learning model to predict freight/load rates using the provided labeled development dataset. The workflow includes data exploration, data-quality handling, feature engineering, model selection, validation, final prediction generation, and December rate forecasting.

The final model is a **CatBoost Regressor** selected after comparing and tuning several hyperparameters using a chronological train/validation split.

## Repository Structure

```text
spotter-freight-rate-ml-assessment/
├── README.md
├── requirements.txt
├── score.py
├── validation_predictions.csv
│
├── data/
│   ├── train-test.csv
│   └── december-chart-inputs.csv
│
├── models/
│   ├── catboost_freight_rate_final.cbm
│   ├── final_preprocessing_artifacts.pkl
│   └── catboost_final_parameters.json
│
├── predictions/
│   └── december-chart-inputs-scored.csv
│
├── figures/
│   └── candidate_december.png
│
├── reports/
│   └── Siam_Al_Qureshi_Spotter_ML_Assessment_Report.pdf
│
└── src/
    ├── pipeline.py
    ├── train.py
    ├── predict.py
    └── december_predict.py
```

> **Note:** The assessment-provided final validation input (`validation.csv`) is not included in this repository. The completed `validation_predictions.csv` contains the required predictions for all 12,000 assessment loads.

## Dataset

### Development Data

`data/train-test.csv` contains the labeled development data used for model development and validation.

The target variable is:

```text
posted_rate
```

Important input features include:

* Pickup location
* Delivery location
* Distance
* Equipment type
* Weight
* Date
* Market index
* Quote signal

### Final Validation Data

The assessment provides a separate `validation.csv` containing 12,000 loads requiring final predictions. This assessment input is not included in the repository.

The resulting predictions are provided in:

```text
validation_predictions.csv
```

with exactly:

```text
load_id,predicted_rate
```

## Validation Strategy

A chronological validation strategy was used to better represent real-world freight-rate prediction.

* **Training period:** January–August 2025
* **Validation period:** September–October 2025
* Training samples: **38,477**
* Validation samples: **9,523**

This approach avoids randomly mixing future observations into the training data and provides a more realistic estimate of performance on future loads.

## Data Quality and Preprocessing

The data-processing pipeline handles several potential data-quality issues:

* Missing numerical values
* Missing categorical values
* Negative weight values
* Date conversion and calendar feature extraction
* Categorical feature handling
* Geographic feature engineering

Weight values are converted to absolute values. Missing numerical values are imputed using statistics derived from the development training data.

Categorical variables such as pickup, delivery, and equipment are handled directly by CatBoost.

## Feature Engineering

Additional features were created to capture freight-rate relationships, including:

* Latitude/longitude differences
* Absolute geographic differences
* Haversine geographic distance
* Midpoint latitude and longitude
* Weight per mile
* Distance × market index
* Distance × quote signal
* Market index × quote signal
* Year
* Month
* Day of month
* Day of week
* Day of year
* Week of year

## Model Selection

CatBoost was selected because the dataset contains important categorical variables such as pickup, delivery, and equipment.

Hyperparameter experiments were performed for:

* Tree depth
* Learning rate
* L2 regularization
* Random strength
* Bagging temperature

The selected configuration was:

```text
Model: CatBoostRegressor
Objective: RMSE
Evaluation metric: MAE
Iterations: 2000
Learning rate: 0.05
Depth: 7
L2 leaf regularization: 3
Random strength: 1
Bagging temperature: 0
Random seed: 42
```

The best validation result achieved approximately:

```text
MAE: 117.02
RMSE: 633.97
```

## Final Predictions

The completed final prediction file is:

```text
validation_predictions.csv
```

It contains exactly 12,000 unique assessment load IDs and their predicted freight rates.

## December Prediction

The assessment also requires predictions for the fixed December scenario.

The December inputs use:

```text
Pickup: Lexington
Delivery: Fort Wayne
Distance: 360 miles
Equipment: Dry Van
Weight: 32,000 lb
Dates: December 1–31, 2025
```

The completed December predictions are stored in:

```text
predictions/december-chart-inputs-scored.csv
```

The provided scorer was successfully executed and validated:

```text
Validated 12,000 final predictions.
Validated 31 fixed December predictions.
Created chart: scorer_results/candidate_december.png
```

The December prediction chart is included with the project materials.

## Installation

Create and activate a Python virtual environment, then install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

## Running the Pipeline

### Train the model

```bash
python src/train.py
```

### Generate predictions

```bash
python src/predict.py
```

### Generate December predictions

```bash
python src/december_predict.py
```

### Validate the final outputs

```bash
python score.py --predictions validation_predictions.csv --december-predictions predictions/december-chart-inputs-scored.csv
```

The scorer validates:

* 12,000 final prediction rows
* Required `load_id` values
* Positive predicted rates
* 31 December dates
* Fixed December route and load characteristics

## Deliverables

The repository contains:

* Complete source code
* Python dependency specification
* Trained CatBoost model
* Preprocessing artifacts
* Final 12,000-row prediction file
* December predictions
* December prediction chart
* Assessment report

## Report

The assessment report is available at:

```text
reports/Siam_Al_Qureshi_Spotter_ML_Assessment_Report.pdf
```

It documents the data exploration, preprocessing, validation strategy, model development, results, and December prediction analysis.

## Author

**Siam Al Qureshi**

Machine Learning Engineer Assessment Submission
