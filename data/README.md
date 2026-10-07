
# Data Directory

This directory contains the datasets required for training, validating, and scoring the Spotter Freight Rate ML model.

## 📁 Required Data Files

| File Name | Rows | Columns | Target / Key | Description |
|---|---|---|---|---|
| `train_test.csv` | 48,000 | 14 | `posted_rate` | Primary labeled development dataset spanning Jan 1 – Oct 31, 2025. Used for model training and internal validation splits. |
| `validation.csv` | 12,000 | 13 | `load_id` | Final unlabeled prediction dataset spanning Nov 1 – Dec 31, 2025. Requires predicted freight rates. |
| `validation_predictions_template.csv` | 12,000 | 2 | `load_id`, `predicted_rate` | Provided template specifying the expected submission schema for `validation_predictions.csv`. |
| `december_chart_inputs.csv` | 31 | 7 | `load_id` | Scenario dataset representing 31 daily loads for a fixed Lexington → Fort Wayne lane in December 2025. |

---

## 📊 Data Schema & Features

- **`load_id`**: Unique identifier for each freight load.
- **`date`**: Load posting date (Format: `YYYY-MM-DD`).
- **`pickup` / `delivery`**: Origin and destination city names.
- **`pickup_lat` / `pickup_lon`**: Geographic coordinates of origin.
- **`delivery_lat` / `delivery_lon`**: Geographic coordinates of destination.
- **`equipment`**: Trailer equipment type (`Dry Van`, `Reefer`, `Flatbed`).
- **`distance`**: Total operational road distance in miles.
- **`weight`**: Total freight payload weight in pounds.
- **`market_index`**: Macro market capacity and demand index.
- **`quote_signal`**: Real-time pricing signal for the lane.
- **`posted_rate`**: **(Target)** Final posted freight rate in USD.

---

## 🧹 Data Quality & Preprocessing Rules

As documented in the analysis report, raw data undergoes leakage-aware preprocessing:
1. **Negative Weights**: Negative payload values are converted to absolute values.
2. **Missing Value Imputation**:
   - Missing `weight` values are imputed using equipment-level medians, falling back to overall training medians.
   - Missing `market_index` values follow a fallback sequence: Date-level median $\rightarrow$ Equipment-level median $\rightarrow$ Global median.
3. **Out-of-Vocabulary Cities**: Unseen origin/destination locations in validation data are handled seamlessly via latitude/longitude spatial calculations (Haversine distance, coordinate differences)[cite: 6].

> **Note**: Raw input files inside `data/` should be treated as read-only. Generated model outputs and predictions belong in `predictions/` or the project root.
