# Freight Rate Prediction Challenge

A machine learning solution for predicting `posted_rate` for freight loads using CatBoost regression.

## Project Overview

The goal of this project is to predict freight rates for new shipment loads based on shipment characteristics such as:

- Pickup and delivery locations
- Distance
- Equipment type
- Weight
- Date-related features
- Location coordinates

The final solution uses a time-based validation strategy and a CatBoost regression model.

## How to Run

Install the required dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the training and prediction pipeline:

```bash
python train_predict.py
```

Run the provided assessment scorer:

```bash
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

The random seed is fixed at `42` for reproducibility.

## Input Data

The training and prediction pipeline expects the assessment data files inside the `data/` directory:

- `train_test.csv`
- `validation.csv`
- `december_chart_inputs.csv`

The assessment data files are not included in this repository.

## Data Preprocessing

Several data quality issues were handled during preprocessing.

### Negative weights

Negative values in `weight` were treated as invalid and converted to missing values.

Missing weight values were then imputed using the median calculated from the training data only.

### Date features

The shipment date was transformed into:

- `month`
- `day`
- `day_of_week`

### Categorical features

The following features were handled as categorical variables using CatBoost:

- `pickup`
- `delivery`
- `equipment`

## Validation Strategy

A chronological split was used instead of a random train/validation split.

### Training set

**January 1, 2025 – August 31, 2025**

38,477 records

### Validation set

**September 1, 2025 – October 31, 2025**

9,523 records

A time-based split was chosen because the real task is to predict future freight rates. This approach better simulates the forecasting scenario and avoids training on future observations.

## Feature Selection

The final model uses features that are available when making the required predictions:

- `pickup`
- `delivery`
- `equipment`
- `distance`
- `weight`
- pickup coordinates
- delivery coordinates
- `month`
- `day`
- `day_of_week`

`market_index` and `quote_signal` were not used in the final model because they are not available in the December prediction input.

This ensures that the final model does not depend on information that would be unavailable at prediction time.

## Modeling Approach

Several regression approaches were explored during development, including tree-based regression, Ridge Regression, XGBoost, and CatBoost.

The final model uses `CatBoostRegressor`.

Instead of predicting the total freight rate directly, the model predicts the rate per mile:

```text
rate_per_mile = posted_rate / distance
```

The predicted rate per mile is then converted back to the final freight rate:

```text
predicted_rate = predicted_rate_per_mile × distance
```

The number of model iterations was selected using early stopping on the chronological validation period.

After model selection, the final model was retrained using all available labeled development data before generating the final predictions.

## Validation Results

The final model achieved the following results on the internal chronological validation set:

| Validation Period      |    MAE |   RMSE |
| ---------------------- | -----: | -----: |
| September–October 2025 | 103.40 | 629.72 |

MAE measures the average absolute prediction error, while RMSE gives greater weight to large errors.

The difference between the two metrics is mainly influenced by a small number of loads with unusually high freight rates.

These are internal validation results. The official assessment metrics are calculated by the assessment scoring system after submission.

## Final Outputs

The training and prediction pipeline generates:

### Validation predictions

`validation_predictions.csv`

Contains predictions for all **12,000 validation loads** with the required columns:

```text
load_id,predicted_rate
```

### December predictions

`december_predictions.csv`

Contains the **31 December input records** with their predicted rates.

### Validation metrics

`metrics.json`

Contains the internal time-based validation metrics.

### December chart

The provided scoring script generates:

```text
scorer_results/candidate_december.png
```

## Output Validation

The final prediction files were checked for:

- Correct row counts
- Correct columns
- Missing values
- Duplicate IDs
- Missing IDs
- Extra IDs
- Negative predictions
- Infinite predictions

All validation prediction checks passed.

The December prediction file was also checked to ensure:

- 31 records were present
- All predictions were valid
- No predictions were missing
- No negative or infinite predictions existed
- Original input fields were preserved

The provided scoring script successfully validated:

- **12,000 final validation predictions**
- **31 December predictions**

and generated the required December chart.

## Project Structure

```text
freight-rate/
│
├── train_predict.py
├── score.py
├── eda.ipynb
├── requirements.txt
├── README.md
│
├── validation_predictions.csv
├── december_predictions.csv
├── metrics.json
│
└── scorer_results/
    └── candidate_december.png
```

## Reproducibility

The pipeline uses a fixed random seed of `42`.

The complete workflow can be reproduced by installing the dependencies, running `train_predict.py`, and then running the provided scoring script.
