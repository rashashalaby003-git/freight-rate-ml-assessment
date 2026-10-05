
"""Freight rate prediction: validate, retrain on all labeled data, predict.

Usage:
    python train_predict.py

Outputs:
    validation_predictions.csv   (load_id, predicted_rate)  -> 12,000 rows
    december_predictions.csv     (december_chart_inputs.csv with predicted_rate filled)
    metrics.json                 (time-split validation metrics, for the report)
"""
import argparse
import json

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 42
TARGET = "posted_rate"
CAT_COLS = ["pickup", "delivery", "equipment"]
MEDIAN_COLS = ["weight"]
SPLIT_DATE = "2025-09-01"  # time-based split: train before, validate from this date
# Not used as features: december_chart_inputs.csv does not contain them, so the model must work without them
DROP_FEATURES = ["market_index", "quote_signal"]


def load(path):
    return pd.read_csv(path, parse_dates=["date"])


def prep(df):
    """Row-wise cleaning + date features (no statistics learned here -> no leakage)."""
    df = df.copy()
    df.loc[df["weight"] < 0, "weight"] = np.nan  # negative weight is invalid
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.dayofweek
    return df


def fill_medians(df, medians):
    df = df.copy()
    for col, val in medians.items():
        df[col] = df[col].fillna(val)
    return df


def to_features(df):
    """Drop non-features and make categoricals safe strings for CatBoost."""
    drop = ["load_id", "date", TARGET, "predicted_rate"] + DROP_FEATURES
    X = df.drop(columns=[c for c in drop if c in df.columns])
    for c in CAT_COLS:
        X[c] = X[c].astype(str).fillna("missing")
    return X


def city_coords(dev):
    """city -> lat/lon lookup built from the labeled data (coordinates are fixed per city)."""
    p = dev[["pickup", "pickup_lat", "pickup_lon"]].drop_duplicates("pickup")
    d = dev[["delivery", "delivery_lat", "delivery_lon"]].drop_duplicates("delivery")
    p.columns = d.columns = ["city", "lat", "lon"]
    return pd.concat([p, d]).drop_duplicates("city").set_index("city")


def complete_december(dec, dev):
    """december_chart_inputs.csv lacks lat/lon -> look them up by city name from the labeled data."""
    dec = dec.copy()
    coords = city_coords(dev)
    for side in ["pickup", "delivery"]:
        for k in ["lat", "lon"]:
            col = f"{side}_{k}"
            if col not in dec.columns:
                dec[col] = dec[side].map(coords[k])
                if dec[col].isna().any():
                    print(f"WARNING: {col} missing for some December cities")
    return dec


def make_model(**kw):
    return CatBoostRegressor(
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        random_seed=SEED,
        cat_features=CAT_COLS,
        verbose=200,
        **kw,
    )


def main(args):
    dev = prep(load(args.train))

    # ---- 1) Time-based validation -------------------------------------
    train_df = dev[dev["date"] < SPLIT_DATE].copy()
    val_df = dev[dev["date"] >= SPLIT_DATE].copy()
    print(f"Train: {train_df.shape}  {train_df['date'].min().date()} -> {train_df['date'].max().date()}")
    print(f"Valid: {val_df.shape}  {val_df['date'].min().date()} -> {val_df['date'].max().date()}")

    medians = train_df[MEDIAN_COLS].median()  # learned from train only
    train_df = fill_medians(train_df, medians)
    val_df = fill_medians(val_df, medians)

    X_tr, y_tr = to_features(train_df), train_df[TARGET]
    X_va, y_va = to_features(val_df), val_df[TARGET]

    model = make_model(iterations=3000, early_stopping_rounds=100)
    # predict rate per mile, then multiply by distance to get the full rate
    model.fit(X_tr, y_tr / X_tr["distance"], eval_set=(X_va, y_va / X_va["distance"]))

    pred = np.clip(model.predict(X_va) * X_va["distance"], 0, None)
    mae = float(mean_absolute_error(y_va, pred))
    rmse = float(np.sqrt(mean_squared_error(y_va, pred)))
    best_iter = int(model.get_best_iteration())
    print(f"Validation MAE={mae:.2f}  RMSE={rmse:.2f}  best_iter={best_iter}")

    with open("metrics.json", "w") as f:
        json.dump(
            {
                "split_date": SPLIT_DATE,
                "dropped_features": DROP_FEATURES,
                "n_train": len(train_df),
                "n_valid": len(val_df),
                "mae": mae,
                "rmse": rmse,
                "best_iteration": best_iter,
            },
            f,
            indent=2,
        )

    # ---- 2) Final model on ALL labeled data ---------------------------
    medians_full = dev[MEDIAN_COLS].median()
    dev_full = fill_medians(dev, medians_full)
    final = make_model(iterations=int((best_iter + 1) * 1.1))  # a bit more data -> a bit more trees
    X_full = to_features(dev_full)
    final.fit(X_full, dev_full[TARGET] / X_full["distance"])
    median_distance = float(X_full["distance"].median())

    def predict_file(path, complete=None):
        df = load(path)
        work = complete(df) if complete else df
        X = to_features(fill_medians(prep(work), medians_full))
        X = X[final.feature_names_]  # same column order as training
        bad = X["distance"].isna() | (X["distance"] <= 0)
        if bad.any():
            print(f"WARNING: {int(bad.sum())} rows in {path} have invalid distance; using median distance")
            X.loc[bad, "distance"] = median_distance
        return df, np.clip(final.predict(X) * X["distance"], 0, None)

    # ---- 3) validation.csv -> validation_predictions.csv --------------
    val_in, val_pred = predict_file(args.validation)
    out = pd.read_csv(args.template)[["load_id"]].merge(
        pd.DataFrame({"load_id": val_in["load_id"], "predicted_rate": val_pred}),
        on="load_id",
        how="left",
    )
    assert len(out) == 12000, f"expected 12000 rows, got {len(out)}"
    assert out["predicted_rate"].notna().all(), "missing predictions"
    out.to_csv("validation_predictions.csv", index=False)
    print("Saved validation_predictions.csv", out.shape)

    # ---- 4) december_chart_inputs.csv -> december_predictions.csv -----
    _, dec_pred = predict_file(args.december, lambda d: complete_december(d, dev))
    dec_out = pd.read_csv(args.december)  # keep the original columns/format untouched
    dec_out["predicted_rate"] = dec_pred
    dec_out.to_csv("december_predictions.csv", index=False)
    print("Saved december_predictions.csv", dec_out.shape)
    print(dec_out["predicted_rate"].describe().round(2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/train_test.csv")
    p.add_argument("--validation", default="data/validation.csv")
    p.add_argument("--template", default="data/validation_predictions_template.csv")
    p.add_argument("--december", default="data/december_chart_inputs.csv")
    main(p.parse_args())