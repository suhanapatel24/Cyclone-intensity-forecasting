import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

NUMERIC = ["wind_kt", "wind_change_6h", "lat", "lon", "pressure_mb", "storm_speed"]
FEATURES = NUMERIC + ["basin"]

def load_ibtracs(path):
    cols = ["SID", "ISO_TIME", "BASIN", "LAT", "LON", "USA_WIND", "WMO_WIND",
            "USA_PRES", "WMO_PRES", "STORM_SPEED"]
    df = pd.read_csv(path, skiprows=[1], usecols=lambda c: c in cols,
                     low_memory=False, na_values=[" ", ""])
    required = {"SID", "ISO_TIME", "BASIN", "LAT", "LON"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing IBTrACS columns: {sorted(required-set(df.columns))}")
    df = df.rename(columns={"SID":"sid","ISO_TIME":"time","BASIN":"basin",
                            "LAT":"lat","LON":"lon","STORM_SPEED":"storm_speed"})
    for col in ["lat","lon","storm_speed","USA_WIND","WMO_WIND","USA_PRES","WMO_PRES"]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["time"] = pd.to_datetime(df["time"], utc=True, errors="coerce")
    def coalesce(first, second):
        a = df[first] if first in df else pd.Series(np.nan, index=df.index)
        b = df[second] if second in df else pd.Series(np.nan, index=df.index)
        return a.fillna(b)
    df["wind_kt"] = coalesce("USA_WIND","WMO_WIND")
    df["pressure_mb"] = coalesce("USA_PRES","WMO_PRES")
    if "storm_speed" not in df:
        df["storm_speed"] = np.nan
    return df[["sid","time","basin","lat","lon","wind_kt","pressure_mb","storm_speed"]]

def make_examples(raw):
    df = raw.dropna(subset=["sid","time","wind_kt"]).copy()
    df = df.sort_values(["sid","time"]).drop_duplicates(["sid","time"])
    groups = df.groupby("sid", sort=False)
    prev_time = groups["time"].shift(1)
    prev_wind = groups["wind_kt"].shift(1)
    delta = (df["time"] - prev_time).dt.total_seconds() / 3600
    df["wind_change_6h"] = (df["wind_kt"] - prev_wind).where(delta == 6)
    next_time = groups["time"].shift(-1)
    next_wind = groups["wind_kt"].shift(-1)
    hours_ahead = (next_time - df["time"]).dt.total_seconds() / 3600
    df["target_wind_6h"] = next_wind.where(hours_ahead == 6)
    df["basin"] = df["basin"].fillna("Unknown").astype(str)
    return df.dropna(subset=["target_wind_6h"]).reset_index(drop=True)

def make_model():
    transformer = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC),
        ("basin", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore"))
        ]), ["basin"])
    ])
    return Pipeline([
        ("features", transformer),
        ("forest", RandomForestRegressor(n_estimators=150, min_samples_leaf=2,
                                         random_state=42, n_jobs=-1))
    ])

def run(input_path, output_dir):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    examples = make_examples(load_ibtracs(input_path))
    if len(examples) < 100 or examples["sid"].nunique() < 10:
        raise ValueError("Need >=100 valid 6-hour examples from >=10 storms")
    split = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr, te = next(split.split(examples, groups=examples["sid"]))
    train, test = examples.iloc[tr], examples.iloc[te]
    model = make_model()
    model.fit(train[FEATURES], train["target_wind_6h"])
    predicted = model.predict(test[FEATURES])
    observed = test["target_wind_6h"].to_numpy()
    metrics = {
        "total_examples": int(len(examples)),
        "unique_storms": int(examples["sid"].nunique()),
        "test_storms": int(test["sid"].nunique()),
        "horizon_hours": 6,
        "mae_knots": float(mean_absolute_error(observed,predicted)),
        "rmse_knots": float(np.sqrt(mean_squared_error(observed,predicted))),
        "r2": float(r2_score(observed,predicted)),
        "persistence_baseline_mae_knots": float(mean_absolute_error(observed,test["wind_kt"]))
    }
    results = test[["sid","time","basin","wind_kt","target_wind_6h"]].copy()
    results["prediction_kt"] = predicted
    results["absolute_error_kt"] = np.abs(predicted-observed)
    metrics["mae_by_basin_knots"] = {
        str(k): float(v) for k,v in results.groupby("basin")["absolute_error_kt"].mean().items()
    }
    results.to_csv(output/"holdout_predictions.csv",index=False)
    importance = model.named_steps["forest"].feature_importances_
    names = model.named_steps["features"].get_feature_names_out()
    pd.DataFrame({"feature":names,"importance":importance}).sort_values(
        "importance",ascending=False).to_csv(output/"feature_importance.csv",index=False)
    joblib.dump(model,output/"random_forest_6h.joblib")
    (output/"metrics.json").write_text(json.dumps(metrics,indent=2))
    print(json.dumps(metrics,indent=2))
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",required=True)
    parser.add_argument("--output",default="outputs")
    args = parser.parse_args()
    run(args.input,args.output)

if __name__ == "__main__":
    main()
