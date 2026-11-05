# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 02 — Example 1: Price prediction with Feature Views
# MAGIC
# MAGIC Trains an XGBoost price model on 97 predictors and registers it with **Databricks Feature Views**, so `fe.score_batch` (and Model Serving) compute the time-series features automatically at inference time.
# MAGIC
# MAGIC * 67 predictors are provided at call time (from the base table).
# MAGIC * 30 predictors are computed on the fly by Feature Views: 15 rolling std-dev + 15 change values over 5/10/15-minute windows.
# MAGIC * Feature Views use `StddevPop` and `First` over a `SlidingWindow`; changes are derived as `current_req − first_in_window` inside a thin pyfunc wrapper.
# MAGIC
# MAGIC **Creates:** 30 Feature Views, the `ice_price_fv_model` UC model, and a pickle of the XGBoost model (used for SHAP in notebook 05).
# MAGIC
# MAGIC Prerequisite: **01 — Data Generation**.

# COMMAND ----------

# DBTITLE 1,Install libraries
# MAGIC %pip install -q databricks-feature-engineering xgboost mlflow cloudpickle
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load shared configuration
# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,Imports and prerequisite checks
import pickle
from datetime import timedelta

import mlflow
import mlflow.pyfunc
import mlflow.xgboost
import numpy as np
import pandas as pd
import xgboost as xgb
from mlflow import MlflowClient
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from databricks.feature_engineering import FeatureEngineeringClient
from databricks.feature_engineering.entities import (
    AggregationFunction,
    DeltaTableSource,
    Feature,
    First,
    SlidingWindow,
    StddevPop,
)

require_tables(HISTORY_TABLE, BASE_TABLE, hint="Run 01_data_generation first.")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(MLFLOW_EXPERIMENT)
print(f"Tables OK | model -> {FV_MODEL_NAME}")

# COMMAND ----------

# DBTITLE 1,Define Feature Views and build the 97-column training matrix
fe = FeatureEngineeringClient()
delta_source = DeltaTableSource(catalog_name=CATALOG, schema_name=SCHEMA, table_name=HISTORY_TABLE_NAME)

# For each source feature and window create two Feature View aggregations:
#   1. StddevPop — volatility over the window
#   2. First     — oldest value in the window (used to derive "change")
std_features, first_features = [], []
first_feature_names = []
for feature_name in TS_SOURCE_FEATURES:
    for window in WINDOWS:
        sliding = SlidingWindow(window_duration=timedelta(minutes=window), slide_duration=timedelta(minutes=1))
        std_features.append(Feature(
            name=f"{feature_name}_{window}_std",
            source=delta_source,
            entity=["instrument_id"],
            timeseries_column="event_ts",
            function=AggregationFunction(StddevPop(input=feature_name), sliding),
        ))
        first_features.append(Feature(
            name=f"{feature_name}_{window}_first",
            source=delta_source,
            entity=["instrument_id"],
            timeseries_column="event_ts",
            function=AggregationFunction(First(input=feature_name), sliding),
        ))
        first_feature_names.append(f"{feature_name}_{window}_first")

# 15 std + 15 first = 30 Feature View aggregations
requested_feature_views = std_features + first_features

def compute_change_features(df: DataFrame) -> DataFrame:
    """Derive change = current_req - first_in_window, then drop the intermediate first-value columns."""
    out = df
    for feature_name in TS_SOURCE_FEATURES:
        for window in WINDOWS:
            out = out.withColumn(
                f"{feature_name}_{window}_change",
                F.col(TS_REQ_MAP[feature_name]) - F.col(f"{feature_name}_{window}_first"),
            )
    return out.drop(*first_feature_names)

def build_feature_matrix(input_df: DataFrame, label_col=None) -> DataFrame:
    """Point-in-time Feature View join, then derive the change columns."""
    training_set = fe.create_training_set(df=input_df, features=requested_feature_views, label=label_col)
    return compute_change_features(training_set.load_df())

train_matrix_sdf = build_feature_matrix(spark.table(BASE_TABLE), label_col="price")
display(train_matrix_sdf.select(["time_id", "instrument_id", "price"] + MODEL_FEATURES).limit(5))

# COMMAND ----------

# DBTITLE 1,Train XGBoost and log to MLflow
# Time-ordered split: first 80% for training, last 20% for evaluation
train_pd = (
    train_matrix_sdf.select(["time_id", "instrument_id", "event_ts", "price"] + MODEL_FEATURES)
    .orderBy("event_ts").toPandas().dropna().reset_index(drop=True)
)
split_idx = int(len(train_pd) * 0.80)
train_split, test_split = train_pd.iloc[:split_idx].copy(), train_pd.iloc[split_idx:].copy()
X_train, y_train = train_split[MODEL_FEATURES], train_split["price"]
X_test, y_test = test_split[MODEL_FEATURES], test_split["price"]

model = xgb.XGBRegressor(
    n_estimators=350,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.85,
    colsample_bytree=0.85,
    objective="reg:squarederror",
    tree_method="hist",
    random_state=SEED,
)
model.fit(X_train, y_train)

test_pred = model.predict(X_test)
rmse = float(np.sqrt(np.mean((y_test - test_pred) ** 2)))
r2 = float(1 - np.sum((y_test - test_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2))

# Pickle the raw XGBoost model (used by the SHAP step in notebook 05)
with open(EX1_PICKLE_PATH, "wb") as f:
    pickle.dump({"model": model, "feature_columns": MODEL_FEATURES, "example": "example_1_feature_views"}, f)

with mlflow.start_run(run_name="ice_example1_feature_views_xgboost"):
    mlflow.log_params({
        "predictor_count": len(MODEL_FEATURES),
        "base_predictor_count": len(BASE_FEATURES),
        "timeseries_predictor_count": len(TS_FEATURES),
        "training_rows": len(train_split),
        "test_rows": len(test_split),
        "history_table": HISTORY_TABLE,
        "base_table": BASE_TABLE,
    })
    mlflow.log_metrics({"rmse": rmse, "r2": r2})
    mlflow.xgboost.log_model(model, artifact_path="model")
    mlflow.log_artifact(EX1_PICKLE_PATH, artifact_path="pickle")

print({"rmse": round(rmse, 4), "r2": round(r2, 4), "train_rows": len(train_split), "pickle": EX1_PICKLE_PATH})

# COMMAND ----------

# DBTITLE 1,Register Feature Views in Unity Catalog
# Register each Feature View (required for fe.log_model lineage).
# If it already exists from a previous run, reuse it.
registered_features = []
for feat in requested_feature_views:
    try:
        registered = fe.register_feature(feature=feat, catalog_name=CATALOG, schema_name=SCHEMA)
    except Exception:
        registered = fe.get_feature(full_name=f"{CATALOG}.{SCHEMA}.{feat.name}")
    registered_features.append(registered)

print(f"{len(registered_features)} Feature Views registered in {CATALOG}.{SCHEMA}")

# COMMAND ----------

# DBTITLE 1,Log model with Feature View lineage and register in UC
# ---------------------------------------------------------------------------
# Pyfunc wrapper: receives (67 base + 15 std + 15 first) from score_batch,
# derives the 15 change features, then runs XGBoost predict.
# ---------------------------------------------------------------------------
class ICEPriceWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, xgb_model, model_features, ts_req_map, windows):
        self.xgb_model = xgb_model
        self.model_features = model_features
        self.ts_req_map = ts_req_map
        self.windows = windows

    def predict(self, context, model_input, params=None):
        df = model_input.copy()
        for feature_name, req_col in self.ts_req_map.items():
            for window in self.windows:
                first_col = f"{feature_name}_{window}_first"
                if first_col in df.columns and req_col in df.columns:
                    df[f"{feature_name}_{window}_change"] = df[req_col] - df[first_col]
        return self.xgb_model.predict(df[self.model_features])

wrapper = ICEPriceWrapper(xgb_model=model, model_features=MODEL_FEATURES, ts_req_map=TS_REQ_MAP, windows=WINDOWS)

# Small training set — only the schema and Feature View lineage matter here
training_set = fe.create_training_set(
    df=spark.table(BASE_TABLE).limit(100),
    features=registered_features,
    label="price",
)

with mlflow.start_run(run_name="ice_example1_fv_model"):
    fe.log_model(
        model=wrapper,
        artifact_path="model",
        flavor=mlflow.pyfunc,
        training_set=training_set,
        registered_model_name=FV_MODEL_NAME,
    )

print(f"Model registered: {FV_MODEL_NAME}  |  latest version {latest_model_version(FV_MODEL_NAME)}")