# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 04 — Example 2: Price prediction with pre-computed Delta features
# MAGIC
# MAGIC Same model as Example 1, but the 30 time-series predictors come **pre-computed** from the `ice_delta_features` materialized view (built by the pipeline in notebook 03) instead of being computed by Feature Views at inference time.
# MAGIC
# MAGIC The model is registered with two `FeatureLookup`s keyed on `(instrument_id, event_ts)`:
# MAGIC
# MAGIC * 67 base features from the base table.
# MAGIC * 30 time-series features from the materialized view.
# MAGIC
# MAGIC At scoring time `fe.score_batch` only performs key-based joins — no window computation.
# MAGIC
# MAGIC **Creates:** the `ice_price_delta_model` UC model and a primary key on the base table.
# MAGIC
# MAGIC Prerequisites: **01 — Data Generation** and **03 — Feature pipeline**.

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

import mlflow
import mlflow.xgboost
import numpy as np
import xgboost as xgb
from pyspark.sql import functions as F

from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

require_tables(HISTORY_TABLE, BASE_TABLE, DELTA_FEATURE_TABLE,
               hint="Run 01_data_generation and 03_example2_feature_pipeline first.")
mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(MLFLOW_EXPERIMENT)
print(f"Tables OK | model -> {DELTA_MODEL_NAME}")

# COMMAND ----------

# DBTITLE 1,Join pre-computed features to the base table
# The 30 time-series features are already computed by the pipeline:
#   {feature}_{window}_std    : stddev_pop over the preceding N rows
#   {feature}_{window}_change : current value minus lag(N)
training_sdf = (
    spark.table(BASE_TABLE).alias("b")
    .join(spark.table(DELTA_FEATURE_TABLE).alias("f"), on=["instrument_id", "time_id", "event_ts"], how="inner")
    .dropna()
)

display(training_sdf.select(["time_id", "instrument_id", "price"] + MODEL_FEATURES).limit(5))

# COMMAND ----------

# DBTITLE 1,Train XGBoost and log to MLflow
# Time-ordered split: first 80% for training, last 20% for evaluation
train_pd = (
    training_sdf.select(["time_id", "instrument_id", "event_ts", "price"] + MODEL_FEATURES)
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

with open(EX2_PICKLE_PATH, "wb") as f:
    pickle.dump({"model": model, "feature_columns": MODEL_FEATURES, "example": "example_2_delta_features"}, f)

with mlflow.start_run(run_name="ice_example2_delta_features_xgboost"):
    mlflow.log_params({
        "predictor_count": len(MODEL_FEATURES),
        "base_predictor_count": len(BASE_FEATURES),
        "timeseries_predictor_count": len(TS_FEATURES),
        "training_rows": len(train_split),
        "test_rows": len(test_split),
        "feature_table": DELTA_FEATURE_TABLE,
    })
    mlflow.log_metrics({"rmse": rmse, "r2": r2})
    mlflow.xgboost.log_model(model, artifact_path="model")
    mlflow.log_artifact(EX2_PICKLE_PATH, artifact_path="pickle")

print({"rmse": round(rmse, 4), "r2": round(r2, 4), "train_rows": len(train_split), "pickle": EX2_PICKLE_PATH})

# COMMAND ----------

# DBTITLE 1,Log model with FeatureLookups and register in UC
fe = FeatureEngineeringClient()

# FeatureLookup requires a primary key on each source table. The materialized
# view declares one in the pipeline; add one to the base table (idempotent).
for col in ["instrument_id", "event_ts"]:
    spark.sql(f"ALTER TABLE {BASE_TABLE} ALTER COLUMN {col} SET NOT NULL")
try:
    spark.sql(f"ALTER TABLE {BASE_TABLE} ADD CONSTRAINT {BASE_TABLE_NAME}_pk PRIMARY KEY (instrument_id, event_ts)")
except Exception as e:
    if "already exists" not in str(e).lower():
        raise

feature_lookups = [
    FeatureLookup(table_name=BASE_TABLE, feature_names=BASE_FEATURES, lookup_key=["instrument_id", "event_ts"]),
    FeatureLookup(table_name=DELTA_FEATURE_TABLE, feature_names=TS_FEATURES, lookup_key=["instrument_id", "event_ts"]),
]

# Small training set — only the schema and lookup lineage matter here
training_set = fe.create_training_set(
    df=spark.table(BASE_TABLE).select("instrument_id", "event_ts", "price").limit(100),
    feature_lookups=feature_lookups,
    label="price",
    exclude_columns=["instrument_id", "event_ts"],
)

# No wrapper needed: the view already has the final _std and _change columns
with mlflow.start_run(run_name="ice_example2_delta_model"):
    fe.log_model(
        model=model,
        artifact_path="model",
        flavor=mlflow.xgboost,
        training_set=training_set,
        registered_model_name=DELTA_MODEL_NAME,
    )

print(f"Model registered: {DELTA_MODEL_NAME}  |  latest version {latest_model_version(DELTA_MODEL_NAME)}")