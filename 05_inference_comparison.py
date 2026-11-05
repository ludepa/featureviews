# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 05 — Inference comparison: Example 1 (Feature Views) vs Example 2 (Delta features)
# MAGIC
# MAGIC * **Performance:** end-to-end `fe.score_batch` time for the same 10k rows with each registered model — model loading, feature computation or lookup, and `predict()`.
# MAGIC * **SHAP values:** per-prediction feature attributions for Example 1, saved to the `ice_example1_shap` Delta table (one row per prediction, one column per feature).
# MAGIC
# MAGIC Prerequisites: **02 — Example 1** and **04 — Example 2**.

# COMMAND ----------

# DBTITLE 1,Install libraries
# MAGIC %pip install -q databricks-feature-engineering xgboost shap mlflow cloudpickle
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load shared configuration
# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,Performance comparison — score_batch latency (10k rows)
import time

import matplotlib.pyplot as plt
import mlflow
from pyspark.sql import functions as F
from databricks.feature_engineering import FeatureEngineeringClient

mlflow.set_registry_uri("databricks-uc")
fe = FeatureEngineeringClient()

# Score the latest registered version of each model
P1_VERSION = latest_model_version(FV_MODEL_NAME)
P2_VERSION = latest_model_version(DELTA_MODEL_NAME)
print(f"Scoring {FV_MODEL_NAME} v{P1_VERSION} and {DELTA_MODEL_NAME} v{P2_VERSION}")

# Same 10k most recent rows for both approaches (label dropped)
scoring_df = spark.table(BASE_TABLE).orderBy(F.desc("event_ts")).limit(10_000).drop("price")

def timed_score(model_name, version):
    """Time score_batch end to end.

    score_batch is lazy, and Spark skips the model if the action never reads
    `prediction` (e.g. count()). Summing the prediction column forces feature
    computation/lookup AND predict() to run for every row.
    """
    t0 = time.perf_counter()
    preds = fe.score_batch(model_uri=f"models:/{model_name}/{version}", df=scoring_df)
    row = preds.agg(F.count("prediction").alias("n"), F.sum("prediction").alias("total")).first()
    return time.perf_counter() - t0, row["n"]

# Example 1: Feature Views compute 30 SlidingWindow aggregations from the history table
p1_seconds, p1_rows = timed_score(FV_MODEL_NAME, P1_VERSION)
# Example 2: key-based lookup against the pre-computed materialized view
p2_seconds, p2_rows = timed_score(DELTA_MODEL_NAME, P2_VERSION)

print(f"Example 1 (Feature Views):  {p1_seconds:6.1f}s  ({p1_rows:,} predictions)")
print(f"Example 2 (Delta Lookup):   {p2_seconds:6.1f}s  ({p2_rows:,} predictions)")
print(f"Speedup (Ex1 / Ex2):        {p1_seconds / p2_seconds:6.2f}x")

fig, ax = plt.subplots(figsize=(6, 3.5))
bars = ax.bar(["Example 1\n(Feature Views)", "Example 2\n(Delta Lookup)"], [p1_seconds, p2_seconds],
              color=["steelblue", "darkorange"], width=0.5)
for bar, val in zip(bars, [p1_seconds, p2_seconds]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}s", ha="center", va="bottom", fontweight="bold")
ax.set_ylabel("Seconds")
ax.set_title("score_batch end-to-end latency — 10k rows")
plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Build the Example 1 feature matrix for SHAP
from datetime import timedelta
from databricks.feature_engineering.entities import (
    AggregationFunction, DeltaTableSource, Feature, First, SlidingWindow, StddevPop,
)

# Same Feature View definitions as Example 1 (not re-registered here)
delta_source = DeltaTableSource(catalog_name=CATALOG, schema_name=SCHEMA, table_name=HISTORY_TABLE_NAME)
feature_views, first_names = [], []
for feat in TS_SOURCE_FEATURES:
    for w in WINDOWS:
        sliding = SlidingWindow(window_duration=timedelta(minutes=w), slide_duration=timedelta(minutes=1))
        feature_views.append(Feature(name=f"{feat}_{w}_std", source=delta_source, entity=["instrument_id"],
                                     timeseries_column="event_ts", function=AggregationFunction(StddevPop(input=feat), sliding)))
        feature_views.append(Feature(name=f"{feat}_{w}_first", source=delta_source, entity=["instrument_id"],
                                     timeseries_column="event_ts", function=AggregationFunction(First(input=feat), sliding)))
        first_names.append(f"{feat}_{w}_first")

fv_df = fe.create_training_set(df=scoring_df, features=feature_views, label=None).load_df()
for feat in TS_SOURCE_FEATURES:
    for w in WINDOWS:
        fv_df = fv_df.withColumn(f"{feat}_{w}_change", F.col(TS_REQ_MAP[feat]) - F.col(f"{feat}_{w}_first"))

p1_pd = (
    fv_df.drop(*first_names)
    .select(["time_id", "instrument_id", "event_ts"] + MODEL_FEATURES)
    .toPandas().dropna().reset_index(drop=True)
)
print(f"Example 1 feature matrix: {len(p1_pd):,} rows x {len(MODEL_FEATURES)} features")

# COMMAND ----------

# DBTITLE 1,Compute Example 1 SHAP values and save to Delta
import pickle

import numpy as np
import pandas as pd
import shap

# TreeExplainer gives exact SHAP values for the XGBoost model trained in Example 1
with open(EX1_PICKLE_PATH, "rb") as f:
    p1_xgb = pickle.load(f)["model"]

explainer = shap.TreeExplainer(p1_xgb)
shap_values = explainer.shap_values(p1_pd[MODEL_FEATURES])
base_value = float(explainer.expected_value)

# Wide format: one row per prediction, one column per feature.
# base_value + sum of the 97 feature columns = model prediction.
shap_pd = pd.concat(
    [
        p1_pd[["time_id", "instrument_id", "event_ts"]].reset_index(drop=True),
        pd.DataFrame({"base_value": np.full(len(p1_pd), base_value)}),
        pd.DataFrame(shap_values.astype("float64"), columns=MODEL_FEATURES),
    ],
    axis=1,
)

(spark.createDataFrame(shap_pd)
    .write.mode("overwrite").option("overwriteSchema", "true")
    .format("delta").saveAsTable(SHAP_TABLE))

# Sanity check: SHAP values + base value reproduce the model's predictions
max_gap = float(np.max(np.abs(shap_values.sum(axis=1) + base_value - p1_xgb.predict(p1_pd[MODEL_FEATURES]))))
print(f"Saved {len(shap_pd):,} predictions x {len(MODEL_FEATURES)} SHAP columns to {SHAP_TABLE}")
print(f"Base value: {base_value:.4f} | max |SHAP sum - prediction|: {max_gap:.2e}")
display(spark.table(SHAP_TABLE).limit(10))