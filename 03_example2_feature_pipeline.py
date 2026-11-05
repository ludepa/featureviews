# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 03 — Example 2: Create and run the feature pipeline
# MAGIC
# MAGIC Example 2 pre-computes the 30 time-series features with a **Lakeflow Spark Declarative Pipeline** instead of computing them at inference time. The pipeline source is in `pipeline/ice_delta_features.py` (next to this notebook) and publishes a materialized view, `ice_delta_features`, with a primary key on `(instrument_id, event_ts)` so it can be used in a `FeatureLookup`.
# MAGIC
# MAGIC This notebook:
# MAGIC
# MAGIC 1. Creates the serverless pipeline (or updates it if it already exists) targeting the catalog/schema from `00_config`.
# MAGIC 2. Runs a pipeline update and waits for it to finish.
# MAGIC 3. Checks that the materialized view and its primary key exist.
# MAGIC
# MAGIC In production you'd schedule the pipeline (for example, hourly) as new market data arrives.
# MAGIC
# MAGIC Prerequisite: **01 — Data Generation**.

# COMMAND ----------

# DBTITLE 1,Install libraries
# MAGIC %pip install -q --upgrade databricks-sdk
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load shared configuration
# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,Create or update the pipeline
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.pipelines import FileLibrary, PipelineLibrary

require_tables(HISTORY_TABLE, hint="Run 01_data_generation first.")

w = WorkspaceClient()

# The pipeline source lives in ./pipeline/ next to this notebook
notebook_path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
PIPELINE_SOURCE = f"{os.path.dirname(notebook_path)}/pipeline/ice_delta_features.py"
w.workspace.get_status(PIPELINE_SOURCE)   # fails fast if the file is missing

pipeline_spec = dict(
    name=PIPELINE_NAME,
    catalog=CATALOG,
    schema=SCHEMA,
    serverless=True,
    continuous=False,
    development=False,
    libraries=[PipelineLibrary(file=FileLibrary(path=PIPELINE_SOURCE))],
    configuration={
        "ice.history_table": HISTORY_TABLE,
        "ice.feature_table": DELTA_FEATURE_TABLE_NAME,
    },
)

existing = list(w.pipelines.list_pipelines(filter=f"name LIKE '{PIPELINE_NAME}'"))
if existing:
    pipeline_id = existing[0].pipeline_id
    w.pipelines.update(pipeline_id=pipeline_id, **pipeline_spec)
    print(f"Updated pipeline {PIPELINE_NAME} ({pipeline_id})")
else:
    pipeline_id = w.pipelines.create(**pipeline_spec).pipeline_id
    print(f"Created pipeline {PIPELINE_NAME} ({pipeline_id})")

print(f"Source: {PIPELINE_SOURCE}")
print(f"Pipeline UI: {w.config.host}/pipelines/{pipeline_id}")

# COMMAND ----------

# DBTITLE 1,Run a pipeline update and wait for completion
update_id = w.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=False).update_id
print(f"Started update {update_id}")

t0 = time.time()
while True:
    state = w.pipelines.get_update(pipeline_id=pipeline_id, update_id=update_id).update.state.value
    if state in ("COMPLETED", "FAILED", "CANCELED"):
        break
    time.sleep(15)

print(f"Update {update_id} finished with state {state} after {time.time() - t0:.0f}s")
if state != "COMPLETED":
    raise RuntimeError(f"Pipeline update {state}. Open {w.config.host}/pipelines/{pipeline_id} to see the error.")

# COMMAND ----------

# DBTITLE 1,Verify the materialized view and its primary key
pk = spark.sql(f"""
    SELECT constraint_name
    FROM {CATALOG}.information_schema.table_constraints
    WHERE table_schema = '{SCHEMA}' AND table_name = '{DELTA_FEATURE_TABLE_NAME}' AND constraint_type = 'PRIMARY KEY'
""").collect()
if not pk:
    raise RuntimeError(f"{DELTA_FEATURE_TABLE} has no primary key; FeatureLookup in notebook 04 would fail.")

n_rows = spark.table(DELTA_FEATURE_TABLE).count()
n_hist = spark.table(HISTORY_TABLE).count()
print(f"{DELTA_FEATURE_TABLE}: {n_rows:,} rows (history: {n_hist:,}) | primary key: {pk[0][0]}")
display(spark.table(DELTA_FEATURE_TABLE).limit(5))