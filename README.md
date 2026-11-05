# ICE Price Prediction Demo: Feature Views vs. Pre-computed Delta Features

An end-to-end demo that trains the same XGBoost price model (97 predictors) two ways and compares how they score in batch:

* **Example 1 — Feature Views:** 30 time-series features are computed on the fly by Databricks Feature Views at inference time.
* **Example 2 — Delta features:** the same 30 features are pre-computed by a Lakeflow Spark Declarative Pipeline into a materialized view, and looked up by key at inference time.

The data is fully synthetic (8 ICE-style instruments, minute bars).

## Contents

| Item | What it does |
| --- | --- |
| `00_config` | Shared settings (catalog/schema widgets, table and model names, feature lists). Loaded by every notebook with `%run`; creates the schema and `models` Volume if missing. |
| `01_data_generation` | Generates the synthetic history table (5 time-series columns) and base table (67 predictors + `price` label). |
| `02_example1_feature_views` | Defines and registers 30 Feature Views, trains XGBoost, and logs/registers `ice_price_fv_model` with `fe.log_model`. |
| `03_example2_feature_pipeline` | Creates (or updates) a serverless pipeline from `pipeline/ice_delta_features.py`, runs it, and checks the materialized view's primary key. |
| `04_example2_delta_features` | Trains XGBoost on the pre-computed features and logs/registers `ice_price_delta_model` with two `FeatureLookup`s. |
| `05_inference_comparison` | Times end-to-end `fe.score_batch` for both models on the same 10k rows, and writes per-prediction SHAP values for Example 1 to a Delta table. |
| `pipeline/ice_delta_features.py` | Pipeline source for the Example 2 materialized view. |

## Prerequisites

* A Unity Catalog-enabled workspace with serverless compute for notebooks and pipelines.
* Permission to use the target catalog and to create a schema in it (or an existing schema where you can create tables, volumes, functions, and models).
* Permission to create pipelines.
* The Feature Views API in `databricks-feature-engineering` (installed by the notebooks) must be available in your workspace.

## How to run

1. Import this folder into your workspace (keep the files together; notebooks reference each other by relative path).
2. Open `00_config` and set the default `catalog` and `schema` (defaults: `main` / `ice_demo`), or set the widgets in each notebook.
3. Run the notebooks in order on serverless compute: `01` → `02` → `03` → `04` → `05`. `02` and `03`/`04` are independent of each other; `05` needs both examples.

To run it as a job instead, create one notebook task per step in that order and pass `catalog` and `schema` as task parameters.

Expected runtime is about 10 minutes end to end on serverless (a test run took about 8.5 minutes), mostly Feature View computation, pipeline startup, and model logging.

## What gets created

In `<catalog>.<schema>`:

* Tables: `ice_market_history`, `ice_model_base`, `ice_example1_shap`
* Materialized view: `ice_delta_features` (owned by the pipeline)
* 30 Feature Views (`<source>_<window>_std` and `<source>_<window>_first`)
* Models: `ice_price_fv_model`, `ice_price_delta_model`
* Volume: `models` (pickled XGBoost models used for SHAP)

Elsewhere:

* Pipeline: `ice_demo_feature_pipeline_<catalog>_<schema>`
* MLflow experiment: `/Users/<you>/ice_price_demo_experiment`

## Reading the results

* **Latency:** each timing covers the full `score_batch` call — model loading, feature computation (Example 1) or key lookup (Example 2), and `predict()`. It ends with an action that reads `prediction`, because Spark is lazy and an action like `count()` would skip the model. The Example 2 time does not include the pipeline refresh that pre-computes its features.
* **SHAP table:** one row per prediction, with `base_value` plus one column per feature. `base_value` plus the 97 feature columns equals the model's prediction.

## Clean up

1. Delete the pipeline `ice_demo_feature_pipeline_<catalog>_<schema>` (this also removes its materialized view).
2. Drop the schema: `DROP SCHEMA <catalog>.<schema> CASCADE` (tables, Feature Views, models, and the Volume).
3. Delete the MLflow experiment `/Users/<you>/ice_price_demo_experiment`.
