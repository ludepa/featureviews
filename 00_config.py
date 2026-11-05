# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 00 — Configuration
# MAGIC
# MAGIC Shared settings for the whole demo. Every other notebook loads this with `%run ./00_config`, so you only change the target location here.
# MAGIC
# MAGIC * **catalog** / **schema** widgets (default `main` / `ice_demo`) control where every table, Feature View, model, and the model Volume are created. You can also pass them as job parameters.
# MAGIC * The schema and the `models` Volume are created automatically if they don't exist.
# MAGIC
# MAGIC You don't need to run this notebook directly.

# COMMAND ----------

# DBTITLE 1,Target location, names, and feature definitions
import re

# ---------------------------------------------------------------------------
# Target catalog / schema (override with the widgets or job parameters)
# ---------------------------------------------------------------------------
dbutils.widgets.text("catalog", "main", "Catalog")
dbutils.widgets.text("schema", "ice_demo", "Schema")

CATALOG = dbutils.widgets.get("catalog").strip()
SCHEMA = dbutils.widgets.get("schema").strip()
for _label, _value in [("catalog", CATALOG), ("schema", SCHEMA)]:
    if not re.fullmatch(r"[A-Za-z0-9_]+", _value):
        raise ValueError(f"Invalid {_label} name {_value!r}: use letters, digits, and underscores only.")

# ---------------------------------------------------------------------------
# Tables, models, and artifacts created by the demo
# ---------------------------------------------------------------------------
HISTORY_TABLE_NAME       = "ice_market_history"     # 5 time-series source columns
BASE_TABLE_NAME          = "ice_model_base"         # 67 request-time predictors + price label
DELTA_FEATURE_TABLE_NAME = "ice_delta_features"     # 30 pre-computed features (SDP materialized view)

HISTORY_TABLE       = f"{CATALOG}.{SCHEMA}.{HISTORY_TABLE_NAME}"
BASE_TABLE          = f"{CATALOG}.{SCHEMA}.{BASE_TABLE_NAME}"
DELTA_FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.{DELTA_FEATURE_TABLE_NAME}"
SHAP_TABLE          = f"{CATALOG}.{SCHEMA}.ice_example1_shap"

FV_MODEL_NAME    = f"{CATALOG}.{SCHEMA}.ice_price_fv_model"     # Example 1 (Feature Views)
DELTA_MODEL_NAME = f"{CATALOG}.{SCHEMA}.ice_price_delta_model"  # Example 2 (Delta features)

MODEL_VOLUME    = f"/Volumes/{CATALOG}/{SCHEMA}/models"
EX1_PICKLE_PATH = f"{MODEL_VOLUME}/example1_xgb_price_model.pkl"
EX2_PICKLE_PATH = f"{MODEL_VOLUME}/example2_xgb_price_model.pkl"

PIPELINE_NAME     = f"ice_demo_feature_pipeline_{CATALOG}_{SCHEMA}"
CURRENT_USER      = spark.sql("SELECT current_user()").first()[0]
MLFLOW_EXPERIMENT = f"/Users/{CURRENT_USER}/ice_price_demo_experiment"

# ---------------------------------------------------------------------------
# Simulation parameters
# ---------------------------------------------------------------------------
SEED = 42
MINUTES_PER_INSTRUMENT = 2500   # ~42 hours of minute bars per instrument
INSTRUMENTS = [
    "ICE_BRENT_F1", "ICE_WTI_F1", "ICE_TTF_GAS_F1", "ICE_POWER_PEAK_F1",
    "ICE_SOFR_F1", "ICE_USDX_F1", "ICE_EUA_CARBON_F1", "ICE_GASOIL_F1",
]

# ---------------------------------------------------------------------------
# Time-series features: 5 source columns x 3 windows x (std, change) = 30
# ---------------------------------------------------------------------------
WINDOWS = [5, 10, 15]   # lookback windows in minutes
TS_SOURCE_FEATURES = ["mid_price", "spread_bps", "trade_size", "implied_vol_30d", "order_imbalance"]
TS_REQ_MAP = {   # history column -> request-time column
    "mid_price": "mid_price_req",
    "spread_bps": "spread_bps_req",
    "trade_size": "trade_size_req",
    "implied_vol_30d": "implied_vol_30d_req",
    "order_imbalance": "order_imbalance_req",
}
STD_FEATURES    = [f"{f}_{w}_std" for f in TS_SOURCE_FEATURES for w in WINDOWS]
CHANGE_FEATURES = [f"{f}_{w}_change" for f in TS_SOURCE_FEATURES for w in WINDOWS]
TS_FEATURES     = STD_FEATURES + CHANGE_FEATURES

# ---------------------------------------------------------------------------
# 67 base (request-time) features grouped by asset-class domain
# ---------------------------------------------------------------------------
RATE_FEATURES = [
    "sofr_1m", "sofr_3m", "fed_funds_rate", "treasury_2y_yield", "treasury_5y_yield", "treasury_10y_yield",
    "treasury_30y_yield", "swap_2y", "swap_5y", "swap_10y", "breakeven_5y", "breakeven_10y",
]
ENERGY_FEATURES = [
    "brent_front", "brent_next", "wti_front", "natgas_front", "gasoil_front", "power_peak_front",
    "carbon_front", "crack_spread_3_2_1", "refining_margin", "freight_index",
]
CREDIT_FEATURES = [
    "investment_grade_spread", "high_yield_spread", "cdx_ig", "cdx_hy",
    "mortgage_spread", "municipal_spread", "repo_rate", "funding_spread",
]
FX_FEATURES = [
    "dxy_index", "eurusd", "usdjpy", "gbpusd", "usdcad", "usdchf", "audusd", "emerging_fx_basket",
]
EQUITY_VOL_FEATURES = [
    "spx_level", "vix", "russell_2000", "nasdaq_100", "bank_equity_index",
    "exchange_volume_index", "clearing_flow_index", "dealer_gamma_proxy",
]
MICROSTRUCTURE_FEATURES = [
    "bid_price", "ask_price", "mid_price_req", "bid_size", "ask_size", "trade_size_req", "order_imbalance_req",
    "depth_5_levels", "depth_10_levels", "trade_count_1m", "volume_1m", "buy_ratio_1m", "sell_ratio_1m",
    "realized_vol_5m_req", "realized_vol_10m_req", "realized_vol_15m_req", "implied_vol_30d_req",
    "implied_vol_90d_req", "spread_bps_req", "open_interest", "basis_front_second",
]
BASE_FEATURES = RATE_FEATURES + ENERGY_FEATURES + CREDIT_FEATURES + FX_FEATURES + EQUITY_VOL_FEATURES + MICROSTRUCTURE_FEATURES
MODEL_FEATURES = BASE_FEATURES + TS_FEATURES
assert len(BASE_FEATURES) == 67 and len(TS_FEATURES) == 30 and len(MODEL_FEATURES) == 97

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def require_tables(*tables, hint):
    """Fail fast with a clear message if a prerequisite table is missing."""
    missing = [t for t in tables if not spark.catalog.tableExists(t)]
    if missing:
        raise ValueError(f"Missing table(s): {missing}. {hint}")

def latest_model_version(model_name):
    """Return the latest registered version of a Unity Catalog model."""
    from mlflow import MlflowClient
    versions = MlflowClient(registry_uri="databricks-uc").search_model_versions(f"name='{model_name}'")
    if not versions:
        raise ValueError(f"No versions found for {model_name}. Run the corresponding Example notebook first.")
    return max(int(v.version) for v in versions)

# Create the schema and the model Volume if needed
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.models")

print(f"Config loaded | target: {CATALOG}.{SCHEMA} | model volume: {MODEL_VOLUME}")