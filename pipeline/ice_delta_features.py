# ---------------------------------------------------------------------------
# ICE Demo — Example 2 feature pipeline (Lakeflow Spark Declarative Pipeline)
#
# Computes the 30 time-series features for Example 2 as a materialized view.
# On serverless, the platform decides between an incremental refresh and a
# full recompute on each update.
#
# Pipeline configuration (set by 03_example2_feature_pipeline):
#   ice.history_table  fully qualified source table (catalog.schema.ice_market_history)
#   ice.feature_table  name of the output materialized view (published to the
#                      pipeline's default catalog/schema)
# ---------------------------------------------------------------------------

from pyspark import pipelines as dp
from pyspark.sql import functions as F
from pyspark.sql.window import Window

SOURCE_TABLE  = spark.conf.get("ice.history_table")
FEATURE_TABLE = spark.conf.get("ice.feature_table", "ice_delta_features")

TS_SOURCE_FEATURES = ["mid_price", "spread_bps", "trade_size", "implied_vol_30d", "order_imbalance"]
WINDOWS = [5, 10, 15]

# Explicit schema with a PRIMARY KEY so the view can be used in a FeatureLookup
# (Feature Engineering requires a primary key on feature tables).
_FEATURE_COLS = ",\n    ".join(
    f"{feature}_{w}_{kind} DOUBLE"
    for feature in TS_SOURCE_FEATURES
    for w in WINDOWS
    for kind in ("std", "change")
)
SCHEMA_DDL = f"""
    instrument_id STRING NOT NULL,
    time_id STRING,
    event_ts TIMESTAMP NOT NULL,
    {_FEATURE_COLS},
    CONSTRAINT {FEATURE_TABLE}_pk PRIMARY KEY (instrument_id, event_ts)
"""


@dp.materialized_view(
    name=FEATURE_TABLE,
    comment=(
        "30 time-series features for ICE price prediction (Example 2): "
        "15 rolling stddev_pop + 15 change (current minus lag) features "
        "across 5 source columns and 3 lookback windows (5/10/15 min)."
    ),
    schema=SCHEMA_DDL,
)
def ice_delta_features():
    history = spark.read.table(SOURCE_TABLE)

    stddev_windows = {
        w: Window.partitionBy("instrument_id").orderBy("event_ts").rowsBetween(-w, -1)
        for w in WINDOWS
    }
    lag_window = Window.partitionBy("instrument_id").orderBy("event_ts")

    df = history.select("instrument_id", "time_id", "event_ts", *TS_SOURCE_FEATURES)
    for feature in TS_SOURCE_FEATURES:
        for w in WINDOWS:
            df = df.withColumn(
                f"{feature}_{w}_std",
                F.stddev_pop(F.col(feature)).over(stddev_windows[w]),
            ).withColumn(
                f"{feature}_{w}_change",
                F.col(feature) - F.lag(F.col(feature), w).over(lag_window),
            )

    # Keep only keys + the 30 derived features
    return df.drop(*TS_SOURCE_FEATURES)
