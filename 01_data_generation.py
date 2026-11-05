# Databricks notebook source
# DBTITLE 1,Overview
# MAGIC %md
# MAGIC # 01 — ICE Synthetic Data Generation
# MAGIC
# MAGIC Generates a realistic Intercontinental Exchange-style minute-bar dataset for **8 ICE instruments** (energy, rates, carbon, FX) and writes two Delta tables used by both examples:
# MAGIC
# MAGIC * **History table** (`ice_market_history`) — 5 time-series source columns used to compute the rolling features.
# MAGIC * **Base table** (`ice_model_base`) — 67 request-time predictors + the `price` label.
# MAGIC
# MAGIC Run this notebook first. Re-running it overwrites both tables.

# COMMAND ----------

# DBTITLE 1,Load shared configuration
# MAGIC %run ./00_config

# COMMAND ----------

# DBTITLE 1,Generate synthetic market data
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Global macro factors — random-walk processes that drive correlated moves
# across all instruments (rates, commodities, liquidity, vol, credit, FX, equity).
# ---------------------------------------------------------------------------
rng = np.random.default_rng(SEED)
timestamps = pd.date_range("2026-01-05 08:00:00", periods=MINUTES_PER_INSTRUMENT, freq="min")
global_factors = pd.DataFrame({
    "event_ts": timestamps,
    "rate_level":      4.20  + np.cumsum(rng.normal(0, 0.0020, MINUTES_PER_INSTRUMENT)),
    "commodity_level": 72    + np.cumsum(rng.normal(0, 0.0550, MINUTES_PER_INSTRUMENT)),
    "liquidity_level": 0.55  + np.cumsum(rng.normal(0, 0.0015, MINUTES_PER_INSTRUMENT)),
    "vol_level":       0.24  + np.cumsum(rng.normal(0, 0.0008, MINUTES_PER_INSTRUMENT)),
    "credit_level":    1.80  + np.cumsum(rng.normal(0, 0.0030, MINUTES_PER_INSTRUMENT)),
    "fx_level":        101   + np.cumsum(rng.normal(0, 0.0300, MINUTES_PER_INSTRUMENT)),
    "equity_level":    5100  + np.cumsum(rng.normal(0, 2.5000, MINUTES_PER_INSTRUMENT)),
})

def noise(scale, size):
    """Helper: i.i.d. Gaussian noise."""
    return rng.normal(0, scale, size)

# ---------------------------------------------------------------------------
# Per-instrument feature generation. Each instrument shares the global factors
# but gets its own bias, seasonal pattern, and idiosyncratic noise.
# ---------------------------------------------------------------------------
frames = []
for idx, instrument in enumerate(INSTRUMENTS):
    n = MINUTES_PER_INSTRUMENT
    inst = global_factors.copy()
    inst["instrument_id"] = instrument
    inst["time_id"] = [f"{instrument}_{ts.strftime('%Y%m%d%H%M')}" for ts in inst["event_ts"]]
    inst_bias = idx * 3.5                              # price-level offset per instrument
    seasonal = np.sin(np.linspace(0, 12, n) + idx)     # slow intraday cycle

    # --- Time-series source features (written to the history table) ----------
    inst["order_imbalance"] = np.clip(0.35 * seasonal + noise(0.12, n), -1.0, 1.0)
    inst["implied_vol_30d"] = 0.20 + 0.08 * inst["vol_level"] + 0.02 * np.abs(inst["order_imbalance"]) + noise(0.01, n)
    inst["mid_price"] = 95 + inst_bias + 0.42 * inst["commodity_level"] - 0.65 * inst["rate_level"] + 4.5 * inst["order_imbalance"] + np.cumsum(noise(0.06, n))
    inst["spread_bps"] = np.maximum(1.0, 8 + 12 * inst["vol_level"] + 3 * np.abs(inst["order_imbalance"]) + noise(0.7, n))
    inst["trade_size"] = np.maximum(50, np.exp(7.0 + 0.20 * inst["liquidity_level"] - 0.15 * inst["vol_level"] + noise(0.12, n)))

    # --- Rates features (12) ------------------------------------------------
    inst["sofr_1m"] = inst["rate_level"] + 0.02 + noise(0.01, n)
    inst["sofr_3m"] = inst["rate_level"] + 0.05 + noise(0.01, n)
    inst["fed_funds_rate"] = inst["rate_level"] - 0.03 + noise(0.008, n)
    inst["treasury_2y_yield"]  = inst["rate_level"] - 0.10 + noise(0.015, n)
    inst["treasury_5y_yield"]  = inst["rate_level"] + 0.02 + noise(0.015, n)
    inst["treasury_10y_yield"] = inst["rate_level"] + 0.18 + noise(0.02, n)
    inst["treasury_30y_yield"] = inst["rate_level"] + 0.35 + noise(0.02, n)
    inst["swap_2y"]  = inst["treasury_2y_yield"] + 0.04 + noise(0.01, n)
    inst["swap_5y"]  = inst["treasury_5y_yield"] + 0.05 + noise(0.01, n)
    inst["swap_10y"] = inst["treasury_10y_yield"] + 0.06 + noise(0.01, n)
    inst["breakeven_5y"]  = 2.10 + 0.08 * inst["commodity_level"] / 100 + noise(0.03, n)
    inst["breakeven_10y"] = 2.25 + 0.09 * inst["commodity_level"] / 100 + noise(0.03, n)

    # --- Energy features (10) -----------------------------------------------
    inst["brent_front"] = inst["commodity_level"] + 0.9 + noise(0.3, n)
    inst["brent_next"]  = inst["commodity_level"] + 1.2 + noise(0.3, n)
    inst["wti_front"]   = inst["commodity_level"] - 2.0 + noise(0.35, n)
    inst["natgas_front"]       = 3.0 + 0.06 * inst["commodity_level"] + noise(0.08, n)
    inst["gasoil_front"]       = 650 + 4.5 * inst["commodity_level"] + noise(3.5, n)
    inst["power_peak_front"]   = 48 + 5.5 * inst["vol_level"] + 0.05 * inst["commodity_level"] + noise(0.6, n)
    inst["carbon_front"]       = 70 + 0.25 * inst["commodity_level"] + noise(0.5, n)
    inst["crack_spread_3_2_1"] = 12 + 0.14 * inst["commodity_level"] + noise(0.4, n)
    inst["refining_margin"]    = 10 + 0.11 * inst["commodity_level"] + noise(0.4, n)
    inst["freight_index"]      = 1800 + 7 * inst["commodity_level"] + noise(12, n)

    # --- Credit features (8) ------------------------------------------------
    inst["investment_grade_spread"] = 95 + 4 * inst["credit_level"] + noise(1.5, n)
    inst["high_yield_spread"] = 320 + 14 * inst["credit_level"] + noise(3.0, n)
    inst["cdx_ig"]   = 62 + 2.0 * inst["credit_level"] + noise(0.9, n)
    inst["cdx_hy"]   = 360 + 10 * inst["credit_level"] + noise(3.0, n)
    inst["mortgage_spread"]  = 120 + 4.5 * inst["credit_level"] + noise(1.5, n)
    inst["municipal_spread"] = 75 + 2.5 * inst["credit_level"] + noise(1.2, n)
    inst["repo_rate"]        = inst["rate_level"] - 0.12 + noise(0.01, n)
    inst["funding_spread"]   = 18 + 1.8 * inst["credit_level"] + noise(0.6, n)

    # --- FX features (8) ----------------------------------------------------
    inst["dxy_index"] = inst["fx_level"] + noise(0.1, n)
    inst["eurusd"] = 1.08 - 0.002 * (inst["fx_level"] - 101) + noise(0.002, n)
    inst["usdjpy"] = 148 + 0.40 * (inst["fx_level"] - 101) + noise(0.10, n)
    inst["gbpusd"] = 1.26 - 0.002 * (inst["fx_level"] - 101) + noise(0.002, n)
    inst["usdcad"] = 1.35 + 0.003 * (inst["fx_level"] - 101) + noise(0.002, n)
    inst["usdchf"] = 0.89 + 0.0015 * (inst["fx_level"] - 101) + noise(0.001, n)
    inst["audusd"] = 0.67 - 0.002 * (inst["fx_level"] - 101) + noise(0.002, n)
    inst["emerging_fx_basket"] = 48 - 0.05 * (inst["fx_level"] - 101) + noise(0.06, n)

    # --- Equity / vol features (8) ------------------------------------------
    inst["spx_level"]    = inst["equity_level"] + noise(8, n)
    inst["vix"]          = 12 + 14 * inst["vol_level"] + noise(0.3, n)
    inst["russell_2000"] = 2200 + 0.32 * inst["equity_level"] + noise(6, n)
    inst["nasdaq_100"]   = 18200 + 2.3 * inst["vol_level"] * 100 + noise(20, n)
    inst["bank_equity_index"]     = 680 + 0.05 * inst["equity_level"] - 2 * inst["credit_level"] + noise(2.0, n)
    inst["exchange_volume_index"] = 100 + 16 * inst["liquidity_level"] + noise(0.8, n)
    inst["clearing_flow_index"]   = 70 + 10 * inst["liquidity_level"] + 2.5 * np.abs(inst["order_imbalance"]) + noise(0.8, n)
    inst["dealer_gamma_proxy"]    = 55 + 20 * inst["vol_level"] - 3 * inst["order_imbalance"] + noise(0.8, n)

    # --- Microstructure features (21) — "_req" = value available at request time
    inst["mid_price_req"]       = inst["mid_price"] + noise(0.03, n)
    inst["spread_bps_req"]      = inst["spread_bps"] + noise(0.15, n)
    inst["trade_size_req"]      = inst["trade_size"] + noise(12, n)
    inst["order_imbalance_req"] = np.clip(inst["order_imbalance"] + noise(0.02, n), -1.0, 1.0)
    inst["implied_vol_30d_req"] = inst["implied_vol_30d"] + noise(0.004, n)
    inst["bid_price"] = inst["mid_price_req"] - inst["spread_bps_req"] / 20000.0 * inst["mid_price_req"]
    inst["ask_price"] = inst["mid_price_req"] + inst["spread_bps_req"] / 20000.0 * inst["mid_price_req"]
    inst["bid_size"]  = np.maximum(10, 0.46 * inst["trade_size_req"] + noise(8, n))
    inst["ask_size"]  = np.maximum(10, 0.44 * inst["trade_size_req"] + noise(8, n))
    inst["depth_5_levels"]  = np.maximum(50, 2.6 * inst["trade_size_req"] + noise(30, n))
    inst["depth_10_levels"] = np.maximum(100, 4.8 * inst["trade_size_req"] + noise(45, n))
    inst["trade_count_1m"]  = np.maximum(1, (18 + 22 * inst["liquidity_level"] + noise(3, n)).round())
    inst["volume_1m"]       = np.maximum(100, inst["trade_count_1m"] * inst["trade_size_req"] * 0.8 + noise(40, n))
    inst["buy_ratio_1m"]    = np.clip(0.50 + 0.22 * inst["order_imbalance_req"] + noise(0.03, n), 0.01, 0.99)
    inst["sell_ratio_1m"]   = np.clip(1.0 - inst["buy_ratio_1m"] + noise(0.01, n), 0.01, 0.99)
    inst["realized_vol_5m_req"]  = np.maximum(0.01, 0.7 * inst["implied_vol_30d_req"] + noise(0.01, n))
    inst["realized_vol_10m_req"] = np.maximum(0.01, 0.8 * inst["implied_vol_30d_req"] + noise(0.01, n))
    inst["realized_vol_15m_req"] = np.maximum(0.01, 0.9 * inst["implied_vol_30d_req"] + noise(0.01, n))
    inst["implied_vol_90d_req"]  = np.maximum(0.01, inst["implied_vol_30d_req"] + 0.03 + noise(0.006, n))
    inst["open_interest"]        = np.maximum(1000, 150000 + idx * 5000 + np.cumsum(noise(80, n)))
    inst["basis_front_second"]   = inst["brent_next"] - inst["brent_front"] + noise(0.04, n)

    frames.append(inst)

print(f"Generated {len(frames)} instruments x {MINUTES_PER_INSTRUMENT} minutes")

# COMMAND ----------

# DBTITLE 1,Compute target label and write Delta tables
# ---------------------------------------------------------------------------
# The price label is a known function of base features + rolling stats + noise,
# so the model is learnable but not trivially linear.
# ---------------------------------------------------------------------------
full_pd = pd.concat(frames, ignore_index=True).sort_values(["instrument_id", "event_ts"]).reset_index(drop=True)

# Rolling statistics used only to build the label
for feature in TS_SOURCE_FEATURES:
    grouped = full_pd.groupby("instrument_id")[feature]
    for window in WINDOWS:
        full_pd[f"__{feature}_{window}_std"]    = grouped.transform(lambda s, w=window: s.rolling(w, min_periods=w).std())
        full_pd[f"__{feature}_{window}_change"] = grouped.transform(lambda s, w=window: s - s.shift(w))

# Drop the first 20 rows per instrument where rolling stats are NaN
full_pd = full_pd.groupby("instrument_id", group_keys=False).apply(lambda pdf: pdf.iloc[20:]).reset_index(drop=True)

full_pd["price"] = (
    0.70  * full_pd["mid_price_req"]
    + 0.11  * full_pd["brent_front"]
    - 0.015 * full_pd["high_yield_spread"]
    + 6.5   * full_pd["order_imbalance_req"]
    + 40.0  * full_pd["implied_vol_30d_req"]
    + 0.45  * full_pd["__mid_price_5_change"]
    - 0.20  * full_pd["__spread_bps_10_std"]
    + 0.002 * full_pd["__trade_size_15_change"]
    - 4.0   * full_pd["__implied_vol_30d_5_std"]
    + 3.0   * full_pd["__order_imbalance_10_change"]
    + noise(0.35, len(full_pd))
)

# ---------------------------------------------------------------------------
# Write the two output tables to Delta
# ---------------------------------------------------------------------------
history_cols = ["instrument_id", "time_id", "event_ts"] + TS_SOURCE_FEATURES
base_cols    = ["instrument_id", "time_id", "event_ts"] + BASE_FEATURES + ["price"]

history_sdf = spark.createDataFrame(full_pd[history_cols])
base_sdf    = spark.createDataFrame(full_pd[base_cols])

history_sdf.write.mode("overwrite").format("delta").saveAsTable(HISTORY_TABLE)
base_sdf.write.mode("overwrite").format("delta").saveAsTable(BASE_TABLE)

print(f"History table: {HISTORY_TABLE}  ({spark.table(HISTORY_TABLE).count():,} rows)")
print(f"Base table:    {BASE_TABLE}  ({spark.table(BASE_TABLE).count():,} rows, {len(BASE_FEATURES)} base features + price)")
display(spark.table(BASE_TABLE).limit(5))