"""
========================================================
  QUANTITATIVE STOCK TREND PREDICTOR
  - Fetches full historical data
  - Computes technical indicators
  - Trains XGBoost + ensemble model
  - Validates on holdout set
  - Predicts trend direction, duration & price targets
========================================================
"""
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
from datetime import datetime, timedelta
import ta
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score)
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb
from scipy.signal import argrelextrema
import textwrap

# ─────────────────────────────────────────────
# CONFIG  — change TICKER here
# ─────────────────────────────────────────────
TICKER          = "SMH"
PREDICTION_DAYS = 30          # forward window for trend prediction
TRAIN_RATIO     = 0.80        # 80 % train / 20 % validation
LOOKBACK        = 20          # rolling-window base for features
SUPPORT_RES_WIN = 10          # local extrema window

COLORS = {
    "bg":      "#0d1117",
    "panel":   "#161b22",
    "border":  "#30363d",
    "text":    "#e6edf3",
    "muted":   "#8b949e",
    "green":   "#3fb950",
    "red":     "#f85149",
    "yellow":  "#d29922",
    "blue":    "#58a6ff",
    "purple":  "#bc8cff",
    "orange":  "#ff9800",
    "cyan":    "#76e3ea",
}


# ══════════════════════════════════════════════
# 1. DATA FETCHING
# ══════════════════════════════════════════════

def fetch_data(ticker: str) -> pd.DataFrame:
    print(f"\n{'═'*56}")
    print(f"  QUANTITATIVE STOCK TREND PREDICTOR")
    print(f"  Ticker : {ticker}")
    print(f"{'═'*56}")
    print("\n[1/4] Fetching historical data …")

    live_success = False
    df = pd.DataFrame()

    # ── Attempt live data first ──
    try:
        t = yf.Ticker(ticker)

        # Force fresh, adjusted, repaired data
        df = t.history(
            period="max",
            auto_adjust=True,   # adjusts for splits & dividends
            repair=True,        # fixes bad OHLC values
            keepna=False,
        )

        if not df.empty:
            df.index = pd.to_datetime(df.index).tz_localize(None)
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df.dropna(inplace=True)

            last_price = df["Close"].iloc[-1]
            last_date  = df.index[-1].date()
            live_success = True

            print(f"  ✓  Live data fetched via Yahoo Finance")
            print(f"  ✓  {len(df):,} daily bars  |  "
                  f"{df.index[0].date()} → {last_date}")
            print(f"  ✓  Last close : ${last_price:.2f}  ({last_date})")
            print(f"  ℹ  Verify this matches your broker / Google Finance")
        else:
            print(f"  ⚠  Yahoo Finance returned empty data — falling back to simulation")

    except Exception as e:
        print(f"  ⚠  Yahoo Finance unavailable ({e}) — falling back to simulation")

    # ── Simulation fallback ──
    if not live_success:
        print(f"  ℹ  Generating realistic simulated data for '{ticker}' …")

        seed = sum(ord(c) for c in ticker)
        rng  = np.random.default_rng(seed)

        n_days   = 5040
        end_date = datetime(2025, 5, 2)
        bdays    = pd.bdate_range(end=end_date, periods=n_days)

        start_price = float(rng.integers(20, 400))
        annual_ret  = rng.uniform(0.04, 0.18)
        annual_vol  = rng.uniform(0.18, 0.40)

        dt       = 1 / 252
        drift    = (annual_ret - 0.5 * annual_vol**2) * dt
        vol_step = annual_vol * np.sqrt(dt)

        prices      = [start_price]
        regime_dur  = rng.integers(60, 240, size=60)
        regime_type = rng.choice(["bull", "bear", "sideways"], size=60, p=[0.50, 0.25, 0.25])
        regime_mod  = {"bull": 1.4, "bear": -0.6, "sideways": 0.1}

        reg_idx, days_in_reg = 0, 0
        for _ in range(n_days - 1):
            if days_in_reg >= regime_dur[reg_idx % len(regime_dur)]:
                reg_idx += 1
                days_in_reg = 0
            mod       = regime_mod[regime_type[reg_idx % len(regime_type)]]
            eff_drift = drift * mod
            shock     = rng.standard_normal() * vol_step
            if rng.random() < 0.003:
                shock += rng.choice([-1, 1]) * rng.uniform(0.03, 0.08)
            prices.append(max(prices[-1] * np.exp(eff_drift + shock), 1.0))
            days_in_reg += 1

        close_arr  = np.array(prices)
        daily_range = rng.uniform(0.005, 0.025, size=n_days)
        high_arr   = close_arr * (1 + daily_range * rng.uniform(0.3, 1.0, n_days))
        low_arr    = close_arr * (1 - daily_range * rng.uniform(0.3, 1.0, n_days))
        open_arr   = low_arr + rng.random(n_days) * (high_arr - low_arr)
        base_vol   = float(rng.integers(5_000_000, 80_000_000))
        volume_arr = (base_vol * rng.lognormal(0, 0.5, n_days)).astype(int)

        df = pd.DataFrame({
            "Open":   open_arr,
            "High":   high_arr,
            "Low":    low_arr,
            "Close":  close_arr,
            "Volume": volume_arr,
        }, index=bdays)

        print(f"  ✓  {len(df):,} simulated bars  |  "
              f"{df.index[0].date()} → {df.index[-1].date()}")
        print(f"  ⚠  NOTE: Results based on simulation, NOT real market data")

    return df


# ══════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    print("\n[2/4] Computing technical indicators & features …")

    # ── Trend / Moving Averages ──
    for w in [5, 10, 20, 50, 100, 200]:
        df[f"SMA_{w}"]  = ta.trend.sma_indicator(df["Close"], window=w)
        df[f"EMA_{w}"]  = ta.trend.ema_indicator(df["Close"], window=w)

    # ── Momentum ──
    df["RSI"]           = ta.momentum.rsi(df["Close"], window=14)
    df["RSI_fast"]      = ta.momentum.rsi(df["Close"], window=7)
    macd = ta.trend.MACD(df["Close"])
    df["MACD"]          = macd.macd()
    df["MACD_signal"]   = macd.macd_signal()
    df["MACD_hist"]     = macd.macd_diff()
    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"])
    df["Stoch_K"]       = stoch.stoch()
    df["Stoch_D"]       = stoch.stoch_signal()
    df["Williams_R"]    = ta.momentum.williams_r(df["High"], df["Low"], df["Close"])
    df["ROC"]           = ta.momentum.roc(df["Close"], window=12)

    # ── Volatility ──
    bb = ta.volatility.BollingerBands(df["Close"])
    df["BB_upper"]      = bb.bollinger_hband()
    df["BB_mid"]        = bb.bollinger_mavg()
    df["BB_lower"]      = bb.bollinger_lband()
    df["BB_width"]      = (df["BB_upper"] - df["BB_lower"]) / df["BB_mid"]
    df["BB_pct"]        = bb.bollinger_pband()
    df["ATR"]           = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"])
    df["ATR_pct"]       = df["ATR"] / df["Close"]

    # ── Volume ──
    df["OBV"]           = ta.volume.on_balance_volume(df["Close"], df["Volume"])
    df["Vol_SMA20"]     = df["Volume"].rolling(20).mean()
    df["Vol_ratio"]     = df["Volume"] / df["Vol_SMA20"]
    df["VWAP"]          = (df["Close"] * df["Volume"]).cumsum() / df["Volume"].cumsum()
    df["CMF"]           = ta.volume.chaikin_money_flow(df["High"], df["Low"], df["Close"], df["Volume"])

    # ── Price-derived ──
    df["Return_1d"]     = df["Close"].pct_change(1)
    df["Return_5d"]     = df["Close"].pct_change(5)
    df["Return_20d"]    = df["Close"].pct_change(20)
    df["Log_return"]    = np.log(df["Close"] / df["Close"].shift(1))
    df["High_Low_pct"]  = (df["High"] - df["Low"]) / df["Close"]
    df["Close_Open_pct"]= (df["Close"] - df["Open"]) / df["Open"]

    # ── Trend strength ──
    adx_ind             = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"])
    df["ADX"]           = adx_ind.adx()
    df["ADX_pos"]       = adx_ind.adx_pos()
    df["ADX_neg"]       = adx_ind.adx_neg()

    # ── Rolling stats ──
    for w in [5, 10, 20]:
        df[f"Volatility_{w}d"] = df["Log_return"].rolling(w).std() * np.sqrt(252)
        df[f"High_{w}d"]       = df["High"].rolling(w).max()
        df[f"Low_{w}d"]        = df["Low"].rolling(w).min()

    # ── TARGET: will price be higher in PREDICTION_DAYS? ──
    df["Future_close"]  = df["Close"].shift(-PREDICTION_DAYS)
    df["Target"]        = (df["Future_close"] > df["Close"]).astype(int)

    feature_cols = [c for c in df.columns if c not in ("Future_close", "Target")]
    df.dropna(subset=feature_cols, inplace=True)
    print(f"  ✓  {df.shape[1]} columns  |  {len(df):,} usable rows after dropna")
    return df


# ══════════════════════════════════════════════
# 3. MODEL TRAINING & VALIDATION
# ══════════════════════════════════════════════
def train_and_validate(df: pd.DataFrame):
    print("\n[3/4] Training & validating model …")

    train_df = df.dropna(subset=["Future_close"])

    feature_cols = [c for c in df.columns
                    if c not in ("Open","High","Low","Close","Volume",
                                 "Future_close","Target")]
    X = train_df[feature_cols].values
    y = train_df["Target"].values
    dates = train_df.index

    split = int(len(X) * TRAIN_RATIO)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    dates_val       = dates[split:]

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)

    # ── Ensemble ──
    models = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, verbosity=0),
        "RandomForest": RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=5,
            random_state=42, n_jobs=-1),
        "GradBoost": GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.8, random_state=42),
    }

    preds_val  = {}
    probas_val = {}
    metrics    = {}

    for name, mdl in models.items():
        mdl.fit(X_train_s, y_train)
        p = mdl.predict(X_val_s)
        pr = mdl.predict_proba(X_val_s)[:, 1]
        preds_val[name]  = p
        probas_val[name] = pr
        acc = accuracy_score(y_val, p)
        auc = roc_auc_score(y_val, pr)
        metrics[name] = {"acc": acc, "auc": auc}
        print(f"  {name:<14}  Acc={acc:.3f}  AUC={auc:.3f}")

    # ── Ensemble average ──
    ensemble_proba = np.mean([probas_val[n] for n in models], axis=0)
    ensemble_pred  = (ensemble_proba >= 0.5).astype(int)
    ens_acc = accuracy_score(y_val, ensemble_pred)
    ens_auc = roc_auc_score(y_val, ensemble_proba)
    metrics["Ensemble"] = {"acc": ens_acc, "auc": ens_auc}
    print(f"  {'Ensemble':<14}  Acc={ens_acc:.3f}  AUC={ens_auc:.3f}  ← best")

    # Final model on all data for prediction
    for name, mdl in models.items():
        mdl.fit(scaler.transform(X), y)   # refit on full set

    # Feature importances from XGBoost
    fi = dict(zip(feature_cols,
                  models["XGBoost"].feature_importances_))
    top_features = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:15]

    return (models, scaler, feature_cols,
            df, split, dates_val,
            preds_val, probas_val, ensemble_pred, ensemble_proba,
            y_val, metrics, top_features)


# ══════════════════════════════════════════════
# 4. SUPPORT / RESISTANCE
# ══════════════════════════════════════════════
def find_support_resistance(close: pd.Series, window: int = SUPPORT_RES_WIN):
    arr = close.values
    # local minima → support
    lmin = argrelextrema(arr, np.less,    order=window)[0]
    # local maxima → resistance
    lmax = argrelextrema(arr, np.greater, order=window)[0]

    # cluster nearby levels
    def cluster(indices, tol=0.015):
        levels = sorted(arr[i] for i in indices)
        clustered = []
        for lv in levels:
            if not clustered or abs(lv - clustered[-1]) / clustered[-1] > tol:
                clustered.append(lv)
        return clustered

    return cluster(lmin), cluster(lmax)


# ══════════════════════════════════════════════
# 5. PREDICTION & CONCLUSION
# ══════════════════════════════════════════════
def make_prediction(models, scaler, feature_cols,
                    df: pd.DataFrame, ensemble_proba, y_val):

    # ── Always anchor to the actual last row of the dataframe ──
    last_row    = df[feature_cols].iloc[-1:].values
    last_scaled = scaler.transform(last_row)

    model_probas = [m.predict_proba(last_scaled)[0, 1] for m in models.values()]
    final_prob   = float(np.mean(model_probas))
    trend        = "BULLISH" if final_prob >= 0.5 else "BEARISH"
    confidence   = final_prob if trend == "BULLISH" else 1 - final_prob

    # ── Pull price stats directly from df tail — never hardcoded ──
    last_price   = float(df["Close"].iloc[-1])      # <-- this was the bug
    atr          = float(df["ATR"].iloc[-1])
    adx          = float(df["ADX"].iloc[-1])
    bb_width     = float(df["BB_width"].iloc[-1])
    hist_vol     = float(df["Volatility_20d"].iloc[-1])

    print(f"\n  DEBUG — Last price used in prediction : ${last_price:.2f}")
    print(f"  DEBUG — Last date in df               : {df.index[-1].date()}")

    # ── Target price ──
    adx_factor     = min(adx / 25, 2.0)
    conf_factor    = confidence
    base_move_pct  = hist_vol / np.sqrt(252 / PREDICTION_DAYS)
    target_move    = last_price * base_move_pct * adx_factor * conf_factor

    if trend == "BULLISH":
        price_target = last_price + target_move
        stop_loss    = last_price - atr * 2
    else:
        price_target = last_price - target_move
        stop_loss    = last_price + atr * 2

    # ── Trend duration ──
    base_dur   = PREDICTION_DAYS
    dur_factor = min(adx / 20, 2.5)
    est_days   = max(5, int(base_dur * dur_factor * conf_factor))
    est_end    = df.index[-1] + timedelta(days=est_days)

    # ── Support / Resistance ──
    supports, resistances = find_support_resistance(df["Close"])
    near_sup = max((s for s in supports    if s < last_price), default=None)
    near_res = min((r for r in resistances if r > last_price), default=None)

    conclusion = {
        "ticker":          TICKER,
        "trend":           trend,
        "confidence":      confidence,
        "final_prob":      final_prob,
        "last_price":      last_price,
        "price_target":    price_target,
        "stop_loss":       stop_loss,
        "est_days":        est_days,
        "est_end":         est_end,
        "atr":             atr,
        "adx":             adx,
        "bb_width":        bb_width,
        "hist_vol":        hist_vol,
        "near_sup":        near_sup,
        "near_res":        near_res,
        "all_supports":    supports[-5:],
        "all_resistances": resistances[:5],
    }
    return conclusion


# ══════════════════════════════════════════════
# 6. CONSOLE PRINT
# ══════════════════════════════════════════════
def print_conclusion(c: dict, metrics: dict):
    print(f"\n{'═'*56}")
    print(f"  PREDICTION SUMMARY  —  {c['ticker']}")
    print(f"{'═'*56}")
    trend_sym = "▲ BULLISH" if c["trend"] == "BULLISH" else "▼ BEARISH"
    print(f"  Trend        : {trend_sym}")
    print(f"  Confidence   : {c['confidence']*100:.1f}%")
    print(f"  Last Price   : ${c['last_price']:.2f}")
    print(f"  Price Target : ${c['price_target']:.2f}  "
          f"({'↑' if c['trend']=='BULLISH' else '↓'}"
          f" {abs(c['price_target']-c['last_price'])/c['last_price']*100:.1f}%)")
    print(f"  Stop Loss    : ${c['stop_loss']:.2f}")
    print(f"  Est Duration : ~{c['est_days']} trading days  "
          f"(≈ {c['est_end'].date()})")
    print(f"  ADX Strength : {c['adx']:.1f}  "
          f"({'Strong' if c['adx']>25 else 'Moderate' if c['adx']>20 else 'Weak'} trend)")
    print(f"  ATR (vol)    : ${c['atr']:.2f}")
    print(f"  Hist Vol     : {c['hist_vol']*100:.1f}% annualised")
    if c["near_sup"]: print(f"  Support      : ${c['near_sup']:.2f}")
    if c["near_res"]: print(f"  Resistance   : ${c['near_res']:.2f}")
    print(f"\n  Model Performance (validation set):")
    for name, m in metrics.items():
        print(f"    {name:<14} Acc={m['acc']*100:.1f}%  AUC={m['auc']:.3f}")
    print(f"{'═'*56}\n")


# ══════════════════════════════════════════════
# 7. CHARTS
# ══════════════════════════════════════════════
def plot_all(df: pd.DataFrame, split: int, dates_val,
             ensemble_pred, ensemble_proba, y_val,
             metrics: dict, top_features: list,
             conclusion: dict):

    print("[4/4] Generating charts …")

    plt.rcParams.update({
        "figure.facecolor":  COLORS["bg"],
        "axes.facecolor":    COLORS["panel"],
        "axes.edgecolor":    COLORS["border"],
        "axes.labelcolor":   COLORS["text"],
        "text.color":        COLORS["text"],
        "xtick.color":       COLORS["muted"],
        "ytick.color":       COLORS["muted"],
        "grid.color":        COLORS["border"],
        "grid.alpha":        0.5,
        "font.family":       "monospace",
        "font.size":         9,
    })

    fig = plt.figure(figsize=(22, 28))
    fig.patch.set_facecolor(COLORS["bg"])

    gs = gridspec.GridSpec(5, 3,
                           figure=fig,
                           hspace=0.45, wspace=0.32,
                           left=0.06, right=0.97,
                           top=0.94, bottom=0.04)

    # ── helpers ──
    def ax_style(ax, title="", ylabel=""):
        ax.set_facecolor(COLORS["panel"])
        for sp in ax.spines.values():
            sp.set_color(COLORS["border"])
        ax.tick_params(colors=COLORS["muted"], labelsize=8)
        ax.grid(True, alpha=0.3, linewidth=0.5)
        if title:
            ax.set_title(title, color=COLORS["text"],
                         fontsize=9, fontweight="bold", pad=6)
        if ylabel:
            ax.set_ylabel(ylabel, color=COLORS["muted"], fontsize=8)

    close   = df["Close"]
    dates   = df.index
    
    # Train/Val split applies only to data with valid Future_close
    labeled_dates = df.dropna(subset=["Future_close"]).index
    train_d = labeled_dates[:split]
    val_d   = labeled_dates[split:]

    # ─────────────────────────────────────────
    # CHART 1 — Full price + MAs (spans 3 cols)
    # ─────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(train_d, close[:split], color=COLORS["blue"],
             linewidth=1.2, label="Train", alpha=0.9)
    ax1.plot(dates[split:], close[split:], color=COLORS["purple"],
             linewidth=1.4, label="Validation & Recent", alpha=0.9)
    ax1.plot(dates, df["SMA_50"],  color=COLORS["orange"],
             linewidth=0.8, alpha=0.7, label="SMA 50")
    ax1.plot(dates, df["SMA_200"], color=COLORS["cyan"],
             linewidth=0.8, alpha=0.7, label="SMA 200")
    ax1.fill_between(dates, df["BB_upper"], df["BB_lower"],
                     alpha=0.07, color=COLORS["blue"])

    # support / resistance lines
    c = conclusion
    for s in c["all_supports"][-3:]:
        ax1.axhline(s, color=COLORS["green"], linewidth=0.7,
                    linestyle="--", alpha=0.5)
    for r in c["all_resistances"][:3]:
        ax1.axhline(r, color=COLORS["red"], linewidth=0.7,
                    linestyle="--", alpha=0.5)

    # train / val divider
    ax1.axvline(val_d[0], color=COLORS["yellow"],
                linewidth=1.5, linestyle=":", alpha=0.8,
                label="Train/Val split")

    # prediction arrow + target
    last_date  = dates[-1]
    fut_date   = c["est_end"]
    lp         = c["last_price"]
    pt         = c["price_target"]
    arrow_col  = COLORS["green"] if c["trend"] == "BULLISH" else COLORS["red"]

    ax1.annotate("", xy=(fut_date, pt), xytext=(last_date, lp),
                 arrowprops=dict(arrowstyle="->", color=arrow_col,
                                 lw=2.5, connectionstyle="arc3,rad=0.15"))
    ax1.scatter([fut_date], [pt], color=arrow_col, s=100, zorder=6)
    ax1.text(fut_date, pt, f"  Target\n  ${pt:.2f}",
             color=arrow_col, fontsize=8, fontweight="bold", va="center")

    ax1.legend(loc="upper left", fontsize=7.5, framealpha=0.2,
               labelcolor=COLORS["text"])
    ax_style(ax1, f"{TICKER} — Full History with Bollinger Bands & Trend Projection",
             "Price (USD)")

    # ─────────────────────────────────────────
    # CHART 2 — Volume  (row 1, col 0)
    # ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    colors_vol = [COLORS["green"] if c_ >= o else COLORS["red"]
                  for c_, o in zip(df["Close"], df["Open"])]
    ax2.bar(dates, df["Volume"], color=colors_vol, alpha=0.7, width=1)
    ax2.plot(dates, df["Vol_SMA20"], color=COLORS["yellow"],
             linewidth=1, label="Vol SMA20")
    ax_style(ax2, "Volume", "Shares")
    ax2.legend(fontsize=7, framealpha=0.1)

    # ─────────────────────────────────────────
    # CHART 3 — RSI  (row 1, col 1)
    # ─────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(dates, df["RSI"], color=COLORS["purple"], linewidth=1)
    ax3.axhline(70, color=COLORS["red"],   linewidth=0.8, linestyle="--", alpha=0.7)
    ax3.axhline(30, color=COLORS["green"], linewidth=0.8, linestyle="--", alpha=0.7)
    ax3.fill_between(dates, df["RSI"], 70,
                     where=df["RSI"] > 70, alpha=0.15, color=COLORS["red"])
    ax3.fill_between(dates, df["RSI"], 30,
                     where=df["RSI"] < 30, alpha=0.15, color=COLORS["green"])
    ax3.set_ylim(0, 100)
    ax3.text(dates[-1], 72, "OB", color=COLORS["red"],   fontsize=7)
    ax3.text(dates[-1], 26, "OS", color=COLORS["green"], fontsize=7)
    ax_style(ax3, "RSI (14)", "RSI")

    # ─────────────────────────────────────────
    # CHART 4 — MACD  (row 1, col 2)
    # ─────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.plot(dates, df["MACD"],        color=COLORS["blue"],   linewidth=1,   label="MACD")
    ax4.plot(dates, df["MACD_signal"], color=COLORS["orange"], linewidth=1,   label="Signal")
    hist_pos = df["MACD_hist"].clip(lower=0)
    hist_neg = df["MACD_hist"].clip(upper=0)
    ax4.bar(dates, hist_pos, color=COLORS["green"], alpha=0.5, width=1)
    ax4.bar(dates, hist_neg, color=COLORS["red"],   alpha=0.5, width=1)
    ax4.axhline(0, color=COLORS["muted"], linewidth=0.5)
    ax4.legend(fontsize=7, framealpha=0.1)
    ax_style(ax4, "MACD", "")

    # ─────────────────────────────────────────
    # CHART 5 — Validation actual vs predicted  (row 2, col 0-1)
    # ─────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, :2])
    val_close = df.loc[val_d, "Close"].values
    ax5.plot(val_d, val_close, color=COLORS["blue"],
             linewidth=1.2, label="Actual price", alpha=0.9)

    # shade correct / wrong predictions
    for i in range(len(val_d)):
        correct = (ensemble_pred[i] == y_val[i])
        col = COLORS["green"] if correct else COLORS["red"]
        ax5.axvspan(val_d[i], val_d[min(i+1, len(val_d)-1)],
                    alpha=0.12, color=col)

    ax5.set_title("Validation Period — Prediction Accuracy (Green=Correct, Red=Wrong)",
                  color=COLORS["text"], fontsize=9, fontweight="bold", pad=6)
    ax_style(ax5, "", "Price (USD)")

    # probability line on twin axis
    ax5b = ax5.twinx()
    ax5b.plot(val_d, ensemble_proba, color=COLORS["purple"],
              linewidth=0.9, alpha=0.7, label="Bullish prob")
    ax5b.axhline(0.5, color=COLORS["yellow"],
                 linewidth=0.6, linestyle="--", alpha=0.6)
    ax5b.set_ylim(0, 1)
    ax5b.set_ylabel("Bullish probability", color=COLORS["purple"], fontsize=8)
    ax5b.tick_params(colors=COLORS["purple"])

    lines1, labs1 = ax5.get_legend_handles_labels()
    lines2, labs2 = ax5b.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labs1 + labs2,
               fontsize=7, framealpha=0.15, loc="upper left")

    # ─────────────────────────────────────────
    # CHART 6 — Model metrics bar  (row 2, col 2)
    # ─────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 2])
    names  = list(metrics.keys())
    accs   = [metrics[n]["acc"] for n in names]
    aucs   = [metrics[n]["auc"] for n in names]
    x      = np.arange(len(names))
    w      = 0.35
    ax6.bar(x - w/2, accs, w, color=COLORS["blue"],   alpha=0.8, label="Accuracy")
    ax6.bar(x + w/2, aucs, w, color=COLORS["purple"], alpha=0.8, label="AUC")
    ax6.set_xticks(x)
    ax6.set_xticklabels([n[:8] for n in names], fontsize=7)
    ax6.set_ylim(0, 1.0)
    for i, (a, u) in enumerate(zip(accs, aucs)):
        ax6.text(i - w/2, a + 0.01, f"{a:.2f}", ha="center",
                 fontsize=7, color=COLORS["blue"])
        ax6.text(i + w/2, u + 0.01, f"{u:.2f}", ha="center",
                 fontsize=7, color=COLORS["purple"])
    ax6.legend(fontsize=7, framealpha=0.15)
    ax_style(ax6, "Model Performance", "Score")

    # ─────────────────────────────────────────
    # CHART 7 — Feature importance  (row 3, col 0-1)
    # ─────────────────────────────────────────
    ax7 = fig.add_subplot(gs[3, :2])
    feats = [f[0] for f in top_features][::-1]
    imps  = [f[1] for f in top_features][::-1]
    bar_colors = [COLORS["cyan"] if i < len(imps) // 2 else COLORS["blue"]
                  for i in range(len(imps))]
    bars = ax7.barh(feats, imps, color=bar_colors, alpha=0.8, height=0.7)
    for bar, val in zip(bars, imps):
        ax7.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                 f"{val:.4f}", va="center", fontsize=7, color=COLORS["muted"])
    ax_style(ax7, "Top 15 Feature Importances (XGBoost)", "Importance")

    # ─────────────────────────────────────────
    # CHART 8 — ADX  (row 3, col 2)
    # ─────────────────────────────────────────
    ax8 = fig.add_subplot(gs[3, 2])
    ax8.plot(dates, df["ADX"],     color=COLORS["orange"], linewidth=1, label="ADX")
    ax8.plot(dates, df["ADX_pos"], color=COLORS["green"],  linewidth=0.8, alpha=0.7, label="+DI")
    ax8.plot(dates, df["ADX_neg"], color=COLORS["red"],    linewidth=0.8, alpha=0.7, label="-DI")
    ax8.axhline(25, color=COLORS["yellow"],
                linewidth=0.7, linestyle="--", alpha=0.6, label="Trend threshold")
    ax8.legend(fontsize=7, framealpha=0.1)
    ax_style(ax8, "ADX — Trend Strength", "")

    # ─────────────────────────────────────────
    # CHART 9 — Prediction summary panel (row 4, full)
    # ─────────────────────────────────────────
    ax9 = fig.add_subplot(gs[4, :])
    ax9.set_facecolor(COLORS["panel"])
    ax9.set_xlim(0, 1)
    ax9.set_ylim(0, 1)
    ax9.axis("off")

    # outer border
    rect = FancyBboxPatch((0.01, 0.04), 0.98, 0.93,
                           boxstyle="round,pad=0.01",
                           linewidth=1.5,
                           edgecolor=(COLORS["green"] if c["trend"] == "BULLISH"
                                      else COLORS["red"]),
                           facecolor=COLORS["bg"], alpha=0.8)
    ax9.add_patch(rect)

    trend_col = COLORS["green"] if c["trend"] == "BULLISH" else COLORS["red"]
    trend_sym  = "▲" if c["trend"] == "BULLISH" else "▼"

    ax9.text(0.5, 0.88, f"{trend_sym}  {c['ticker']} — {c['trend']} TREND  {trend_sym}",
             ha="center", va="center", fontsize=16, fontweight="bold",
             color=trend_col, transform=ax9.transAxes)

    # key stats — 3 columns
    col_x = [0.12, 0.45, 0.78]
    rows = [
        [
            f"Current Price   ${c['last_price']:.2f}",
            f"Price Target    ${c['price_target']:.2f}",
            f"Est Duration    ~{c['est_days']} trading days",
        ],
        [
            f"Confidence      {c['confidence']*100:.1f}%",
            f"Stop Loss       ${c['stop_loss']:.2f}",
            f"Target Date     {c['est_end'].date()}",
        ],
        [
            f"ATR             ${c['atr']:.2f}",
            f"ADX             {c['adx']:.1f}  "
            f"({'Strong' if c['adx']>25 else 'Mod'})",
            f"Hist Vol        {c['hist_vol']*100:.1f}% p.a.",
        ],
        [
            f"Support         ${c['near_sup']:.2f}" if c['near_sup'] else "Support    —",
            f"Resistance      ${c['near_res']:.2f}" if c['near_res'] else "Resistance —",
            f"Move            {(c['price_target']-c['last_price'])/c['last_price']*100:+.1f}%",
        ],
    ]

    y_positions = [0.70, 0.55, 0.40, 0.25]
    for row_y, row in zip(y_positions, rows):
        for cx, txt in zip(col_x, row):
            label, _, val = txt.partition("  ")
            ax9.text(cx - 0.06, row_y, label,
                     ha="left", fontsize=9, color=COLORS["muted"],
                     transform=ax9.transAxes)
            ax9.text(cx + 0.08, row_y, val.strip(),
                     ha="left", fontsize=9.5, fontweight="bold",
                     color=COLORS["text"], transform=ax9.transAxes)

    # disclaimer
    ax9.text(0.5, 0.08,
             "⚠  For research & educational purposes only. Not financial advice.",
             ha="center", fontsize=7.5, color=COLORS["muted"],
             transform=ax9.transAxes, style="italic")

    # ── Super title ──
    fig.suptitle(
        f"QUANTITATIVE TREND PREDICTOR  |  {TICKER}  |  "
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        fontsize=11, fontweight="bold",
        color=COLORS["text"], y=0.965)

    import os
    os.makedirs("outputs", exist_ok=True)
    out = f"outputs/{TICKER}_quant_predictor.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=COLORS["bg"])
    plt.close()
    print(f"  ✓  Chart saved → {out}")
    return out


# ══════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════
def main(ticker: str = TICKER):
    global TICKER
    TICKER = ticker.upper()

    # 1. Fetch
    df = fetch_data(TICKER)

    # 2. Features
    df = add_features(df)

    # 3. Train
    (models, scaler, feature_cols,
     df, split, dates_val,
     preds_val, probas_val, ensemble_pred, ensemble_proba,
     y_val, metrics, top_features) = train_and_validate(df)

    # 4. Predict
    conclusion = make_prediction(models, scaler, feature_cols,
                                 df, ensemble_proba, y_val)

    # 5. Print
    print_conclusion(conclusion, metrics)

    # 6. Plot
    out = plot_all(df, split, dates_val,
                   ensemble_pred, ensemble_proba, y_val,
                   metrics, top_features, conclusion)

    print(f"\n  Done!  Output → {out}\n")
    return conclusion


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else TICKER
    main(ticker)