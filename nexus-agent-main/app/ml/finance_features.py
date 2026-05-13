from __future__ import annotations

import numpy as np
import pandas as pd
import yfinance as yf

TRADING_DAYS_PER_YEAR = 252


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def load_ohlcv(ticker: str, period: str = "6mo") -> pd.DataFrame:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=period, auto_adjust=True)
        if df is None or df.empty:
            return pd.DataFrame()
        # Flatten MultiIndex columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]
        # Normalize column names
        df.columns = [str(c).strip() for c in df.columns]
        required = ["Open", "High", "Low", "Close", "Volume"]
        available = [c for c in required if c in df.columns]
        return df[available].dropna()
    except Exception:
        return pd.DataFrame()


def build_finance_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    close = df["Close"].astype(float)
    high = df["High"].astype(float)
    low = df["Low"].astype(float)

    frame = df.copy()
    frame["return_1d"] = close.pct_change()
    frame["rolling_volatility_20d"] = (
        frame["return_1d"].rolling(20).std() * np.sqrt(TRADING_DAYS_PER_YEAR)
    )
    frame["ma_20"] = close.rolling(20).mean()
    frame["ma_50"] = close.rolling(50).mean()
    frame["rsi_14"] = _rsi(close)
    # ATR
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs(),
    ], axis=1).max(axis=1)
    frame["atr_14"] = tr.rolling(14).mean()
    frame["trend_signal"] = np.where(
        close > frame["ma_20"], "Bullish", "Bearish"
    )
    return frame.dropna()


def summarize_finance_features(ticker: str, period: str = "6mo") -> dict:
    df = load_ohlcv(ticker=ticker, period=period)
    if df.empty:
        return {"error": f"No data found for ticker '{ticker}'."}

    feature_frame = build_finance_feature_frame(df)
    if feature_frame.empty:
        return {"error": "Not enough data points to compute rolling indicators."}

    last = feature_frame.iloc[-1]
    close_series = df["Close"].astype(float)
    current_price = float(close_series.iloc[-1])
    prev_price = float(close_series.iloc[-2]) if len(close_series) > 1 else current_price
    daily_change_pct = round((current_price - prev_price) / prev_price * 100, 2)

    return {
        "feature_frame": feature_frame,
        "current_price": current_price,
        "daily_change_pct": daily_change_pct,
        "rsi_14": round(float(last["rsi_14"]), 2) if not pd.isna(last["rsi_14"]) else None,
        "rolling_volatility": round(float(last["rolling_volatility_20d"]), 4) if not pd.isna(last["rolling_volatility_20d"]) else None,
        "ma_20": round(float(last["ma_20"]), 2) if not pd.isna(last["ma_20"]) else None,
        "ma_50": round(float(last["ma_50"]), 2) if not pd.isna(last["ma_50"]) else None,
        "atr_14": round(float(last["atr_14"]), 4) if not pd.isna(last["atr_14"]) else None,
        "trend_signal": str(last["trend_signal"]),
        "close_prices": close_series.tolist(),
        "dates": [str(d.date()) for d in feature_frame.index],
    }
