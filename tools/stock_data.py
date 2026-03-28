import yfinance as yf
import pandas as pd
from langchain_core.tools import tool
from typing import Optional


def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, float("inf"))
    rsi   = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2)


def _analyze_stock_internal(ticker: str, period: str = "3mo") -> dict:
    df = yf.download(ticker, period=period, progress=False)

    if df.empty:
        return {"error": f"No data found for ticker '{ticker}'."}

    close       = df["Close"].squeeze()
    first_price = float(close.iloc[0])
    last_price  = float(close.iloc[-1])
    returns     = close.pct_change().dropna()

    return_pct   = round(((last_price - first_price) / first_price) * 100, 2)
    volatility   = round(float(returns.std()) * 100, 2)
    rsi          = _compute_rsi(close)
    trend        = "Bullish" if return_pct > 0 else "Bearish"

    recent = df.tail(5)[["Open", "High", "Low", "Close", "Volume"]].round(2)

    return {
        "ticker":         ticker,
        "period":         period,
        "current_price":  round(last_price, 2),
        "return_pct":     return_pct,
        "volatility_pct": volatility,
        "rsi":            rsi,
        "trend":          trend,
        "data_points":    len(df),
        "recent_ohlcv":   recent.to_dict(orient="records")
    }


@tool
def get_stock_data(ticker: str, period: Optional[str] = "3mo") -> dict:
    """
    Fetch live stock market data and compute key financial metrics.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g., 'AAPL', 'RELIANCE.NS', 'TCS.NS').
    period : str, optional
        Time period for historical data.
        Valid values: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y'.
        Default: '3mo'.

    Returns
    -------
    dict
        {
            "ticker":         "AAPL",
            "period":         "3mo",
            "current_price":  213.49,
            "return_pct":     8.34,       # % return over the period
            "volatility_pct": 1.82,       # daily volatility %
            "rsi":            58.3,        # 14-day RSI
            "trend":          "Bullish",
            "data_points":    63,
            "recent_ohlcv":   [...]        # last 5 trading days
        }
    """
    return _analyze_stock_internal(ticker, period)
