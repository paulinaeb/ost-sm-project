# analytics/forecasting.py
import pandas as pd

def naive_forecast(daily_counts: pd.DataFrame, periods: int = 7) -> pd.DataFrame:
    """
    Baseline forecast: extend the last 7-day average forward.
    Input: DataFrame with ['date','jobs']
    Output: DataFrame with ['date','yhat']
    """
    if daily_counts.empty:
        return pd.DataFrame(columns=["date","yhat"])

    s = (daily_counts.set_index("date")
                        .asfreq("D")["jobs"]
                        .fillna(0)
                        .sort_index())
    window = min(7, len(s))
    avg = s.tail(window).mean() if window else 0.0
    last_day = s.index.max() if len(s) else pd.Timestamp.today().normalize()
    future_idx = pd.date_range(last_day + pd.Timedelta(days=1),
                               periods=periods, freq="D")
    return pd.DataFrame({"date": future_idx, "yhat": [avg]*periods})
