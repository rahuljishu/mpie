# utils/cleaner.py
import re
import warnings
import numpy as np
import pandas as pd

_dollar_re  = re.compile(r"^\$?\s*([-\d.,]+)$")
_percent_re = re.compile(r"^([-\d.,]+)%$")
_comma_re   = re.compile(r",")

def _coerce_numeric_series(s: pd.Series) -> pd.Series:
    """Coerce $,%,comma strings → floats; if <80% valid, leave as-is."""
    if s.dtype != object:
        return s
    cleaned = s.str.strip()
    cleaned = cleaned.str.replace(_comma_re, "",    regex=True)
    cleaned = cleaned.str.replace(_dollar_re, r"\1", regex=True)
    cleaned = cleaned.str.replace(_percent_re, r"\1", regex=True)
    num = pd.to_numeric(cleaned, errors="coerce")
    return num if (num.notna().mean() > 0.8) else s

def _coerce_date_series(s: pd.Series, threshold: float = 0.8) -> pd.Series:
    """
    Try MDY vs DMY parsing (vectorized). Pick whichever ≥threshold parses.
    Otherwise leave the series untouched.
    """
    if s.dtype != object:
        return s

    # vectorized parses – silence pandas warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        mdy = pd.to_datetime(s, errors="coerce", dayfirst=False,
                             infer_datetime_format=True)
        dmy = pd.to_datetime(s, errors="coerce", dayfirst=True,
                             infer_datetime_format=True)
    r1, r2 = mdy.notna().mean(), dmy.notna().mean()

    if max(r1, r2) >= threshold:
        best = dmy if (r2 > r1) else mdy
        return best.astype("int64")   # nanoseconds since epoch
    return s

def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    ▸ Coerce $,%,comma strings → floats  
    ▸ Parse date-like strings → int64 ns (MDY or DMY)  
    ▸ Factorize any leftover categorical/text columns  
    ▸ Keep numeric cols, drop all-NaN columns  
    ▸ Down-scale giant ints (>1e12) → days  
    ▸ Replace ±inf with NaN, fill NaNs with medians  
    ▸ Cast to float32
    """
    df = df.copy()

    # 1) Coerce numeric & date strings
    for col in df.select_dtypes(include=["object"]):
        df[col] = _coerce_numeric_series(df[col])
        df[col] = _coerce_date_series(df[col])

    # 2) Factorize any remaining object/category columns
    for col in df.select_dtypes(include=["object", "category"]):
        df[col], _ = pd.factorize(df[col])

    # 3) Keep only numeric columns, drop all-NaN
    df = df.select_dtypes(include=[np.number]).dropna(axis=1, how="all")

    # 4) Down-scale huge ints → days
    for col in df.columns:
        if df[col].abs().max() > 1e12:
            df[col] = df[col] / 86_400_000_000_000.0

    # 5) Replace infs, fill NaNs with median, cast to float32
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(df.median()).astype(np.float32)

    return df
