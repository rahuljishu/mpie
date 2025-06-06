# utils/loader.py
import os, csv, pandas as pd
from pathlib import Path

# ------------------------------------------------------------
def _sniff_delimiter(sample: str) -> str:
    """Guess delimiter from a short text sample."""
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except csv.Error:
        common = [",", ";", "\t", "|", " "]
        counts = {d: sample.count(d) for d in common}
        return max(counts, key=counts.get)

# ------------------------------------------------------------
def _detect_encoding(raw: bytes) -> str:
    """
    Try UTF-8 first, then fall back to latin-1 (never fails).
    You can install 'charset-normalizer' for smarter detection,
    but this two-step approach solves most public datasets.
    """
    try:
        raw.decode("utf-8")
        return "utf-8"
    except UnicodeDecodeError:
        return "latin1"          # single-byte fallback

# ------------------------------------------------------------
def load_table(path: str, sniff_lines: int = 20) -> pd.DataFrame:
    """
    Universal loader
      • Excel (xls, xlsx, xlsm)
      • CSV/TXT with unknown delimiter
      • Handles non-UTF-8 encodings
      • Treats '?' as NaN
    """
    ext = Path(path).suffix.lower()
    if ext in {".xls", ".xlsx", ".xlsm"}:
        return pd.read_excel(path)

    # read a header block as bytes for encoding + delimiter sniff
    with open(path, "rb") as fh:
        head = fh.read(64_000)                # 64 KB is plenty

    encoding = _detect_encoding(head)
    text     = head.decode(encoding, errors="ignore")
    delim    = _sniff_delimiter("\n".join(text.splitlines()[:sniff_lines]))

    return pd.read_csv(path,
                       sep=delim,
                       na_values=["?"],
                       low_memory=False,
                       encoding=encoding,
                       encoding_errors="ignore")
