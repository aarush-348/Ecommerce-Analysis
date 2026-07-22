"""
data_loader — Load e-commerce transaction data from local CSV files.

Supports two modes:
    • Full dataset  (~541K rows) — ``data/online_retail_500k.csv``
    • Sample dataset (~5K rows)  — ``data/sample_transactions.csv``
"""

from pathlib import Path

import pandas as pd

# ── File paths (relative to project root) ────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_FULL_DATA_PATH = _PROJECT_ROOT / "data" / "online_retail_500k.csv"
_SAMPLE_DATA_PATH = _PROJECT_ROOT / "data" / "sample_transactions.csv"


def load_data(full: bool = False) -> pd.DataFrame:
    """Load the e-commerce transaction dataset.

    Parameters
    ----------
    full : bool, default False
        If *True*, load the full ~541K-row dataset.  Otherwise load the
        5K-row sample.  Falls back to the sample if the full file is
        missing.

    Returns
    -------
    pd.DataFrame
        Raw, uncleaned transaction data with columns:
        ``InvoiceNo``, ``StockCode``, ``Description``, ``Quantity``,
        ``InvoiceDate``, ``UnitPrice``, ``CustomerID``, ``Country``.
    """
    if full and _FULL_DATA_PATH.exists():
        path = _FULL_DATA_PATH
        print(f"  Loading FULL dataset: {path.name}")
    else:
        if full:
            print(f"  ⚠  Full dataset not found — falling back to sample.")
        path = _SAMPLE_DATA_PATH
        print(f"  Loading SAMPLE dataset: {path.name}")

    df = pd.read_csv(
        path,
        encoding="latin-1",
        dtype={
            "InvoiceNo": str,
            "StockCode": str,
            "Description": str,
            "Country": str,
        },
        parse_dates=["InvoiceDate"],
        dayfirst=False,
    )

    print(f"  ✓ Loaded {len(df):,} rows × {df.shape[1]} columns")
    return df
