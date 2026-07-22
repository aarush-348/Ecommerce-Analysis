"""
cleaner — Data-cleaning pipeline for UCI Online Retail transactions.

Handles:
    • Missing ``CustomerID`` rows (dropped — required for churn/LTV).
    • Cancellations (``InvoiceNo`` prefixed with ``C``).
    • Negative / zero ``Quantity`` and ``UnitPrice``.
    • Duplicate rows.
    • Derived ``TotalPrice`` and ``Category`` columns.
"""

import numpy as np
import pandas as pd

# ── Top product-description buckets used to derive a Category column ─────────
_CATEGORY_KEYWORDS = {
    "BAG": "Bags & Accessories",
    "CANDLE": "Candles & Holders",
    "HOLDER": "Candles & Holders",
    "T-LIGHT": "Candles & Holders",
    "LANTERN": "Candles & Holders",
    "CHRISTMAS": "Christmas & Seasonal",
    "XMAS": "Christmas & Seasonal",
    "VINTAGE": "Vintage & Retro",
    "RETRO": "Vintage & Retro",
    "HEART": "Home Décor",
    "SIGN": "Home Décor",
    "DECORATION": "Home Décor",
    "HANGING": "Home Décor",
    "FRAME": "Home Décor",
    "CLOCK": "Home Décor",
    "MUG": "Kitchen & Dining",
    "CAKE": "Kitchen & Dining",
    "PLATE": "Kitchen & Dining",
    "NAPKIN": "Kitchen & Dining",
    "BOTTLE": "Kitchen & Dining",
    "JAR": "Kitchen & Dining",
    "BOX": "Storage & Organisation",
    "BASKET": "Storage & Organisation",
    "TIN": "Storage & Organisation",
    "SET": "Gift Sets",
    "WRAP": "Stationery & Craft",
    "CARD": "Stationery & Craft",
    "PAPER": "Stationery & Craft",
    "TOY": "Toys & Games",
    "GAME": "Toys & Games",
    "GARDEN": "Garden & Outdoor",
}


def _derive_category(description: pd.Series) -> pd.Series:
    """Map product descriptions to broader categories via keyword matching."""
    upper = description.fillna("").str.upper()
    category = pd.Series("Other", index=description.index)

    # Iterate from most-specific to least; later matches do NOT overwrite.
    for keyword, cat in _CATEGORY_KEYWORDS.items():
        mask = upper.str.contains(keyword, na=False) & (category == "Other")
        category = category.where(~mask, cat)

    return category


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean raw transaction data for downstream analytics.

    Parameters
    ----------
    df : pd.DataFrame
        Raw output from :func:`src.data_loader.load_data`.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame with added ``TotalPrice`` and ``Category``
        columns, sorted chronologically.
    """
    print("\n" + "=" * 60)
    print("  DATA CLEANING")
    print("=" * 60)

    df_clean = df.copy()
    n_raw = len(df_clean)

    # ── 1. Drop rows without a CustomerID ────────────────────────────────
    n_null_cust = df_clean["CustomerID"].isnull().sum()
    df_clean = df_clean.dropna(subset=["CustomerID"])
    df_clean["CustomerID"] = df_clean["CustomerID"].astype(int).astype(str)
    print(f"  ✗ Dropped {n_null_cust:,} rows with missing CustomerID")

    # ── 2. Remove cancellations (InvoiceNo starting with 'C') ───────────
    cancel_mask = df_clean["InvoiceNo"].str.startswith("C", na=False)
    n_cancel = cancel_mask.sum()
    df_clean = df_clean[~cancel_mask]
    print(f"  ✗ Removed {n_cancel:,} cancellation rows")

    # ── 3. Filter out non-positive Quantity and UnitPrice ────────────────
    valid_mask = (df_clean["Quantity"] > 0) & (df_clean["UnitPrice"] > 0)
    n_invalid = (~valid_mask).sum()
    df_clean = df_clean[valid_mask]
    print(f"  ✗ Removed {n_invalid:,} rows with non-positive Qty/Price")

    # ── 4. Drop exact duplicates ─────────────────────────────────────────
    n_dupes = df_clean.duplicated().sum()
    df_clean = df_clean.drop_duplicates()
    print(f"  ✗ Removed {n_dupes:,} duplicate rows")

    # ── 5. Derive TotalPrice ─────────────────────────────────────────────
    df_clean["TotalPrice"] = np.round(
        df_clean["Quantity"] * df_clean["UnitPrice"], 2
    )

    # ── 6. Derive Category from Description ──────────────────────────────
    df_clean["Category"] = _derive_category(df_clean["Description"])
    n_cats = df_clean["Category"].nunique()
    print(f"  ✓ Derived {n_cats} product categories from Description")

    # ── 7. Ensure InvoiceDate is datetime ────────────────────────────────
    df_clean["InvoiceDate"] = pd.to_datetime(
        df_clean["InvoiceDate"], errors="coerce"
    )
    df_clean = df_clean.dropna(subset=["InvoiceDate"])

    # ── 8. Sort chronologically & reset index ────────────────────────────
    df_clean = (
        df_clean
        .sort_values("InvoiceDate")
        .reset_index(drop=True)
    )

    print(f"\n  ✓ Cleaning complete: {n_raw:,} → {len(df_clean):,} rows "
          f"({len(df_clean) / n_raw * 100:.1f}% retained)")

    return df_clean
