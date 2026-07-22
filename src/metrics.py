"""
metrics — Vectorized business-metric calculations.

Provides:
    • ``compute_churn_rate``  — 60-day rolling-window churn classification.
    • ``compute_aov``         — Average Order Value (overall, by category, monthly).
    • ``compute_ltv``         — Customer Lifetime Value with quartile segmentation.
"""

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# 1. CHURN RATE
# ─────────────────────────────────────────────────────────────────────────────

def compute_churn_rate(
    df: pd.DataFrame,
    window_days: int = 60,
) -> dict:
    """Compute 60-day churn rate and per-category churn breakdown.

    A customer is *churned* when their most recent purchase is more than
    ``window_days`` before the dataset's latest transaction date.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned data with ``CustomerID``, ``InvoiceDate``, ``Category``.
    window_days : int, default 60
        Inactivity threshold in days.

    Returns
    -------
    dict
        Keys: ``total_customers``, ``active_customers``,
        ``churned_customers``, ``churn_rate``, ``category_churn``.
    """
    analysis_date = df["InvoiceDate"].max()
    cutoff = analysis_date - pd.Timedelta(days=window_days)

    # Vectorised: one groupby to get each customer's last purchase
    last_purchase = df.groupby("CustomerID")["InvoiceDate"].max()
    is_churned = last_purchase < cutoff

    total = len(last_purchase)
    churned = int(is_churned.sum())
    active = total - churned
    rate = (churned / total * 100) if total else 0.0

    # Per-category churn: merge flag back, then group by Category
    churn_flag = is_churned.rename("IsChurned").reset_index()
    merged = df.merge(churn_flag, on="CustomerID")
    category_churn = (
        merged
        .groupby("Category")["IsChurned"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
    )

    return {
        "total_customers": total,
        "active_customers": active,
        "churned_customers": churned,
        "churn_rate": round(rate, 2),
        "category_churn": category_churn,
        "analysis_date": analysis_date,
        "cutoff_date": cutoff,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 2. AVERAGE ORDER VALUE
# ─────────────────────────────────────────────────────────────────────────────

def compute_aov(df: pd.DataFrame) -> dict:
    """Compute Average Order Value overall, by category, and monthly.

    Each row in the cleaned dataset represents an invoice line-item.
    We aggregate at the **invoice level** (``InvoiceNo``) so AOV
    reflects the true per-order value.

    Returns
    -------
    dict
        Keys: ``total_revenue``, ``total_orders``, ``overall_aov``,
        ``aov_by_category`` (DataFrame), ``monthly_aov`` (Series).
    """
    # Invoice-level totals for true AOV
    invoice_totals = (
        df.groupby("InvoiceNo")["TotalPrice"]
        .sum()
    )
    total_revenue = float(invoice_totals.sum())
    total_orders = len(invoice_totals)
    overall_aov = total_revenue / total_orders if total_orders else 0.0

    # Category-level AOV (per line-item, for category comparison)
    aov_by_category = (
        df.groupby("Category")["TotalPrice"]
        .agg(["mean", "median", "count", "sum"])
        .rename(columns={
            "mean": "AOV_Mean",
            "median": "AOV_Median",
            "count": "Orders",
            "sum": "Revenue",
        })
        .sort_values("Revenue", ascending=False)
    )

    # Monthly AOV trend (invoice-level)
    df_m = df.copy()
    df_m["Month"] = df_m["InvoiceDate"].dt.to_period("M")
    monthly_aov = (
        df_m.groupby(["Month", "InvoiceNo"])["TotalPrice"]
        .sum()
        .groupby("Month")
        .mean()
    )

    return {
        "total_revenue": round(total_revenue, 2),
        "total_orders": total_orders,
        "overall_aov": round(overall_aov, 2),
        "aov_by_category": aov_by_category,
        "monthly_aov": monthly_aov,
    }


# ─────────────────────────────────────────────────────────────────────────────
# 3. CUSTOMER LIFETIME VALUE
# ─────────────────────────────────────────────────────────────────────────────

def compute_ltv(df: pd.DataFrame) -> dict:
    """Compute Customer Lifetime Value with quartile segmentation.

    Formula::

        LTV = AOV × Purchase Frequency × Customer Lifespan (years)

    Customers with a single purchase are assigned a floor lifespan of
    30 days (~1 month).

    Returns
    -------
    dict
        Keys: ``avg_aov``, ``avg_frequency``, ``avg_lifespan_years``,
        ``overall_ltv``, ``median_ltv``, ``customer_stats`` (DataFrame),
        ``segment_summary`` (DataFrame).
    """
    cust = df.groupby("CustomerID").agg(
        total_spent=("TotalPrice", "sum"),
        order_count=("InvoiceNo", "nunique"),
        avg_order_value=("TotalPrice", "mean"),
        first_purchase=("InvoiceDate", "min"),
        last_purchase=("InvoiceDate", "max"),
    )

    # Lifespan in years with a 30-day floor
    cust["lifespan_days"] = (
        (cust["last_purchase"] - cust["first_purchase"]).dt.days
    )
    cust["lifespan_years"] = np.maximum(
        cust["lifespan_days"] / 365.25,
        30 / 365.25,
    )

    # LTV = AOV × frequency × lifespan
    cust["LTV"] = (
        cust["avg_order_value"]
        * cust["order_count"]
        * cust["lifespan_years"]
    )

    avg_aov = float(cust["avg_order_value"].mean())
    avg_freq = float(cust["order_count"].mean())
    avg_lifespan = float(cust["lifespan_years"].mean())
    overall_ltv = float(cust["LTV"].mean())
    median_ltv = float(cust["LTV"].median())

    # Quartile segmentation
    cust["LTV_Segment"] = pd.qcut(
        cust["LTV"], q=4,
        labels=["Low", "Medium", "High", "Premium"],
    )
    segment_summary = cust.groupby("LTV_Segment", observed=True).agg(
        customers=("LTV", "count"),
        avg_ltv=("LTV", "mean"),
        total_revenue=("total_spent", "sum"),
        avg_orders=("order_count", "mean"),
    )

    return {
        "avg_aov": round(avg_aov, 2),
        "avg_frequency": round(avg_freq, 2),
        "avg_lifespan_years": round(avg_lifespan, 2),
        "overall_ltv": round(overall_ltv, 2),
        "median_ltv": round(median_ltv, 2),
        "customer_stats": cust,
        "segment_summary": segment_summary,
    }
