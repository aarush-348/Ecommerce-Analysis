"""
test_metrics — Pytest suite validating Churn, AOV, and LTV calculations.

Uses a small hand-crafted fixture DataFrame that mirrors the cleaned
UCI Online Retail schema so expected outputs can be computed by hand.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure ``src`` is importable.
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.metrics import compute_churn_rate, compute_aov, compute_ltv


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """Create a small deterministic DataFrame for testing.

    Layout (8 transactions, 3 customers):
        - Customer A: 3 purchases spread over 100 days  → active
        - Customer B: 2 purchases, last 90 days ago      → churned
        - Customer C: 3 purchases, last 80 days ago      → churned
    """
    latest = pd.Timestamp("2011-12-09")

    data = {
        "InvoiceNo": ["I001", "I002", "I003", "I004", "I005", "I006", "I007", "I008"],
        "StockCode": ["S1"] * 8,
        "Description": [
            "CANDLE HOLDER", "MUG SET", "CHRISTMAS CARD",
            "BAG VINTAGE", "TOY TRAIN",
            "GARDEN LANTERN", "PLATE SET", "FRAME OAK",
        ],
        "Quantity": [2, 1, 5, 3, 1, 2, 4, 1],
        "InvoiceDate": [
            latest - pd.Timedelta(days=100),  # A
            latest - pd.Timedelta(days=50),   # A
            latest,                            # A  (analysis date)
            latest - pd.Timedelta(days=90),   # B
            latest - pd.Timedelta(days=90),   # B
            latest - pd.Timedelta(days=80),   # C
            latest - pd.Timedelta(days=80),   # C
            latest - pd.Timedelta(days=80),   # C
        ],
        "UnitPrice": [5.0, 10.0, 2.0, 8.0, 15.0, 4.0, 3.0, 20.0],
        "CustomerID": ["A", "A", "A", "B", "B", "C", "C", "C"],
        "Country": ["United Kingdom"] * 8,
        "TotalPrice": [10.0, 10.0, 10.0, 24.0, 15.0, 8.0, 12.0, 20.0],
        "Category": [
            "Candles & Holders", "Kitchen & Dining", "Christmas & Seasonal",
            "Bags & Accessories", "Toys & Games",
            "Candles & Holders", "Gift Sets", "Home Décor",
        ],
    }
    return pd.DataFrame(data)


# ─────────────────────────────────────────────────────────────────────────────
# CHURN TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestChurnRate:
    """Validate 60-day churn classification."""

    def test_total_customers(self, sample_df):
        result = compute_churn_rate(sample_df, window_days=60)
        assert result["total_customers"] == 3

    def test_churned_count(self, sample_df):
        """B and C's last purchases are >60 days before analysis date."""
        result = compute_churn_rate(sample_df, window_days=60)
        assert result["churned_customers"] == 2

    def test_active_count(self, sample_df):
        """Only customer A purchased within the 60-day window."""
        result = compute_churn_rate(sample_df, window_days=60)
        assert result["active_customers"] == 1

    def test_churn_rate_value(self, sample_df):
        result = compute_churn_rate(sample_df, window_days=60)
        expected = round(2 / 3 * 100, 2)
        assert result["churn_rate"] == expected

    def test_all_active_with_large_window(self, sample_df):
        """With a 200-day window everyone should be active."""
        result = compute_churn_rate(sample_df, window_days=200)
        assert result["churned_customers"] == 0
        assert result["churn_rate"] == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# AOV TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestAOV:
    """Validate Average Order Value calculations."""

    def test_total_revenue(self, sample_df):
        result = compute_aov(sample_df)
        expected_revenue = sample_df["TotalPrice"].sum()
        assert result["total_revenue"] == round(expected_revenue, 2)

    def test_total_orders(self, sample_df):
        """Each unique InvoiceNo is one order."""
        result = compute_aov(sample_df)
        assert result["total_orders"] == sample_df["InvoiceNo"].nunique()

    def test_overall_aov(self, sample_df):
        result = compute_aov(sample_df)
        invoice_totals = sample_df.groupby("InvoiceNo")["TotalPrice"].sum()
        expected = round(float(invoice_totals.mean()), 2)
        assert result["overall_aov"] == expected

    def test_category_breakdown_exists(self, sample_df):
        result = compute_aov(sample_df)
        assert not result["aov_by_category"].empty


# ─────────────────────────────────────────────────────────────────────────────
# LTV TESTS
# ─────────────────────────────────────────────────────────────────────────────

class TestLTV:
    """Validate Customer Lifetime Value formula."""

    def test_ltv_components(self, sample_df):
        result = compute_ltv(sample_df)
        assert result["avg_aov"] > 0
        assert result["avg_frequency"] > 0
        assert result["avg_lifespan_years"] > 0

    def test_ltv_formula_customer_a(self, sample_df):
        """Manually verify LTV for customer A.

        A: total_spent=30, orders=3 (unique invoices), avg=10,
           lifespan=100 days = 0.2738 years
           LTV = 10 * 3 * 0.2738 ≈ 8.21
        """
        result = compute_ltv(sample_df)
        cust_a = result["customer_stats"].loc["A"]
        lifespan = 100 / 365.25
        expected_ltv = cust_a["avg_order_value"] * cust_a["order_count"] * lifespan
        assert abs(cust_a["LTV"] - expected_ltv) < 0.01

    def test_lifespan_floor(self, sample_df):
        """Customers with 0-day lifespan should get the 30-day floor."""
        # Create a customer with a single purchase
        single = sample_df.iloc[:1].copy()
        single["CustomerID"] = "Z"
        df = pd.concat([sample_df, single], ignore_index=True)
        result = compute_ltv(df)
        z_lifespan = result["customer_stats"].loc["Z", "lifespan_years"]
        assert z_lifespan == pytest.approx(30 / 365.25, rel=1e-3)

    def test_segment_labels(self, sample_df):
        result = compute_ltv(sample_df)
        segments = set(result["customer_stats"]["LTV_Segment"].dropna().unique())
        assert segments.issubset({"Low", "Medium", "High", "Premium"})

    def test_overall_ltv_positive(self, sample_df):
        result = compute_ltv(sample_df)
        assert result["overall_ltv"] > 0
        assert result["median_ltv"] > 0
