#!/usr/bin/env python3
"""
main.py — CLI entry point for E-Commerce Customer Retention & Churn Analysis.

Usage::

    python main.py               # Run on 5K-row sample dataset
    python main.py --full-data   # Run on full 541K-row dataset
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Ensure project root is on sys.path so ``src`` is importable.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.data_loader import load_data
from src.cleaner import clean_data
from src.metrics import compute_churn_rate, compute_aov, compute_ltv
from src.visualization import (
    plot_category_churn,
    plot_ltv_distribution,
    plot_monthly_aov,
)

_ASSETS_DIR = _ROOT / "assets" / "images"


def _print_executive_summary(churn: dict, aov: dict, ltv: dict) -> None:
    """Print a formatted executive summary to the terminal."""
    print("\n")
    print("╔" + "═" * 68 + "╗")
    print("║" + "  📊  EXECUTIVE SUMMARY — E-COMMERCE ANALYTICS".center(68) + "║")
    print("╠" + "═" * 68 + "╣")
    print("║" + "".center(68) + "║")

    line1 = (f"  🔄 Churn Rate        :  {churn['churn_rate']:.2f}%"
             f"  ({churn['churned_customers']:,d} of "
             f"{churn['total_customers']:,d} customers)")
    print("║" + line1.ljust(68) + "║")

    line2 = f"  💰 Avg Order Value   :  ${aov['overall_aov']:,.2f}"
    print("║" + line2.ljust(68) + "║")

    line3 = (f"  👥 Customer LTV      :  ${ltv['overall_ltv']:,.2f}"
             f"  (median: ${ltv['median_ltv']:,.2f})")
    print("║" + line3.ljust(68) + "║")

    print("║" + "".center(68) + "║")

    line4 = f"  📦 Total Orders      :  {aov['total_orders']:,d}"
    print("║" + line4.ljust(68) + "║")

    line5 = f"  💵 Total Revenue     :  ${aov['total_revenue']:,.2f}"
    print("║" + line5.ljust(68) + "║")

    line6 = f"  🟢 Active Customers  :  {churn['active_customers']:,d}"
    print("║" + line6.ljust(68) + "║")

    line7 = f"  🔴 Churned Customers :  {churn['churned_customers']:,d}"
    print("║" + line7.ljust(68) + "║")

    print("║" + "".center(68) + "║")
    print("╚" + "═" * 68 + "╝")


def _print_metric_details(churn: dict, aov: dict, ltv: dict) -> None:
    """Print detailed metric breakdowns to the terminal."""

    # ── Churn details ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  METRIC 1 — CHURN RATE (60-Day Window)")
    print("=" * 60)
    print(f"  Analysis date : {churn['analysis_date'].date()}")
    print(f"  Cutoff date   : {churn['cutoff_date'].date()}")
    print(f"  Total         : {churn['total_customers']:>8,d}")
    print(f"  Active        : {churn['active_customers']:>8,d}")
    print(f"  Churned       : {churn['churned_customers']:>8,d}")
    print(f"  ► Churn Rate  : {churn['churn_rate']:>7.2f}%")

    print("\n  Churn Rate by Category:")
    for cat, rate in churn["category_churn"].head(10).items():
        bar = "█" * int(rate / 3)
        print(f"    {cat:<28s} {rate:5.1f}%  {bar}")

    # ── AOV details ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  METRIC 2 — AVERAGE ORDER VALUE (AOV)")
    print("=" * 60)
    print(f"  Total Revenue   : ${aov['total_revenue']:>12,.2f}")
    print(f"  Total Orders    : {aov['total_orders']:>12,d}")
    print(f"  ► Overall AOV   : ${aov['overall_aov']:>12,.2f}")

    print("\n  AOV by Category (top 10 by revenue):")
    cat_df = aov["aov_by_category"].head(10)
    print(f"    {'Category':<28s} {'AOV':>9s}  {'Orders':>7s}  {'Revenue':>12s}")
    print(f"    {'─' * 60}")
    for cat, row in cat_df.iterrows():
        print(f"    {cat:<28s} ${row['AOV_Mean']:>8,.2f}"
              f"  {int(row['Orders']):>7,d}"
              f"  ${row['Revenue']:>11,.2f}")

    print("\n  Monthly AOV Trend:")
    for month, val in aov["monthly_aov"].items():
        bar = "▓" * int(val / 3)
        print(f"    {str(month):<10s}  ${val:>8,.2f}  {bar}")

    # ── LTV details ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  METRIC 3 — CUSTOMER LIFETIME VALUE (LTV)")
    print("=" * 60)
    print(f"  Avg AOV              : ${ltv['avg_aov']:>10,.2f}")
    print(f"  Avg Frequency        : {ltv['avg_frequency']:>10,.1f} orders")
    print(f"  Avg Lifespan         : {ltv['avg_lifespan_years']:>10,.2f} years")
    print(f"  ► Mean LTV           : ${ltv['overall_ltv']:>10,.2f}")
    print(f"  ► Median LTV         : ${ltv['median_ltv']:>10,.2f}")

    print("\n  LTV Segments:")
    seg = ltv["segment_summary"]
    print(f"    {'Segment':<10s}  {'Customers':>10s}  {'Avg LTV':>10s}"
          f"  {'Revenue':>12s}  {'Avg Orders':>10s}")
    print(f"    {'─' * 58}")
    for s, row in seg.iterrows():
        print(f"    {s:<10s}  {int(row['customers']):>10,d}"
              f"  ${row['avg_ltv']:>9,.2f}"
              f"  ${row['total_revenue']:>11,.2f}"
              f"  {row['avg_orders']:>10.1f}")

    # Top 10 customers
    top10 = ltv["customer_stats"].nlargest(10, "LTV")[
        ["total_spent", "order_count", "avg_order_value", "lifespan_years", "LTV"]
    ]
    print("\n  Top 10 Most Valuable Customers:")
    print(f"    {'CustomerID':<12s}  {'Spent':>11s}  {'Orders':>7s}"
          f"  {'AOV':>9s}  {'Lifespan':>9s}  {'LTV':>12s}")
    print(f"    {'─' * 66}")
    for cid, row in top10.iterrows():
        print(f"    {cid:<12s}  ${row['total_spent']:>10,.2f}"
              f"  {int(row['order_count']):>7,d}"
              f"  ${row['avg_order_value']:>8,.2f}"
              f"  {row['lifespan_years']:>8.2f}y"
              f"  ${row['LTV']:>11,.2f}")


def main() -> None:
    """Run the full analysis pipeline."""
    parser = argparse.ArgumentParser(
        description="E-Commerce Customer Retention & Churn Analysis",
    )
    parser.add_argument(
        "--full-data",
        action="store_true",
        help="Use the full 541K-row dataset instead of the 5K sample.",
    )
    args = parser.parse_args()

    # ── 1. Load ──────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  STEP 1 — LOADING DATA")
    print("=" * 60)
    raw = load_data(full=args.full_data)

    # ── 2. Clean ─────────────────────────────────────────────────────────
    df = clean_data(raw)

    # ── 3. Metrics ───────────────────────────────────────────────────────
    churn = compute_churn_rate(df, window_days=60)
    aov = compute_aov(df)
    ltv = compute_ltv(df)

    # ── 4. Detailed output ───────────────────────────────────────────────
    _print_metric_details(churn, aov, ltv)

    # ── 5. Visualizations ────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  GENERATING VISUALIZATIONS")
    print("=" * 60)
    plot_category_churn(churn, save_dir=_ASSETS_DIR)
    plot_ltv_distribution(ltv, save_dir=_ASSETS_DIR)
    plot_monthly_aov(aov, save_dir=_ASSETS_DIR)

    # ── 6. Executive summary ─────────────────────────────────────────────
    _print_executive_summary(churn, aov, ltv)
    print("\n  ✅ Analysis complete. Charts saved to assets/images/\n")


if __name__ == "__main__":
    main()
