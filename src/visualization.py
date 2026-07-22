"""
visualization — Polished Seaborn chart generation for e-commerce analytics.

Generates three publication-ready charts:
    1. Category Churn Risk & Drop-off
    2. Customer LTV & Spend Distribution
    3. Monthly AOV Trend Line
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless environments

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd

# ── Global Seaborn theme ─────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.1)
_PALETTE = sns.color_palette("mako", 12)
_ACCENT = "#e63946"
_SAVE_DPI = 180


def _ensure_dir(path: Path) -> Path:
    """Create the output directory if it doesn't exist."""
    path.mkdir(parents=True, exist_ok=True)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 1. CATEGORY CHURN RISK
# ─────────────────────────────────────────────────────────────────────────────

def plot_category_churn(
    churn_metrics: dict,
    save_dir: str | Path = "assets/images",
) -> Path:
    """Horizontal bar chart of churn rate by product category.

    Parameters
    ----------
    churn_metrics : dict
        Output from :func:`src.metrics.compute_churn_rate`.
    save_dir : str or Path
        Directory to save the chart PNG.

    Returns
    -------
    Path
        Path to the saved image.
    """
    save_dir = _ensure_dir(Path(save_dir))
    cat_churn: pd.Series = churn_metrics["category_churn"].head(15)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("flare", n_colors=len(cat_churn))

    bars = ax.barh(
        cat_churn.index[::-1],
        cat_churn.values[::-1],
        color=colors[::-1],
        edgecolor="white",
        linewidth=0.5,
    )

    # Annotate bars with percentage values
    for bar, val in zip(bars, cat_churn.values[::-1]):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%",
            va="center",
            fontsize=9,
            fontweight="bold",
            color="#333",
        )

    ax.set_xlabel("Churn Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Category Churn Risk & Drop-off (60-Day Window)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.xaxis.set_major_formatter(ticker.PercentFormatter())
    ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    out = save_dir / "category_churn_risk.png"
    fig.savefig(out, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2. LTV DISTRIBUTION
# ─────────────────────────────────────────────────────────────────────────────

def plot_ltv_distribution(
    ltv_metrics: dict,
    save_dir: str | Path = "assets/images",
) -> Path:
    """Histogram + box plot of Customer LTV by quartile segment.

    Parameters
    ----------
    ltv_metrics : dict
        Output from :func:`src.metrics.compute_ltv`.
    save_dir : str or Path
        Directory to save the chart PNG.

    Returns
    -------
    Path
        Path to the saved image.
    """
    save_dir = _ensure_dir(Path(save_dir))
    cust = ltv_metrics["customer_stats"].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [3, 2]})

    # ── Left: histogram ──────────────────────────────────────────────────
    ax_hist = axes[0]
    # Cap at 99th percentile for readability
    cap = cust["LTV"].quantile(0.99)
    plot_data = cust["LTV"].clip(upper=cap)

    sns.histplot(
        plot_data,
        bins=50,
        kde=True,
        color=_PALETTE[3],
        edgecolor="white",
        linewidth=0.4,
        ax=ax_hist,
    )
    ax_hist.axvline(
        ltv_metrics["median_ltv"],
        color=_ACCENT,
        ls="--",
        lw=2,
        label=f"Median: ${ltv_metrics['median_ltv']:,.0f}",
    )
    ax_hist.set_xlabel("Customer LTV ($)", fontweight="bold")
    ax_hist.set_ylabel("Number of Customers", fontweight="bold")
    ax_hist.set_title("Customer LTV Distribution", fontsize=13, fontweight="bold")
    ax_hist.legend(fontsize=10)

    # ── Right: box plot by segment ───────────────────────────────────────
    ax_box = axes[1]
    segment_order = ["Low", "Medium", "High", "Premium"]
    cust_plot = cust[cust["LTV"] <= cap].copy()
    sns.boxplot(
        data=cust_plot,
        x="LTV_Segment",
        y="LTV",
        order=segment_order,
        palette="mako",
        ax=ax_box,
        fliersize=2,
    )
    ax_box.set_xlabel("LTV Segment", fontweight="bold")
    ax_box.set_ylabel("LTV ($)", fontweight="bold")
    ax_box.set_title("LTV by Quartile Segment", fontsize=13, fontweight="bold")

    plt.tight_layout()
    out = save_dir / "ltv_distribution.png"
    fig.savefig(out, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 3. MONTHLY AOV TREND
# ─────────────────────────────────────────────────────────────────────────────

def plot_monthly_aov(
    aov_metrics: dict,
    save_dir: str | Path = "assets/images",
) -> Path:
    """Line chart of monthly Average Order Value with trend annotation.

    Parameters
    ----------
    aov_metrics : dict
        Output from :func:`src.metrics.compute_aov`.
    save_dir : str or Path
        Directory to save the chart PNG.

    Returns
    -------
    Path
        Path to the saved image.
    """
    save_dir = _ensure_dir(Path(save_dir))
    monthly: pd.Series = aov_metrics["monthly_aov"]

    # Convert Period index to timestamps for plotting
    x = monthly.index.to_timestamp()
    y = monthly.values

    fig, ax = plt.subplots(figsize=(11, 5))

    ax.plot(
        x, y,
        marker="o",
        markersize=6,
        linewidth=2.5,
        color=_PALETTE[4],
        markeredgecolor="white",
        markeredgewidth=1.2,
        zorder=3,
    )
    ax.fill_between(x, y, alpha=0.15, color=_PALETTE[4])

    # Overall AOV reference line
    overall = aov_metrics["overall_aov"]
    ax.axhline(
        overall,
        color=_ACCENT,
        ls="--",
        lw=1.5,
        label=f"Overall AOV: ${overall:,.2f}",
    )

    ax.set_xlabel("Month", fontsize=12, fontweight="bold")
    ax.set_ylabel("Average Order Value ($)", fontsize=12, fontweight="bold")
    ax.set_title(
        "Monthly Average Order Value Trend",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    ax.legend(fontsize=10, loc="upper left")
    ax.yaxis.set_major_formatter(ticker.StrMethodFormatter("${x:,.0f}"))
    ax.spines[["top", "right"]].set_visible(False)

    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    out = save_dir / "monthly_aov_trend.png"
    fig.savefig(out, dpi=_SAVE_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out}")
    return out
