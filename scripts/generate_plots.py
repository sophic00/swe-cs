"""
Technical Debt / Hotspot Analysis — Plot Generator
Repos: huggingface/transformers  &  django/django
Run: python scripts/generate_plots.py
Input: ./data/*.csv
Output: ./plots/  (12 PNG files at 150 dpi, ready for LaTeX)
"""

import warnings
from pathlib import Path
from typing import cast

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings("ignore")
matplotlib.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    }
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
PLOTS_DIR = ROOT / "plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_TRF = "#E8601C"  # transformers — warm orange
C_DJG = "#1C7BE8"  # django        — cool blue
C_RISK = "#D62728"  # high-risk red
C_SAFE = "#2CA02C"  # low-risk green
ALPHA = 0.72


def as_frame(value: object) -> pd.DataFrame:
    return cast(pd.DataFrame, value)


def column(df: pd.DataFrame, name: str) -> pd.Series:
    return cast(pd.Series, df[name])


def top_rows(df: pd.DataFrame, n: int, sort_by: str) -> pd.DataFrame:
    return as_frame(df.sort_values(by=sort_by, ascending=False).head(n))


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & MERGE DATA
# ══════════════════════════════════════════════════════════════════════════════


def load_and_merge(history_csv: Path | str, complexity_csv: Path | str) -> pd.DataFrame:
    h = as_frame(pd.read_csv(history_csv))
    c = as_frame(pd.read_csv(complexity_csv))
    h.loc[:, "filename"] = (
        column(h, "filename").astype(str).str.replace("\\", "/", regex=False)
    )
    c.loc[:, "filename"] = (
        column(c, "filename").astype(str).str.replace("\\", "/", regex=False)
    )
    df = as_frame(pd.merge(h, c, on="filename", suffixes=("_h", "_c")))
    df = as_frame(df[column(df, "avg_complexity") > 0].copy())
    # Normalised scores  (0-1)
    for col in ["churn", "avg_complexity", "bug_fixes"]:
        values = column(df, col)
        mx = float(values.max())
        df.loc[:, f"{col}_n"] = values / mx if mx > 0 else 0.0
    # Composite hotspot score  (equal weights)
    df.loc[:, "hotspot_score"] = (
        column(df, "churn_n")
        + column(df, "avg_complexity_n")
        + column(df, "bug_fixes_n")
    ) / 3
    # Short label for plots
    df.loc[:, "label"] = [
        "/".join(str(filename).split("/")[-2:]) for filename in column(df, "filename")
    ]
    return df


df_t = load_and_merge(
    DATA_DIR / "transformers_history.csv",
    DATA_DIR / "transformers_complexity.csv",
)
df_d = load_and_merge(
    DATA_DIR / "django_history.csv",
    DATA_DIR / "django_complexity.csv",
)

print(f"Transformers merged: {len(df_t)} files")
print(f"Django       merged: {len(df_d)} files")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — save figure
# ══════════════════════════════════════════════════════════════════════════════
def save(name):
    path = PLOTS_DIR / f"{name}.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Hotspot Scatter: Complexity vs Churn  (side-by-side, bubble=bug_fixes)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/12] Hotspot scatter …")


def hotspot_scatter(ax, df, color, title, top_n=10):
    bug_fixes = column(df, "bug_fixes")
    churn = column(df, "churn")
    complexity = column(df, "avg_complexity")
    sizes = (bug_fixes / float(bug_fixes.max()) * 400).clip(lower=10)
    sc = ax.scatter(
        churn,
        complexity,
        s=sizes,
        c=color,
        alpha=ALPHA,
        linewidths=0.3,
        edgecolors="white",
    )
    # Quadrant lines (medians)
    med_x = float(churn.median())
    med_y = float(complexity.median())
    ax.axvline(med_x, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(med_y, color="grey", lw=0.8, ls="--", alpha=0.6)
    # Quadrant labels
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.text(
        med_x * 1.02,
        med_y * 1.02,
        "High Risk",
        color=C_RISK,
        fontsize=8,
        fontweight="bold",
        alpha=0.8,
    )
    ax.text(
        0.02 * float(churn.max()),
        med_y * 1.02,
        "High Complexity\nLow Churn",
        color="saddlebrown",
        fontsize=7,
        alpha=0.7,
    )
    ax.text(
        med_x * 1.02,
        ylim[0] + 0.5,
        "High Churn\nLow Complexity",
        color="steelblue",
        fontsize=7,
        alpha=0.7,
    )
    # Annotate top-N by hotspot score
    top = top_rows(df, top_n, "hotspot_score")
    for _, row in top.iterrows():
        ax.annotate(
            row["label"],
            (row["churn"], row["avg_complexity"]),
            fontsize=6,
            alpha=0.85,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Churn  (# commits touching file)")
    ax.set_ylabel("Avg Cyclomatic Complexity")
    return sc


fig, axes = plt.subplots(1, 2, figsize=(15, 6))
hotspot_scatter(axes[0], df_t, C_TRF, "huggingface/transformers")
hotspot_scatter(axes[1], df_d, C_DJG, "django/django")
fig.suptitle(
    "Figure 1 — Hotspot Map: Cyclomatic Complexity vs Churn\n"
    "(bubble size ∝ bug-fix commit count)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
save("fig01_hotspot_scatter")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Top 15 Files by Composite Hotspot Score
# ══════════════════════════════════════════════════════════════════════════════
print("[2/12] Top hotspot bar chart …")


def top_hotspot_bar(ax, df, color, title, n=15):
    top = as_frame(
        top_rows(df, n, "hotspot_score")[["label", "hotspot_score"]].iloc[::-1]
    )
    bars = ax.barh(
        column(top, "label"),
        column(top, "hotspot_score"),
        color=color,
        alpha=0.82,
        edgecolor="white",
    )
    ax.set_xlabel("Composite Hotspot Score  (0–1)")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, column(top, "hotspot_score")):
        ax.text(
            val + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=7,
        )


fig, axes = plt.subplots(1, 2, figsize=(15, 7))
top_hotspot_bar(axes[0], df_t, C_TRF, "Transformers — Top 15 Hotspot Files")
top_hotspot_bar(axes[1], df_d, C_DJG, "Django — Top 15 Hotspot Files")
fig.suptitle(
    "Figure 2 — Ranked Hotspot Files by Composite Risk Score\n"
    "(Churn + Complexity + Bug-Fix Density, equally weighted)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
save("fig02_top_hotspots")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Top 15 by Churn
# ══════════════════════════════════════════════════════════════════════════════
print("[3/12] Top churn …")


def top_bar(ax, df, metric, color, xlabel, title, n=15):
    top = as_frame(top_rows(df, n, metric)[["label", metric]].iloc[::-1])
    ax.barh(
        column(top, "label"),
        column(top, metric),
        color=color,
        alpha=0.82,
        edgecolor="white",
    )
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold")


fig, axes = plt.subplots(1, 2, figsize=(15, 7))
top_bar(
    axes[0],
    df_t,
    "churn",
    C_TRF,
    "Commit churn count",
    "Transformers — Top 15 by Churn",
)
top_bar(axes[1], df_d, "churn", C_DJG, "Commit churn count", "Django — Top 15 by Churn")
fig.suptitle(
    "Figure 3 — Files with Highest Commit Churn (change frequency)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
save("fig03_top_churn")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Top 15 by Avg Cyclomatic Complexity
# ══════════════════════════════════════════════════════════════════════════════
print("[4/12] Top complexity …")

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
top_bar(
    axes[0],
    df_t,
    "avg_complexity",
    C_TRF,
    "Avg Cyclomatic Complexity",
    "Transformers — Top 15 by Complexity",
)
top_bar(
    axes[1],
    df_d,
    "avg_complexity",
    C_DJG,
    "Avg Cyclomatic Complexity",
    "Django — Top 15 by Complexity",
)
fig.suptitle(
    "Figure 4 — Files with Highest Avg Cyclomatic Complexity",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
save("fig04_top_complexity")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Top 15 by Bug-Fix Commit Count
# ══════════════════════════════════════════════════════════════════════════════
print("[5/12] Top bug-fix files …")

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
top_bar(
    axes[0],
    df_t,
    "bug_fixes",
    C_TRF,
    "Bug-fix commit count",
    "Transformers — Top 15 by Bug-Fix Commits",
)
top_bar(
    axes[1],
    df_d,
    "bug_fixes",
    C_DJG,
    "Bug-fix commit count",
    "Django — Top 15 by Bug-Fix Commits",
)
fig.suptitle(
    "Figure 5 — Files Most Frequently Touched by Bug-Fix Commits",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
save("fig05_top_bugfixes")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("[6/12] Correlation heatmap …")


def corr_heatmap(ax, df, title):
    cols = ["churn", "bug_fixes", "avg_complexity", "nloc", "num_functions"]
    nice = ["Churn", "Bug Fixes", "Avg Complexity", "NLOC", "Num Functions"]
    corr = as_frame(df[cols].corr())
    corr.columns = nice
    corr.index = nice
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(nice)))
    ax.set_xticklabels(nice, rotation=30, ha="right")
    ax.set_yticks(range(len(nice)))
    ax.set_yticklabels(nice)
    for i in range(len(nice)):
        for j in range(len(nice)):
            ax.text(
                j,
                i,
                f"{corr.iloc[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color="black",
            )
    ax.set_title(title, fontweight="bold")
    return im


fig = plt.figure(figsize=(14, 5))
gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.35)
axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
cax = fig.add_subplot(gs[0, 2])
im = corr_heatmap(axes[0], df_t, "Transformers")
corr_heatmap(axes[1], df_d, "Django")
fig.colorbar(im, cax=cax, label="Pearson r")
fig.suptitle(
    "Figure 6 — Correlation Matrix: Code Metrics", fontsize=13, fontweight="bold"
)
fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
save("fig06_correlation_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Bug-Fix Ratio  (bug_fixes / churn)  by Complexity Quartile
# ══════════════════════════════════════════════════════════════════════════════
print("[7/12] Bug-fix ratio by complexity quartile …")


def bugfix_ratio_quartile(ax, df, color, title):
    df = as_frame(df.copy())
    df.loc[:, "bug_ratio"] = column(df, "bug_fixes") / column(df, "churn").clip(lower=1)
    df.loc[:, "ccn_quartile"] = pd.qcut(
        column(df, "avg_complexity"), 4, labels=["Q1\n(Low)", "Q2", "Q3", "Q4\n(High)"]
    )
    grouped = cast(
        pd.Series, df.groupby("ccn_quartile", observed=True)["bug_ratio"].mean()
    )
    bars = ax.bar(
        grouped.index,
        grouped.values,
        color=color,
        alpha=0.82,
        edgecolor="white",
        width=0.55,
    )
    ax.set_ylabel("Mean Bug-Fix Ratio  (bug-fix commits / total commits)")
    ax.set_xlabel("Complexity Quartile")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, grouped.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01,
            f"{val:.2f}",
            ha="center",
            fontsize=9,
        )


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
bugfix_ratio_quartile(axes[0], df_t, C_TRF, "Transformers")
bugfix_ratio_quartile(axes[1], df_d, C_DJG, "Django")
fig.suptitle(
    "Figure 7 — Bug-Fix Ratio vs Cyclomatic Complexity Quartile\n"
    "(validates: higher complexity → more bug-fix commits)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
save("fig07_bugfix_ratio_by_quartile")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Churn Distribution  (log-scale histogram + KDE)
# ══════════════════════════════════════════════════════════════════════════════
print("[8/12] Churn distribution …")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, df, color, title in [
    (axes[0], df_t, C_TRF, "Transformers"),
    (axes[1], df_d, C_DJG, "Django"),
]:
    churn = column(df, "churn")
    ax.hist(churn, bins=50, color=color, alpha=0.75, edgecolor="white", log=True)
    ax.set_xlabel("Churn (# commits)")
    ax.set_ylabel("File count  (log scale)")
    ax.set_title(title, fontweight="bold")
    med = float(churn.median())
    ax.axvline(med, color="black", lw=1.2, ls="--", label=f"Median = {med:.0f}")
    p90 = float(churn.quantile(0.9))
    ax.axvline(p90, color=C_RISK, lw=1.2, ls=":", label=f"P90 = {p90:.0f}")
    ax.legend()

fig.suptitle(
    "Figure 8 — Churn Distribution Across Files  (log-scale)\n"
    "Power-law tail: a small fraction of files absorb most changes",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
save("fig08_churn_distribution")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Complexity Distribution (histogram)
# ══════════════════════════════════════════════════════════════════════════════
print("[9/12] Complexity distribution …")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
thresholds = {10: ("yellow", "Moderate (10)"), 20: (C_RISK, "High (20)")}
for ax, df, color, title in [
    (axes[0], df_t, C_TRF, "Transformers"),
    (axes[1], df_d, C_DJG, "Django"),
]:
    ax.hist(
        column(df, "avg_complexity"),
        bins=40,
        color=color,
        alpha=0.75,
        edgecolor="white",
    )
    for thresh, (tc, label) in thresholds.items():
        ax.axvline(thresh, color=tc, lw=1.5, ls="--", label=label)
    ax.set_xlabel("Avg Cyclomatic Complexity")
    ax.set_ylabel("File count")
    ax.set_title(title, fontweight="bold")
    ax.legend(title="Risk thresholds")

fig.suptitle(
    "Figure 9 — Distribution of Avg Cyclomatic Complexity per File\n"
    "McCabe thresholds: >10 Moderate risk, >20 High risk",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
save("fig09_complexity_distribution")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — NLOC vs Avg Complexity  (size = churn)
# ══════════════════════════════════════════════════════════════════════════════
print("[10/12] NLOC vs Complexity …")


def nloc_complexity(ax, df, color, title):
    churn = column(df, "churn")
    nloc = column(df, "nloc")
    complexity = column(df, "avg_complexity")
    sizes = (churn / float(churn.max()) * 300).clip(lower=8)
    ax.scatter(
        nloc,
        complexity,
        s=sizes,
        c=color,
        alpha=ALPHA,
        edgecolors="white",
        linewidths=0.3,
    )
    # Regression line
    slope, intercept, r, p, _ = cast(
        tuple[float, float, float, float, float],
        stats.linregress(nloc, complexity),
    )
    x_line = np.linspace(0, float(nloc.quantile(0.98)), 100)
    ax.plot(
        x_line,
        slope * x_line + intercept,
        color="black",
        lw=1.3,
        ls="--",
        label=f"r = {r:.2f}  (p {'< 0.001' if p < 0.001 else f'= {p:.3f}'})",
    )
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("NLOC (non-commented lines of code)")
    ax.set_ylabel("Avg Cyclomatic Complexity")
    ax.set_title(title, fontweight="bold")
    ax.legend()


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
nloc_complexity(axes[0], df_t, C_TRF, "Transformers")
nloc_complexity(axes[1], df_d, C_DJG, "Django")
fig.suptitle(
    "Figure 10 — File Size (NLOC) vs Avg Cyclomatic Complexity\n"
    "(bubble size ∝ churn; dashed = OLS regression)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
save("fig10_nloc_vs_complexity")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11 — Lines Added vs Lines Deleted (churn breakdown)
# ══════════════════════════════════════════════════════════════════════════════
print("[11/12] Lines added vs deleted …")


def lines_scatter(ax, df, color, title, n=15):
    lines_added = column(df, "lines_added")
    lines_deleted = column(df, "lines_deleted")
    ax.scatter(
        lines_added,
        lines_deleted,
        c=color,
        alpha=0.45,
        s=20,
        edgecolors="white",
        linewidths=0.2,
    )
    # 45° reference line
    mx = max(float(lines_added.quantile(0.97)), float(lines_deleted.quantile(0.97)))
    ax.plot([0, mx], [0, mx], "k--", lw=0.8, alpha=0.5, label="Equal add/delete")
    ax.set_xlim(0, float(lines_added.quantile(0.98)))
    ax.set_ylim(0, float(lines_deleted.quantile(0.98)))
    # Annotate top by (added+deleted)
    df = as_frame(df.copy())
    df.loc[:, "total_delta"] = column(df, "lines_added") + column(df, "lines_deleted")
    top = top_rows(df, n, "total_delta")
    for _, row in top.iterrows():
        ax.annotate(
            row["label"],
            (row["lines_added"], row["lines_deleted"]),
            fontsize=6,
            alpha=0.8,
            xytext=(3, 3),
            textcoords="offset points",
        )
    ax.set_xlabel("Lines Added")
    ax.set_ylabel("Lines Deleted")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)


fig, axes = plt.subplots(1, 2, figsize=(13, 5))
lines_scatter(axes[0], df_t, C_TRF, "Transformers")
lines_scatter(axes[1], df_d, C_DJG, "Django")
fig.suptitle(
    "Figure 11 — Lines Added vs Lines Deleted per File\n"
    "Files above the diagonal lose more code than they gain (refactoring / removal)",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
save("fig11_lines_added_deleted")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12 — Remediation Priority Matrix  (top 20 combined, 4-quadrant)
# ══════════════════════════════════════════════════════════════════════════════
print("[12/12] Remediation priority matrix …")


def priority_matrix(ax, df, color, title, n=20):
    df = as_frame(df.copy())
    med_c = float(column(df, "churn").median())
    med_x = float(column(df, "avg_complexity").median())
    top = top_rows(df, n, "hotspot_score")
    max_bug_fixes = float(column(df, "bug_fixes").max())

    def quadrant(row):
        hi_churn = row["churn"] >= med_c
        hi_ccn = row["avg_complexity"] >= med_x
        if hi_churn and hi_ccn:
            return "Refactor Now", C_RISK, "★★★"
        if hi_churn:
            return "Reduce Churn", "darkorange", "★★"
        if hi_ccn:
            return "Simplify", "goldenrod", "★★"
        return "Monitor", C_SAFE, "★"

    for _, row in top.iterrows():
        label, c, priority = quadrant(row)
        ax.scatter(
            row["churn"],
            row["avg_complexity"],
            s=row["bug_fixes"] / max_bug_fixes * 500 + 30,
            color=c,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
            zorder=3,
        )
        ax.annotate(
            f"{row['label']}\n{priority}",
            (row["churn"], row["avg_complexity"]),
            fontsize=5.5,
            alpha=0.9,
            xytext=(4, 4),
            textcoords="offset points",
        )

    ax.axvline(med_c, color="grey", lw=0.9, ls="--", alpha=0.6)
    ax.axhline(med_x, color="grey", lw=0.9, ls="--", alpha=0.6)

    # Keep quadrant headers pinned to the corners so they don't collide when
    # medians fall close to the plot edge.
    label_box = {"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 1.5}
    ax.text(
        0.02,
        0.98,
        "SIMPLIFY\n(low churn, high complexity)",
        transform=ax.transAxes,
        ha="left",
        va="top",
        color="goldenrod",
        fontsize=7,
        bbox=label_box,
    )
    ax.text(
        0.98,
        0.98,
        "▲ REFACTOR NOW\n(high churn, high complexity)",
        transform=ax.transAxes,
        ha="right",
        va="top",
        color=C_RISK,
        fontsize=7,
        fontweight="bold",
        bbox=label_box,
    )
    ax.text(
        0.98,
        0.03,
        "REDUCE CHURN\n(high churn, low complexity)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        color="darkorange",
        fontsize=7,
        bbox=label_box,
    )
    ax.text(
        0.02,
        0.03,
        "MONITOR\n(low churn, low complexity)",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        color=C_SAFE,
        fontsize=7,
        bbox=label_box,
    )

    legend_elements = [
        mpatches.Patch(color=C_RISK, label="Refactor Now  ★★★"),
        mpatches.Patch(color="darkorange", label="Reduce Churn  ★★"),
        mpatches.Patch(color="goldenrod", label="Simplify       ★★"),
        mpatches.Patch(color=C_SAFE, label="Monitor        ★"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)
    ax.set_xlabel("Churn  (# commits touching file)")
    ax.set_ylabel("Avg Cyclomatic Complexity")
    ax.set_title(title, fontweight="bold")


fig, axes = plt.subplots(1, 2, figsize=(16, 7))
priority_matrix(axes[0], df_t, C_TRF, "Transformers — Remediation Priority Matrix")
priority_matrix(axes[1], df_d, C_DJG, "Django — Remediation Priority Matrix")
fig.suptitle(
    "Figure 12 — Remediation Priority Matrix  (Top 20 Hotspot Files)\n"
    "Bubble size ∝ bug-fix commit count; stars = recommended action urgency",
    fontsize=13,
    fontweight="bold",
)
plt.tight_layout()
save("fig12_remediation_matrix")


# ══════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY TABLES (for use in LaTeX)
# ══════════════════════════════════════════════════════════════════════════════
print("\n──────────────────────────────────────────────────────────────────")
print("TOP 10 HOTSPOTS — TRANSFORMERS")
print("──────────────────────────────────────────────────────────────────")
cols_show = ["label", "churn", "bug_fixes", "avg_complexity", "nloc", "hotspot_score"]
top_transformers = as_frame(top_rows(df_t, 10, "hotspot_score")[cols_show])
print(top_transformers.to_string(index=False))

print("\n──────────────────────────────────────────────────────────────────")
print("TOP 10 HOTSPOTS — DJANGO")
print("──────────────────────────────────────────────────────────────────")
top_django = as_frame(top_rows(df_d, 10, "hotspot_score")[cols_show])
print(top_django.to_string(index=False))

# Stats for LaTeX
print("\n── STATS ──")
for name, df in [("Transformers", df_t), ("Django", df_d)]:
    r, p = stats.pearsonr(df["churn"], df["bug_fixes"])
    r2, p2 = stats.pearsonr(df["avg_complexity"], df["bug_fixes"])
    print(
        f"{name}: churn↔bug_fixes r={r:.3f} (p={p:.2e})  |  complexity↔bug_fixes r={r2:.3f} (p={p2:.2e})"
    )

print("\n✅  All 12 plots saved to ./plots/")
