"""
Technical Debt / Hotspot Analysis — Plot Generator
Repos: huggingface/transformers  &  django/django
Run: python generate_plots.py
Output: ./plots/  (12 PNG files at 150 dpi, ready for LaTeX)
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
from scipy import stats

warnings.filterwarnings("ignore")
matplotlib.rcParams.update({
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
})

os.makedirs("plots", exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_TRF  = "#E8601C"   # transformers — warm orange
C_DJG  = "#1C7BE8"   # django        — cool blue
C_RISK = "#D62728"   # high-risk red
C_SAFE = "#2CA02C"   # low-risk green
ALPHA  = 0.72

# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & MERGE DATA
# ══════════════════════════════════════════════════════════════════════════════

def load_and_merge(history_csv, complexity_csv):
    h = pd.read_csv(history_csv)
    c = pd.read_csv(complexity_csv)
    h["filename"] = h["filename"].str.replace("\\", "/", regex=False)
    c["filename"] = c["filename"].str.replace("\\", "/", regex=False)
    df = pd.merge(h, c, on="filename", suffixes=("_h", "_c"))
    df = df[df["avg_complexity"] > 0].copy()
    # Normalised scores  (0-1)
    for col in ["churn", "avg_complexity", "bug_fixes"]:
        mx = df[col].max()
        df[f"{col}_n"] = df[col] / mx if mx > 0 else 0
    # Composite hotspot score  (equal weights)
    df["hotspot_score"] = (df["churn_n"] + df["avg_complexity_n"] + df["bug_fixes_n"]) / 3
    # Short label for plots
    df["label"] = df["filename"].apply(lambda x: "/".join(x.split("/")[-2:]))
    return df

df_t = load_and_merge(
    "./transformers_history.csv",
    "./transformers_complexity.csv",
)
df_d = load_and_merge(
    "./django_history.csv",
    "./django_complexity.csv",
)

print(f"Transformers merged: {len(df_t)} files")
print(f"Django       merged: {len(df_d)} files")


# ══════════════════════════════════════════════════════════════════════════════
# HELPER — save figure
# ══════════════════════════════════════════════════════════════════════════════
def save(name):
    path = f"plots/{name}.png"
    plt.savefig(path)
    plt.close()
    print(f"  Saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Hotspot Scatter: Complexity vs Churn  (side-by-side, bubble=bug_fixes)
# ══════════════════════════════════════════════════════════════════════════════
print("\n[1/12] Hotspot scatter …")

def hotspot_scatter(ax, df, color, title, top_n=10):
    sizes = (df["bug_fixes"] / df["bug_fixes"].max() * 400).clip(lower=10)
    sc = ax.scatter(
        df["churn"], df["avg_complexity"],
        s=sizes, c=color, alpha=ALPHA, linewidths=0.3, edgecolors="white",
    )
    # Quadrant lines (medians)
    med_x, med_y = df["churn"].median(), df["avg_complexity"].median()
    ax.axvline(med_x, color="grey", lw=0.8, ls="--", alpha=0.6)
    ax.axhline(med_y, color="grey", lw=0.8, ls="--", alpha=0.6)
    # Quadrant labels
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    ax.text(med_x * 1.02, med_y * 1.02, "High Risk", color=C_RISK,
            fontsize=8, fontweight="bold", alpha=0.8)
    ax.text(0.02 * df["churn"].max(), med_y * 1.02, "High Complexity\nLow Churn",
            color="saddlebrown", fontsize=7, alpha=0.7)
    ax.text(med_x * 1.02, ylim[0] + 0.5, "High Churn\nLow Complexity",
            color="steelblue", fontsize=7, alpha=0.7)
    # Annotate top-N by hotspot score
    top = df.nlargest(top_n, "hotspot_score")
    for _, row in top.iterrows():
        ax.annotate(
            row["label"], (row["churn"], row["avg_complexity"]),
            fontsize=6, alpha=0.85,
            xytext=(4, 4), textcoords="offset points",
        )
    ax.set_title(title, fontweight="bold")
    ax.set_xlabel("Churn  (# commits touching file)")
    ax.set_ylabel("Avg Cyclomatic Complexity")
    return sc

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
hotspot_scatter(axes[0], df_t, C_TRF, "huggingface/transformers")
hotspot_scatter(axes[1], df_d, C_DJG, "django/django")
fig.suptitle("Figure 1 — Hotspot Map: Cyclomatic Complexity vs Churn\n"
             "(bubble size ∝ bug-fix commit count)", fontsize=13, fontweight="bold")
plt.tight_layout()
save("fig01_hotspot_scatter")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Top 15 Files by Composite Hotspot Score
# ══════════════════════════════════════════════════════════════════════════════
print("[2/12] Top hotspot bar chart …")

def top_hotspot_bar(ax, df, color, title, n=15):
    top = df.nlargest(n, "hotspot_score")[["label", "hotspot_score"]].iloc[::-1]
    bars = ax.barh(top["label"], top["hotspot_score"], color=color, alpha=0.82, edgecolor="white")
    ax.set_xlabel("Composite Hotspot Score  (0–1)")
    ax.set_title(title, fontweight="bold")
    ax.set_xlim(0, 1.05)
    for bar, val in zip(bars, top["hotspot_score"]):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                f"{val:.3f}", va="center", fontsize=7)

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
top_hotspot_bar(axes[0], df_t, C_TRF, "Transformers — Top 15 Hotspot Files")
top_hotspot_bar(axes[1], df_d, C_DJG, "Django — Top 15 Hotspot Files")
fig.suptitle("Figure 2 — Ranked Hotspot Files by Composite Risk Score\n"
             "(Churn + Complexity + Bug-Fix Density, equally weighted)", fontsize=13, fontweight="bold")
plt.tight_layout()
save("fig02_top_hotspots")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Top 15 by Churn
# ══════════════════════════════════════════════════════════════════════════════
print("[3/12] Top churn …")

def top_bar(ax, df, metric, color, xlabel, title, n=15):
    top = df.nlargest(n, metric)[["label", metric]].iloc[::-1]
    ax.barh(top["label"], top[metric], color=color, alpha=0.82, edgecolor="white")
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontweight="bold")

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
top_bar(axes[0], df_t, "churn", C_TRF, "Commit churn count", "Transformers — Top 15 by Churn")
top_bar(axes[1], df_d, "churn", C_DJG, "Commit churn count", "Django — Top 15 by Churn")
fig.suptitle("Figure 3 — Files with Highest Commit Churn (change frequency)", fontsize=13, fontweight="bold")
plt.tight_layout()
save("fig03_top_churn")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Top 15 by Avg Cyclomatic Complexity
# ══════════════════════════════════════════════════════════════════════════════
print("[4/12] Top complexity …")

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
top_bar(axes[0], df_t, "avg_complexity", C_TRF, "Avg Cyclomatic Complexity",
        "Transformers — Top 15 by Complexity")
top_bar(axes[1], df_d, "avg_complexity", C_DJG, "Avg Cyclomatic Complexity",
        "Django — Top 15 by Complexity")
fig.suptitle("Figure 4 — Files with Highest Avg Cyclomatic Complexity", fontsize=13, fontweight="bold")
plt.tight_layout()
save("fig04_top_complexity")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Top 15 by Bug-Fix Commit Count
# ══════════════════════════════════════════════════════════════════════════════
print("[5/12] Top bug-fix files …")

fig, axes = plt.subplots(1, 2, figsize=(15, 7))
top_bar(axes[0], df_t, "bug_fixes", C_TRF, "Bug-fix commit count",
        "Transformers — Top 15 by Bug-Fix Commits")
top_bar(axes[1], df_d, "bug_fixes", C_DJG, "Bug-fix commit count",
        "Django — Top 15 by Bug-Fix Commits")
fig.suptitle("Figure 5 — Files Most Frequently Touched by Bug-Fix Commits", fontsize=13, fontweight="bold")
plt.tight_layout()
save("fig05_top_bugfixes")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 6 — Correlation Heatmap
# ══════════════════════════════════════════════════════════════════════════════
print("[6/12] Correlation heatmap …")

def corr_heatmap(ax, df, title):
    cols = ["churn", "bug_fixes", "avg_complexity", "nloc", "num_functions"]
    nice = ["Churn", "Bug Fixes", "Avg Complexity", "NLOC", "Num Functions"]
    corr = df[cols].corr()
    corr.columns = nice
    corr.index = nice
    im = ax.imshow(corr, cmap="RdYlGn", vmin=-1, vmax=1)
    ax.set_xticks(range(len(nice))); ax.set_xticklabels(nice, rotation=30, ha="right")
    ax.set_yticks(range(len(nice))); ax.set_yticklabels(nice)
    for i in range(len(nice)):
        for j in range(len(nice)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color="black")
    ax.set_title(title, fontweight="bold")
    return im

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
im = corr_heatmap(axes[0], df_t, "Transformers")
corr_heatmap(axes[1], df_d, "Django")
fig.colorbar(im, ax=axes, label="Pearson r", shrink=0.8)
fig.suptitle("Figure 6 — Correlation Matrix: Code Metrics", fontsize=13, fontweight="bold")
plt.tight_layout()
save("fig06_correlation_heatmap")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Bug-Fix Ratio  (bug_fixes / churn)  by Complexity Quartile
# ══════════════════════════════════════════════════════════════════════════════
print("[7/12] Bug-fix ratio by complexity quartile …")

def bugfix_ratio_quartile(ax, df, color, title):
    df = df.copy()
    df["bug_ratio"] = df["bug_fixes"] / df["churn"].clip(lower=1)
    df["ccn_quartile"] = pd.qcut(df["avg_complexity"], 4,
                                  labels=["Q1\n(Low)", "Q2", "Q3", "Q4\n(High)"])
    grouped = df.groupby("ccn_quartile", observed=True)["bug_ratio"].mean()
    bars = ax.bar(grouped.index, grouped.values, color=color, alpha=0.82, edgecolor="white", width=0.55)
    ax.set_ylabel("Mean Bug-Fix Ratio  (bug-fix commits / total commits)")
    ax.set_xlabel("Complexity Quartile")
    ax.set_title(title, fontweight="bold")
    ax.set_ylim(0, 1.0)
    for bar, val in zip(bars, grouped.values):
        ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                f"{val:.2f}", ha="center", fontsize=9)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
bugfix_ratio_quartile(axes[0], df_t, C_TRF, "Transformers")
bugfix_ratio_quartile(axes[1], df_d, C_DJG, "Django")
fig.suptitle("Figure 7 — Bug-Fix Ratio vs Cyclomatic Complexity Quartile\n"
             "(validates: higher complexity → more bug-fix commits)", fontsize=13, fontweight="bold")
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
    ax.hist(df["churn"], bins=50, color=color, alpha=0.75, edgecolor="white", log=True)
    ax.set_xlabel("Churn (# commits)")
    ax.set_ylabel("File count  (log scale)")
    ax.set_title(title, fontweight="bold")
    med = df["churn"].median()
    ax.axvline(med, color="black", lw=1.2, ls="--", label=f"Median = {med:.0f}")
    p90 = df["churn"].quantile(0.9)
    ax.axvline(p90, color=C_RISK, lw=1.2, ls=":", label=f"P90 = {p90:.0f}")
    ax.legend()

fig.suptitle("Figure 8 — Churn Distribution Across Files  (log-scale)\n"
             "Power-law tail: a small fraction of files absorb most changes", fontsize=13, fontweight="bold")
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
    ax.hist(df["avg_complexity"], bins=40, color=color, alpha=0.75, edgecolor="white")
    for thresh, (tc, label) in thresholds.items():
        ax.axvline(thresh, color=tc, lw=1.5, ls="--", label=label)
    ax.set_xlabel("Avg Cyclomatic Complexity")
    ax.set_ylabel("File count")
    ax.set_title(title, fontweight="bold")
    ax.legend(title="Risk thresholds")

fig.suptitle("Figure 9 — Distribution of Avg Cyclomatic Complexity per File\n"
             "McCabe thresholds: >10 Moderate risk, >20 High risk", fontsize=13, fontweight="bold")
plt.tight_layout()
save("fig09_complexity_distribution")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 10 — NLOC vs Avg Complexity  (size = churn)
# ══════════════════════════════════════════════════════════════════════════════
print("[10/12] NLOC vs Complexity …")

def nloc_complexity(ax, df, color, title):
    sizes = (df["churn"] / df["churn"].max() * 300).clip(lower=8)
    ax.scatter(df["nloc"], df["avg_complexity"], s=sizes, c=color, alpha=ALPHA,
               edgecolors="white", linewidths=0.3)
    # Regression line
    slope, intercept, r, p, _ = stats.linregress(df["nloc"], df["avg_complexity"])
    x_line = np.linspace(0, df["nloc"].quantile(0.98), 100)
    ax.plot(x_line, slope * x_line + intercept, color="black", lw=1.3, ls="--",
            label=f"r = {r:.2f}  (p {'< 0.001' if p < 0.001 else f'= {p:.3f}'})")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("NLOC (non-commented lines of code)")
    ax.set_ylabel("Avg Cyclomatic Complexity")
    ax.set_title(title, fontweight="bold")
    ax.legend()

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
nloc_complexity(axes[0], df_t, C_TRF, "Transformers")
nloc_complexity(axes[1], df_d, C_DJG, "Django")
fig.suptitle("Figure 10 — File Size (NLOC) vs Avg Cyclomatic Complexity\n"
             "(bubble size ∝ churn; dashed = OLS regression)", fontsize=13, fontweight="bold")
plt.tight_layout()
save("fig10_nloc_vs_complexity")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 11 — Lines Added vs Lines Deleted (churn breakdown)
# ══════════════════════════════════════════════════════════════════════════════
print("[11/12] Lines added vs deleted …")

def lines_scatter(ax, df, color, title, n=15):
    ax.scatter(df["lines_added"], df["lines_deleted"], c=color, alpha=0.45,
               s=20, edgecolors="white", linewidths=0.2)
    # 45° reference line
    mx = max(df["lines_added"].quantile(0.97), df["lines_deleted"].quantile(0.97))
    ax.plot([0, mx], [0, mx], "k--", lw=0.8, alpha=0.5, label="Equal add/delete")
    ax.set_xlim(0, df["lines_added"].quantile(0.98))
    ax.set_ylim(0, df["lines_deleted"].quantile(0.98))
    # Annotate top by (added+deleted)
    df = df.copy()
    df["total_delta"] = df["lines_added"] + df["lines_deleted"]
    top = df.nlargest(n, "total_delta")
    for _, row in top.iterrows():
        ax.annotate(row["label"], (row["lines_added"], row["lines_deleted"]),
                    fontsize=6, alpha=0.8, xytext=(3, 3), textcoords="offset points")
    ax.set_xlabel("Lines Added")
    ax.set_ylabel("Lines Deleted")
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
lines_scatter(axes[0], df_t, C_TRF, "Transformers")
lines_scatter(axes[1], df_d, C_DJG, "Django")
fig.suptitle("Figure 11 — Lines Added vs Lines Deleted per File\n"
             "Files above the diagonal lose more code than they gain (refactoring / removal)", fontsize=13, fontweight="bold")
plt.tight_layout()
save("fig11_lines_added_deleted")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 12 — Remediation Priority Matrix  (top 20 combined, 4-quadrant)
# ══════════════════════════════════════════════════════════════════════════════
print("[12/12] Remediation priority matrix …")

def priority_matrix(ax, df, color, title, n=20):
    df = df.copy()
    med_c = df["churn"].median()
    med_x = df["avg_complexity"].median()
    top = df.nlargest(n, "hotspot_score")

    def quadrant(row):
        hi_churn = row["churn"] >= med_c
        hi_ccn   = row["avg_complexity"] >= med_x
        if hi_churn and hi_ccn:   return "Refactor Now",   C_RISK,    "★★★"
        if hi_churn:              return "Reduce Churn",   "darkorange", "★★"
        if hi_ccn:                return "Simplify",       "goldenrod",  "★★"
        return                           "Monitor",        C_SAFE,    "★"

    for _, row in top.iterrows():
        label, c, priority = quadrant(row)
        ax.scatter(row["churn"], row["avg_complexity"],
                   s=row["bug_fixes"] / df["bug_fixes"].max() * 500 + 30,
                   color=c, alpha=0.85, edgecolors="white", linewidths=0.5, zorder=3)
        ax.annotate(f"{row['label']}\n{priority}",
                    (row["churn"], row["avg_complexity"]),
                    fontsize=5.5, alpha=0.9, xytext=(4, 4), textcoords="offset points")

    ax.axvline(med_c, color="grey", lw=0.9, ls="--", alpha=0.6)
    ax.axhline(med_x, color="grey", lw=0.9, ls="--", alpha=0.6)

    # Quadrant labels
    xmax = top["churn"].max() * 1.1
    ymax = top["avg_complexity"].max() * 1.08
    ax.text(med_c * 1.02, ymax * 0.95, "▲ REFACTOR NOW\n(high churn, high complexity)",
            color=C_RISK, fontsize=7, fontweight="bold")
    ax.text(0.01 * xmax, ymax * 0.95, "SIMPLIFY\n(low churn, high complexity)",
            color="goldenrod", fontsize=7)
    ax.text(med_c * 1.02, med_x * 0.15, "REDUCE CHURN\n(high churn, low complexity)",
            color="darkorange", fontsize=7)
    ax.text(0.01 * xmax, med_x * 0.15, "MONITOR\n(low churn, low complexity)",
            color=C_SAFE, fontsize=7)

    legend_elements = [
        mpatches.Patch(color=C_RISK,      label="Refactor Now  ★★★"),
        mpatches.Patch(color="darkorange", label="Reduce Churn  ★★"),
        mpatches.Patch(color="goldenrod",  label="Simplify       ★★"),
        mpatches.Patch(color=C_SAFE,       label="Monitor        ★"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=7)
    ax.set_xlabel("Churn  (# commits touching file)")
    ax.set_ylabel("Avg Cyclomatic Complexity")
    ax.set_title(title, fontweight="bold")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))
priority_matrix(axes[0], df_t, C_TRF, "Transformers — Remediation Priority Matrix")
priority_matrix(axes[1], df_d, C_DJG, "Django — Remediation Priority Matrix")
fig.suptitle("Figure 12 — Remediation Priority Matrix  (Top 20 Hotspot Files)\n"
             "Bubble size ∝ bug-fix commit count; stars = recommended action urgency",
             fontsize=13, fontweight="bold")
plt.tight_layout()
save("fig12_remediation_matrix")


# ══════════════════════════════════════════════════════════════════════════════
# PRINT SUMMARY TABLES (for use in LaTeX)
# ══════════════════════════════════════════════════════════════════════════════
print("\n──────────────────────────────────────────────────────────────────")
print("TOP 10 HOTSPOTS — TRANSFORMERS")
print("──────────────────────────────────────────────────────────────────")
cols_show = ["label", "churn", "bug_fixes", "avg_complexity", "nloc", "hotspot_score"]
print(df_t.nlargest(10, "hotspot_score")[cols_show].to_string(index=False))

print("\n──────────────────────────────────────────────────────────────────")
print("TOP 10 HOTSPOTS — DJANGO")
print("──────────────────────────────────────────────────────────────────")
print(df_d.nlargest(10, "hotspot_score")[cols_show].to_string(index=False))

# Stats for LaTeX
print("\n── STATS ──")
for name, df in [("Transformers", df_t), ("Django", df_d)]:
    r, p = stats.pearsonr(df["churn"], df["bug_fixes"])
    r2, p2 = stats.pearsonr(df["avg_complexity"], df["bug_fixes"])
    print(f"{name}: churn↔bug_fixes r={r:.3f} (p={p:.2e})  |  complexity↔bug_fixes r={r2:.3f} (p={p2:.2e})")

print("\n✅  All 12 plots saved to ./plots/")
