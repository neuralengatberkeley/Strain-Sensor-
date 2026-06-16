# Seaborn-styled Matplotlib plots.
# Saves figures to .svg if save_path is provided.

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import seaborn as sns

from .utils import sec_to_mmss
from .palettes import participant_color_map, app_marker_map

def set_seaborn_theme():
    """Call once in the notebook to apply Seaborn styling."""
    sns.set_theme(context="talk", style="whitegrid")

def _fmt_mmss(y, _pos=None):
    return sec_to_mmss(y)

def plot_app_or_rem_grouped(df, which="app", title="", save_path=None, ax=None):
    """
    Grouped bars for DT/TT per APP/REM with participant dots and DT->TT connectors.
    df for APP: columns = participant, method, app_label, app_index, time_sec
    df for REM: columns = participant, method, rem_label, rem_index, time_sec
    which = "app" or "rem"
    """
    assert which in {"app", "rem"}
    label_col = "app_label" if which == "app" else "rem_label"

    methods = ["DT", "TT"]
    labels = ["APP1","APP2","APP3"] if which=="app" else ["REM1","REM2","REM3"]
    participants = sorted(df["participant"].unique())

    pal = sns.color_palette("deep")
    method_colors = {"DT": pal[0], "TT": pal[1]}
    part_colors = participant_color_map(participants)

    plot_data = []
    for lb in labels:
        for m in methods:
            vals = df[(df[label_col]==lb) & (df["method"]==m)]["time_sec"].to_numpy()
            plot_data.append((lb, m, float(vals.mean()), vals))

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(plot_data))
    last_dt_vals = None

    # Bars + dots + connectors
    for i, (lb, m, mean_val, indiv_vals) in enumerate(plot_data):
        ax.bar(i, mean_val, color=method_colors[m], alpha=0.85, width=0.8)
        for p_idx, v in enumerate(indiv_vals):
            p = participants[p_idx]
            ax.scatter(i, v, color=part_colors[p], edgecolors="white", linewidths=0.5, zorder=10)
        if m == "DT":
            last_dt_vals = indiv_vals
        else:
            for p_idx, (d, t) in enumerate(zip(last_dt_vals, indiv_vals)):
                p = participants[p_idx]
                ax.plot([i-1, i], [d, t], color=part_colors[p], linestyle="--", alpha=0.85)

    centers = [i + 0.5 for i in range(0, len(plot_data), 2)]
    ax.set_xticks(centers)
    ax.set_xticklabels(labels)
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_mmss))
    ax.set_ylabel("Time (min:sec)")
    ax.set_title(title or (f"{which.upper()} Times: DT vs TT"))

    method_handles = [
        Patch(facecolor=method_colors["DT"], alpha=0.85, label="DT"),
        Patch(facecolor=method_colors["TT"], alpha=0.85, label="TT"),
    ]
    part_handles = [
        Line2D([0],[0], color=part_colors[p], linestyle="--", marker="o", label=p)
        for p in participants
    ]
    ax.legend(handles=method_handles + part_handles, frameon=False, ncol=2, loc="lower left")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, format="svg")
    return ax

def plot_collapsed_method(df, which="app", title=None, save_path=None, ax=None):
    """
    Collapsed DT vs TT mean bars with all paired points overlaid and connected.
    df columns (APP): participant, method, app_index, time_sec
    df columns (REM): participant, method, rem_index, time_sec
    which = "app" or "rem"
    """
    assert which in {"app", "rem"}
    idx_col = "app_index" if which == "app" else "rem_index"
    participants = sorted(df["participant"].unique())

    pal = sns.color_palette("deep")
    method_colors = {"DT": pal[0], "TT": pal[1]}
    collapsed = df.groupby("method")["time_sec"].mean()

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.bar([0,1], [collapsed["DT"], collapsed["TT"]],
           color=[method_colors["DT"], method_colors["TT"]],
           alpha=0.9, width=0.6)

    part_colors = participant_color_map(participants)
    mark_map = app_marker_map(df[idx_col].tolist())

    for p in participants:
        for idx in [1,2,3]:
            d = df[(df.participant==p)&(df.method=="DT")&(df[idx_col]==idx)]["time_sec"].iloc[0]
            t = df[(df.participant==p)&(df.method=="TT")&(df[idx_col]==idx)]["time_sec"].iloc[0]
            ax.scatter(0, d, color=part_colors[p], marker=mark_map[idx], s=90, edgecolors="white", linewidths=0.5, zorder=10)
            ax.scatter(1, t, color=part_colors[p], marker=mark_map[idx], s=90, edgecolors="white", linewidths=0.5, zorder=10)
            ax.plot([0,1], [d, t], color=part_colors[p], linestyle="--", alpha=0.85)

    ax.set_xticks([0,1]); ax.set_xticklabels(["DT","TT"])
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_mmss))
    ax.set_ylabel("Time (min:sec)")
    ax.set_title(title or f"Collapsed {which.upper()} Times: DT vs TT")

    method_handles = [
        Patch(facecolor=method_colors["DT"], alpha=0.9, label="DT"),
        Patch(facecolor=method_colors["TT"], alpha=0.9, label="TT"),
    ]
    part_handles = [
        Line2D([0],[0], color=participant_color_map(participants)[p], linestyle="--", marker="o", label=p)
        for p in participants
    ]
    idx_handles = [
        Line2D([0],[0], color="black", marker=app_marker_map([1,2,3])[i], linestyle="None", label=f"{which.upper()}{i}")
        for i in [1,2,3]
    ]
    ax.legend(handles=method_handles + part_handles + idx_handles, frameon=False, ncol=2, loc="lower left")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, format="svg")
    return ax

def plot_method_split_app_rem(df, which="app", title=None, save_path=None, ax=None):
    """
    Bars for APP1–APP3 split by method: [DT APP1, DT APP2, DT APP3, (gap), TT APP1, TT APP2, TT APP3]
    Overlays participant dots (no connectors).
    df (APP): participant, method, app_index, time_sec
    df (REM): participant, method, rem_index, time_sec
    which = "app" or "rem"
    """
    assert which in {"app", "rem"}
    idx_col = "app_index" if which == "app" else "rem_index"
    label_prefix = "APP" if which == "app" else "REM"
    participants = sorted(df["participant"].unique())

    # Colors
    pal = sns.color_palette("deep")
    method_colors = {"DT": pal[0], "TT": pal[1]}
    part_colors = participant_color_map(participants)

    # Prepare means and individual values
    # Positions: DT @ [0,1,2], TT @ [4,5,6] (gap between 2 and 4)
    dt_positions = [0, 1, 2]
    tt_positions = [4, 5, 6]
    idxs = [1, 2, 3]

    # Compute means for bars
    means_dt = [float(df[(df.method=="DT") & (df[idx_col]==i)]["time_sec"].mean()) for i in idxs]
    means_tt = [float(df[(df.method=="TT") & (df[idx_col]==i)]["time_sec"].mean()) for i in idxs]

    # Build figure/axes
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot bars
    ax.bar(dt_positions, means_dt, color=method_colors["DT"], alpha=0.9, width=0.8, label="DT")
    ax.bar(tt_positions, means_tt, color=method_colors["TT"], alpha=0.9, width=0.8, label="TT")

    # Overlay participant dots
    for p in participants:
        c = part_colors[p]
        for i, x in zip(idxs, dt_positions):
            vals = df[(df.participant==p)&(df.method=="DT")&(df[idx_col]==i)]["time_sec"].to_numpy()
            if len(vals):
                ax.scatter(x, vals[0], color=c, s=80, edgecolors="white", linewidths=0.5, zorder=10)
        for i, x in zip(idxs, tt_positions):
            vals = df[(df.participant==p)&(df.method=="TT")&(df[idx_col]==i)]["time_sec"].to_numpy()
            if len(vals):
                ax.scatter(x, vals[0], color=c, s=80, edgecolors="white", linewidths=0.5, zorder=10)

    # connect within-method points per participant (APP1->APP2->APP3 or REM1->REM2->REM3)
    for p in participants:
        c = part_colors[p]
        # DT polyline
        y_dt = []
        for i in idxs:
            vals = df[(df.participant==p)&(df.method=="DT")&(df[idx_col]==i)]["time_sec"].to_numpy()
            if len(vals):
                y_dt.append(vals[0])
        if len(y_dt) == len(idxs):
            ax.plot(dt_positions, y_dt, color=c, linestyle="--", alpha=0.85)

        # TT polyline
        y_tt = []
        for i in idxs:
            vals = df[(df.participant==p)&(df.method=="TT")&(df[idx_col]==i)]["time_sec"].to_numpy()
            if len(vals):
                y_tt.append(vals[0])
        if len(y_tt) == len(idxs):
            ax.plot(tt_positions, y_tt, color=c, linestyle="--", alpha=0.85)

    
    # X tick labels: APP/REM labels repeated on both sides
    xticks = dt_positions + tt_positions
    xticklabels = [f"{label_prefix}{i}" for i in idxs] + [f"{label_prefix}{i}" for i in idxs]
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)

    # place group labels in axis-fraction coords (won't collide with tick labels)
    ax.text(1, -0.15, "DT", ha="center", va="top", transform=ax.get_xaxis_transform(), clip_on=False)
    ax.text(5, -0.15, "TT", ha="center", va="top", transform=ax.get_xaxis_transform(), clip_on=False)

    # Y axis formatting
    ax.yaxis.set_major_formatter(FuncFormatter(_fmt_mmss))
    ax.set_ylabel("Time (min:sec)")
    default_title = "Application Times: DT vs TT" if which=="app" else "Removal Times: DT vs TT"
    ax.set_title(title or default_title)

    # Legend (methods + participants)
    method_handles = [
        Patch(facecolor=method_colors["DT"], alpha=0.9, label="DT"),
        Patch(facecolor=method_colors["TT"], alpha=0.9, label="TT"),
    ]
    part_handles = [
        Line2D([0],[0], color=part_colors[p], linestyle="--", marker="o", label=p)
        for p in participants
    ]
    ax.legend(handles=method_handles + part_handles, frameon=False, ncol=2, loc="lower left")

    # Layout & save
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format="svg")
    return ax


def plot_ratings_by_metric(df_rate, title="Survey Ratings by Metric: DT vs TT", save_path=None, ax=None):
    """
    Grouped bars (DT/TT) by metric with 9 points per bar and DT->TT connectors.
    expected columns:
      participant, method, app_label, app_index, metric, rating
    """
    metrics = ["ease","movement","stability","adhesion","wires","overall"]
    methods = ["DT","TT"]
    participants = sorted(df_rate["participant"].unique())
    pal = sns.color_palette("deep")
    method_colors = {"DT": pal[0], "TT": pal[1]}
    part_colors = participant_color_map(participants)
    mark_map = app_marker_map(df_rate["app_index"].tolist())

    means = (df_rate.groupby(["metric","method"], as_index=False)["rating"]
                    .mean().rename(columns={"rating":"mean_rating"}))

    x_pairs = [(m, meth) for m in metrics for meth in methods]
    x = np.arange(len(x_pairs))

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    mean_map = {(r["metric"], r["method"]): r["mean_rating"] for _, r in means.iterrows()}

    # bars
    for i, (met, meth) in enumerate(x_pairs):
        ax.bar(i, mean_map[(met, meth)], color=method_colors[meth], alpha=0.9, width=0.8)

    # dots + connectors
    for met in metrics:
        x_dt = x_pairs.index((met, "DT"))
        x_tt = x_pairs.index((met, "TT"))
        for app_i in [1,2,3]:
            dt_vals = (df_rate[(df_rate.metric==met)&(df_rate.method=="DT")&(df_rate.app_index==app_i)]
                        .sort_values("participant")["rating"].to_numpy())
            tt_vals = (df_rate[(df_rate.metric==met)&(df_rate.method=="TT")&(df_rate.app_index==app_i)]
                        .sort_values("participant")["rating"].to_numpy())
            for p_idx, p in enumerate(participants):
                color = part_colors[p]; marker = mark_map[app_i]
                ax.scatter(x_dt, dt_vals[p_idx], color=color, marker=marker, s=70, edgecolors="white", linewidths=0.5, zorder=10)
                ax.scatter(x_tt, tt_vals[p_idx], color=color, marker=marker, s=70, edgecolors="white", linewidths=0.5, zorder=10)
                ax.plot([x_dt, x_tt], [dt_vals[p_idx], tt_vals[p_idx]],
                        color=color, linestyle="--", alpha=0.85)

    centers = [i*2 + 0.5 for i in range(len(metrics))]
    ax.set_xticks(centers); ax.set_xticklabels(metrics)
    ax.set_ylim(0, 10.2)
    ax.set_ylabel("Survey Response (1-10)")
    ax.set_title(title)

    method_handles = [
        Patch(facecolor=method_colors["DT"], alpha=0.9, label="DT"),
        Patch(facecolor=method_colors["TT"], alpha=0.9, label="TT"),
    ]
    part_handles = [Line2D([0],[0], color=part_colors[p], linestyle="--", marker="o", label=p)
                    for p in participants]
    app_handles = [Line2D([0],[0], color="black", marker=mark_map[i], linestyle="None", label=f"APP{i}")
                    for i in [1,2,3]]
    ax.legend(handles=method_handles + part_handles + app_handles, frameon=False, ncol=3, loc="lower left")

    if save_path:
        plt.tight_layout()
        plt.savefig(save_path, format="svg")
    return ax
