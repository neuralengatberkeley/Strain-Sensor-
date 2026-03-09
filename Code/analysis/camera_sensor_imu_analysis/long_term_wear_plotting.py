from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def mask_to_segments(df, mask, label):
    mask = pd.Series(mask, index=df.index).fillna(False).astype(bool)
    segments = []

    if mask.sum() == 0:
        return segments

    in_segment = False
    start_idx = None

    for i, val in enumerate(mask):
        if val and not in_segment:
            in_segment = True
            start_idx = i
        elif not val and in_segment:
            end_idx = i - 1
            segments.append({
                "start_min": float(df.iloc[start_idx]["elapsed_min"]),
                "end_min": float(df.iloc[end_idx]["elapsed_min"]),
                "label": label,
            })
            in_segment = False

    if in_segment:
        end_idx = len(mask) - 1
        segments.append({
            "start_min": float(df.iloc[start_idx]["elapsed_min"]),
            "end_min": float(df.iloc[end_idx]["elapsed_min"]),
            "label": label,
        })

    return segments


def merge_close_segments(segments, max_gap_min=0.005):
    if not segments:
        return []

    segs = sorted(segments, key=lambda s: (s["start_min"], s["end_min"]))
    merged = [segs[0].copy()]

    for seg in segs[1:]:
        prev = merged[-1]

        same_label = prev["label"] == seg["label"]
        same_channel = prev.get("channel") == seg.get("channel")
        gap = seg["start_min"] - prev["end_min"]

        if same_label and same_channel and gap <= max_gap_min:
            prev["end_min"] = max(prev["end_min"], seg["end_min"])
        else:
            merged.append(seg.copy())

    return merged


def drop_tiny_segments(segments, min_duration_min=0.002):
    out = []
    for seg in segments:
        dur = seg["end_min"] - seg["start_min"]
        if dur >= min_duration_min:
            out.append(seg)
    return out


def subtract_interval(seg, blocker):
    s0, s1 = seg["start_min"], seg["end_min"]
    b0, b1 = blocker["start_min"], blocker["end_min"]

    if b1 <= s0 or b0 >= s1:
        return [seg]

    pieces = []

    if b0 > s0:
        left = seg.copy()
        left["end_min"] = b0
        if left["end_min"] > left["start_min"]:
            pieces.append(left)

    if b1 < s1:
        right = seg.copy()
        right["start_min"] = b1
        if right["end_min"] > right["start_min"]:
            pieces.append(right)

    return pieces


def subtract_blockers(segments, blockers):
    result = []
    for seg in segments:
        pieces = [seg]
        for blocker in blockers:
            new_pieces = []
            for piece in pieces:
                new_pieces.extend(subtract_interval(piece, blocker))
            pieces = new_pieces
            if not pieces:
                break
        result.extend(pieces)
    return result


def get_segments_for_column(col, manual_segments, bluetooth_segments, adc_disconnect_segments, imu2_disconnect_segments):
    manual_segments = [s.copy() for s in manual_segments]
    bluetooth_segments = [s.copy() for s in bluetooth_segments]
    adc_segments = [s.copy() for s in adc_disconnect_segments if s.get("channel") == col]
    imu2_segments = [s.copy() for s in imu2_disconnect_segments]

    is_imu_plot = col in [
        "IMU1_H", "IMU1_P", "IMU1_R",
        "IMU2_H", "IMU2_P", "IMU2_R",
        "IMU1_W", "IMU1_X", "IMU1_Y", "IMU1_Z",
        "IMU2_W", "IMU2_X", "IMU2_Y", "IMU2_Z",
    ]
    is_adc_plot = col in ["ADC_ch0", "ADC_ch1"]

    blockers = []
    final_segments = []

    final_segments += bluetooth_segments
    blockers += bluetooth_segments

    if is_imu_plot:
        imu2_segments = subtract_blockers(imu2_segments, blockers)
        final_segments += imu2_segments
        blockers += imu2_segments

    if is_adc_plot:
        adc_segments = subtract_blockers(adc_segments, blockers)
        final_segments += adc_segments
        blockers += adc_segments

    manual_segments = subtract_blockers(manual_segments, blockers)
    final_segments += manual_segments

    return sorted(final_segments, key=lambda s: (s["start_min"], s["end_min"]))


def add_segment_shading(ax, segments, label_colors=None, x_units="elapsed_min"):
    if label_colors is None:
        label_colors = {}

    used = set()
    scale = 1 / 60 if x_units == "elapsed_hr" else 1.0

    for seg in segments:
        label = seg["label"]
        color = label_colors.get(label, "lightgray")

        if label in ["sensor disconnected", "IMU2 disconnected"]:
            alpha = 0.70
        elif label == "bluetooth down":
            alpha = 0.40
        else:
            alpha = 0.40

        x0 = seg["start_min"] * scale
        x1 = seg["end_min"] * scale

        show_label = label if label not in used else None
        ax.axvspan(x0, x1, color=color, alpha=alpha, label=show_label, zorder=0)
        used.add(label)


def add_event_markers(
    ax,
    segments,
    color_map=None,
    y_text=0.98,
    rotation=90,
    label_every_event=False,
    linestyle="--",
    linewidth=1.2,
    x_units="elapsed_min",
):
    if color_map is None:
        color_map = {}

    used_labels = set()
    scale = 1 / 60 if x_units == "elapsed_hr" else 1.0

    for seg in segments:
        label = seg["label"]
        x = seg["start_min"] * scale
        color = color_map.get(label, "red")

        ax.axvline(x, color=color, linestyle=linestyle, linewidth=linewidth, alpha=0.95, zorder=4)

        show_text = label_every_event or (label not in used_labels)
        if show_text:
            ax.text(
                x,
                y_text,
                label,
                rotation=rotation,
                color=color,
                fontsize=8,
                ha="left",
                va="top",
                transform=ax.get_xaxis_transform(),
                zorder=5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7, pad=1.2),
            )

        used_labels.add(label)


def plot_group(
    df_plot,
    cols,
    title,
    manual_segments=None,
    bluetooth_segments=None,
    adc_disconnect_segments=None,
    imu2_disconnect_segments=None,
    label_colors=None,
    display_name_map=None,
    group_title_map=None,
    adc_ylim_map=None,
    xcol="elapsed_min",
):
    if manual_segments is None:
        manual_segments = []
    if bluetooth_segments is None:
        bluetooth_segments = []
    if adc_disconnect_segments is None:
        adc_disconnect_segments = []
    if imu2_disconnect_segments is None:
        imu2_disconnect_segments = []
    if label_colors is None:
        label_colors = {}
    if display_name_map is None:
        display_name_map = {}
    if group_title_map is None:
        group_title_map = {}
    if adc_ylim_map is None:
        adc_ylim_map = {}

    fig, axes = plt.subplots(len(cols), 1, figsize=(15, 2.6 * len(cols)), sharex=True)
    if len(cols) == 1:
        axes = [axes]

    for ax, col in zip(axes, cols):
        ax.plot(df_plot[xcol], df_plot[col], linewidth=0.8, zorder=2)
        ax.set_ylabel(display_name_map.get(col, col))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        col_segments = get_segments_for_column(
            col,
            manual_segments=manual_segments,
            bluetooth_segments=bluetooth_segments,
            adc_disconnect_segments=adc_disconnect_segments,
            imu2_disconnect_segments=imu2_disconnect_segments,
        )

        add_segment_shading(ax, col_segments, label_colors, x_units=xcol)

        disconnect_like = [seg for seg in col_segments if seg["label"] in ["sensor disconnected", "IMU2 disconnected"]]
        bluetooth_like = [seg for seg in col_segments if seg["label"] == "bluetooth down"]

        add_event_markers(
            ax,
            disconnect_like,
            color_map=label_colors,
            label_every_event=True,
            linestyle="--",
            linewidth=1.3,
            x_units=xcol,
        )

        add_event_markers(
            ax,
            bluetooth_like,
            color_map=label_colors,
            label_every_event=True,
            linestyle="--",
            linewidth=1.1,
            x_units=xcol,
        )

        if col in adc_ylim_map:
            ax.set_ylim(*adc_ylim_map[col])

    axes[0].set_title(group_title_map.get(title, title))

    if xcol == "elapsed_min":
        axes[-1].set_xlabel("Time from start (min)")
    elif xcol == "elapsed_hr":
        axes[-1].set_xlabel("Time from start (hr)")
    else:
        axes[-1].set_xlabel(xcol)

    legend_exclude = {"bluetooth down", "sensor disconnected", "IMU2 disconnected"}
    legend_items = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l in legend_exclude:
                continue
            if l not in legend_items:
                legend_items[l] = h

    if legend_items:
        axes[0].legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            loc="upper right",
            ncol=min(4, len(legend_items))
        )

    plt.tight_layout()
    plt.show()


def plot_all_signals_one_figure(
    df_plot,
    plot_groups,
    manual_segments=None,
    bluetooth_segments=None,
    adc_disconnect_segments=None,
    imu2_disconnect_segments=None,
    label_colors=None,
    display_name_map=None,
    group_title_map=None,
    adc_ylim_map=None,
    xcol="elapsed_min",
):
    if manual_segments is None:
        manual_segments = []
    if bluetooth_segments is None:
        bluetooth_segments = []
    if adc_disconnect_segments is None:
        adc_disconnect_segments = []
    if imu2_disconnect_segments is None:
        imu2_disconnect_segments = []
    if label_colors is None:
        label_colors = {}
    if display_name_map is None:
        display_name_map = {}
    if group_title_map is None:
        group_title_map = {}
    if adc_ylim_map is None:
        adc_ylim_map = {}

    ordered = []
    for group_name, cols in plot_groups.items():
        for col in cols:
            ordered.append((group_name, col))

    n = len(ordered)
    fig, axes = plt.subplots(n, 1, figsize=(16, 2.1 * n), sharex=True)
    if n == 1:
        axes = [axes]

    prev_group = None

    for ax, (group_name, col) in zip(axes, ordered):
        ax.plot(df_plot[xcol], df_plot[col], linewidth=0.8, zorder=2)
        ax.set_ylabel(display_name_map.get(col, col), fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

        if group_name != prev_group:
            ax.set_title(group_title_map.get(group_name, group_name), loc="left", fontsize=11, fontweight="bold")
            prev_group = group_name

        col_segments = get_segments_for_column(
            col,
            manual_segments=manual_segments,
            bluetooth_segments=bluetooth_segments,
            adc_disconnect_segments=adc_disconnect_segments,
            imu2_disconnect_segments=imu2_disconnect_segments,
        )

        add_segment_shading(ax, col_segments, label_colors, x_units=xcol)

        disconnect_like = [seg for seg in col_segments if seg["label"] in ["sensor disconnected", "IMU2 disconnected"]]
        bluetooth_like = [seg for seg in col_segments if seg["label"] == "bluetooth down"]

        add_event_markers(
            ax,
            disconnect_like,
            color_map=label_colors,
            label_every_event=True,
            linestyle="--",
            linewidth=1.3,
            x_units=xcol,
        )

        add_event_markers(
            ax,
            bluetooth_like,
            color_map=label_colors,
            label_every_event=True,
            linestyle="--",
            linewidth=1.1,
            x_units=xcol,
        )

        if col in adc_ylim_map:
            ax.set_ylim(*adc_ylim_map[col])

    if xcol == "elapsed_min":
        axes[-1].set_xlabel("Time from start (min)")
    elif xcol == "elapsed_hr":
        axes[-1].set_xlabel("Time from start (hr)")
    else:
        axes[-1].set_xlabel(xcol)

    legend_exclude = {"bluetooth down", "sensor disconnected", "IMU2 disconnected"}
    legend_items = {}
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        for h, l in zip(handles, labels):
            if l in legend_exclude:
                continue
            if l not in legend_items:
                legend_items[l] = h

    if legend_items:
        axes[0].legend(
            list(legend_items.values()),
            list(legend_items.keys()),
            loc="upper right",
            ncol=min(4, len(legend_items))
        )

    plt.tight_layout()
    plt.show()

def keep_min_duration_segments(segments, min_duration_min=0.0083):
    """
    Keep only segments at least this long.
    0.0083 min ~= 0.5 s
    """
    out = []
    for seg in segments:
        dur = seg["end_min"] - seg["start_min"]
        if dur >= min_duration_min:
            out.append(seg)
    return out