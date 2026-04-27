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


# Maps derived angle columns back to their source ADC channel so that
# ADC disconnect shading is correctly inherited by the angle plots.
_ADC_DERIVED_CHANNEL_MAP = {
    "index_mcp_deg": "ADC_ch0",
    "thumb_mp_deg":  "ADC_ch1",
}


def get_segments_for_column(col, manual_segments, bluetooth_segments, adc_disconnect_segments, imu2_disconnect_segments):
    manual_segments = [s.copy() for s in manual_segments]
    bluetooth_segments = [s.copy() for s in bluetooth_segments]

    # Derived angle columns inherit disconnect shading from their source ADC channel
    channel_for_adc = _ADC_DERIVED_CHANNEL_MAP.get(col, col)
    adc_segments = [s.copy() for s in adc_disconnect_segments if s.get("channel") == channel_for_adc]

    imu2_segments = [s.copy() for s in imu2_disconnect_segments]

    is_imu_plot = col in [
        "IMU1_H", "IMU1_P", "IMU1_R",
        "IMU2_H", "IMU2_P", "IMU2_R",
        "IMU1_W", "IMU1_X", "IMU1_Y", "IMU1_Z",
        "IMU2_W", "IMU2_X", "IMU2_Y", "IMU2_Z",
        "wrist_flex_ext_deg",
        "imu_bend_deg",
        "imu_pitch_deg",
        "imu_azimuth_deg",
    ]
    is_adc_plot = col in ["ADC_ch0", "ADC_ch1"] or col in _ADC_DERIVED_CHANNEL_MAP

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
            alpha = 0.40
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


def add_wrist_flex_ext_from_imus(
    df,
    *,
    quat_order="wxyz",
    fixed_axis="y",
    moving_axis="y",
    plane_normal_axis="z",
    fe_source_col="imu_azimuth_deg", # or '_bend_' or '_pitch_'
    out_col="wrist_flex_ext_deg",
    zero_baseline=True,
    baseline_window_sec=1.0,
    baseline_stat="median",   # "median" or "mean" ?
    abs_value=False,
    sign=1.0,
):
    """
    Derive a wrist flexion/extension trace from IMU1 + IMU2 quaternions using the same quaternion math as analysis_imu_cam.py.

    Will compute azimuth, bend, and pitch -- can visaully inspect 

    Assumes the unified wear CSV has columns:
      IMU1_W, IMU1_X, IMU1_Y, IMU1_Z,
      IMU2_W, IMU2_X, IMU2_Y, IMU2_Z

    Returns a copy of df with:
      - imu_bend_deg
      - imu_pitch_deg
      - imu_azimuth_deg
      - wrist_flex_ext_deg   (default = signed imu_azimuth_deg)
    """
    required = [
        "IMU1_W", "IMU1_X", "IMU1_Y", "IMU1_Z",
        "IMU2_W", "IMU2_X", "IMU2_Y", "IMU2_Z",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required IMU quaternion columns: {missing}")

    d = df.copy()

    # Alias wear-file column names to what IMU_cam expects
    alias_map = {
        "IMU1_W": "euler1_w",
        "IMU1_X": "euler1_x",
        "IMU1_Y": "euler1_y",
        "IMU1_Z": "euler1_z",
        "IMU2_W": "euler2_w",
        "IMU2_X": "euler2_x",
        "IMU2_Y": "euler2_y",
        "IMU2_Z": "euler2_z",
    }
    for src, dst in alias_map.items():
        d[dst] = pd.to_numeric(d[src], errors="coerce")

    # use the existing quaternion-angle pipeline
    from analysis_imu_cam import IMU_cam
    imu = IMU_cam.__new__(IMU_cam)

    aug_trials, _ = imu.compute_imu_bend_pitch_azimuth_by_trial(
        [d],
        set_label="wear",
        quat_cols=("euler1", "euler2"),
        fixed_axis=fixed_axis,
        moving_axis=moving_axis,
        plane_normal_axis=plane_normal_axis,
        quat_order=quat_order,
        bend_col="imu_bend_deg",
        pitch_col="imu_pitch_deg",
        azim_col="imu_azimuth_deg",
        time_col="elapsed_sec",
        trial_len_sec=None,
        zero_baseline_bend=False,
        zero_baseline_pitch=False,
        zero_baseline_azim=False,
    )

    d = aug_trials[0]

    # For structured on-hand testing, azimuth was the useful FE-like trace -- not sure now -- manually inspect
    if fe_source_col not in d.columns:
        raise KeyError(
            f"fe_source_col='{fe_source_col}' not found. "
            f"Valid options: 'imu_bend_deg', 'imu_pitch_deg', 'imu_azimuth_deg'."
        )
    d[out_col] = sign * pd.to_numeric(d[fe_source_col], errors="coerce")

    if zero_baseline:
        if "elapsed_sec" in d.columns:
            mask0 = d["elapsed_sec"] <= (d["elapsed_sec"].min() + baseline_window_sec)
            base_vals = pd.to_numeric(d.loc[mask0, out_col], errors="coerce").dropna()
        else:
            base_vals = pd.to_numeric(d[out_col], errors="coerce").dropna().iloc[:200]

        if len(base_vals) > 0:
            baseline = base_vals.median() if baseline_stat == "median" else base_vals.mean()
            d[out_col] = d[out_col] - baseline

    if abs_value:
        d[out_col] = d[out_col].abs()

    return d

from scipy.signal import welch, find_peaks

def build_activity_segments_df(manual_segments):
    """Convert manual_segments list-of-dicts → tidy DataFrame with segment_id."""
    if not manual_segments:
        return pd.DataFrame(columns=["segment_id", "label", "start_min", "end_min", "duration_sec"])
    seg_df = pd.DataFrame(manual_segments).copy()
    seg_df = seg_df.sort_values(["start_min", "end_min"]).reset_index(drop=True)
    seg_df["segment_id"] = np.arange(1, len(seg_df) + 1)
    seg_df["duration_sec"] = 60.0 * (seg_df["end_min"] - seg_df["start_min"])
    return seg_df[["segment_id", "label", "start_min", "end_min", "duration_sec"]]


def print_label_table(manual_segments):
    """
    Print a readable label table from manual_segments.

    Example output
    --------------
    #   label      start (min)   end (min)   duration
    1   walking       22.44        37.56       15 min 5 s
    2   eating        37.56        51.32       13 min 45 s
    ...
    """
    if not manual_segments:
        print("No manual segments defined.")
        return

    seg_df = build_activity_segments_df(manual_segments)

    header = f"{'#':<4}  {'label':<14}  {'start (min)':>11}  {'end (min)':>9}  {'duration':>12}"
    print(header)
    print("-" * len(header))

    for _, row in seg_df.iterrows():
        dur_s  = row["duration_sec"]
        mins   = int(dur_s // 60)
        secs   = int(dur_s % 60)
        dur_str = f"{mins} min {secs:02d} s" if mins else f"{secs} s"
        print(
            f"{int(row['segment_id']):<4}  {row['label']:<14}  "
            f"{row['start_min']:>11.3f}  {row['end_min']:>9.3f}  {dur_str:>12}"
        )

def slice_df_by_segment(df, start_min, end_min, pad_sec=0.0):
    """Return rows of df within [start_min - pad, end_min + pad]."""
    pad_min = pad_sec / 60.0
    return df.loc[
        (df["elapsed_min"] >= start_min - pad_min) &
        (df["elapsed_min"] <= end_min   + pad_min)
    ].copy()


def _label_for_window(manual_segments, start_min, end_min):
    """
    Return a title string describing which activity labels overlap [start_min, end_min].

    If the window sits entirely inside one segment → "walking"
    If it spans multiple               → "walking / eating"
    If none overlap                    → "" (empty)
    """
    hits = []
    seen = set()
    for seg in sorted(manual_segments, key=lambda s: s["start_min"]):
        # overlap check
        if seg["end_min"] <= start_min or seg["start_min"] >= end_min:
            continue
        lbl = seg["label"]
        if lbl not in seen:
            hits.append(lbl)
            seen.add(lbl)
    return " / ".join(hits)


def plot_time_window(
    df,
    start_min,
    end_min,
    columns=None,
    manual_segments=None,
    label_colors=None,
    display_name_map=None,
    xcol="elapsed_min",
):
    """
    Plot a specific time window with activity shading and an auto-generated title.

    The title shows which activity label(s) fall within the window, so you can
    call this repeatedly with different start/end times to inspect each segment.

    Requires ``%matplotlib widget`` (ipympl) in the calling cell for pan/zoom.

    Parameters
    ----------
    df            : DataFrame containing the signals
    start_min     : window start in minutes
    end_min       : window end in minutes
    columns       : list of column names to plot (one subplot each)
    manual_segments : list-of-dicts used to determine the title label
    label_colors  : dict label → hex colour for shading
    display_name_map : dict col → display name for y-axis labels
    xcol          : x-axis column (default "elapsed_min")
    """
    manual_segments  = manual_segments  or []
    label_colors     = label_colors     or {}
    display_name_map = display_name_map or {}

    if columns is None:
        columns = [c for c in ["wrist_flex_ext_deg", "index_mcp_deg", "thumb_mp_deg",
                                "ADC_ch0", "ADC_ch1"] if c in df.columns]
    columns = [c for c in columns if c in df.columns]
    if not columns:
        raise ValueError("No valid columns found in df.")

    df_win = slice_df_by_segment(df, start_min, end_min)
    activity_label = _label_for_window(manual_segments, start_min, end_min)
    dur_s  = (end_min - start_min) * 60
    title  = (
        f"{activity_label}  |  {start_min:.2f} - {end_min:.2f} min  "
        f"({dur_s:.0f} s)"
        if activity_label
        else f"{start_min:.2f} - {end_min:.2f} min  ({dur_s:.0f} s)"
    )

    fig, axes = plt.subplots(len(columns), 1,
                             figsize=(14, 2.4 * len(columns)),
                             sharex=True)
    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        ax.plot(df_win[xcol], df_win[col], linewidth=0.9, zorder=2)
        ax.set_ylabel(display_name_map.get(col, col), fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # shade any segments that overlap this window
        for seg in manual_segments:
            x0 = max(seg["start_min"], start_min)
            x1 = min(seg["end_min"],   end_min)
            if x1 <= x0:
                continue
            color = label_colors.get(seg["label"], "lightgray")
            ax.axvspan(x0, x1, color=color, alpha=0.30, zorder=0)

    plt.title(activity_label)

    axes[0].set_title(title, fontsize=11)
    axes[-1].set_xlabel("Time (min)")
    plt.tight_layout()
    plt.show()


import re


def set_segments_to_fixed_duration(
    segments,
    *,
    duration_sec=10.0,
    end_cap_min=None,
):
    """
    Return a copy of ``segments`` where every segment duration is forced to
    ``duration_sec`` from its start time.
    """
    dur_min = float(duration_sec) / 60.0
    out = []

    for seg in segments:
        seg2 = dict(seg)
        start_min = float(seg2["start_min"])
        end_min = start_min + dur_min
        if end_cap_min is not None:
            end_min = min(end_min, float(end_cap_min))
        seg2["end_min"] = end_min
        out.append(seg2)

    return out


def extract_calibration_segments_from_notes(
    note_rows,
    *,
    duration_sec=10.0,
    end_cap_min=None,
    time_col="elapsed_min",
    note_col="Note",
    sensor_channel_map=None,
):
    """
    Parse note-labeled calibration holds from a unified kinwatch CSV.

    Expected note styles include:
      - "channel 0 is index"
      - "START 0"
      - "START 22.5"
      - "Thumb START 45"
      - "Index START 67.5"
      - "Above was index"

    Plain "START X" notes inherit the most recent sensor context.
    Returned segments are fixed-duration windows starting at each START note.
    """
    if sensor_channel_map is None:
        sensor_channel_map = {"index": "ADC_ch0", "thumb": "ADC_ch1"}

    if note_rows is None or len(note_rows) == 0:
        return []

    notes = note_rows.copy()
    notes = notes.sort_values(time_col).reset_index(drop=True)

    current_sensor = "index"
    out = []

    channel_sensor_pat = re.compile(
        r"channel\s*(\d+)\s+is\s+(index|thumb)", flags=re.IGNORECASE
    )
    explicit_start_pat = re.compile(
        r"^(index|thumb)\s+start\s+(-?\d+(?:\.\d+)?)$",
        flags=re.IGNORECASE,
    )
    plain_start_pat = re.compile(
        r"^start\s+(-?\d+(?:\.\d+)?)$",
        flags=re.IGNORECASE,
    )
    above_was_pat = re.compile(r"above\s+was\s+(index|thumb)", flags=re.IGNORECASE)

    for _, row in notes.iterrows():
        note = str(row.get(note_col, "")).strip()
        if not note:
            continue

        m_chan = channel_sensor_pat.search(note)
        if m_chan:
            ch_num = int(m_chan.group(1))
            sensor = m_chan.group(2).lower()
            sensor_channel_map[sensor] = f"ADC_ch{ch_num}"
            current_sensor = sensor
            continue

        m_above = above_was_pat.search(note)
        if m_above:
            current_sensor = m_above.group(1).lower()
            continue

        sensor = None
        angle_deg = None

        m_explicit = explicit_start_pat.match(note)
        if m_explicit:
            sensor = m_explicit.group(1).lower()
            angle_deg = float(m_explicit.group(2))
        else:
            m_plain = plain_start_pat.match(note)
            if m_plain:
                sensor = current_sensor
                angle_deg = float(m_plain.group(1))

        if sensor is None or angle_deg is None:
            continue

        start_min = float(row[time_col])
        end_min = start_min + float(duration_sec) / 60.0
        if end_cap_min is not None:
            end_min = min(end_min, float(end_cap_min))

        label_angle = int(angle_deg) if float(angle_deg).is_integer() else angle_deg
        out.append(
            {
                "start_min": start_min,
                "end_min": end_min,
                "label": f"{sensor} {label_angle}",
                "sensor": sensor,
                "channel": sensor_channel_map.get(sensor),
                "angle_deg": float(angle_deg),
                "note": note,
            }
        )

    return out


def build_adc_calibration_table(
    df,
    calib_segments,
    *,
    time_col="elapsed_min",
    adc_col_override=None,
    agg="mean",
):
    """
    Compute one calibration point per labeled hold segment.
    """
    if df is None or df.empty or not calib_segments:
        return pd.DataFrame(
            columns=[
                "sensor", "channel", "angle_deg",
                "start_min", "end_min", "adc_mean", "adc_std", "n", "label",
            ]
        )

    rows = []
    for seg in calib_segments:
        adc_col = adc_col_override or seg.get("channel")
        if adc_col is None or adc_col not in df.columns:
            continue

        start_min = float(seg["start_min"])
        end_min = float(seg["end_min"])

        mask = (
            (df[time_col] >= start_min) &
            (df[time_col] <= end_min)
        )
        vals = pd.to_numeric(df.loc[mask, adc_col], errors="coerce").dropna()

        if vals.empty:
            continue

        if agg == "median":
            adc_mean = float(vals.median())
        else:
            adc_mean = float(vals.mean())

        rows.append(
            {
                "sensor": seg.get("sensor"),
                "channel": adc_col,
                "angle_deg": float(seg.get("angle_deg")),
                "start_min": start_min,
                "end_min": end_min,
                "adc_mean": adc_mean,
                "adc_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                "n": int(len(vals)),
                "label": seg.get("label"),
            }
        )

    calib_df = pd.DataFrame(rows)
    if calib_df.empty:
        return calib_df

    return calib_df.sort_values(["sensor", "angle_deg", "start_min"]).reset_index(drop=True)


def build_adc_calibration_table_from_spec(
    df,
    calib_spec_df,
    *,
    time_col="elapsed_min",
    agg="mean",
):
    rows = []

    if df is None or df.empty or calib_spec_df is None or calib_spec_df.empty:
        return pd.DataFrame()

    for _, seg in calib_spec_df.iterrows():
        adc_col = seg["channel"]
        start_min = float(seg["start_min"])
        end_min = float(seg["end_min"])

        if adc_col not in df.columns:
            continue

        mask = (df[time_col] >= start_min) & (df[time_col] <= end_min)
        vals = pd.to_numeric(df.loc[mask, adc_col], errors="coerce").dropna()

        if vals.empty:
            continue

        if agg == "median":
            adc_mean = float(vals.median())
        else:
            adc_mean = float(vals.mean())

        rows.append({
            "set_label": seg.get("set_label"),
            "sensor": seg.get("sensor"),
            "channel": adc_col,
            "angle_deg": float(seg.get("angle_deg")),
            "start_min": start_min,
            "end_min": end_min,
            "adc_mean": adc_mean,
            "adc_std": float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
            "n": int(len(vals)),
            "label": seg.get("label"),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    return out.sort_values(["set_label", "sensor", "angle_deg", "start_min"]).reset_index(drop=True)



def _transform_adc_for_model(
    adc_vals,
    model,
    *,
    clip_normalized=False,
):
    """
    Convert raw ADC values into the input space expected by ``model``.

    If the model was fit with ``normalize_adc=True``, this maps raw ADC to:
        (adc - adc_min) / (adc_max - adc_min)

    using the fit-set min/max stored in the model. Optionally clips the
    normalized values to [0, 1].
    """
    vals = np.asarray(adc_vals, dtype=float)

    if model is None or not model.get("normalize_adc", False):
        return vals

    adc_min = float(model["adc_min"])
    adc_max = float(model["adc_max"])
    denom = adc_max - adc_min

    if denom == 0 or not np.isfinite(denom):
        out = np.full_like(vals, np.nan, dtype=float)
    else:
        out = (vals - adc_min) / denom
        if clip_normalized:
            out = np.clip(out, 0.0, 1.0)

    out[np.isnan(vals)] = np.nan
    return out


def _evaluate_adc_angle_model(
    model,
    adc_vals,
    *,
    clip_normalized=False,
):
    """
    Evaluate an ADC -> angle model on raw ADC input values.
    """
    adc_vals = np.asarray(adc_vals, dtype=float)
    x_eval = _transform_adc_for_model(
        adc_vals,
        model,
        clip_normalized=clip_normalized,
    )
    theta = np.asarray(model["poly"](x_eval), dtype=float)
    theta[np.isnan(adc_vals)] = np.nan
    return theta


def fit_adc_angle_models(
    calib_df,
    *,
    poly_order=2,
    group_col="sensor",
    normalize_adc=False,
):
    """
    Fit ADC -> angle polynomial models.

    If ``normalize_adc=True``, the model is fit in normalized-ADC space using
    the fit-set's own raw ADC min/max:
        adc_norm = (adc - adc_min) / (adc_max - adc_min)

    The raw ``adc_min`` / ``adc_max`` from the chosen fit set are stored in the
    model and later reused when applying the model to wear data.
    """
    models = {}
    if calib_df is None or calib_df.empty:
        return models

    for group_value, sub in calib_df.groupby(group_col):
        sub = sub.dropna(subset=["adc_mean", "angle_deg"]).copy()
        sub = sub.sort_values("angle_deg")

        if len(sub) < 2:
            continue

        fit_order = min(int(poly_order), len(sub) - 1)
        x_raw = sub["adc_mean"].to_numpy(dtype=float)
        y = sub["angle_deg"].to_numpy(dtype=float)

        adc_min = float(np.nanmin(x_raw))
        adc_max = float(np.nanmax(x_raw))
        denom = adc_max - adc_min

        if normalize_adc:
            if denom == 0 or not np.isfinite(denom):
                print(
                    f"Skipping {group_value}: cannot normalize because "
                    f"adc_max == adc_min."
                )
                continue
            x_fit = (x_raw - adc_min) / denom
        else:
            x_fit = x_raw

        coeffs = np.polyfit(x_fit, y, fit_order)
        models[group_value] = {
            "coeffs": coeffs,
            "poly": np.poly1d(coeffs),
            "fit_order": fit_order,
            "channel": sub["channel"].iloc[0] if "channel" in sub.columns else None,
            "adc_min": adc_min,
            "adc_max": adc_max,
            "angle_min": float(np.nanmin(y)),
            "angle_max": float(np.nanmax(y)),
            "n_points": int(len(sub)),
            "normalize_adc": bool(normalize_adc),
            "calib_df": sub.reset_index(drop=True),
        }

    return models


def summarize_adc_angle_models(models):
    """
    Convert model dict from ``fit_adc_angle_models`` into a small summary table.
    """
    rows = []
    for sensor, model in (models or {}).items():
        rows.append(
            {
                "sensor": sensor,
                "channel": model.get("channel"),
                "fit_order": model.get("fit_order"),
                "n_points": model.get("n_points"),
                "normalize_adc": bool(model.get("normalize_adc", False)),
                "adc_min": model.get("adc_min"),
                "adc_max": model.get("adc_max"),
                "angle_min": model.get("angle_min"),
                "angle_max": model.get("angle_max"),
                "coeffs": np.array2string(
                    np.asarray(model.get("coeffs")),
                    precision=6,
                    separator=", ",
                ),
            }
        )
    return pd.DataFrame(rows)


def apply_adc_angle_models(
    df,
    models,
    *,
    source_col_map=None,
    output_col_map=None,
    clamp=True,
    clamp_range_map=None,
    clip_normalized=True,
):
    """
    Apply fitted ADC -> angle polynomials to a dataframe.

    If a model was fit with ``normalize_adc=True``, raw ADC is first normalized
    using that model's stored fit-set ``adc_min`` / ``adc_max``. If
    ``clip_normalized=True``, normalized ADC is clipped to [0, 1] before the
    polynomial is evaluated.
    """
    if output_col_map is None:
        output_col_map = {"index": "index_mcp_deg", "thumb": "thumb_mp_deg"}

    out = df.copy()

    for sensor, model in (models or {}).items():
        src_col = (
            source_col_map.get(sensor)
            if source_col_map is not None and sensor in source_col_map
            else model.get("channel")
        )
        out_col = output_col_map.get(sensor, f"{sensor}_deg")

        if src_col is None or src_col not in out.columns:
            continue

        adc_vals = pd.to_numeric(out[src_col], errors="coerce").to_numpy(dtype=float)
        theta = _evaluate_adc_angle_model(
            model,
            adc_vals,
            clip_normalized=clip_normalized,
        )

        if clamp:
            if clamp_range_map is not None and sensor in clamp_range_map:
                lo, hi = clamp_range_map[sensor]
            else:
                lo = model.get("angle_min", np.nanmin(theta))
                hi = model.get("angle_max", np.nanmax(theta))
            theta = np.clip(theta, lo, hi)

        theta[np.isnan(adc_vals)] = np.nan
        out[out_col] = theta

    return out


def plot_adc_calibration_models(
    calib_df,
    models,
    *,
    group_col="sensor",
    figsize=(10, 4),
    normal=False,
):
    """
    Scatter calibration points and overlay fit with:
    x-axis = angle (deg)
    y-axis = ADC value, or normalized ADC if normal=True
    """
    import numpy as np
    import matplotlib.pyplot as plt

    if calib_df is None or calib_df.empty:
        print("No calibration points to plot.")
        return

    sensors = list(calib_df[group_col].dropna().unique())
    if not sensors:
        print("No calibration groups found.")
        return

    fig, axes = plt.subplots(1, len(sensors), figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for ax, sensor in zip(axes, sensors):
        sub = calib_df[calib_df[group_col] == sensor].copy()
        sub = sub.sort_values("angle_deg")

        model = (models or {}).get(sensor)

        if normal:
            if model is None:
                print(f"Skipping normalization for {sensor}: no model found.")
                y_vals = sub["adc_mean"].to_numpy(dtype=float)
                y_label = "ADC value"
            else:
                adc_min = float(model["adc_min"])
                adc_max = float(model["adc_max"])
                denom = adc_max - adc_min
                if denom == 0:
                    print(f"Skipping normalization for {sensor}: adc_max == adc_min.")
                    y_vals = sub["adc_mean"].to_numpy(dtype=float)
                    y_label = "ADC value"
                else:
                    y_vals = (sub["adc_mean"].to_numpy(dtype=float) - adc_min) / denom
                    y_label = "Normalized ADC"
        else:
            y_vals = sub["adc_mean"].to_numpy(dtype=float)
            y_label = "ADC value"

        ax.scatter(sub["angle_deg"], y_vals, s=35, alpha=0.9)

        for (_, row), y in zip(sub.iterrows(), y_vals):
            ax.annotate(
                f"{row['angle_deg']:.1f}°",
                (row["angle_deg"], y),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8,
            )

        if model is not None:
            adc_grid = np.linspace(
                min(sub["adc_mean"].min(), model["adc_min"]),
                max(sub["adc_mean"].max(), model["adc_max"]),
                300,
            )
            angle_grid = _evaluate_adc_angle_model(model, adc_grid, clip_normalized=False)

            if normal:
                adc_min = float(model["adc_min"])
                adc_max = float(model["adc_max"])
                denom = adc_max - adc_min
                if denom != 0:
                    y_grid = (adc_grid - adc_min) / denom
                else:
                    y_grid = adc_grid
            else:
                y_grid = adc_grid

            ax.plot(angle_grid, y_grid, linewidth=2)

        ax.set_title(f"{sensor.capitalize()} calibration")
        ax.set_xlabel("Angle (°)")
        ax.set_ylabel(y_label)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.show()

def plot_calibration_sets_by_sensor(
    df_raw,
    calib_spec_df,
    calib_summary_df,
    models,
    *,
    time_col="elapsed_min",
    figsize=(12, 5),
    normal=False,
    alpha_raw=0.18,
    raw_marker_size=10,
    mean_marker_size=42,
    line_width=2.0,
    set_order=None,
    set_color_map=None,
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if calib_spec_df is None or len(calib_spec_df) == 0:
        print("No calibration spec rows to plot.")
        return
    if calib_summary_df is None or calib_summary_df.empty:
        print("No calibration summary table to plot.")
        return

    spec = calib_spec_df.copy()
    summ = calib_summary_df.copy()

    if set_order is None:
        set_order = list(pd.unique(spec["set_label"]))

    if set_color_map is None:
        default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        set_color_map = {
            label: default_colors[i % len(default_colors)]
            for i, label in enumerate(set_order)
        }

    sensors_present = [s for s in ["index", "thumb"] if s in spec["sensor"].unique()]
    if not sensors_present:
        print("No index/thumb sensors found.")
        return

    fig, axes = plt.subplots(1, len(sensors_present), figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for ax, sensor in zip(axes, sensors_present):
        spec_sensor = spec[spec["sensor"] == sensor].copy()
        summ_sensor = summ[summ["sensor"] == sensor].copy()

        sensor_set_labels = [lbl for lbl in set_order if lbl in spec_sensor["set_label"].unique()]

        for set_label in sensor_set_labels:
            color = set_color_map.get(set_label, None)
            curve_key = f"{sensor} | {set_label}"
            model = models.get(curve_key)

            spec_sub = spec_sensor[spec_sensor["set_label"] == set_label].copy()
            spec_sub = spec_sub.sort_values("angle_deg")

            summ_sub = summ_sensor[summ_sensor["set_label"] == set_label].copy()
            summ_sub = summ_sub.sort_values("angle_deg")

            # raw points
            raw_x_all = []
            raw_y_all = []

            for _, seg in spec_sub.iterrows():
                adc_col = seg["channel"]
                if adc_col not in df_raw.columns:
                    continue

                start_min = float(seg["start_min"])
                end_min = float(seg["end_min"])
                angle_deg = float(seg["angle_deg"])

                mask = (df_raw[time_col] >= start_min) & (df_raw[time_col] <= end_min)
                vals = pd.to_numeric(df_raw.loc[mask, adc_col], errors="coerce").dropna().to_numpy(dtype=float)

                if len(vals) == 0:
                    continue

                if normal and model is not None:
                    adc_min = float(model["adc_min"])
                    adc_max = float(model["adc_max"])
                    denom = adc_max - adc_min
                    vals_plot = (vals - adc_min) / denom if denom != 0 else vals
                else:
                    vals_plot = vals

                raw_x_all.append(np.full(len(vals_plot), angle_deg))
                raw_y_all.append(vals_plot)

            if raw_x_all:
                ax.scatter(
                    np.concatenate(raw_x_all),
                    np.concatenate(raw_y_all),
                    s=raw_marker_size,
                    alpha=alpha_raw,
                    color=color,
                )

            # mean points
            if not summ_sub.empty:
                mean_x = summ_sub["angle_deg"].to_numpy(dtype=float)
                mean_adc = summ_sub["adc_mean"].to_numpy(dtype=float)

                if normal and model is not None:
                    adc_min = float(model["adc_min"])
                    adc_max = float(model["adc_max"])
                    denom = adc_max - adc_min
                    mean_y = (mean_adc - adc_min) / denom if denom != 0 else mean_adc
                else:
                    mean_y = mean_adc

                ax.scatter(
                    mean_x,
                    mean_y,
                    s=mean_marker_size,
                    alpha=1.0,
                    color=color,
                    edgecolor="none",
                    label=set_label,
                    zorder=3,
                )

            # fitted curve
            if model is not None:
                adc_grid = np.linspace(float(model["adc_min"]), float(model["adc_max"]), 400)
                angle_grid = _evaluate_adc_angle_model(model, adc_grid, clip_normalized=False)

                if normal:
                    adc_min = float(model["adc_min"])
                    adc_max = float(model["adc_max"])
                    denom = adc_max - adc_min
                    y_grid = (adc_grid - adc_min) / denom if denom != 0 else adc_grid
                else:
                    y_grid = adc_grid

                #order = np.argsort(angle_grid)
                ax.plot(
                    angle_grid,
                    y_grid,
                    linewidth=line_width,
                    color=color,
                )

        ax.set_title(f"{sensor.capitalize()} calibration")
        ax.set_xlabel("Angle (°)")
        ax.set_ylabel("Normalized ADC" if normal else "ADC value")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(title="Calibration set")

    plt.tight_layout()
    plt.show()

def plot_activity_window(
    df,
    *,
    start_min,
    window_sec,
    columns=None,
    manual_segments=None,
    label_colors=None,
    display_name_map=None,
    ylim_map=None,
):
    """
    Plot a zoomed time window for activity inspection.

    Inputs
    ------
    start_min : float
        Start time of the window, in minutes from beginning of recording.
    window_sec : float
        Length of the window to show, in seconds.
    columns : list[str] or None
        Signals to plot. If None, uses a default set if present.
    manual_segments : list[dict]
        Used to determine overlapping activity labels and shading.
    label_colors : dict
        Activity label -> color.
    display_name_map : dict
        Column name -> y-axis display label.

    Behavior
    --------
    - X-axis is shown in seconds relative to the chosen window start.
    - Title includes the overlapping activity label(s).
    """
    manual_segments = manual_segments or []
    label_colors = label_colors or {}
    display_name_map = display_name_map or {}
    ylim_map = ylim_map or {}

    end_min = start_min + (window_sec / 60.0)

    if columns is None:
        columns = [
            c for c in [
                "wrist_flex_ext_deg",
                "index_mcp_deg",
                "thumb_mp_deg",
                "ADC_ch0",
                "ADC_ch1",
                "IMU1_W", "IMU1_X", "IMU1_Y", "IMU1_Z",
                "IMU2_W", "IMU2_X", "IMU2_Y", "IMU2_Z",
            ]
            if c in df.columns
        ]
    columns = [c for c in columns if c in df.columns]
    if not columns:
        raise ValueError("No valid columns found in df.")

    df_win = df.loc[
        (df["elapsed_min"] >= start_min) &
        (df["elapsed_min"] <= end_min)
    ].copy()

    if df_win.empty:
        raise ValueError("No rows found in the requested time window.")

    # relative x-axis in seconds
    df_win["window_sec"] = (df_win["elapsed_min"] - start_min) * 60.0

    # figure out overlapping activity label(s)
    overlapping_labels = []
    seen = set()
    for seg in sorted(manual_segments, key=lambda s: s["start_min"]):
        if seg["end_min"] <= start_min or seg["start_min"] >= end_min:
            continue
        lbl = seg["label"]
        if lbl not in seen:
            overlapping_labels.append(lbl)
            seen.add(lbl)

    activity_str = " / ".join(overlapping_labels) if overlapping_labels else "Unlabeled window"
    title = f"{activity_str} | start={start_min:.3f} min | window={window_sec:.0f} s"

    fig, axes = plt.subplots(
        len(columns), 1,
        figsize=(14, 2.5 * len(columns)),
        sharex=True
    )
    if len(columns) == 1:
        axes = [axes]

    for ax, col in zip(axes, columns):
        ax.plot(df_win["window_sec"], df_win[col], linewidth=0.9, zorder=2)
        ax.set_ylabel(display_name_map.get(col, col), fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if col in ylim_map and ylim_map[col] is not None:
            ax.set_ylim(*ylim_map[col])

        # shade overlapping manual activity segments, converted into window-relative seconds
        for seg in manual_segments:
            seg0 = max(seg["start_min"], start_min)
            seg1 = min(seg["end_min"], end_min)
            if seg1 <= seg0:
                continue

            x0 = (seg0 - start_min) * 60.0
            x1 = (seg1 - start_min) * 60.0
            color = label_colors.get(seg["label"], "lightgray")
            ax.axvspan(x0, x1, color=color, alpha=0.30, zorder=0)

    axes[0].set_title(title, fontsize=11)
    axes[-1].set_xlabel("Time from chosen start (s)")
    plt.tight_layout()
    plt.show()


def plot_adc_calibration_models_with_raw(
    df,
    calib_segments,
    models,
    *,
    time_col="elapsed_min",
    group_key="sensor",
    figsize=(10, 4),
    alpha_raw=0.15,
    alpha_mean=1.0,
    normal=False,
):
    """
    Plot all raw ADC values used during each calibration hold, plus mean points
    and fitted curves, with:
    x-axis = angle (deg)
    y-axis = ADC value, or normalized ADC if normal=True
    """
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    if df is None or df.empty or not calib_segments:
        print("No calibration data to plot.")
        return

    seg_df = pd.DataFrame(calib_segments)
    if seg_df.empty:
        print("No calibration segments to plot.")
        return

    sensors = list(seg_df[group_key].dropna().unique())
    if not sensors:
        print("No sensors found in calibration segments.")
        return

    fig, axes = plt.subplots(1, len(sensors), figsize=figsize, squeeze=False)
    axes = axes.ravel()

    for ax, sensor in zip(axes, sensors):
        sensor_segments = seg_df[seg_df[group_key] == sensor].copy()
        sensor_segments = sensor_segments.sort_values("angle_deg")

        model = models.get(sensor)

        mean_x = []
        mean_y = []

        for _, seg in sensor_segments.iterrows():
            adc_col = seg["channel"]
            if adc_col not in df.columns:
                continue

            start_min = float(seg["start_min"])
            end_min = float(seg["end_min"])
            angle_deg = float(seg["angle_deg"])

            mask = (df[time_col] >= start_min) & (df[time_col] <= end_min)
            vals = pd.to_numeric(df.loc[mask, adc_col], errors="coerce").dropna()

            if vals.empty:
                continue

            vals = vals.to_numpy(dtype=float)

            if normal and model is not None:
                adc_min = float(model["adc_min"])
                adc_max = float(model["adc_max"])
                denom = adc_max - adc_min
                if denom != 0:
                    vals_plot = (vals - adc_min) / denom
                else:
                    vals_plot = vals
            else:
                vals_plot = vals

            x = np.full(len(vals_plot), angle_deg, dtype=float)

            ax.scatter(
                x,
                vals_plot,
                s=10,
                alpha=alpha_raw,
            )

            mean_x.append(angle_deg)
            mean_y.append(float(np.mean(vals_plot)))

        if mean_x:
            ax.scatter(
                mean_x,
                mean_y,
                s=45,
                alpha=alpha_mean,
                marker="o",
                label="segment mean",
            )

            for x, y in zip(mean_x, mean_y):
                ax.annotate(
                    f"{x:.1f}°",
                    (x, y),
                    textcoords="offset points",
                    xytext=(4, 4),
                    fontsize=8,
                )

        if model is not None:
            adc_grid = np.linspace(model["adc_min"], model["adc_max"], 400)
            angle_grid = _evaluate_adc_angle_model(model, adc_grid, clip_normalized=False)

            if normal:
                adc_min = float(model["adc_min"])
                adc_max = float(model["adc_max"])
                denom = adc_max - adc_min
                if denom != 0:
                    y_grid = (adc_grid - adc_min) / denom
                else:
                    y_grid = adc_grid
            else:
                y_grid = adc_grid

            ax.plot(angle_grid, y_grid, linewidth=2, label="fit")

        ax.set_title(f"{sensor.capitalize()} calibration")
        ax.set_xlabel("Angle (°)")
        ax.set_ylabel("Normalized ADC" if normal else "ADC value")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend()

    plt.tight_layout()
    plt.show()
