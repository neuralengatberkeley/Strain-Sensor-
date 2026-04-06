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


def plot_pooled_calibration_fit(
    pooled_calib_df,
    *,
    poly_order=2,
    clamp_deg=False,
    adc_scale_factor=1.0,
    deg_min=0.0,
    deg_max=90.0,
    participant_col="participant",
    title="Pooled ADC calibration — all participants & applications",
):
    """
    Diagnostic plot: scatter of all pooled calibration points colored by
    participant, with the fitted polynomial overlaid.

    Uses the same np.polyfit / np.poly1d logic as
    ADC_CAM.calibrate_trials_with_camera(), applied to the concatenated
    calib_df produced in the notebook.

    Parameters
    ----------
    pooled_calib_df : pd.DataFrame
        Concatenated output of ADC_CAM.extract_calib_means_by_set() across
        all participants, with an added 'participant' column for color-coding.
        Must contain columns 'adc_mean' and 'angle_snap_deg'.
    poly_order : int
        Polynomial order for the ADC → angle fit (match what you pass to
        apply_pooled_adc_calibration).
    deg_min, deg_max : float
        Angle range used to draw the fitted curve.
    participant_col : str
        Column used to color-code scatter points (default 'participant').
    title : str
        Figure title.
    """
    df = pooled_calib_df.dropna(subset=["adc_mean", "angle_snap_deg"]).copy()

    if df.empty:
        print("[plot_pooled_calibration_fit] No valid calibration rows — nothing to plot.")
        return

    x_all = df["adc_mean"].to_numpy(dtype=float)
    y_all = df["angle_snap_deg"].to_numpy(dtype=float)
    coeffs = np.polyfit(x_all, y_all, poly_order)
    poly = np.poly1d(coeffs)

    fig, ax = plt.subplots(figsize=(7, 5))

    # Scatter colored by participant
    if participant_col in df.columns:
        participants = df[participant_col].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(participants)))
        for pid, color in zip(participants, colors):
            sub = df[df[participant_col] == pid]
            ax.scatter(sub["adc_mean"], sub["angle_snap_deg"],
                       label=str(pid), alpha=0.6, s=18, color=color, zorder=3)
    else:
        ax.scatter(x_all, y_all, alpha=0.6, s=18, zorder=3, label="calibration points")

    # Fitted curve
    x_curve = np.linspace(x_all.min(), x_all.max(), 300)
    y_curve = np.clip(poly(x_curve), deg_min, deg_max)
    if clamp_deg:
        y_curve = np.clip(y_curve, deg_min, deg_max)
    ax.plot(x_curve, y_curve, color="black", linewidth=2,
            label=f"poly order={poly_order}", zorder=4)

    ax.set_xlabel("ADC value" + (f" × {adc_scale_factor:.2e}" if adc_scale_factor != 1.0 else ""))
    ax.set_ylabel("Angle (°)")
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.show()

    n_parts = df[participant_col].nunique() if participant_col in df.columns else "?"
    print(f"Pooled fit ({len(df)} points, {n_parts} participants)")
    print(f"Coefficients (scaled): {coeffs}")
    print(f"Expected angle at wear-data min ADC "
          f"(inspect x-axis — should overlap the wear ADC range ~1.28-1.38 V)")
    return coeffs


def apply_pooled_adc_calibration(
    df,
    pooled_calib_df,
    *,
    convert_index_mcp=True,
    convert_thumb_mp=False,
    adc_col_index="ADC_ch0",
    adc_col_thumb="ADC_ch1",
    out_col_index="index_mcp_deg",
    out_col_thumb="thumb_mp_deg",
    poly_order=2,
    clamp_deg=True,
    deg_min=0.0,
    deg_max=90.0,
    zero_baseline=True,
    baseline_window_sec=1.0,
    baseline_stat="median",
):
    """
    Convert raw ADC channels to bend angles (degrees) using a polynomial fit
    trained on pooled calibration data from multiple participants/applications.

    The polynomial is fit once on all rows of pooled_calib_df
    (np.polyfit on adc_mean → angle_snap_deg), replicating the logic in
    ADC_CAM.calibrate_trials_with_camera() but across a pooled dataset.

    Important caveats
    -----------------
    - Calibration was collected only for Index MCP (adc_ch3 / ADC_ch0).
    - Thumb MP (ADC_ch1) has no calibration data.  If convert_thumb_mp=True,
      the same Index MCP polynomial is applied as a rough proxy — the shape
      will be meaningful but the absolute values are not validated.
    - Because individual sensor instances differ in resting ADC value,
      zero_baseline=True is best: it expresses angles as
      *degrees of change from the starting posture* rather than absolute angle.

    Parameters
    ----------
    df : pd.DataFrame
        df_plot with ADC_ch0 / ADC_ch1 already NaN-filled for disconnected rows.
    pooled_calib_df : pd.DataFrame
        Concatenated output of ADC_CAM.extract_calib_means_by_set() across
        all participants.  Must contain 'adc_mean' and 'angle_snap_deg'.
    convert_index_mcp : bool
        If True, convert ADC_ch0 → index_mcp_deg.
    convert_thumb_mp : bool
        If True, apply the Index MCP polynomial to ADC_ch1 → thumb_mp_deg.
    poly_order : int
        Polynomial order for the ADC → angle mapping (default 2).
    clamp_deg : bool
        Clamp output to [deg_min, deg_max].
    zero_baseline : bool
        Subtract the median (or mean) of the first baseline_window_sec of
        valid samples so the trace starts near 0.
    baseline_window_sec : float
        Duration of the leading window used for baseline estimation.
    baseline_stat : {"median", "mean"}
        Statistic for the baseline value.

    Returns
    -------
    df : pd.DataFrame (copy) with index_mcp_deg and/or thumb_mp_deg added.
    """
    calib = pooled_calib_df.dropna(subset=["adc_mean", "angle_snap_deg"]).copy()
    if calib.empty:
        raise ValueError("pooled_calib_df has no valid rows after dropping NaN.")


    # Fit polynomial in scaled (volt-equivalent) space
    # (same np.polyfit / np.poly1d logic as ADC_CAM.calibrate_trials_with_camera)
    x_cal = calib["adc_mean"].to_numpy(dtype=float)
    y_cal = calib["angle_snap_deg"].to_numpy(dtype=float)
    coeffs = np.polyfit(x_cal, y_cal, poly_order)
    poly = np.poly1d(coeffs)

    df = df.copy()

    def _convert_and_baseline(adc_col, out_col, label):
        if adc_col not in df.columns:
            print(f"[apply_pooled_adc_calibration] '{adc_col}' not found — skipping {out_col}.")
            return

        adc_vals = pd.to_numeric(df[adc_col], errors="coerce").to_numpy(dtype=float)
        theta = poly(adc_vals)

        if clamp_deg:
            theta = np.clip(theta, deg_min, deg_max)

        # Propagate NaNs from source ADC column
        theta[np.isnan(adc_vals)] = np.nan
        df[out_col] = theta

        if zero_baseline:
            if "elapsed_sec" in df.columns:
                t0 = df["elapsed_sec"].min()
                mask0 = df["elapsed_sec"] <= (t0 + baseline_window_sec)
                base_vals = pd.to_numeric(df.loc[mask0, out_col], errors="coerce").dropna()
            else:
                base_vals = pd.to_numeric(df[out_col], errors="coerce").dropna().iloc[:500]

            if len(base_vals) > 0:
                baseline = (base_vals.median() if baseline_stat == "median"
                            else base_vals.mean())
                df[out_col] = df[out_col] - baseline

        n_valid = int(np.isfinite(df[out_col].to_numpy()).sum())

    if convert_index_mcp:
        _convert_and_baseline(adc_col_index, out_col_index, "Index MCP")
    if convert_thumb_mp:
        _convert_and_baseline(adc_col_thumb, out_col_thumb,
                               "Thumb MP [proxy — Index MCP curve applied]")

    return df
