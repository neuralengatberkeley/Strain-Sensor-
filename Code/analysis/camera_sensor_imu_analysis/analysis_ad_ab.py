from __future__ import annotations

from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
_PARENT_DIR = _THIS_DIR.parent

# local folder first: use camera_sensor_imu_analysis/analysis.py
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# parent folder second: expose config.py
if str(_PARENT_DIR) not in sys.path:
    sys.path.insert(1, str(_PARENT_DIR))

from typing import Callable, Iterable, Optional, Sequence

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from config import path_to_repository
from analysis_adc_cam import ADC_CAM

# streamlined defaults
DEFAULT_ADAB_CALIB_KWARGS = dict(
    adc_column="adc_ch3",
    exclude_name_contains=("C_Block",),
    exclude_sets=(3, 4),
    make_plot=False,
    overlay_mean=False,
    point_alpha=0.25,
    point_size=10,
    jitter=0.25,
    snap_tol_deg=4.0,
    plot_all_data=False,
    canonical_angles=(0, 22, 45, 67), 
)

# helper methods for time parsing
def parse_hhmmssffffff(series: pd.Series) -> pd.Series:
    """
    Robust parser for timestamp values stored like HHMMSSffffff.
    Handles numeric-looking values that may come in as floats.
    """
    s = (
        pd.Series(series)
        .astype(str)
        .str.replace(r"\.0$", "", regex=True)
        .str.strip()
        .str.zfill(12)
    )
    return pd.to_datetime(s, format="%H%M%S%f", errors="coerce")

def add_elapsed_time(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    out_col: str = "time_sec",
) -> pd.DataFrame:
    """
    Add elapsed seconds from the first valid timestamp.
    Falls back to row index if timestamps cannot be parsed.
    """
    df2 = df.copy()

    if timestamp_col not in df2.columns:
        df2[out_col] = np.arange(len(df2), dtype=float)
        return df2

    ts_dt = parse_hhmmssffffff(df2[timestamp_col])
    valid = ts_dt.notna()

    if not valid.any():
        df2[out_col] = np.arange(len(df2), dtype=float)
        return df2

    t0 = ts_dt.loc[valid].iloc[0]
    df2[out_col] = (ts_dt - t0).dt.total_seconds()
    return df2

# helper methods for window
def normalize_window_spec(window) -> dict:
    """
    Accept either:
      (rest_start, rest_end, move_start, move_end)
    or:
      {"rest": (a, b), "move": (c, d)}
    """
    if isinstance(window, dict):
        rest = window["rest"]
        move = window["move"]
        return {
            "rest_start": float(rest[0]),
            "rest_end": float(rest[1]),
            "move_start": float(move[0]),
            "move_end": float(move[1]),
        }

    if len(window) != 4:
        raise ValueError(
            "Window must be either a dict with 'rest'/'move' or a "
            "4-tuple: (rest_start, rest_end, move_start, move_end)."
        )

    return {
        "rest_start": float(window[0]),
        "rest_end": float(window[1]),
        "move_start": float(window[2]),
        "move_end": float(window[3]),
    }

def get_window_values(
    df: pd.DataFrame,
    value_col: str,
    start_s: float,
    end_s: float,
    time_col: str = "time_sec",
) -> pd.Series:
    if value_col not in df.columns or time_col not in df.columns:
        return pd.Series(dtype=float)

    vals = pd.to_numeric(df[value_col], errors="coerce")
    t = pd.to_numeric(df[time_col], errors="coerce")

    mask = (t >= start_s) & (t <= end_s)
    return vals.loc[mask].dropna()

def summarize_series(values: pd.Series, prefix: str) -> dict:
    arr = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)

    if arr.size == 0:
        return {
            f"{prefix}_n": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
            f"{prefix}_ptp": np.nan,
        }

    return {
        f"{prefix}_n": int(arr.size),
        f"{prefix}_mean": float(np.nanmean(arr)),
        f"{prefix}_std": float(np.nanstd(arr, ddof=1)) if arr.size > 1 else 0.0,
        f"{prefix}_min": float(np.nanmin(arr)),
        f"{prefix}_max": float(np.nanmax(arr)),
        f"{prefix}_ptp": float(np.nanmax(arr) - np.nanmin(arr)),
    }


# loading 
def load_adab_participant(
    participant: str,
    root_dir: str,
    *,
    folder_suffix_first: str = "B1_adab",
    folder_suffix_second: str = "B2_adab",
    adc_column: str = "adc_ch3",
    poly_order: int = 2,
    calib_kwargs: Optional[dict] = None,
    clamp_theta: bool = False,
    deg_min: float = 0.0,
    deg_max: float = 67.0,
) -> dict:
    """
    Load one participant's ad/ab data and convert ADC traces to parasitic-angle
    estimates using the fixed-angle calib folders.
    """
    calib_kwargs_use = dict(DEFAULT_ADAB_CALIB_KWARGS)
    if calib_kwargs:
        calib_kwargs_use.update(calib_kwargs)

    cam = ADC_CAM(
        root_dir=root_dir,
        path_to_repo=path_to_repository,
        folder_suffix_first=folder_suffix_first,
        folder_suffix_second=folder_suffix_second,
    )

    first_trials = cam.load_first()
    second_trials = cam.load_second()

    adc_trials_first = cam.extract_adc_dfs_by_trial(first_trials)
    adc_trials_second = cam.extract_adc_dfs_by_trial(second_trials)

    out_cam = cam.calibrate_trials_with_camera(
        adc_trials_first=adc_trials_first,
        adc_trials_second=adc_trials_second,
        adc_column=adc_column,
        poly_order=poly_order,
        calib_kwargs=calib_kwargs_use,
        clamp_theta=clamp_theta,
        deg_min=deg_min,
        deg_max=deg_max,
    )

    adc_first_theta = []
    for i, df in enumerate(out_cam["adc_trials_first_theta"], start=1):
        df2 = add_elapsed_time(df)
        df2["participant"] = participant
        df2["application"] = "app1"
        df2["trial"] = i
        adc_first_theta.append(df2)

    adc_second_theta = []
    for i, df in enumerate(out_cam["adc_trials_second_theta"], start=1):
        df2 = add_elapsed_time(df)
        df2["participant"] = participant
        df2["application"] = "app2"
        df2["trial"] = i
        adc_second_theta.append(df2)

    calib_df = out_cam["calib_df"].copy()
    if not calib_df.empty:
        calib_df["participant"] = participant
        calib_df["application"] = calib_df["set"].map({1: "app1", 2: "app2"}).fillna("extra")

    return {
        "participant": participant,
        "root_dir": root_dir,
        "cam": cam,
        "first_trials": first_trials,
        "second_trials": second_trials,
        "adc_trials_first": adc_trials_first,
        "adc_trials_second": adc_trials_second,
        "adc_first_theta": adc_first_theta,
        "adc_second_theta": adc_second_theta,
        "calib_df": calib_df,
        "coeffs": out_cam["coeffs"],
    }

# trial summaries 
def summarize_trial_windows(
    df: pd.DataFrame,
    participant: str,
    application: str,
    trial: int,
    window,
    *,
    angle_col: str = "theta_cam_cal",
    adc_col: str = "adc_ch3",
    pred_bend_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> dict:
    w = normalize_window_spec(window)

    rest_angle = get_window_values(df, angle_col, w["rest_start"], w["rest_end"])
    move_angle = get_window_values(df, angle_col, w["move_start"], w["move_end"])

    row = {
        "participant": participant,
        "application": application,
        "trial": trial,
        "rest_start_s": w["rest_start"],
        "rest_end_s": w["rest_end"],
        "move_start_s": w["move_start"],
        "move_end_s": w["move_end"],
    }

    row.update(summarize_series(rest_angle, "rest_angle"))
    row.update(summarize_series(move_angle, "move_angle"))

    row["move_mean_minus_rest_mean_angle"] = (
        row["move_angle_mean"] - row["rest_angle_mean"]
        if pd.notna(row["move_angle_mean"]) and pd.notna(row["rest_angle_mean"])
        else np.nan
    )

    # requested movement metrics
    row["movement_mean_angle_deg"] = row["move_angle_mean"]
    row["movement_peak_angle_deg"] = row["move_angle_max"]
    row["movement_ptp_angle_deg"] = row["move_angle_ptp"]

    if adc_col in df.columns:
        rest_adc = get_window_values(df, adc_col, w["rest_start"], w["rest_end"])
        move_adc = get_window_values(df, adc_col, w["move_start"], w["move_end"])

        row.update(summarize_series(rest_adc, "rest_adc"))
        row.update(summarize_series(move_adc, "move_adc"))
        row["move_mean_minus_rest_mean_adc"] = (
            row["move_adc_mean"] - row["rest_adc_mean"]
            if pd.notna(row["move_adc_mean"]) and pd.notna(row["rest_adc_mean"])
            else np.nan
        )

        if pred_bend_func is not None:
            move_adc_arr = pd.to_numeric(move_adc, errors="coerce").dropna().to_numpy(dtype=float)
            if move_adc_arr.size:
                pred_bend = np.asarray(pred_bend_func(move_adc_arr), dtype=float)
                row["movement_mean_pred_bend_deg"] = float(np.nanmean(pred_bend))
                row["movement_peak_pred_bend_deg"] = float(np.nanmax(pred_bend))
                row["movement_ptp_pred_bend_deg"] = float(np.nanmax(pred_bend) - np.nanmin(pred_bend))
            else:
                row["movement_mean_pred_bend_deg"] = np.nan
                row["movement_peak_pred_bend_deg"] = np.nan
                row["movement_ptp_pred_bend_deg"] = np.nan

    return row

def summarize_participant_adab(
    result: dict,
    windows_first: Sequence,
    windows_second: Sequence,
    *,
    angle_col: str = "theta_cam_cal",
    adc_col: str = "adc_ch3",
    pred_bend_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> pd.DataFrame:
    rows = []

    for i, (df, win) in enumerate(zip(result["adc_first_theta"], windows_first), start=1):
        rows.append(
            summarize_trial_windows(
                df=df,
                participant=result["participant"],
                application="app1",
                trial=i,
                window=win,
                angle_col=angle_col,
                adc_col=adc_col,
                pred_bend_func=pred_bend_func,
            )
        )

    for i, (df, win) in enumerate(zip(result["adc_second_theta"], windows_second), start=1):
        rows.append(
            summarize_trial_windows(
                df=df,
                participant=result["participant"],
                application="app2",
                trial=i,
                window=win,
                angle_col=angle_col,
                adc_col=adc_col,
                pred_bend_func=pred_bend_func,
            )
        )

    return pd.DataFrame(rows)

# helper methods for calibration table 
def collect_calibration_points(
    result: dict,
    *,
    pred_bend_func: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> pd.DataFrame:
    """
    Return one tidy table of fixed-angle parasitic calibration points
    for one participant.
    """
    calib_df = result["calib_df"].copy()
    if calib_df.empty:
        return calib_df

    calib_df = calib_df[calib_df["set"].isin([1, 2])].copy()
    calib_df = calib_df[calib_df["angle_snap_deg"].notna()].copy()
    calib_df["application"] = calib_df["set"].map({1: "app1", 2: "app2"})

    if pred_bend_func is not None:
        calib_df["pred_bend_deg"] = pred_bend_func(calib_df["adc_mean"].to_numpy(dtype=float))

    return calib_df.sort_values(["participant", "set", "angle_snap_deg"]).reset_index(drop=True)


# plotting
def _lighten(color, amount: float = 0.35):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb + (1.0 - rgb) * amount)


def _darken(color, amount: float = 0.20):
    rgb = np.array(mcolors.to_rgb(color))
    return tuple(rgb * (1.0 - amount))

def plot_adab_trial_grid(
    result: dict,
    windows_first: Sequence,
    windows_second: Sequence,
    *,
    y_col: str = "theta_cam_cal",
    y_label: str = "Angle (deg)",
    fig_title: Optional[str] = None,
    fig_subtitle: Optional[str] = None,
    app_labels: tuple[str, str] = ("1st application", "2nd application"),
    trial_prefix: str = "Trial",
    figsize: tuple[float, float] = (12, 10),
    ylim: Optional[tuple[float, float]] = None,
    line_kwargs: Optional[dict] = None,
):
    """
    3x2 style plot of ad/ab trial traces for one participant.
    """
    nrows = max(len(result["adc_first_theta"]), len(result["adc_second_theta"]))
    if nrows == 0:
        raise ValueError(f"No trials found for participant {result['participant']}.")

    if line_kwargs is None:
        line_kwargs = dict(linewidth=1.5)

    fig, axes = plt.subplots(nrows, 2, figsize=figsize, sharex=False, sharey=True)
    if nrows == 1:
        axes = np.array([axes])

    def _plot_one(ax, df, window, panel_title):
        w = normalize_window_spec(window)

        x = pd.to_numeric(df["time_sec"], errors="coerce")
        y = pd.to_numeric(df[y_col], errors="coerce") if y_col in df.columns else pd.Series(dtype=float)

        ax.plot(x, y, **line_kwargs)
        ax.axvspan(w["rest_start"], w["rest_end"], alpha=0.12, color="gray", label="Rest")
        ax.axvspan(w["move_start"], w["move_end"], alpha=0.12, color="orange", label="Movement")
        ax.set_title(panel_title, fontsize=11)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(y_label)
        ax.grid(alpha=0.25)

        if ylim is not None:
            ax.set_ylim(*ylim)

    for i in range(nrows):
        ax = axes[i, 0]
        if i < len(result["adc_first_theta"]):
            _plot_one(
                ax,
                result["adc_first_theta"][i],
                windows_first[i],
                f"{trial_prefix} {i+1} — {app_labels[0]}",
            )
        else:
            ax.axis("off")

        ax = axes[i, 1]
        if i < len(result["adc_second_theta"]):
            _plot_one(
                ax,
                result["adc_second_theta"][i],
                windows_second[i],
                f"{trial_prefix} {i+1} — {app_labels[1]}",
            )
        else:
            ax.axis("off")

    if fig_title is None:
        fig_title = f"Ad/abduction trial traces — participant {result['participant']}"

    fig.suptitle(fig_title, y=0.985, fontsize=14)

    if fig_subtitle is not None:
        fig.text(0.5, 0.955, fig_subtitle, ha="center", va="center", fontsize=10)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper right", frameon=False)

    plt.tight_layout(rect=[0, 0, 0.96, 0.94 if fig_subtitle is None else 0.92])
    return fig, axes

def add_calibration_variability(
    calib_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each calib row, read the raw ADC CSV at source_path and compute
    n, std, and sem for the ADC hold.
    """
    out = calib_df.copy()

    adc_n = []
    adc_std = []
    adc_sem = []

    for _, row in out.iterrows():
        csv_path = row["source_path"]
        adc_col = row["adc_column"]

        try:
            df = pd.read_csv(csv_path)
        except UnicodeDecodeError:
            df = pd.read_csv(csv_path, encoding="latin-1")

        if adc_col not in df.columns:
            vals = pd.Series(dtype=float)
        else:
            vals = pd.to_numeric(df[adc_col], errors="coerce").dropna()

        n = int(len(vals))
        sd = float(vals.std(ddof=1)) if n > 1 else np.nan
        sem = float(sd / np.sqrt(n)) if n > 1 else np.nan

        adc_n.append(n)
        adc_std.append(sd)
        adc_sem.append(sem)

    out["adc_n"] = adc_n
    out["adc_std"] = adc_std
    out["adc_sem"] = adc_sem
    return out

def plot_combined_parasitic_relationship(
    calib_all: pd.DataFrame,
    *,
    y_col: str = "adc_mean",
    y_label: Optional[str] = None,
    title: Optional[str] = None,
    error_col: Optional[str] = None,
    connect_points: bool = True,
):
    if calib_all.empty:
        raise ValueError("calib_all is empty.")

    fig, ax = plt.subplots(figsize=(9, 6))

    base_colors = {
        "11_2_25": "#4C78A8",
        "11_4_25": "#F58518",
        "11_7_25": "#54A24B",
    }

    participants = list(pd.unique(calib_all["participant"]))

    for participant in participants:
        base = base_colors.get(participant, "#4C72B0")

        for application in ["app1", "app2"]:
            sub = calib_all[
                (calib_all["participant"] == participant)
                & (calib_all["application"] == application)
            ].copy()

            if sub.empty or y_col not in sub.columns:
                continue

            sub = sub.sort_values("angle_snap_deg")
            color = _lighten(base, 0.35) if application == "app1" else _darken(base, 0.15)

            x = sub["angle_snap_deg"].to_numpy(dtype=float)
            y = pd.to_numeric(sub[y_col], errors="coerce").to_numpy(dtype=float)

            if connect_points and len(sub) >= 2:
                ax.plot(
                    x, y,
                    linewidth=2.0,
                    color=color,
                    alpha=0.95,
                )

            ax.scatter(
                x, y,
                s=65,
                color=color,
                edgecolor="black",
                linewidth=0.4,
                alpha=0.95,
                label=f"{participant} {application}",
                zorder=3,
            )

            if error_col is not None and error_col in sub.columns:
                yerr = pd.to_numeric(sub[error_col], errors="coerce").to_numpy(dtype=float)
                ax.errorbar(
                    x, y, yerr=yerr,
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.2,
                    capsize=3,
                    alpha=0.9,
                    zorder=2,
                )

    ax.set_xlabel("Parasitic ad/ab angle (deg)")
    ax.set_ylabel(y_label if y_label is not None else y_col)
    ax.set_title(title if title is not None else f"{y_col} vs parasitic ad/ab angle")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    return fig, ax
