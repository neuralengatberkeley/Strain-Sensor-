from __future__ import annotations

# Standard library
import os
import re
import glob
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, List

# Third-party
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.io import loadmat

from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import PolynomialFeatures
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression

# Local
from config import path_to_repository



# Ensure DLC3DBendAngles is available in scope
# from your_module import DLC3DBendAngles


# --- Required imports (put these once at the top of your file) ---
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Sequence, Union
import numpy as np
import pandas as pd


class BallBearingData:
    """
    Loader + calibration for ball-bearing / beaker trials.
    """

    def __init__(
        self,
        root_dir: str,
        path_to_repo: str,
        n_trials_per_set: int = 9,
        files_per_trial: int = 7,
        folder_suffix: str = "R_mar",
    ):
        self.root_dir = Path(path_to_repo) / Path(root_dir)
        self.n_trials_per_set = int(n_trials_per_set)
        self.files_per_trial = int(files_per_trial)
        self.folder_suffix = str(folder_suffix)
        self._all_folders: List[Path] = []
        self._first_set: List[Path] = []
        self._second_set: List[Path] = []

        # calibration state
        self.calib: Dict[str, float] = {}   # {c0,c1,c2,y0,y90,source}

    # ---------- file system helpers ----------
    @staticmethod
    def _is_bad_dot_underscore(p: Path) -> bool:
        return p.name.startswith("._")

    @staticmethod
    def _read_csv_safe(p: Path) -> Optional[pd.DataFrame]:
        if not p.exists() or p.stat().st_size == 0:
            return None
        if BallBearingData._is_bad_dot_underscore(p):
            return None
        for enc in ("utf-8", "utf-8-sig", "latin1"):
            try:
                return pd.read_csv(p, encoding=enc)
            except Exception:
                continue
        print(f"[WARN] Skipping unreadable file: {p} (failed UTF-8/UTF-8-SIG/latin1)")
        return None

    def _find_trial_folders(self) -> List[Path]:
        """
        Find trial root folders whose name ends with `_<folder_suffix>` (case-insensitive).
        Prefer direct children of root (current layout), and fall back to a recursive
        search if none are found.
        """
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        suf = str(self.folder_suffix).strip()
        # Pass 1: non-recursive (direct children)
        direct = [p for p in self.root_dir.glob(f"*_{suf}") if p.is_dir()]
        if not direct:
            # Pass 2: recursive fallback, case-insensitive endswith
            direct = []
            suf_lower = f"_{suf}".lower()
            for p in self.root_dir.rglob("*"):
                if p.is_dir() and p.name.lower().endswith(suf_lower):
                    direct.append(p)

        folders = sorted(set(direct), key=lambda p: p.name)

        if not folders:
            print(f"[DEBUG] No *_{self.folder_suffix} dirs under {self.root_dir}")
            print("[DEBUG] Show a few children of root:")
            for i, p in enumerate(sorted(self.root_dir.glob("*"))[:12]):
                print("  -", p)
            return []

        print(f"Found {len(folders)} *_{self.folder_suffix} folders total (case-insensitive).")
        print("  example:", folders[0])
        self._all_folders = folders
        return folders

    def _split_sets(self) -> Tuple[List[Path], List[Path]]:
        folders = self._all_folders or self._find_trial_folders()
        if len(folders) < 2 * self.n_trials_per_set:
            n = len(folders) // 2
            first, second = folders[:n], folders[n: n*2]
        else:
            first  = folders[: self.n_trials_per_set]
            second = folders[self.n_trials_per_set: 2 * self.n_trials_per_set]

        if first:
            print(f"First set range: {first[0].name} → {first[-1].name}")
        if second:
            print(f"Second set range: {second[0].name} → {second[-1].name}")

        self._first_set, self._second_set = first, second
        return first, second

    # ---------- public loaders ----------
    def load_first(self) -> List[str]:
        if not self._all_folders:
            self._find_trial_folders()
        first, _ = self._split_sets()
        self._warn_counts("ball_bearing_first", first)
        return [str(p) for p in first]

    def load_second(self) -> List[str]:
        if not self._all_folders:
            self._find_trial_folders()
        _, second = self._split_sets()
        self._warn_counts("ball_bearing_second", second)
        return [str(p) for p in second]

    def _warn_counts(self, label: str, folders: List[Path]):
        problems = []
        for i, f in enumerate(folders, start=1):
            # Try top-level first (old layout)
            csvs = [p for p in f.glob("*.csv") if not self._is_bad_dot_underscore(p)]
            # If that looks short, count nested too (new layout)
            if len(csvs) < self.files_per_trial:
                csvs = [p for p in f.glob("**/*.csv") if not self._is_bad_dot_underscore(p)]
            if len(csvs) != self.files_per_trial:
                problems.append((i, f.name, len(csvs)))

        if problems:
            print(f"[WARN] {label}: Some trials do not have exactly {self.files_per_trial} CSVs:")
            for i, name, n in problems:
                print(f"  • Trial {i:02d}: {name} has {n} CSVs (including nested)")

    # ---------- generic per-trial extractor ----------
    def _extract_by_glob(self, trial_folders: Sequence[Union[str, Path]], pattern: Union[str, Sequence[str]]):
        pats = [pattern] if isinstance(pattern, str) else list(pattern)
        out: List[pd.DataFrame] = []
        for folder in trial_folders:
            fp = Path(folder)
            cands: List[Path] = []
            for pat in pats:
                cands.extend([p for p in fp.rglob(pat) if not self._is_bad_dot_underscore(p)])
            if not cands:
                out.append(pd.DataFrame()); continue
            # Prefer the largest (usually the full-length recording)
            cands = sorted(set(cands), key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
            df = self._read_csv_safe(cands[0])
            out.append(df if df is not None else pd.DataFrame())
        return out

    def compare_block_quadratic_vs_angle(
            self,
            *,
            h_cal_path_first,
            ranges_first: list[tuple[int, int]],
            h_cal_path_second,
            ranges_second: list[tuple[int, int]],
            angles: list[float],
            adc_col: str = "adc_ch3",
            max_points_per_range: int | None = None,
            figsize=(9.5, 5.0),
            jitter: float = 0.25,
            scatter_alpha: float = 0.35,
            scatter_size: float = 8,
            color_first: str = "red",
            color_second: str = "C0",
            ax=None,
            show: bool = True,
    ):
        """
        Build ADC-vs-ANGLE datasets from row-index windows, do quadratic fits for two block
        calibrations, and plot both (first=red, second=blue).

        Parameters
        ----------
        h_cal_path_first, h_cal_path_second : str | Path | array-like | pd.Series
            CSV path containing `adc_col`, or raw ADC series/array.
        ranges_first, ranges_second : list[(start, end)]
            One window per angle; len(ranges_*) must equal len(angles). end is exclusive.
        angles : list[float]
            Angles corresponding to each (start, end) window, e.g., [0, 22.5, 45, 67.5, 90].
        adc_col : str
            ADC column name when loading from CSV.
        max_points_per_range : int | None
            If set, keeps the flattest consecutive window of this many points inside each range.
        """

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from pathlib import Path

        # ---- helpers ----
        def _load_adc_series(src):
            if isinstance(src, (list, tuple, np.ndarray, pd.Series)):
                s = pd.Series(src, dtype="float64")
                return pd.to_numeric(s, errors="coerce")
            if isinstance(src, (str, Path)):
                p = Path(src)
                if not p.exists():
                    raise FileNotFoundError(f"CSV not found: {p}")
                df = self._read_csv_safe(p)
                if df is None or df.empty:
                    raise ValueError(f"Empty/unreadable CSV: {p}")
                col = adc_col if adc_col in df.columns else next(
                    (c for c in df.columns if str(c).lower().startswith("adc")), None
                )
                if col is None:
                    raise KeyError(f"No ADC-like column in {p.name}")
                return pd.to_numeric(df[col], errors="coerce")
            return pd.to_numeric(pd.Series(src), errors="coerce")

        def _flattest_window(vals: np.ndarray, k: int) -> np.ndarray:
            if k is None or vals.size <= (k or 0):
                return vals
            k = int(k)
            if k < 1 or vals.size <= k:
                return vals
            c1 = np.concatenate(([0.0], np.cumsum(vals)))
            c2 = np.concatenate(([0.0], np.cumsum(vals * vals)))
            sum_y = c1[k:] - c1[:-k]
            sum_y2 = c2[k:] - c2[:-k]
            mean_y = sum_y / k
            var_y = np.maximum(sum_y2 / k - mean_y ** 2, 0.0)
            i0 = int(np.argmin(var_y))
            return vals[i0:i0 + k]

        def _build_angle_adc_table(y_series: pd.Series, win_list: list[tuple[int, int]],
                                   angs: list[float]) -> pd.DataFrame:
            if len(win_list) != len(angs):
                raise ValueError("ranges length must equal angles length.")
            y = pd.to_numeric(pd.Series(y_series), errors="coerce").to_numpy(float)
            n = len(y)
            parts = []
            for (start, end), ang in zip(win_list, angs):
                start = int(max(0, start));
                end = int(min(n, end))
                if end <= start:
                    continue
                vals = y[start:end]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                if max_points_per_range is not None and vals.size > max_points_per_range:
                    vals = _flattest_window(vals, int(max_points_per_range))
                parts.append(pd.DataFrame({"angle": float(ang), "adc": vals}))
            return (pd.concat(parts, ignore_index=True)
                    if parts else pd.DataFrame(columns=["angle", "adc"]))

        def _quad_fit_xy(x: np.ndarray, y: np.ndarray):
            uniq = np.unique(x[np.isfinite(x)])
            if x.size < 3 or uniq.size < 3:
                return np.nan, np.nan, np.nan, np.nan
            c2, c1, c0 = np.polyfit(x, y, deg=2)
            yhat = c0 + c1 * x + c2 * (x ** 2)
            ss_res = np.sum((y - yhat) ** 2)
            ss_tot = np.sum((y - np.nanmean(y)) ** 2)
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            return float(c0), float(c1), float(c2), float(r2)

        # ---- load data & make tables ----
        y_first = _load_adc_series(h_cal_path_first)
        y_second = _load_adc_series(h_cal_path_second)

        df_first = _build_angle_adc_table(y_first, ranges_first, angles)
        df_second = _build_angle_adc_table(y_second, ranges_second, angles)

        x1 = df_first["angle"].to_numpy(float) if not df_first.empty else np.array([])
        y1 = df_first["adc"].to_numpy(float) if not df_first.empty else np.array([])
        x2 = df_second["angle"].to_numpy(float) if not df_second.empty else np.array([])
        y2 = df_second["adc"].to_numpy(float) if not df_second.empty else np.array([])

        c0_1, c1_1, c2_1, r2_1 = _quad_fit_xy(x1, y1)
        c0_2, c1_2, c2_2, r2_2 = _quad_fit_xy(x2, y2)

        # ---- plot ----
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        # jittered scatter per angle
        def _scatter_by_angle(df, color, label):
            if df.empty:
                return
            rng = np.random.default_rng(12345)
            for ang in sorted(set(df["angle"].astype(float))):
                ys = df.loc[df["angle"] == ang, "adc"].to_numpy(float)
                if ys.size:
                    xs = ang + rng.uniform(-jitter, jitter, size=ys.size)
                    ax.scatter(xs, ys, s=scatter_size, alpha=scatter_alpha, color=color, label=None)
            # tiny transparent handle for legend consistency
            ax.scatter([], [], color=color, alpha=scatter_alpha, s=scatter_size, label=label)

        _scatter_by_angle(df_first, color_first, "First (raw)")
        _scatter_by_angle(df_second, color_second, "Second (raw)")

        if len(angles) >= 2:
            xx = np.linspace(min(angles), max(angles), 400)
            if np.isfinite([c0_1, c1_1, c2_1]).all():
                ax.plot(xx, c0_1 + c1_1 * xx + c2_1 * (xx ** 2),
                        color=color_first, linewidth=2.0, label=f"First fit (R²={r2_1:.4f})")
            if np.isfinite([c0_2, c1_2, c2_2]).all():
                ax.plot(xx, c0_2 + c1_2 * xx + c2_2 * (xx ** 2),
                        color=color_second, linewidth=2.0, label=f"Second fit (R²={r2_2:.4f})")

        ax.set_xlabel("Angle (deg)")
        ax.set_ylabel(adc_col if isinstance(h_cal_path_first, (str, Path)) else "adc_ch3 (raw)")
        ax.set_title("Quadratic fit: ADC vs Angle — First (red) vs Second (blue)")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False)
        fig.tight_layout()
        if show:
            plt.show()

        return {
            "first": {"c0": c0_1, "c1": c1_1, "c2": c2_1, "r2": r2_1, "n": int(y1.size)},
            "second": {"c0": c0_2, "c1": c1_2, "c2": c2_2, "r2": r2_2, "n": int(y2.size)},
            "df_first": df_first,
            "df_second": df_second,
            "fig": fig, "ax": ax,
        }

    # Specific extractors
    def extract_adc_dfs_by_trial(self, trial_folders):
        return self._extract_by_glob(trial_folders, ["data_adc*.csv", "data_adc*"])

    def extract_imu_dfs_by_trial(self, trial_folders):
        return self._extract_by_glob(trial_folders, ["data_imu*.csv", "data_imu*"])

    def extract_spacebar_dfs_by_trial(self, trial_folders):
        return self._extract_by_glob(trial_folders, ["data_spacebar*.csv", "data_spacebar*"])

    def extract_rotenc_dfs_by_trial(self, trial_folders):
        return self._extract_by_glob(trial_folders, ["data_rotenc*.csv", "data_rotenc*"])

    # --- add inside class BallBearingData ---

    def extract_dlc3d_dfs_by_trial(
            self,
            trial_folders: list[str] | list[Path],
            file_patterns: tuple[str, ...] = ("*DLC*.csv", "*DLC3D*.csv", "*dlc3d*.csv", "*3d*.csv", "*3D*.csv"),
            *,
            # NEW labeling controls
            add_labels: bool = True,
            trial_labels: Optional[List[int]] = None,
            trial_base: int = 1,
            set_label: Optional[str] = None,
            set_labels: Optional[List[str]] = None,
            include_path: bool = False,
    ) -> list[pd.DataFrame]:
        """
        Find DLC 3D csv per trial folder. Picks the largest matching csv in each folder (or nested).
        Returns a list of DataFrames (MultiIndex if possible). If add_labels=True, stamps 'trial' and 'set_label'.
        """
        import pandas as pd
        from pathlib import Path

        out: list[pd.DataFrame] = []

        n = len(trial_folders)
        if trial_labels is None:
            labels_trial = [trial_base + i for i in range(n)]
        else:
            if len(trial_labels) != n:
                raise ValueError("trial_labels length must match trial_folders length.")
            labels_trial = trial_labels

        if set_labels is not None and len(set_labels) != n:
            raise ValueError("set_labels length must match trial_folders length.")

        def _label_for(i: int) -> tuple[Optional[int], Optional[str]]:
            tlabel = labels_trial[i] if add_labels else None
            slabel = None
            if add_labels:
                slabel = set_labels[i] if set_labels is not None else set_label
            return tlabel, slabel

        for i, folder in enumerate(trial_folders):
            fp = Path(folder)

            # find candidates
            cands: list[Path] = []
            for pat in file_patterns:
                cands.extend([p for p in fp.glob(pat) if not self._is_bad_dot_underscore(p)])
            if not cands:
                for pat in file_patterns:
                    cands.extend([p for p in fp.rglob(pat) if not self._is_bad_dot_underscore(p)])

            if not cands:
                df = pd.DataFrame()
                tlabel, slabel = _label_for(i)
                if add_labels:
                    if tlabel is not None: df["trial"] = [tlabel]
                    if slabel is not None: df["set_label"] = [slabel]
                if include_path: df["source_path"] = [str(fp)]
                out.append(df)
                continue

            # prefer largest file (full export)
            cands = sorted(set(cands), key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
            df0 = self._read_csv_safe(cands[0])
            if df0 is None or df0.empty:
                df = pd.DataFrame()
                tlabel, slabel = _label_for(i)
                if add_labels:
                    if tlabel is not None: df["trial"] = [tlabel]
                    if slabel is not None: df["set_label"] = [slabel]
                if include_path: df["source_path"] = [str(cands[0])]
                out.append(df)
                continue

            # normalize to MultiIndex if it looks like a flat DLC export
            try:
                df_coerced = self._coerce_dlc3d_multiindex(df0)
            except Exception:
                df_coerced = df0

            # stamp labels (don’t overwrite if already present)
            if add_labels and not df_coerced.empty:
                tlabel, slabel = _label_for(i)
                if ("trial" not in df_coerced.columns) and (tlabel is not None):
                    df_coerced = df_coerced.copy();
                    df_coerced["trial"] = tlabel
                if ("set_label" not in df_coerced.columns) and (slabel is not None):
                    df_coerced = df_coerced.copy();
                    df_coerced["set_label"] = slabel

            if include_path:
                df_coerced = df_coerced.copy();
                df_coerced["source_path"] = str(cands[0])

            out.append(df_coerced)
        return out

    def calibrate_four_from_overlays_and_stream(
            self,
            overlays: dict,
            *,
            adc_trials_first: list | None = None,
            adc_trials_second: list | None = None,
            anchors_mode: str = "stream_per_set",  # "stream_per_set" | "stream_per_cal" | "endpoints_per_cal"
            q_hi: float = 0.999,
            q_lo: float = 0.001,
            deg_min: float = 0.0,
            deg_max: float = 90.0,
            clip_z: bool = True,
            extrapolate: bool = False,
            clamp_theta: bool = True,
            trial_len_sec: float = 10.0,
            plot: bool = True,
            # styling
            linewidth: float = 1.8,
            alpha_curves: float = 0.95,
            figsize=(10.5, 4.2),
            label_cam1="Camera 1", label_blk1="Block 1",
            label_cam2="Camera 2", label_blk2="Block 2",
            verbose: bool = True,
    ):
        import numpy as _np
        import pandas as _pd
        import copy
        import matplotlib.pyplot as _plt

        # ---------- helpers ----------
        def _coeffs(tag):
            ov = overlays.get(tag, {})
            if not ov or not ov.get("ok"):
                return None
            c0, c1, c2 = ov.get("coeffs", (None, None, None))
            if c0 is None or c1 is None or c2 is None:
                return None
            return float(c0), float(c1), float(c2)

        def _poly_y(th_deg, c0, c1, c2):
            th = float(th_deg)
            return c0 + c1 * th + c2 * th * th

        def _stack_adc(trials):
            parts = []
            for df in (trials or []):
                if df is None or df.empty:
                    continue
                col = "adc_ch3" if "adc_ch3" in df.columns else next(
                    (c for c in df.columns if str(c).lower().startswith("adc")), None
                )
                if col is None:
                    continue
                parts.append(_pd.to_numeric(df[col], errors="coerce"))
            return _pd.concat(parts, ignore_index=True) if parts else _pd.Series(dtype=float)

        def _anchors_from_stream(trials, q_hi, q_lo):
            s = _stack_adc(trials)
            if s.empty:
                return _np.nan, _np.nan, True
            y_raw = _np.asarray(s, float)
            return float(_np.nanquantile(y_raw, q_hi)), float(_np.nanquantile(y_raw, q_lo)), False

        def _anchors_from_endpoints(c0, c1, c2):
            return float(_poly_y(0.0, c0, c1, c2)), float(_poly_y(90.0, c0, c1, c2))

        def _clone_and_install_inverse(base_bb, c0, c1, c2, y0, y90, source_tag):
            b = copy.deepcopy(base_bb)
            if not hasattr(b, "calib") or not isinstance(getattr(b, "calib", None), dict):
                b.calib = {}
            b.calib.update(c0=float(c0), c1=float(c1), c2=float(c2), y0=float(y0), y90=float(y90),
                           source=source_tag)
            b.build_poly_inverse(clip_z=clip_z, extrapolate=extrapolate,
                                 deg_min=deg_min, deg_max=deg_max)
            return b

        def _tall(bb_obj, trials, set_label):
            if (not trials) or (bb_obj is None):
                return _pd.DataFrame(columns=["set_label", "trial", "time_s", "timestamp",
                                              "theta_pred_deg", "adc_ch3"])
            return bb_obj.trials_to_tall_df(
                trials, set_label=set_label, trial_len_sec=trial_len_sec,
                use_poly_inverse=True, clamp_theta=clamp_theta
            )

        def _attach_oob(df, y0, y90):
            if df.empty:
                return df
            lo, hi = (y90, y0) if y90 < y0 else (y0, y90)
            col = "adc_ch3" if "adc_ch3" in df.columns else next(
                (c for c in df.columns if str(c).lower().startswith("adc")), None
            )
            if col is None:
                return df
            v = _pd.to_numeric(df[col], errors="coerce").to_numpy(float)
            over = v > hi
            under = v < lo
            oob = _pd.DataFrame({
                "adc_raw": v, "oob_over_hi": over, "oob_under_lo": under,
                "oob_margin": _np.where(over, v - hi, _np.where(under, lo - v, 0.0)),
            })
            return _pd.concat([df.reset_index(drop=True), oob], axis=1)

        def _trial_ids_union(df_a, df_b):
            ids = set()
            if (df_a is not None) and (not df_a.empty) and ("trial" in df_a.columns):
                ids |= set(_pd.to_numeric(df_a["trial"], errors="coerce").dropna().astype(int).unique())
            if (df_b is not None) and (not df_b.empty) and ("trial" in df_b.columns):
                ids |= set(_pd.to_numeric(df_b["trial"], errors="coerce").dropna().astype(int).unique())
            return sorted(list(ids))

        # NEW: attach original timestamps from raw trials by (set_label, trial, row-order)
        def _attach_timestamp_from_trials(tall_df, trials, set_label):
            """
            Append original integer 'timestamp' from `trials` into `tall_df`,
            aligning by (set_label, trial, row-order). No dtype conversion.
            If tall_df already has 'timestamp', this is a no-op.
            """
            if tall_df is None or tall_df.empty:
                return tall_df
            if "timestamp" in tall_df.columns:
                return tall_df  # already present

            df = tall_df.copy()
            df["_row_idx"] = df.groupby(["set_label", "trial"]).cumcount()

            parts = []
            for t_id, raw in enumerate(trials or [], start=1):  # 1-based to match your trials
                if raw is None or len(raw) == 0 or "timestamp" not in raw.columns:
                    continue
                ts = _pd.to_numeric(raw["timestamp"], errors="coerce")
                idx = _pd.DataFrame({
                    "set_label": set_label,
                    "trial": t_id,
                    "_row_idx": _np.arange(len(ts), dtype=int),
                    "timestamp": ts.to_numpy()
                })
                parts.append(idx)

            if not parts:
                return df.drop(columns=["_row_idx"], errors="ignore")

            attach = _pd.concat(parts, ignore_index=True)
            df = _pd.merge(df, attach, on=["set_label", "trial", "_row_idx"], how="left")
            return df.drop(columns=["_row_idx"], errors="ignore")

        # ---------- grab coeffs ----------
        c_blk1 = _coeffs("block1")
        c_blk2 = _coeffs("block2")
        c_cam1 = _coeffs("cam1")
        c_cam2 = _coeffs("cam2")

        if verbose:
            print("[four] coeffs found:",
                  f"blk1={c_blk1 is not None}, blk2={c_blk2 is not None}, cam1={c_cam1 is not None}, cam2={c_cam2 is not None}")

        # ---------- compute anchors per mode ----------
        anchors_per_cal = {}  # y0,y90 per calibration key
        mode = anchors_mode.lower()

        if mode not in {"stream_per_set", "stream_per_cal", "endpoints_per_cal"}:
            raise ValueError("anchors_mode must be one of {'stream_per_set','stream_per_cal','endpoints_per_cal'}")

        if mode == "stream_per_set":
            # first set (shared by blk1 + cam1)
            y0_f, y90_f, empty_f = _anchors_from_stream(adc_trials_first, q_hi, q_lo)
            if empty_f and (c_blk1 or c_cam1):
                pool = [c for c in (c_blk1, c_cam1) if c]
                y0_f = float(_np.mean([_poly_y(0, *c) for c in pool]))
                y90_f = float(_np.mean([_poly_y(90, *c) for c in pool]))
            # second set (shared by blk2 + cam2)
            y0_s, y90_s, empty_s = _anchors_from_stream(adc_trials_second, q_hi, q_lo)
            if empty_s and (c_blk2 or c_cam2):
                pool = [c for c in (c_blk2, c_cam2) if c]
                y0_s = float(_np.mean([_poly_y(0, *c) for c in pool]))
                y90_s = float(_np.mean([_poly_y(90, *c) for c in pool]))

            anchors_per_cal["blk1"] = {"y0": y0_f, "y90": y90_f, "mode": mode}
            anchors_per_cal["cam1"] = {"y0": y0_f, "y90": y90_f, "mode": mode}
            anchors_per_cal["blk2"] = {"y0": y0_s, "y90": y90_s, "mode": mode}
            anchors_per_cal["cam2"] = {"y0": y0_s, "y90": y90_s, "mode": mode}

        elif mode == "stream_per_cal":
            for key, coeffs, trials in [
                ("blk1", c_blk1, adc_trials_first),
                ("cam1", c_cam1, adc_trials_first),
                ("blk2", c_blk2, adc_trials_second),
                ("cam2", c_cam2, adc_trials_second),
            ]:
                if coeffs is None:
                    anchors_per_cal[key] = {"y0": _np.nan, "y90": _np.nan, "mode": mode}
                    continue
                y0, y90, empty = _anchors_from_stream(trials, q_hi, q_lo)
                if empty:
                    y0, y90 = _anchors_from_endpoints(*coeffs)
                anchors_per_cal[key] = {"y0": y0, "y90": y90, "mode": mode}

        else:  # endpoints_per_cal
            for key, coeffs in [("blk1", c_blk1), ("cam1", c_cam1), ("blk2", c_blk2), ("cam2", c_cam2)]:
                if coeffs is None:
                    anchors_per_cal[key] = {"y0": _np.nan, "y90": _np.nan, "mode": mode}
                else:
                    y0, y90 = _anchors_from_endpoints(*coeffs)
                    anchors_per_cal[key] = {"y0": y0, "y90": y90, "mode": mode}

        if verbose:
            for k, v in anchors_per_cal.items():
                print(f"[four] anchors {k}: y0={v['y0']:.6g}, y90={v['y90']:.6g}, mode={v['mode']}")

        # ---------- build 4 calibrated clones ----------
        bb_blk1 = _clone_and_install_inverse(self, *(c_blk1 or (0, 1, 0)),
                                             anchors_per_cal.get("blk1", {}).get("y0", _np.nan),
                                             anchors_per_cal.get("blk1", {}).get("y90", _np.nan),
                                             "four_from_overlays") if c_blk1 else None
        bb_cam1 = _clone_and_install_inverse(self, *(c_cam1 or (0, 1, 0)),
                                             anchors_per_cal.get("cam1", {}).get("y0", _np.nan),
                                             anchors_per_cal.get("cam1", {}).get("y90", _np.nan),
                                             "four_from_overlays") if c_cam1 else None
        bb_blk2 = _clone_and_install_inverse(self, *(c_blk2 or (0, 1, 0)),
                                             anchors_per_cal.get("blk2", {}).get("y0", _np.nan),
                                             anchors_per_cal.get("blk2", {}).get("y90", _np.nan),
                                             "four_from_overlays") if c_blk2 else None
        bb_cam2 = _clone_and_install_inverse(self, *(c_cam2 or (0, 1, 0)),
                                             anchors_per_cal.get("cam2", {}).get("y0", _np.nan),
                                             anchors_per_cal.get("cam2", {}).get("y90", _np.nan),
                                             "four_from_overlays") if c_cam2 else None

        # ---------- tall tables ----------
        theta_blk1_first = _tall(bb_blk1, adc_trials_first, set_label="first_block") if bb_blk1 else _pd.DataFrame()
        theta_cam1_first = _tall(bb_cam1, adc_trials_first, set_label="first_cam") if bb_cam1 else _pd.DataFrame()
        theta_blk2_second = _tall(bb_blk2, adc_trials_second, set_label="second_block") if bb_blk2 else _pd.DataFrame()
        theta_cam2_second = _tall(bb_cam2, adc_trials_second, set_label="second_cam") if bb_cam2 else _pd.DataFrame()

        # --- append raw timestamps from input trials (preserve integer ticks) ---
        theta_blk1_first = _attach_timestamp_from_trials(theta_blk1_first, adc_trials_first, "first_block")
        theta_cam1_first = _attach_timestamp_from_trials(theta_cam1_first, adc_trials_first, "first_cam")
        theta_blk2_second = _attach_timestamp_from_trials(theta_blk2_second, adc_trials_second, "second_block")
        theta_cam2_second = _attach_timestamp_from_trials(theta_cam2_second, adc_trials_second, "second_cam")

        # ---------- OOB ----------
        theta_blk1_first = _attach_oob(theta_blk1_first,
                                       anchors_per_cal.get("blk1", {}).get("y0", _np.nan),
                                       anchors_per_cal.get("blk1", {}).get("y90", _np.nan))
        theta_cam1_first = _attach_oob(theta_cam1_first,
                                       anchors_per_cal.get("cam1", {}).get("y0", _np.nan),
                                       anchors_per_cal.get("cam1", {}).get("y90", _np.nan))
        theta_blk2_second = _attach_oob(theta_blk2_second,
                                        anchors_per_cal.get("blk2", {}).get("y0", _np.nan),
                                        anchors_per_cal.get("blk2", {}).get("y90", _np.nan))
        theta_cam2_second = _attach_oob(theta_cam2_second,
                                        anchors_per_cal.get("cam2", {}).get("y0", _np.nan),
                                        anchors_per_cal.get("cam2", {}).get("y90", _np.nan))

        def _oob_count(df):
            if df.empty:
                return {"over_hi": 0, "under_lo": 0}
            return {
                "over_hi": int(
                    _pd.to_numeric(df.get("oob_over_hi"), errors="coerce").sum()) if "oob_over_hi" in df else 0,
                "under_lo": int(
                    _pd.to_numeric(df.get("oob_under_lo"), errors="coerce").sum()) if "oob_under_lo" in df else 0
            }

        oob_counts = {
            "first": _oob_count(_pd.concat([theta_cam1_first, theta_blk1_first], axis=0)),
            "second": _oob_count(_pd.concat([theta_cam2_second, theta_blk2_second], axis=0)),
        }

        # ---------- combine outputs ----------
        tall_all = _pd.concat(
            [theta_cam1_first, theta_blk1_first, theta_cam2_second, theta_blk2_second],
            ignore_index=True
        )

        # prefer timestamp in pairwise wide merges
        def _wide_pair(df_a, df_b, name_a, name_b):
            if (df_a is None or df_a.empty) and (df_b is None or df_b.empty):
                return _pd.DataFrame()

            join_priority = [
                ["set_label", "trial", "timestamp"],
                ["set_label", "trial", "time_s"],
                ["timestamp"],
                ["time_s"],
            ]

            def _keys_in_both(keys, A, B):
                return all(k in A.columns for k in keys) and all(k in B.columns for k in keys)

            keys = next((k for k in join_priority if _keys_in_both(k, df_a, df_b)), [])

            dfA = df_a.rename(columns={"theta_pred_deg": f"theta_{name_a}"})
            dfB = df_b.rename(columns={"theta_pred_deg": f"theta_{name_b}"})

            if keys:
                out = _pd.merge(
                    dfA[keys + [f"theta_{name_a}"]],
                    dfB[keys + [f"theta_{name_b}"]],
                    on=keys, how="outer"
                )
                return out.sort_values(keys)
            else:
                return _pd.concat([dfA[[f"theta_{name_a}"]], dfB[[f"theta_{name_b}"]]], axis=1)

        wide_first = _wide_pair(theta_cam1_first, theta_blk1_first, "cam1", "blk1")
        wide_second = _wide_pair(theta_cam2_second, theta_blk2_second, "cam2", "blk2")

        # merge wide tables on common keys (prefer timestamp)
        priority_orders = [
            ["set_label", "trial", "timestamp"],
            ["set_label", "trial", "time_s"],
            ["timestamp"],
            ["time_s"],
        ]

        def _common_keys(a, b, prefs):
            for keys in prefs:
                if all(k in a.columns for k in keys) and all(k in b.columns for k in keys):
                    return keys
            return []

        common_keys = _common_keys(wide_first, wide_second, priority_orders)
        if common_keys:
            theta_all_wide = _pd.merge(wide_first, wide_second, how="outer", on=common_keys).sort_values(common_keys)
        else:
            theta_all_wide = _pd.concat([wide_first, wide_second], axis=1)

        # ---------- plotting ----------
        fig = axes = None
        if plot:
            fig, axes = _plt.subplots(1, 2, figsize=figsize, sharey=True)

            def _plot_panel(ax, df_blk, df_cam, label_blk, label_cam, panel_title):
                trial_ids = _trial_ids_union(df_blk, df_cam)
                colors = _plt.rcParams["axes.prop_cycle"].by_key().get(
                    "color",
                    ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
                )
                for i, t in enumerate(trial_ids):
                    col = colors[i % len(colors)]
                    if (df_blk is not None) and (not df_blk.empty):
                        sub = df_blk[df_blk["trial"] == t]
                        if not sub.empty:
                            ax.plot(sub["time_s"], sub["theta_pred_deg"],
                                    color=col, linewidth=linewidth, alpha=alpha_curves,
                                    linestyle="-", label=f"{label_blk} • trial {t}")
                    if (df_cam is not None) and (not df_cam.empty):
                        sub = df_cam[df_cam["trial"] == t]
                        if not sub.empty:
                            ax.plot(sub["time_s"], sub["theta_pred_deg"],
                                    color=col, linewidth=linewidth, alpha=alpha_curves,
                                    linestyle="--", label=f"{label_cam} • trial {t}")
                handles, labels = ax.get_legend_handles_labels()
                seen = set()
                dedup = [(h, l) for h, l in zip(handles, labels) if not (l in seen or seen.add(l))]
                if dedup:
                    ax.legend(*zip(*dedup), frameon=False, ncol=1)
                ax.set_title(panel_title)
                ax.set_xlabel("Time (s)")
                ax.set_ylim(deg_min - 2, deg_max + 2)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.grid(alpha=0.25, axis="y")

            ax = axes[0]
            _plot_panel(ax,
                        theta_blk1_first, theta_cam1_first,
                        f"{label_blk1}", f"{label_cam1}",
                        "First application: Block 1 & Cam 1 → adc_trials_first")
            ax.set_ylabel("Angle (deg)")

            ax = axes[1]
            _plot_panel(ax,
                        theta_blk2_second, theta_cam2_second,
                        f"{label_blk2}", f"{label_cam2}",
                        "Second application: Block 2 & Cam 2 → adc_trials_second")

            fig.tight_layout()

        if verbose:
            print(f"[four] OOB first: over={oob_counts['first']['over_hi']}, under={oob_counts['first']['under_lo']}")
            print(
                f"[four] OOB second: over={oob_counts['second']['over_hi']}, under={oob_counts['second']['under_lo']}")

        # optional: stable time sort before return
        def _maybe_sort_time(df):
            if df is None or df.empty: return df
            if "timestamp" in df.columns: return df.sort_values(["set_label", "trial", "timestamp"])
            if "time_s" in df.columns:    return df.sort_values(["set_label", "trial", "time_s"])
            return df

        theta_cam1_first = _maybe_sort_time(theta_cam1_first)
        theta_blk1_first = _maybe_sort_time(theta_blk1_first)
        theta_cam2_second = _maybe_sort_time(theta_cam2_second)
        theta_blk2_second = _maybe_sort_time(theta_blk2_second)
        tall_all = _maybe_sort_time(tall_all)
        theta_all_wide = _maybe_sort_time(theta_all_wide)

        return {
            "anchors_mode": anchors_mode,
            "anchors_per_cal": anchors_per_cal,
            "theta_cam1_first": theta_cam1_first,
            "theta_blk1_first": theta_blk1_first,
            "theta_cam2_second": theta_cam2_second,
            "theta_blk2_second": theta_blk2_second,
            "theta_all_tall": tall_all,
            "theta_all_wide": theta_all_wide,
            "oob_counts": oob_counts,
            "fig": fig, "axes": axes,
        }

    def calibrate_from_pairs_and_stream(
            self,
            angle_adc_df: pd.DataFrame,
            *,
            adc_trials_first: list[pd.DataFrame] | None = None,
            adc_trials_second: list[pd.DataFrame] | None = None,
            adc_col_pairs: str = "adc_ch3",
            enc_col_pairs: str = "angle",
            # how many trials to stack for the external stream (first set)
            stack_limit: int = 15,
            # robust external anchors from quantiles of the stream
            q_hi: float = 0.999,  # maps to 0° (y0)
            q_lo: float = 0.001,  # maps to 90° (y90)
            # inversion config
            deg_min: float = 0.0,
            deg_max: float = 90.0,
            clip_z: bool = True,
            extrapolate: bool = False,
            # tall-table config
            set_label_first: str = "first",
            set_label_second: str = "second",
            trial_len_sec: float = 10.0,
            clamp_theta: bool = True,
            verbose: bool = True,
    ) -> dict:
        """
        Convenience orchestrator:
          1) Fit quadratic (c0,c1,c2) on labeled angle–ADC pairs WITHOUT setting anchors.
          2) Build an external ADC stream from adc_trials_first (or fall back to pairs).
          3) Compute robust anchors (y0,y90) from stream quantiles and update self.calib.
          4) Build quadratic inverse (ADC→θ).
          5) Produce tall θ tables for first/second sets, with OOB flags against anchors.

        Returns dict with:
          {
            'anchors': {'y0':..., 'y90':..., 'source':...},
            'theta_first': <DataFrame>,
            'theta_second': <DataFrame>,
            'theta_all': <DataFrame>,
            'oob_counts': {'first': {'over_hi': int, 'under_lo': int},
                           'second': {'over_hi': int, 'under_lo': int}}
          }
        """
        import numpy as _np
        import pandas as _pd

        # -------- 1) Fit only the shape on labeled pairs --------
        if verbose:
            print("[cal] fitting quadratic (shape only) on labeled pairs…")
        if enc_col_pairs not in angle_adc_df.columns:
            raise KeyError(f"enc_col_pairs '{enc_col_pairs}' missing from angle_adc_df")
        if adc_col_pairs not in angle_adc_df.columns:
            # permissive fallback: pick first 'adc*' column
            cand = next((c for c in angle_adc_df.columns if str(c).lower().startswith("adc")), None)
            if cand is None:
                raise KeyError(f"adc_col_pairs '{adc_col_pairs}' missing and no 'adc*' fallback found.")
            adc_col_pairs = cand

        angle_adc_local = angle_adc_df[[enc_col_pairs, adc_col_pairs]].rename(
            columns={enc_col_pairs: "angle", adc_col_pairs: "adc"}
        ).copy()
        angle_adc_local["angle"] = _pd.to_numeric(angle_adc_local["angle"], errors="coerce")
        angle_adc_local["adc"] = _pd.to_numeric(angle_adc_local["adc"], errors="coerce")
        angle_adc_local = angle_adc_local.dropna()
        # restrict to [0,90] with a forgiving widen-then-clip if sparse
        m = angle_adc_local["angle"].between(0.0, 90.0)
        if m.sum() < 5:
            m = angle_adc_local["angle"].between(-5.0, 95.0)
        angle_adc_local = angle_adc_local.loc[m].copy()
        angle_adc_local["angle"] = angle_adc_local["angle"].clip(deg_min, deg_max)

        # Fit quadratic + set calib c0,c1,c2; DO NOT set y0,y90 yet
        self.fit_and_set_calibration(
            angle_adc_local.rename(columns={"angle": "angle", "adc": adc_col_pairs}),
            angle_col="angle",
            adc_col=adc_col_pairs,
            robust=True,
            anchors_source="fit_only",
            deg_min=deg_min,
            deg_max=deg_max,
        )
        if verbose:
            print(f"[cal] calib (shape): c0={self.calib.get('c0'):.6g}, "
                  f"c1={self.calib.get('c1'):.6g}, c2={self.calib.get('c2'):.6g}")

        # -------- 2) Build an external ADC stream (prefer first-set trials) --------
        def _stack_adc(trials, limit=15, adc_col="adc_ch3"):
            parts = []
            for df in trials[:limit]:
                if df is None or df.empty:
                    continue
                col = adc_col if adc_col in df.columns else next(
                    (c for c in df.columns if str(c).lower().startswith("adc")), None
                )
                if col is None:
                    continue
                parts.append(_pd.to_numeric(df[col], errors="coerce"))
            return _pd.concat(parts, ignore_index=True) if parts else _pd.Series(dtype=float)

        if verbose:
            print("[cal] building external ADC stream for robust anchors…")
        adc_series_calib = _pd.Series(dtype=float)
        if isinstance(adc_trials_first, (list, tuple)) and len(adc_trials_first) > 0:
            adc_series_calib = _stack_adc(adc_trials_first, limit=stack_limit, adc_col="adc_ch3")

        if adc_series_calib.empty:
            # fallback: use the pairs table
            base_adc_col = adc_col_pairs if adc_col_pairs in angle_adc_df.columns else next(
                (c for c in angle_adc_df.columns if str(c).lower().startswith("adc")), None
            )
            if base_adc_col is None:
                raise KeyError("No ADC column available in angle_adc_df for fallback.")
            adc_series_calib = _pd.to_numeric(angle_adc_df[base_adc_col], errors="coerce").dropna()

        y_raw = _np.asarray(adc_series_calib, float)
        y0 = float(_np.nanquantile(y_raw, q_hi))  # near-maximum -> maps to 0°
        y90 = float(_np.nanquantile(y_raw, q_lo))  # near-minimum -> maps to 90°
        self.calib.update(y0=y0, y90=y90, source="fit+external_minmax")
        if verbose:
            print(f"[cal] anchors from stream: y0={y0:.6g} (q={q_hi}), y90={y90:.6g} (q={q_lo})")

        # -------- 3) Build inverse (ADC→θ) --------
        self.build_poly_inverse(clip_z=clip_z, extrapolate=extrapolate, deg_min=deg_min, deg_max=deg_max)

        # -------- 4) Tall θ tables for first & second sets --------
        if verbose:
            print("[cal] building tall θ tables (use_poly_inverse=True)…")
        theta_first = self.trials_to_tall_df(
            adc_trials_first if isinstance(adc_trials_first, (list, tuple)) else [],
            set_label=set_label_first,
            trial_len_sec=trial_len_sec,
            use_poly_inverse=True,
            clamp_theta=clamp_theta,
        ) if adc_trials_first else _pd.DataFrame(
            columns=["set_label", "trial", "time_s", "timestamp", "theta_pred_deg", "adc_ch3"])

        theta_second = self.trials_to_tall_df(
            adc_trials_second if isinstance(adc_trials_second, (list, tuple)) else [],
            set_label=set_label_second,
            trial_len_sec=trial_len_sec,
            use_poly_inverse=True,
            clamp_theta=clamp_theta,
        ) if adc_trials_second else _pd.DataFrame(
            columns=["set_label", "trial", "time_s", "timestamp", "theta_pred_deg", "adc_ch3"])

        # -------- 5) OOB flags v. anchors --------
        lo, hi = (y90, y0) if y90 < y0 else (y0, y90)

        def _flag_oob(adc_array):
            adc = _np.asarray(adc_array, float)
            over = adc > hi
            under = adc < lo
            return _pd.DataFrame({
                "adc_raw": adc,
                "oob_over_hi": over,
                "oob_under_lo": under,
                "oob_margin": _np.where(over, adc - hi, _np.where(under, lo - adc, 0.0)),
            })

        def _attach_flags(df):
            if df.empty:
                return df
            if "adc_ch3" not in df.columns:
                # permissive fallback to any 'adc*' col
                cand = next((c for c in df.columns if str(c).lower().startswith("adc")), None)
                if cand is None:
                    return df
                adc_col_use = cand
            else:
                adc_col_use = "adc_ch3"
            vals = _pd.to_numeric(df[adc_col_use], errors="coerce")
            flags = _flag_oob(vals)
            return _pd.concat([df.reset_index(drop=True), flags], axis=1)

        theta_first_f = _attach_flags(theta_first)
        theta_second_f = _attach_flags(theta_second)
        theta_all = _pd.concat([theta_first_f, theta_second_f],
                               ignore_index=True) if not theta_first_f.empty or not theta_second_f.empty else _pd.DataFrame()

        oob_counts = {
            "first": {
                "over_hi": int(theta_first_f["oob_over_hi"].sum()) if not theta_first_f.empty else 0,
                "under_lo": int(theta_first_f["oob_under_lo"].sum()) if not theta_first_f.empty else 0,
            },
            "second": {
                "over_hi": int(theta_second_f["oob_over_hi"].sum()) if not theta_second_f.empty else 0,
                "under_lo": int(theta_second_f["oob_under_lo"].sum()) if not theta_second_f.empty else 0,
            },
        }
        if verbose:
            print(f"OOB counts: first(over={oob_counts['first']['over_hi']}, under={oob_counts['first']['under_lo']}), "
                  f"second(over={oob_counts['second']['over_hi']}, under={oob_counts['second']['under_lo']})")

        # quick θ range sanity
        for name, df in [("first", theta_first_f), ("second", theta_second_f)]:
            if df.empty or "theta_pred_deg" not in df.columns:
                continue
            s = _pd.to_numeric(df["theta_pred_deg"], errors="coerce")
            if s.notna().any() and verbose:
                print(f"[{name}] θ_pred (deg): min..max {float(_np.nanmin(s)):.3f} .. "
                      f"{float(_np.nanmax(s)):.3f} (median {float(_np.nanmedian(s)):.3f}) n={int(s.notna().sum())}")

        return {
            "anchors": {"y0": y0, "y90": y90, "source": "fit+external_minmax"},
            "theta_first": theta_first_f,
            "theta_second": theta_second_f,
            "theta_all": theta_all,
            "oob_counts": oob_counts,
        }

    from typing import Optional, Union, List
    import pandas as pd
    import numpy as np

    from typing import Optional, Union, List
    import pandas as pd
    import numpy as np
    from pandas.api.types import is_timedelta64_dtype

    # --- Paste this method into BallBearingData in analysis.py ---

    def align_theta_all_to_cam_for_set(
            self,
            theta_all_set: pd.DataFrame,
            cam_trials: list[pd.DataFrame],
            *,
            enc_time_col: str = "timestamp",
            cam_time_col: str | None = None,
            cam_time_prefix: str = "ts",
            tolerance: int | float | str = 50000,  # you call with 50 ms in microseconds; keep numeric
            direction: str = "nearest",
            theta_col: str = "theta_pred_deg",
            keep_time_delta: bool = True,
            drop_unmatched: bool = True,
            return_concatenated: bool = False,
            trial_labels: list[int] | None = None,
            require_set_label_match: bool = True,
    ) -> list[pd.DataFrame] | pd.DataFrame:
        """
        Align per-trial camera rows to encoder (theta_all_set) by nearest time.
        - Converts various time formats to int64 nanoseconds for robust asof-merge
        - Optional strict matching on set_label
        - Returns list of aligned dfs (or concatenated if return_concatenated=True)
        """
        import numpy as np
        import pandas as pd
        from pandas.api.types import is_timedelta64_dtype  # <-- ensure available in this scope

        # ---------- helpers ----------
        def _coerce_tolerance_to_ns(tol) -> int | None:
            """Return tolerance in nanoseconds if tol is str/Timedelta; None if numeric (we'll post-filter)."""
            if isinstance(tol, (int, float)):
                # numeric → treat as microseconds to preserve your current call-pattern (50000 = 50 ms)
                return int(float(tol) * 1_000)  # us → ns
            # string like "50ms", "10ms", etc.
            td = pd.to_timedelta(tol)
            return int(td / pd.to_timedelta(1, "ns"))

        def _rel_seconds_from_ns(ns: pd.Series) -> pd.Series:
            n = pd.to_numeric(ns, errors="coerce").astype("float64")
            if n.notna().sum() == 0:
                return pd.Series(np.nan, index=ns.index, dtype="float64")
            n0 = np.nanmin(n)
            return (n - n0) / 1e9

        def _to_ns_generic(s: pd.Series) -> pd.Series:
            """Best-effort conversion of many time representations to int64 nanoseconds."""

            # 1) Try HHMMSSffffff (even if numeric floats)
            def _try_hhmmssffffff(ss: pd.Series) -> pd.Series:
                try:
                    ints = pd.to_numeric(ss, errors="coerce").dropna().apply(
                        lambda v: int(np.floor(float(v) + 0.5))
                    )
                except Exception:
                    return pd.Series(pd.NaT, index=ss.index)
                z = ints.astype("string").str.zfill(12)
                ok = z.str.fullmatch(r"\d{12}")
                if ok.mean() < 0.7:
                    return pd.Series(pd.NaT, index=ss.index)
                hh = z.str.slice(0, 2).astype(int)
                mm = z.str.slice(2, 4).astype(int)
                ss_ = z.str.slice(4, 6).astype(int)
                use = z.str.slice(6, 12).astype(int)
                td = (pd.to_timedelta(hh, unit="h") + pd.to_timedelta(mm, unit="m")
                      + pd.to_timedelta(ss_, unit="s") + pd.to_timedelta(use, unit="us"))
                out = pd.Series(td, index=ints.index).reindex(ss.index)
                return out

            td_hms = _try_hhmmssffffff(s)
            if is_timedelta64_dtype(td_hms) and td_hms.notna().any():
                return td_hms.astype("timedelta64[ns]").astype("int64")

            # 2) Timedelta-like strings via DLC helper if present
            try:
                td = DLC3DBendAngles._series_time_of_day_to_timedelta(s)  # if your class is available
            except Exception:
                td = pd.Series(pd.NaT, index=s.index)
            if is_timedelta64_dtype(td) and td.notna().any():
                return td.astype("timedelta64[ns]").astype("int64")

            # 3) Datetime strings/values
            dt = pd.to_datetime(s, errors="coerce", utc=True)
            if dt.notna().any():
                return dt.astype("int64")

            # 4) Numeric epochs or counters
            sn = pd.to_numeric(s, errors="coerce")
            if sn.notna().any():
                snn = sn.dropna()
                # Try epochs with guessed units
                for unit in ("ns", "us", "ms", "s"):
                    dte = pd.to_datetime(snn, unit=unit, errors="coerce", utc=True)
                    if dte.notna().mean() > 0.8 and dte.dt.year.between(2005, 2100).mean() > 0.8:
                        dt_all = pd.to_datetime(sn, unit=unit, errors="coerce", utc=True)
                        return dt_all.astype("int64")
                # Fallback: relative counter -> heuristic scale to ns
                rng = float(snn.max() - snn.min())
                if not np.isfinite(rng) or rng <= 0:
                    return sn.round().astype("int64")
                if rng < 1e6:
                    sc = 1e9  # seconds-range
                elif rng < 1e9:
                    sc = 1e6  # milliseconds-range
                elif rng < 1e12:
                    sc = 1e3  # microseconds-range
                else:
                    sc = 1.0  # already ns-range
                return (sn * sc).round().astype("int64")

            # nothing worked
            return pd.Series(np.nan, index=s.index, dtype="float64")

        # ---------- checks & setup ----------
        if theta_col not in theta_all_set.columns:
            raise KeyError(f"'{theta_col}' not found in theta_all_set.")
        if enc_time_col not in theta_all_set.columns:
            raise KeyError(f"'{enc_time_col}' not found in theta_all_set.")

        # map cam_trials -> labels
        if trial_labels is None:
            labels = list(range(1, len(cam_trials) + 1))
        else:
            if len(trial_labels) != len(cam_trials):
                raise ValueError("trial_labels length must match cam_trials length.")
            labels = trial_labels

        tol_ns = _coerce_tolerance_to_ns(tolerance)
        numeric_postfilter = isinstance(tolerance, (int, float))

        merged_trials: list[pd.DataFrame] = []

        # ---------- main loop ----------
        for cam_df, trial_id in zip(cam_trials, labels):
            # subset encoder rows for this trial (and possibly set_label)
            th_df = theta_all_set.loc[
                pd.to_numeric(theta_all_set.get("trial"), errors="coerce") == trial_id
                ].copy()

            # If enforcing set_label and camera has one, filter encoder side to match
            if require_set_label_match and isinstance(cam_df, pd.DataFrame) and ("set_label" in cam_df.columns):
                cam_sets = pd.Series(cam_df["set_label"]).dropna().astype(str).unique().tolist()
                if cam_sets and ("set_label" in th_df.columns):
                    th_df = th_df.loc[th_df["set_label"].astype(str).isin(cam_sets)].copy()

            if cam_df is None or cam_df.empty or th_df.empty:
                merged_trials.append(pd.DataFrame())
                continue

            # choose camera time column (explicit or by prefix)
            if cam_time_col is None:
                cam_cands = [c for c in cam_df.columns if str(c).lower().startswith(cam_time_prefix)]
                cam_col = cam_cands[0] if cam_cands else None
            else:
                cam_col = cam_time_col

            if cam_col is None or cam_col not in cam_df.columns:
                print(f"[alignθ] Trial {trial_id}: camera time column not found (prefix='{cam_time_prefix}').")
                merged_trials.append(pd.DataFrame())
                continue

            # LEFT = camera
            left = cam_df.copy()
            left["_t_ns"] = _to_ns_generic(left[cam_col])
            left = left.dropna(subset=["_t_ns"]).sort_values("_t_ns").reset_index(drop=False)
            left["_t_ns"] = left["_t_ns"].astype("int64")  # ensure int64 for merge_asof
            left["cam_time_s"] = _rel_seconds_from_ns(left["_t_ns"])

            # RIGHT = encoder subset
            extra_cols = ["time_s", "adc_ch3", "calib", "set_label"]
            right_cols = [enc_time_col, theta_col] + [c for c in extra_cols if c in th_df.columns]
            right = th_df[right_cols].copy()
            right["_t_ns_right"] = _to_ns_generic(right[enc_time_col])
            right = right.dropna(subset=["_t_ns_right"]).sort_values("_t_ns_right").reset_index(drop=True)
            right["_t_ns_right"] = right["_t_ns_right"].astype("int64")
            right["enc_time_s"] = _rel_seconds_from_ns(right["_t_ns_right"])

            if left.empty or right.empty:
                merged_trials.append(pd.DataFrame())
                continue

            # merge_asof (tolerance only if non-numeric; for numeric we post-filter by abs delta)
            merged = pd.merge_asof(
                left,
                right,
                left_on="_t_ns",
                right_on="_t_ns_right",
                direction=direction,
                allow_exact_matches=True,
                tolerance=None if numeric_postfilter else pd.to_timedelta(tol_ns, unit="ns")
            )

            # apply numeric tolerance post-filter
            merged["_delta_ns"] = (merged["_t_ns_right"] - merged["_t_ns"]).astype("float64")
            if numeric_postfilter and tol_ns is not None:
                merged = merged[merged["_delta_ns"].abs() <= tol_ns].copy()

            if drop_unmatched:
                merged = merged[pd.notna(merged["_t_ns_right"])].copy()

            if keep_time_delta:
                merged["_delta_ms"] = merged["_delta_ns"] / 1e6
                merged["_delta_sec"] = merged["_delta_ns"] / 1e9

            # stamp trial; keep/propagate set_label if present
            merged["trial"] = trial_id
            if "set_label" not in merged.columns:
                if "set_label" in right.columns:
                    merged["set_label"] = right["set_label"]
                elif "set_label" in cam_df.columns:
                    merged["set_label"] = cam_df["set_label"]

            # restore original camera row order
            merged = merged.set_index("index").sort_index().reset_index(drop=True)
            merged_trials.append(merged)

        if return_concatenated:
            non_empty = [df for df in merged_trials if not df.empty]
            return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()

        return merged_trials

    def plot_adc_and_dlc_twopanel(
            self,
            out: dict,
            *,
            cam_plus_dlc_first: list,
            cam_plus_dlc_second: list,
            dlc_angle_col: str = "metric_mcp_bend_deg_deg_dlc",
            cam_time_col: str = "ts_25185174",
            aligned_first: list | None = None,  # from align_theta_all_to_cam_for_set(...)
            aligned_second: list | None = None,  # from align_theta_all_to_cam_for_set(...)
            use_aligned: bool = True,
            # trial IDs (1-based ids used in the tall ADC tables)
            trial_first: int | None = None,
            trial_second: int | None = None,
            title_left: str = "Set1: Block1 + Cam1 (ADC) vs DLC(1) — aligned",
            title_right: str = "Set2: Block2 + Cam2 (ADC) vs DLC(2) — aligned",
            deg_min: float = 0.0,
            deg_max: float = 90.0,
            figsize=(12.5, 5.2),
            linewidth: float = 1.8,
            alpha_curves: float = 0.95,
            # --- drawing order & emphasis ---
            dlc_on_top: bool = True,
            dlc_zorder: int = 10,
            adc_zorder: int = 2,
            dlc_linewidth: float = 2.2,
            dlc_alpha: float = 1.0,
            # --- DLC y preprocessing (optional) ---
            auto_convert_rad: bool = True,  # auto-convert DLC radians→degrees if detected
            dlc_abs: bool = False,  # plot |DLC| if True
            verbose: bool = True,
    ):
        """
        Plot ADC theta (blk+cam) against DLC angles for one trial in each set.
        Key fixes:
          - DLC(aligned) retrieved by trial id -> index = trial_id - 1
          - DLC(raw cam) retrieved by matching stamped 'trial' in each df (not by list position)
        """

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        # ---- fetch tall tables from 'out' ----
        need = ["theta_blk1_first", "theta_cam1_first", "theta_blk2_second", "theta_cam2_second"]
        for k in need:
            if k not in out or out[k] is None:
                raise KeyError(f"Missing '{k}' in out.")
        blk1_first = out["theta_blk1_first"].copy()
        cam1_first = out["theta_cam1_first"].copy()
        blk2_second = out["theta_blk2_second"].copy()
        cam2_second = out["theta_cam2_second"].copy()

        def _first_trial_or_none(df):
            if df is None or df.empty:
                return None
            t = pd.to_numeric(df.get("trial"), errors="coerce").dropna().astype(int)
            return int(np.min(t)) if not t.empty else None

        # Auto-pick a trial id for each set if not provided (use min trial id present)
        if trial_first is None:
            trial_first = _first_trial_or_none(blk1_first if not blk1_first.empty else cam1_first)
        if trial_second is None:
            trial_second = _first_trial_or_none(blk2_second if not blk2_second.empty else cam2_second)

        if verbose:
            print(f"[twopanel] trial ids → set1: {trial_first}, set2: {trial_second}")

        # --- helpers: consistent selection by TRIAL ID ---
        def _subset_trial_adc(df, trial_id):
            """Rows with df['trial'] == trial_id (for ADC tall tables)."""
            if df is None or df.empty or trial_id is None:
                return pd.DataFrame(columns=df.columns if df is not None else ["time_s", "theta_pred_deg"])
            tt = pd.to_numeric(df.get("trial"), errors="coerce").astype("Int64")
            return df.loc[tt == int(trial_id)].copy()

        def _find_cam_df_for_trial_id(cam_list, trial_id):
            """
            Return the df from cam_list whose stamped 'trial' equals trial_id (by mode),
            ignoring position. Returns None if not found.
            """
            if cam_list is None or trial_id is None:
                return None
            for df in cam_list:
                if df is None or df.empty or "trial" not in df.columns:
                    continue
                s = pd.to_numeric(df["trial"], errors="coerce").dropna().astype(int)
                if not s.empty and int(s.mode().iat[0]) == int(trial_id):
                    return df
            return None

        def _get_aligned_df_for_trial_id(aligned_list, trial_id):
            """
            Aligned lists produced by remap_aligned_by_trial are 0-based by (trial_id - 1).
            """
            if not use_aligned or aligned_list is None or trial_id is None:
                return None
            idx = int(trial_id) - 1
            if 0 <= idx < len(aligned_list):
                df = aligned_list[idx]
                return None if (df is None or df.empty) else df
            return None

        def _rel_seconds(ts_series):
            """Convert any ts-like series to relative seconds starting at 0."""
            ts = pd.to_numeric(ts_series, errors="coerce").to_numpy()
            if ts.size == 0 or not np.isfinite(np.nanmin(ts)):
                return ts
            x = ts - np.nanmin(ts)
            mx = float(np.nanmax(x)) if np.isfinite(np.nanmax(x)) else 0.0
            if mx > 1e9:
                x = x / 1e9
            elif mx > 1e6:
                x = x / 1e6
            elif mx > 1e3:
                x = x / 1e3
            return x

        # ---------- DLC helpers: robust x, unit handling, warnings ----------
        def _prepare_dlc_series(y_raw, *, auto_convert_rad=True, dlc_abs=False):
            y = pd.to_numeric(y_raw, errors="coerce").to_numpy(dtype=float)
            if y.size == 0:
                return y
            # heuristic: if 95th percentile < ~6 rad, assume radians → degrees
            if auto_convert_rad:
                finite = y[np.isfinite(y)]
                if finite.size:
                    y95 = np.nanpercentile(np.abs(finite), 95)
                    if y95 < 6.0:
                        y = np.degrees(y)
            if dlc_abs:
                y = np.abs(y)
            return y

        def _pick_x_seconds_from_aligned(df, fallback_ts_prefixes=("ts_",)):
            """
            Priority for DLC x-axis:
              1) cam_time_s
              2) enc_time_s
              3) time_s or t_sec
              4) first ts_* column → relative seconds
            Returns (x, used_col_name, is_constant)
            """
            for name in ("cam_time_s", "enc_time_s", "time_s", "t_sec"):
                if name in df.columns:
                    x = pd.to_numeric(df[name], errors="coerce").to_numpy(dtype=float)
                    if x.size:
                        is_const = (np.nanmax(x) - np.nanmin(x)) < 1e-9
                        return x, name, is_const
            ts_cols = [c for c in df.columns if any(str(c).startswith(p) for p in fallback_ts_prefixes)]
            if ts_cols:
                s = _rel_seconds(df[ts_cols[0]])
                if s.size:
                    is_const = (np.nanmax(s) - np.nanmin(s)) < 1e-9
                    return s, ts_cols[0], is_const
            return np.array([]), None, False

        # ---------- plotting helpers ----------
        def _plot_adc(ax, df, color, linestyle, label):
            if df is None or df.empty:
                return False
            x = pd.to_numeric(df.get("time_s"), errors="coerce")
            y = pd.to_numeric(df.get("theta_pred_deg"), errors="coerce")
            if x.notna().any() and y.notna().any():
                ax.plot(
                    x, y,
                    color=color, linestyle=linestyle,
                    linewidth=linewidth, alpha=alpha_curves,
                    label=label, zorder=adc_zorder
                )
                return True
            return False

        def _plot_dlc_aligned(ax, aligned_list, trial_id, color, label):
            """Use aligned seconds for DLC if available (prefers cam_time_s)."""
            if not use_aligned or aligned_list is None or trial_id is None:
                return False
            # aligned_list is 0-based by trial_id-1
            slot = int(trial_id) - 1
            if not (0 <= slot < len(aligned_list)):
                return False
            df = aligned_list[slot]
            if df is None or df.empty or (dlc_angle_col not in df.columns):
                return False

            x, xname, is_const = _pick_x_seconds_from_aligned(df)
            y = _prepare_dlc_series(df[dlc_angle_col], auto_convert_rad=auto_convert_rad, dlc_abs=dlc_abs)
            if x.size and y.size:
                if is_const and verbose:
                    print(f"[twopanel] WARNING: DLC x-axis '{xname}' is constant for trial {trial_id}.")
                order = np.argsort(x, kind="stable")
                ax.plot(
                    x[order], y[order],
                    color=color, linestyle=":",
                    linewidth=dlc_linewidth, alpha=dlc_alpha,
                    label=label,
                    zorder=(dlc_zorder if dlc_on_top else adc_zorder + 1)
                )
                return True
            return False

        def _plot_dlc_raw(ax, cam_list, trial_id, color, label):
            """When passing a *raw* per-trial camera list (0-based by position), also translate trial_id→index."""
            if cam_list is None or trial_id is None:
                return False
            slot = int(trial_id) - 1
            if not (0 <= slot < len(cam_list)):
                return False
            df = cam_list[slot]
            if df is None or df.empty or (cam_time_col not in df.columns) or (dlc_angle_col not in df.columns):
                return False
            s = _rel_seconds(df[cam_time_col])
            if s.size == 0:
                return False
            is_const = (np.nanmax(s) - np.nanmin(s)) < 1e-9
            if is_const and verbose:
                print(f"[twopanel] WARNING: raw cam x-axis '{cam_time_col}' is constant for trial {trial_id}.")
            y = _prepare_dlc_series(df[dlc_angle_col], auto_convert_rad=auto_convert_rad, dlc_abs=dlc_abs)
            order = np.argsort(s, kind="stable")
            ax.plot(
                s[order], y[order],
                color=color, linestyle=":",
                linewidth=dlc_linewidth, alpha=dlc_alpha,
                label=label,
                zorder=(dlc_zorder if dlc_on_top else adc_zorder + 1)
            )
            return True

        # fixed colors
        colors = {
            "blk1": "tab:blue",
            "cam1": "tab:orange",
            "blk2": "tab:green",
            "cam2": "tab:red",
            "dlc1": "tab:purple",
            "dlc2": "tab:brown",
        }

        # subset per panel from ADC tall tables by trial ID
        df_b1_L = _subset_trial_adc(blk1_first, trial_first)
        df_c1_L = _subset_trial_adc(cam1_first, trial_first)
        df_b2_R = _subset_trial_adc(blk2_second, trial_second)
        df_c2_R = _subset_trial_adc(cam2_second, trial_second)

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

        # ================= LEFT: set1 ADC(blk1, cam1) + DLC(1) =================
        ax = axes[0]
        _plot_adc(ax, df_b1_L, colors["blk1"], "-", f"Blk1→set1 (trial {trial_first})")
        _plot_adc(ax, df_c1_L, colors["cam1"], "--", f"Cam1→set1 (trial {trial_first})")
        # DLC first: aligned → raw fallback
        if not _plot_dlc_aligned(ax, aligned_first, trial_first, colors["dlc1"], f"DLC(1) trial {trial_first}"):
            _plot_dlc_raw(ax, cam_plus_dlc_first, trial_first, colors["dlc1"], f"DLC(1) trial {trial_first}")

        ax.set_title(title_left)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angle (deg)")
        ax.set_ylim(deg_min - 2, deg_max + 2)
        ax.grid(alpha=0.25, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, ncol=1, loc="best")

        # ================= RIGHT: set2 ADC(blk2, cam2) + DLC(2) =================
        ax = axes[1]
        _plot_adc(ax, df_b2_R, colors["blk2"], "-", f"Blk2→set2 (trial {trial_second})")
        _plot_adc(ax, df_c2_R, colors["cam2"], "--", f"Cam2→set2 (trial {trial_second})")
        if not _plot_dlc_aligned(ax, aligned_second, trial_second, colors["dlc2"], f"DLC(2) trial {trial_second}"):
            _plot_dlc_raw(ax, cam_plus_dlc_second, trial_second, colors["dlc2"], f"DLC(2) trial {trial_second}")

        ax.set_title(title_right)
        ax.set_xlabel("Time (s)")
        ax.set_ylim(deg_min - 2, deg_max + 2)
        ax.grid(alpha=0.25, axis="y")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, ncol=1, loc="best")

        fig.tight_layout()
        if verbose:
            print("[twopanel] done.")
        return {"fig": fig, "axes": axes, "trial_first": trial_first, "trial_second": trial_second}

    def remap_aligned_by_trial(
            self,
            aligned_list: list[pd.DataFrame],
            *,
            start_from_zero: bool = True,  # True → return index = trial_id - 1
            fallback_use_index: bool = True,  # If missing trial, fall back to i+1
            prefer_last: bool = True,  # If duplicates, last wins
            verbose: bool = True,
    ) -> list[pd.DataFrame]:
        """
        Build a dense list keyed by trial id.
          - If start_from_zero=True, output index k holds trial_id=(k+1)
          - If start_from_zero=False, output index t holds trial_id=t (index 0 unused/None)
        Robust to frames with no/NaN trial: optionally fall back to list index (i+1).
        """
        if not aligned_list:
            return []

        # 1) collect (trial_id, df) pairs
        pairs: list[tuple[int, pd.DataFrame]] = []
        bad_idxs: list[int] = []
        for i, df in enumerate(aligned_list):
            if df is None or df.empty:
                continue
            if "trial" in df.columns:
                tser = pd.to_numeric(df["trial"], errors="coerce").dropna().astype(int)
                if not tser.empty:
                    t = int(tser.mode().iloc[0])
                    pairs.append((t, df))
                    continue
            # no valid trial in df
            if fallback_use_index:
                t = i + 1
                pairs.append((t, df))
            else:
                bad_idxs.append(i)

        if not pairs:
            if verbose:
                print("[remap] No frames had usable trial IDs; returning empty mapping.")
            return []

        # 2) resolve duplicates: first or last wins
        by_trial: dict[int, pd.DataFrame] = {}
        for t, df in (pairs if not prefer_last else pairs):
            by_trial[t] = df  # overwrites prior if prefer_last=True (default)

        # 3) build dense list
        max_t = max(by_trial.keys())
        if start_from_zero:
            out = [None] * max_t  # index k → trial (k+1)
            for t, df in by_trial.items():
                idx = t - 1
                if 0 <= idx < len(out):
                    out[idx] = df
        else:
            out = [None] * (max_t + 1)  # index t → trial t, 0 unused
            for t, df in by_trial.items():
                out[t] = df

        # 4) optional logging
        if verbose:
            present = [i + 1 for i, df in enumerate(out) if df is not None] if start_from_zero else [i for i, df in
                                                                                                     enumerate(out) if
                                                                                                     i > 0 and df is not None]
            missing = [i + 1 for i, df in enumerate(out) if df is None] if start_from_zero else [i for i, df in
                                                                                                 enumerate(out) if
                                                                                                 i > 0 and df is None]
            print(f"[remap] Trials present: {present} | missing: {missing}")
            if bad_idxs:
                print(f"[remap] Frames without trial & not remapped (fallback_use_index=False): {bad_idxs}")

        return out

    def plot_dlc_abs_error_boxplots_from_aligned(
            self,
            *,
            aligned_first: list,
            aligned_second: list,
            dlc_angle_col: str = "metric_mcp_bend_deg_deg_dlc",
            adc_theta_col: str = "theta_pred_deg",
            calib_col: str = "calib",
            max_abs_dt_ms: float | None = 100.0,
            trial_first: int | None = None,
            trial_second: int | None = None,
            figsize=(10.5, 4.6),
            whisker: float = 1.5,
            verbose: bool = True,
    ):
        """
        Boxplots of |ADC - DLC| using pre-aligned lists (no re-alignment).
        Left:  |Blk1−DLC(1)|, |Cam1−DLC(1)|
        Right: |Blk2−DLC(2)|, |Cam2−DLC(2)|
        All boxes blue, black median, no outliers shown.

        Robust to:
          - delta-time column being named `_delta_ms` or `time_delta_ms`
          - deriving delta-time from enc_time_s/cam_time_s or `time_delta`
          - DLC column name differences (auto-detects a likely DLC column if missing)
          - trial index confusion (accepts 0-based or 1-based)
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        def _trial_indices(label, n):
            """Return the list of indices to use into aligned list."""
            if n <= 0:
                return []
            if label is None:
                return list(range(n))
            # accept 1-based or 0-based
            i1 = int(label) - 1
            if 0 <= i1 < n:
                return [i1]
            i0 = int(label)
            if 0 <= i0 < n:
                return [i0]
            return []

        def _ensure_time_delta_ms(df: pd.DataFrame) -> pd.Series:
            """Get/derive a time delta in milliseconds as a Series aligned to df.index."""
            if df is None or df.empty:
                return pd.Series(dtype=float)
            # 1) preferred column names
            if "time_delta_ms" in df.columns:
                return pd.to_numeric(df["time_delta_ms"], errors="coerce")
            if "_delta_ms" in df.columns:
                return pd.to_numeric(df["_delta_ms"], errors="coerce")
            # 2) derive from enc_time_s - cam_time_s
            if {"enc_time_s", "cam_time_s"}.issubset(df.columns):
                enc = pd.to_numeric(df["enc_time_s"], errors="coerce")
                cam = pd.to_numeric(df["cam_time_s"], errors="coerce")
                return (enc - cam) * 1000.0
            # 3) derive heuristically from generic 'time_delta' (unknown units)
            if "time_delta" in df.columns:
                td = pd.to_numeric(df["time_delta"], errors="coerce").astype(float)
                if td.notna().any():
                    p95 = float(np.nanpercentile(np.abs(td.dropna()), 95))
                    # crude unit guess
                    if p95 > 1e7:  # ns → ms
                        return td / 1e6
                    elif p95 > 1e4:  # us → ms
                        return td / 1e3
                    elif p95 > 10:  # ms already
                        return td
                    else:  # seconds → ms
                        return td * 1e3
            # fallback: all NaN
            return pd.Series(np.nan, index=df.index, dtype=float)

        def _pick_dlc_col(df: pd.DataFrame, wanted: str) -> str | None:
            """Return an existing DLC angle column name, preferring 'wanted'."""
            if wanted in df.columns:
                return wanted
            # try to find a likely candidate
            cand = next((c for c in df.columns
                         if ("dlc" in str(c).lower()) and ("mcp" in str(c).lower())
                         and ("deg" in str(c).lower() or "angle" in str(c).lower())),
                        None)
            return cand

        def _collect_abs_err(al_list, idxs, want_calibs, panel_tag="set"):
            vals = {c: [] for c in want_calibs}
            dbg_counts = {c: dict(rows=0, kept=0) for c in want_calibs}

            for i in idxs:
                if i is None or i >= len(al_list):
                    continue
                df = al_list[i]
                if df is None or df.empty:
                    if verbose:
                        print(f"[{panel_tag}] trial idx {i}: empty")
                    continue

                # ensure calib present
                if calib_col not in df.columns:
                    raise KeyError(
                        f"[{panel_tag}] trial idx {i} missing '{calib_col}'. "
                        "Keep it during alignment by including 'calib' among the right-hand columns."
                    )

                # time-delta filter
                dt_ms = _ensure_time_delta_ms(df)
                mask_dt = pd.Series(True, index=df.index)
                if max_abs_dt_ms is not None:
                    mask_dt = dt_ms.abs() <= float(max_abs_dt_ms)

                # dlc/adc series
                dlc_col = _pick_dlc_col(df, dlc_angle_col)
                if dlc_col is None:
                    if verbose:
                        print(f"[{panel_tag}] trial idx {i}: no DLC column found (looked for '{dlc_angle_col}')")
                    continue

                s_dlc = pd.to_numeric(df[dlc_col], errors="coerce")
                s_adc = pd.to_numeric(df.get(adc_theta_col), errors="coerce")
                s_cal = df[calib_col].astype(str)

                ok = mask_dt & s_dlc.notna() & s_adc.notna() & s_cal.notna()
                if verbose:
                    total_rows = int(len(df))
                    kept_rows = int(ok.sum())
                    print(f"[{panel_tag}] trial idx {i}: rows total={total_rows}, kept_by_dt={kept_rows}, "
                          f"dt_ms 5/95% ~ {np.nanpercentile(dt_ms, 5) if dt_ms.notna().any() else np.nan:.1f} / "
                          f"{np.nanpercentile(dt_ms, 95) if dt_ms.notna().any() else np.nan:.1f}")

                if not ok.any():
                    continue

                sub = pd.DataFrame({"dlc": s_dlc[ok], "adc": s_adc[ok], "cal": s_cal[ok]})
                sub["abs_err"] = (sub["adc"] - sub["dlc"]).abs()

                for c in want_calibs:
                    arr = sub.loc[sub["cal"] == c, "abs_err"].to_numpy()
                    dbg_counts[c]["rows"] += int((s_cal == c).sum())
                    dbg_counts[c]["kept"] += int((ok & (s_cal == c)).sum())
                    if arr.size:
                        vals[c].append(arr)

            for c in list(vals.keys()):
                vals[c] = np.concatenate(vals[c]) if vals[c] else np.array([], dtype=float)

            if verbose:
                for c in want_calibs:
                    print(f"[{panel_tag}] calib='{c}': kept {dbg_counts[c]['kept']} / rows {dbg_counts[c]['rows']}")

            return vals

        def _summ(arr: np.ndarray):
            if arr.size == 0:
                return dict(n=0, mean=np.nan, median=np.nan, p90=np.nan, iqr=np.nan, mad=np.nan)
            q25, med, q75, p90 = np.percentile(arr, [25, 50, 75, 90])
            iqr = float(q75 - q25)
            mad = float(np.median(np.abs(arr - med)))
            return dict(n=int(arr.size), mean=float(np.mean(arr)), median=float(med),
                        p90=float(p90), iqr=iqr, mad=mad)

        want1, want2 = ("blk1", "cam1"), ("blk2", "cam2")
        idxs1 = _trial_indices(trial_first, len(aligned_first) if aligned_first is not None else 0)
        idxs2 = _trial_indices(trial_second, len(aligned_second) if aligned_second is not None else 0)

        vals1 = _collect_abs_err(aligned_first, idxs1, want1, panel_tag="set1")
        vals2 = _collect_abs_err(aligned_second, idxs2, want2, panel_tag="set2")

        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, constrained_layout=True)

        data_L = [vals1["blk1"], vals1["cam1"]]
        data_R = [vals2["blk2"], vals2["cam2"]]

        # blue boxes, black median, hide outliers
        boxprops = dict(linewidth=1.2)
        medianprops = dict(color="black", linewidth=1.8)
        whiskerprops = dict(color="0.4", linewidth=1.2)
        capprops = dict(color="0.4", linewidth=1.2)

        bpl = axes[0].boxplot(
            data_L, labels=["|Blk1 − DLC(1)|", "|Cam1 − DLC(1)|"],
            whis=whisker, showfliers=False, patch_artist=True,
            boxprops=boxprops, medianprops=medianprops,
            whiskerprops=whiskerprops, capprops=capprops
        )
        bpr = axes[1].boxplot(
            data_R, labels=["|Blk2 − DLC(2)|", "|Cam2 − DLC(2)|"],
            whis=whisker, showfliers=False, patch_artist=True,
            boxprops=boxprops, medianprops=medianprops,
            whiskerprops=whiskerprops, capprops=capprops
        )

        def _blueify(bx):
            for patch in bx["boxes"]:
                patch.set_facecolor("tab:blue")
                patch.set_alpha(0.35)

        _blueify(bpl);
        _blueify(bpr)

        for ax in axes:
            ax.axhline(0, color="k", linewidth=0.8, alpha=0.5)
            ax.grid(axis="y", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        axes[0].set_title("Set 1: |error| vs DLC")
        axes[0].set_ylabel("|θ_ADC − θ_DLC| (deg)")
        axes[1].set_title("Set 2: |error| vs DLC")

        stats = {
            "set1": {"blk1_abs": _summ(data_L[0]), "cam1_abs": _summ(data_L[1])},
            "set2": {"blk2_abs": _summ(data_R[0]), "cam2_abs": _summ(data_R[1])},
            "params": {
                "dlc_angle_col": dlc_angle_col,
                "adc_theta_col": adc_theta_col,
                "calib_col": calib_col,
                "max_abs_dt_ms": max_abs_dt_ms,
                "trial_first": trial_first, "trial_second": trial_second,
            },
        }
        if verbose:
            print("[abs-err boxplots] done.")
        return {"fig": fig, "axes": axes, "stats": stats}

    def remap_aligned_by_trial_concat(self, aligned_list: list[pd.DataFrame]) -> list[pd.DataFrame]:
        """
        Return a 0-based dense list where index (t-1) holds a CONCATENATION
        of *all* aligned dfs whose stamped 'trial' == t.
        Preserves both calib streams (e.g., blk1 and cam1) for the same trial.
        """
        import pandas as pd
        from collections import defaultdict

        if aligned_list is None or len(aligned_list) == 0:
            return []

        buckets = defaultdict(list)
        max_t = 0

        for df in aligned_list:
            if df is None or df.empty or "trial" not in df.columns:
                continue
            tt = pd.to_numeric(df["trial"], errors="coerce").dropna().astype(int)
            if tt.empty:
                continue
            t = int(tt.mode().iat[0])
            max_t = max(max_t, t)
            buckets[t].append(df)

        if max_t == 0:
            return []

        out = [pd.DataFrame()] * max_t  # slots 0..(max_t-1)
        for t in range(1, max_t + 1):
            parts = buckets.get(t, [])
            if parts:
                cat = pd.concat(parts, ignore_index=True, sort=False)
                # nice-to-haves: sort by any available relative-time columns
                sort_cols = [c for c in ("cam_time_s", "enc_time_s", "time_s", "t_sec") if c in cat.columns]
                if sort_cols:
                    cat = cat.sort_values(sort_cols, kind="stable").reset_index(drop=True)
                out[t - 1] = cat
            else:
                out[t - 1] = pd.DataFrame()

        print(
            f"[remap+concat] trials present: {sorted(buckets.keys())} | missing: {[t for t in range(1, max_t + 1) if t not in buckets]}")
        return out

    import pandas as pd
    import numpy as np

    @staticmethod
    def _collect_aligned_rows(
            aligned_list: list,
            *,
            set_tag: str,
            dlc_angle_col: str = "metric_mcp_bend_deg_deg_dlc",
            adc_theta_col: str = "theta_pred_deg",
            calib_col: str = "calib",
            max_abs_dt_ms: float | None = None,
    ):
        """Flatten one aligned list into a single DataFrame with abs_err + filters applied."""
        rows = []
        for i, df in enumerate(aligned_list or []):
            if df is None or len(getattr(df, "columns", [])) == 0:
                continue

            if calib_col not in df.columns:
                raise KeyError(f"Missing '{calib_col}' in aligned df[{i}]")

            # Filter by |Δt| if provided and present
            mask_dt = pd.Series(True, index=df.index)
            if (max_abs_dt_ms is not None) and ("_delta_ms" in df.columns):
                mask_dt = pd.to_numeric(df["_delta_ms"], errors="coerce").abs() <= float(max_abs_dt_ms)

            s_dlc = pd.to_numeric(df.get(dlc_angle_col), errors="coerce")
            s_adc = pd.to_numeric(df.get(adc_theta_col), errors="coerce")
            s_cal = df[calib_col].astype(str)
            s_trial = pd.to_numeric(df.get("trial"), errors="coerce").astype("Int64")

            ok = mask_dt & s_dlc.notna() & s_adc.notna() & s_cal.notna() & s_trial.notna()
            if not ok.any():
                continue

            sub = pd.DataFrame({
                "set": set_tag,
                "trial": s_trial[ok].astype(int).to_numpy(),
                "trial_index": i,  # index within the list (0-based)
                "calib": s_cal[ok].to_numpy(),
                "theta_adc_deg": s_adc[ok].to_numpy(),
                "theta_dlc_deg": s_dlc[ok].to_numpy(),
            })
            sub["abs_err_deg"] = (sub["theta_adc_deg"] - sub["theta_dlc_deg"]).astype(float).abs()

            # (Optional) carry a reasonable x-axis if present (helps you revisit the spot)
            for col in ("cam_time_s", "enc_time_s", "time_s", "t_sec"):
                if col in df.columns:
                    sub[col] = pd.to_numeric(df.loc[ok, col], errors="coerce").to_numpy()
                    break

            rows.append(sub)

        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
            columns=["set", "trial", "trial_index", "calib", "theta_adc_deg", "theta_dlc_deg", "abs_err_deg"]
        )

    @staticmethod
    def find_max_abs_errors_across_sets(
            *,
            aligned_first: list,
            aligned_second: list,
            dlc_angle_col: str = "metric_mcp_bend_deg_deg_dlc",
            adc_theta_col: str = "theta_pred_deg",
            calib_col: str = "calib",
            max_abs_dt_ms: float | None = None,
            top_k: int = 10,
    ):
        """
        Returns:
          {
            "per_trial_calib_max": DataFrame  # one worst row per (set, trial, calib)
            "top_overall": DataFrame          # top-K worst rows overall across both sets
          }
        """
        import pandas as pd

        df1 = BallBearingData._collect_aligned_rows(
            aligned_first, set_tag="set1",
            dlc_angle_col=dlc_angle_col, adc_theta_col=adc_theta_col,
            calib_col=calib_col, max_abs_dt_ms=max_abs_dt_ms
        )
        df2 = BallBearingData._collect_aligned_rows(
            aligned_second, set_tag="set2",
            dlc_angle_col=dlc_angle_col, adc_theta_col=adc_theta_col,
            calib_col=calib_col, max_abs_dt_ms=max_abs_dt_ms
        )

        if df1.empty and df2.empty:
            empty = BallBearingData._collect_aligned_rows([], set_tag="set1")
            return {"per_trial_calib_max": empty, "top_overall": empty}

        all_rows = pd.concat([df1, df2], ignore_index=True)

        # ---- robust 1-D idxmax over groups ----
        # (avoid multidimensional key errors in different pandas versions)
        idx = (
            all_rows
            .groupby(["set", "trial", "calib"], sort=False)["abs_err_deg"]
            .idxmax()
        )

        # ensure we pass a 1-D indexer to .loc
        idx = getattr(idx, "to_numpy", lambda: idx)()

        per_trial_calib_max = (
            all_rows
            .loc[idx]
            .sort_values(["set", "trial", "calib"])
            .reset_index(drop=True)
        )

        top_overall = (
            all_rows
            .sort_values("abs_err_deg", ascending=False)
            .head(int(top_k))
            .reset_index(drop=True)
        )

        return {
            "per_trial_calib_max": per_trial_calib_max,
            "top_overall": top_overall
        }

    def extract_calib_means_by_set(
            self,
            *,
            adc_col_preferred: str = "adc_ch3",
            file_patterns: tuple[str, ...] = ("data_adc*.csv", "data_adc*"),
            angles_expected: tuple[float, ...] = (0.0, 22.5, 45.0, 67.5, 90.0),
            exclude_name_contains: tuple[str, ...] = ("C_Block",),
            exclude_sets: tuple[object, ...] = (),  # e.g., (3, 4) or ("3rd","fourth","set3")
            make_plot: bool = False,
            overlay_mean: bool = False,
            point_alpha: float = 0.25,
            point_size: float = 10,
            jitter: float = 0.25,
            ax=None,
            # NEW: snap near-miss folder angles (e.g., calib86/87) to nearest expected angle
            snap_tol_deg: float = 4.0,
    ) -> pd.DataFrame:
        """
        Find folders whose names contain 'calib' immediately followed by an angle number.
        Accepts calib0, calib22, calib22.5/22p5/22-5/22_5, calib67.5, calib90 (etc.).
        Returns tidy DataFrame with columns:
            ['set_label','set_idx','angle_deg','mean_adc','n','folder','adc_col'].

        If make_plot=True, scatters all points and (optionally) overlays per-set means.
        """
        import re as _re
        import numpy as _np
        import pandas as _pd
        from pathlib import Path as _Path

        if not self.root_dir.exists():
            raise FileNotFoundError(f"Root directory not found: {self.root_dir}")

        angles_set = set(float(a) for a in angles_expected)

        # -------- parsing helpers (no regex needed) --------
        def _parse_after_calib_token(name: str):
            s = name.lower()
            k = s.find("calib")
            if k < 0:
                return None, None
            i = k + 5  # start after 'calib'
            n = len(s)

            # read integer part
            start = i
            while i < n and s[i].isdigit():
                i += 1
            if i == start:  # no digits after 'calib'
                return None, None

            whole = int(s[start:i])

            # optional fractional part
            frac = None
            if i < n and s[i] in ('.', 'p', '-', '_'):
                i += 1
                start2 = i
                while i < n and s[i].isdigit():
                    i += 1
                if i > start2:
                    frac = int(s[start2:i])

            return whole, frac

        def _angle_from_tokens(whole: int, frac: int | None) -> float | None:
            # CHANGED: compute raw value first
            if frac is not None:
                scale = 10.0 if frac < 10 else (100.0 if frac < 100 else 1.0)
                val = float(whole) + (float(frac) / scale)
            else:
                if float(whole) in angles_set:
                    val = float(whole)
                elif (float(whole) + 0.5) in angles_set and (float(whole) not in angles_set):
                    val = float(whole) + 0.5
                else:
                    val = float(whole)

            # exact match is fine
            if val in angles_set:
                return val

            # NEW: snap near-misses to nearest expected angle within tolerance
            nearest = min(angles_set, key=lambda a: abs(a - val))
            if abs(nearest - val) <= snap_tol_deg:
                return nearest

            return None

        # -------- collect candidate folders --------
        def _has_calib_token(p: _Path) -> bool:
            return "calib" in p.name.lower()

        cands = [p for p in self.root_dir.glob("*") if p.is_dir() and _has_calib_token(p)]
        if not cands:
            cands = [p for p in self.root_dir.rglob("*") if p.is_dir() and _has_calib_token(p)]

        # Exclude blocked substrings (case-insensitive)
        if exclude_name_contains:
            bad_l = tuple(s.lower() for s in exclude_name_contains)
            cands = [p for p in cands if all(b not in p.name.lower() for b in bad_l)]

        if not cands:
            print("[extract_calib_means_by_set] No calib folders found after exclusions.")
            return _pd.DataFrame(columns=["set_label", "set_idx", "angle_deg", "mean_adc", "n", "folder", "adc_col"])

        cands = sorted(set(cands), key=lambda p: p.name)

        rows_summary = []
        rows_points = []  # store every individual adc value

        for folder in cands:
            tok = _parse_after_calib_token(folder.name)
            # NEW: be robust to (None, None)
            if tok is None or tok == (None, None):
                continue
            whole, frac = tok
            ang = _angle_from_tokens(whole, frac)
            if ang is None:
                continue

            # pick largest data_adc*.csv
            adc_files = []
            for pat in file_patterns:
                adc_files.extend([p for p in folder.glob(pat) if not self._is_bad_dot_underscore(p)])
            if not adc_files:
                for pat in file_patterns:
                    adc_files.extend([p for p in folder.rglob(pat) if not self._is_bad_dot_underscore(p)])
            if not adc_files:
                print(f"[WARN] No ADC CSVs in calib folder: {folder}")
                continue

            adc_files = sorted(
                set(adc_files),
                key=lambda p: p.stat().st_size if p.exists() else 0,
                reverse=True
            )
            df0 = self._read_csv_safe(adc_files[0])
            if df0 is None or df0.empty:
                print(f"[WARN] Unreadable/empty ADC CSV in calib folder: {folder}")
                continue

            # choose ADC column
            adc_col = adc_col_preferred if adc_col_preferred in df0.columns else None
            if adc_col is None:
                adc_like = [c for c in df0.columns if str(c).lower().startswith("adc")]
                if not adc_like:
                    print(f"[WARN] No ADC-like columns in {adc_files[0].name}")
                    continue
                adc_col = adc_like[0]

            vals = _pd.to_numeric(df0[adc_col], errors="coerce").to_numpy(float)
            vals = vals[_np.isfinite(vals)]
            if vals.size == 0:
                print(f"[WARN] All-NaN ADC in {adc_files[0].name}")
                continue

            # summary row
            rows_summary.append(dict(angle_deg=float(ang),
                                     mean_adc=float(_np.nanmean(vals)),
                                     n=int(vals.size),
                                     folder=str(folder),
                                     adc_col=str(adc_col)))

            # point rows
            for v in vals:
                rows_points.append(dict(angle_deg=float(ang),
                                        adc_val=float(v),
                                        folder=str(folder),
                                        adc_col=str(adc_col)))

        if not rows_summary:
            return _pd.DataFrame(columns=["set_label", "set_idx", "angle_deg", "mean_adc", "n", "folder", "adc_col"])

        import pandas as _pd
        df = _pd.DataFrame(rows_summary).sort_values(["folder", "angle_deg"]).reset_index(drop=True)
        df_pts = _pd.DataFrame(rows_points).sort_values(["folder", "angle_deg"]).reset_index(drop=True)

        # Assign set indices/labels by blocks of k in sorted order
        # (This is what makes cam1/cam2 work when you pass a single DataFrame.)
        k = len(angles_expected)
        set_idx = [(i // k) + 1 for i in range(len(df))]  # 1-based
        df["set_idx"] = set_idx

        def _label_for(idx: int) -> str:
            return "first" if idx == 1 else ("second" if idx == 2 else f"set{idx}")

        df["set_label"] = [_label_for(i) for i in df["set_idx"]]

        # Map same set info onto point table
        df["_block_key"] = list(zip(df["folder"], df["angle_deg"].astype(float)))
        df_pts["_block_key"] = list(zip(df_pts["folder"], df_pts["angle_deg"].astype(float)))
        df_pts = df_pts.merge(
            df[["_block_key", "set_idx", "set_label"]],
            on="_block_key",
            how="left"
        ).drop(columns=["_block_key"])

        # Exclude sets if requested
        import re as _re
        def _parse_set_token(tok) -> int | None:
            if isinstance(tok, int):
                return tok
            s = str(tok).strip().lower()
            named = {
                "first": 1, "1st": 1,
                "second": 2, "2nd": 2,
                "third": 3, "3rd": 3,
                "fourth": 4, "4th": 4,
                "fifth": 5, "5th": 5,
            }
            if s in named:
                return named[s]
            m = _re.match(r"^(?:set)?\s*(\d+)(?:st|nd|rd|th)?$", s)
            if m:
                return int(m.group(1))
            return None

        if exclude_sets:
            bad_idxs = {i for i in (_parse_set_token(t) for t in exclude_sets) if i is not None}
            if bad_idxs:
                df = df[~df["set_idx"].isin(bad_idxs)].reset_index(drop=True)
                df_pts = df_pts[~df_pts["set_idx"].isin(bad_idxs)].reset_index(drop=True)

        # Order columns & angle category
        if df.empty:
            return _pd.DataFrame(columns=["set_label", "set_idx", "angle_deg", "mean_adc", "n", "folder", "adc_col"])

        df["angle_deg"] = _pd.Categorical(df["angle_deg"], categories=list(angles_expected), ordered=True)
        df = df.sort_values(["set_idx", "angle_deg"]).reset_index(drop=True)

        # ---------- PLOTTING ----------
        if make_plot and not df_pts.empty:
            import numpy as _np
            import matplotlib.pyplot as plt
            if ax is None:
                _, ax = plt.subplots(figsize=(6, 4))

            angle_to_x = {float(a): float(a) for a in angles_expected}

            for (idx, lbl), sub_pts in df_pts.groupby(["set_idx", "set_label"], sort=False):
                x = sub_pts["angle_deg"].astype(float).map(angle_to_x).to_numpy()
                x = x + _np.random.uniform(-jitter, jitter, size=x.size)
                ax.scatter(x, sub_pts["adc_val"].to_numpy(), s=point_size, alpha=point_alpha, label=None)

                if overlay_mean:
                    sub_mean = (
                        sub_pts.groupby("angle_deg", sort=False)["adc_val"]
                        .mean()
                        .reset_index()
                    )
                    ax.plot(sub_mean["angle_deg"].astype(float), sub_mean["adc_val"], lw=1.0, label=str(lbl))

            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("ADC")
            ax.set_title("Calibration samples (ADC vs angle)")
            ax.grid(alpha=0.25)
            if overlay_mean:
                ax.legend(title="Set")

        return df



    @staticmethod
    def _polyfit_quadratic(x: np.ndarray, y: np.ndarray, robust: bool) -> Tuple[float, float, float]:
        x = np.asarray(x, float).reshape(-1)
        y = np.asarray(y, float).reshape(-1)
        mask = np.isfinite(x) & np.isfinite(y)
        x, y = x[mask], y[mask]
        if x.size < 3:
            raise ValueError("Need at least 3 points for quadratic fit.")
        if not robust:
            c2, c1, c0 = np.polyfit(x, y, 2)
            return c0, c1, c2
        w = np.ones_like(y)
        for _ in range(3):
            X = np.vstack([np.ones_like(x), x, x**2]).T
            W = np.diag(w)
            beta = np.linalg.lstsq(W @ X, W @ y, rcond=None)[0]
            yhat = X @ beta
            r = y - yhat
            s = 1.4826 * np.median(np.abs(r - np.median(r))) if np.any(r) else 1.0
            s = max(s, 1e-8)
            k = 1.345 * s
            w = np.where(np.abs(r) <= k, 1.0, k / (np.abs(r) + 1e-12))
        return float(beta[0]), float(beta[1]), float(beta[2])

    def fit_and_set_calibration(
            self,
            angle_adc_df: pd.DataFrame,
            angle_col: str = "angle",
            adc_col: str = "adc_ch3",
            robust: bool = True,
            anchors_source: str = "fit_only",  # 'fit_only' | 'empirical' | 'empirical_minmax'
            deg_min: float = 0.0,
            deg_max: float = 90.0,
    ) -> Dict[str, float]:
        x = angle_adc_df[angle_col].to_numpy(float)
        y = angle_adc_df[adc_col].to_numpy(float)
        c0, c1, c2 = self._polyfit_quadratic(x, y, robust=robust)
        p = np.poly1d([c2, c1, c0])

        if anchors_source == "empirical_minmax":
            # Map max ADC -> 0°, min ADC -> 90°
            y0 = float(np.nanmax(y))   # ADC at 0°
            y90 = float(np.nanmin(y))  # ADC at 90°
        elif anchors_source == "empirical":
            def endpoint_avg(target_deg: float, win: float = 2.5):
                m = np.isfinite(x) & np.isfinite(y) & (np.abs(x - target_deg) <= win)
                return float(np.nanmean(y[m])) if np.any(m) else float(p(target_deg))
            y0 = endpoint_avg(deg_min)
            y90 = endpoint_avg(deg_max)
        else:  # 'fit_only'
            y0 = float(p(deg_min))
            y90 = float(p(deg_max))

        self.calib = dict(c0=c0, c1=c1, c2=c2, y0=y0, y90=y90, source=anchors_source)

        yhat = p(x)
        ss_res = np.sum((y - yhat) ** 2)
        ss_tot = np.sum((y - np.nanmean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        return dict(
            anchors_source=anchors_source,
            c0=c0, c1=c1, c2=c2,
            y0_emp=y0, y90_emp=y90,
            r2=r2,
        )

    # ---------- normalization & mapping ----------
    def adc_to_norm(self, adc: np.ndarray) -> np.ndarray:
        if not self.calib:
            raise RuntimeError("Calibration not set. Call fit_and_set_calibration(...) first.")
        y0, y90 = self.calib["y0"], self.calib["y90"]
        adc = np.asarray(adc, float)
        if y90 >= y0:
            z = (adc - y0) / max(1e-12, (y90 - y0))
        else:
            z = (y0 - adc) / max(1e-12, (y0 - y90))
        return np.clip(z, 0.0, 1.0)

    @staticmethod
    def norm_to_angle(z: np.ndarray, deg_min: float = 0.0, deg_max: float = 90.0) -> np.ndarray:
        z = np.asarray(z, float)
        return deg_min + z * (deg_max - deg_min)

    def _adc_to_theta_deg(self, adc_vals: np.ndarray, clamp: bool = True) -> np.ndarray:
        if not self.calib:
            raise RuntimeError("Calibration not set. Call fit_and_set_calibration(...) first.")
        y0  = float(self.calib["y0"])
        y90 = float(self.calib["y90"])
        adc_vals = np.asarray(adc_vals, dtype=float)
        if np.isnan(y0) or np.isnan(y90):
            theta = np.full_like(adc_vals, np.nan, dtype=float)
        elif y90 >= y0:
            theta = (adc_vals - y0) / max(1e-12, (y90 - y0)) * 90.0
        else:
            theta = (y0 - adc_vals) / max(1e-12, (y0 - y90)) * 90.0
        if clamp:
            theta = np.clip(theta, 0.0, 90.0)
        return theta

    # ---------- quadratic inverse (NEW) ----------
    def _adc_to_theta_deg_poly(self, adc_vals: np.ndarray, *, clamp: bool = True) -> np.ndarray:
        """
        Invert the fitted quadratic y = c2*x^2 + c1*x + c0 to get angle x (deg) from ADC y.
        Chooses the root consistent with the global slope between anchors. Falls back to
        linear normalized mapping when allowed by config and no real root exists.
        """
        if not self.calib:
            raise RuntimeError("Calibration not set. Call fit_and_set_calibration(...) first.")
        if not getattr(self, "_poly_inv_cfg", None):
            raise RuntimeError("Poly inverse not built. Call build_poly_inverse(...) first.")

        c0 = float(self.calib.get("c0", np.nan))
        c1 = float(self.calib.get("c1", np.nan))
        c2 = float(self.calib.get("c2", np.nan))
        y0 = float(self.calib.get("y0", np.nan))
        y90 = float(self.calib.get("y90", np.nan))

        deg_min = float(self._poly_inv_cfg["deg_min"])
        deg_max = float(self._poly_inv_cfg["deg_max"])
        extrap  = bool(self._poly_inv_cfg["extrapolate"])
        clip_z  = bool(self._poly_inv_cfg["clip_z"])

        y = np.asarray(adc_vals, float)
        out = np.full_like(y, np.nan, dtype=float)

        # Optional: clip ADC to anchor range before inversion
        if clip_z and np.isfinite(y0) and np.isfinite(y90):
            lo, hi = (y90, y0) if y90 < y0 else (y0, y90)
            y = np.clip(y, lo, hi)

        # Desired overall slope sign between endpoints (secant)
        secant = (y90 - y0) / max(1e-12, (deg_max - deg_min))
        desired_sign = np.sign(secant) if np.isfinite(secant) else 0.0

        a = c2
        b = c1
        for i, yi in enumerate(y):
            if not np.isfinite(yi):
                continue
            c = c0 - yi
            if abs(a) < 1e-14:  # effectively linear
                x = (yi - c0) / b if abs(b) >= 1e-14 else np.nan
            else:
                disc = b*b - 4.0*a*c
                if disc >= 0.0:
                    s = np.sqrt(disc)
                    r1 = (-b + s) / (2.0*a)
                    r2 = (-b - s) / (2.0*a)
                    candidates = np.array([r1, r2], float)

                    # Prefer roots inside [deg_min, deg_max]
                    in_rng = (candidates >= deg_min) & (candidates <= deg_max)
                    cand = candidates[in_rng] if in_rng.any() else candidates

                    if desired_sign != 0:
                        deriv_sign = np.sign(2.0*a*cand + b)  # y'(x) = 2ax + b
                        ok = np.where(deriv_sign == desired_sign)[0]
                        if ok.size > 0:
                            x = float(cand[ok[0]])
                        else:
                            # fallback: pick the one closest to range (or first if tie)
                            x = float(cand[np.argmin(np.minimum(np.abs(cand - deg_min), np.abs(cand - deg_max)))])
                    else:
                        x = float(cand[0])
                else:
                    if extrap and np.isfinite(y0) and np.isfinite(y90):
                        # Linear normalized fallback
                        z = (yi - y0) / max(1e-12, (y90 - y0))
                        x = deg_min + z * (deg_max - deg_min)
                    else:
                        x = np.nan

            if clamp:
                x = float(np.clip(x, deg_min, deg_max))
            out[i] = x

        return out

    def build_poly_inverse(
        self,
        *,
        clip_z: bool = True,
        extrapolate: bool = False,
        deg_min: float = 0.0,
        deg_max: float = 90.0,
    ) -> None:
        """
        Prepare quadratic inverse mapping ADC→angle using current calib {c0,c1,c2,y0,y90}.
        - clip_z:      clip ADC to [min(y0,y90), max(y0,y90)] before inversion
        - extrapolate: when quadratic has no real root, fall back to linear anchor mapping
        - deg_min/max: expected angle range used for clamping and slope selection
        """
        if not self.calib or not all(k in self.calib for k in ("c0","c1","c2","y0","y90")):
            raise RuntimeError("Calibration incomplete. Fit and set calibration first (c0,c1,c2,y0,y90).")
        self._poly_inv_cfg = dict(
            clip_z=bool(clip_z),
            extrapolate=bool(extrapolate),
            deg_min=float(deg_min),
            deg_max=float(deg_max),
        )

    # ---------- tall DF builder ----------
    def trials_to_tall_df(
            self,
            adc_trials: List[pd.DataFrame],
            set_label: str,
            trial_len_sec: float = 10.0,
            adc_col: str = "adc_ch3",
            time_col_options: Tuple[str, ...] = ("timestamp", "time", "t_sec"),
            include_endpoint: bool = True,
            clamp_theta: bool = False,      # no clipping by default
            use_poly_inverse: bool = False, # choose quadratic inverse
    ) -> pd.DataFrame:
        if not self.calib:
            raise RuntimeError("Calibration not set. Call fit_and_set_calibration(...) first.")

        use_poly = bool(use_poly_inverse) and getattr(self, "_poly_inv_cfg", None) is not None

        parts = []
        for trial_idx, df in enumerate(adc_trials, start=1):
            if df is None or df.empty:
                continue
            df = df.copy()

            # choose ADC column
            if adc_col not in df.columns:
                adc_like = [c for c in df.columns if str(c).lower().startswith("adc")]
                if not adc_like:
                    continue
                adc_use = adc_like[0]
            else:
                adc_use = adc_col

            n = len(df)
            time_s = np.linspace(0.0, float(trial_len_sec), num=n, endpoint=include_endpoint, dtype=float)

            # passthrough timestamp if present
            ts = None
            for tcol in time_col_options:
                if tcol in df.columns:
                    ts = df[tcol].copy()
                    break
            if ts is None:
                ts = pd.Series([np.nan] * n)

            adc_vals = pd.to_numeric(df[adc_use], errors="coerce").to_numpy(float)

            if use_poly:
                theta_deg = self._adc_to_theta_deg_poly(adc_vals, clamp=clamp_theta)
            else:
                theta_deg = self._adc_to_theta_deg(adc_vals, clamp=clamp_theta)

            parts.append(pd.DataFrame({
                "set_label": set_label,
                "trial": trial_idx,
                "time_s": time_s,
                "timestamp": ts,
                "theta_pred_deg": theta_deg,
                adc_use: adc_vals,
            }))

        if not parts:
            return pd.DataFrame(columns=["set_label", "trial", "time_s", "timestamp", "theta_pred_deg", adc_col])

        return pd.concat(parts, ignore_index=True)

    def compute_dlc3d_angles_by_trial(
            self,
            dlc3d_trials: List[pd.DataFrame],
            *,
            set_label: str,
            signed_in_plane: bool = True,
            add_plane_ok: bool = True,
    ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        """
        From per-trial DLC3D DataFrames (3-row MultiIndex columns), compute:
          • wrist_bend_deg  = angle(forearm→hand, hand→MCP)
          • mcp_bend_deg    = angle(hand→MCP, MCP→PIP)
          • mcp_bend_in_wrist_plane_deg = MCP bend projected into wrist plane

        Returns (augmented_trials, tall_df).

        CHANGE: Each per-trial 'augmented' df now also carries scalar columns:
          - 'set_label' (string)
          - 'trial'     (int, 1-based)
        """
        import numpy as np
        import pandas as pd

        augmented: List[pd.DataFrame] = []
        parts: List[pd.DataFrame] = []

        for trial_idx, dlc_df in enumerate(dlc3d_trials, start=1):
            if dlc_df is None or dlc_df.empty:
                augmented.append(pd.DataFrame());
                continue

            cam = DLC3DBendAngles(dlc_df)

            # MCP bend
            hand_pts = cam.get_points("hand")
            mcp_pts = cam.get_points("MCP")
            pip_pts = cam.get_points("PIP")
            v1_mcp = cam.vector(hand_pts, mcp_pts)  # hand→MCP
            v2_mcp = cam.vector(mcp_pts, pip_pts)  # MCP→PIP

            # Wrist bend (+ plane)
            forearm_pts = cam.get_points("forearm")
            v1_wrist = cam.vector(forearm_pts, hand_pts)  # forearm→hand
            v2_wrist = cam.vector(hand_pts, mcp_pts)  # hand→MCP

            angles_mcp = cam.angle_from_vectors(v1_mcp, v2_mcp)
            angles_wrist = cam.angle_from_vectors(v1_wrist, v2_wrist)

            angles_mcp_plane, _, _, plane_ok = cam.angle_from_vectors_in_plane(
                v1=v1_mcp, v2=v2_mcp, plane_v1=v1_wrist, plane_v2=v2_wrist, signed=signed_in_plane
            )

            df_out = cam.df.copy()
            df_out[("metric", "mcp_bend_deg", "deg")] = angles_mcp
            df_out[("metric", "wrist_bend_deg", "deg")] = angles_wrist
            df_out[("metric", "mcp_bend_in_wrist_plane_deg", "deg")] = angles_mcp_plane
            if add_plane_ok:
                df_out[("metric", "wrist_plane_ok", "")] = plane_ok

            # === NEW: stamp labels on each per-trial augmented df ===
            # Use flat (single-level) columns for easy downstream access
            df_out["set_label"] = set_label
            df_out["trial"] = int(trial_idx)

            augmented.append(df_out)

            # try to carry a useful time axis if present
            time_col = None
            for c in df_out.columns:
                if "time" in str(c).lower() or "timestamp" in str(c).lower():
                    time_col = c;
                    break
            time_vals = df_out[time_col] if time_col is not None else pd.Series([np.nan] * len(df_out))

            parts.append(pd.DataFrame({
                "set_label": set_label,
                "trial": trial_idx,
                "frame": np.arange(len(df_out), dtype=int),
                "time_or_timestamp": pd.to_numeric(time_vals, errors="coerce"),
                "mcp_bend_deg": pd.to_numeric(df_out[("metric", "mcp_bend_deg", "deg")], errors="coerce"),
                "wrist_bend_deg": pd.to_numeric(df_out[("metric", "wrist_bend_deg", "deg")], errors="coerce"),
                "mcp_bend_in_wrist_plane_deg": pd.to_numeric(
                    df_out[("metric", "mcp_bend_in_wrist_plane_deg", "deg")], errors="coerce"),
                "wrist_plane_ok": plane_ok if add_plane_ok else True,
            }))

        tall = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["set_label", "trial", "frame", "time_or_timestamp",
                     "mcp_bend_deg", "wrist_bend_deg", "mcp_bend_in_wrist_plane_deg", "wrist_plane_ok"]
        )
        return augmented, tall

    # --- DLC3D header coercion ---
    @staticmethod
    def _coerce_dlc3d_multiindex(df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert common DLC 3D flat export into MultiIndex (scorer, bodypart, coord).
        Expected flat form:
            col0='scorer', row0='bodyparts', row1='coords', data from row2 onward.
        If already MultiIndex, returns a normalized copy (coords lowercased).
        """
        if isinstance(df.columns, pd.MultiIndex):
            new_cols = []
            for c in df.columns:
                if isinstance(c, tuple) and len(c) >= 3:
                    new_cols.append((str(c[0]), str(c[1]), str(c[2]).lower()))
                else:
                    new_cols.append(c)
            out = df.copy()
            out.columns = pd.MultiIndex.from_tuples(new_cols)
            return out

        # Quick sanity checks
        if df.shape[0] < 3 or df.shape[1] < 4:
            return df

        try:
            first_col_name = str(df.columns[0]).strip().lower()
            row0_first = str(df.iloc[0, 0]).strip().lower()
            row1_first = str(df.iloc[1, 0]).strip().lower()
        except Exception:
            return df

        looks_like_dlc3d_flat = ("scorer" in first_col_name and
                                 row0_first.startswith("bodypart") and
                                 row1_first.startswith("coord"))
        if not looks_like_dlc3d_flat:
            return df

        tuples = []
        use_cols = []
        for j in range(1, df.shape[1]):  # skip first 'scorer' column
            scorer = str(df.columns[j]).strip()
            bpart  = str(df.iloc[0, j]).strip()
            coord  = str(df.iloc[1, j]).strip().lower()
            if bpart == "" and coord == "":
                continue
            if coord not in ("x", "y", "z"):
                continue
            tuples.append((scorer, bpart, coord))
            use_cols.append(j)

        if not tuples:
            return df

        df2 = df.iloc[2:, use_cols].reset_index(drop=True).copy()
        df2.columns = pd.MultiIndex.from_tuples(tuples)
        df2 = df2.apply(pd.to_numeric, errors="coerce")
        return df2

    # ---------- triggers / camera timestamps ----------
    def extract_trigger_dfs_by_trial(self, trial_folders: List[str]) -> List[pd.DataFrame]:
        dfs = self._extract_by_glob(trial_folders, pattern="data_trigger_time*.csv")
        if all((df is None or df.empty) for df in dfs):
            dfs = self._extract_by_glob(trial_folders, pattern="*trigger*.csv")
        return dfs

    def extract_mat_dfs_by_trial(
            self,
            trial_folders: List[str],
            mat_name: str = "flir.mat",
            prefix: str = "ts",
            *,
            # NEW labeling controls
            add_labels: bool = True,
            trial_labels: Optional[List[int]] = None,  # else trial_base..N-1
            trial_base: int = 1,  # set 0 for zero-based
            set_label: Optional[str] = None,  # single label for all
            set_labels: Optional[List[str]] = None,  # per-trial labels
            include_path: bool = False,  # optionally store source path in each df
    ) -> List[pd.DataFrame]:
        """
        For each trial folder, find a FLIR .mat file and load variables whose names
        start with `prefix` (e.g., 'ts*') into a single DataFrame per trial.
        If add_labels=True, stamps 'trial' and 'set_label' columns on each df.
        """
        import pandas as pd
        from pathlib import Path

        out: List[pd.DataFrame] = []

        # build labels
        n = len(trial_folders)
        if trial_labels is None:
            labels_trial = [trial_base + i for i in range(n)]
        else:
            if len(trial_labels) != n:
                raise ValueError("trial_labels length must match trial_folders length.")
            labels_trial = trial_labels

        if set_labels is not None and len(set_labels) != n:
            raise ValueError("set_labels length must match trial_folders length.")

        def _label_for(i: int) -> tuple[Optional[int], Optional[str]]:
            tlabel = labels_trial[i] if add_labels else None
            slabel = None
            if add_labels:
                slabel = set_labels[i] if set_labels is not None else set_label
            return tlabel, slabel

        for i, folder in enumerate(trial_folders):
            fp = Path(folder)

            cands: List[Path] = []
            exact = fp / mat_name
            if exact.exists() and not self._is_bad_dot_underscore(exact):
                cands.append(exact)
            if not cands:
                cands = [p for p in fp.glob("flir*.mat") if not self._is_bad_dot_underscore(p)]
            if not cands:
                cands = [p for p in fp.glob("*.mat") if not self._is_bad_dot_underscore(p)]
            if not cands:
                nested = list(fp.glob("**/*.mat"))
                cands = [p for p in nested if not self._is_bad_dot_underscore(p)]

            if not cands:
                # return an empty df but still stamp labels if requested
                df = pd.DataFrame()
                tlabel, slabel = _label_for(i)
                if add_labels:
                    if tlabel is not None: df["trial"] = [tlabel]
                    if slabel is not None: df["set_label"] = [slabel]
                if include_path: df["source_path"] = [str(fp)]
                out.append(df)
                continue

            cands_sorted = sorted(cands, key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
            mat_path = cands_sorted[0]

            try:
                df = self._mat_to_df(mat_path, prefix=prefix)
            except Exception as e:
                print(f"[WARN] Skipping unreadable MAT: {mat_path} ({e})")
                df = pd.DataFrame()

            # stamp labels (don’t overwrite if already present)
            if add_labels and not df.empty:
                tlabel, slabel = _label_for(i)
                if ("trial" not in df.columns) and (tlabel is not None):
                    df = df.copy();
                    df["trial"] = tlabel
                if ("set_label" not in df.columns) and (slabel is not None):
                    df = df.copy();
                    df["set_label"] = slabel
            elif add_labels and df.empty:
                # ensure at least a one-row df with labels, if you prefer; or keep empty
                pass

            if include_path:
                df = df.copy();
                df["source_path"] = str(mat_path)

            out.append(df)
        return out

    def attach_dlc_angles_to_cam_by_trial(
            self,
            cam_trials: List[pd.DataFrame],
            dlc_aug_trials: List[pd.DataFrame],
            *,
            cam_time_col: Optional[str] = None,  # e.g. 'ts_25183199'; auto-detect if None
            cam_time_prefix: str = "ts",  # auto-detect columns starting with this
            tolerance: Union[str, float, int] = "10ms",
            direction: str = "nearest",
            suffix: str = "_dlc",
            # --- NEW labeling controls ---
            add_labels: bool = True,  # add both 'trial' and 'set_label'
            trial_labels: Optional[List[int]] = None,  # per-trial ids; else trial_base..N-1
            trial_base: int = 1,  # default 1..N; set to 0 for 0..N-1
            set_label: Optional[str] = None,  # single label for all trials (e.g., "first")
            set_labels: Optional[List[str]] = None,  # per-trial labels (same length as cam_trials)
    ) -> List[pd.DataFrame]:
        """
        Append DLC angle columns (from compute_dlc3d_angles_by_trial) to each camera ts_* DataFrame.
        Returns a list of per-trial DataFrames. If add_labels=True, each df gets:
          - 'trial'     : from trial_labels or a default sequence
          - 'set_label' : from set_label (single) or set_labels (per-trial)
        Preserves these labels if they already exist.
        """
        import numpy as np
        import pandas as pd

        out: List[pd.DataFrame] = []

        # ---------- helpers ----------
        def _pick_cam_time(df: pd.DataFrame) -> Optional[str]:
            if cam_time_col and cam_time_col in df.columns:
                return cam_time_col
            cands = [c for c in df.columns if str(c).lower().startswith(cam_time_prefix)]
            return cands[0] if cands else None

        def _find_dlc_time_col(df: pd.DataFrame):
            for c in df.columns:
                s = str(c).lower()
                if "cam_time_s_from_ts" in s:  # preferred prepared-seconds column if present
                    return c
                if "ts" in s or "timestamp" in s or s.endswith("_time_s") or s in ("time_s", "t_sec"):
                    return c
            return None

        def _flatten_metrics(df: pd.DataFrame) -> pd.DataFrame:
            cols = [
                ("metric", "mcp_bend_deg", "deg"),
                ("metric", "wrist_bend_deg", "deg"),
                ("metric", "mcp_bend_in_wrist_plane_deg", "deg"),
                ("metric", "wrist_plane_ok", ""),
            ]
            present = [c for c in cols if c in df.columns]
            if not present:
                return pd.DataFrame(index=df.index)
            sub = df[present].copy()
            sub.columns = [
                ("_".join([x for x in c if str(x) != ""]) if isinstance(c, tuple) else str(c)) + suffix
                for c in sub.columns
            ]
            return sub

        def _tol_seconds_local(tol) -> float:
            return pd.to_timedelta(tol).total_seconds() if isinstance(tol, str) else float(tol)

        # ---------- build labels ----------
        n = len(cam_trials)
        if trial_labels is None:
            labels_trial = [trial_base + i for i in range(n)]
        else:
            if len(trial_labels) != n:
                raise ValueError("trial_labels length must match cam_trials length.")
            labels_trial = trial_labels

        if set_labels is not None and len(set_labels) != n:
            raise ValueError("set_labels length must match cam_trials length.")

        def _label_for(i: int) -> tuple[Optional[int], Optional[str]]:
            tlabel = labels_trial[i] if add_labels else None
            slabel = None
            if add_labels:
                slabel = set_labels[i] if set_labels is not None else set_label
            return tlabel, slabel

        # ---------- main loop ----------
        for i, (cam_df, dlc_df) in enumerate(zip(cam_trials, dlc_aug_trials)):
            tlabel, slabel = _label_for(i)

            if cam_df is None or cam_df.empty or dlc_df is None or dlc_df.empty:
                df_out = cam_df.copy() if isinstance(cam_df, pd.DataFrame) else pd.DataFrame()
                if add_labels and not df_out.empty:
                    if tlabel is not None and "trial" not in df_out.columns:
                        df_out["trial"] = tlabel
                    if slabel is not None and "set_label" not in df_out.columns:
                        df_out["set_label"] = slabel
                out.append(df_out)
                continue

            dlc_metrics = _flatten_metrics(dlc_df)

            # Case A: identical lengths → fast index join
            if len(cam_df) == len(dlc_metrics) and len(dlc_metrics) > 0:
                joined = cam_df.reset_index(drop=True).join(dlc_metrics.reset_index(drop=True))
                if add_labels:
                    if tlabel is not None and "trial" not in joined.columns:
                        joined["trial"] = tlabel
                    if slabel is not None and "set_label" not in joined.columns:
                        joined["set_label"] = slabel
                out.append(joined)
                continue

            # Case B: time-based merge_asof if possible
            cam_col = _pick_cam_time(cam_df)
            dlc_tcol = _find_dlc_time_col(dlc_df)

            if cam_col is not None and dlc_tcol is not None and len(dlc_metrics) > 0:
                left = cam_df[[cam_col]].copy()
                right = dlc_df[[dlc_tcol]].copy()

                # prefer precomputed seconds if available
                if "cam_time_s_from_ts" in cam_df.columns:
                    left["_t"] = pd.to_numeric(cam_df["cam_time_s_from_ts"], errors="coerce")
                else:
                    left["_t"] = self._coerce_time_series_numeric_seconds(left[cam_col])

                if dlc_tcol in ("cam_time_s_from_ts", "time_s", "t_sec") or str(dlc_tcol).endswith("_time_s"):
                    right["_t_enc"] = pd.to_numeric(right[dlc_tcol], errors="coerce")
                else:
                    right["_t_enc"] = self._coerce_time_series_numeric_seconds(right[dlc_tcol])

                left = left.dropna(subset=["_t"]).sort_values("_t").reset_index(drop=False)
                right = right.dropna(subset=["_t_enc"]).sort_values("_t_enc").reset_index(drop=False)
                right = right.join(dlc_metrics.reset_index(drop=True))

                m = pd.merge_asof(
                    left, right,
                    left_on="_t", right_on="_t_enc",
                    direction=direction,
                    tolerance=_tol_seconds_local(tolerance),
                    allow_exact_matches=True,
                )

                m = m.set_index("index").reindex(cam_df.index)
                keep_cols = [c for c in m.columns if c.endswith(suffix)]
                joined = pd.concat([cam_df, m[keep_cols]], axis=1)

                if add_labels:
                    if tlabel is not None and "trial" not in joined.columns:
                        joined["trial"] = tlabel
                    if slabel is not None and "set_label" not in joined.columns:
                        joined["set_label"] = slabel

                out.append(joined)
                continue

            # Case C: fallback to frame-index nearest
            cam_tmp = cam_df.copy()
            dlc_tmp = dlc_metrics.copy()
            cam_tmp["_frame"] = np.arange(len(cam_tmp), dtype=float)
            dlc_tmp["_frame"] = np.linspace(0, max(0, len(cam_tmp) - 1), num=len(dlc_tmp))

            cam_small = cam_tmp[["_frame"]].copy().reset_index(drop=False)
            dlc_small = dlc_tmp.copy().reset_index(drop=True)

            m = pd.merge_asof(
                cam_small.sort_values("_frame"),
                dlc_small.sort_values("_frame"),
                on="_frame",
                direction=direction,
                allow_exact_matches=True,
            ).set_index("index").reindex(cam_df.index)

            keep_cols = [c for c in m.columns if c.endswith(suffix)]
            joined = pd.concat([cam_df, m[keep_cols]], axis=1)
            if add_labels:
                if tlabel is not None and "trial" not in joined.columns:
                    joined["trial"] = tlabel
                if slabel is not None and "set_label" not in joined.columns:
                    joined["set_label"] = slabel

            out.append(joined)

        return out

    def _mat_to_df(self, mat_path: Path, prefix: str = "ts") -> pd.DataFrame:
        import numpy as _np
        wanted: Dict[str, np.ndarray] = {}
        pref_l = str(prefix).lower()

        # try classic MAT
        try:
            from scipy.io import loadmat
            loaded = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
            keys = [k for k in loaded.keys() if not k.startswith("__")]
            for k in keys:
                if not k.lower().startswith(pref_l):
                    continue
                arr = _np.asarray(loaded[k])
                arr = _np.squeeze(arr)
                if arr.ndim == 1:
                    wanted[k] = _np.asarray(arr, dtype=float)
                elif arr.ndim == 2 and 1 in arr.shape:
                    wanted[k] = _np.asarray(arr.reshape(-1), dtype=float)
        except Exception:
            pass

        # try HDF5 (v7.3)
        if not wanted:
            try:
                import h5py
                with h5py.File(str(mat_path), "r") as f:
                    def _visit(name, obj):
                        if not isinstance(obj, h5py.Dataset):
                            return
                        key = name.split("/")[-1]
                        if not key.lower().startswith(pref_l):
                            return
                        data = _np.array(obj)
                        data = _np.squeeze(data)
                        if data.ndim == 1:
                            wanted[key] = _np.asarray(data, dtype=float)
                        elif data.ndim == 2 and 1 in data.shape:
                            wanted[key] = _np.asarray(data.reshape(-1), dtype=float)
                    f.visititems(_visit)
            except Exception as e:
                raise RuntimeError(f"Failed to open as HDF5 v7.3: {e}")

        if not wanted:
            return pd.DataFrame()

        df = pd.DataFrame({k: v for k, v in wanted.items()})
        if df.shape[1] == 1:
            only = df.columns[0]
            if only.lower().startswith(pref_l):
                df = df.rename(columns={only: "timestamp"})
        return df

    # ============================================================
    # Heuristic helper (kept for convenience, not used by DLC flow)
    # ============================================================
    @staticmethod
    def _infer_imu_angle_deg(df: pd.DataFrame) -> np.ndarray:
        """
        Best-effort extractor for a single joint angle (deg) from a trial IMU df.
        Preference order:
          1) existing angle columns
          2) roll/rx (deg or rad)
          3) pitch/py (deg or rad)
          4) quaternion -> pitch (deg), ZYX
        """
        cols = {c.lower(): c for c in df.columns}

        for key in ("imu_joint_deg_rx_py", "theta_deg", "angle_deg"):
            if key in cols:
                return pd.to_numeric(df[cols[key]], errors="coerce").to_numpy(float)

        for key in ("rx_deg", "roll_deg", "roll (deg)", "roll (degrees)"):
            if key in cols:
                return pd.to_numeric(df[cols[key]], errors="coerce").to_numpy(float)

        for key in ("rx", "roll", "rx_rad", "roll_rad"):
            if key in cols:
                v = pd.to_numeric(df[cols[key]], errors="coerce").to_numpy(float)
                if np.nanmax(np.abs(v)) <= 3.5:
                    v = np.degrees(v)
                return v

        for key in ("py_deg", "pitch_deg", "pitch (deg)", "pitch (degrees)"):
            if key in cols:
                return pd.to_numeric(df[cols[key]], errors="coerce").to_numpy(float)

        for key in ("py", "pitch", "py_rad", "pitch_rad"):
            if key in cols:
                v = pd.to_numeric(df[cols[key]], errors="coerce").to_numpy(float)
                if np.nanmax(np.abs(v)) <= 3.5:
                    v = np.degrees(v)
                return v

        qnames = {k: cols[k] for k in ("qw", "qx", "qy", "qz") if k in cols}
        if len(qnames) == 4:
            w = pd.to_numeric(df[qnames["qw"]], errors="coerce").to_numpy(float)
            x = pd.to_numeric(df[qnames["qx"]], errors="coerce").to_numpy(float)
            y = pd.to_numeric(df[qnames["qy"]], errors="coerce").to_numpy(float)
            z = pd.to_numeric(df[qnames["qz"]], errors="coerce").to_numpy(float)
            s = 2.0 * (w * y - z * x)
            s = np.clip(s, -1.0, 1.0)
            pitch_rad = np.arcsin(s)
            return np.degrees(pitch_rad)

        return np.full(len(df), np.nan, dtype=float)

    # =========================
    # IMU (DLC-driven) pipeline
    # =========================
    def imu_augment_trials_inplace(
        self,
        trigger_trials,  # unused; kept for API symmetry
        imu_trials: List[pd.DataFrame],
        trial_len_sec: float = 10.0,
        quat_cols: Tuple[str, str] = ("quat1", "quat2"),
        fixed_axis: str = "y",
        moving_axis: str = "y",
        quat_order: str = "wxyz",
    ) -> None:
        dlc = DLC3DBendAngles(pd.DataFrame({"_": []}))
        augmented, _ = dlc.compute_joint_angle_trials(
            imu_trials,
            set_label="unused",
            trial_len_sec=trial_len_sec,
            quat_cols=quat_cols,
            fixed_axis=fixed_axis,
            moving_axis=moving_axis,
            quat_order=quat_order,
        )

        for i, df_src in enumerate(imu_trials):
            if df_src is None or df_src.empty:
                continue

            n = len(df_src)
            df_src["time_s"] = np.linspace(0.0, float(trial_len_sec), num=n, endpoint=False, dtype=float)

            if "imu_joint_deg_rx_py" not in df_src.columns:
                df_src["imu_joint_deg_rx_py"] = np.nan

            df_aug = augmented[i]
            if df_aug is None or df_aug.empty or "imu_joint_deg_rx_py" not in df_aug.columns:
                continue

            if len(df_aug) == n:
                df_src["imu_joint_deg_rx_py"] = df_aug["imu_joint_deg_rx_py"].to_numpy()
                continue

            if "timestamp" in df_src.columns and "timestamp" in df_aug.columns:
                left = df_src[["timestamp"]].copy()
                right = df_aug[["timestamp", "imu_joint_deg_rx_py"]].copy()
                try:
                    lt = pd.to_numeric(left["timestamp"], errors="coerce")
                    rt = pd.to_numeric(right["timestamp"], errors="coerce")
                    if lt.notna().any() and rt.notna().any():
                        left["_t"] = lt; right["_t"] = rt
                        left = left.sort_values("_t")
                        right = right.sort_values("_t")
                        merged = pd.merge_asof(left, right, on="_t", direction="nearest", tolerance=None).sort_index()
                        df_src["imu_joint_deg_rx_py"] = merged["imu_joint_deg_rx_py"].to_numpy()
                        continue
                except Exception:
                    pass
                merged = left.merge(right.astype({"timestamp": str}), on="timestamp", how="left")
                df_src["imu_joint_deg_rx_py"] = merged["imu_joint_deg_rx_py"].to_numpy()
            else:
                print(f"[imu_augment_trials_inplace] Trial {i + 1}: length mismatch "
                      f"({n} vs {len(df_aug)}), no timestamp to align; leaving angle as NaN.")

    def imu_collect_tall(
        self,
        imu_trials: List[pd.DataFrame],
        set_label: str,
        trial_len_sec: float = 10.0,
        quat_cols: Tuple[str, str] = ("quat1", "quat2"),
        fixed_axis: str = "y",
        moving_axis: str = "y",
        quat_order: str = "wxyz",
    ) -> pd.DataFrame:
        dlc = DLC3DBendAngles(pd.DataFrame({"_": []}))
        _, tall = dlc.compute_joint_angle_trials(
            imu_trials,
            set_label=set_label,
            trial_len_sec=trial_len_sec,
            quat_cols=quat_cols,
            fixed_axis=fixed_axis,
            moving_axis=moving_axis,
            quat_order=quat_order,
        )
        return tall

    # ======== ADC→θ builders + ADC↔CAM alignment ========
    def _detect_cam_time_col(self, cam_df: pd.DataFrame, prefix: str = "ts") -> Optional[str]:
        """Pick the first column that looks like a camera timestamp (e.g., 'ts_25183199')."""
        cands = [c for c in cam_df.columns if str(c).lower().startswith(prefix)]
        return cands[0] if cands else None

    def _coerce_time_series_numeric_seconds(self, s: pd.Series) -> pd.Series:
        """
        Robustly coerce timestamps to a numeric 'seconds' axis for merge_asof alignment.
        Tries numeric first; otherwise tries datetime-like and converts to seconds from min.
        """
        sn = pd.to_numeric(s, errors="coerce")
        if sn.notna().any():
            m = sn[sn.notna()]
            if len(m) > 2:
                span = float(m.max() - m.min())
                if span > 1e6:  # likely µs/ns domain
                    med = float(m.abs().median())
                    scale = 1e9 if med > 1e9 else (1e6 if med > 1e6 else 1.0)
                    return sn.astype(float) / scale
            return sn.astype(float)

        try:
            dt = pd.to_datetime(s, errors="coerce", utc=True)
            if dt.notna().any():
                t0 = dt.min()
                return (dt - t0).dt.total_seconds()
        except Exception:
            pass

        return pd.Series(np.nan, index=s.index, dtype=float)

    

    def _theta_trials_from_adc(
        self,
        adc_trials: List[pd.DataFrame],
        trial_len_sec: float = 10.0,
        adc_col: str = "adc_ch3",
        time_col_options: Tuple[str, ...] = ("timestamp", "time", "t_sec"),
        include_endpoint: bool = True,
    ) -> List[pd.DataFrame]:
        """
        Produce per-trial frames with columns:
          ['time_s','timestamp','theta_pred_deg', <adc_col>]
        """
        out = []
        for df in adc_trials:
            if df is None or df.empty:
                out.append(pd.DataFrame()); continue
            tall_one = self.trials_to_tall_df(
                [df],
                set_label="set",
                trial_len_sec=float(trial_len_sec),
                adc_col=adc_col,
                time_col_options=time_col_options,
                include_endpoint=include_endpoint,
            )
            out.append(
                tall_one.drop(columns=["set_label", "trial"], errors="ignore").reset_index(drop=True)
            )
        return out

    def align_adc_to_cam_for_set(
        self,
        adc_trials: List[pd.DataFrame],
        cam_trials: List[pd.DataFrame],
        *,
        trial_len_sec: float = 10.0,
        adc_col: str = "adc_ch3",
        enc_time_col: str = "timestamp",
        cam_time_col: Optional[str] = None,  # auto-detect if None (e.g., 'ts_25183199')
        cam_time_prefix: str = "ts",
        tolerance: str | float = "10ms",
        direction: str = "nearest",
        attach_cols: Tuple[str, ...] = ("theta_pred_deg",),
        keep_time_delta: bool = True,
        drop_unmatched: bool = True,
        dlc_cam_obj: Optional[object] = None,
    ) -> List[pd.DataFrame]:
        """
        Align per-trial angle-converted ADC (θ_pred from self.calib) to camera timestamps.
        """
        # 1) Safety: need calibration first
        if not self.calib:
            raise RuntimeError("Calibration not set. Call fit_and_set_calibration(...) before aligning.")

        # 2) Build θ-from-ADC tables per trial
        theta_trials = self._theta_trials_from_adc(
            adc_trials,
            trial_len_sec=trial_len_sec,
            adc_col=adc_col,
            time_col_options=(enc_time_col, "timestamp", "time", "t_sec"),
            include_endpoint=True,
        )

        merged = []
        use_dlc = (
            dlc_cam_obj is not None
            and hasattr(dlc_cam_obj, "find_matching_indices")
            and hasattr(dlc_cam_obj, "attach_encoder_using_match")
        )

        for i, (cam_df, th_df) in enumerate(zip(cam_trials, theta_trials), start=1):
            if cam_df is None or cam_df.empty or th_df is None or th_df.empty:
                merged.append(pd.DataFrame()); continue

            cam_col = cam_time_col or self._detect_cam_time_col(cam_df, prefix=cam_time_prefix)
            if cam_col is None or cam_col not in cam_df.columns:
                print(f"[align] Trial {i}: camera time column not found (prefix='{cam_time_prefix}').")
                merged.append(pd.DataFrame()); continue

            if enc_time_col not in th_df.columns:
                print(f"[align] Trial {i}: encoder time column '{enc_time_col}' not found in theta trial.")
                merged.append(pd.DataFrame()); continue

            if use_dlc:
                try:
                    dlc_cam_obj.find_matching_indices(
                        encoder_df=th_df.rename(columns={enc_time_col: "timestamp"}),
                        cam_time_col=(cam_col, "", ""),
                        enc_time_col="timestamp",
                        tolerance=tolerance,
                        direction=direction,
                    )
                    mdf = dlc_cam_obj.attach_encoder_using_match(
                        encoder_df=th_df.rename(columns={enc_time_col: "timestamp"}),
                        columns=list(attach_cols) if attach_cols else None,
                        suffix="_adc",
                        keep_time_delta=keep_time_delta,
                        drop_unmatched=drop_unmatched,
                    )
                    merged.append(mdf); continue
                except Exception as e:
                    print(f"[align] Trial {i}: DLC path failed ({e}); falling back to merge_asof).")

            left  = cam_df[[cam_col]].copy()
            right = th_df[[enc_time_col] + [c for c in attach_cols if c in th_df.columns]].copy()

            left["_t"]  = self._coerce_time_series_numeric_seconds(left[cam_col])
            right["_t"] = self._coerce_time_series_numeric_seconds(right[enc_time_col])

            left  = left.loc[left["_t"].notna()].sort_values("_t")
            right = right.loc[right["_t"].notna()].sort_values("_t")

            m = pd.merge_asof(
                left, right,
                on="_t",
                direction=direction,
                tolerance=self._tol_seconds(tolerance),
                allow_exact_matches=True,
            )

            if keep_time_delta:
                try:
                    lraw = pd.to_numeric(cam_df.loc[left.index, cam_col], errors="coerce")
                    rr = right[["_t", enc_time_col]].copy()
                    mm = m.merge(rr, on="_t", how="left", suffixes=("", ""))
                    rraw = pd.to_numeric(mm[enc_time_col], errors="coerce")
                    if lraw.notna().any() and rraw.notna().any():
                        lraw_aligned = lraw.reset_index(drop=True)
                        rraw_aligned = rraw.reset_index(drop=True)
                        m["_delta_raw"] = rraw_aligned - lraw_aligned
                except Exception:
                    pass

            if drop_unmatched and attach_cols:
                first_col = next((c for c in attach_cols if c in m.columns), None)
                if first_col is not None:
                    m = m.loc[m[first_col].notna()].copy()

            m[cam_col] = left[cam_col].values
            merged.append(m.reset_index(drop=True))

        return merged


    def plot_imu_dlc_abs_error_boxplots(
            self,
            *,
            aligned_first_imu: list,
            aligned_second_imu: list,
            dlc_angle_col: str = "metric_wrist_bend_deg_deg_dlc",
            imu_angle_col: str = "imu_joint_deg_rx_py",
            trials_first: list | tuple | None = None,  # e.g., [1,3] (accepts 1- or 0-based). None => auto-pick 2
            trials_second: list | tuple | None = None,
            max_abs_dt_ms: float | None = 100.0,  # None -> no Δt filter
            figsize=(10.5, 4.6),
            whisker: float = 1.5,
            show_counts: bool = True,
            verbose: bool = True,
    ):
        """
        Make boxplots of |IMU - DLC| for two trials in Set 1 and two trials in Set 2,
        using pre-aligned lists (aligned_first_imu, aligned_second_imu).

        Left subplot: 2 boxes for Set 1 (one per chosen trial).
        Right subplot: 2 boxes for Set 2 (one per chosen trial).

        Returns: {"fig","axes","stats","chosen_trials":{"set1":[...],"set2":[...]}}
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        def _nonempty_indices(al):
            return [i for i, df in enumerate(al or []) if df is not None and not df.empty]

        def _resolve_trials(al, trials):
            """Accept 1- or 0-based labels; return valid 0-based indices (max 2)."""
            ne = _nonempty_indices(al)
            if not ne:
                return []
            if trials is None:
                return ne[:2]
            idxs = []
            for t in trials:
                t = int(t)
                i1 = t - 1
                if 0 <= i1 < len(al) and al[i1] is not None and not al[i1].empty:
                    idxs.append(i1);
                    continue
                if 0 <= t < len(al) and al[t] is not None and not al[t].empty:
                    idxs.append(t)
            # keep first two unique
            seen, out = set(), []
            for i in idxs:
                if i not in seen:
                    out.append(i);
                    seen.add(i)
                if len(out) == 2:
                    break
            if not out:
                out = ne[:2]
            return out

        def _abs_err_for_trial(df):
            if df is None or df.empty:
                return np.array([], dtype=float)
            if (dlc_angle_col not in df.columns) or (imu_angle_col not in df.columns):
                return np.array([], dtype=float)
            s_dlc = pd.to_numeric(df[dlc_angle_col], errors="coerce")
            s_imu = pd.to_numeric(df[imu_angle_col], errors="coerce")
            ok = s_dlc.notna() & s_imu.notna()
            if max_abs_dt_ms is not None and "_delta_ms" in df.columns:
                dt_ok = pd.to_numeric(df["_delta_ms"], errors="coerce").abs() <= float(max_abs_dt_ms)
                ok = ok & dt_ok
            v = (s_imu[ok] - s_dlc[ok]).abs().to_numpy()
            return v if v.size else np.array([], dtype=float)

        # choose trials
        idxs1 = _resolve_trials(aligned_first_imu, trials_first)
        idxs2 = _resolve_trials(aligned_second_imu, trials_second)
        if verbose:
            print(f"[imu-dlc boxplots] set1 trials (0-based): {idxs1}, set2 trials (0-based): {idxs2}")

        # compute arrays
        vals1 = [_abs_err_for_trial(aligned_first_imu[i]) for i in idxs1]
        vals2 = [_abs_err_for_trial(aligned_second_imu[i]) for i in idxs2]

        # pad with empties if fewer than 2
        while len(vals1) < 2: vals1.append(np.array([], dtype=float))
        while len(vals2) < 2: vals2.append(np.array([], dtype=float))

        # labels (show as 1-based to user)
        labels1 = [f"Set1 T{(i + 1)}" if i is not None else "Set1 ?" for i in idxs1] + [""] * (2 - len(idxs1))
        labels2 = [f"Set2 T{(i + 1)}" if i is not None else "Set2 ?" for i in idxs2] + [""] * (2 - len(idxs2))

        # plot
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, constrained_layout=True)

        boxprops = dict(linewidth=1.2)
        medianprops = dict(color="black", linewidth=1.8)
        whiskerprops = dict(color="0.4", linewidth=1.2)
        capprops = dict(color="0.4", linewidth=1.2)

        bpl = axes[0].boxplot(
            vals1, labels=labels1, whis=whisker, showfliers=False, patch_artist=True,
            boxprops=boxprops, medianprops=medianprops,
            whiskerprops=whiskerprops, capprops=capprops
        )
        bpr = axes[1].boxplot(
            vals2, labels=labels2, whis=whisker, showfliers=False, patch_artist=True,
            boxprops=boxprops, medianprops=medianprops,
            whiskerprops=whiskerprops, capprops=capprops
        )

        # make all boxes blue with slight alpha
        for bx in (bpl, bpr):
            for patch in bx["boxes"]:
                patch.set_facecolor("tab:blue");
                patch.set_alpha(0.35)
            for med in bx["medians"]:
                med.set_color("black");
                med.set_linewidth(1.8)

        axes[0].set_title("|IMU − DLC| (Set 1)")
        axes[0].set_ylabel("|angle error| (deg)")
        axes[1].set_title("|IMU − DLC| (Set 2)")

        for ax in axes:
            ax.axhline(0, color="k", linewidth=0.8, alpha=0.5)
            ax.grid(axis="y", alpha=0.25)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # counts
        if show_counts:
            def _annot(ax, arrs):
                top = ax.get_ylim()[1] if np.isfinite(ax.get_ylim()[1]) else 1.0
                for i, arr in enumerate(arrs, start=1):
                    n = int(arr.size) if arr is not None else 0
                    ax.text(i, top * 0.95, f"n={n}", ha="center", va="top", fontsize=9)

            _annot(axes[0], vals1);
            _annot(axes[1], vals2)

        def _summ(a):
            if a.size == 0:
                return dict(n=0, mean=np.nan, median=np.nan, p90=np.nan, iqr=np.nan, mad=np.nan)
            q25, med, q75, p90 = np.percentile(a, [25, 50, 75, 90])
            return dict(
                n=int(a.size),
                mean=float(np.mean(a)),
                median=float(med),
                p90=float(p90),
                iqr=float(q75 - q25),
                mad=float(np.median(np.abs(a - med))),
            )

        stats = {
            "set1": {labels1[0]: _summ(vals1[0]), labels1[1]: _summ(vals1[1])},
            "set2": {labels2[0]: _summ(vals2[0]), labels2[1]: _summ(vals2[1])},
            "chosen_trials": {"set1": [i for i in idxs1], "set2": [i for i in idxs2]},
            "params": {
                "dlc_angle_col": dlc_angle_col,
                "imu_angle_col": imu_angle_col,
                "max_abs_dt_ms": max_abs_dt_ms,
                "whisker": whisker,
            },
        }
        if verbose: print("[imu-dlc boxplots] done.")
        return {"fig": fig, "axes": axes, "stats": stats}

    def plot_imu_vs_dlc_twopanel(
            self,
            *,
            aligned_first_imu: list,
            aligned_second_imu: list,
            dlc_angle_col: str = "metric_wrist_bend_deg_deg_dlc",
            imu_angle_col: str = "imu_joint_deg_rx_py",
            trials_first: list | tuple | None = None,  # e.g., [1,3] or [0,2]; None => auto first two
            trials_second: list | tuple | None = None,
            max_abs_dt_ms: float | None = 100.0,
            figsize=(12.0, 4.8),
            linewidth: float = 1.6,
            alpha_dlc: float = 0.9,
            alpha_imu: float = 0.9,
            verbose: bool = True,
    ):
        """
        Plot IMU vs DLC angle as time series, overlaying up to TWO trials per set.
          LEFT:  Set 1 (two chosen trials)
          RIGHT: Set 2 (two chosen trials)
        Uses pre-aligned lists (aligned_first_imu / aligned_second_imu).
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        def _nonempty_indices(al):
            return [i for i, df in enumerate(al or []) if df is not None and not df.empty]

        def _resolve_two(al, trials_like):
            """Return up to two 0-based indices, accepting 1- or 0-based."""
            ne = _nonempty_indices(al)
            if not ne:
                return []
            if trials_like is None:
                return ne[:2]
            idxs = []
            for t in trials_like:
                t = int(t)
                i1 = t - 1
                if 0 <= i1 < len(al) and al[i1] is not None and not al[i1].empty:
                    idxs.append(i1);
                    continue
                if 0 <= t < len(al) and al[t] is not None and not al[t].empty:
                    idxs.append(t)
            # dedupe and cap at 2
            out, seen = [], set()
            for i in idxs:
                if i not in seen:
                    out.append(i);
                    seen.add(i)
                if len(out) == 2:
                    break
            return out if out else ne[:2]

        def _rel_seconds(series):
            s = pd.to_numeric(series, errors="coerce").to_numpy()
            if s.size == 0 or not np.isfinite(s).any():
                return s
            s = s - np.nanmin(s)
            mx = np.nanmax(s)
            if not np.isfinite(mx) or mx == 0:
                return s
            if mx > 1e9: return s / 1e9
            if mx > 1e6: return s / 1e6
            if mx > 1e3: return s / 1e3
            return s

        def _pick_x(df):
            for xname in ("enc_time_s", "time_s", "t_sec"):
                if xname in df.columns:
                    return pd.to_numeric(df[xname], errors="coerce").to_numpy()
            ts_cols = [c for c in df.columns if str(c).startswith("ts_")]
            if ts_cols:
                return _rel_seconds(df[ts_cols[0]])
            return np.arange(len(df), dtype=float)

        def _plot_trial(ax, df, label_prefix, color):
            if df is None or df.empty: return False
            if (dlc_angle_col not in df.columns) or (imu_angle_col not in df.columns): return False

            x = _pick_x(df)
            ok = pd.Series(True, index=df.index)
            if (max_abs_dt_ms is not None) and ("_delta_ms" in df.columns):
                ok = pd.to_numeric(df["_delta_ms"], errors="coerce").abs() <= float(max_abs_dt_ms)
                ok = ok.fillna(False)

            y_dlc = pd.to_numeric(df[dlc_angle_col], errors="coerce")
            y_imu = pd.to_numeric(df[imu_angle_col], errors="coerce")
            good = ok & y_dlc.notna() & y_imu.notna()
            if not good.any(): return False

            xx = np.asarray(x)[good.to_numpy()]
            ax.plot(xx, y_dlc[good].to_numpy(), linestyle=":", linewidth=linewidth,
                    alpha=alpha_dlc, color=color, label=f"{label_prefix} DLC")
            ax.plot(xx, y_imu[good].to_numpy(), linestyle="-", linewidth=linewidth,
                    alpha=alpha_imu, color=color, label=f"{label_prefix} IMU")
            return True

        # choose up to two trials per set
        idxs1 = _resolve_two(aligned_first_imu, trials_first)
        idxs2 = _resolve_two(aligned_second_imu, trials_second)
        if verbose:
            print(f"[imu-vs-dlc 2x] set1 trials (0-based): {idxs1} | set2 trials (0-based): {idxs2}")

        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, constrained_layout=True)

        # colors per trial overlay (two distinct colors per panel)
        colors_L = ["tab:blue", "tab:orange"]
        colors_R = ["tab:green", "tab:red"]

        # LEFT panel (Set 1)
        any_L = False
        for k, i in enumerate(idxs1[:2]):
            lbl = f"T{i + 1}"
            any_L |= _plot_trial(axes[0], aligned_first_imu[i], lbl, colors_L[k % len(colors_L)])
        axes[0].set_title("Set 1 — IMU vs DLC (two trials)" + ("" if any_L else " — no data"))
        axes[0].set_xlabel("time (s)")
        axes[0].set_ylabel("angle (deg)")
        axes[0].grid(alpha=0.25)
        axes[0].spines["top"].set_visible(False);
        axes[0].spines["right"].set_visible(False)
        if any_L: axes[0].legend(frameon=False, ncol=2, fontsize=9)

        # RIGHT panel (Set 2)
        any_R = False
        for k, i in enumerate(idxs2[:2]):
            lbl = f"T{i + 1}"
            any_R |= _plot_trial(axes[1], aligned_second_imu[i], lbl, colors_R[k % len(colors_R)])
        axes[1].set_title("Set 2 — IMU vs DLC (two trials)" + ("" if any_R else " — no data"))
        axes[1].set_xlabel("time (s)")
        axes[1].grid(alpha=0.25)
        axes[1].spines["top"].set_visible(False);
        axes[1].spines["right"].set_visible(False)
        if any_R: axes[1].legend(frameon=False, ncol=2, fontsize=9)

        return {"fig": fig, "axes": axes, "trials_used": {"set1": idxs1[:2], "set2": idxs2[:2]}}

    def align_adc_to_cam_both_sets(
        self,
        adc_trials_first: List[pd.DataFrame],
        cam_trials_first: List[pd.DataFrame],
        adc_trials_second: List[pd.DataFrame],
        cam_trials_second: List[pd.DataFrame],
        **kwargs,
    ) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Convenience wrapper: runs align_adc_to_cam_for_set for first and second sets.
        kwargs are passed through to align_adc_to_cam_for_set.
        """
        first  = self.align_adc_to_cam_for_set(adc_trials_first,  cam_trials_first,  **kwargs)
        second = self.align_adc_to_cam_for_set(adc_trials_second, cam_trials_second, **kwargs)
        return first, second





class DLC3DBendAngles:
    """
    DLC 3D (MultiIndex header) bend-angle calculator.

    angle_type:
      - 'mcp'   : angle(hand→MCP, MCP→PIP)
      - 'wrist' : angle(forearm→hand, hand→MCP)
    """

    def __init__(self, data, bodyparts: Optional[Dict[str, str]] = None):
        """
        Initialize from:
          - one or two DLC 3D CSV file paths
          - OR an existing pandas DataFrame
        """
        self.csv_path = data
        self._match_map: Optional[pd.DataFrame] = None
        self.df = None
        self.imu_df = None
        self.enc_df = None

        if isinstance(data, pd.DataFrame):
            self.df = data
        elif isinstance(data, str):
            df = pd.read_csv(data, header=[0, 1, 2])
            if not isinstance(df.columns, pd.MultiIndex):
                raise ValueError("Expected DLC 3D CSV with a 3-row MultiIndex header.")
            self.df = df
        elif isinstance(data, (list, tuple)) and len(data) == 2:
            dfs = []
            for path in data:
                df = pd.read_csv(path, header=[0, 1, 2])
                if not isinstance(df.columns, pd.MultiIndex):
                    raise ValueError(f"{path} does not have a 3-row MultiIndex header.")
                dfs.append(df)
            if len(dfs[0]) != len(dfs[1]):
                raise ValueError(f"Row count mismatch: {len(dfs[0])} vs {len(dfs[1])}")
            self.df = pd.concat(dfs, axis=1)
        else:
            raise ValueError("Input must be a DataFrame, a CSV path, or a list/tuple of two CSV paths.")

        self.bp = {
            "forearm": "forearm",
            "hand": "hand",
            "MCP": "MCP",
            "PIP": "PIP",
        }
        if bodyparts:
            self.bp.update(bodyparts)

    # ---------- IMU augmentation & tall collector ----------

    def imu_augment_trials_inplace(
        self,
        trigger_trials: List[pd.DataFrame],  # accepted for API symmetry; unused
        imu_trials: List[pd.DataFrame],
        trial_len_sec: float = 10.0,
        time_col_options: Tuple[str, ...] = ("timestamp", "time", "t_sec"),
    ) -> None:
        """
        In-place augments each IMU trial DataFrame with:
          - time_s: synthetic time from 0..trial_len_sec, length=len(rows)
          - imu_joint_deg_rx_py: best-available IMU joint angle in degrees
        Preserves any existing timestamp column (not modified/overwritten).
        """

        def _deg_from_quat(df: pd.DataFrame) -> Optional[np.ndarray]:
            qcols = {"qw", "qx", "qy", "qz"} & set(df.columns)
            if len(qcols) == 4:
                qw = pd.to_numeric(df["qw"], errors="coerce").to_numpy(float)
                qx = pd.to_numeric(df["qx"], errors="coerce").to_numpy(float)
                qy = pd.to_numeric(df["qy"], errors="coerce").to_numpy(float)
                qz = pd.to_numeric(df["qz"], errors="coerce").to_numpy(float)
                # roll = atan2(2(w x + y z), 1 - 2(x^2 + y^2))
                num = 2.0 * (qw * qx + qy * qz)
                den = 1.0 - 2.0 * (qx * qx + qy * qy)
                roll_rad = np.arctan2(num, den)
                return np.degrees(roll_rad)
            return None

        def _compute_imu_deg(df: pd.DataFrame) -> np.ndarray:
            for name in ["imu_joint_deg_rx_py", "imu_joint_deg", "imu_deg", "theta_deg", "angle_deg"]:
                if name in df.columns:
                    return pd.to_numeric(df[name], errors="coerce").to_numpy(float)
            for name in ["rx_deg", "roll_deg", "imu_rx_deg"]:
                if name in df.columns:
                    return pd.to_numeric(df[name], errors="coerce").to_numpy(float)
            for name in ["rx", "roll", "imu_rx"]:
                if name in df.columns:
                    v = pd.to_numeric(df[name], errors="coerce").to_numpy(float)
                    return np.degrees(v)
            v = _deg_from_quat(df)
            if v is not None:
                return v
            return np.full(len(df), np.nan, dtype=float)

        for i, df in enumerate(imu_trials):
            if df is None or df.empty:
                continue

            n = len(df)
            df["time_s"] = np.linspace(0.0, float(trial_len_sec), num=n, endpoint=False, dtype=float)

            # ensure timestamp exists (pass-through if present, else create)
            if not any(c in df.columns for c in time_col_options):
                df["timestamp"] = np.nan

            df["imu_joint_deg_rx_py"] = _compute_imu_deg(df)

    def imu_collect_tall(
        self,
        imu_trials: List[pd.DataFrame],
        set_label: str,
        time_col_options: Tuple[str, ...] = ("timestamp", "time", "t_sec"),
    ) -> pd.DataFrame:
        """
        Collects augmented IMU trials into a single tall DataFrame with:
          ['set_label','trial','time_s','timestamp','imu_joint_deg_rx_py']
        Assumes `imu_augment_trials_inplace` has run.
        """
        parts: List[pd.DataFrame] = []

        def _pick_timestamp_col(d: pd.DataFrame) -> Optional[str]:
            for c in time_col_options:
                if c in d.columns:
                    return c
            return None

        for trial_idx, df in enumerate(imu_trials, start=1):
            if df is None or df.empty:
                continue
            dfx = df.copy()
            if "time_s" not in dfx.columns:
                n = len(dfx)
                dfx["time_s"] = np.linspace(0.0, 10.0, num=n, endpoint=False, dtype=float)
            if "imu_joint_deg_rx_py" not in dfx.columns:
                # fallback: try to compute quickly
                dfx["imu_joint_deg_rx_py"] = np.nan
            tcol = _pick_timestamp_col(dfx)
            ts = dfx[tcol] if tcol else pd.Series([np.nan] * len(dfx))

            parts.append(pd.DataFrame({
                "set_label": set_label,
                "trial": trial_idx,
                "time_s": pd.to_numeric(dfx["time_s"], errors="coerce"),
                "timestamp": ts,
                "imu_joint_deg_rx_py": pd.to_numeric(dfx["imu_joint_deg_rx_py"], errors="coerce"),
            }))

        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["set_label", "trial", "time_s", "timestamp", "imu_joint_deg_rx_py"]
        )

    def add_dataframe(self, other: pd.DataFrame, prefix: str | None = None) -> None:
        """
        Append columns from `other` into `self.df` by row index.
        If lengths differ, `other` is reindexed to self.df and missing rows become NaN.
        Optionally add a `prefix` to the added column names to avoid collisions.
        """
        if self.df is None or not isinstance(self.df, pd.DataFrame):
            raise ValueError("self.df must be a pandas DataFrame before adding columns.")
        if other is None or other.empty:
            return

        # align by index length
        other2 = other.copy().reindex(range(len(self.df))).reset_index(drop=True)

        # optionally prefix new column names
        if prefix:
            other2 = other2.add_prefix(f"{prefix}:")

        # avoid accidental name clashes: rename duplicates
        dup = set(self.df.columns) & set(other2.columns)
        if dup:
            other2 = other2.rename(columns={c: f"{c}_ts" for c in dup})

        self.df = pd.concat([self.df.reset_index(drop=True), other2], axis=1)

    # ---------------- helpers ----------------
    @staticmethod
    def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Flatten MultiIndex columns to single strings (non-destructive copy)."""
        if isinstance(df.columns, pd.MultiIndex):
            out = df.copy()
            out.columns = [
                "_".join([str(x) for x in tup if str(x) != ""]).strip()
                for tup in out.columns.to_list()
            ]
            return out
        return df

    # ---------- Column + point helpers ----------
    def get_xyz(self, bodypart_key: str) -> Tuple[Tuple[str, str, str], Tuple[str, str, str], Tuple[str, str, str]]:
        """Return MultiIndex columns for (x, y, z) of the given bodypart."""
        label = self.bp[bodypart_key]
        cols = self.df.columns
        cand = [c for c in cols if len(c) >= 3 and str(c[1]).strip().lower() == label.strip().lower()]
        if not cand:
            raise KeyError(f"Bodypart '{label}' not found.")
        coord_map = {str(c[2]).strip().lower(): c for c in cand}
        try:
            return (coord_map["x"], coord_map["y"], coord_map["z"])
        except KeyError:
            raise KeyError(f"Missing x/y/z for bodypart '{label}'.")

    def get_xy(self, bodypart_key: str) -> Tuple[Tuple[str, str, str], Tuple[str, str, str]]:
        """Return MultiIndex columns for (x, y) of the given bodypart."""
        label = self.bp[bodypart_key]
        cols = self.df.columns
        cand = [c for c in cols if len(c) >= 3 and str(c[1]).strip().lower() == label.strip().lower()]
        if not cand:
            raise KeyError(f"Bodypart '{label}' not found.")
        coord_map = {str(c[2]).strip().lower(): c for c in cand}
        try:
            return (coord_map["x"], coord_map["y"])
        except KeyError:
            raise KeyError(f"Missing x and/or y for bodypart '{label}'.")

    def get_points(
        self,
        bodypart_key: str,
        allow_2d: bool = True,
        pad_2d_with_z0: bool = True
    ) -> np.ndarray:
        """
        Return Nx3 (preferred) or Nx2 coordinates for the bodypart.
        """
        try:
            xyz_cols = self.get_xyz(bodypart_key)
            return self.df[list(xyz_cols)].to_numpy(dtype=float)
        except KeyError:
            if not allow_2d:
                raise
            x_col, y_col = self.get_xy(bodypart_key)
            xy = self.df[[x_col, y_col]].to_numpy(dtype=float)
            if pad_2d_with_z0:
                z = np.zeros((xy.shape[0], 1), dtype=float)
                return np.hstack([xy, z])
            else:
                return xy

    # ---------- Vector + angle math ----------
    @staticmethod
    def vector(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Row-wise vector from A to B."""
        return B - A

    @staticmethod
    def angle_from_vectors(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
        """Row-wise angle (deg) via dot product."""
        n1 = np.linalg.norm(v1, axis=1)
        n2 = np.linalg.norm(v2, axis=1)
        denom = n1 * n2
        with np.errstate(invalid="ignore", divide="ignore"):
            cosang = np.sum(v1 * v2, axis=1) / denom
        cosang = np.clip(cosang, -1.0, 1.0)
        ang = np.degrees(np.arccos(cosang))
        ang[~np.isfinite(ang)] = np.nan
        return ang

    @staticmethod
    def _rowwise_unit(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
        """Normalize rows; rows with ||x|| < eps become 0."""
        n = np.linalg.norm(x, axis=1, keepdims=True)
        out = np.zeros_like(x)
        good = (n[:, 0] > eps)
        out[good] = x[good] / n[good]
        return out

    @staticmethod
    def _project_onto_plane(vec: np.ndarray, n_hat: np.ndarray) -> np.ndarray:
        """Project rows of vec onto plane with unit normal n_hat (row-wise)."""
        dot = np.sum(vec * n_hat, axis=1, keepdims=True)
        return vec - dot * n_hat

    def angle_from_vectors_in_plane(
        self,
        v1: np.ndarray,
        v2: np.ndarray,
        plane_v1: np.ndarray,
        plane_v2: np.ndarray,
        signed: bool = False,
        eps: float = 1e-12,
    ):
        """
        Project v1 and v2 onto plane spanned by plane_v1 and plane_v2, then angle between them.
        Returns: (ang_deg, p1, p2, valid_plane_mask)
        """
        v1 = np.asarray(v1, float)
        v2 = np.asarray(v2, float)
        p1a = np.asarray(plane_v1, float)
        p2a = np.asarray(plane_v2, float)

        n = np.cross(p1a, p2a)
        n_norm = np.linalg.norm(n, axis=1)
        valid_plane = n_norm > eps
        n_hat = np.zeros_like(n)
        n_hat[valid_plane] = (n[valid_plane] / n_norm[valid_plane, None])

        p1 = self._project_onto_plane(v1, n_hat)
        p2 = self._project_onto_plane(v2, n_hat)

        p1n = np.linalg.norm(p1, axis=1)
        p2n = np.linalg.norm(p2, axis=1)
        good = valid_plane & (p1n > eps) & (p2n > eps)

        ang = np.full(v1.shape[0], np.nan, dtype=float)
        if not np.any(good):
            return ang, p1, p2, valid_plane

        if signed:
            p1u = np.zeros_like(p1); p2u = np.zeros_like(p2)
            p1u[good] = p1[good] / p1n[good, None]
            p2u[good] = p2[good] / p2n[good, None]
            cross_p = np.cross(p1u[good], p2u[good])
            sin_th = np.sum(cross_p * n_hat[good], axis=1)
            cos_th = np.sum(p1u[good] * p2u[good], axis=1)
            ang[good] = np.degrees(np.arctan2(sin_th, cos_th))
        else:
            cosang = np.sum(p1[good] * p2[good], axis=1) / (p1n[good] * p2n[good])
            cosang = np.clip(cosang, -1.0, 1.0)
            ang[good] = np.degrees(np.arccos(cosang))

        return ang, p1, p2, valid_plane

    def compute_vectors(self, angle_type: str = "mcp") -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (v1, v2) for the requested angle.
          - 'mcp'   : v1 = hand→MCP, v2 = MCP→PIP
          - 'wrist' : v1 = forearm→hand, v2 = hand→MCP
        """
        angle_type = angle_type.lower()
        hand = self.get_points("hand")
        mcp = self.get_points("MCP")

        if angle_type == "mcp":
            pip = self.get_points("PIP")
            v1 = self.vector(hand, mcp)
            v2 = self.vector(mcp, pip)
        elif angle_type == "wrist":
            forearm = self.get_points("forearm")
            v1 = self.vector(forearm, hand)
            v2 = self.vector(hand, mcp)
        else:
            raise ValueError("angle_type must be one of {'mcp','wrist'}")
        return v1, v2

    def compute_bend_angle(self, angle_type: str = "mcp") -> np.ndarray:
        """Compute bend angle per frame for the requested angle_type."""
        v1, v2 = self.compute_vectors(angle_type)
        return self.angle_from_vectors(v1, v2)



    # ---------------- column resolution & typing helpers ----------------
    @staticmethod
    def _resolve_col_key(df: pd.DataFrame, col_spec):
        """
        Resolve a column given:
          - exact name present in df.columns
          - a tuple (true MultiIndex key)
          - a substring (returns first match)
        """
        if col_spec in df.columns:
            return col_spec
        candidates = [c for c in df.columns if str(col_spec) in str(c)]
        if len(candidates) == 0:
            raise KeyError(f"Column matching '{col_spec}' not found.")
        return candidates[0]

    @staticmethod
    def _ensure_float_series(s: pd.Series) -> pd.Series:
        """Coerce to float64 for numeric asof matching."""
        if not np.issubdtype(s.dtype, np.number):
            s = pd.to_numeric(s, errors="coerce")
        return s.astype("float64")

    # ---------------- Robust MAT loader (patched) ----------------
    def load_mat_as_df(self, mat_path: str, prefix: str | None = None) -> pd.DataFrame:
        from scipy.io import loadmat

        def _flatten_struct(name, obj):
            cols = {}
            if hasattr(obj, "dtype") and obj.dtype.names:
                for f in obj.dtype.names:
                    v = obj[f]
                    col_name = f"{name}.{f}"
                    if isinstance(v, np.ndarray):
                        if v.ndim == 0:
                            cols[col_name] = pd.Series([v.item()])
                        elif v.ndim == 1:
                            cols[col_name] = pd.Series(v.reshape(-1))
                        else:
                            if v.shape[0] == 0:
                                cols[col_name] = pd.Series([], dtype=float)
                            else:
                                flat = v.reshape(v.shape[0], -1)
                                for j in range(flat.shape[1]):
                                    cols[f"{col_name}_{j}"] = pd.Series(flat[:, j])
                    else:
                        cols[col_name] = pd.Series([v])
            return pd.DataFrame(cols) if cols else pd.DataFrame()

        mat = loadmat(mat_path, squeeze_me=True, struct_as_record=False)
        keys = [k for k in mat.keys() if not k.startswith("__")]
        if prefix:
            keys = [k for k in keys if str(k).startswith(prefix)]

        dfs = []
        for k in keys:
            val = mat[k]
            if hasattr(val, "dtype") and getattr(val.dtype, "names", None):
                df_k = _flatten_struct(k, val)
                if not df_k.empty:
                    dfs.append(df_k)
                continue
            if isinstance(val, np.ndarray):
                if val.ndim == 0:
                    df_k = pd.DataFrame({k: [val.item()]})
                elif val.ndim == 1:
                    df_k = pd.DataFrame({k: val.reshape(-1)})
                elif val.ndim == 2:
                    df_k = pd.DataFrame(val, columns=[f"{k}_{i}" for i in range(val.shape[1])])
                else:
                    flat = val.reshape(val.shape[0], -1)
                    df_k = pd.DataFrame(flat, columns=[f"{k}_{i}" for i in range(flat.shape[1])])
                dfs.append(df_k)
                continue
            dfs.append(pd.DataFrame({k: [val]}))

        if dfs:
            try:
                return pd.concat(dfs, axis=1)
            except Exception:
                return pd.concat(dfs, axis=1, join="outer").reset_index(drop=True)

        raise ValueError(f"No variables found in {mat_path} matching prefix='{prefix}'")

    # ======================
    # 2) Quaternion → Euler
    # ======================
    def imu_quat_to_euler(
        self,
        imu_cols=('euler1', 'euler2'),
        quat_order='wxyz',
        sequence='zyx',
        degrees=True,
        out_prefix=('imu1', 'imu2'),
    ):
        """
        Convert quaternion columns to Euler angles (roll, pitch, yaw).
        Stores results in self.imu_df with prefixes.
        """
        def parse_tuple_string(s):
            if isinstance(s, (list, tuple, np.ndarray)):
                arr = np.array(s, dtype=float).reshape(-1)
                if arr.size == 4:
                    return arr
                out = np.full(4, np.nan, dtype=float)
                n = min(4, arr.size)
                out[:n] = arr[:n]
                return out
            if s is None:
                return np.full(4, np.nan, dtype=float)
            s = str(s).strip()
            if s == "" or s.lower() == "none":
                return np.full(4, np.nan, dtype=float)
            s = s.strip("[](){}")
            parts = [p for p in s.replace(",", " ").split() if p]
            vals = []
            for p in parts[:4]:
                try:
                    vals.append(float(p))
                except Exception:
                    vals.append(np.nan)
            if len(vals) < 4:
                vals += [np.nan] * (4 - len(vals))
            return np.array(vals, dtype=float)

        def normalize(q):
            q = np.asarray(q, dtype=float)
            n = np.linalg.norm(q, axis=-1, keepdims=True)
            with np.errstate(invalid='ignore', divide='ignore'):
                qn = q / n
            return qn

        def quat_to_euler(q, order='wxyz', seq='zyx', deg=True):
            if order == 'wxyz':
                w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
            elif order == 'xyzw':
                x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
            else:
                raise ValueError("quat_order must be 'wxyz' or 'xyzw'.")

            ww, xx, yy, zz = w * w, x * x, y * y, z * z
            R00 = 1 - 2 * (yy + zz)
            R10 = 2 * (x * y + z * w)
            R20 = 2 * (x * z - y * w)
            R21 = 2 * (y * z + x * w)
            R22 = 1 - 2 * (xx + yy)

            if seq == 'zyx':
                yaw = np.arctan2(R10, R00)
                pitch = np.arcsin(-R20)
                roll = np.arctan2(R21, R22)
            else:
                raise ValueError("Unsupported sequence; use 'zyx'.")

            if deg:
                roll, pitch, yaw = np.degrees(roll), np.degrees(pitch), np.degrees(yaw)
            return roll, pitch, yaw

        for col, prefix in zip(imu_cols, out_prefix):
            if col not in self.imu_df.columns:
                raise KeyError(f"Missing IMU quaternion column: {col}")
            q_raw = self.imu_df[col].apply(parse_tuple_string).to_list()
            q = normalize(np.array(q_raw))
            r, p, y = quat_to_euler(q, order=quat_order, seq=sequence, deg=degrees)
            self.imu_df[f'{prefix}_roll'] = r
            self.imu_df[f'{prefix}_pitch'] = p
            self.imu_df[f'{prefix}_yaw'] = y

    # ======================
    # 3) Euler → unit vector
    # ======================
    def euler_to_unit_vec(
        self,
        prefix='imu1',
        sequence='zyx',
        axis='z',
        degrees=True,
        out_col=None,
    ):
        r = self.imu_df[f'{prefix}_roll'].to_numpy()
        p = self.imu_df[f'{prefix}_pitch'].to_numpy()
        y = self.imu_df[f'{prefix}_yaw'].to_numpy()

        if degrees:
            r, p, y = np.deg2rad(r), np.deg2rad(p), np.deg2rad(y)

        def Rx(a): return np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        def Ry(a): return np.array([[np.cos(a), 0, np.sin(a)], [0, 1, 0], [-np.sin(a), 0, np.cos(a)]])
        def Rz(a): return np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])

        base = {'x': [1, 0, 0], 'y': [0, 1, 0], 'z': [0, 0, 1]}[axis]
        vecs = []
        for ri, pi, yi in zip(r, p, y):
            R = Rz(yi) @ Ry(pi) @ Rx(ri) if sequence == 'zyx' else np.eye(3)
            vecs.append(tuple((R @ base).tolist()))

        if out_col is None:
            out_col = f'{prefix}_{axis}vec'
        self.imu_df[out_col] = vecs

    # ======================
    # 4) Vector → angle
    # ======================
    def angle_between_vectors(
        self,
        vec_col_a,
        vec_col_b,
        out_col='imu_vec_angle_deg',
        degrees=True
    ):
        a = np.stack(self.imu_df[vec_col_a].apply(lambda v: np.array(v, float)).to_numpy())
        b = np.stack(self.imu_df[vec_col_b].apply(lambda v: np.array(v, float)).to_numpy())

        na = np.linalg.norm(a, axis=1, keepdims=True)
        nb = np.linalg.norm(b, axis=1, keepdims=True)
        mask = (na[:, 0] > 0) & (nb[:, 0] > 0)

        ang = np.full(len(a), np.nan, float)
        if np.any(mask):
            aa = a[mask] / na[mask]
            bb = b[mask] / nb[mask]
            dot = np.clip(np.sum(aa * bb, axis=1), -1.0, 1.0)
            val = np.arccos(dot)
            if degrees:
                val = np.degrees(val)
            ang[mask] = val

        self.imu_df[out_col] = ang

    # ======================
    # 5) Add angle column
    # ======================
    def add_imu_angle_column(
        self,
        values,
        name=('metric', 'imu_joint_deg', 'deg'),
        inplace=True
    ):
        target = self.imu_df if inplace else self.imu_df.copy()
        colname = name
        if isinstance(name, tuple) and not isinstance(target.columns, pd.MultiIndex):
            colname = "_".join([str(x) for x in name])
        target[colname] = values
        if inplace:
            self.imu_df = target
            return self.imu_df
        return target

    # ---------------- Timestamp parsing helpers for matching ----------------
    @staticmethod
    def _series_time_of_day_to_timedelta(s: pd.Series) -> pd.Series:
        """
        Convert various time formats to Timedelta since midnight.
        Special handling for HHMMSSffffff strings/ints from strftime("%H%M%S%f").
        """
        s = pd.Series(s, copy=False)

        def parse_hhmmssfff(val):
            if pd.isna(val):
                return pd.NaT
            val_str = str(int(val)).zfill(12)
            try:
                t = datetime.strptime(val_str, "%H%M%S%f").time()
                return (pd.to_timedelta(t.hour, unit="h") +
                        pd.to_timedelta(t.minute, unit="m") +
                        pd.to_timedelta(t.second, unit="s") +
                        pd.to_timedelta(t.microsecond, unit="us"))
            except Exception:
                return pd.NaT

        if np.issubdtype(s.dtype, np.number) or s.dtype == object:
            if s.dropna().astype(str).str.fullmatch(r"\d{9,12}").all():
                return s.apply(parse_hhmmssfff)

        try:
            return pd.to_timedelta(s, errors="coerce")
        except Exception:
            return pd.Series(pd.NaT, index=s.index)

    @staticmethod
    def _coerce_tolerance_to_timedelta(tolerance) -> pd.Timedelta:
        """
        Tolerance: int→µs, float→s, str→pandas offset (e.g., '10ms','500us','1s').
        """
        if isinstance(tolerance, (np.integer, int)):
            return pd.to_timedelta(int(tolerance), unit="us")
        if isinstance(tolerance, (np.floating, float)):
            return pd.to_timedelta(float(tolerance), unit="s")
        return pd.to_timedelta(tolerance)

    # ---------------- Core matching API ----------------
    def find_matching_indices(
        self,
        encoder_df: pd.DataFrame,
        cam_time_col,
        enc_time_col,
        tolerance,
        direction: str = "nearest",
    ) -> pd.DataFrame:
        """
        Build mapping of camera rows to encoder rows where |t_cam - t_enc| <= tolerance.
        Stores time_delta in milliseconds (float).
        """
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError("self.df must be a pandas DataFrame.")

        cam_col = self._resolve_col_key(self.df, cam_time_col)
        enc_col = self._resolve_col_key(encoder_df, enc_time_col)

        cam_td = self._series_time_of_day_to_timedelta(self.df[cam_col]).rename("t_cam_td")
        enc_td = self._series_time_of_day_to_timedelta(encoder_df[enc_col]).rename("t_enc_td")

        cam_small = pd.DataFrame({"t_cam_td": cam_td, "cam_index": self.df.index})
        enc_small = pd.DataFrame({"t_enc_td": enc_td, "enc_index": encoder_df.index})

        cam_small = cam_small.dropna(subset=["t_cam_td"]).sort_values("t_cam_td")
        enc_small = enc_small.dropna(subset=["t_enc_td"]).sort_values("t_enc_td")

        if cam_small.empty or enc_small.empty:
            self._match_map = pd.DataFrame(columns=["cam_index", "enc_index", "time_delta"])
            return self._match_map

        cam_small["_t_cam_ns"] = cam_small["t_cam_td"].view("i8")
        enc_small["_t_enc_ns"] = enc_small["t_enc_td"].view("i8")

        merged = pd.merge_asof(
            cam_small,
            enc_small,
            left_on="_t_cam_ns",
            right_on="_t_enc_ns",
            direction=direction,
            allow_exact_matches=True,
        )

        merged["time_delta_td"] = pd.to_timedelta(merged["_t_enc_ns"] - merged["_t_cam_ns"], unit="ns")
        merged["time_delta_ms"] = merged["time_delta_td"].dt.total_seconds() * 1000.0

        tol_td = self._coerce_tolerance_to_timedelta(tolerance)
        keep = merged["time_delta_td"].abs() <= tol_td

        out = (merged.loc[keep, ["cam_index", "enc_index", "time_delta_ms"]]
               .rename(columns={"time_delta_ms": "time_delta"})
               .drop_duplicates(subset=["cam_index"], keep="first")
               .reset_index(drop=True))

        self._match_map = out
        return out

    # --------- Row cleaner for IMU quaternion/euler strings ---------
    @staticmethod
    def drop_rows_with_none_in_euler(df, euler_cols=("euler1", "euler2")):
        df = df.copy()
        for c in euler_cols:
            if c not in df.columns:
                raise KeyError(f"Missing expected IMU column: {c}")
            df[c] = df[c].replace(["None", "none", "NULL", "null", ""], np.nan)
        mask_valid = df[list(euler_cols)].notna().all(axis=1)
        return df.loc[mask_valid].copy()

    # --------- Trial-wise quaternion pipeline ---------
    def compute_joint_angle_trials(
        self,
        imu_trials: list[pd.DataFrame],
        set_label: str,
        trial_len_sec: float = 10.0,
        quat_cols: tuple[str, str] = ("quat1", "quat2"),
        fixed_axis: str = "y",
        moving_axis: str = "y",
        quat_order: str = "wxyz",
    ):
        augmented: list[pd.DataFrame] = []
        parts: list[pd.DataFrame] = []

        for trial_idx, df in enumerate(imu_trials, start=1):
            if df is None or df.empty:
                augmented.append(pd.DataFrame());
                continue

            d = self.drop_rows_with_none_in_euler(df, euler_cols=quat_cols).copy()

            self.imu_df = d
            self.imu_quat_to_euler(
                imu_cols=quat_cols,
                quat_order=quat_order,
                sequence="zyx",
                degrees=True,
                out_prefix=("imu1", "imu2")
            )

            self.euler_to_unit_vec(prefix="imu1", axis=fixed_axis, out_col=f"imu1_{fixed_axis}vec")
            self.euler_to_unit_vec(prefix="imu2", axis=moving_axis, out_col=f"imu2_{moving_axis}vec")

            self.angle_between_vectors(
                vec_col_a=f"imu1_{fixed_axis}vec",
                vec_col_b=f"imu2_{moving_axis}vec",
                out_col="imu_joint_deg_rx_py",
                degrees=True
            )

            n = len(self.imu_df)
            self.imu_df["time_s"] = np.linspace(0.0, float(trial_len_sec), num=n, endpoint=False, dtype=float)
            if "timestamp" not in self.imu_df.columns:
                self.imu_df["timestamp"] = np.nan

            augmented.append(self.imu_df.copy())
            ts = self.imu_df["timestamp"]
            parts.append(pd.DataFrame({
                "set_label": set_label,
                "trial": trial_idx,
                "time_s": pd.to_numeric(self.imu_df["time_s"], errors="coerce"),
                "timestamp": ts,
                "imu_joint_deg_rx_py": pd.to_numeric(self.imu_df["imu_joint_deg_rx_py"], errors="coerce"),
            }))

        tall = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=["set_label", "trial", "time_s", "timestamp", "imu_joint_deg_rx_py"]
        )
        return augmented, tall

    # --------- Encoder attach utilities ---------
    def attach_encoder_using_match(
        self,
        encoder_df: pd.DataFrame,
        columns: Optional[List] = None,
        suffix: str = "_renc",
        keep_time_delta: bool = True,
        drop_unmatched: bool = True,
    ) -> pd.DataFrame:
        if self._match_map is None or self._match_map.empty:
            raise RuntimeError("No matches stored. Run find_matching_indices(...) first.")

        m = self._match_map
        cam_to_enc = dict(zip(m["cam_index"].tolist(), m["enc_index"].tolist()))

        if drop_unmatched:
            keep_idx = m["cam_index"].unique()
            self.df = self.df.loc[keep_idx]
        self.df = self.df.sort_index()

        if columns is None:
            cols_to_take = encoder_df.columns
        else:
            cols_to_take = [self._resolve_col_key(encoder_df, c) for c in columns]

        mapped_enc_idx = self.df.index.to_series().map(cam_to_enc)
        enc_selected = encoder_df.loc[mapped_enc_idx.values, cols_to_take].copy()
        enc_selected.index = self.df.index

        enc_selected = self._flatten_columns(enc_selected)
        enc_selected.columns = [f"{c}{suffix}" for c in enc_selected.columns]

        if keep_time_delta:
            td = m.set_index("cam_index")["time_delta"]
            enc_selected[f"time_delta{suffix}"] = self.df.index.to_series().map(td).values

        self.df = pd.concat([self.df, enc_selected], axis=1)
        return self.df

    def match_encoder_to_imu(
        self,
        enc_time_col="timestamp",
        imu_time_col="timestamp",
        tolerance="200ms",
        direction="nearest",
        columns=None,
        suffix="_imu",
        keep_time_delta=True,
        drop_unmatched=True,
    ):
        if getattr(self, "enc_df", None) is None or getattr(self, "imu_df", None) is None:
            raise RuntimeError("enc_df and imu_df must be loaded before calling match_encoder_to_imu().")

        _prev_df = getattr(self, "df", None)
        self.df = self.enc_df

        self.find_matching_indices(
            encoder_df=self.imu_df,
            cam_time_col=enc_time_col,
            enc_time_col=imu_time_col,
            tolerance=tolerance,
            direction=direction,
        )

        self.attach_encoder_using_match(
            encoder_df=self.imu_df,
            columns=columns,
            suffix=suffix,
            keep_time_delta=keep_time_delta,
            drop_unmatched=drop_unmatched,
        )

        self.enc_df = self.df
        self.df = _prev_df
        return self.enc_df

    def align_and_attach_encoder(
        self,
        encoder_df: pd.DataFrame,
        cam_time_col,
        enc_time_col,
        tolerance=None,
        direction: str = "nearest",
        suffix: str = "_enc",
        drop_unmatched: bool = True,
        keep_time_delta: bool = True,
        inplace: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        left = self._flatten_columns(self.df)
        right = self._flatten_columns(encoder_df)

        cam_time_col = self._resolve_col_key(left, cam_time_col)
        enc_time_col = self._resolve_col_key(right, enc_time_col)

        left = left.copy()
        right = right.copy()
        left["_t"] = self._ensure_float_series(left[cam_time_col])
        right["_t_enc"] = self._ensure_float_series(right[enc_time_col])

        left = left.dropna(subset=["_t"]).sort_values("_t").reset_index(drop=False)
        right = right.dropna(subset=["_t_enc"]).sort_values("_t_enc").reset_index(drop=False)

        rename_map = {c: f"{c}{suffix}" for c in right.columns if c not in ("_t_enc", "index")}
        right = right.rename(columns=rename_map)

        merged = pd.merge_asof(
            left,
            right,
            left_on="_t",
            right_on="_t_enc",
            direction=direction,
            tolerance=tolerance,
            suffixes=("", suffix),
        )

        if keep_time_delta:
            with np.errstate(invalid="ignore"):
                merged[f"abs_dt{suffix}"] = (merged["_t"] - merged["_t_enc"]).abs()

        matched = merged[merged["_t_enc"].notna()].copy() if drop_unmatched else merged

        if verbose:
            print(f"[align_and_attach_encoder] camera rows in: {len(left)}, kept after match: {len(matched)}")

        matched = matched.set_index("index").sort_index()
        if inplace:
            self.df = matched
            return self.df
        return matched

    def load_imu_p_enc(self, imu_path, enc_path):
        def _to_str_path(p):
            if p is None:
                return None
            try:
                return str(p)
            except Exception:
                raise TypeError(f"Unsupported path type: {type(p)}")

        imu_path = _to_str_path(imu_path)
        enc_path = _to_str_path(enc_path)

        if imu_path is not None:
            if not os.path.exists(imu_path):
                raise FileNotFoundError(f"IMU path not found: {imu_path}")
            self.imu_df = pd.read_csv(imu_path)

        if enc_path is not None:
            if not os.path.exists(enc_path):
                raise FileNotFoundError(f"Encoder path not found: {enc_path}")
            self.enc_df = pd.read_csv(enc_path)

        if not hasattr(self, "imu_df") or self.imu_df is None:
            raise ValueError("IMU CSV was not loaded (imu_path was None or invalid).")
        if not hasattr(self, "enc_df") or self.enc_df is None:
            raise ValueError("Encoder CSV was not loaded (enc_path was None or invalid).")

    # ======================
    # Body-axes joint angle
    # ======================
    def compute_joint_angle_from_body_axes(
        self,
        fixed_prefix='imu1', fixed_axis='x',
        moving_prefix='imu2', moving_axis='z',
        out_col='imu_joint_deg_axes',
        degrees=True
    ):
        colA = f'{fixed_prefix}_{fixed_axis}vec'
        colB = f'{moving_prefix}_{moving_axis}vec'
        if colA not in self.imu_df.columns:
            self.euler_to_unit_vec(prefix=fixed_prefix, axis=fixed_axis, out_col=colA)
        if colB not in self.imu_df.columns:
            self.euler_to_unit_vec(prefix=moving_prefix, axis=moving_axis, out_col=colB)

        a = np.stack(self.imu_df[colA].apply(lambda v: np.array(v, float)).to_numpy())
        b = np.stack(self.imu_df[colB].apply(lambda v: np.array(v, float)).to_numpy())
        na = np.linalg.norm(a, axis=1, keepdims=True)
        nb = np.linalg.norm(b, axis=1, keepdims=True)
        mask = (na[:, 0] > 0) & (nb[:, 0] > 0)

        ang = np.full(len(a), np.nan, float)
        if np.any(mask):
            aa = a[mask] / na[mask]
            bb = b[mask] / nb[mask]
            dot = np.clip(np.sum(aa * bb, axis=1), -1.0, 1.0)
            val = np.arccos(dot)
            if degrees:
                val = np.degrees(val)
            ang[mask] = val

        self.imu_df[out_col] = ang
        return out_col

    def apply_fixed_mount_rotation(self, quat_col, axis='x', angle_deg=90.0, quat_order='wxyz', out_col=None):
        """
        Left-multiply a fixed mount rotation to every quaternion in `quat_col`:
            q_corrected = q_fixed ⊗ q_measured
        """
        def _parse_q(s):
            if isinstance(s, (list, tuple, np.ndarray)):
                arr = np.array(s, dtype=float)
                if arr.size == 4: return arr
                out = np.full(4, np.nan); out[:min(4, arr.size)] = arr[:min(4, arr.size)]
                return out
            if s is None or (isinstance(s, float) and np.isnan(s)): return np.full(4, np.nan)
            s = str(s).strip().strip('[](){}')
            parts = [p for p in s.replace(',', ' ').split() if p]
            vals = []
            for p in parts[:4]:
                try: vals.append(float(p))
                except: vals.append(np.nan)
            if len(vals) < 4: vals += [np.nan] * (4 - len(vals))
            return np.array(vals, dtype=float)

        def _qmul(q1, q2, order='wxyz'):
            if order == 'wxyz':
                w1, x1, y1, z1 = q1; w2, x2, y2, z2 = q2
                return np.array([
                    w1*w2 - x1*x2 - y1*y2 - z1*z2,
                    w1*x2 + x1*w2 + y1*z2 - z1*y2,
                    w1*y2 - x1*z2 + y1*w2 + z1*x2,
                    w1*z2 + x1*y2 - y1*x2 + z1*w2
                ], dtype=float)
            else:  # 'xyzw'
                x1, y1, z1, w1 = q1; x2, y2, z2, w2 = q2
                x = w1*x2 + x1*w2 + y1*z2 - z1*y2
                y = w1*y2 - x1*z2 + y1*w2 + z1*x2
                z = w1*z2 + x1*y2 - y1*x2 + z1*w2
                w = w1*w2 - x1*x2 - y1*y2 - z1*z2
                return np.array([x, y, z, w], dtype=float)

        ang = np.deg2rad(angle_deg)
        v = {'x': np.array([1,0,0.0]), 'y': np.array([0,1,0.0]), 'z': np.array([0,0,1.0])}[axis.lower()]
        s = np.sin(ang/2.0); c = np.cos(ang/2.0)
        if quat_order == 'wxyz':
            q_fix = np.array([c, v[0]*s, v[1]*s, v[2]*s], dtype=float)
        else:
            q_fix = np.array([v[0]*s, v[1]*s, v[2]*s, c], dtype=float)

        q_meas = self.imu_df[quat_col].apply(_parse_q).to_list()
        q_corr = []
        for q in q_meas:
            if np.any(~np.isfinite(q)):
                q_corr.append(np.array([np.nan]*4))
            else:
                qc = _qmul(q_fix, q, quat_order)
                qc = qc / (np.linalg.norm(qc) or 1.0)
                q_corr.append(qc)

        out = out_col or quat_col
        self.imu_df[out] = q_corr
        return out






class bender_class:
    
    '''
    class to manage data loading from CSVs, model training and testing, and data plotting
    '''

    def __init__(self, path=None):
        '''
        Initialize values for data, accuracy, model, and polynomial features
        '''
        self.data = None  # dataframe containing data from all csv files analyzed -> m rows by 4 columns
        self.acc = None  # accuracy from quadratic curve fitting class method:  quadriatic_fit(self)
        self.model = None  # To store the trained model
        self.poly_features = None  # To store polynomial features
        self.model_types = None # model-ftting type
        self.all_accuracies = []  # Initialize as an empty list for collecting accuracies
        self.accuracy_angle = np.arange(1, 16)  # Angle thresholds for accuracy calculations

        if path is None:
            self.repo_path = path_to_repository
        else: 
            self.repo_path = path

    def __str__(self):
        """
        human-readable, or informal, string representation of object
        """
        return (f"Bender Class: \n"
                f"  Number of data points: {self.data.shape[0] if self.data is not None else 0}\n"
                f"  Number of features: {self.data.shape[1] if self.data is not None else 0}\n"
                f"  Current Accuracy: {self.acc:.2f}% if self.acc is not None else 'N/A'\n")

    def __repr__(self):
        """
       more information-rich, or official, string representation of an object
       """

        return (f"Bender_class(data={self.data.head() if self.data is not None else 'None'}, "
                f"acc={self.acc}, "
                f"model={self.model.__class__.__name__ if self.model else 'None'}, "
                f"poly_features={self.poly_features.__class__.__name__ if self.poly_features else 'None'})")

    def load_merged_df(self, merged_df, enc_col="angle_renc", adc_col="adc_ch0"):
        """
        Load an already merged (aligned) dataframe into bender_class.
        Picks out encoder and ADC columns, converts encoder to degrees,
        then back into raw counts (so normalization + model methods still work).

        Parameters
        ----------
        merged_df : pd.DataFrame
            DataFrame containing at least encoder and ADC/strain columns.
        enc_col : str
            Column name for encoder angle in DEGREES (e.g., "angle_renc_renc").
        adc_col : str
            Column name for ADC/strain channel (e.g., "adc_ch0").
        """
        import pandas as pd
        import numpy as np

        if enc_col not in merged_df.columns:
            raise KeyError(f"Encoder column '{enc_col}' not found in merged_df. "
                           f"Available columns: {list(merged_df.columns)}")

        if adc_col not in merged_df.columns:
            raise KeyError(f"ADC column '{adc_col}' not found in merged_df. "
                           f"Available columns: {list(merged_df.columns)}")

        # 1) Get encoder degrees, convert to raw counts
        enc_counts = pd.to_numeric(merged_df[enc_col], errors="coerce")
      

        # 2) Get ADC values
        adc_vals = pd.to_numeric(merged_df[adc_col], errors="coerce")

        # 3) Build internal dataframe with 4 cols (like load_data)
        df = pd.DataFrame({
            "Rotary Encoder": enc_counts,
            "ADC Value": adc_vals,
            "C3": 0,
            "C4": 0,
        })

        self.data = df
        self.columns = df.columns
        self.adc_normalized = False
        self.normalize_type = None

        print(f"Loaded merged_df with {len(df)} rows into bender_class.")
        return self.data

    def load_data(self, regex_path):
        '''
        method to load data from csv files that match the regular expression string "regex_path"
        '''

        # Check that regex_path is a string
        if not isinstance(regex_path, str):
            raise TypeError("Expected 'path' to be a string.")

        # Use glob to get all the files in the folder that match the regex pattern
        csv_files = glob.glob(regex_path)
        print(csv_files)

        # Check that csv_files is not empty
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in the specified path: {regex_path}")

        # Load all the data
        dataframes = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                continue

            # Check if the DataFrame has exactly 4 columns
            if df.shape[1] != 4:
                raise ValueError(
                    f"Error: The file {f} does not contain exactly 4 columns. It has {df.shape[1]} columns.")

            # Remove rows where all columns equal "100" (usually first few rows)
            ix_ok = (df.iloc[:, 0] != 100) & (df.iloc[:, 1] != 100) & (df.iloc[:, 2] != 100) & (df.iloc[:, 3] != 100)
            df = df[ix_ok]

            # Convert rotary encoder to angle (degrees) -> ADC is Arduino Uno 10 bit (2**10 = 1024), rotary encoder has 320 degrees of rotation
            df['Rotary Encoder'] = df['Rotary Encoder'] * 320 / 1024

            # Shift rotary encoder angles to start tests at 0 degrees
            df['Rotary Encoder'] = df['Rotary Encoder'] - df['Rotary Encoder'].values[0]

            # make all rotary encoder angles > 0 so when plate is bent, it is at + 90 deg...if left alone angles go from 0 to -90 deg
            df['Rotary Encoder'] = df['Rotary Encoder'] * -1

            # Append the DataFrame to the list
            dataframes.append(df)

        # Concatenate all DataFrames in the list into a single DataFrame
        self.data = pd.concat(dataframes, ignore_index=True)

        # Add column names to make it so we dont need to remember the column numbers
        self.columns = self.data.columns

        # Have not yet normalized ADC data
        self.adc_normalized = False

    def normalize_adc_bw_01(self):
        """
        Normalizes ADC values to be between 0 and 1 while ensuring:
        - The first value starts at 0.
        - The data increases positively (mirrored if necessary).
        """

        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        if self.adc_normalized:
            raise ValueError("ADC data already normalized.")

        # Get min and max ADC values
        max_val = self.data['ADC Value'].max()
        min_val = self.data['ADC Value'].min()

        # Normalize ADC values between 0 and 1
        self.data['ADC Value'] = (self.data['ADC Value'] - min_val) / (max_val - min_val)

        # Identify ADC at 0° and near 90° (or closest available)
        adc_at_0 = self.data.loc[self.data['Rotary Encoder'].idxmin(), 'ADC Value']
        adc_at_90 = self.data.iloc[(self.data['Rotary Encoder'] - 90).abs().idxmin()]['ADC Value']

        # Check if ADC values decrease with increasing angle
        if adc_at_90 < adc_at_0:
            self.data['ADC Value'] = 1 - self.data['ADC Value']  # Mirror the dataset

        # Mark as normalized
        self.adc_normalized = True
        self.normalize_type = 'MinMax --> 0-1'

        print('ADC normalized bw 0-1. ADC max: ', self.data['ADC Value'].max(), 'ADC min: ',
              self.data['ADC Value'].min())

    def normalize_adc_over_R0(self):
        """
        Normalize ADC values to (R - R₀) / R₀ where R₀ is the initial resistance at the first strain value.
        This ensures normalized resistance starts near zero at strain = 0.
        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        if self.adc_normalized:
            raise ValueError("ADC data already normalized.")

        # Use min abs (ADC) value as R₀
        #R0 = self.data['ADC Value'].iloc[0]
        # Step 1: Check for negative values and shift if needed
        min_val = self.data['ADC Value'].min()

        if min_val < 0:
            self.data['ADC Value'] = self.data['ADC Value'] - min_val
        else:
            shift_note = ""

        # Step 2: Normalize using the first (possibly shifted) value as R₀
        R0 = self.data['ADC Value'].iloc[0]


        # Ensure R₀ is not zero to avoid division errors
        if R0 == 0:
            raise ValueError("Initial ADC value (R₀) is zero. Cannot normalize.")

        # Normalize the ADC values
        self.data['ADC Value'] = (self.data['ADC Value'] - R0) / R0

        # Mark as normalized
        self.adc_normalized = True
        self.normalize_type = '(R - R₀) / R₀'
        print(f"ADC normalized with initial value R₀: {R0}")

    def normalize_adc_over_largest(self, desired_max=None):
        """
        Normalize ADC values to largest (R - R₀) / R₀ value in all datasets

        """
        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        if self.adc_normalized:
            raise ValueError("ADC data already normalized.")

        # Normalize the ADC values to max value
        self.normalize_adc_over_R0()
        print(self.data['ADC Value'].max())
        self.data['ADC Value'] = self.data['ADC Value'].abs()
        max_value = self.data['ADC Value'].max()
        self.data['ADC Value'] =  self.data['ADC Value'] * desired_max / max_value

        # Mark as normalized
        self.adc_normalized = True
        self.normalize_type = '0 to max (R - R₀) / R₀'

    @staticmethod
    def _quad_fit_with_r2(x, y):
        """Return (poly1d p, r2, (c2,c1,c0)). NaNs are ignored; needs ≥3 points."""
        import numpy as _np
        x = _np.asarray(x, float).ravel()
        y = _np.asarray(y, float).ravel()
        m = _np.isfinite(x) & _np.isfinite(y)
        x, y = x[m], y[m]
        if x.size < 3:
            return None, _np.nan, (None, None, None)
        c2, c1, c0 = _np.polyfit(x, y, 2)
        p = _np.poly1d([c2, c1, c0])
        yhat = p(x)
        ss_res = float(_np.sum((y - yhat) ** 2))
        ss_tot = float(_np.sum((y - _np.nanmean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else _np.nan
        return p, r2, (c2, c1, c0)

    def plot_data(self, scatter=False, title=''):
        """
        method to plot normalized ADC values vs Rotary Encoder angles (blue dots)
        option to do a scatter plot where color == sample index
        """
        if self.data is None:
            raise ValueError("Data not available. Please read the data first.")
        
        if self.adc_normalized == False: 
            raise ValueError("ADC data not normalized. Please normalize data first.")

        # Create scatter plot: 
        f, ax = plt.subplots()

        # Plotting Rotary Encoder data
        if scatter:
            # Set color equal to data.index 
            ax.scatter(self.data['Rotary Encoder'], self.data['ADC Value'], c=self.data.index, cmap='viridis', s=5,
                    label='Rotary Encoder')
        else:
            ax.plot(self.data['Rotary Encoder'], self.data['ADC Value'], 'b.', markersize=5,
                 label='Rotary Encoder')  # Blue dots for Rotary Encoder
        
        # Setting labels
        ax.set_xlabel('Angle (deg)')
        ax.set_ylabel('Normalized ADC \n %s'%self.normalize_type) # state normalization type
        ax.set_title(title)

        self.data_ax = ax; 
        self.data_fig = f

    def plot_mech_model_data(self, thick, l_ch, l_sam, area, res, scatter=False,
                             data_color='blue', model_color='green',
                             data_label='Experimental Data', model_label='Theoretical Model',
                             normalize_by='over_R0', ax=None):
        """
        Class method to plot normalized data vs strain (ε) for both experimental data
        and a theoretical mechanics model. Supports normalization by:

        - '01': MinMax scaling between [0,1]
        - 'over_R0': (R - R₀) / R₀ normalization

        Parameters:
            normalize_by (str): '01' for MinMax normalization, 'over_R0' for (R - R₀) / R₀.
        """
        # Ensure data is loaded
        if self.data is None:
            raise ValueError("No data loaded. Please load data using the load_data method.")

        # Ensure data is normalized with the correct method
        if not self.adc_normalized:
            raise ValueError("Data not normalized. Please normalize the data first.")

        if self.normalize_type == 'MinMax --> 0-1' and normalize_by != '01':
            raise ValueError("Data was normalized using MinMax [0,1], but 'over_R0' normalization was requested.")

        if self.normalize_type == '(R - R₀) / R₀' and normalize_by != 'over_R0':
            raise ValueError("Data was normalized using (R - R₀) / R₀, but '01' normalization was requested.")

        # Compute strain (ε) for experimental data

        ##### PK notes -- why are values multiplied by 0.0254 all the time? Conversion from inches to meters?
        ##### PK notes -- Since strain is unitless can we just use inches? 
        #self.data['Strain (ε)'] = (thick * 0.0254) * (self.data['Rotary Encoder'] * np.pi / 180) / (l_sam * 0.0254)
        self.data['Strain (ε)'] = (thick) * (self.data['Rotary Encoder'] * np.pi / 180) / (l_sam)

        # Prepare theoretical model data
        theta = np.arange(0, np.pi / 2 + 0.1, 0.1)  # Include up to 90 degrees
        rho = 29.4 * 10 ** -8  # Electrical resistivity of galinstan
        eps_model = (thick) * theta / (l_sam)  # Strain (ε) for theoretical model

        ### PK notes -- here looks like  you need the inches_to_meters: 
        inches_to_meters = 0.0254

        ### PK notes -- What is this 0.000645 value? Can this be defined?
        # Convert area from in² to m² using 1 in² = 0.000645 m²
        dr_model = (rho * eps_model * (l_ch * inches_to_meters) * (8 - eps_model) /
                    ((area * 0.000645) * (2 - eps_model) ** 2))   # Resistance change ΔR

        # Apply selected normalization method
        if normalize_by == 'over_R0':
            model_data = dr_model / res  # ΔR / R₀
            y_label = r'Normalized ADC $(\Delta R / R_{0})$'
        elif normalize_by == '01':
            # Normalize theoretical model using MinMax scaling to [0,1]
            model_data = (dr_model - dr_model.min()) / (dr_model.max() - dr_model.min())
            y_label = 'Normalized ADC (0-1)'
        else:
            raise ValueError("Invalid normalization method. Choose '01' for MinMax scaling or 'over_R0' for ΔR / R₀.")

        # Interpolate the model to get predictions at the experimental strain values
        f_interp = interp1d(eps_model, model_data, kind='linear', fill_value='extrapolate')
        model_at_data = f_interp(self.data['Strain (ε)'])

        # Compute the R² value
        # This is an R2 for how well the model fits the data 
        ss_res = np.sum((self.data['ADC Value'] - model_at_data) ** 2)
        ss_tot = np.sum((self.data['ADC Value'] - np.mean(self.data['ADC Value'])) ** 2)
        r2_model2data = 1 - (ss_res / ss_tot)

        # Create new plot or add to existing one
        if ax is None:
            fig, ax = plt.subplots()

        # Plot experimental data
        if scatter:
            ax.scatter(self.data['Strain (ε)'], self.data['ADC Value'], c=self.data.index,
                       cmap='viridis', s=5, label=data_label)
        else:
            ax.plot(self.data['Strain (ε)'], self.data['ADC Value'], '.',
                    markersize=5, color=data_color, label=data_label)

        # Plot theoretical model with the R² value included in the legend label
        ax.plot(eps_model, model_data, '--', color=model_color,
                label=f"{model_label} (R² = {r2_model2data:.3f})")

        # Set labels and legend only if a new figure was created
        if ax.get_title() == '':
            ax.set_xlabel('Strain (ε)')
            ax.set_ylabel(y_label)
            ax.set_title('Experimental vs Theoretical Model')
        ax.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))

        # Return the axes object to allow further plotting
        return ax

    def train_model_test_accuracy(self, perc_train=0.8, niter=10, degree=1):
        """
        Trains a polynomial model (linear by default) to predict rotary encoder angles
        from normalized ADC values.

        Parameters:
        - perc_train: float (0–1), percent of data used for training
        - niter: number of train/test iterations
        - degree: degree of the polynomial fit (1=linear, 2=quadratic, etc.)
        """

        if self.data is None:
            raise ValueError("Data not available. Please read the data first.")

        if not self.adc_normalized:
            raise ValueError("ADC data not normalized. Please normalize data first.")

        self.accuracy_angle = np.arange(1, 16, 0.2)  # accuracy tested up to 15 deg
        self.accuracy = np.zeros((niter, len(self.accuracy_angle)))
        self.abs_angular_error = []

        self.poly = PolynomialFeatures(degree=degree, include_bias=True)

        for i in range(niter):
            # Cross-validation
            dataTrain, dataTest = train_test_split(self.data, test_size=1.0 - perc_train, shuffle=True)

            # Extract and transform training data
            X_train_raw = dataTrain['ADC Value'].values.reshape(-1, 1)
            X_train = self.poly.fit_transform(X_train_raw)
            y_train = dataTrain['Rotary Encoder'].values

            self.model = LinearRegression()
            self.model.fit(X_train, y_train)

            # Predict using test data
            X_test_raw = dataTest['ADC Value'].values.reshape(-1, 1)
            X_test = self.poly.transform(X_test_raw)
            Y_test = dataTest['Rotary Encoder'].values
            Y_pred = self.model.predict(X_test)

            self.abs_angular_error.append(np.abs(Y_test - Y_pred))

            for j, angle_accuracy in enumerate(self.accuracy_angle):
                self.accuracy[i, j] = self.accuracy_by_angle(Y_test, Y_pred, angle_accuracy)

        self.all_accuracies.append(self.accuracy)



    def plot_pairwise_min_angle_heatmap(self, df_results, group_dict, group_colors, label):
        """
        Generate a heatmap with colored lines around defined groups, ensuring dataset order.

        Parameters:
        - df_results: Pandas DataFrame with ["train_dataset", "test_dataset", "min_angle_100"]
        - group_dict: Dictionary mapping dataset names to group labels
        - group_colors: Dictionary mapping group labels to border colors
        """

        if df_results.empty:
            print("No results to display.")
            return

        # Extract dataset names and enforce numeric order
        def extract_number(ds_name):
            """ Extract the numeric part from DS names (e.g., 'DS10' -> 10) """
            return int(ds_name[2:])  # Assuming 'DS' prefix, so we take the numeric part

        # Get all unique dataset names and sort them
        dataset_order = sorted(set(df_results["train_dataset"]).union(df_results["test_dataset"]), key=extract_number)

        # Convert results into a pivot table, ensuring correct order
        df_pivot = df_results.pivot(index="train_dataset", columns="test_dataset", values="min_angle_100")
        df_pivot = df_pivot.reindex(index=dataset_order, columns=dataset_order)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        sns.heatmap(df_pivot, cmap="coolwarm", linewidths=0.5, cbar=True, cbar_kws={'label': label}, ax=ax)

        # Identify group positions
        dataset_positions = {dataset: i for i, dataset in enumerate(dataset_order)}

        group_positions = {}
        for dataset, group in group_dict.items():
            if group not in group_positions:
                group_positions[group] = []
            if dataset in dataset_positions:
                group_positions[group].append(dataset_positions[dataset])

        # Draw rectangles around groups
        for group, indices in group_positions.items():
            if len(indices) > 1:
                min_idx, max_idx = min(indices), max(indices) + 1  # Extend to include last dataset
                color = group_colors.get(group, "black")  # Default to black if no color is specified

                # Draw group boundary lines
                ax.hlines(y=max_idx, xmin=min_idx, xmax=max_idx, colors=color, linewidth=2)
                ax.hlines(y=min_idx, xmin=min_idx, xmax=max_idx, colors=color, linewidth=2)
                ax.vlines(x=min_idx, ymin=min_idx, ymax=max_idx, colors=color, linewidth=2)
                ax.vlines(x=max_idx, ymin=min_idx, ymax=max_idx, colors=color, linewidth=2)

        # Labels and title
        ax.set_title("Pairwise Min Angle (Accuracy 100%) Heatmap with Group Outlines")
        ax.set_xlabel("Test Dataset")
        ax.set_ylabel("Train Dataset")

        plt.show()

    def plot_compact_pairwise_comparison(
            self,
            pairwise_min_accuracy,
            pairwise_abs_error,
            xlabel_flat,
            group_size=3,
            title1='Min Angle for 100% Accuracy',
            title2='Mean Absolute Error',
            ylim=(0, 15),
            # Optional distributions for error bars on Min Angle bars
            self_minangle_dists=None,   # list[list[np.ndarray]]
            cross_minangle_dists=None,  # list[list[np.ndarray]]; []/None for first in group
            err_metric='sd',            # 'sd' or 'sem'
            capsize=4,
            # NEW: embed into an existing axes and align with left plot spacing
            ax_top=None,                # if provided, draw ONLY the top bars into this axes
            x_sample_centers=None,      # list of sample centers (same order as xlabel_flat)
            group_centers=None,         # list of one center per group (for xticks)
            group_labels=None,          # list of group labels (tick labels)
            bar_w=0.24,                 # bar width for side-by-side bars
            paired=True,                # if True, place red next to blue
            show_sample_numbers=True,   # annotate "1,2,3,..."
    ):
        import numpy as np
        import matplotlib.pyplot as plt

        assert pairwise_min_accuracy.shape[0] == len(xlabel_flat)
        assert pairwise_min_accuracy.shape[0] % group_size == 0
        num_groups = pairwise_min_accuracy.shape[0] // group_size

        def mean_sd(arr):
            arr = np.asarray(arr, float)
            if arr.size == 0:
                return np.nan, np.nan
            m = np.nanmean(arr)
            s = np.nanstd(arr, ddof=1) if arr.size > 1 else np.nan
            return m, s

        def mean_sem(arr):
            arr = np.asarray(arr, float)
            n = np.sum(np.isfinite(arr))
            if n == 0:
                return np.nan, np.nan
            m = np.nanmean(arr)
            s = (np.nanstd(arr, ddof=1) / np.sqrt(n)) if n > 1 else np.nan
            return m, s

        use_err = (self_minangle_dists is not None) and (cross_minangle_dists is not None)
        agg = mean_sem if err_metric.lower() == 'sem' else mean_sd

        # Standalone mode (old behavior) OR embedded into provided ax_top
        standalone = ax_top is None
        if standalone:
            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            ax_min = axes[0]
            ax_err = axes[1]
        else:
            ax_min = ax_top
            ax_err = None

        # ----- Build flat arrays in sample order -----
        min_self  = []
        min_cross = []
        self_yerr = []
        cross_yerr = []

        for g in range(num_groups):
            base = g * group_size
            for i in range(group_size):
                sidx = base + i
                # bars (means) from matrices
                min_self.append(pairwise_min_accuracy[sidx, sidx])   # diagonal
                # cross is first in group -> others
                if i == 0:
                    min_cross.append(np.nan)  # no red on model-1
                else:
                    min_cross.append(pairwise_min_accuracy[base, sidx])

                if use_err:
                    s_arr = self_minangle_dists[g][i]
                    _, s_err = agg(s_arr)
                    self_yerr.append(s_err)

                    c_arr = cross_minangle_dists[g][i]
                    if (i == 0) or (c_arr is None) or (len(c_arr) == 0):
                        cross_yerr.append(np.nan)
                    else:
                        _, c_err = agg(c_arr)
                        cross_yerr.append(c_err)

        min_self  = np.asarray(min_self, dtype=float)
        min_cross = np.asarray(min_cross, dtype=float)
        self_yerr = np.asarray(self_yerr, dtype=float) if use_err else None
        cross_yerr = np.asarray(cross_yerr, dtype=float) if use_err else None

        # ----- X positions -----
        if not paired:
            # overlay mode (legacy): blue full-width, red inner-width
            x_idx = []
            for g in range(num_groups):
                x_idx.extend(list(np.arange(group_size) + g * (group_size + 1.0)))
            x_idx = np.asarray(x_idx, dtype=float)
            ax_min.bar(x_idx, min_self, width=0.60, color='blue', label='Self-trained')
            ax_min.bar(x_idx, min_cross, width=0.40, color='red', alpha=0.7, label='Cross-trained (model 1)')
            if use_err:
                ax_min.errorbar(x_idx, min_self,  yerr=self_yerr,  fmt='none', ecolor='k', elinewidth=1.1, capsize=capsize)
                ax_min.errorbar(x_idx, min_cross, yerr=cross_yerr, fmt='none', ecolor='k', elinewidth=1.1, capsize=capsize)
            # ticks per sample (legacy)
            ax_min.set_xticks(x_idx)
            ax_min.set_xticklabels(xlabel_flat, rotation=45, ha='right')
        else:
            # side-by-side mode aligned to provided sample centers
            assert x_sample_centers is not None, "Provide x_sample_centers for paired=True"
            x_centers = np.asarray(x_sample_centers, dtype=float)
            x_self  = x_centers - bar_w/2
            x_cross = x_centers + bar_w/2
            # mask out first-in-group for cross bars
            cross_mask = []
            for g in range(num_groups):
                cross_mask.extend([False] + [True]*(group_size-1))
            cross_mask = np.asarray(cross_mask, dtype=bool)

            # draw bars
            ax_min.bar(x_self, min_self,  width=bar_w, color='blue', label='Self-trained')
            ax_min.bar(x_cross[cross_mask], min_cross[cross_mask], width=bar_w, color='red', alpha=0.7,
                       label='Cross-trained (model 1)')

            # error bars
            if use_err:
                ax_min.errorbar(x_self, min_self, yerr=self_yerr, fmt='none', ecolor='k', elinewidth=1.1, capsize=capsize)
                ax_min.errorbar(x_cross[cross_mask], min_cross[cross_mask],
                                yerr=cross_yerr[cross_mask], fmt='none', ecolor='k', elinewidth=1.1, capsize=capsize)

            # group-level ticks
            assert (group_centers is not None) and (group_labels is not None), \
                "Provide group_centers and group_labels for paired=True"
            ax_min.set_xticks(group_centers)
            ax_min.set_xticklabels(group_labels)

            # sample numbers above each sample center
            if show_sample_numbers:
                # numbers 1..group_size repeating per group
                sample_nums = []
                for g in range(num_groups):
                    sample_nums.extend([str(i+1) for i in range(group_size)])
                # y placement: max of self(+err) and cross(+err)
                ymin, ymax = ylim
                y_pad = 0.02 * (ymax - ymin)
                for i, xc in enumerate(x_centers):
                    y_self_top = min_self[i] + (self_yerr[i] if (use_err and np.isfinite(self_yerr[i])) else 0.0)
                    if cross_mask[i]:
                        y_cross_top = min_cross[i] + (cross_yerr[i] if (use_err and np.isfinite(cross_yerr[i])) else 0.0)
                        y_top = max(y_self_top, y_cross_top)
                    else:
                        y_top = y_self_top
                    ax_min.text(xc, y_top + y_pad, sample_nums[i], ha='center', va='bottom', fontsize=12, fontweight='bold', clip_on=False)

        # style & labels
        ax_min.set_ylabel('Min Angle (deg)')
        ax_min.set_title(title1)
        ax_min.set_ylim(ylim)
        ax_min.legend(frameon=False)

        # clean style per your request
        ax_min.grid(False)
        ax_min.spines['top'].set_visible(False)
        ax_min.spines['right'].set_visible(False)

        if standalone:
            # Bottom subplot (legacy mode) if you ever call without ax_top
            # (kept unchanged; you said you only need the top when embedding)
            ax_err.set_ylabel('Mean Error (deg)')
            ax_err.set_xlabel('Sample')
            ax_err.set_ylim(ylim)
            ax_err.set_yticks(np.arange(ylim[0], ylim[1] + 1, 1))
            ax_err.set_title(title2)
            ax_err.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig("compact_pairwise_comp.png", dpi=300, bbox_inches='tight')
            plt.show()


    def plot_pairwise_mean_error_heatmap(self, df_results, group_dict, group_colors, label):
        """
        Generate a heatmap with colored lines around defined groups, using mean angular error.

        Parameters:
        - df_results: Pandas DataFrame with ["train_dataset", "test_dataset", "mean_error"]
        - group_dict: Dictionary mapping dataset names to group labels
        - group_colors: Dictionary mapping group labels to border colors
        - label: Label for the color bar, typically "Mean Angular Error (deg)"
        """
        if df_results.empty:
            print("No results to display.")
            return

        # Extract dataset names and enforce numeric order
        def extract_number(ds_name):
            """ Extract the numeric part from DS names (e.g., 'DS10' -> 10) """
            return int(ds_name[2:])  # Assuming 'DS' prefix, so we extract the numeric part

        # Get all unique dataset names and sort them numerically
        dataset_order = sorted(set(df_results["train_dataset"]).union(df_results["test_dataset"]), key=extract_number)

        # Convert results into a pivot table, ensuring correct dataset order
        df_pivot = df_results.pivot(index="train_dataset", columns="test_dataset", values="mean_error")
        df_pivot = df_pivot.reindex(index=dataset_order, columns=dataset_order)

        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot heatmap
        sns.heatmap(df_pivot, cmap="coolwarm", linewidths=0.5, cbar=True, cbar_kws={'label': label}, ax=ax)

        # Identify group positions
        dataset_positions = {dataset: i for i, dataset in enumerate(dataset_order)}

        group_positions = {}
        for dataset, group in group_dict.items():
            if group not in group_positions:
                group_positions[group] = []
            if dataset in dataset_positions:
                group_positions[group].append(dataset_positions[dataset])

        # Draw rectangles around groups
        for group, indices in group_positions.items():
            if len(indices) > 1:
                min_idx, max_idx = min(indices), max(indices) + 1  # Extend to include last dataset
                color = group_colors.get(group, "black")  # Default to black if no color is specified

                # Draw group boundary lines
                ax.hlines(y=max_idx, xmin=min_idx, xmax=max_idx, colors=color, linewidth=2)
                ax.hlines(y=min_idx, xmin=min_idx, xmax=max_idx, colors=color, linewidth=2)
                ax.vlines(x=min_idx, ymin=min_idx, ymax=max_idx, colors=color, linewidth=2)
                ax.vlines(x=max_idx, ymin=min_idx, ymax=max_idx, colors=color, linewidth=2)

        # Labels and title
        ax.set_title("Pairwise Mean Angular Error Heatmap with Group Outlines")
        ax.set_xlabel("Test Dataset")
        ax.set_ylabel("Train Dataset")

        plt.show()

    import numpy as np
    from scipy.optimize import curve_fit
    import matplotlib.pyplot as plt

    def _norm_theoretical(self, theta_rad, r, L=2.0):
        """
        Normalized theoretical curve vs angle θ (radians).
        alpha = r / L.
        y(0)=0 and y(pi/2)=1 by construction.
        """
        theta = np.asarray(theta_rad, dtype=float)
        alpha = r / L

        num = theta * (8.0 - alpha * theta) / (2.0 - alpha * theta) ** 2
        th_max = np.pi / 2.0
        den = th_max * (8.0 - alpha * th_max) / (2.0 - alpha * th_max) ** 2
        return num / den

    def fit_knuckle_radius_from_normalized(
            self,
            # --- theory fit target (now Block 1 normalized points) ---
            L=2.0,
            r0=0.7,
            bounds=(1e-4, 2.45),

            # --- BLOCK sources: ranges + angles (this method will build angle→adc tables) ---
            h_cal_path_first=None,  # str | pd.Series | np.ndarray (ADC stream for Block 1)
            ranges_first=None,  # list[(start,end)] aligned with 'angles'
            h_cal_path_second=None,  # str | pd.Series | np.ndarray (ADC stream for Block 2)
            ranges_second=None,  # list[(start,end)] aligned with 'angles'
            angles=(0.0, 22.5, 45.0, 67.5, 90.0),
            max_points_per_range=None,  # e.g., 200 (keeps flattest window), or None for all

            # --- CAMERA sources (means or points) ---
            cam_df=None,  # DataFrame (both sets with set_idx) OR list/tuple of DFs
            cam_index_first=None,  # which item is Camera 1 if cam_df is a sequence
            cam_index_second=None,  # which item is Camera 2 if cam_df is a sequence
            cam_angle_col_hint="angle_deg",
            cam_adc_col_hint="mean_adc",

            # --- general behaviors ---
            restrict_to_0_90=True,
            make_angles_positive=True,
            flip_data=False,  # flips normalized y → (1-y) everywhere (incl. theory fit)
            show_quadratic=True,
            plot=False,
            ax=None,

            # --- labels ---
            block1_label="Block 1",
            block2_label="Block 2",
            cam1_label="Camera 1",
            cam2_label="Camera 2",
            theory_label="Theory fit",

            # --- styles ---
            style_theory=dict(color="tab:orange", linewidth=2),
            style_block1_points=dict(color="tab:green", s=22, alpha=0.75),
            style_block1_curve=dict(color="tab:green", linestyle="--", linewidth=2),
            style_block2_points=dict(color="tab:red", s=22, alpha=0.75),
            style_block2_curve=dict(color="tab:red", linestyle="--", linewidth=2),
            style_cam1_points=dict(color="tab:blue", s=28, alpha=0.80, marker="o"),
            style_cam1_curve=dict(color="tab:blue", linestyle="-.", linewidth=2),
            style_cam2_points=dict(color="tab:purple", s=28, alpha=0.80, marker="o"),
            style_cam2_curve=dict(color="tab:purple", linestyle="-.", linewidth=2),

            # --- outputs ---
            return_curve=True,
            theta_grid=None,

            # --- diagnostics ---
            verbose=False,
    ):
        """
        Build Block 1/2 angle→ADC from (ranges, angles) and CSV/Series, do per-set quadratic fits
        normalized to [0,1] via each fit's y(0°), y(90°). Do the same for Camera 1/2.
        Fit the theoretical model (normalized) to Block 1 normalized *points* and report r̂, R².
        Optionally plot overlays and return all fit artifacts.

        Returns:
          {
            "r_hat", "r_se", "r_ci95", "r2", "params_cov",
            "theta", "y_model",                       # theory curve if return_curve
            "overlays": {
               "block1": {"ok","coeffs","r2","x_pts","y_pts_norm","theta","y_curve_norm"},
               "block2": {...},
               "cam1":   {...},
               "cam2":   {...}
            }
          }
        """
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from scipy.optimize import curve_fit
        import copy
        from pathlib import Path

        # ---------------- helpers ----------------
        def _load_adc_series(src):
            # Accept path, Series, ndarray, list-like
            if src is None:
                return None
            if isinstance(src, (pd.Series, list, tuple, np.ndarray)):
                v = np.asarray(src).astype(float)
                return v[np.isfinite(v)]
            # path-like → CSV → column pick
            p = Path(str(src))
            if not p.exists() or p.suffix.lower() != ".csv":
                raise FileNotFoundError(f"ADC source not found or not a CSV: {src}")
            df = pd.read_csv(p)
            cand = "adc_ch3" if "adc_ch3" in df.columns else next(
                (c for c in df.columns if str(c).lower().startswith("adc")), None
            )
            if cand is None:
                raise KeyError(f"No ADC-like column in {p.name}")
            y = pd.to_numeric(df[cand], errors="coerce").to_numpy(float)
            y = y[np.isfinite(y)]
            return y

        def _flattest_window(vals, k):
            # pick consecutive window length k with minimal variance
            if k is None or len(vals) <= 0 or k >= len(vals):
                return vals
            import numpy as _np
            c1 = _np.concatenate(([0.0], _np.cumsum(vals)))
            c2 = _np.concatenate(([0.0], _np.cumsum(vals * vals)))
            sum_y = c1[k:] - c1[:-k]
            sum_y2 = c2[k:] - c2[:-k]
            mean_y = sum_y / k
            var_y = _np.maximum(sum_y2 / k - mean_y ** 2, 0.0)
            i0 = int(_np.argmin(var_y))
            return vals[i0:i0 + k]

        def _build_block_angle_df(adc_stream, ranges, angles_list):
            if adc_stream is None or ranges is None or angles_list is None:
                return None
            parts = []
            n = len(adc_stream)
            for (start, end), ang in zip(ranges, angles_list):
                i0 = int(max(0, start))
                i1 = int(min(n, end))
                if i1 <= i0:
                    continue
                vals = adc_stream[i0:i1]
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                if max_points_per_range is not None and vals.size > max_points_per_range:
                    vals = _flattest_window(vals, int(max_points_per_range))
                parts.append(pd.DataFrame({"angle": float(ang), "adc": vals}))
            return (pd.concat(parts, ignore_index=True)
                    if parts else pd.DataFrame(columns=["angle", "adc"]))

        def _tidy_angle_adc(df, angle_hint=None, adc_hint=None, tag="?"):
            if df is None or df.empty:
                return None
            a = angle_hint if (angle_hint and angle_hint in df.columns) else None
            d = adc_hint if (adc_hint and adc_hint in df.columns) else None
            if a is None:
                a = next((c for c in ["angle", "angle_deg", "theta", "Rotary Encoder"] if c in df.columns), None)
                if a is None:
                    a = next((c for c in df.columns if "angle" in str(c).lower()), None)
            if d is None:
                d = next((c for c in ["adc_norm", "mean_adc", "adc", "adc_ch3", "ADC Value"] if c in df.columns), None)
                if d is None:
                    d = next((c for c in df.columns if str(c).lower().startswith("adc")), None)
            if a is None or d is None:
                raise KeyError(f"[{tag}] cannot find angle/adc columns in {list(df.columns)}")
            out = df[[a, d]].rename(columns={a: "angle", d: "adc"}).copy()
            out["angle"] = pd.to_numeric(out["angle"], errors="coerce")
            out["adc"] = pd.to_numeric(out["adc"], errors="coerce")
            out = out.dropna()
            m = out["angle"].between(0.0, 90.0)
            sub = out.loc[m].copy()
            if sub.shape[0] < 5:
                m2 = out["angle"].between(-5.0, 95.0)
                sub = out.loc[m2].copy()
                sub["angle"] = sub["angle"].clip(0.0, 90.0)
            return sub

        def _quadfit_norm(df, theta_line, tag="?"):
            # fit: adc = c0 + c1*θ + c2*θ² ; normalize by fit's y(0), y(90)
            import numpy as _np
            if df is None or df.shape[0] < 3:
                return {"ok": False, "reason": "too_few_points"}
            x = df["angle"].to_numpy(float)
            y = df["adc"].to_numpy(float)
            c2, c1, c0 = _np.polyfit(x, y, deg=2)
            p = _np.poly1d([c2, c1, c0])
            yhat = p(x)
            ss_res = float(_np.sum((y - yhat) ** 2))
            ss_tot = float(_np.sum((y - float(_np.mean(y))) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else _np.nan
            y0, y90 = float(p(0.0)), float(p(90.0))
            y_grid = p(theta_line)
            if y90 >= y0:
                y_curve_n = (y_grid - y0) / max(1e-12, (y90 - y0))
                y_pts_n = (y - y0) / max(1e-12, (y90 - y0))
            else:
                y_curve_n = (y0 - y_grid) / max(1e-12, (y0 - y90))
                y_pts_n = (y0 - y) / max(1e-12, (y0 - y90))
            return {
                "ok": True,
                "coeffs": (float(c0), float(c1), float(c2)),
                "r2": float(r2),
                "theta": theta_line,
                "x_pts": x,
                "y_pts_norm": y_pts_n,
                "y_curve_norm": y_curve_n,
            }

        def _split_cam_sets(cam_in):
            """Return (cam1_df, cam2_df) as tidy angle/adc tables."""
            if cam_in is None:
                return None, None
            # sequence → pick by indices
            if isinstance(cam_in, (list, tuple)):
                c1 = _tidy_angle_adc(cam_in[cam_index_first or 0],
                                     angle_hint=cam_angle_col_hint,
                                     adc_hint=cam_adc_col_hint,
                                     tag="cam1")
                c2 = None
                if cam_index_second is not None and cam_index_second < len(cam_in):
                    c2 = _tidy_angle_adc(cam_in[cam_index_second],
                                         angle_hint=cam_angle_col_hint,
                                         adc_hint=cam_adc_col_hint,
                                         tag="cam2")
                return c1, c2
            # single DF → try set_idx or set_label to split
            df = cam_in.copy()
            key = "set_idx" if "set_idx" in df.columns else ("set_label" if "set_label" in df.columns else None)
            if key is None:
                # assume it's already one set → cam1 only
                c1 = _tidy_angle_adc(df, angle_hint=cam_angle_col_hint, adc_hint=cam_adc_col_hint, tag="cam1")
                return c1, None
            c1_raw = df[df[key].astype(str).str.lower().isin(["1", "first", "1st"])].copy()
            c2_raw = df[df[key].astype(str).str.lower().isin(["2", "second", "2nd"])].copy()
            c1 = _tidy_angle_adc(c1_raw, angle_hint=cam_angle_col_hint, adc_hint=cam_adc_col_hint,
                                 tag="cam1") if not c1_raw.empty else None
            c2 = _tidy_angle_adc(c2_raw, angle_hint=cam_angle_col_hint, adc_hint=cam_adc_col_hint,
                                 tag="cam2") if not c2_raw.empty else None
            return c1, c2

        # ---------------- build Block 1/2 angle→adc from ranges ----------------
        theta_grid = np.linspace(0.0, 90.0, 400) if theta_grid is None else np.asarray(theta_grid, float)
        blk1_df = None
        blk2_df = None
        if h_cal_path_first is not None and ranges_first is not None:
            adc1 = _load_adc_series(h_cal_path_first)
            blk1_df = _build_block_angle_df(adc1, ranges_first, angles)
        if h_cal_path_second is not None and ranges_second is not None:
            adc2 = _load_adc_series(h_cal_path_second)
            blk2_df = _build_block_angle_df(adc2, ranges_second, angles)

        # tidy camera sets
        cam1_df, cam2_df = _split_cam_sets(cam_df)

        # ---------------- quadratic fits (normalized) for each overlay ----------------
        overlays = {}
        if show_quadratic and blk1_df is not None and not blk1_df.empty:
            overlays["block1"] = _quadfit_norm(blk1_df, theta_grid, tag="block1")
        if show_quadratic and blk2_df is not None and not blk2_df.empty:
            overlays["block2"] = _quadfit_norm(blk2_df, theta_grid, tag="block2")
        if show_quadratic and cam1_df is not None and not cam1_df.empty:
            overlays["cam1"] = _quadfit_norm(cam1_df, theta_grid, tag="cam1")
        if show_quadratic and cam2_df is not None and not cam2_df.empty:
            overlays["cam2"] = _quadfit_norm(cam2_df, theta_grid, tag="cam2")

        # ---------------- theory fit (normalized) using Block 1 normalized points ----------------
        # pick Block 1 normalized points (fall back to any available set)
        fit_source = None
        fit_label = None
        if "block1" in overlays and overlays["block1"].get("ok"):
            fit_source, fit_label = overlays["block1"], block1_label
        elif "cam1" in overlays and overlays["cam1"].get("ok"):
            fit_source, fit_label = overlays["cam1"], cam1_label
        elif "block2" in overlays and overlays["block2"].get("ok"):
            fit_source, fit_label = overlays["block2"], block2_label
        elif "cam2" in overlays and overlays["cam2"].get("ok"):
            fit_source, fit_label = overlays["cam2"], cam2_label
        else:
            raise ValueError("No valid overlay found to fit the theoretical model against.")

        x_deg_fit = np.asarray(fit_source["x_pts"], float)
        y_fit = np.asarray(fit_source["y_pts_norm"], float)

        if make_angles_positive:
            x_deg_fit = np.abs(x_deg_fit)
        if restrict_to_0_90:
            mfit = (x_deg_fit >= 0.0) & (x_deg_fit <= 90.0) & np.isfinite(y_fit)
            x_deg_fit = x_deg_fit[mfit];
            y_fit = y_fit[mfit]
        if flip_data:
            y_fit = 1.0 - y_fit
            for key in overlays:
                if overlays[key].get("ok"):
                    overlays[key]["y_pts_norm"] = 1.0 - np.asarray(overlays[key]["y_pts_norm"], float)
                    overlays[key]["y_curve_norm"] = 1.0 - np.asarray(overlays[key]["y_curve_norm"], float)

        if x_deg_fit.size < 5:
            raise ValueError("Not enough points (need ≥5) in 0–90° from the selected fit source to fit theory.")

        theta_data = np.deg2rad(x_deg_fit)

        def _norm_theoretical(theta_r, r, L=L):
            alpha = r / L
            num = theta_r * (8.0 - alpha * theta_r) / (2.0 - alpha * theta_r) ** 2
            th_max = np.pi / 2.0
            den = th_max * (8.0 - alpha * th_max) / (2.0 - alpha * th_max) ** 2
            return num / den

        popt, pcov = curve_fit(lambda th, r: _norm_theoretical(th, r, L),
                               theta_data, y_fit, p0=[r0], bounds=bounds, maxfev=20000)
        r_hat = float(popt[0])
        y_pred = _norm_theoretical(theta_data, r_hat, L)
        ss_res = float(np.sum((y_fit - y_pred) ** 2))
        ss_tot = float(np.sum((y_fit - float(np.mean(y_fit))) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        r_se = float(np.sqrt(pcov[0, 0])) if pcov.size else np.nan
        ci95 = (r_hat - 1.96 * r_se, r_hat + 1.96 * r_se) if np.isfinite(r_se) else (np.nan, np.nan)

        # theory curve grid
        theta_grid = np.asarray(theta_grid, float)
        theta_grid_rad = np.deg2rad(theta_grid)
        y_theory = _norm_theoretical(theta_grid_rad, r_hat, L)
        if flip_data:
            y_theory = 1.0 - y_theory

        # ---------------- plotting ----------------
        if plot:
            if ax is None:
                _, ax = plt.subplots(figsize=(6.9, 4.4))
            ax.set_xlim(0, 90)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Normalized ADC (0–1)")
            ax.set_title(f"{theory_label} on {fit_label} (L={L} in)")

            legend_items = []

            # theory
            th = ax.plot(theta_grid, y_theory,
                         color=style_theory.get("color", "tab:orange"),
                         linewidth=style_theory.get("linewidth", 2),
                         linestyle=style_theory.get("linestyle", "-"),
                         label="_nolegend_")[0]
            legend_items.append((th, f"{theory_label} (r={r_hat:.3f}, R²={r2:.4f})"))

            # overlays
            def _plot_one(tag, pts_style, curve_style, label_pts, label_curve):
                ov = overlays.get(tag, {})
                if not ov or not ov.get("ok"):
                    return
                sc = ax.scatter(ov["x_pts"], ov["y_pts_norm"], **pts_style)
                ln, = ax.plot(ov["theta"], ov["y_curve_norm"], **curve_style)
                legend_items.append((sc, f"{label_pts} (points)"))
                legend_items.append((ln, f"{label_curve} (quad, R²={ov['r2']:.4f})"))

            _plot_one("block1", style_block1_points, style_block1_curve, block1_label, block1_label)
            _plot_one("block2", style_block2_points, style_block2_curve, block2_label, block2_label)
            _plot_one("cam1", style_cam1_points, style_cam1_curve, cam1_label, cam1_label)
            _plot_one("cam2", style_cam2_points, style_cam2_curve, cam2_label, cam2_label)

            if legend_items:
                handles, labels = zip(*legend_items)
                ax.legend(list(handles), list(labels), loc="best", frameon=True)
            ax.grid(alpha=0.25, axis="y")
            plt.tight_layout()

        # ---------------- return ----------------
        out = {
            "r_hat": r_hat,
            "r_se": r_se,
            "r_ci95": ci95,
            "r2": r2,
            "params_cov": pcov,
            "overlays": overlays,
        }
        if return_curve:
            out.update({"theta": theta_grid, "y_model": y_theory})
        return out

    def append_theta_from_normalized(self,
                                     r_hand,
                                     L=2.0,
                                     value_col="ADC Value",
                                     out_col="theta_pred_deg",
                                     flip_data=False,
                                     clip=True,
                                     tol_deg=1e-3,
                                     max_iters=30):
        """
        Convert an existing normalized 0–1 column to predicted angle (deg) using r_hand.
        Appends results to self.data[out_col].
        """
        import numpy as np
        import pandas as pd

        if self.data is None:
            raise ValueError("No data loaded.")
        if value_col not in self.data.columns:
            raise KeyError(f"'{value_col}' not found in self.data.")

        y = pd.to_numeric(self.data[value_col], errors="coerce").to_numpy(dtype=float)
        if flip_data:
            y = 1.0 - y
        if clip:
            y = np.clip(y, 0.0, 1.0)

        theta_rad = np.array([
            self.theta_from_y(v, r=r_hand, L=L, tol_deg=tol_deg, max_iters=max_iters)
            if np.isfinite(v) else np.nan
            for v in y
        ])
        self.data[out_col] = np.rad2deg(theta_rad)
        return self.data[out_col]

    def fig_1_lin_vs_quad(self, perc_train=0.8, random_state=None,
                          data_color='blue', lin_color='red', quad_color='green',
                          data_label='Test Data', lin_label='Linear Fit', quad_label='Quadratic Fit'):
        """
        Plots test data from a single train/test split along with both a linear and a quadratic model fit,
        with Rotary Encoder angle on the x-axis and ADC Value on the y-axis.

        The models are trained on the training data from the split, then evaluated on the test data.
        The R² values for each model (evaluated on the test set) are included in the legend.

        Assumes:
          - self.data is a Pandas DataFrame containing 'ADC Value' and 'Rotary Encoder'.
          - self.adc_normalized is True.
        """
        # Check that data is available and normalized
        if self.data is None:
            raise ValueError("Data not available. Please read the data first.")
        if not self.adc_normalized:
            raise ValueError("ADC data not normalized. Please normalize data first.")

        # Create a train/test split (mimicking the first iteration)
        dataTrain, dataTest = train_test_split(self.data, test_size=(1 - perc_train),
                                               shuffle=True, random_state=random_state)

        # For plotting: use Rotary Encoder angle as x and ADC Value as y
        X_train = dataTrain[['Rotary Encoder']]
        y_train = dataTrain['ADC Value']
        X_test = dataTest[['Rotary Encoder']]
        y_test = dataTest['ADC Value']

        # ---- Linear Model ----
        model_linear = LinearRegression()
        model_linear.fit(X_train, y_train)
        y_pred_linear = model_linear.predict(X_test)
        ss_res_linear = np.sum((y_test - y_pred_linear) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r2_linear = 1 - (ss_res_linear / ss_tot)

        # ---- Quadratic Model ----
        # Use np.polyfit to fit a quadratic model on the training data
        x_train_vals = X_train.values.flatten()  # Rotary Encoder angles (1D array)
        y_train_vals = y_train.values  # ADC Values
        p_quad = np.polyfit(x_train_vals, y_train_vals, 2)
        # Predict on the test data using the quadratic model
        x_test_vals = X_test.values.flatten()
        y_pred_quad = np.polyval(p_quad, x_test_vals)
        ss_res_quad = np.sum((y_test - y_pred_quad) ** 2)
        r2_quad = 1 - (ss_res_quad / ss_tot)

        # ---- Prepare smooth curves for plotting the model fits ----
        x_range = np.linspace(np.min(X_test.values), np.max(X_test.values), 200)
        y_range_linear = model_linear.predict(x_range.reshape(-1, 1))
        y_range_quad = np.polyval(p_quad, x_range)

        # ---- Plotting ----
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(X_test, y_test, color=data_color, label=data_label)
        ax.plot(x_range, y_range_linear, '--', color=lin_color,
                label=f'{lin_label} (R² = {r2_linear:.3f})')
        ax.plot(x_range, y_range_quad, '-.', color=quad_color,
                label=f'{quad_label} (R² = {r2_quad:.3f})')

        ax.set_xlabel('Rotary Encoder Angle')
        ax.set_ylabel('Normalized ADC \n %s' % self.normalize_type)  # state normalization type
        ax.set_title('Test Data with Linear vs Quadratic Model Fits')
        ax.legend()

        return ax

    def plot_trained_model_on_existing(self, ax=None, title=None):
        """
        Overlays the trained model predictions on an existing plot (from plot_data).

        - Requires that `plot_data()` has already been run and `ax` is provided.
        - Requires that `train_model_test_accuracy()` has been run first.

        Parameters:
        - ax (matplotlib.axes.Axes, optional): The existing plot to overlay the model curve.
        - title (str): Title of the plot (optional).

        If ax is not provided, it uses the stored `self.data_ax` from plot_data().
        """

        # Ensure data is available
        if self.data is None:
            raise ValueError("No data loaded. Please load data first using load_data().")

        # Ensure ADC values are normalized before plotting
        if not self.adc_normalized:
            raise ValueError(
                "ADC data is not normalized. Please normalize it using normalize_adc_over_R0() or normalize_adc_bw_01().")

        # Ensure model is trained
        if self.model is None:
            raise ValueError("Model has not been trained yet. Please run train_model_test_accuracy first.")

        # Use existing plot if no ax is provided
        if ax is None:
            if not hasattr(self, 'data_ax') or self.data_ax is None:
                raise ValueError("No existing plot found. Please run plot_data() first or provide an ax object.")
            ax = self.data_ax  # Use the stored axis from plot_data()

        # Generate model predictions across a range of ADC values
        adc_values = np.linspace(self.data['ADC Value'].min(), self.data['ADC Value'].max(), 100).reshape(-1, 1)
        adc_values = np.hstack((adc_values, np.ones(adc_values.shape)))  # Add intercept term

        # Predict angles using the trained model
        predicted_angles = self.model.predict(adc_values)

        # Plot the model's predicted curve on the existing ax
        ax.plot(predicted_angles, adc_values[:, 0], color='red', linestyle='--', linewidth=2, label="Trained Linear Model")

        # Update title if necessary
        ax.set_title(title)

        # Add legend if not already present
        ax.legend()

        # Show the updated plot
        plt.show()

    @staticmethod
    def _coerce_to_df_sequence(obj, prefer_first_in_tuple=True):
        """
        Turn obj into a list of DataFrames so index selection actually changes data.
        Handles: DataFrame, list/tuple of DFs, list/tuple of (points_df, ...),
                 dicts of any of the above.
        """
        import pandas as pd

        def pick_df(item):
            # if tuple/list like (points_df, means_df) → pick the first by default
            if isinstance(item, (list, tuple)) and item:
                return item[0] if prefer_first_in_tuple else item[-1]
            return item

        # Single DF → [DF]
        if isinstance(obj, pd.DataFrame):
            return [obj]

        # List/tuple → flatten one level selecting first elem if tuples
        if isinstance(obj, (list, tuple)):
            out = []
            for it in obj:
                it = pick_df(it)
                if isinstance(it, pd.DataFrame):
                    out.append(it)
            return out

        # Dict → look through values
        if isinstance(obj, dict):
            out = []
            for v in obj.values():
                if isinstance(v, (list, tuple)):
                    for it in v:
                        it = pick_df(it)
                        if hasattr(it, "columns"):
                            out.append(it)
                elif hasattr(v, "columns"):
                    out.append(v)
            return out

        # Fallback: nothing usable
        return []

    def cross_validation_angular_error(self, degree=1):
        """
        Performs 10-fold cross-validation using polynomial regression of specified degree.
        - Each fold uses 9/10 of the data for training and 1/10 for testing.
        - Computes the mean and standard deviation of the angular error.

        Args:
            degree (int): Degree of the polynomial for fitting (1=linear, 2=quadratic, etc.)

        Returns:
            mean_error (float): Mean of absolute angular errors.
            std_error (float): Standard deviation of angular errors.
            predictions_df (pd.DataFrame): DataFrame with actual and predicted angles.
        """

        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        if not self.adc_normalized:
            raise ValueError("ADC data is not normalized. Please normalize it first.")

        from sklearn.linear_model import LinearRegression
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.pipeline import make_pipeline

        N = len(self.data)
        num_splits = 10
        indices = np.arange(N)
        np.random.shuffle(indices)
        split_size = N // num_splits

        all_predictions = np.zeros(N)
        all_errors = np.zeros(N)

        for i in range(num_splits):
            if i < (num_splits - 1):
                test_indices = indices[i * split_size:(i + 1) * split_size]
            else:
                test_indices = indices[i * split_size:]

            train_indices = np.setdiff1d(indices, test_indices)

            train_data = self.data.iloc[train_indices]
            test_data = self.data.iloc[test_indices]

            X_train = train_data['ADC Value'].values.reshape(-1, 1)
            y_train = train_data['Rotary Encoder'].values
            X_test = test_data['ADC Value'].values.reshape(-1, 1)

            # Polynomial model of given degree
            model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            all_predictions[test_indices] = y_pred
            all_errors[test_indices] = np.abs(y_pred - test_data['Rotary Encoder'].values)

        mean_error = np.mean(all_errors)
        std_error = np.std(all_errors)

        predictions_df = self.data.copy()
        predictions_df['Predicted Angle'] = all_predictions
        predictions_df['Absolute Error'] = all_errors

        return mean_error, std_error, predictions_df




    def cross_validation_external_test(self, external_datasets, n_splits=10):
        """
        Cross-validation method where:
        - One dataset is split into 10 parts (9N/10 for training, N/10 for testing).
        - The trained model is used to predict on the concatenated external datasets.
        - This process is repeated 10 times to compute mean and std errors.

        Parameters:
            external_datasets (list of bender_class instances): Other datasets to test on.
            n_splits (int): Number of cross-validation splits (default: 10).

        Returns:
            tuple: (mean_error, std_error, predictions_df)
                   - mean_error: Mean of absolute angular errors.
                   - std_error: Standard deviation of absolute angular errors.
                   - predictions_df: DataFrame with true angles and predicted angles.
        """

        if self.data is None:
            raise ValueError("No data loaded. Please load data first.")

        if self.adc_normalized is False:
            raise ValueError("ADC data must be normalized before running cross-validation.")

        # Ensure external datasets are loaded and normalized
        external_data = []
        for dataset in external_datasets:
            if dataset.data is None:
                raise ValueError("One of the external datasets has no data loaded.")
            if dataset.adc_normalized is False:
                raise ValueError("All external datasets must be normalized before testing.")
            external_data.append(dataset.data)

        # Concatenate external datasets
        external_df = pd.concat(external_data, ignore_index=True)

        # Prepare K-Fold cross-validation
        kf = KFold(n_splits=n_splits, shuffle=True)
        errors = []

        predictions_list = []

        for train_idx, test_idx in kf.split(self.data):
            train_data = self.data.iloc[train_idx]
            test_data = self.data.iloc[test_idx]  # Held-out test set (N/10)

            # Train the model using the 9N/10 split
            X_train = train_data['ADC Value'].values.reshape(-1, 1)
            X_train = np.hstack((X_train, np.ones(X_train.shape)))  # Add intercept
            y_train = train_data['Rotary Encoder'].values

            self.model = LinearRegression()
            self.model.fit(X_train, y_train)

            # Predict on the concatenated external dataset
            X_external = external_df['ADC Value'].values.reshape(-1, 1)
            X_external = np.hstack((X_external, np.ones(X_external.shape)))  # Add intercept
            y_true_external = external_df['Rotary Encoder'].values  # True external angles

            y_pred_external = self.model.predict(X_external)

            # Compute absolute error for this split
            abs_errors = np.abs(y_pred_external - y_true_external)
            errors.append(abs_errors)

            # Store predictions
            predictions_df = pd.DataFrame({
                'True Angle': y_true_external,
                'Predicted Angle': y_pred_external,
                'Absolute Error': abs_errors
            })
            predictions_list.append(predictions_df)

        # Combine all predictions into one DataFrame
        final_predictions_df = pd.concat(predictions_list, ignore_index=True)

        # Compute mean and std error
        mean_error = np.mean([np.mean(err) for err in errors])
        std_error = np.std([np.mean(err) for err in errors])

        return mean_error, std_error, final_predictions_df

    def plot_error_violin(self, error_dfs, labels=None):
        """
        Creates side-by-side violin plots for angular prediction errors from multiple datasets.

        Parameters:
            error_dfs (list of pd.DataFrame): List of DataFrames, each containing an 'Absolute Error' column.
            labels (list of str, optional): List of labels corresponding to each dataset. Default is None.

        Raises:
            ValueError: If any DataFrame does not contain the expected column.
        """

        # Check if input is a list of DataFrames
        if not isinstance(error_dfs, list) or not all(isinstance(df, pd.DataFrame) for df in error_dfs):
            raise ValueError("error_dfs must be a list of pandas DataFrames.")

        # Ensure all DataFrames contain the 'Absolute Error' column
        for df in error_dfs:
            if "Absolute Error" not in df.columns:
                raise ValueError("Each DataFrame must contain an 'Absolute Error' column.")

        # Create a combined DataFrame for plotting
        plot_data = []
        dataset_labels = []

        for i, df in enumerate(error_dfs):
            plot_data.extend(df["Absolute Error"].tolist())  # Store error values
            dataset_label = labels[i] if labels and i < len(labels) else f"Dataset {i + 1}"
            dataset_labels.extend([dataset_label] * len(df))

        # Convert to DataFrame
        plot_df = pd.DataFrame({"Absolute Error": plot_data, "Dataset": dataset_labels})

        # Create the violin plot
        plt.figure(figsize=(10, 6))
        sns.violinplot(x="Dataset", y="Absolute Error", data=plot_df, inner="point", palette="coolwarm")

        # Labels and title
        plt.xlabel("Datasets")
        plt.ylabel("Absolute Angular Error (degrees)")
        plt.title("Distribution of Angular Prediction Errors Across Datasets")
        plt.ylim(0, 15)
        # Show the plot
        plt.show()

    def plot_box_plot(self, data_dict, group_dict, group_colors, group_names,
                      box_alpha=0.5, data_alpha=0.5, jitter=0.2):
        """
        Plot box plots for each sample but group them visually using custom group labels and colors.

        Parameters:
        - data_dict: A dictionary where keys are sample names and values are lists of data (absolute errors).
        - group_dict: A dictionary where keys are sample names and values are group labels.
        - group_colors: A list of colors corresponding to each group.
        - group_names: A list of custom group labels to display on the x-axis.
        - box_alpha: Transparency level for the box plots (default=0.5).
        - data_alpha: Transparency level for individual data points (default=0.5).
        - jitter: Jitter amount for individual data points (default=0.2).

        The function plots each sample separately but assigns colors based on their groups.
        """

        # Convert data_dict into a DataFrame for easier plotting
        all_samples = []
        all_values = []
        all_groups = []

        for sample, values in data_dict.items():
            all_samples.extend([sample] * len(values))  # Repeat sample name for each data point
            all_values.extend(values)  # Store values
            all_groups.extend([group_dict[sample]] * len(values))  # Assign groups

        df = pd.DataFrame({'Sample': all_samples, 'Value': all_values, 'Group': all_groups})

        # Ensure group order is consistent and assign colors
        unique_groups = sorted(set(group_dict.values()))  # Extract unique groups
        if len(unique_groups) != len(group_names):
            raise ValueError("Length of group_names must match the number of unique groups.")

        group_palette = {group: color for group, color in zip(unique_groups, group_colors)}

        # Get sample order
        sample_order = list(df['Sample'].unique())  # Order of samples as they appear

        # Create figure
        plt.figure(figsize=(12, 6))

        # Box plot with adjustable transparency
        ax = sns.boxplot(
            x='Sample', y='Value', data=df,
            palette=[group_palette[group_dict[sample]] for sample in sample_order],
            boxprops=dict(alpha=box_alpha),  # Set box transparency
            medianprops=dict(alpha=box_alpha),
            whiskerprops=dict(alpha=box_alpha),
            capprops=dict(alpha=box_alpha),
            flierprops=dict(marker='')  # Remove outlier markers
        )

        # Add jittered individual data points with adjustable transparency
        sns.stripplot(
            x='Sample', y='Value', data=df, hue='Group', palette=group_palette,
            jitter=jitter, alpha=data_alpha, dodge=False, size=4, edgecolor='w', linewidth=0.5
        )

        # Modify x-axis to show **custom** group labels instead of sample labels
        group_positions = [
            np.mean([sample_order.index(sample) for sample in df[df['Group'] == group]['Sample'].unique()])
            for group in unique_groups
        ]
        plt.xticks(group_positions, group_names, rotation=45, ha='right')  # Use custom group names

        # Remove legend (since x-axis already shows groups)
        plt.legend([], [], frameon=False)

        # Improve layout

        plt.xlabel('Group')
        plt.ylabel('Absolute Angular Error (deg)')
        plt.title('Box Plot of Absolute Errors by Group')
        plt.ylim(0, 7)  # Change the range as needed

        plt.savefig("box plot", dpi=300, bbox_inches='tight')

        plt.show()

    def accuracy_by_angle(self, y_true, y_pred, angle_accuracy):
        '''
        Method to calculate the accuracy of the model for specific thresholds of angle accuracy
        '''

        # Accuracy determined by finding the number of test data that predicts
        # actual angle correctly to within +/- deg_accuracy
        pos = np.sum(np.abs(y_true - y_pred) < angle_accuracy)
        total = len(y_true)

        return 100. * pos / total

    def plot_accuracy(self, accuracy = None, title=''):
       
        if not hasattr(self, 'accuracy'):
            assert(accuracy is not None)
        
        if accuracy is not None:
            pass
        else:
            accuracy = self.accuracy

        f, ax = plt.subplots()

        # Plotting accuracy
        ax.plot(self.accuracy_angle, np.mean(accuracy, axis=0), 'b-', label='Accuracy')
        ax.errorbar(self.accuracy_angle, np.mean(accuracy, axis=0), np.std(accuracy, axis=0), marker='|', color='k')
        ax.set_xlabel('Angle Accuracy (degrees)')
        ax.set_ylabel('Percent Accurate (held out data)')
        ax.set_title(title)
        ax.set_ylim([0, 100])

    def get_min_accuracy_100(self, accuracy_matrix=None):
        """
        Finds the smallest angle threshold where the mean accuracy reaches 100%.

        Parameters:
            accuracy_matrix (np.array, optional): If provided, uses this external accuracy data instead of self.accuracy.

        Returns:
            tuple: (min_angle_100, accuracy_value)
                   where min_angle_100 is the smallest angle where accuracy is 100%.
        """

        # Determine which accuracy matrix to use
        if accuracy_matrix is None:
            if not hasattr(self, 'accuracy') or self.accuracy is None:
                raise ValueError("Accuracy has not been computed. Please run train_model_test_accuracy first.")
            accuracy_matrix = self.accuracy  # Default to self.accuracy: niter x nangles

        # Compute mean accuracy across runs
        mean_accuracy = np.mean(accuracy_matrix, axis=0)

        # Also report list of all accuracies: 
        all_min_angle_100 = []
        for i in range(accuracy_matrix.shape[0]):
            ix_tmp = np.where(accuracy_matrix[i, :] == 100)[0]
            if len(ix_tmp) > 0: 
                all_min_angle_100.append(self.accuracy_angle[ix_tmp[0]]) # lowest value
            else: 
                pass

        # Find the first occurrence where accuracy reaches > 99%
        indices_100 = np.where(mean_accuracy == 100)[0]

        if len(indices_100) == 0:
            print("No angle threshold reaches 100% accuracy.")
            return None, []

        # Get the smallest angle threshold where accuracy is 100%
        min_angle_100 = self.accuracy_angle[indices_100[0]]
        accuracy_value = mean_accuracy[indices_100[0]]
        assert(accuracy_value == 100) # by definition this has to be 100

        return min_angle_100, all_min_angle_100

    def plot_bar_chart(self, data, labels, title, ylabel, colors, ylim=None):
        """
        Plots grouped bars with different colors for each group.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker

        num_groups = len(data)
        max_bars = max(len(group) for group in data)
        bar_width = 0.2

        x_positions = np.arange(num_groups)

        plt.figure(figsize=(8, 6))

        for i, (group_values, color) in enumerate(zip(data, colors)):
            num_bars = len(group_values)
            x_offsets = np.linspace(-bar_width * (num_bars - 1) / 2,
                                    bar_width * (num_bars - 1) / 2, num_bars)

            for j, (val, offset) in enumerate(zip(group_values, x_offsets)):
                if isinstance(val, float):
                    plt.bar(x_positions[i] + offset, val, width=bar_width,
                            color=color, label=f"Sample {j + 1}" if i == 0 else "")
                elif isinstance(val, list):
                    vals = np.array(val)
                    plt.bar(x_positions[i] + offset, np.mean(vals),
                            width=bar_width, color=color,
                            label=f"Sample {j + 1}" if i == 0 else "",
                            yerr=np.std(vals), capsize=5,
                            error_kw={'elinewidth': 1.5})

        # X labels/titles
        plt.xticks(x_positions, labels)
        plt.ylabel(ylabel)
        plt.title(title)

        if ylim:
            plt.ylim(ylim)

        ax = plt.gca()
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.tick_params(axis='both', which='major', labelsize=20)

        # ---- Force y-ticks every 1 unit (robust) ----
        ax.yaxis.set_major_locator(mticker.MultipleLocator(1.0))
        # Optional: nicer integer formatting (no decimals)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d'))
        # Optional: minor ticks between majors
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))

        plt.savefig("min_angle_bar_chart.svg", dpi=300, bbox_inches='tight')
        plt.show()

    def plot_double_bar_chart(self, list1, list2, labels,
                              std1=None, std2=None,
                              title1="RMSE per Sample", title2="Min Angle for 100% Accuracy",
                              ylabel1="RMSE (deg)", ylabel2="Min Angle (deg)",
                              color1='b', color2='r', ylim1=(0, 15), ylim2=(0, 15)):
        """
        Creates two stacked bar plots for visualizing RMSE and Min Angle where Mean Accuracy = 100,
        with optional error bars.

        Parameters:
        - list1 (list): Data for the first bar plot (e.g., RMSE values).
        - list2 (list): Data for the second bar plot (e.g., min angle where accuracy is 100%).
        - labels (list): Labels for the bars (should be the same length as list1 and list2).
        - std1 (list, optional): Standard deviations for the first bar plot (RMSE) (default: None).
        - std2 (list, optional): Standard deviations for the second bar plot (Min Angle) (default: None).
        - title1 (str): Title for the first bar plot.
        - title2 (str): Title for the second bar plot.
        - ylabel1 (str): Y-axis label for the first plot.
        - ylabel2 (str): Y-axis label for the second plot.
        - color1 (str): Color for the first bar plot (default: 'b' for blue).
        - color2 (str): Color for the second bar plot (default: 'r' for red).
        - ylim1 (tuple): Y-axis limits for the first plot (default: (0, 15)).
        - ylim2 (tuple): Y-axis limits for the second plot (default: (0, 15)).
        """

        # Ensure input lists have the same length
        if not (len(list1) == len(list2) == len(labels)):
            raise ValueError("list1, list2, and labels must have the same length.")

        # Convert std to None if not provided to avoid errors in error bars
        if std1 is None:
            std1 = [0] * len(list1)  # No error bars if std1 is not provided
        if std2 is None:
            std2 = [0] * len(list2)  # No error bars if std2 is not provided

        # Ensure std lists are the same length as data lists
        if not (len(std1) == len(list1) and len(std2) == len(list2)):
            raise ValueError("Standard deviation lists must have the same length as their corresponding data lists.")

        # Set figure size
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        # Define bar positions
        x_pos = np.arange(len(labels))

        # Create first bar plot (RMSE)
        axes[0].bar(x_pos, list1, color=color1, alpha=0.7, yerr=std1, capsize=5, error_kw={'elinewidth': 1.5})
        axes[0].set_ylabel(ylabel1)
        axes[0].set_title(title1)
        axes[0].set_ylim(ylim1)  # Customizable y-axis limit
        axes[0].grid(axis='y', linestyle='--', alpha=0.6)  # Adds grid lines for readability

        # Create second bar plot (Min Angle where Accuracy = 100)
        axes[1].bar(x_pos, list2, color=color2, alpha=0.7, yerr=std2, capsize=5, error_kw={'elinewidth': 1.5})
        axes[1].set_xlabel("Sample")
        axes[1].set_ylabel(ylabel2)
        axes[1].set_title(title2)
        axes[1].set_ylim(ylim2)  # Customizable y-axis limit
        axes[1].grid(axis='y', linestyle='--', alpha=0.6)

        # Set x-axis labels
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels(labels, rotation=45, ha="right")

        # Adjust layout
        plt.tight_layout()
        plt.show()

    def predict_new_data(self, new_data_df):
        """
        Uses result from trained model method from single dataset/test to make predictions on new data/test from another bender_class object
        and calculates the accuracy of those predictions 
        """
        # Ensure that the model has been trained
        if self.model is None: 
            raise Exception("Model has not been trained yet. Please run the train_test method first.")
        
        # Prepare new data for prediction
        #X_new = new_data_df['ADC Value'].values.reshape(-1, 1) #-1 * new_data.iloc[:, 2].values.reshape(-1, 1)  # Assuming the same structure
        #X_new = np.hstack((X_new, np.ones((X_new.shape[0], 1))))

        # Prepare new data for prediction using the same transformation
        X_new_raw = new_data_df['ADC Value'].values.reshape(-1, 1)
        X_new = self.poly.transform(X_new_raw)
        y_new = new_data_df['Rotary Encoder'].values  # Assuming the same structure

        # Predict using the trained model
        y_pred_new = self.model.predict(X_new)

        # Calculate accuracy based on the setpoint
        # Assuming the actual angles are in the fourth column of the new data
        accuracy = np.zeros((len(self.accuracy_angle)))
        for j, angle_accuracy in enumerate(self.accuracy_angle):
            accuracy[j] = self.accuracy_by_angle(y_new, y_pred_new, angle_accuracy)

        abs_error = np.abs(y_new - y_pred_new)

        return accuracy, abs_error

    def plot_combined_accuracy(self, title='Combined Accuracy vs Angle'):
        """
        Combine all accuracy plots into one showing average accuracy and standard deviation as a plot with error bars
        """
        if not self.all_accuracies or len(self.all_accuracies) == 0:
            raise ValueError("No accuracy data available. Train and test the model first.")

        # Concatenate all accuracy arrays
        all_accuracies_combined = np.vstack(self.all_accuracies)  # Shape: (runs * niter, len(accuracy_angle))

        # Calculate mean and standard deviation
        mean_accuracy = np.mean(all_accuracies_combined, axis=0)
        std_dev = np.std(all_accuracies_combined, axis=0)

        # Plot
        f, ax = plt.subplots()
        accuracy_angle = self.accuracy_angle  # Use angle thresholds from the class
        ax.plot(self.accuracy_angle, mean_accuracy, 'k-', label='Average Accuracy')  # Mean accuracy as a blue line
        ax.errorbar(self.accuracy_angle, mean_accuracy, std_dev,  marker='|', color='k')  # Mean + STD
        ax.set_xlabel('Angle Accuracy (degrees)')
        ax.set_ylabel('Percent Accurate')
        ax.set_ylim([0, 100])
        ax.set_title(title)
        ax.legend()
        plt.tight_layout()
        plt.show()

    def dynamic_data(self, period):
        """
        Code used to extract timestamp info from dynamic autobending test.  After plate would bend 90 deg,
        'period' seconds wait before collecting 15 data points. Same when going back to 0 degrees.
        Test was conducted over 100 cycles.

        """
        # Add timestamp column
        num_rows = self.data.shape[0]  # Get the number of rows
        timestamps = pd.Series(range(num_rows)) * period  # 0.3 seconds (300 ms)
        self.data['Timestamp'] = timestamps

        # Ensure no mismatch in lengths
        self.data['ADC Value'] = self.data['ADC Value'].dropna()  # Remove NaN values in the y-data
        self.data['Timestamp'] = self.data['Timestamp'][:len(self.data['ADC Value'])]  # Match the lengths of x and y

    def plot_dynamic(self, time):

        """
        Code used to plot data from dynamic autobending test.
        'time' is the time domain to plot in 2nd subplot.

        """

        # Create subplots (1 row, 2 columns)
        fig, ax = plt.subplots(1, 2, figsize=(15, 5))

        # Plot the full range on the first subplot
        ax[0].plot(self.data['Timestamp'], self.data['ADC Value'], 'b')
        ax[0].set_xlabel('Time (sec)')
        ax[0].set_ylabel('$\Delta R/R_o$')
        ax[0].set_title('Full Plot')

        # Zoomed-in plot for time range between 0 and 4 seconds
        ax[1].plot(self.data['Timestamp'], self.data['ADC Value'], 'b')
        ax[1].set_xlim(0, time)  # Set x-axis to zoom between 0 and 4 seconds
        ax[1].set_xlabel('Time (sec)')
        ax[1].set_ylabel('$\Delta R/R_o$')
        ax[1].set_title('Zoomed-in Plot')

        # Display the subplots
        plt.tight_layout()  # Adjust the spacing
        plt.show()


class original_bender_class:  
    
    def __init__(self, data):
        
        # Ensure data is a pandas DataFrame
        if not isinstance(data, pd.DataFrame):
            raise TypeError("Expected 'data' to be a pandas DataFrame.")
        
        self.data = data #dataframe containing data from all csv files analyzed -> m rows by 4 columns
        self.acc = None  #accuracy from quadratic curve fitting class method:  quadriatic_fit(self)
        self.model = None  # To store the trained model
        self.poly_features = None  # To store polynomial features
        
    def __str__(self):
        """
        human-readable, or informal, string representation of object
        """
        return (f"Bender Class: \n"
            f"  Number of data points: {self.data.shape[0] if self.data is not None else 0}\n"
            f"  Number of features: {self.data.shape[1] if self.data is not None else 0}\n"
            f"  Current Accuracy: {self.acc:.2f}% if self.acc is not None else 'N/A'\n")
    
    def __repr__(self):
        """
        more information-rich, or official, string representation of an object
        """
        return (f"Bender_class(data={self.data.head() if self.data is not None else 'None'}, "
            f"acc={self.acc}, "
            f"model={self.model.__class__.__name__ if self.model else 'None'}, "
            f"poly_features={self.poly_features.__class__.__name__ if self.poly_features else 'None'})")
    
    def read_data(self, path):
        """
        NOT USED IN LATEST ANALYSIS:  class method to extract all csv files in path and concatenate data in pandas dataframe
        class method also normalizes ADC around 0 and converts rotary encoder angle to deg, then 
        shifts angles to start at 0 deg
        """
        
        # use glob to get all the csv files
        # in the folder
        csv_files = glob.glob(path)
        
        if not isinstance(path, str):
            raise TypeError("Expected 'path' to be a string.")
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in the specified path: {path}")
        
        # Initialize an empty list to hold the DataFrames
        dataframes = []

        # loop over the list of csv files
        for f in csv_files:
                  
            try:
                # Read the csv file
                df = pd.read_csv(f)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                continue
            
            # Check if the DataFrame has exactly 4 columns
            if df.shape[1] != 4:
                raise ValueError(f"Error: The file {f} does not contain exactly 4 columns. It has {df.shape[1]} columns.")
            
            # Remove rows where the first column equals 100
            df = df[df.iloc[:, 0] != 100]
            df = df[df.iloc[:, 1] < 95]
            
            # center ADC values around 0 (normalize ADC values)
            #df.iloc[:, 2] = (df.iloc[:, 2] - df.iloc[1, 2]) / df.iloc[1, 2]

            
            #Changes made 10/14/2024:  Normalization of data from 0 -> 1:  
            #https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.MinMaxScaler.html
            
            y = df.iloc[:, 2].values #returns a numpy array
            y = y.reshape(-1, 1)
            # Initialize the MinMaxScaler
            min_max_scaler = preprocessing.MinMaxScaler()
            # Fit and transform the data
            y_scaled = min_max_scaler.fit_transform(y)
            # Update the DataFrame with scaled values
            df.iloc[:, 2] = y_scaled.flatten()  # Flatten back to 1D for assignment
                
            #convert rotary encoder to angle (deg) -> ADC is arduino Uno 10 bit
            df.iloc[:, 3] = df.iloc[:, 3] * 320 / 1024

            #shift rotary encoder angles to start tests at 0 deg
            df.iloc[:, 3] = df.iloc[:, 3] - df.iloc[1, 3]

            

            # Append the DataFrame to the list
            dataframes.append(df)
            
         # Concatenate all DataFrames in the list into a single DataFrame
        self.data = pd.concat(dataframes, ignore_index=True)
        
        return self.data
    
    def read_data_norm(self, path):
        """
        Delta R / R_o  method: extracts all csv files in path and concatenate data in pandas dataframe
        class method also normalizes ADC around 0 and converts rotary encoder angle to deg, then 
        shifts angles to start at 0 deg
        """
    
        # use glob to get all the csv files in the folder
        csv_files = glob.glob(path)
    
        if not isinstance(path, str):
            raise TypeError("Expected 'path' to be a string.")
        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in the specified path: {path}")
    
        # Initialize an empty list to hold the DataFrames
        dataframes = []

        # loop over the list of csv files
        for f in csv_files:
              
            try:
                # Read the csv file
                df = pd.read_csv(f)
            except Exception as e:
                print(f"Error reading {f}: {e}")
                continue
        
            # Check if the DataFrame has exactly 4 columns
            if df.shape[1] != 4:
                raise ValueError(f"Error: The file {f} does not contain exactly 4 columns. It has {df.shape[1]} columns.")
        
            # Remove rows where the first column equals 100
            df = df[df.iloc[:, 0] != 100]
       
                
        
            df = df[df.iloc[:, 1] > -93]
            # Ensure slopes are positive
            df.iloc[:, 2] = df.iloc[:, 2].abs()
        
            # Normalize column 2
            y = df.iloc[:, 2].values # Returns a numpy array
            min_value = np.min(y)  # Find the smallest value in column 2
            y_change = (y - min_value) / min_value  # Calculate relative change from the smallest value
        
            # Update the DataFrame with the calculated change
            df.iloc[:, 2] = y_change  # Replace original values with relative change
        
            # Convert rotary encoder to angle (deg)
            df.iloc[:, 3] = df.iloc[:, 3] * 320 / 1024

            # Shift rotary encoder angles to start tests at 0 deg
            df.iloc[:, 3] = df.iloc[:, 3] - df.iloc[1, 3]

            # Append the DataFrame to the list
            dataframes.append(df)
    
        # Concatenate all DataFrames in the list into a single DataFrame
        self.data = pd.concat(dataframes, ignore_index=True)
    
        # Plot the change in column 2 over the smallest value
        #plt.plot(self.data.iloc[:, 3], self.data.iloc[:, 2], marker='o', label='Change in Column 2')
        #plt.xlabel('Rotary Encoder Angle (deg)')
        #plt.ylabel('$\Delta R/R_o$')
        #plt.legend()
        #plt.grid()
        #plt.show()
    
        return self.data
    
    def read_data_2(self, path):
        """
        MinMax normalization method:  Method extracts all csv files in path and concatenate data in pandas dataframe.
        Y-axis data range from 0 to 1, converts rotary encoder angle to degrees.  Data also drops NaN rows and has the potential to get rid of 
        very extreme data points (looks like a handful in each dataset)!
        """

        # Use glob to get all the csv files in the folder
        csv_files = glob.glob(path)
        print(f"Found {len(csv_files)} CSV files.")  # Debug statement
    
        # Initialize an empty list to hold the DataFrames
        dataframes = []

        # Loop over the list of csv files
        for f in csv_files:
            df = pd.read_csv(f)
        
            # Remove rows where the first column equals 100
            df = df[df.iloc[:, 0] != 100]
            
                    
            # Remove rows with any NaN values
            df = df.dropna()
        
            # center ADC values around 0 (normalize ADC values)
            #df.iloc[:, 2] = (df.iloc[:, 2] - df.iloc[15, 2]) / df.iloc[1, 2]
        
             # Ensure slopes are positive
            df.iloc[:, 2] = df.iloc[:, 2].abs()
            
            #Changes made 10/14/2024:  Normalization of data from 0 -> 1:  
            #https://scikit-learn.org/1.5/modules/generated/sklearn.preprocessing.MinMaxScaler.html
            
            y = df.iloc[:, 2].values #returns a numpy array
            y = y.reshape(-1, 1)
            # Initialize the MinMaxScaler
            min_max_scaler = preprocessing.MinMaxScaler()
            # Fit and transform the data
            y_scaled = min_max_scaler.fit_transform(y)
            # Update the DataFrame with scaled values
            df.iloc[:, 2] = y_scaled.flatten()  # Flatten back to 1D for assignment
                
        
            # Convert rotary encoder to angle (degrees) -> ADC is Arduino Uno 10 bit
            df.iloc[:, 3] = df.iloc[:, 3] * 320 / 1024
        
            # Shift rotary encoder angles to start tests at 0 degrees
            initial_angle = df.iloc[20, 3]  # Save initial angle for later adjustment
            df.iloc[:, 3] = df.iloc[:, 3] - initial_angle 
            df = df[df.iloc[:, 3] > -93]
                 
            # Append the DataFrame to the list
            dataframes.append(df)

        # Concatenate all DataFrames in the list into a single DataFrame
        self.data = pd.concat(dataframes, ignore_index=True)

        return self.data
    
    
    
    def plot_data(self):
        """
        class method to plot normalized ADC values vs Rotary Encoder angles (blue dots) AND IMU angles (red dots)
        """
        if self.data is None:
            raise ValueError("Data not available. Please read the data first.")
        
        # Plotting Rotary Encoder data
        plt.plot(-1 * self.data.iloc[:, 3], self.data.iloc[:, 2], 'bo', label='Rotary Encoder')  # Blue circles for Rotary Encoder
        
        
        # Plotting IMU data
        #plt.plot(-1 * self.data.iloc[:, 1], self.data.iloc[:, 2], 'ro', label='IMU')  # Red circles for IMU
        #plt.yscale("log")
        #plt.xscale("log")
        
        # Setting labels
        plt.xlabel('Angle (deg)')
        plt.ylabel('MinMax(ADC)')

        # Adding legend
        plt.legend()
        plt.show()
        
        return
    
    
    def model_data(self, thick, l_ch, l_sam, area, res):
        """
        Class method to plot normalized data (delta R / Ro) vs bend angle as well as theoretical curve  based on mechanics model .
        """
        if self.data is None:
            raise ValueError("Data not available. Please read the data first.")

        # Plot Rotary Encoder data
        fig, ax1 = plt.subplots()

        ax1.plot(-1 * self.data.iloc[:, 3], self.data.iloc[:, 2], 'bo', label='Rotary Encoder Data')  # Blue circles for Rotary Encoder    

        theta = np.arange(0, np.pi/2 + 0.1, 0.1)  # Include 90 by adding increment
        rho = 29.4 * 10**-8
        eps = (thick * 0.0254) * theta / (l_sam * 0.0254)
        dr = (rho * eps * (l_ch * 0.0254) * (8 - eps) / ((area * 0.000645) * (2 - eps)**2))
        drrt = dr / res
        ax1.plot(theta * 180 / np.pi, drrt, 'g', label='Theoretical Model')  

        # Setting up bottom axis
        ax1.set_xlabel('Angle (deg)')
        ax1.set_ylabel('$\Delta R/R_o$', color='b')
        ax1.tick_params(axis='y', labelcolor='b')
        ax1.legend(loc="upper left")

         # Define 5 evenly spaced tick positions for theta
        theta_ticks = np.linspace(0, np.pi/2, 5)  # 5 points from 0 to 90 degrees in radians
        eps_ticks = (thick * 0.0254 + 0.00635) * theta_ticks / (l_sam * 0.0254)  # Compute corresponding eps values

        # Create top x-axis for eps
        ax2 = ax1.twiny()  # Twin the x-axis to share the y-axis
        ax2.set_xlim(ax1.get_xlim())  # Synchronize with bottom x-axis

        ax1.set_xticks(theta_ticks * 180 / np.pi)  # Set bottom x-axis (theta) ticks in degrees
        ax2.set_xticks(theta_ticks * 180 / np.pi)  # Match top x-axis ticks to bottom x-axis
        ax2.set_xticklabels([f"{e:.2f}" for e in eps_ticks])  # Set top x-axis labels with eps values
        ax2.set_xlabel('$\epsilon$ (strain)')

        plt.tight_layout()
        plt.show()
        
    def train_test(self, deg_accuracy):
        """
        class method to determine how well a model that predicts angle based on normalized ADC value input up to +/- deg_accuracy 
        """
        
        if self.data is None:
            raise ValueError("Data not available. Please read the data first.")
            
               
        # Cross-validation (train: 80%, test: 20%)
        dataTrain, dataTest = train_test_split(self.data, test_size=0.2, random_state=None)

        # Fit a polynomial of degree X to the training data
        poly_features = PolynomialFeatures(degree=1)  # degree of 1 corresponds to linear fit, 2 would be quadratic
        X_train = -1 * dataTrain.iloc[:, 2].values.reshape(-1, 1)  # Reshapes the 1D array into a 2D array with one... 
        # column and as many rows as needed for compatibility with sklearn functions
        y_train = dataTrain.iloc[:, 3].values  # Converts Pandas Series to a NumPy array
        X_train_poly = poly_features.fit_transform(X_train)

        self.model = LinearRegression()
        self.model.fit(X_train_poly, y_train)
        self.poly_features = poly_features  # Store the polynomial features

        # Predicting using the test set
        X_test = -1 * dataTest.iloc[:, 2].values.reshape(-1, 1)  
        X_test_poly = self.poly_features.transform(X_test)
        y_test = self.model.predict(X_test_poly)

        # Accuracy determined by finding the number of test data that predicts
        # actual angle correctly to within +/- deg_accuracy
        pos = np.sum(np.abs(y_test - dataTest.iloc[:, 3].values) < deg_accuracy)  
        total = len(dataTest)

        # Calculate accuracy
        self.acc = pos * 100 / total if total > 0 else 0  # Avoid division by zero
        
         # Optionally print the accuracy
        #print(f'Accuracy: {self.acc:.2f}%')
        
        return self.acc
    
    def train_test_log(self, deg_accuracy):
        """
        Class method to determine how well a model predicts angle based on 
        normalized ADC value input up to +/- deg_accuracy using log-log fitting.
        """
    
        if self.data is None:
            raise ValueError("Data not available. Please read the data first.")
        
        # Cross-validation (train: 80%, test: 20%)
        dataTrain, dataTest = train_test_split(self.data, test_size=0.2, random_state=None)

        # Prepare the training data using log transformation
        # Ensure we add a small constant to avoid log(0)
        X_train = -1 * dataTrain.iloc[:, 3].values + 1e-10  # Normalized ADC values
        y_train = dataTrain.iloc[:, 2].values +  1e-10        # Rotary Encoder angles
        
              
        # Fit a linear model to the log-log data
        self.model = LinearRegression()
        self.model.fit(X_train.reshape(-1, 1), y_train)  # Reshape for sklearn

        # Prepare the test data for predictions
        X_test = -1 * dataTest.iloc[:, 3].values + 1e-10  # Normalized ADC values
        y_test = dataTest.iloc[:, 2].values + 1e-10        # Rotary Encoder angles
        
       

        # Apply log transformation
        #X_test_log = np.log10(X_test)
        y_test_log_pred = self.model.predict(X_test.reshape(-1, 1))

        # Convert predictions back to original scale
        y_test_pred = 10 ** y_test_log_pred

        # Accuracy determined by finding the number of test data that predicts
        # actual angle correctly to within +/- deg_accuracy
        pos = np.sum(np.abs(y_test_pred - dataTest.iloc[:, 3].values) < deg_accuracy)  
        total = len(dataTest)

        # Calculate accuracy
        self.acc = pos * 100 / total if total > 0 else 0  # Avoid division by zero
    
        return self.acc
    
    def predict_new_data(self, new_data_obj, deg_accuracy):
        """
        Uses result from trained model method from single dataset/test to make predictions on new data/test from another bender_class object
        and calculates the accuracy of those predictions based on a specified degree of accuracy. 

        :param new_data_obj: An instance of bender_class containing new data for predictions.
        :param deg_accuracy: The degree of accuracy within which the predictions are considered correct.
        :return: A float representing the accuracy of the predictions.
        """
        # Ensure that the model has been trained
        if self.model is None or self.poly_features is None:
            raise Exception("Model has not been trained yet. Please run the train_test method first.")
        
        
        # Retrieve new data
        new_data = new_data_obj.data

        # Prepare new data for prediction
        X_new = -1 * new_data.iloc[:, 2].values.reshape(-1, 1)  # Assuming the same structure
        X_new_poly = self.poly_features.transform(X_new)

        # Predict using the trained model
        y_pred_new = self.model.predict(X_new_poly)

        # Calculate accuracy based on the setpoint
        # Assuming the actual angles are in the fourth column of the new data
        actual_angles = new_data.iloc[:, 3].values
        pos = np.sum(np.abs(y_pred_new - actual_angles) < deg_accuracy)  # Count how many predictions are within the specified accuracy
        total = len(actual_angles)

        # Calculate accuracy
        accuracy = pos * 100 / total if total > 0 else 0  # Avoid division by zero

        # Optionally print the accuracy
        #print(f'Prediction Accuracy: {accuracy:.2f}%')

        # Create a DataFrame to hold the new data and the predictions
        results = pd.DataFrame(new_data)
        results['Predicted_Angle'] = y_pred_new  # Add predictions as a new column

        return accuracy, results

