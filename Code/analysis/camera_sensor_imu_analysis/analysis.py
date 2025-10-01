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

    # Specific extractors
    def extract_adc_dfs_by_trial(self, trial_folders):
        return self._extract_by_glob(trial_folders, ["data_adc*.csv", "data_adc*"])

    def extract_imu_dfs_by_trial(self, trial_folders):
        return self._extract_by_glob(trial_folders, ["data_imu*.csv", "data_imu*"])

    def extract_spacebar_dfs_by_trial(self, trial_folders):
        return self._extract_by_glob(trial_folders, ["data_spacebar*.csv", "data_spacebar*"])

    def extract_rotenc_dfs_by_trial(self, trial_folders):
        return self._extract_by_glob(trial_folders, ["data_rotenc*.csv", "data_rotenc*"])

    def extract_dlc3d_dfs_by_trial(
        self,
        trial_folders: List[str],
        patterns: Tuple[str, ...] = ("data_DLC3D*.csv", "DLC3D*.csv", "*dlc3d*.csv"),
    ) -> List[pd.DataFrame]:
        """Find per-trial DLC3D CSV(s); returns one DataFrame per trial."""
        return self._extract_by_glob(trial_folders, pattern=list(patterns))

    # ---------- calibration ----------
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
            y0 = float(np.nanmax(y))  # ADC at 0°
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

    
    # ---------- tall DF builder ----------
    def trials_to_tall_df(
            self,
            adc_trials: List[pd.DataFrame],
            set_label: str,
            trial_len_sec: float = 10.0,
            adc_col: str = "adc_ch3",
            time_col_options: Tuple[str, ...] = ("timestamp", "time", "t_sec"),
            include_endpoint: bool = True,
            clamp_theta: bool = False,  # <-- new: no clipping by default
    ) -> pd.DataFrame:
        if not self.calib:
            raise RuntimeError("Calibration not set. Call fit_and_set_calibration(...) first.")

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
            # <-- key change: no clipping unless you ask for it
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
        """
        augmented: List[pd.DataFrame] = []
        parts: List[pd.DataFrame] = []

        for trial_idx, dlc_df in enumerate(dlc3d_trials, start=1):
            if dlc_df is None or dlc_df.empty:
                augmented.append(pd.DataFrame()); continue

            cam = DLC3DBendAngles(dlc_df)

            # MCP bend
            hand_pts = cam.get_points("hand")
            mcp_pts  = cam.get_points("MCP")
            pip_pts  = cam.get_points("PIP")
            v1_mcp = cam.vector(hand_pts, mcp_pts)  # hand→MCP
            v2_mcp = cam.vector(mcp_pts, pip_pts)  # MCP→PIP

            # Wrist bend (+ plane)
            forearm_pts = cam.get_points("forearm")
            v1_wrist = cam.vector(forearm_pts, hand_pts)  # forearm→hand
            v2_wrist = cam.vector(hand_pts, mcp_pts)      # hand→MCP

            angles_mcp   = cam.angle_from_vectors(v1_mcp, v2_mcp)
            angles_wrist = cam.angle_from_vectors(v1_wrist, v2_wrist)

            angles_mcp_plane, _, _, plane_ok = cam.angle_from_vectors_in_plane(
                v1=v1_mcp, v2=v2_mcp, plane_v1=v1_wrist, plane_v2=v2_wrist, signed=signed_in_plane
            )

            df_out = cam.df.copy()
            df_out[("metric", "mcp_bend_deg", "deg")]                 = angles_mcp
            df_out[("metric", "wrist_bend_deg", "deg")]               = angles_wrist
            df_out[("metric", "mcp_bend_in_wrist_plane_deg", "deg")]  = angles_mcp_plane
            if add_plane_ok:
                df_out[("metric", "wrist_plane_ok", "")] = plane_ok

            augmented.append(df_out)

            # try to carry a useful time axis if present
            time_col = None
            for c in df_out.columns:
                if "time" in str(c).lower() or "timestamp" in str(c).lower():
                    time_col = c; break
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
            columns=["set_label","trial","frame","time_or_timestamp",
                     "mcp_bend_deg","wrist_bend_deg","mcp_bend_in_wrist_plane_deg","wrist_plane_ok"]
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
    ) -> List[pd.DataFrame]:
        """
        For each trial folder, find a FLIR .mat file and load variables whose names start with `prefix`
        (e.g., 'ts*') into a single DataFrame per trial.
        """
        out: List[pd.DataFrame] = []
        for folder in trial_folders:
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
                out.append(pd.DataFrame()); continue

            cands_sorted = sorted(cands, key=lambda p: p.stat().st_size if p.exists() else 0, reverse=True)
            mat_path = cands_sorted[0]

            try:
                df = self._mat_to_df(mat_path, prefix=prefix)
            except Exception as e:
                print(f"[WARN] Skipping unreadable MAT: {mat_path} ({e})")
                df = pd.DataFrame()

            out.append(df)
        return out

    def attach_dlc_angles_to_cam_by_trial(
        self,
        cam_trials: List[pd.DataFrame],
        dlc_aug_trials: List[pd.DataFrame],
        *,
        cam_time_col: Optional[str] = None,   # e.g. 'ts_25183199'; auto-detect if None
        cam_time_prefix: str = "ts",          # auto-detect columns starting with this
        tolerance: Union[str, float, int] = "10ms",  # '10ms' or seconds(float) or microseconds(int)
        direction: str = "nearest",
        suffix: str = "_dlc",                 # suffix for appended metric columns
    ) -> List[pd.DataFrame]:
        """
        Append DLC angle columns (from compute_dlc3d_angles_by_trial) to each camera ts_* DataFrame.

        Per trial:
          • If lengths match: index-wise join (fast path).
          • Else if both sides have a time column: merge_asof on coerced seconds.
          • Else: nearest join on frame index.
        """
        out: List[pd.DataFrame] = []

        def _pick_cam_time(df: pd.DataFrame) -> Optional[str]:
            if cam_time_col and cam_time_col in df.columns:
                return cam_time_col
            cands = [c for c in df.columns if str(c).lower().startswith(cam_time_prefix)]
            return cands[0] if cands else None

        def _find_dlc_time_col(df: pd.DataFrame):
            for c in df.columns:
                s = str(c)
                if "ts" in s.lower() or "timestamp" in s.lower():
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
            new_names = []
            for c in sub.columns:
                base = "_".join([x for x in c if str(x) != ""]) if isinstance(c, tuple) else str(c)
                new_names.append(base + suffix)
            sub.columns = new_names
            return sub

        def _tol_seconds_local(tol) -> float:
            if isinstance(tol, str):
                return pd.to_timedelta(tol).total_seconds()
            return float(tol)

        for cam_df, dlc_df in zip(cam_trials, dlc_aug_trials):
            if cam_df is None or cam_df.empty or dlc_df is None or dlc_df.empty:
                out.append(cam_df if isinstance(cam_df, pd.DataFrame) else pd.DataFrame())
                continue

            dlc_metrics = _flatten_metrics(dlc_df)

            # Case A: identical lengths → fast index join
            if len(cam_df) == len(dlc_metrics) and len(dlc_metrics) > 0:
                joined = cam_df.reset_index(drop=True).join(dlc_metrics.reset_index(drop=True))
                out.append(joined)
                continue

            cam_col  = _pick_cam_time(cam_df)
            dlc_tcol = _find_dlc_time_col(dlc_df)

            # Case B: time-based merge_asof if possible
            if cam_col is not None and dlc_tcol is not None and len(dlc_metrics) > 0:
                left = cam_df[[cam_col]].copy()
                right = dlc_df[[dlc_tcol]].copy()

                left["_t"]     = self._coerce_time_series_numeric_seconds(left[cam_col])
                right["_t_enc"] = self._coerce_time_series_numeric_seconds(right[dlc_tcol])

                left  = left.dropna(subset=["_t"]).sort_values("_t").reset_index(drop=False)
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
                out.append(joined)
                continue

            # Case C: fallback to frame-index nearest
            if len(dlc_metrics) == 0:
                out.append(cam_df.copy())
                continue

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

    def align_theta_all_to_cam_for_set(
        self,
        theta_all_set: pd.DataFrame,
        cam_trials: List[pd.DataFrame],
        *,
        enc_time_col: str = "timestamp",
        cam_time_col: Optional[str] = None,
        cam_time_prefix: str = "ts",
        tolerance: Union[str, int, float] = "10ms",
        direction: str = "nearest",
        theta_col: str = "theta_pred_deg",
        keep_time_delta: bool = True,
        drop_unmatched: bool = True,
        return_concatenated: bool = False,
    ) -> List[pd.DataFrame] | pd.DataFrame:
        """
        Align a tall per-set θ table to per-trial camera ts_* DataFrames.
        Keeps ALL camera columns; appends theta/ADC/time_s, plus deltas in ns/ms/s.
        """
        def _coerce_tolerance_to_timedelta(tol) -> pd.Timedelta:
            if isinstance(tol, (np.integer, int)):
                return pd.to_timedelta(int(tol), unit="us")
            if isinstance(tol, (np.floating, float)):
                return pd.to_timedelta(float(tol), unit="s")
            return pd.to_timedelta(tol)

        def _to_int64_ns(series: pd.Series) -> pd.Series:
            if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
                return series.view("i8").astype("Int64")
            sn = pd.to_numeric(series, errors="coerce")
            if sn.notna().any():
                return sn.round().astype("Int64")
            dt = pd.to_datetime(series, errors="coerce", utc=True)
            if dt.notna().any():
                return dt.view("i8").astype("Int64")
            return pd.Series(pd.array([pd.NA] * len(series), dtype="Int64"), index=series.index)

        if theta_col not in theta_all_set.columns:
            raise KeyError(f"'{theta_col}' not found in theta_all_set.")
        if enc_time_col not in theta_all_set.columns:
            raise KeyError(f"'{enc_time_col}' not found in theta_all_set.")

        tol_td = _coerce_tolerance_to_timedelta(tolerance)
        tol_ns = int(tol_td / pd.to_timedelta(1, unit="ns"))
        merged_trials: List[pd.DataFrame] = []

        for trial_idx, cam_df in enumerate(cam_trials, start=1):
            th_df = theta_all_set.loc[theta_all_set["trial"] == trial_idx].copy()
            if cam_df is None or cam_df.empty or th_df.empty:
                merged_trials.append(pd.DataFrame()); continue

            if cam_time_col is None:
                cam_cands = [c for c in cam_df.columns if str(c).lower().startswith(cam_time_prefix)]
                cam_col = cam_cands[0] if cam_cands else None
            else:
                cam_col = cam_time_col

            if cam_col is None or cam_col not in cam_df.columns:
                print(f"[alignθ] Trial {trial_idx}: camera time column not found (prefix='{cam_time_prefix}').")
                merged_trials.append(pd.DataFrame()); continue

            left = cam_df.copy()
            left["_t_ns"] = _to_int64_ns(left[cam_col])
            left = left.dropna(subset=["_t_ns"]).sort_values("_t_ns")

            extra_cols = ["time_s", "adc_ch3"]
            right_cols = [enc_time_col, theta_col] + [c for c in extra_cols if c in th_df.columns]
            right = th_df[right_cols].copy()
            right["_t_ns"] = _to_int64_ns(right[enc_time_col])
            right = right.dropna(subset=["_t_ns"]).sort_values("_t_ns")
            right["_t_enc_ns"] = right["_t_ns"].copy()

            if left.empty or right.empty:
                merged_trials.append(pd.DataFrame()); continue

            m = pd.merge_asof(
                left, right,
                left_on="_t_ns", right_on="_t_ns",
                direction=direction,
                tolerance=tol_ns,
                allow_exact_matches=True,
            )

            if keep_time_delta and "_t_enc_ns" in m.columns:
                m["_delta_ns"]  = (m["_t_enc_ns"] - m["_t_ns"]).astype("Int64")
                m["_delta_ms"]  = m["_delta_ns"].astype("float64") / 1e6
                m["_delta_sec"] = m["_delta_ns"].astype("float64") / 1e9

            if drop_unmatched and theta_col in m.columns:
                m = m.loc[m[theta_col].notna()].copy()

            m["trial"] = trial_idx
            if "set_label" in theta_all_set.columns:
                vals = th_df["set_label"].dropna()
                if not vals.empty:
                    m["set_label"] = vals.mode().iat[0]

            merged_trials.append(m.reset_index(drop=True))

        if return_concatenated:
            non_empty = [df for df in merged_trials if not df.empty]
            return pd.concat(non_empty, ignore_index=True) if non_empty else pd.DataFrame()

        return merged_trials

    def _tol_seconds(self, tol):
        """Return tolerance in float seconds (handles strings like '10ms')."""
        if isinstance(tol, str):
            return pd.to_timedelta(tol).total_seconds()
        return float(tol)

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

    def plot_compact_pairwise_comparison(self, pairwise_min_accuracy, pairwise_abs_error,
                                         xlabel_flat, group_size=3,
                                         title1='Min Angle for 100% Accuracy',
                                         title2='Mean Absolute Error', ylim=(0, 15)):
        """
        Compact overlayed double-bar plot showing:
        - Blue = model trained/tested on same sample (self-trained)
        - Red = model trained on first sample in group, tested on others (cross-trained)

        Parameters:
        - pairwise_min_accuracy (np.array): Matrix of min angle for 100% accuracy
        - pairwise_abs_error (np.array): Matrix of mean absolute errors
        - xlabel_flat (list): List of sample labels corresponding to DS_flat
        - group_size (int): Number of samples per group
        - title1 (str): Title for top subplot
        - title2 (str): Title for bottom subplot
        - ylim (tuple): Y-axis limits for both plots (e.g., (0, 15))
        """
        import matplotlib.pyplot as plt
        import numpy as np

        assert pairwise_min_accuracy.shape[0] == len(xlabel_flat)
        assert pairwise_min_accuracy.shape[0] % group_size == 0

        num_groups = pairwise_min_accuracy.shape[0] // group_size

        fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        bar_width = 0.6
        inner_width = 0.4  # width for the red overlaid bar
        x_all = []

        for group_idx in range(num_groups):
            base_idx = group_idx * group_size
            x_group = np.arange(group_size) + group_idx * (group_size + 1.0)
            x_all.extend(x_group)

            min_self = []
            min_cross = []
            err_self = []
            err_cross = []

            for i in range(group_size):
                idx = base_idx + i
                min_self.append(pairwise_min_accuracy[idx, idx])
                err_self.append(pairwise_abs_error[idx, idx])

                # Cross-trained using first sample in group
                cross_idx = base_idx
                if idx != cross_idx:
                    min_cross.append(pairwise_min_accuracy[cross_idx, idx])
                    err_cross.append(pairwise_abs_error[cross_idx, idx])
                else:
                    min_cross.append(np.nan)  # skip duplicate
                    err_cross.append(np.nan)

            # --- Plot Top Subplot: Min Angle ---
            axes[0].bar(x_group, min_self, width=bar_width, color='blue',
                        label='Self-trained' if group_idx == 0 else "")
            axes[0].bar(x_group, min_cross, width=inner_width, color='red', alpha=0.7,
                        label='Cross-trained (model 1)' if group_idx == 0 else "")

            # --- Plot Bottom Subplot: Mean Error ---
            axes[1].bar(x_group, err_self, width=bar_width, color='blue')
            axes[1].bar(x_group, err_cross, width=inner_width, color='red', alpha=0.7)

        # --- Format Top Subplot ---
        axes[0].set_ylabel('Min Angle (deg)')
        axes[0].set_ylim(ylim)
        axes[0].set_yticks(np.arange(ylim[0], ylim[1] + 1, 1))
        axes[0].set_title(title1)
        axes[0].legend()
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # --- Format Bottom Subplot ---
        axes[1].set_ylabel('Mean Error (deg)')
        axes[1].set_xlabel('Sample')
        axes[1].set_ylim(ylim)
        axes[1].set_yticks(np.arange(ylim[0], ylim[1] + 1, 1))
        axes[1].set_title(title2)
        axes[1].grid(True, linestyle='--', alpha=0.5)

        # --- Annotated X-Axis Labels ---
        xtick_labels = []
        for group_idx in range(num_groups):
            base_idx = group_idx * group_size
            for i in range(group_size):
                idx = base_idx + i
                label = xlabel_flat[idx]
                if i == 0:
                    label += " (model 1)"
                xtick_labels.append(label)

        axes[1].set_xticks(x_all)
        axes[1].set_xticklabels(xtick_labels, rotation=45, ha='right')

        plt.tight_layout()
        plt.savefig("compact pairwise comp.png", dpi=300, bbox_inches='tight')
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

    def fit_knuckle_radius_from_normalized(self,
                                           L=2.0,
                                           angle_col="Rotary Encoder",
                                           value_col="ADC Value",
                                           r0=0.7,
                                           bounds=(1e-4, 2.45),
                                           restrict_to_0_90=True,
                                           make_angles_positive=True,
                                           plot=False,
                                           ax=None,
                                           flip_data=False):
        """
        Fit the normalized theoretical model to data in self.data to estimate knuckle radius r.
        Now supports flipping the data so that 0°→0 and 90°→1 always match the model.
        """
        import numpy as np
        from scipy.optimize import curve_fit
        import matplotlib.pyplot as plt

        if self.data is None:
            raise ValueError("No data loaded. Use load_merged_df(...) or load_data(...) first.")

        # Pull columns
        x_deg = np.asarray(self.data[angle_col], dtype=float)
        y = np.asarray(self.data[value_col], dtype=float)

        # Flip data if needed
        if flip_data:
            y = 1.0 - y

        # Clean/guard
        m = np.isfinite(x_deg) & np.isfinite(y)
        x_deg = x_deg[m]
        y = y[m]

        if make_angles_positive:
            x_deg = np.abs(x_deg)

        if restrict_to_0_90:
            mm = (x_deg >= 0.0) & (x_deg <= 95.0)
            x_deg = x_deg[mm]
            y = y[mm]

        if x_deg.size < 5:
            raise ValueError("Not enough points (need ≥5) in the 0–90° range to fit.")

        # Convert to radians
        theta = np.deg2rad(x_deg)

        # Theoretical model
        def _norm_theoretical(theta, r, L=L):
            alpha = r / L
            num = theta * (8.0 - alpha * theta) / (2.0 - alpha * theta) ** 2
            th_max = np.pi / 2.0
            den = th_max * (8.0 - alpha * th_max) / (2.0 - alpha * th_max) ** 2
            return num / den

        # Fit r
        popt, pcov = curve_fit(lambda th, r: _norm_theoretical(th, r, L),
                               theta, y, p0=[r0], bounds=bounds, maxfev=20000)
        r_hat = float(popt[0])

        # Predictions and R²
        y_pred = _norm_theoretical(theta, r_hat, L)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Std error & 95% CI
        r_se = float(np.sqrt(pcov[0, 0])) if pcov.size else np.nan
        ci95 = (r_hat - 1.96 * r_se, r_hat + 1.96 * r_se) if np.isfinite(r_se) else (np.nan, np.nan)

        # Optional plot
        if plot:
            if ax is None:
                fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(x_deg, y, ".", markersize=3, label="Data")
            th_fit = np.linspace(0, np.deg2rad(max(95.0, x_deg.max())), 600)
            y_fit = _norm_theoretical(th_fit, r_hat, L)
            ax.plot(np.rad2deg(th_fit), y_fit, "-", label=f"Fit r={r_hat:.3f} in (R²={r2:.4f})")
            ax.set_xlabel("Angle (deg)")
            ax.set_ylabel("Normalized ADC (0–1)")
            ax.set_title(f"Theoretical fit (L={L} in)")
            ax.legend()
            plt.tight_layout()

        return {
            "r_hat": r_hat,
            "r_se": r_se,
            "r_ci95": ci95,
            "r2": r2,
            "params_cov": pcov,
        }

    def theta_from_y(self, y, r, L=2.0, tol_deg=1e-3, max_iters=30):
        """
        Invert normalized theoretical curve y(θ; r, L) on [0, 90°].
        No other class methods required. Returns radians.
        """
        import numpy as np
        # clamp for safety
        y = float(np.clip(y, 0.0, 1.0))

        # inline normalized model (y = 1 at 90°)
        def f(theta):
            a = r / L
            num = theta * (8.0 - a * theta)
            den = (2.0 - a * theta) ** 2
            th90 = np.pi / 2
            num90 = th90 * (8.0 - a * th90) / (2.0 - a * th90) ** 2
            return (num / den) / num90

        lo, hi = 0.0, np.pi / 2
        for _ in range(max_iters):
            mid = 0.5 * (lo + hi)
            if f(mid) < y:
                lo = mid
            else:
                hi = mid
            if (hi - lo) <= np.deg2rad(tol_deg):
                break
        return 0.5 * (lo + hi)

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

        Parameters:
        - data: List of lists, where each sublist represents values for a group.
        - labels: List of group names.
        - title: Chart title.
        - ylabel: Y-axis label.
        - colors: List of colors for each group.
        - ylim: Tuple (ymin, ymax) for y-axis limit.
        """
        num_groups = len(data)  # Number of groups (e.g., 3)
        max_bars = max(len(group) for group in data)  # Find the maximum datasets in any group
        bar_width = 0.2  # Controls spacing between bars in a group

        x_positions = np.arange(num_groups)  # X positions for the groups

        plt.figure(figsize=(8, 6))

        for i, (group_values, color) in enumerate(zip(data, colors)):
            num_bars = len(group_values)
            x_offsets = np.linspace(-bar_width * (num_bars - 1) / 2, bar_width * (num_bars - 1) / 2, num_bars)

            # Plot bars for this group, slightly offset from center position
            for j, (val, offset) in enumerate(zip(group_values, x_offsets)):
                print(type(val))
                if type(val) is float: 
                    plt.bar(x_positions[i] + offset, val, width=bar_width, color=color,
                        label=f"Sample {j + 1}" if i == 0 else "")
                elif type(val) is list:
                    plt.bar(x_positions[i] + offset, np.mean(np.array(val)), width=bar_width, color=color,
                        label=f"Sample {j + 1}" if i == 0 else "", yerr=np.std(np.array(val)), capsize=5, error_kw={'elinewidth': 1.5})
                    
        # Set x-ticks to group labels
        plt.xticks(x_positions, labels)
        plt.ylabel(ylabel)
        plt.title(title)

        if ylim:
            plt.ylim(ylim)

        plt.savefig("min_angle_bar_chart.png", dpi=300, bbox_inches='tight')

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

