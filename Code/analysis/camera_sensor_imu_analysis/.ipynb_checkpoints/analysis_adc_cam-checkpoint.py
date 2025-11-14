from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


class ADC_CAM:
    """
    Simple helper to group ADC + camera + IMU + encoder + trigger + FLIR
    files into two sets of trials based on folder suffixes.

    Example
    -------
    cam = ADC_CAM(
        root_dir=root_dir,
        path_to_repo=path_to_repository,
        folder_suffix_first="B1_slow",
        folder_suffix_second="B2_slow",
    )

    first_trials  = cam.load_first()   # list of dicts (one per trial)
    second_trials = cam.load_second()  # list of dicts (one per trial)
    """

    def __init__(
        self,
        root_dir: str,
        path_to_repo: str,
        folder_suffix_first: str,
        folder_suffix_second: str,
    ) -> None:
        # match BallBearingData behavior: repo_root / root_dir
        self.root_dir = Path(path_to_repo) / Path(root_dir)
        self.folder_suffix_first = str(folder_suffix_first).strip()
        self.folder_suffix_second = str(folder_suffix_second).strip()

        if not self.root_dir.exists():
            raise FileNotFoundError(f"root_dir does not exist: {self.root_dir}")

        if not self.folder_suffix_first:
            raise ValueError("folder_suffix_first must be a non-empty string.")
        if not self.folder_suffix_second:
            raise ValueError("folder_suffix_second must be a non-empty string.")

        self._first_folders: List[Path] = []
        self._second_folders: List[Path] = []

    # ---------- internal helpers ----------

    def _find_trial_folders_for_suffix(self, suffix: str) -> List[Path]:
        """
        Find trial folders whose name ends with `_<suffix>` (case-insensitive).

        Example: suffix='B1_slow' matches folders named like:
            9_19_25_B1_slow
            10_03_25_B1_slow
        """
        suffix = str(suffix).strip()
        if not suffix:
            return []

        # First try direct children: *_{suffix}
        direct = [
            p for p in self.root_dir.glob(f"*_{suffix}")
            if p.is_dir()
        ]

        # If none found, fall back to recursive, case-insensitive search
        if not direct:
            suf_lower = f"_{suffix}".lower()
            direct = []
            for p in self.root_dir.rglob("*"):
                if p.is_dir() and p.name.lower().endswith(suf_lower):
                    direct.append(p)

        folders = sorted(set(direct), key=lambda p: p.name)
        return folders

    def _collect_trial_files_for_folder(self, trial_folder: Path) -> Dict[str, Optional[Path]]:
        """
        Given a single trial folder, collect all expected files:

            *camera-1*.csv
            *data_adc*.csv
            *data_imu*.csv
            *data_rotenc*.csv
            *data_trigger_count*.csv
            *data_trigger_time*.csv
            *flir_data*.mat

        Returns
        -------
        info : dict
            {
              "folder":         Path,
              "camera_csv":     Path or None,
              "adc_csv":        Path or None,
              "imu_csv":        Path or None,
              "rotenc_csv":     Path or None,
              "trig_count_csv": Path or None,
              "trig_time_csv":  Path or None,
              "flir_mat":       Path or None,
            }
        """
        trial_folder = Path(trial_folder)

        def _one(pattern: str, ext: str) -> Optional[Path]:
            # pattern like "camera-1" or "data_adc"
            # IMPORTANT: ignore macOS resource-fork files that start with "._"
            matches = [
                m for m in trial_folder.glob(f"*{pattern}*{ext}")
                if not m.name.startswith("._")
            ]
            if not matches:
                return None
            if len(matches) > 1:
                # If you ever want logging, you could log here instead of print.
                pass
            return matches[0]

        camera_csv     = _one("camera-1",          ".csv")
        adc_csv        = _one("data_adc",          ".csv")
        imu_csv        = _one("data_imu",          ".csv")
        rotenc_csv     = _one("data_rotenc",       ".csv")
        trig_count_csv = _one("data_trigger_count", ".csv")
        trig_time_csv  = _one("data_trigger_time",  ".csv")

        # flir_data is a .mat file
        flir_matches = list(trial_folder.glob("*flir_data*.mat"))
        flir_mat: Optional[Path] = None
        if flir_matches:
            # If more than one, just take the first; no prints.
            flir_mat = flir_matches[0]

        return {
            "folder":         trial_folder,
            "camera_csv":     camera_csv,
            "adc_csv":        adc_csv,
            "imu_csv":        imu_csv,
            "rotenc_csv":     rotenc_csv,
            "trig_count_csv": trig_count_csv,
            "trig_time_csv":  trig_time_csv,
            "flir_mat":       flir_mat,
        }

    def _build_set(self, folders: List[Path]) -> List[Dict[str, Optional[Path]]]:
        trials: List[Dict[str, Optional[Path]]] = []
        for tf in folders:
            info = self._collect_trial_files_for_folder(tf)
            trials.append(info)
        return trials

    # ---------- public API ----------

    def load_first(self) -> List[Dict[str, Optional[Path]]]:
        """
        Load all 'first application' trials (folder_suffix_first).

        Returns
        -------
        trials : list[dict]
            One dict per trial folder, with keys:
              'folder', 'camera_csv', 'adc_csv', 'imu_csv',
              'rotenc_csv', 'trig_count_csv', 'trig_time_csv', 'flir_mat'
        """
        if not self._first_folders:
            self._first_folders = self._find_trial_folders_for_suffix(self.folder_suffix_first)
        return self._build_set(self._first_folders)

    def load_second(self) -> List[Dict[str, Optional[Path]]]:
        """
        Load all 'reapplication' trials (folder_suffix_second).

        Returns
        -------
        trials : list[dict]
            One dict per trial folder, with keys:
              'folder', 'camera_csv', 'adc_csv', 'imu_csv',
              'rotenc_csv', 'trig_count_csv', 'trig_time_csv', 'flir_mat'
        """
        if not self._second_folders:
            self._second_folders = self._find_trial_folders_for_suffix(self.folder_suffix_second)
        return self._build_set(self._second_folders)

    import pandas as pd
    from pathlib import Path
    from typing import List, Dict, Optional

    # ... your ADC_CAM class definition above ...

    def extract_adc_dfs_by_trial(
            self,
            trials: List[Dict[str, Optional[Path]]],
            *,
            add_metadata: bool = True,
    ) -> List[pd.DataFrame]:
        """
        Given a list of trial dicts (as returned by load_first/load_second),
        load the ADC CSV for each trial into a pandas DataFrame.

        Parameters
        ----------
        trials : list[dict]
            Output of load_first() or load_second(). Each dict is expected
            to have an 'adc_csv' key with a Path or None.
        add_metadata : bool, default True
            If True, add 'trial_index' and 'source_path' columns to each
            DataFrame for easier debugging/merging later.

        Returns
        -------
        dfs : list[pandas.DataFrame]
            One DataFrame per trial that has a non-None adc_csv path.
            Trials with missing adc_csv are silently skipped.
        """
        dfs: List[pd.DataFrame] = []

        for idx, trial in enumerate(trials):
            adc_path = trial.get("adc_csv", None)
            if adc_path is None:
                # silently skip trials without adc_csv
                continue

            # Robust CSV read: try UTF-8, fall back to latin-1
            try:
                df = pd.read_csv(adc_path)
            except UnicodeDecodeError:
                df = pd.read_csv(adc_path, encoding="latin-1")

            if add_metadata:
                df = df.copy()
                df["trial_index"] = idx
                df["source_path"] = str(adc_path)

            dfs.append(df)

        return dfs

    def extract_calib_means_by_set(
        self,
        adc_column: Optional[str] = None,
        *,
        exclude_name_contains: tuple = (),
        exclude_sets: tuple = (),
        make_plot: bool = True,
        overlay_mean: bool = False,
        point_alpha: float = 0.25,
        point_size: float = 10.0,
        jitter: float = 0.25,
        snap_tol_deg: float = 4.0,
        plot_all_data: bool = False,
    ) -> pd.DataFrame:
        """
        Scan root_dir for calibration folders (siblings of B1/B2), read ADC
        CSVs (data_adc), compute a per-folder mean ADC value, assign folders
        to sets (1st, 2nd, 3rd, ...) per angle, and optionally plot ADC vs
        angle.

        NEW: if plot_all_data=True, the plot shows ALL ADC samples for each
        calib folder (scatter cloud) rather than just one point per folder.

        Example calib folder name:
            2025_11_02_14_06_51_B_calib0

        Structure under self.root_dir:
            2025_11_02_14_06_51_B1_slow/
            2025_11_02_14_06_51_B2_slow/
            2025_11_02_14_06_51_B_calib0/
            2025_11_02_14_13_33_B_calib22/
            ...

        Each calib folder should contain an ADC CSV:
            *data_adc*.csv
        (we explicitly ignore any file whose name starts with "._")

        Angle handling
        --------------
        * Extract angle from 'calibXX' in the folder name (XX in degrees).
        * Snap to nearest canonical angle in [0, 22, 45, 67, 90]
          if |raw - canonical| <= snap_tol_deg.

        Set assignment (per angle)
        --------------------------
        For each snapped angle, sort matching calib folders by name:
          * first folder  -> set 1 (first application)
          * second folder -> set 2 (second application)
          * third+        -> set 3, 4, ... (optionally excluded via exclude_sets)

        ADC value
        ---------
        * If adc_column is provided and exists in the ADC CSV, use that.
        * Else, if 'adc_ch3' exists, use 'adc_ch3'.
        * Else, use the first numeric column.
        * Take the mean of that column for the folder (adc_mean).

        Parameters
        ----------
        adc_column : str or None, default None
            Name of the ADC column to average. If None, the first numeric
            column in the ADC CSV is used (preferring 'adc_ch3' if present).
        exclude_name_contains : tuple of str, default ()
            Skip calib folders whose name contains any of these substrings
            (e.g. ('C_Block',)).
        exclude_sets : tuple of int, default ()
            Set labels to drop from the final DataFrame/plot (e.g. (3, 4)
            to ignore 3rd+ calibrations).
        make_plot : bool, default True
            If True, plot ADC vs angle_snap_deg colored by set.
        overlay_mean : bool, default False
            If True, overlay per-angle mean adc_mean for each set as lines.
        point_alpha : float, default 0.25
            Transparency of scatter points.
        point_size : float, default 10.0
            Size of scatter markers.
        jitter : float, default 0.25
            Horizontal jitter so points don't overlap exactly.
        snap_tol_deg : float, default 4.0
            Maximum |angle_raw - canonical| before snapping.
        plot_all_data : bool, default False
            If True, plot ALL ADC samples (one point per row in each CSV)
            instead of just the per-folder mean.

        Returns
        -------
        calib_df : pandas.DataFrame
            Columns:
              'set'            (1 = first, 2 = second, 3+ for extras)
              'folder_name'
              'angle_raw_deg'
              'angle_snap_deg'
              'adc_column'     (column used)
              'adc_mean'       (mean ADC for that calib)
              'source_path'
        """
        canonical_angles = np.array([0, 22, 45, 67, 90], dtype=float)

        def _extract_angle_from_name(name: str) -> Optional[float]:
            m = re.search(r"calib[_-]?(\d+)", name.lower())
            if not m:
                return None
            return float(m.group(1))

        def _snap_angle(angle: float) -> Optional[float]:
            diffs = np.abs(canonical_angles - angle)
            idx = int(np.argmin(diffs))
            if diffs[idx] <= snap_tol_deg:
                return float(canonical_angles[idx])
            return None

        def _adc_values_from_csv(csv_path: Path) -> tuple[Optional[str], Optional[np.ndarray]]:
            # Robust read: UTF-8, then Latin-1
            try:
                df_adc = pd.read_csv(csv_path)
            except UnicodeDecodeError:
                df_adc = pd.read_csv(csv_path, encoding="latin-1")

            col_to_use: Optional[str] = None
            if adc_column is not None and adc_column in df_adc.columns:
                col_to_use = adc_column
            elif "adc_ch3" in df_adc.columns:
                col_to_use = "adc_ch3"
            else:
                numeric_cols = df_adc.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) == 0:
                    return None, None
                col_to_use = numeric_cols[0]

            vals = pd.to_numeric(df_adc[col_to_use], errors="coerce").dropna().to_numpy()
            if vals.size == 0:
                return None, None

            return col_to_use, vals

        # 1. Find calib folders as direct children of root_dir
        calib_folders = []
        for p in self.root_dir.iterdir():
            if not p.is_dir():
                continue
            name = p.name
            if "calib" not in name.lower():
                continue
            if any(excl in name for excl in exclude_name_contains):
                continue

            angle_raw = _extract_angle_from_name(name)
            if angle_raw is None:
                continue
            angle_snap = _snap_angle(angle_raw)
            calib_folders.append((p, name, angle_raw, angle_snap))

        if not calib_folders:
            return pd.DataFrame()

        # 2. Group by snapped angle and assign set labels per angle
        rows = []
        all_samples = []  # for optional full-data plotting
        by_angle: dict[float | None, list[tuple[Path, str, float]]] = {}
        for p, name, angle_raw, angle_snap in calib_folders:
            by_angle.setdefault(angle_snap, []).append((p, name, angle_raw))

        for angle_snap, items in by_angle.items():
            items_sorted = sorted(items, key=lambda t: t[1])
            for set_idx, (folder_path, folder_name, angle_raw) in enumerate(
                items_sorted, start=1
            ):
                if set_idx in exclude_sets:
                    continue

                # 3. Find ADC CSV inside this calib folder (skip "._" files)
                adc_matches = [
                    q for q in folder_path.glob("*data_adc*.csv")
                    if not q.name.startswith("._")
                ]
                if not adc_matches:
                    continue
                adc_path = adc_matches[0]

                col_used, vals = _adc_values_from_csv(adc_path)
                if col_used is None or vals is None or vals.size == 0:
                    continue

                adc_mean = float(vals.mean())

                rows.append(
                    {
                        "set": set_idx,
                        "folder_name": folder_name,
                        "angle_raw_deg": angle_raw,
                        "angle_snap_deg": angle_snap,
                        "adc_column": col_used,
                        "adc_mean": adc_mean,
                        "source_path": str(adc_path),
                    }
                )

                if plot_all_data:
                    # store every sample for plotting
                    for v in vals:
                        all_samples.append(
                            {
                                "set": set_idx,
                                "angle_snap_deg": angle_snap,
                                "adc_value": float(v),
                            }
                        )

        if not rows:
            return pd.DataFrame()

        calib_df = pd.DataFrame(rows)

        # 3. Plot, if requested
        if make_plot and not calib_df.empty:
            fig, ax = plt.subplots(figsize=(8, 5))

            if plot_all_data and all_samples:
                # scatter all ADC samples
                all_df = pd.DataFrame(all_samples)
                plot_df = all_df[~all_df["set"].isin(exclude_sets)].copy()
                unique_sets = sorted(plot_df["set"].unique())
                colors = ["C0", "C1", "C2", "C3"]

                for idx, set_label in enumerate(unique_sets):
                    sub = plot_df[plot_df["set"] == set_label]
                    if sub.empty:
                        continue

                    x = sub["angle_snap_deg"].to_numpy()
                    if jitter > 0:
                        x = x + np.random.randn(len(x)) * jitter
                    y = sub["adc_value"].to_numpy()

                    ax.scatter(
                        x,
                        y,
                        s=point_size,
                        alpha=point_alpha,
                        color=colors[idx % len(colors)],
                        label=f"Set {set_label}" if idx == 0 else None,
                    )

            else:
                # fallback: plot only per-folder means (previous behavior)
                plot_df = calib_df[~calib_df["set"].isin(exclude_sets)].copy()
                unique_sets = sorted(plot_df["set"].unique())
                colors = ["C0", "C1", "C2", "C3"]

                for idx, set_label in enumerate(unique_sets):
                    sub = plot_df[plot_df["set"] == set_label]
                    if sub.empty:
                        continue

                    x = sub["angle_snap_deg"].to_numpy()
                    if jitter > 0:
                        x = x + np.random.randn(len(x)) * jitter
                    y = sub["adc_mean"].to_numpy()

                    ax.scatter(
                        x,
                        y,
                        s=point_size,
                        alpha=point_alpha,
                        color=colors[idx % len(colors)],
                        label=f"Set {set_label}",
                    )

                    if overlay_mean:
                        means = (
                            sub.groupby("angle_snap_deg")["adc_mean"]
                            .mean()
                            .reset_index()
                        )
                        ax.plot(
                            means["angle_snap_deg"],
                            means["adc_mean"],
                            marker="o",
                            linestyle="-",
                            color=colors[idx % len(colors)],
                        )

            ax.set_xlabel("Calibration Angle (deg, snapped)")
            ax.set_ylabel("ADC value" if plot_all_data else "Mean ADC value")
            ax.set_title("ADC Calibration by Set")
            present = sorted(
                a
                for a in canonical_angles
                if a in calib_df["angle_snap_deg"].unique()
            )
            if present:
                ax.set_xticks(present)
            ax.grid(True, alpha=0.3)
            ax.legend()
            plt.tight_layout()

        return calib_df

    def calibrate_trials_with_camera(
            self,
            adc_trials_first,
            adc_trials_second,
            *,
            adc_column: str = "adc_ch3",
            poly_order: int = 2,
            calib_kwargs: dict | None = None,
            clamp_theta: bool = True,
            deg_min: float = 0.0,
            deg_max: float = 90.0,
    ):
        """
        Use camera-based ADC calibration (calib folders) to convert ALL ADC
        samples in adc_trials_first / adc_trials_second into angles.

        For each calibration set (set=1 for first application, set=2 for
        second application), we:

          1. Call extract_calib_means_by_set(...) to get calib_df with
             columns like 'adc_mean' and 'angle_snap_deg'.
          2. Fit a polynomial mapping ADC -> angle:
                 angle_deg ≈ p_set(adc_value)
             using np.polyfit(adc_mean, angle_snap_deg, poly_order).
          3. Apply that polynomial to every ADC sample in the corresponding
             trial set, appending a new column:
                 'theta_cam_cal'   (in degrees)

        Parameters
        ----------
        adc_trials_first : list[pd.DataFrame]
            List of ADC DataFrames for first-application trials.
        adc_trials_second : list[pd.DataFrame]
            List of ADC DataFrames for second-application trials.
        adc_column : str, default 'adc_ch3'
            Name of the ADC column to use in each trial DataFrame.
        poly_order : int, default 2
            Polynomial order to use for ADC->angle mapping.
        calib_kwargs : dict or None
            Extra keyword args passed to extract_calib_means_by_set.
        clamp_theta : bool, default True
            If True, clamp theta_cam_cal to [deg_min, deg_max].
        deg_min : float, default 0.0
            Minimum allowed angle (deg) if clamp_theta is True.
        deg_max : float, default 90.0
            Maximum allowed angle (deg) if clamp_theta is True.

        Returns
        -------
        result : dict
            {
              "calib_df"               : full calibration DataFrame,
              "coeffs"                 : {1: coeffs_first, 2: coeffs_second},
              "adc_trials_first_theta" : list[pd.DataFrame],
              "adc_trials_second_theta": list[pd.DataFrame],
            }

            Each DataFrame in adc_trials_*_theta is a COPY of the input
            trial DataFrame with an extra column:
                'theta_cam_cal'
        """
        import numpy as np
        import pandas as pd

        # 1) Build kwargs for extract_calib_means_by_set without duplicating keys
        if calib_kwargs is None:
            calib_kwargs = {}
        else:
            calib_kwargs = dict(calib_kwargs)  # shallow copy

        # Force make_plot=False here so this method doesn't spam plots
        calib_kwargs["make_plot"] = False

        calib_df = self.extract_calib_means_by_set(**calib_kwargs)

        if calib_df.empty:
            # No calibration data found; just return copies with no theta column
            first_out = [df.copy() for df in adc_trials_first]
            second_out = [df.copy() for df in adc_trials_second]
            return {
                "calib_df": calib_df,
                "coeffs": {},
                "adc_trials_first_theta": first_out,
                "adc_trials_second_theta": second_out,
            }

        # 2) Fit one polynomial per set (1 and 2) using adc_mean -> angle_snap_deg
        coeffs: dict[int, np.ndarray] = {}
        for set_label in sorted(calib_df["set"].unique()):
            sub = calib_df[
                (calib_df["set"] == set_label)
                & calib_df["angle_snap_deg"].notna()
                ]
            # Need at least poly_order+1 points to fit that order
            if len(sub) < poly_order + 1:
                continue

            x = sub["adc_mean"].to_numpy()
            y = sub["angle_snap_deg"].to_numpy()
            c = np.polyfit(x, y, poly_order)
            coeffs[int(set_label)] = c

        # 3) Helper to apply a given set's polynomial to all trials
        def _apply_to_trials(trials, set_label: int):
            out = []
            c = coeffs.get(set_label, None)
            if c is None:
                # No calibration for this set -> just copy with no angle column
                return [df.copy() for df in trials]

            p = np.poly1d(c)
            for df in trials:
                if adc_column not in df.columns:
                    out.append(df.copy())
                    continue
                df2 = df.copy()
                adc_vals = pd.to_numeric(df2[adc_column], errors="coerce")
                theta = p(adc_vals.to_numpy())
                if clamp_theta:
                    theta = np.clip(theta, deg_min, deg_max)
                df2["theta_cam_cal"] = theta
                out.append(df2)
            return out

        adc_first_theta = _apply_to_trials(adc_trials_first, set_label=1)
        adc_second_theta = _apply_to_trials(adc_trials_second, set_label=2)

        return {
            "calib_df": calib_df,
            "coeffs": coeffs,
            "adc_trials_first_theta": adc_first_theta,
            "adc_trials_second_theta": adc_second_theta,
        }










