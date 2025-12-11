from pathlib import Path
import pandas as pd
import re
import numpy as np
from typing import List, Dict, Optional, Tuple, Sequence
import matplotlib.pyplot as plt
from typing import Optional
from analysis import DLC3DBendAngles # class for taking DLC 3d point data and converting to angles
import seaborn as sns
COLOR_FIRST = "#4C72B0"   # blue
COLOR_SECOND = "#DD8452"  # orange


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
            canonical_angles: Optional[Sequence[float]] = None,
    ) -> pd.DataFrame:
        """
        Scan root_dir for calibration folders (siblings of B1/B2), read ADC
        CSVs (data_adc), compute a per-folder mean ADC value, assign folders
        to sets (1st, 2nd, 3rd, ...) per angle, and optionally plot ADC vs
        angle.

        NEW:
          * If plot_all_data=True, the plot shows ALL ADC samples for each
            calib folder (scatter cloud) rather than just one point per folder.
          * canonical_angles can be specified (e.g., (0, 22, 45, 67) to
            exclude 90° from snapping).

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
        * Snap to nearest canonical angle in canonical_angles
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
        canonical_angles : sequence of float or None, default None
            Canonical angles (deg) to snap to. If None, defaults to
            (0, 22, 45, 67, 90).

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
        # Default canonical angles if not provided
        if canonical_angles is None:
            canonical_angles = (0, 22, 45, 67, 90)
        canonical_angles_arr = np.array(canonical_angles, dtype=float)

        def _extract_angle_from_name(name: str) -> Optional[float]:
            m = re.search(r"calib[_-]?(\d+)", name.lower())
            if not m:
                return None
            return float(m.group(1))

        def _snap_angle(angle: float) -> Optional[float]:
            diffs = np.abs(canonical_angles_arr - angle)
            idx = int(np.argmin(diffs))
            if diffs[idx] <= snap_tol_deg:
                return float(canonical_angles_arr[idx])
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
            fig, ax = plt.subplots(figsize=(8.5, 6))

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
            ax.set_ylabel("ADC value" if plot_all_data else "Mean ADC Value")
            ax.set_title("ADC Calibration by Set")

            present = sorted(
                a
                for a in canonical_angles_arr
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

    def _mat_to_df(self, mat_path: Path, prefix: str = "ts") -> pd.DataFrame:
        """
        Light-weight MAT loader for FLIR files.

        Grabs all 1D numeric variables whose names start with `prefix`
        (e.g. 'ts_123456', 'ts_foo'), returns them as columns.

        If there is exactly one such column and its name starts with prefix,
        it is renamed to 'timestamp'.

        If arrays have different lengths, they are all trimmed to the
        shortest length so that a DataFrame can be constructed.
        """
        import numpy as np

        mat_path = Path(mat_path)
        if not mat_path.exists():
            return pd.DataFrame()

        pref_l = str(prefix).lower()
        wanted: Dict[str, np.ndarray] = {}

        # --- First try classic MATLAB (v7 and below) ---
        try:
            from scipy.io import loadmat
            loaded = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
            keys = [k for k in loaded.keys() if not k.startswith("__")]
            for k in keys:
                if not k.lower().startswith(pref_l):
                    continue
                arr = np.asarray(loaded[k])
                arr = np.squeeze(arr)
                if arr.ndim == 1:
                    wanted[k] = np.asarray(arr, dtype=float)
        except Exception as e:
            print(f"[WARN] loadmat failed on {mat_path}: {e}")

        # --- If nothing found, optionally try HDF5/v7.3 ---
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
                        data = np.asarray(obj[...])
                        data = np.squeeze(data)
                        if data.ndim == 1:
                            wanted[key] = np.asarray(data, dtype=float)

                    f.visititems(_visit)
            except Exception as e:
                print(f"[WARN] HDF5 read failed on {mat_path}: {e}")

        if not wanted:
            return pd.DataFrame()

        # --- Make sure all arrays have the same length by trimming to min ---
        lengths = {k: len(v) for k, v in wanted.items() if v is not None}
        if not lengths:
            return pd.DataFrame()

        min_len = min(lengths.values())

        if len(set(lengths.values())) > 1:
            print(
                f"[INFO] _mat_to_df: trimming arrays in {mat_path.name} "
                f"to min length {min_len} (lengths were: {lengths})"
            )

        trimmed = {k: v[:min_len] for k, v in wanted.items()}

        df = pd.DataFrame(trimmed)

        if df.shape[1] == 1:
            only = df.columns[0]
            if str(only).lower().startswith(pref_l):
                df = df.rename(columns={only: "timestamp"})

        return df

    # ---------- FLIR .mat per trial ----------

    def extract_mat_dfs_by_trial(
        self,
        trials: List[Dict[str, Optional[Path]]],
        mat_name: str = "flir_data.mat",
        prefix: str = "ts",
        *,
        add_labels: bool = True,
        trial_labels: Optional[List[int]] = None,  # else trial_base..N-1
        trial_base: int = 1,
        set_label: Optional[str] = None,          # e.g. "first_cam", "second_cam"
        set_labels: Optional[List[str]] = None,   # per-trial labels
        include_path: bool = False,
    ) -> List[pd.DataFrame]:
        """
        For each trial dict, load the corresponding FLIR .mat (if present)
        into a DataFrame of ts_* signals.

        Keeps 1:1 alignment with `trials`:
            len(output) == len(trials)

        If a trial has no readable .mat, returns an empty DataFrame
        (optionally with label columns) for that trial.
        """
        out: List[pd.DataFrame] = []
        n = len(trials)

        # ---- build labels ----
        if trial_labels is None:
            labels_trial = [trial_base + i for i in range(n)]
        else:
            if len(trial_labels) != n:
                raise ValueError("trial_labels length must match len(trials).")
            labels_trial = trial_labels

        if set_labels is not None and len(set_labels) != n:
            raise ValueError("set_labels length must match len(trials).")

        def _label_for(i: int):
            if not add_labels:
                return None, None
            tlabel = labels_trial[i]
            slabel = set_labels[i] if set_labels is not None else set_label
            return tlabel, slabel

        # ---- per-trial loop ----
        for i, trial in enumerate(trials):
            folder = Path(trial.get("folder", ".")) if trial.get("folder") else None
            flir_path = trial.get("flir_mat", None)

            cands: List[Path] = []
            if flir_path is not None and Path(flir_path).exists():
                cands.append(Path(flir_path))

            # Fallback: search in folder if needed
            if (not cands) and folder is not None and folder.exists():
                exact = folder / mat_name
                if exact.exists() and not exact.name.startswith("._"):
                    cands.append(exact)
                if not cands:
                    cands = [p for p in folder.glob("flir*.mat") if not p.name.startswith("._")]
                if not cands:
                    cands = [p for p in folder.glob("*.mat") if not p.name.startswith("._")]
                if not cands:
                    nested = [p for p in folder.rglob("*.mat") if not p.name.startswith("._")]
                    cands = nested

            # If still nothing, return empty df with labels/path (optionally)
            if not cands:
                df = pd.DataFrame()
                tlabel, slabel = _label_for(i)
                if add_labels:
                    if tlabel is not None:
                        df["trial"] = [tlabel]
                    if slabel is not None:
                        df["set_label"] = [slabel]
                if include_path and folder is not None:
                    df["source_path"] = [str(folder)]
                out.append(df)
                continue

            # Use largest .mat candidate
            cands_sorted = sorted(
                cands,
                key=lambda p: p.stat().st_size if p.exists() else 0,
                reverse=True,
            )
            mat_path = cands_sorted[0]

            try:
                df = self._mat_to_df(mat_path, prefix=prefix)
            except Exception as e:
                print(f"[WARN] Skipping unreadable MAT: {mat_path} ({e})")
                df = pd.DataFrame()

            # Stamp labels (don’t overwrite if already present)
            if add_labels and not df.empty:
                tlabel, slabel = _label_for(i)
                df = df.copy()
                if ("trial" not in df.columns) and (tlabel is not None):
                    df["trial"] = tlabel
                if ("set_label" not in df.columns) and (slabel is not None):
                    df["set_label"] = slabel
            elif add_labels and df.empty:
                # you could choose to create a 1-row df with labels here instead
                pass

            if include_path:
                df = df.copy()
                df["source_path"] = str(mat_path)

            out.append(df)

        return out

    # ---------- DLC (3D) CSV per trial ----------

    def extract_dlc3d_dfs_by_trial(
        self,
        trials: List[Dict[str, Optional[Path]]],
        file_patterns: tuple[str, ...] = (
            "*DLC*.csv", "*DLC3D*.csv", "*dlc3d*.csv", "*3d*.csv", "*3D*.csv"
        ),
        *,
        add_labels: bool = True,
        trial_labels: Optional[List[int]] = None,
        trial_base: int = 1,
        set_label: Optional[str] = None,
        set_labels: Optional[List[str]] = None,
        include_path: bool = False,
    ) -> List[pd.DataFrame]:
        """
        For each trial dict, load the DLC CSV (2D or 3D) for camera-1.

        Priority:
          1) trial["camera_csv"] if present
          2) any CSV in the folder matching file_patterns (largest by size)

        Keeps len(output) == len(trials).
        """
        out: List[pd.DataFrame] = []
        n = len(trials)

        if trial_labels is None:
            labels_trial = [trial_base + i for i in range(n)]
        else:
            if len(trial_labels) != n:
                raise ValueError("trial_labels length must match len(trials).")
            labels_trial = trial_labels

        if set_labels is not None and len(set_labels) != n:
            raise ValueError("set_labels length must match len(trials).")

        def _label_for(i: int):
            if not add_labels:
                return None, None
            tlabel = labels_trial[i]
            slabel = set_labels[i] if set_labels is not None else set_label
            return tlabel, slabel

        for i, trial in enumerate(trials):
            folder = Path(trial.get("folder", ".")) if trial.get("folder") else None
            cam_path = trial.get("camera_csv", None)

            cands: List[Path] = []
            if cam_path is not None and Path(cam_path).exists():
                cands.append(Path(cam_path))

            if (not cands) and folder is not None and folder.exists():
                # first try direct children
                for pat in file_patterns:
                    cands.extend([p for p in folder.glob(pat) if not p.name.startswith("._")])
                # fallback: recursive
                if not cands:
                    for pat in file_patterns:
                        cands.extend([p for p in folder.rglob(pat) if not p.name.startswith("._")])

            if not cands:
                df = pd.DataFrame()
                tlabel, slabel = _label_for(i)
                if add_labels:
                    if tlabel is not None:
                        df["trial"] = [tlabel]
                    if slabel is not None:
                        df["set_label"] = [slabel]
                if include_path and folder is not None:
                    df["source_path"] = [str(folder)]
                out.append(df)
                continue

            # Prefer largest file (full export)
            cands = sorted(
                set(cands),
                key=lambda p: p.stat().st_size if p.exists() else 0,
                reverse=True,
            )
            df0 = pd.read_csv(cands[0])
            if df0 is None or df0.empty:
                df = pd.DataFrame()
                tlabel, slabel = _label_for(i)
                if add_labels:
                    if tlabel is not None:
                        df["trial"] = [tlabel]
                    if slabel is not None:
                        df["set_label"] = [slabel]
                if include_path:
                    df["source_path"] = [str(cands[0])]
                out.append(df)
                continue

            df = df0.copy()

            # Stamp labels if requested and not present
            tlabel, slabel = _label_for(i)
            if add_labels:
                if ("trial" not in df.columns) and (tlabel is not None):
                    df["trial"] = tlabel
                if ("set_label" not in df.columns) and (slabel is not None):
                    df["set_label"] = slabel
            if include_path:
                df["source_path"] = str(cands[0])

            out.append(df)

        return out

    # ---------- DLC 3D angles per trial (MCP + wrist) ----------

    def compute_dlc3d_angles_by_trial(
        self,
        dlc3d_trials: List[pd.DataFrame],
        *,
        set_label: str,
        signed_in_plane: bool = True,
        add_plane_ok: bool = True,
    ) -> tuple[list[pd.DataFrame], pd.DataFrame]:
        """
        From per-trial DLC3D DataFrames (3-row MultiIndex columns), compute:
          • wrist_bend_deg  = angle(forearm→hand, hand→MCP)
          • mcp_bend_deg    = angle(hand→MCP, MCP→PIP)
          • mcp_bend_in_wrist_plane_deg = MCP bend projected into wrist plane

        This is the same logic as BallBearingData.compute_dlc3d_angles_by_trial,
        but attached to ADC_CAM.

        Returns (augmented_trials, tall_df).
        """
        import numpy as np
        import pandas as pd

        augmented: List[pd.DataFrame] = []
        parts: List[pd.DataFrame] = []

        for trial_idx, dlc_df in enumerate(dlc3d_trials, start=1):
            if dlc_df is None or dlc_df.empty:
                augmented.append(pd.DataFrame())
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
            v2_wrist = cam.vector(hand_pts, mcp_pts)      # hand→MCP

            angles_mcp = cam.angle_from_vectors(v1_mcp, v2_mcp)
            angles_wrist = cam.angle_from_vectors(v1_wrist, v2_wrist)

            angles_mcp_plane, _, _, plane_ok = cam.angle_from_vectors_in_plane(
                v1=v1_mcp,
                v2=v2_mcp,
                plane_v1=v1_wrist,
                plane_v2=v2_wrist,
                signed=signed_in_plane,
            )

            df_out = cam.df.copy()
            df_out[("metric", "mcp_bend_deg", "deg")] = angles_mcp
            df_out[("metric", "wrist_bend_deg", "deg")] = angles_wrist
            df_out[("metric", "mcp_bend_in_wrist_plane_deg", "deg")] = angles_mcp_plane
            if add_plane_ok:
                df_out[("metric", "wrist_plane_ok", "")] = plane_ok

            # Stamp flat labels for downstream merging
            df_out["set_label"] = set_label
            df_out["trial"] = int(trial_idx)

            augmented.append(df_out)

            # Try to carry a useful time axis if present
            time_col = None
            for c in df_out.columns:
                if "time" in str(c).lower() or "timestamp" in str(c).lower():
                    time_col = c
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
                    df_out[("metric", "mcp_bend_in_wrist_plane_deg", "deg")], errors="coerce"
                ),
                "wrist_plane_ok": plane_ok if add_plane_ok else True,
            }))

        tall = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(
            columns=[
                "set_label", "trial", "frame", "time_or_timestamp",
                "mcp_bend_deg", "wrist_bend_deg",
                "mcp_bend_in_wrist_plane_deg", "wrist_plane_ok",
            ]
        )
        return augmented, tall

        # ------------------------------------------------------------------
        # DLC3D helper: coerce flat DLC columns into a MultiIndex
        # ------------------------------------------------------------------

    @staticmethod
    def _coerce_dlc3d_multiindex(df: pd.DataFrame) -> pd.DataFrame:
        """
        Coerce a DLC 3D wide-format DataFrame into a 3-level MultiIndex
        (scorer, bodypart, coord), so it can be used with DLC3DBendAngles.

        Handles:
        - Standard DLC style where row 0 is 'bodyparts', row 1 is 'coords',
          and columns are ['scorer', scorer, scorer, ..., 'trial', 'set_label', 'source_path'].
        - Already-MultiIndex columns (returns a normalized copy).
        - Simple flat columns like 'PIP_x', 'PIP_y', etc. (2-level MI).

        Extra columns like 'trial', 'set_label', 'source_path' are
        preserved with ('meta', colname, '') so DLC3DBendAngles ignores them.
        """
        import pandas as pd
        import re

        if df is None or df.empty:
            return df

        cols = df.columns

        # --- Case 1: already a MultiIndex -> just normalize coord level ---
        if isinstance(cols, pd.MultiIndex):
            new_tuples = []
            for tup in cols:
                # ensure we have at least 3 levels; pad if needed
                if len(tup) == 1:
                    scorer, bp, coord = "meta", str(tup[0]), ""
                elif len(tup) == 2:
                    scorer, bp, coord = "meta", str(tup[0]), str(tup[1])
                else:
                    scorer, bp, coord = tup[0], tup[1], tup[2]

                # normalize coord to lowercase x/y/z/likelihood when possible
                c = str(coord).strip().lower()
                if c in ("x", "y", "z", "likelihood"):
                    coord = c
                new_tuples.append((scorer, bp, coord))

            df2 = df.copy()
            df2.columns = pd.MultiIndex.from_tuples(new_tuples)
            return df2

        # --- Case 2: DLC "wide" format with header rows for bodyparts/coords ---
        # We expect something like:
        #   row 0: ['bodyparts', 'PIP', 'PIP', 'PIP', 'MCP', 'MCP', ...]
        #   row 1: ['coords',    'x',   'y',   'likelihood', 'x', 'y', ...]
        try:
            first_col_name = str(df.columns[0]).lower()
            first_row0 = str(df.iloc[0, 0]).lower()
            first_row1 = str(df.iloc[1, 0]).lower()
        except Exception:
            first_col_name = first_row0 = first_row1 = ""

        if first_row0 == "bodyparts" and first_row1 == "coords":
            # build 3-level MI: (scorer, bodypart, coord) for DLC columns
            body_row = df.iloc[0]
            coord_row = df.iloc[1]
            new_cols = []
            meta_targets = {"trial", "set_label", "source_path"}

            for j, col in enumerate(df.columns):
                scorer = str(col)  # DLC scorer name or 'scorer' / 'trial' etc.
                bp = str(body_row[j])
                coord = str(coord_row[j])

                # For meta columns, put them in 'meta' so DLC3DBendAngles ignores.
                if bp in meta_targets or coord in meta_targets or scorer in meta_targets:
                    new_cols.append(("meta", scorer, ""))
                # The very first column is usually ('scorer', 'bodyparts', 'coords')
                elif scorer == "scorer" and bp == "bodyparts" and coord == "coords":
                    new_cols.append(("meta", "scorer", ""))
                else:
                    # Normalize coord: x/y/z/likelihood in lowercase
                    c = coord.strip().lower()
                    if c in ("x", "y", "z", "likelihood"):
                        coord = c
                    new_cols.append((scorer, bp, coord))

            df2 = df.iloc[2:].copy()  # drop header rows
            df2 = df2.reset_index(drop=True)
            df2.columns = pd.MultiIndex.from_tuples(new_cols)
            return df2

        # --- Case 3: flat columns with patterns like "hand_x", "MCP_y" ---
        if df.shape[1] >= 3:
            pat = re.compile(r"^(?P<bp>.+?)_(?P<coord>x|y|z)$", re.IGNORECASE)
            tuples = []
            good = 0
            for col in df.columns:
                name = str(col)
                m = pat.match(name)
                if m:
                    bp = m.group("bp")
                    coord = m.group("coord").lower()
                    tuples.append(("dlc", bp, coord))
                    good += 1
                else:
                    tuples.append(("meta", name, ""))

            if good >= 3:  # at least one full xyz set
                df2 = df.copy()
                df2.columns = pd.MultiIndex.from_tuples(tuples)
                return df2

        # Fallback: return as-is
        return df

    def attach_cam_timestamps_to_angles(
        self,
        cam_trials: List[pd.DataFrame],
        angle_trials: List[pd.DataFrame],
        *,
        time_col_name: str = "timestamp",
        new_col_name: str = "cam_timestamp",
    ) -> List[pd.DataFrame]:
        """
        Attach a camera timestamp column from cam_trials to the corresponding
        DLC angle trial DataFrames.

        Parameters
        ----------
        cam_trials : list[pd.DataFrame]
            List of FLIR MAT-derived DataFrames (e.g. from extract_mat_dfs_by_trial).
        angle_trials : list[pd.DataFrame]
            List of DLC angle DataFrames (e.g. from compute_dlc3d_angles_by_trial).
        time_col_name : str, default "timestamp"
            Name of the column in cam_trials to use as the time axis. If not
            found, the first numeric column is used.
        new_col_name : str, default "cam_timestamp"
            Name of the new column to add to each angle trial DataFrame.

        Returns
        -------
        out : list[pd.DataFrame]
            New list of DataFrames, same length as angle_trials, where each
            DataFrame is a copy of angle_trials[i] with one extra column
            new_col_name containing the FLIR time signal aligned by index:
              - if len(cam) >= len(angle): truncated to len(angle)
              - if len(cam) <  len(angle): padded with NaN at the end
        """
        import numpy as np

        if len(cam_trials) != len(angle_trials):
            raise ValueError(
                f"cam_trials (len={len(cam_trials)}) and angle_trials "
                f"(len={len(angle_trials)}) must have the same length."
            )

        out: List[pd.DataFrame] = []

        for idx, (cam_df, ang_df) in enumerate(zip(cam_trials, angle_trials)):
            # If angle trial is empty, just return a copy
            if ang_df is None or ang_df.empty:
                out.append(ang_df.copy())
                continue

            # If cam df is missing or empty -> create NaN timestamps
            if cam_df is None or cam_df.empty:
                ts = np.full(len(ang_df), np.nan)
            else:
                # Pick time column from cam_df
                if time_col_name in cam_df.columns:
                    time_series = pd.to_numeric(cam_df[time_col_name], errors="coerce").to_numpy()
                else:
                    # fall back: first numeric column
                    num_cols = cam_df.select_dtypes(include=[np.number]).columns
                    if len(num_cols) == 0:
                        time_series = np.full(len(cam_df), np.nan)
                    else:
                        time_series = pd.to_numeric(cam_df[num_cols[0]], errors="coerce").to_numpy()

                n_ang = len(ang_df)
                n_cam = len(time_series)

                if n_cam >= n_ang:
                    ts = time_series[:n_ang]
                else:
                    ts = np.full(n_ang, np.nan)
                    ts[:n_cam] = time_series

            df_out = ang_df.copy()
            # add as a flat column; mixing with MultiIndex is fine
            df_out[new_col_name] = ts
            out.append(df_out)

        return out

    def align_adc_theta_to_dlc_angles_for_set(
            self,
            dlc_angle_trials: List[pd.DataFrame],
            adc_theta_trials: List[pd.DataFrame],
            *,
            dlc_time_col: str = "cam_timestamp",
            adc_time_col: str = "timestamp",
            adc_cols: Optional[Sequence[str]] = None,
            time_unit: str = "s",  # <---- NEW: 's' for seconds, 'ms' for milliseconds, etc.
            tolerance="10ms",
            direction: str = "nearest",
            suffix: str = "_adc",
            keep_time_delta: bool = True,
            drop_unmatched: bool = True,
    ) -> List[pd.DataFrame]:
        """
        Align ADC-derived theta trials to DLC 3D angle trials using the same
        matching/attach logic as DLC3DBendAngles.find_matching_indices(...)
        and attach_encoder_using_match(...), but first convert numeric
        time columns to Timedelta with the given time_unit.

        Parameters
        ----------
        dlc_angle_trials : list[pd.DataFrame]
            List of per-trial DLC angle DataFrames, e.g. dlc3D_angles_first.
            Each must have a time column dlc_time_col (usually 'cam_timestamp').
        adc_theta_trials : list[pd.DataFrame]
            List of per-trial ADC theta DataFrames (adc_first_theta, etc.).
            Each must have a time column adc_time_col.
        dlc_time_col : str, default 'cam_timestamp'
            Column in dlc_angle_trials used as the camera time axis.
        adc_time_col : str, default 'timestamp'
            Column in adc_theta_trials used as the ADC time axis.
        adc_cols : sequence of str, optional
            Which ADC columns to attach (e.g. ['theta_cam_cal']).
            If None, all columns except adc_time_col are attached.
        time_unit : str, default 's'
            Unit for interpreting numeric timestamps in dlc_time_col /
            adc_time_col when converting to Timedelta. Common choices:
              - 's'  : seconds
              - 'ms' : milliseconds
              - 'us' : microseconds
        tolerance : str or number, default '10ms'
            Maximum allowed |t_cam - t_adc| difference. Can be '10ms',
            '0.05s', 5000 (us), etc. Passed directly to DLC3DBendAngles.
        direction : {'nearest','forward','backward'}, default 'nearest'
            Matching direction for asof-style merge.
        suffix : str, default '_adc'
            Suffix for attached ADC column names.
        keep_time_delta : bool, default True
            If True, keep a 'time_delta_adc' column (ms) to inspect match quality.
        drop_unmatched : bool, default True
            If True, drop DLC rows that have no matching ADC row.

        Returns
        -------
        merged_trials : list[pd.DataFrame]
            One merged DataFrame per trial, same length as dlc_angle_trials.
        """
        if len(dlc_angle_trials) != len(adc_theta_trials):
            raise ValueError(
                f"dlc_angle_trials (len={len(dlc_angle_trials)}) and "
                f"adc_theta_trials (len={len(adc_theta_trials)}) must have the same length."
            )

        merged: List[pd.DataFrame] = []

        for trial_idx, (dlc_df, adc_df) in enumerate(
                zip(dlc_angle_trials, adc_theta_trials), start=1
        ):
            # If either side is missing, return empty
            if dlc_df is None or dlc_df.empty or adc_df is None or adc_df.empty:
                merged.append(pd.DataFrame())
                continue

            # Ensure time columns exist
            if dlc_time_col not in dlc_df.columns:
                print(
                    f"[align] Trial {trial_idx}: DLC time column '{dlc_time_col}' not found."
                )
                merged.append(pd.DataFrame())
                continue

            if adc_time_col not in adc_df.columns:
                print(
                    f"[align] Trial {trial_idx}: ADC time column '{adc_time_col}' not found."
                )
                merged.append(pd.DataFrame())
                continue

            # Decide which ADC columns to attach
            if adc_cols is None:
                attach_cols = [c for c in adc_df.columns if c != adc_time_col]
            else:
                attach_cols = list(adc_cols)

            # --- Make local copies and coerce time to Timedelta ---
            dlc_local = dlc_df.copy()
            adc_local = adc_df.copy()

            try:
                dlc_local["_t_cam_td"] = pd.to_timedelta(
                    pd.to_numeric(dlc_local[dlc_time_col], errors="coerce"),
                    unit=time_unit,
                )
                adc_local["_t_adc_td"] = pd.to_timedelta(
                    pd.to_numeric(adc_local[adc_time_col], errors="coerce"),
                    unit=time_unit,
                )
            except Exception as e:
                print(
                    f"[align] Trial {trial_idx}: failed to convert timestamps to Timedelta "
                    f"using unit='{time_unit}' ({e}); returning empty."
                )
                merged.append(pd.DataFrame())
                continue

            # --- Use DLC3DBendAngles purely as a matching helper ---
            try:
                dlc_cam_obj = DLC3DBendAngles(dlc_local)

                # 1) Build match map: note we pass our temporary TD columns as keys
                dlc_cam_obj.find_matching_indices(
                    encoder_df=adc_local,
                    cam_time_col="_t_cam_td",
                    enc_time_col="_t_adc_td",
                    tolerance=tolerance,
                    direction=direction,
                )

                # 2) Attach ADC columns onto DLC rows
                merged_df = dlc_cam_obj.attach_encoder_using_match(
                    encoder_df=adc_local,
                    columns=attach_cols,
                    suffix=suffix,
                    keep_time_delta=keep_time_delta,
                    drop_unmatched=drop_unmatched,
                )
                merged.append(merged_df)
                continue

            except Exception as e:
                print(
                    f"[align] Trial {trial_idx}: DLC-based alignment failed ({e}); "
                    f"returning empty DataFrame."
                )
                merged.append(pd.DataFrame())
                continue

        return merged

    @staticmethod
    def _refine_alignment_by_rmse_single(
        df: pd.DataFrame,
        dlc_col=("metric", "mcp_bend_deg", "deg"),
        adc_col: str = "theta_cam_cal_adc",
        max_lag_samples: int = 5,
    ) -> Tuple[pd.DataFrame, int, float]:
        """
        Internal helper: given a merged trial (DLC + ADC on same rows),
        search over small integer lags of the ADC signal to find the shift
        that minimizes RMSE between DLC and ADC.

        Parameters
        ----------
        df : pd.DataFrame
            One merged trial from merged_first / merged_second.
        dlc_col : column label
            DLC angle column (MultiIndex or flat).
        adc_col : str
            ADC angle column.
        max_lag_samples : int
            Max |lag| in samples to search (e.g. 5 => -5..+5).

        Returns
        -------
        df_aligned : pd.DataFrame
            Copy of df with a new column adc_col + '_rmse' containing the
            best-lag ADC curve, and only rows where both DLC and ADC exist.
        best_lag : int
            Best lag in samples (ADC shifted relative to DLC).
            >0 means ADC shifted left (ADC originally later),
            <0 means shifted right.
        best_rmse : float
            RMSE (deg) at the best lag.
        """
        theta_dlc = df[dlc_col].to_numpy(dtype=float)
        theta_adc = df[adc_col].to_numpy(dtype=float)

        best_rmse = np.inf
        best_lag = 0
        best_shifted = None

        for lag in range(-max_lag_samples, max_lag_samples + 1):
            shifted = np.full_like(theta_adc, np.nan, dtype=float)

            if lag > 0:
                # ADC later -> shift left
                shifted[:-lag] = theta_adc[lag:]
            elif lag < 0:
                # ADC earlier -> shift right
                L = -lag
                shifted[L:] = theta_adc[:-L]
            else:
                shifted[:] = theta_adc

            both_valid = np.isfinite(theta_dlc) & np.isfinite(shifted)
            if both_valid.sum() < 5:
                continue

            diff = theta_dlc[both_valid] - shifted[both_valid]
            rmse = np.sqrt(np.nanmean(diff**2))

            if rmse < best_rmse:
                best_rmse = rmse
                best_lag = lag
                best_shifted = shifted

        if best_shifted is None:
            # nothing worked; return original (no extra column)
            return df.copy(), 0, np.nan

        df_out = df.copy()
        new_col = adc_col + "_rmse"
        df_out[new_col] = best_shifted

        # keep only overlap where both DLC and best-shifted ADC exist
        both_valid = np.isfinite(theta_dlc) & np.isfinite(best_shifted)
        df_out = df_out.loc[both_valid].reset_index(drop=True)

        return df_out, best_lag, best_rmse

    def refine_alignment_by_rmse_for_set(
            self,
            merged_trials: List[pd.DataFrame],
            *,
            dlc_col=("metric", "mcp_bend_deg", "deg"),
            adc_col: str = "theta_cam_cal_adc",
            max_lag_samples: int = 5,
            time_col=("_t_cam_td", "", ""),  # or ("cam_timestamp", "", "")
            plot_indices: Optional[Sequence[int]] = None,
            set_name: str = "set",
            do_refine: bool = True,  # <--- NEW FLAG
    ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        """
        Refine ADC vs DLC alignment for an entire set of merged trials by
        searching over small integer lags (in samples) to minimize RMSE.
        Optionally plots selected trials.

        do_refine : bool, default True
            If True, search over ±max_lag_samples to minimize RMSE.
            If False, no further alignment is done (lag constrained to 0),
            but the _rmse column is still computed for diagnostics.

        Plot behavior:
          - Time axis is always rescaled to 0–10 s (data unchanged).
          - A black vertical dashed line is drawn at the sample with the
            largest absolute difference between the two curves, connecting
            the DLC and ADC values at that time.

        Parameters
        ----------
        merged_trials : list[pd.DataFrame]
            Output from align_adc_theta_to_dlc_angles_for_set (e.g. merged_first).
        dlc_col : column label
            DLC angle column (MultiIndex or flat). Default is MCP bend in deg.
        adc_col : str, default 'theta_cam_cal_adc'
            ADC angle column name to refine.
        max_lag_samples : int, default 5
            Max |lag| in samples to search (e.g. 5 => -5..+5).
        time_col : column label, default ('_t_cam_td', '', '')
            Column used for plotting time. If not found, falls back to
            ('cam_timestamp', '', '') or to sample index.
        plot_indices : sequence of int or None
            If not None, indices of trials to plot after refinement.
        set_name : str
            Label used in plot titles (e.g. 'FIRST', 'SECOND').

        Returns
        -------
        refined_trials : list[pd.DataFrame]
            List of per-trial DataFrames with an extra column adc_col + '_rmse'
            containing the best-lag ADC curve, trimmed to overlapping rows.
        summary : pd.DataFrame
            Table with columns:
              ['trial_index', 'best_lag_samples', 'rmse_deg', 'n_points']
        """

        # If do_refine is False, force lag search to be 0 only (no extra alignment)
        effective_max_lag = max_lag_samples if do_refine else 0
        refined: List[pd.DataFrame] = []
        records = []

        for i, df in enumerate(merged_trials):
            if df is None or df.empty:
                refined.append(pd.DataFrame())
                records.append(
                    dict(
                        trial_index=i,
                        best_lag_samples=0,
                        rmse_deg=np.nan,
                        n_points=0,
                    )
                )
                continue

            df_aligned, best_lag, best_rmse = self._refine_alignment_by_rmse_single(
                df,
                dlc_col=dlc_col,
                adc_col=adc_col,
                max_lag_samples=effective_max_lag,
            )
            refined.append(df_aligned)
            records.append(
                dict(
                    trial_index=i,
                    best_lag_samples=best_lag,
                    rmse_deg=best_rmse,
                    n_points=len(df_aligned),
                )
            )

        summary = pd.DataFrame.from_records(records)

        # --- Optional plotting ---
        if plot_indices is not None:
            for idx in plot_indices:
                if idx < 0 or idx >= len(refined):
                    continue
                df_pl = refined[idx]
                if df_pl is None or df_pl.empty:
                    continue

                # ----- Build time axis in seconds -----
                if time_col in df_pl.columns:
                    t_raw = df_pl[time_col]
                    if hasattr(t_raw, "dt"):
                        t = (t_raw - t_raw.iloc[0]).dt.total_seconds().to_numpy()
                    else:
                        t = t_raw.to_numpy(dtype=float)
                        t = t - t[0]
                elif ("_t_cam_td", "", "") in df_pl.columns:
                    t_raw = df_pl[("_t_cam_td", "", "")]
                    t = (t_raw - t_raw.iloc[0]).dt.total_seconds().to_numpy()
                elif ("cam_timestamp", "", "") in df_pl.columns:
                    t = df_pl[("cam_timestamp", "", "")].to_numpy(dtype=float)
                    t = t - t[0]
                else:
                    t = np.arange(len(df_pl), dtype=float)

                # ----- Rescale time axis to 0–10 s for plotting -----
                if len(t) > 1 and np.nanmax(t) > np.nanmin(t):
                    t_plot = (t - np.nanmin(t)) / (np.nanmax(t) - np.nanmin(t)) * 10.0
                else:
                    t_plot = np.zeros_like(t, dtype=float)

                theta_dlc = df_pl[dlc_col]
                theta_adc_rmse = df_pl[adc_col + "_rmse"]

                # Compute absolute error to find worst point
                dlc_arr = theta_dlc.to_numpy(dtype=float)
                adc_arr = theta_adc_rmse.to_numpy(dtype=float)
                both_valid = np.isfinite(dlc_arr) & np.isfinite(adc_arr)

                idx_max = None
                if both_valid.sum() > 0:
                    abs_diff = np.full_like(dlc_arr, np.nan, dtype=float)
                    abs_diff[both_valid] = np.abs(dlc_arr[both_valid] - adc_arr[both_valid])
                    idx_max = int(np.nanargmax(abs_diff))

                # ----- Plot -----
                plt.figure(figsize=(10, 4))
                plt.plot(t_plot, theta_dlc, label="DLC MCP angle")
                plt.plot(t_plot, theta_adc_rmse, label="ADC MCP angle (RMSE-aligned)")

                # Vertical line at max absolute difference
                if idx_max is not None and np.isfinite(dlc_arr[idx_max]) and np.isfinite(adc_arr[idx_max]):
                    x0 = t_plot[idx_max]
                    y1 = dlc_arr[idx_max]
                    y2 = adc_arr[idx_max]
                    plt.vlines(
                        x0,
                        ymin=min(y1, y2),
                        ymax=max(y1, y2),
                        colors="k",
                        linestyles="--",
                        linewidth=2,
                        label="max |Δ|",
                    )

                plt.xlabel("Scaled time (0-10 s)")
                plt.ylabel("Angle (°)")
                plt.title(
                    f"{set_name} trial {idx + 1}: DLC vs ADC MCP (RMSE-optimal shift)"
                )
                plt.xlim(0.0, 10.0)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()

        return refined, summary

    def plot_abs_error_box_two_sets(
            self,
            refined_first: List[pd.DataFrame],
            summary_first: pd.DataFrame,
            refined_second: List[pd.DataFrame],
            summary_second: pd.DataFrame,
            *,
            dlc_col=("metric", "mcp_bend_deg", "deg"),
            adc_rmse_col: str = "theta_cam_cal_adc_rmse",
            trial_indices_first: Optional[Sequence[int]] = None,
            trial_indices_second: Optional[Sequence[int]] = None,
            n_best_first: int = 2,
            n_best_second: int = 2,
            label_first: str = "FIRST",
            label_second: str = "SECOND",
    ) -> None:
        """
        Plot TWO side-by-side boxplots of absolute error:

            left  box = combined abs error from selected FIRST-set trials
            right box = combined abs error from selected SECOND-set trials

        You can either provide explicit trial indices for each set or let this
        method choose the n_best trials with smallest RMSE in each summary.

        Parameters
        ----------
        refined_first, refined_second : list[pd.DataFrame]
            Outputs from refine_alignment_by_rmse_for_set for each set.
        summary_first, summary_second : pd.DataFrame
            Summaries from refine_alignment_by_rmse_for_set. Must contain
            ['trial_index', 'rmse_deg', 'n_points'].
        dlc_col : column label, default ('metric','mcp_bend_deg','deg')
            DLC angle column used to compute error.
        adc_rmse_col : str, default 'theta_cam_cal_adc_rmse'
            Column name containing RMSE-aligned ADC angle in each trial.
        trial_indices_first, trial_indices_second : sequence[int] or None
            If provided, use these trials for each set. If None, choose
            n_best_first / n_best_second based on smallest RMSE.
        n_best_first, n_best_second : int
            Number of best-RMSE trials to use when explicit indices are not given.
        label_first, label_second : str
            Labels for the two boxplots (e.g. 'FIRST', 'SECOND').
        """

        def _collect_abs_errors(
                refined: List[pd.DataFrame],
                summary: pd.DataFrame,
                trial_indices: Optional[Sequence[int]],
                n_best: int,
        ) -> np.ndarray:
            if summary is None or summary.empty:
                return np.array([])

            valid = summary.dropna(subset=["rmse_deg"])
            valid = valid[valid["n_points"] > 0]
            if valid.empty:
                return np.array([])

            # Decide which trials
            if trial_indices is not None:
                idxs = [int(i) for i in trial_indices]
                chosen = valid[valid["trial_index"].isin(idxs)]
                if chosen.empty:
                    return np.array([])
            else:
                chosen = valid.sort_values("rmse_deg", ascending=True).head(n_best)

            all_errs = []

            for _, row in chosen.iterrows():
                idx = int(row["trial_index"])
                if idx < 0 or idx >= len(refined):
                    continue
                df_trial = refined[idx]
                if df_trial is None or df_trial.empty:
                    continue
                if adc_rmse_col not in df_trial.columns:
                    continue

                theta_dlc = df_trial[dlc_col].to_numpy(dtype=float)
                theta_adc = df_trial[adc_rmse_col].to_numpy(dtype=float)

                mask = np.isfinite(theta_dlc) & np.isfinite(theta_adc)
                if mask.sum() < 5:
                    continue

                abs_err = np.abs(theta_dlc[mask] - theta_adc[mask])
                all_errs.append(abs_err)

            if not all_errs:
                return np.array([])

            return np.concatenate(all_errs)

        # Collect abs error from each set
        errs_first = _collect_abs_errors(
            refined_first, summary_first, trial_indices_first, n_best_first
        )
        errs_second = _collect_abs_errors(
            refined_second, summary_second, trial_indices_second, n_best_second
        )

        if errs_first.size == 0 and errs_second.size == 0:
            print("[boxplot-two] No valid errors from either set; nothing to plot.")
            return

        data = []
        labels = []

        if errs_first.size > 0:
            data.append(errs_first)
            labels.append(f"{label_first}\nN={errs_first.size}")
        else:
            print(f"[boxplot-two] No valid abs error for {label_first} set.")

        if errs_second.size > 0:
            data.append(errs_second)
            labels.append(f"{label_second}\nN={errs_second.size}")
        else:
            print(f"[boxplot-two] No valid abs error for {label_second} set.")

        if not data:
            return

        # Two blue-ish boxes, no fliers
        plt.figure(figsize=(7, 5))
        bp = plt.boxplot(
            data,
            labels=labels,
            showfliers=False,  # no black circles
            patch_artist=True,  # enable facecolor
            widths=0.3, 
        )

        # Color boxes: both blue-ish, slightly different shades
        colors = [COLOR_FIRST, COLOR_SECOND]
        for box, color in zip(bp["boxes"], colors[: len(bp["boxes"])]):
            box.set_facecolor(color)
            box.set_alpha(0.65)

        plt.ylabel("Absolute error |DLC - ADC| (°)")
        plt.title("Absolute error comparison between sets")
        plt.grid(axis="y", alpha=0.4)
        plt.tight_layout()
        plt.show()

    def collect_abs_error_for_set(
            self,
            refined_trials: List[pd.DataFrame],
            summary: pd.DataFrame,
            *,
            dlc_col=("metric", "mcp_bend_deg", "deg"),
            adc_rmse_col: str = "theta_cam_cal_adc_rmse",
            trial_indices: Optional[Sequence[int]] = None,
            n_best: int = 2,
    ) -> np.ndarray:
        """
        Collect a single concatenated array of |DLC - ADC| errors for one set
        (FIRST or SECOND), given refined trials + summary.

        You will typically provide trial_indices explicitly, e.g. [0, 1].
        If trial_indices is None, it falls back to the n_best trials with
        smallest RMSE.

        Parameters
        ----------
        refined_trials : list[pd.DataFrame]
            Output from refine_alignment_by_rmse_for_set (e.g. refined_first).
        summary : pd.DataFrame
            Summary from refine_alignment_by_rmse_for_set, must contain
            ['trial_index', 'rmse_deg', 'n_points'].
        dlc_col : column label
            DLC angle column (MultiIndex or flat).
        adc_rmse_col : str, default 'theta_cam_cal_adc_rmse'
            Column name containing RMSE-aligned ADC angle.
        trial_indices : sequence[int] or None
            If provided, use exactly these trial indices.
        n_best : int, default 2
            If trial_indices is None, choose the n_best trials with
            smallest RMSE.

        Returns
        -------
        abs_err_all : np.ndarray
            Concatenated absolute errors |DLC - ADC| in degrees.
            Empty array if nothing valid.
        """
        if summary is None or summary.empty:
            return np.array([])

        valid = summary.dropna(subset=["rmse_deg"])
        valid = valid[valid["n_points"] > 0]
        if valid.empty:
            return np.array([])

        # Decide which trials
        if trial_indices is not None:
            idxs = [int(i) for i in trial_indices]
            chosen = valid[valid["trial_index"].isin(idxs)]
            if chosen.empty:
                return np.array([])
        else:
            chosen = valid.sort_values("rmse_deg", ascending=True).head(n_best)

        all_errs = []

        for _, row in chosen.iterrows():
            idx = int(row["trial_index"])
            if idx < 0 or idx >= len(refined_trials):
                continue

            df_trial = refined_trials[idx]
            if df_trial is None or df_trial.empty:
                continue

            if adc_rmse_col not in df_trial.columns:
                continue

            theta_dlc = df_trial[dlc_col].to_numpy(dtype=float)
            theta_adc = df_trial[adc_rmse_col].to_numpy(dtype=float)

            mask = np.isfinite(theta_dlc) & np.isfinite(theta_adc)
            if mask.sum() < 5:
                continue

            abs_err = np.abs(theta_dlc[mask] - theta_adc[mask])
            all_errs.append(abs_err)

        if not all_errs:
            return np.array([])

        return np.concatenate(all_errs)

    def reapply_first_calibration_to_second(
            self,
            calib_df: pd.DataFrame,
            adc_trials_second: List[pd.DataFrame],
            *,
            adc_column: str = "adc_ch3",
            poly_order: int = 2,
            deg_min: float = 0.0,
            deg_max: float = 90.0,
            new_col: str = "theta_cam_cal_xtrained",
            clamp_theta: bool = True,
    ) -> dict:
        """
        Build a *cross-trained* calibration for the second application (set=2),
        using the SHAPE of the first application's calibration curve, but
        re-scaled to the endpoint ADC range of the second calibration data.

        Physical intent
        ----------------
        For both applications we want:
            ADC(0 deg)  = max ADC
            ADC(90 deg) = min ADC

        Here we:
          1) Fit an inverse polynomial q1(theta) ~ ADC_1(theta) from set=1.
          2) Evaluate that on [deg_min, deg_max] to get adc1_grid(theta).
          3) Normalize adc1_grid using the *endpoint ADCs* at 0° and 90°,
             so orientation is preserved.
          4) Rescale this normalized curve to match the 0° and 90° ADC
             endpoints of set=2, giving adc2_cross_grid(theta).
          5) Fit p_cross(adc) ~ theta on (adc2_cross_grid, theta_grid).
          6) Apply p_cross to ALL second-application trials, writing into
             `new_col` (default 'theta_cam_cal_xtrained').

        Parameters
        ----------
        calib_df : pd.DataFrame
            Output of self.calibrate_trials_with_camera()['calib_df'].
            Must contain columns 'set', 'angle_snap_deg', 'adc_mean'.
        adc_trials_second : list[pd.DataFrame]
            Raw ADC DataFrames for second application (B2_*).
        adc_column : str, default 'adc_ch3'
            ADC column name in the trials.
        poly_order : int, default 2
            Polynomial order for the inverse and cross fits.
        deg_min, deg_max : float
            Angle range (deg) to span when constructing the grid.
        new_col : str, default 'theta_cam_cal_xtrained'
            Name of the angle column to add to each second-application trial.
        clamp_theta : bool, default True
            If True, clamp angles to [deg_min, deg_max].

        Returns
        -------
        result : dict
            {
              "coeffs_cross_2_from_1": np.ndarray or None,
              "adc_trials_second_theta_cross": list[pd.DataFrame],
            }
        """
        import numpy as np
        import pandas as pd

        if calib_df is None or calib_df.empty:
            second_out = [df.copy() for df in adc_trials_second]
            return {
                "coeffs_cross_2_from_1": None,
                "adc_trials_second_theta_cross": second_out,
            }

        # --- Split calib_df into set 1 and set 2 ---
        sub1 = calib_df[
            (calib_df["set"] == 1) & calib_df["angle_snap_deg"].notna()
            ].copy()
        sub2 = calib_df[
            (calib_df["set"] == 2) & calib_df["angle_snap_deg"].notna()
            ].copy()

        if sub1.empty or sub2.empty:
            second_out = [df.copy() for df in adc_trials_second]
            return {
                "coeffs_cross_2_from_1": None,
                "adc_trials_second_theta_cross": second_out,
            }

        if len(sub1) < poly_order + 1:
            second_out = [df.copy() for df in adc_trials_second]
            return {
                "coeffs_cross_2_from_1": None,
                "adc_trials_second_theta_cross": second_out,
            }

        # --- 1) Fit inverse polynomial for set 1: angle -> ADC_1 ---
        theta1 = sub1["angle_snap_deg"].to_numpy(dtype=float)
        adc1 = sub1["adc_mean"].to_numpy(dtype=float)

        coeffs_inv_1 = np.polyfit(theta1, adc1, poly_order)  # q1(theta) -> ADC

        # --- 2) Build angle grid and ADC_1 curve ---
        theta_grid = np.linspace(deg_min, deg_max, 181)  # e.g., 0..90 in 0.5° steps
        adc1_grid = np.polyval(coeffs_inv_1, theta_grid)

        # ---------- NORMALIZATION / RESCALING PATCH START ----------
        # We normalize using endpoint ADCs at 0° and 90°
        def _angle_specific_adc(sub, angle_target):
            sub_a = sub[np.isclose(sub["angle_snap_deg"], angle_target)]
            if not sub_a.empty:
                return float(sub_a["adc_mean"].mean())
            return None

        # Endpoint ADCs for set 1
        adc1_0 = _angle_specific_adc(sub1, deg_min)  # ~ ADC(0°)
        adc1_90 = _angle_specific_adc(sub1, deg_max)  # ~ ADC(90°)

        # Fallbacks if exact endpoints are missing
        if adc1_0 is None:
            # For your sensor, 0° ≈ max ADC
            adc1_0 = float(sub1["adc_mean"].max())
        if adc1_90 is None:
            # For your sensor, 90° ≈ min ADC
            adc1_90 = float(sub1["adc_mean"].min())

        if adc1_90 == adc1_0:
            second_out = [df.copy() for df in adc_trials_second]
            return {
                "coeffs_cross_2_from_1": None,
                "adc_trials_second_theta_cross": second_out,
            }

        # Normalize so that:
        #   z(theta = 0°)  = 0
        #   z(theta = 90°) = 1
        z_grid = (adc1_grid - adc1_0) / (adc1_90 - adc1_0)

        # Endpoint ADCs for set 2 (target range)
        adc2_0 = _angle_specific_adc(sub2, deg_min)  # ADC_2(0°)
        adc2_90 = _angle_specific_adc(sub2, deg_max)  # ADC_2(90°)

        if adc2_0 is None:
            adc2_0 = float(sub2["adc_mean"].max())
        if adc2_90 is None:
            adc2_90 = float(sub2["adc_mean"].min())

        if adc2_90 == adc2_0:
            second_out = [df.copy() for df in adc_trials_second]
            return {
                "coeffs_cross_2_from_1": None,
                "adc_trials_second_theta_cross": second_out,
            }

        # Rescale normalized shape to set 2 endpoints:
        #   z=0 -> adc2_0  (0°)
        #   z=1 -> adc2_90 (90°)
        adc2_cross_grid = adc2_0 + z_grid * (adc2_90 - adc2_0)
        # ---------- NORMALIZATION / RESCALING PATCH END ----------

        # --- 3) Fit forward polynomial p_cross(adc) ≈ theta using cross grid ---
        coeffs_cross = np.polyfit(adc2_cross_grid, theta_grid, poly_order)

        # --- 4) Apply p_cross to ALL second-application ADC trials ---
        second_out: List[pd.DataFrame] = []
        for df in adc_trials_second:
            if df is None or df.empty or adc_column not in df.columns:
                second_out.append(df.copy())
                continue

            df2 = df.copy()
            adc_vals = pd.to_numeric(df2[adc_column], errors="coerce").to_numpy()
            theta_est = np.polyval(coeffs_cross, adc_vals)

            if clamp_theta:
                theta_est = np.clip(theta_est, deg_min, deg_max)

            df2[new_col] = theta_est
            second_out.append(df2)

        return {
            "coeffs_cross_2_from_1": coeffs_cross,
            "adc_trials_second_theta_cross": second_out,
        }

    def filter_angle_trials_by_likelihood(
            self,
            angle_trials: Sequence[pd.DataFrame],
            *,
            bodyparts: Sequence[str] = ("MCP", "PIP", "hand"),
            min_likelihood: float = 0.95,
            inplace: bool = False,
    ) -> List[pd.DataFrame]:
        """
        Filter each DLC-angle trial DataFrame by DeepLabCut likelihood.

        Any row in which *any* of the specified bodyparts has likelihood
        < min_likelihood is dropped.

        Works for:
        - MultiIndex columns (scorer, bodypart, coord) with coord = 'likelihood'
        - Flat columns like 'MCP_likelihood'

        Parameters
        ----------
        angle_trials : sequence of DataFrame
            Typically the output of compute_dlc3d_angles_by_trial(...)
            or attach_cam_timestamps_to_angles(...).
        bodyparts : sequence of str, default ("MCP", "PIP", "hand")
            Bodyparts whose likelihood columns must exceed the threshold.
            Matching is case-insensitive.
        min_likelihood : float, default 0.95
            Threshold that likelihood values must meet or exceed.
        inplace : bool, default False
            If True, returns filtered views of the original DataFrames.
            If False, returns copies.

        Returns
        -------
        out : list[DataFrame]
            One filtered DataFrame per input trial.
        """
        out: List[pd.DataFrame] = []

        # Normalize bodypart names for case-insensitive matching
        bp_norm = {bp.lower() for bp in bodyparts}

        for df in angle_trials:
            if df is None or df.empty:
                out.append(df)
                continue

            keep_mask = pd.Series(True, index=df.index)
            cols = df.columns

            # ---------- Case 1: MultiIndex DLC columns ----------
            if isinstance(cols, pd.MultiIndex):
                # levels: (scorer, bodypart, coord)
                level1 = cols.get_level_values(1).to_numpy().astype(str)
                level2 = cols.get_level_values(2).to_numpy().astype(str)

                for bp in bp_norm:
                    is_bp = np.char.lower(level1) == bp
                    is_lik = np.char.lower(level2) == "likelihood"
                    bp_lik_cols = cols[is_bp & is_lik]

                    if len(bp_lik_cols) == 0:
                        # This bodypart isn't present; skip
                        continue

                    # Coerce to numeric in case they're stored as strings
                    lik_vals = df[bp_lik_cols].apply(
                        pd.to_numeric, errors="coerce"
                    )

                    lik_ok = (lik_vals >= min_likelihood).all(axis=1)
                    # treat NaNs as failures (drop those rows)
                    lik_ok = lik_ok & lik_vals.notna().all(axis=1)

                    keep_mask &= lik_ok

            # ---------- Case 2: flat columns like "MCP_likelihood" ----------
            else:
                flat_cols = [str(c) for c in cols]
                flat_lower = [c.lower() for c in flat_cols]

                for bp in bp_norm:
                    candidate = None
                    target = f"{bp}_likelihood"
                    for name, low in zip(flat_cols, flat_lower):
                        if low == target:
                            candidate = name
                            break

                    if candidate is None:
                        continue

                    lik_vals = pd.to_numeric(df[candidate], errors="coerce")
                    lik_ok = lik_vals >= min_likelihood
                    lik_ok = lik_ok & lik_vals.notna()

                    keep_mask &= lik_ok

            # Apply mask
            if inplace:
                out.append(df.loc[keep_mask])
            else:
                out.append(df.loc[keep_mask].copy())

        return out

    # ---------- trigger-time CSV per trial ----------

    def extract_trigger_time_dfs_by_trial(
        self,
        trials: List[Dict[str, Optional[Path]]],
        *,
        add_labels: bool = True,
        trial_labels: Optional[List[int]] = None,  # else trial_base..N-1
        trial_base: int = 1,
        set_label: Optional[str] = None,          # e.g. "first_cam", "second_cam"
        set_labels: Optional[List[str]] = None,   # per-trial labels
        include_path: bool = False,
    ) -> List[pd.DataFrame]:
        """
        For each trial dict, load the trigger-time CSV (if present)
        into a small DataFrame of timestamps.

        Priority (for each trial):
          1) trial["trig_time_csv"] if present
          2) any CSV in the folder matching '*data_trigger_time*.csv'

        If a trial has no trigger-time CSV, an empty DataFrame is returned
        for that trial (optionally with label/path columns).
        """
        out: List[pd.DataFrame] = []
        n = len(trials)

        # ---- build labels ----
        if trial_labels is None:
            labels_trial = [trial_base + i for i in range(n)]
        else:
            if len(trial_labels) != n:
                raise ValueError("trial_labels length must match len(trials).")
            labels_trial = trial_labels

        if set_labels is not None and len(set_labels) != n:
            raise ValueError("set_labels length must match len(trials).")

        def _label_for(i: int):
            if not add_labels:
                return None, None
            tlabel = labels_trial[i]
            slabel = set_labels[i] if set_labels is not None else set_label
            return tlabel, slabel

        # ---- per-trial loop ----
        for i, trial in enumerate(trials):
            folder = Path(trial.get("folder", ".")) if trial.get("folder") else None
            trig_path = trial.get("trig_time_csv", None)

            cands: List[Path] = []
            if trig_path is not None and Path(trig_path).exists():
                cands.append(Path(trig_path))

            # Fallback: search in folder if needed
            if (not cands) and folder is not None and folder.exists():
                direct = [
                    p for p in folder.glob("*data_trigger_time*.csv")
                    if not p.name.startswith("._")
                ]
                cands.extend(direct)
                if not cands:
                    nested = [
                        p for p in folder.rglob("*data_trigger_time*.csv")
                        if not p.name.startswith("._")
                    ]
                    cands.extend(nested)

            # If still nothing, return empty df with labels/path (optionally)
            if not cands:
                df = pd.DataFrame()
                tlabel, slabel = _label_for(i)
                if add_labels:
                    if tlabel is not None:
                        df["trial"] = [tlabel]
                    if slabel is not None:
                        df["set_label"] = [slabel]
                if include_path and folder is not None:
                    df["source_path"] = [str(folder)]
                out.append(df)
                continue

            # Use largest CSV candidate
            cands_sorted = sorted(
                cands,
                key=lambda p: p.stat().st_size if p.exists() else 0,
                reverse=True,
            )
            csv_path = cands_sorted[0]

            try:
                df = pd.read_csv(csv_path)
            except Exception as e:
                print(f"[WARN] Skipping unreadable trigger-time CSV: {csv_path} ({e})")
                df = pd.DataFrame()

            # If there is exactly one column and it looks like time, rename to "timestamp"
            if not df.empty and df.shape[1] == 1:
                only = df.columns[0]
                name_l = str(only).lower()
                if ("time" in name_l) or ("ts" in name_l):
                    df = df.rename(columns={only: "timestamp"})

            # Add labels if requested
            tlabel, slabel = _label_for(i)
            if add_labels and not df.empty:
                df = df.copy()
                if ("trial" not in df.columns) and (tlabel is not None):
                    df["trial"] = tlabel
                if ("set_label" not in df.columns) and (slabel is not None):
                    df["set_label"] = slabel
            elif add_labels and df.empty:
                # already handled above in the "no cands" branch
                pass

            if include_path:
                df = df.copy()
                df["source_path"] = str(csv_path)

            out.append(df)

        return out

    @staticmethod
    def plot_adc_abs_error_summary_grid(
            results_adc: dict,
            *,
            speed_titles: dict | None = None,
            speed_order: list[str] | None = None,
            dlc_angle_col=("metric", "mcp_bend_deg", "deg"),
            adc_angle_col: str = "theta_cam_cal_adc",
            # Example row metadata (for titles/labels only)
            example1_participant: str = "P4",
            example1_speed: str = "slow",
            example1_trial_idx: int = 0,
            example2_participant: str = "P3",
            example2_speed: str = "vfas",
            example2_trial_idx: int = 0,
            # explicit aligned example DataFrames
            example1_df=None,
            example2_df=None,
            # calibration data to show in row 1, col 2
            calib_df=None,
            calib_set_label: int = 2,  # 1 = FIRST (blue), 2 = SECOND (orange)
            example_col_start: int = 1,
            calib_x_shift: float = 0.0,  # shift calib panel to the right # layout edit
            figsize: tuple[float, float] = (8.5, 8), # (20,12)
            title_fontsize: int = 12,
            label_fontsize: int = 10,
            tick_fontsize: int = 9,
            title_weight: str = "bold",
            label_weight: str = "normal",
            # global y-limit (fallback)
            abs_err_ylim: tuple[float, float] | None = None,
            # separate tuners
            abs_err_ylim_examples: tuple[float, float] | None = None,
            abs_err_ylim_summary: tuple[float, float] | None = None,
            # horizontal spacing for whole grid
            boxplot_wspace: float = 0.02,
            # gap between TS and box in rows 0–1
            example_col_gap_scale: float = 1.0,
            # fixed gap between AFAP boxplot and summary bar (rows 2–3)
            bar_gap_from_afap: float = 0.33,
            # drop specific participant–set combos from summary rows
            exclude_participant_set_for_summary: list[tuple[str, str]] | None = None,
    ):
        """
        Seaborn summary figure for ADC vs DLC abs-error results.

        Rows 1–2 use *explicit* aligned example DataFrames (example1_df, example2_df).
        Rows 3–4 use abs-error summaries from results_adc.

        Row 1:
          [0, calib_col]          : calibration scatter (SECOND, orange; 11/02/25)
          [0, example_col_start+1]: DLC vs ADC (0–10 s) example 1
          [0, example_col_start+2]: boxplot of |DLC–ADC| example 1

        Row 2:
          [1, example_col_start+1]: DLC vs ADC (0–10 s) example 2
          [1, example_col_start+2]: boxplot of |DLC–ADC| example 2

        Row 3:
          FIRST set – per-speed boxplots by participant (cols 0–4),
          plus mean |error| vs speed bar plot (col 5).

        Row 4:
          SECOND set – per-speed boxplots by participant with TWO boxes
          per participant (self blue, cross red) in cols 0–4,
          plus mean |error| vs speed (self vs cross) bar plot in col 5.
        """

        # -----------------------------
        # 1) Build tall DataFrame of abs errors (incl. SECOND-cross)
        # -----------------------------
        records = []

        speed_xtick_label_map = {
            "slow": "30",
            "medm": "60",
            "fast": "120",
            "vfas": "240",
            "afap": "AFAP",
        }

        for (pname, speed), res in results_adc.items():
            if res is None:
                continue

            # FIRST self
            arr_first = res.get("abs_err_first")
            if arr_first is not None:
                vals = np.asarray(arr_first).ravel()
                vals = vals[~np.isnan(vals)]
                for v in vals:
                    records.append(
                        dict(
                            participant=pname,
                            speed=speed,
                            set="FIRST",
                            calib="self",
                            abs_err_deg=float(v),
                        )
                    )

            # SECOND self
            arr_second = res.get("abs_err_second")
            if arr_second is not None:
                vals = np.asarray(arr_second).ravel()
                vals = vals[~np.isnan(vals)]
                for v in vals:
                    records.append(
                        dict(
                            participant=pname,
                            speed=speed,
                            set="SECOND",
                            calib="self",
                            abs_err_deg=float(v),
                        )
                    )

            # SECOND cross
            arr_second_cross = res.get("abs_err_second_cross")
            if arr_second_cross is not None:
                vals = np.asarray(arr_second_cross).ravel()
                vals = vals[~np.isnan(vals)]
                for v in vals:
                    records.append(
                        dict(
                            participant=pname,
                            speed=speed,
                            set="SECOND",
                            calib="cross",
                            abs_err_deg=float(v),
                        )
                    )

        if not records:
            print("[plot_adc_abs_error_summary_grid] No error data found.")
            return

        df = pd.DataFrame.from_records(records)

        # Pretty speed labels
        if speed_titles is not None:
            df["speed_label"] = df["speed"].map(lambda s: speed_titles.get(s, s))
        else:
            df["speed_label"] = df["speed"]

        # drop participant–set combos from summary rows if requested
        if exclude_participant_set_for_summary:
            mask_excl = pd.Series(False, index=df.index)
            for pname, set_label in exclude_participant_set_for_summary:
                mask_excl |= (
                        (df["participant"] == pname)
                        & (df["set"] == set_label)
                )
            df = df.loc[~mask_excl].copy()

        if speed_order is None:
            speed_order = sorted(df["speed"].unique())

        # -----------------------------
        # 2) Seaborn theme & figure
        # -----------------------------
        sns.set_theme(style="ticks", context="talk")
        nrows, ncols = 4, 6
        fig = plt.figure(figsize=figsize)

        # Outer 2×1 grid: top block (rows 0–1), bottom block (rows 2–3)
        outer_gs = fig.add_gridspec(
            2, 1,
            height_ratios=[1, 1],
            hspace=0.4,  # vertical space between top and bottom blocks
        )

        # Top: rows 0–1 with their own width_ratios
        top_gs = outer_gs[0].subgridspec(
            2, 6,
            width_ratios=[0.8, 0.8, 0.8, 0.2, 1.5, 0.4],
            wspace=boxplot_wspace,
            hspace=1.6,
        )

        # Bottom: rows 2–3 with different width_ratios
        bottom_gs = outer_gs[1].subgridspec(
            2, 6,
            width_ratios=[0.8, 0.8, 0.8, 0.8, 0.8, 1.4],
            wspace=boxplot_wspace,
            hspace=1,
        )

        # Build a 4×6 axes array to keep the rest of your code unchanged
        axes = np.empty((4, 6), dtype=object)

        # Fill rows 0–1 from top_gs
        for r in range(2):
            for c in range(6):
                axes[r, c] = fig.add_subplot(top_gs[r, c])

        # Fill rows 2–3 from bottom_gs
        for r in range(2):
            for c in range(6):
                axes[r + 2, c] = fig.add_subplot(bottom_gs[r, c])

        def _style_ax(ax):
            sns.despine(ax=ax, top=True, right=True)
            ax.grid(False)
            ax.tick_params(labelsize=tick_fontsize)

        def _hide_y_axis(ax):
            # ax.yaxis.set_visible(False)
            # ax.tick_params(axis="y", left=False, labelleft=False)
            # for side in ("left", "right"):
            #     if side in ax.spines:
            #         ax.spines[side].set_visible(False)
            ax.tick_params(axis="y", labelleft=False)  # hide text
            ax.tick_params(axis="y", left=False)        # keep tick marks

        # -----------------------------
        # Helpers for example rows (use explicit DataFrames)
        # -----------------------------
        def _get_example_from_df(df_ex: pd.DataFrame):
            """
            Extract DLC + ADC angles and build a 0–10 s time axis.
            """
            if df_ex is None or df_ex.empty:
                return None

            if dlc_angle_col not in df_ex.columns or adc_angle_col not in df_ex.columns:
                return None

            dlc_angle = df_ex[dlc_angle_col].to_numpy(dtype=float)
            adc_angle = df_ex[adc_angle_col].to_numpy(dtype=float)

            n = len(df_ex)

            # Build a raw time axis in seconds
            if "cam_timestamp" in df_ex.columns:
                # assume nanoseconds
                t_raw = df_ex["cam_timestamp"].to_numpy(dtype=float)
                t_sec = (t_raw - t_raw[0]) / 1e9
            elif "_t_cam_td" in df_ex.columns:
                t_td = df_ex["_t_cam_td"]
                t_sec = t_td.dt.total_seconds().to_numpy()
                t_sec = t_sec - t_sec[0]
            else:
                # no timestamp info -> use sample index as a proxy
                t_sec = np.arange(n, dtype=float)

            # Normalize to 0–10 s ALWAYS
            if n > 1 and np.nanmax(t_sec) > np.nanmin(t_sec):
                t_plot = (t_sec - np.nanmin(t_sec)) / (np.nanmax(t_sec) - np.nanmin(t_sec)) * 10.0
            else:
                t_plot = np.zeros_like(t_sec)

            abs_err = np.abs(dlc_angle - adc_angle)
            return t_plot, dlc_angle, adc_angle, abs_err

        def _plot_example_timeseries(
                ax,
                df_ex,
                participant,
                speed,
                trial_idx,
                title_prefix,
                show_legend: bool = True,
                legend_inside: bool = False,
        ):
            res_ex = _get_example_from_df(df_ex)
            if res_ex is None:
                ax.text(
                    0.5, 0.5,
                    f"No example data\n(participant={participant}, speed={speed}, trial={trial_idx})",
                    ha="center", va="center", fontsize=10,
                )
                _style_ax(ax)
                return

            t_plot, dlc_angle, adc_angle, _ = res_ex

            sns.lineplot(x=t_plot, y=dlc_angle, ax=ax, label="Camera\n(CAM) Angle", linewidth=2, color=COLOR_FIRST, alpha=0.65)
            sns.lineplot(x=t_plot, y=adc_angle, ax=ax, label="Strain Sensor\n(SS) Angle", linewidth=2, color=COLOR_SECOND, alpha=0.65)

            # Always show 0–10 s with ticks every 2 s
            ax.set_xlabel("Time (s)", fontsize=label_fontsize, fontweight=label_weight)
            ax.set_ylabel("Angle (°)", fontsize=label_fontsize, fontweight=label_weight)
            ax.set_xlim(0.0, 10.0)
            ax.set_xticks(np.arange(0, 10.01, 2.0))

            bpm_label = speed_xtick_label_map.get(speed, speed)

            bpm_label = speed_xtick_label_map.get(speed, speed)

            ax.set_title(
                f"{title_prefix}: {participant}\nTrial {trial_idx + 1} - {bpm_label} BPM",
                fontsize=title_fontsize,
                fontweight=title_weight,
            )
            _style_ax(ax)

            if show_legend:
                if legend_inside:
                    # inside top-left
                    ax.legend(
                        fontsize=8,
                        frameon=False,
                        loc="upper center",
                        ncol=2,           # horizontal
                        #columnspacing=0.4,
                        #labelspacing=0.1,
                        handlelength=0.9, # shorter line segments
                        handletextpad=0.4,
                        borderaxespad=0.2,
                        bbox_to_anchor=(0.5, 1.125),
                    )
                else:
                    # outside to the right (old version)
                    ax.legend(
                        fontsize=tick_fontsize,
                        frameon=False,
                        loc="upper left",
                        bbox_to_anchor=(1.02, 1.0),
                        borderaxespad=0.0,
                    )
            else: 
                ax.get_legend().remove()

        def _plot_example_box(
                ax,
                df_ex,
                participant,
                speed,
                trial_idx,
                title_prefix,
        ):
            res_ex = _get_example_from_df(df_ex)
            if res_ex is None:
                ax.text(
                    0.5, 0.5,
                    f"No example data\n(participant={participant}, speed={speed}, trial={trial_idx})",
                    ha="center", va="center", fontsize=10,
                )
                _style_ax(ax)
                return

            _, _, _, abs_err = res_ex
            vals = abs_err[np.isfinite(abs_err)]
            if vals.size == 0:
                ax.text(0.5, 0.5, "No valid samples",
                        ha="center", va="center", fontsize=10)
                _style_ax(ax)
                return

            df_box = pd.DataFrame({"abs_err_deg": vals})
            sns.boxplot(
                data=df_box,
                y="abs_err_deg",
                ax=ax,
                showfliers=False,
                width=0.2,      # skinnier box
                color=COLOR_FIRST,
                boxprops=dict(alpha=0.65),
                whiskerprops=dict(alpha=0.65),
                capprops=dict(alpha=0.65),
                medianprops=dict(alpha=0.65),
            )
            ax.set_xlabel("", fontsize=label_fontsize, fontweight=label_weight)
            ax.set_ylabel("|Error| (°)", fontsize=label_fontsize, fontweight=label_weight)
            ax.set_title(
                f"{title_prefix}: \n|CAM-SS|",
                fontsize=title_fontsize,
                fontweight=title_weight,
            )

            ylim_ex = abs_err_ylim_examples or abs_err_ylim
            if ylim_ex is not None:
                ax.set_ylim(ylim_ex)

            _style_ax(ax)

        # -----------------------------
        # 3) Column positions (shift examples right)
        # -----------------------------
        # layout edit
        example_col_start = ncols - 4  # this will be 2 when ncols = 6

        calib_col = example_col_start - 1       # 3
        ts_col    = example_col_start + 2     # 4
        box_col   = example_col_start + 3     # 5

        # -----------------------------
        # 4) Row 1 – calibration + example 1
        # -----------------------------
        for j in range(ncols):
            # keep the calibration axis, TS axis, and box axis
            if j not in (calib_col, ts_col, box_col):
                axes[0, j].axis("off")

        # --- calibration panel: FIRST application set (blue) from calib_df ---
        ax_cal = axes[0, calib_col]
        has_calib = calib_df is not None and not getattr(calib_df, "empty", False)

        if has_calib:
            sub = calib_df.copy()

            # keep only requested set if 'set' column exists
            if "set" in sub.columns and calib_set_label is not None:
                sub = sub[sub["set"] == calib_set_label]

            # keep only angles up to 67.5 deg
            if "angle_snap_deg" in sub.columns:
                sub = sub[sub["angle_snap_deg"] <= 67.5]

            if (
                    not sub.empty
                    and "angle_snap_deg" in sub.columns
                    and "adc_mean" in sub.columns
            ):
                ax_cal.scatter(
                    sub["angle_snap_deg"].to_numpy(),
                    sub["adc_mean"].to_numpy(),
                    s=30,
                    color=COLOR_FIRST,  # FIRST / blue
                    alpha = 0.65
                )
                ax_cal.set_xlabel(
                    "Calibration Angle (°)",
                    fontsize=label_fontsize,
                    fontweight=label_weight,
                )
                ax_cal.set_ylabel(
                    "Mean ADC Value",
                    fontsize=label_fontsize,
                    fontweight=label_weight,
                )
                ax_cal.set_title(
                    "1st App Calibration (P3)",
                    fontsize=title_fontsize,
                    fontweight=title_weight,
                )
                _style_ax(ax_cal)

                # Tweak scientific-notation offset text on y-axis
                offset = ax_cal.yaxis.get_offset_text()
                offset.set_fontsize(tick_fontsize)   # match y-tick label size
                offset.set_x(-0.25)                  # nudge left (tune as needed)
            else:
                ax_cal.axis("off")
        else:
            ax_cal.axis("off")

        # --- Example 1 time-series in shifted column ---
        _plot_example_timeseries(
            axes[0, ts_col], # layout edit
            df_ex=example1_df,
            participant=example1_participant,
            speed=example1_speed,
            trial_idx=example1_trial_idx,
            title_prefix="Example",
            show_legend=True,
            legend_inside=True, 
        )
        axes[0, ts_col].set_yticks([0, 30, 60, 90])

        # --- Example 1 error boxplot ---
        _plot_example_box(
            axes[0, box_col], # layout edit
            df_ex=example1_df,
            participant=example1_participant,
            speed=example1_speed,
            trial_idx=example1_trial_idx,
            title_prefix="30 BPM",
        )

        # layout edit
        # -----------------------------
        # 5) Row 2 – example 2 (uses example2_df)
        # -----------------------------
        for j in range(ncols):
            if j not in (ts_col, box_col):
                axes[1, j].axis("off")

        _plot_example_timeseries(
            axes[1, ts_col], # layout edit
            df_ex=example2_df,
            participant=example2_participant,
            speed=example2_speed,
            trial_idx=example2_trial_idx,
            title_prefix="Example",
            show_legend=False,
        )
        axes[1, ts_col].set_yticks([0, 25, 50])

        _plot_example_box(
            axes[1, box_col], # layout edit
            df_ex=example2_df,
            participant=example2_participant,
            speed=example2_speed,
            trial_idx=example2_trial_idx,
            title_prefix="240 BPM",
        )

        # -----------------------------
        # 6) Row 3 – FIRST summary
        # -----------------------------
        df_first = df[df["set"] == "FIRST"]
        ylim_sum = abs_err_ylim_summary or abs_err_ylim

        for i, spd in enumerate(speed_order[:5]):
            ax = axes[2, i] # layout edit
            sub = df_first[df_first["speed"] == spd]
            if sub.empty:
                ax.axis("off")
                continue

            sns.boxplot(
                data=sub,
                x="participant",
                y="abs_err_deg",
                ax=ax,
                showfliers=False,
                width=0.35,         # skinnier
                color=COLOR_FIRST,  # blue
                boxprops=dict(alpha=0.65),
                whiskerprops=dict(alpha=0.65),
                capprops=dict(alpha=0.65),
                medianprops=dict(alpha=0.65),
            )

            label = speed_titles.get(spd, spd) if speed_titles else spd
            ax.set_title(
                f"{label}",
                fontsize=title_fontsize,
                fontweight=title_weight,
            )
            #ax.set_xlabel("Participant", fontsize=label_fontsize, fontweight=label_weight)
            ax.set_xlabel("", fontsize=label_fontsize, fontweight=label_weight)

            if ylim_sum is not None:
                ax.set_ylim(ylim_sum)

            ax.set_yticks([0, 10, 20, 30])

            from matplotlib.patches import Patch

            if i == 0:
                ax.set_ylabel(
                    "|Error| (°)",
                    fontsize=label_fontsize,
                    fontweight=label_weight,
                )
                _style_ax(ax)

            else:
                ax.set_ylabel("")
                _style_ax(ax)
                _hide_y_axis(ax)

        ax_bar_first = axes[2, ncols - 1] # layout edit
        means_first = []
        #labels_first = []
        for spd in speed_order:
            sub = df_first[df_first["speed"] == spd]["abs_err_deg"]
            means_first.append(sub.mean() if not sub.empty else np.nan)
            #labels_first.append(speed_titles.get(spd, spd) if speed_titles else spd)

        x = np.arange(len(speed_order))

        sns.barplot(
            x=x,
            y=means_first,
            ax=ax_bar_first,
            color=COLOR_FIRST,
            alpha = 0.65,
            errorbar=None,
        )

        # Map speeds -> desired tick labels
        xtick_labels_first = [speed_xtick_label_map[spd] for spd in speed_order]
        ax_bar_first.set_xticks(x)
        ax_bar_first.set_xticklabels(xtick_labels_first, rotation=0)

        ax_bar_first.set_title(
            "Mean |Error|",
            fontsize=title_fontsize,
            fontweight=title_weight,
        )
        ax_bar_first.set_xlabel("", fontsize=label_fontsize, fontweight=label_weight)
        # ax_bar_first.set_ylabel(
        #     "Mean |Error| (°)",
        #     fontsize=label_fontsize,
        #     fontweight=label_weight,
        # )
        _style_ax(ax_bar_first)                 # applies tick_params(labelsize=tick_fontsize)
        # (optional, redundant but explicit)
        ax_bar_first.tick_params(axis="both", labelsize=tick_fontsize)
        ax_bar_first.set_ylim(0, 6)
        ax_bar_first.set_yticks([0, 2, 4, 6, 8])

        # -----------------------------
        # 7) Row 4 – SECOND summary (self vs cross)
        # -----------------------------
        df_second = df[df["set"] == "SECOND"]

        # spacing knobs
        box_width = 0.25
        paired_offset = 0.30
        intra_group_spacing = 0.55
        gap_between_groups = 0.2

        for i, spd in enumerate(speed_order[:5]):
            ax = axes[3, i] # layout edit
            sub = df_second[df_second["speed"] == spd]
            if sub.empty:
                ax.axis("off")
                continue

            # participants in this speed, sorted
            participants = sorted(sub["participant"].unique())

            data_list = []   # one entry per box
            positions = []   # x-position for each box
            colors = []      # COLOR_FIRST / COLOR_SECOND for each box
            centers = []     # center x of each participant group

            for j, p in enumerate(participants):
                center = j * (intra_group_spacing + gap_between_groups)
                centers.append(center)

                for calib, sign, color in [
                    ("self", -1, COLOR_FIRST),
                    ("cross", +1, COLOR_SECOND),
                ]:
                    vals = sub[
                        (sub["participant"] == p) & (sub["calib"] == calib)
                    ]["abs_err_deg"].to_numpy()

                    if vals.size == 0:
                        continue  # skip if no data for this calib/participant

                    pos = center + sign * (paired_offset / 2.0)
                    positions.append(pos)
                    data_list.append(vals)
                    colors.append(color)

            # draw the boxes at explicit positions
            bp = ax.boxplot(
                data_list,
                positions=positions,
                widths=box_width,
                showfliers=False,
                patch_artist=True,
            )

            # color + alpha
            for box, color in zip(bp["boxes"], colors):
                box.set_facecolor(color)
                box.set_edgecolor(color)
                box.set_alpha(0.65)

            for item in bp["whiskers"] + bp["caps"]:
                item.set_alpha(0.65)

            for med in bp["medians"]:
                med.set_color("black")
                #med.set_linewidth(1)
                med.set_alpha(0.65)

            # x-ticks at group centers labeled by participant
            ax.set_xticks(centers)
            ax.set_xticklabels(participants, fontsize=tick_fontsize)

            # tighten x-limits around first/last groups
            if centers:
                padding = box_width + gap_between_groups / 2.0
                ax.set_xlim(centers[0] - padding, centers[-1] + padding)

 
            # sns.boxplot(
            #     data=sub,
            #     x="participant",
            #     y="abs_err_deg",
            #     hue="calib",
            #     ax=ax,
            #     showfliers=False,
            #     width=0.35,  # skinnier
            #     linewidth=1.4,
            #     palette={"self": COLOR_FIRST, "cross": COLOR_SECOND},
            #     boxprops=dict(alpha=0.65), 
            #     whiskerprops=dict(alpha=0.65),
            #     capprops=dict(alpha=0.65),
            #     medianprops=dict(alpha=0.65),
            # )

            # # reduce outer whitespace between the first/last participants and axis edges
            # ax.margins(x=0.05)

            # if ax.get_legend() is not None:
            #     ax.get_legend().remove()

            label = speed_titles.get(spd, spd) if speed_titles else spd
            ax.set_title(
                f"{label}",
                fontsize=title_fontsize,
                fontweight=title_weight,
            )
            #ax.set_xlabel("Participant", fontsize=label_fontsize, fontweight=label_weight)
            if i == 2:
                ax.set_xlabel("Participant", fontsize=label_fontsize, fontweight=label_weight)
            else:
                ax.set_xlabel("", fontsize=label_fontsize, fontweight=label_weight)

            if ylim_sum is not None:
                ax.set_ylim(ylim_sum)

            ax.set_yticks([0, 10, 20, 30])


            if i == 0:
                ax.set_ylabel(
                    "|Error| (°)",
                    fontsize=label_fontsize,
                    fontweight=label_weight,
                )
                _style_ax(ax)
                # Add legend INSIDE slow-speed panel (row 4, col 0)
                handles = [
                    Patch(facecolor=COLOR_FIRST, edgecolor=COLOR_FIRST, alpha=0.65, label="self"),
                    Patch(facecolor=COLOR_SECOND, edgecolor=COLOR_SECOND, alpha=0.65, label="xtrain"),
                ]
                ax.legend(
                    handles=handles,
                    loc="upper left",   # or "upper left" if you prefer
                    frameon=False,
                    fontsize=tick_fontsize,
                    handlelength=0.8,   # shorter color blocks
                    handleheight=0.6,   # less tall
                    borderpad=0.2,      # tighter box padding
                    labelspacing=0.3,   # less space between entries
                    handletextpad=0.4,  # space between block and text
                )
            else:
                ax.set_ylabel("")
                _style_ax(ax)
                _hide_y_axis(ax)

        ax_bar_second = axes[3, ncols - 1] # layout edit
        means_second_self = []
        means_second_cross = []
        labels_second = []

        for spd in speed_order:
            sub = df_second[df_second["speed"] == spd]
            sub_self = sub[sub["calib"] == "self"]["abs_err_deg"]
            sub_cross = sub[sub["calib"] == "cross"]["abs_err_deg"]

            means_second_self.append(sub_self.mean() if not sub_self.empty else np.nan)
            means_second_cross.append(sub_cross.mean() if not sub_cross.empty else np.nan)
            labels_second.append(speed_titles.get(spd, spd) if speed_titles else spd)

        x = np.arange(len(speed_order))
        width = 0.35

        ax_bar_second.bar(
            x - width / 2, means_second_self,
            width=width, color=COLOR_FIRST, alpha = 0.65, label="self",
        )
        ax_bar_second.bar(
            x + width / 2, means_second_cross,
            width=width, color=COLOR_SECOND, alpha = 0.65, label="xtrain",
        )

        xtick_labels_second = [speed_xtick_label_map[spd] for spd in speed_order]
        ax_bar_second.set_xticks(x)
        ax_bar_second.set_xticklabels(xtick_labels_second, rotation=0)
        ax_bar_second.set_title(
            "Mean |Error|",
            fontsize=title_fontsize,
            fontweight=title_weight,
        )
        ax_bar_second.set_xlabel("Speed (BPM)", fontsize=label_fontsize, fontweight=label_weight)
        # ax_bar_second.set_ylabel(
        #     "Mean |Error| (°)",
        #     fontsize=label_fontsize,
        #     fontweight=label_weight,
        # )
        ax_bar_second.tick_params(axis="x", labelsize=tick_fontsize)
        _style_ax(ax_bar_second)
        # ax_bar_second.legend(fontsize=tick_fontsize, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.05))

        # -----------------------------
        # 8) Layout + base spacing
        # -----------------------------
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.8) #wspace=boxplot_wspace, 

        # -----------------------------
        # 8b) Optional: shift calibration plot to the right
        # -----------------------------
        if calib_x_shift != 0.0:
            ax_cal = axes[0, calib_col]
            pos = ax_cal.get_position()
            ax_cal.set_position([
                pos.x0 + calib_x_shift,
                pos.y0,
                pos.width,
                pos.height,
            ])

        # layout edit
        # -----------------------------
        # 9) Extra gap scaling for first two rows (TS vs box)
        # -----------------------------
        if example_col_gap_scale != 1.0:
            ts_ax0 = axes[0, ts_col]
            box_ax0 = axes[0, box_col]

            ts_pos = ts_ax0.get_position()
            box_pos = box_ax0.get_position()

            ts_right = ts_pos.x0 + ts_pos.width
            current_gap = box_pos.x0 - ts_right
            new_gap = current_gap * example_col_gap_scale

            delta = (ts_right + new_gap) - box_pos.x0

            for row_idx in (0, 1):
                ax = axes[row_idx, box_col]
                pos = ax.get_position()
                ax.set_position([
                    pos.x0 + delta,
                    pos.y0,
                    pos.width,
                    pos.height,
                ])

        # -----------------------------
        # 10) Force bar plots to be bar_gap_from_afap away from AFAP boxplots
        # -----------------------------
        for row_idx in (2, 3): # layout edit
            afap_ax = axes[row_idx, ncols - 2]
            bar_ax = axes[row_idx, ncols - 1]

            afap_pos = afap_ax.get_position()
            bar_pos = bar_ax.get_position()

            afap_right = afap_pos.x0 + afap_pos.width
            delta = (afap_right + bar_gap_from_afap) - bar_pos.x0

            new_x0 = bar_pos.x0 + delta
            bar_ax.set_position([
                new_x0,
                bar_pos.y0,
                bar_pos.width,
                bar_pos.height,
            ])

        return fig, axes