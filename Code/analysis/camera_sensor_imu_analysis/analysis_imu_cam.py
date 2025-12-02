from pathlib import Path
from typing import List, Dict, Optional, Tuple, Sequence, Any
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from analysis_adc_cam import ADC_CAM  # or from .analysis_adc_cam import ADC_CAM
from analysis import DLC3DBendAngles      # IMU angle math lives here (old pipeline)
import seaborn as sns


class IMU_cam(ADC_CAM):
    """
    Helper for loading IMU CSVs, DLC CSVs, and FLIR MATs trial-by-trial using the
    same folder discovery logic as ADC_CAM, plus IMU↔DLC angle pipelines.

    Typical usage
    -------------
    cam = IMU_cam(
        root_dir=root_dir,
        path_to_repo=path_to_repository,
        folder_suffix_first="B1_slow",
        folder_suffix_second="B2_slow",
    )

    first_trials  = cam.load_first()
    second_trials = cam.load_second()

    imu_first_dfs  = cam.extract_imu_dfs_first()
    imu_second_dfs = cam.extract_imu_dfs_second()
    """

    # ------------------------------------------------------------------
    # IMU CSV extraction
    # ------------------------------------------------------------------
    def extract_imu_dfs_by_trial(
        self,
        trials: List[Dict[str, Optional[Path]]],
        *,
        add_metadata: bool = True,
    ) -> List[Optional[pd.DataFrame]]:
        """
        Given a list of trial dicts (from load_first/load_second),
        load the IMU CSV for each trial into a pandas DataFrame.

        IMPORTANT: we preserve list length by appending None when a trial has
        no IMU CSV, so indices stay aligned with DLC trials.
        """
        dfs: List[Optional[pd.DataFrame]] = []

        for idx, trial in enumerate(trials):
            imu_path = trial.get("imu_csv", None)
            if imu_path is None:
                dfs.append(None)
                continue

            # Robust CSV read: try UTF-8, fall back to latin-1
            try:
                df = pd.read_csv(imu_path)
            except UnicodeDecodeError:
                df = pd.read_csv(imu_path, encoding="latin-1")

            if add_metadata:
                df = df.copy()
                df["trial_index"] = idx
                df["source_path"] = str(imu_path)

            dfs.append(df)

        return dfs

    def extract_imu_dfs_first(
        self,
        *,
        add_metadata: bool = True,
    ) -> List[Optional[pd.DataFrame]]:
        """Load first-application trials and return IMU DataFrames (or None) per trial."""
        trials_first = self.load_first()
        return self.extract_imu_dfs_by_trial(trials_first, add_metadata=add_metadata)

    def extract_imu_dfs_second(
        self,
        *,
        add_metadata: bool = True,
    ) -> List[Optional[pd.DataFrame]]:
        """Load reapplication trials and return IMU DataFrames (or None) per trial."""
        trials_second = self.load_second()
        return self.extract_imu_dfs_by_trial(trials_second, add_metadata=add_metadata)

    # ------------------------------------------------------------------
    # FLIR MAT extraction (convenience wrappers)
    # ------------------------------------------------------------------
    def extract_mat_dfs_first(
        self,
        *,
        mat_name: str = "flir_data.mat",
        prefix: str = "ts",
        add_labels: bool = True,
        trial_base: int = 1,
        set_label: str = "first_cam",
        include_path: bool = True,
    ):
        """Load first-application trials and return FLIR/other MAT DataFrames per trial."""
        trials_first = self.load_first()
        return self.extract_mat_dfs_by_trial(
            trials_first,
            mat_name=mat_name,
            prefix=prefix,
            add_labels=add_labels,
            trial_base=trial_base,
            set_label=set_label,
            include_path=include_path,
        )

    def extract_mat_dfs_second(
        self,
        *,
        mat_name: str = "flir_data.mat",
        prefix: str = "ts",
        add_labels: bool = True,
        trial_base: int = 1,
        set_label: str = "second_cam",
        include_path: bool = True,
    ):
        """Load second-application trials and return FLIR/other MAT DataFrames per trial."""
        trials_second = self.load_second()
        return self.extract_mat_dfs_by_trial(
            trials_second,
            mat_name=mat_name,
            prefix=prefix,
            add_labels=add_labels,
            trial_base=trial_base,
            set_label=set_label,
            include_path=include_path,
        )

    # ------------------------------------------------------------------
    # DLC 3D extract + angle computation + filtering
    # ------------------------------------------------------------------
    def coerce_dlc3d_multiindex_list(
        self,
        dlc3d_trials: List[pd.DataFrame],
    ) -> List[pd.DataFrame]:
        """Apply ADC_CAM._coerce_dlc3d_multiindex to each trial."""
        return [self._coerce_dlc3d_multiindex(df) for df in dlc3d_trials]

    def compute_dlc3d_angles_first(
        self,
        dlc3d_trials: List[pd.DataFrame],
        *,
        signed_in_plane: bool = True,
        add_plane_ok: bool = True,
    ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        """
        Compute MCP + wrist bend angles for the 'first_cam' set using the
        same logic as ADC_CAM.compute_dlc3d_angles_by_trial.
        """
        dlc3d_trials = self.coerce_dlc3d_multiindex_list(dlc3d_trials)

        return self.compute_dlc3d_angles_by_trial(
            dlc3d_trials=dlc3d_trials,
            set_label="first_cam",
            signed_in_plane=signed_in_plane,
            add_plane_ok=add_plane_ok,
        )

    def compute_dlc3d_angles_second(
        self,
        dlc3d_trials: List[pd.DataFrame],
        *,
        signed_in_plane: bool = True,
        add_plane_ok: bool = True,
    ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        """Compute MCP + wrist bend angles for the 'second_cam' set."""
        dlc3d_trials = self.coerce_dlc3d_multiindex_list(dlc3d_trials)

        return self.compute_dlc3d_angles_by_trial(
            dlc3d_trials=dlc3d_trials,
            set_label="second_cam",
            signed_in_plane=signed_in_plane,
            add_plane_ok=add_plane_ok,
        )

    def extract_dlc3d_dfs_first(
        self,
        *,
        add_labels: bool = True,
        trial_base: int = 1,
        set_label: str = "first_cam",
        include_path: bool = True,
        file_patterns: Tuple[str, ...] = (
            "*DLC*.csv", "*DLC3D*.csv", "*dlc3d*.csv", "*3d*.csv", "*3D*.csv"
        ),
    ) -> List[pd.DataFrame]:
        """Load first-application trials and return DLC 3D DataFrames per trial."""
        trials_first = self.load_first()
        return self.extract_dlc3d_dfs_by_trial(
            trials_first,
            file_patterns=file_patterns,
            add_labels=add_labels,
            trial_base=trial_base,
            set_label=set_label,
            include_path=include_path,
        )

    def extract_dlc3d_dfs_second(
        self,
        *,
        add_labels: bool = True,
        trial_base: int = 1,
        set_label: str = "second_cam",
        include_path: bool = True,
        file_patterns: Tuple[str, ...] = (
            "*DLC*.csv", "*DLC3D*.csv", "*dlc3d*.csv", "*3d*.csv", "*3D*.csv"
        ),
    ) -> List[pd.DataFrame]:
        """Load second-application trials and return DLC 3D DataFrames per trial."""
        trials_second = self.load_second()
        return self.extract_dlc3d_dfs_by_trial(
            trials_second,
            file_patterns=file_patterns,
            add_labels=add_labels,
            trial_base=trial_base,
            set_label=set_label,
            include_path=include_path,
        )

    # ------------------------------------------------------------------
    # Attach FLIR timestamps to DLC angle trials
    # ------------------------------------------------------------------
    def attach_cam_timestamps_first(
        self,
        angle_trials: List[pd.DataFrame],
        *,
        cam_trials: Optional[List[pd.DataFrame]] = None,
        time_col_name: str = "timestamp",
        new_col_name: str = "cam_timestamp",
    ) -> List[pd.DataFrame]:
        """Attach FLIR camera timestamps to DLC angle trials for the first set."""
        if cam_trials is None:
            trials_first = self.load_first()
            cam_trials = self.extract_mat_dfs_by_trial(
                trials_first,
                mat_name="flir_data.mat",
                prefix="ts",
                add_labels=True,
                trial_base=1,
                set_label="first_cam",
                include_path=True,
            )

        return self.attach_cam_timestamps_to_angles(
            cam_trials=cam_trials,
            angle_trials=angle_trials,
            time_col_name=time_col_name,
            new_col_name=new_col_name,
        )

    def attach_cam_timestamps_second(
        self,
        angle_trials: List[pd.DataFrame],
        *,
        cam_trials: Optional[List[pd.DataFrame]] = None,
        time_col_name: str = "timestamp",
        new_col_name: str = "cam_timestamp",
    ) -> List[pd.DataFrame]:
        """Attach FLIR camera timestamps to DLC angle trials for the second set."""
        if cam_trials is None:
            trials_second = self.load_second()
            cam_trials = self.extract_mat_dfs_by_trial(
                trials_second,
                mat_name="flir_data.mat",
                prefix="ts",
                add_labels=True,
                trial_base=1,
                set_label="second_cam",
                include_path=True,
            )

        return self.attach_cam_timestamps_to_angles(
            cam_trials=cam_trials,
            angle_trials=angle_trials,
            time_col_name=time_col_name,
            new_col_name=new_col_name,
        )

    # ------------------------------------------------------------------
    # DLC angle filtering by likelihood
    # ------------------------------------------------------------------
    def filter_dlc3d_angles_first(
        self,
        angle_trials: List[pd.DataFrame],
        *,
        bodyparts: Sequence[str] = ("MCP", "PIP", "hand"),
        min_likelihood: float = 0.6,
        inplace: bool = False,
    ) -> List[pd.DataFrame]:
        """Filter DLC angle trials for the first set using DLC likelihood."""
        return self.filter_angle_trials_by_likelihood(
            angle_trials=angle_trials,
            bodyparts=bodyparts,
            min_likelihood=min_likelihood,
            inplace=inplace,
        )

    def filter_dlc3d_angles_second(
        self,
        angle_trials: List[pd.DataFrame],
        *,
        bodyparts: Sequence[str] = ("MCP", "PIP", "hand"),
        min_likelihood: float = 0.6,
        inplace: bool = False,
    ) -> List[pd.DataFrame]:
        """Filter DLC angle trials for the second set using DLC likelihood."""
        return self.filter_angle_trials_by_likelihood(
            angle_trials=angle_trials,
            bodyparts=bodyparts,
            min_likelihood=min_likelihood,
            inplace=inplace,
        )

    # ------------------------------------------------------------------
    # Low-level IMU helpers: axis, quaternion parsing, rotmat, time
    # ------------------------------------------------------------------
    def _imu_axis_vector(self, axis: str) -> np.ndarray:
        """Map 'x'/'y'/'z' to a unit 3D axis vector in the IMU body frame."""
        axis = axis.lower()
        if axis == "x":
            return np.array([1.0, 0.0, 0.0], dtype=float)
        if axis == "y":
            return np.array([0.0, 1.0, 0.0], dtype=float)
        if axis == "z":
            return np.array([0.0, 0.0, 1.0], dtype=float)
        raise ValueError(f"Unknown axis '{axis}', expected 'x', 'y', or 'z'.")

    def _imu_quat_from_cols(
        self,
        df: pd.DataFrame,
        base_name: str,
        quat_order: str = "wxyz",
    ) -> np.ndarray:
        """
        Extract quaternion samples for a given base_name in one of two formats:

        A) Separate columns: base_name+'_w', base_name+'_x', base_name+'_y', base_name+'_z'
        B) Single column containing a 4-tuple or tuple-like string: base_name

        Handles 'None'/None/empty strings by substituting identity [1,0,0,0].
        Returns (N,4) in canonical [w,x,y,z] order.
        """
        quat_order = quat_order.lower()
        if set(quat_order) != {"w", "x", "y", "z"} or len(quat_order) != 4:
            raise ValueError(f"quat_order must be a permutation of 'wxyz', got {quat_order!r}")

        comp_cols = {
            "w": f"{base_name}_w",
            "x": f"{base_name}_x",
            "y": f"{base_name}_y",
            "z": f"{base_name}_z",
        }

        # ---------- Case A: separate component columns ----------
        if all(col in df.columns for col in comp_cols.values()):
            cols_in_file_order = [comp_cols[c] for c in quat_order]
            Q = df[cols_in_file_order].to_numpy(dtype=float)

            idx_w = quat_order.index("w")
            idx_x = quat_order.index("x")
            idx_y = quat_order.index("y")
            idx_z = quat_order.index("z")

            Q_canon = np.stack(
                [Q[:, idx_w], Q[:, idx_x], Q[:, idx_y], Q[:, idx_z]],
                axis=1,
            )

        # ---------- Case B: single tuple/tuple-string column ----------
        elif base_name in df.columns:
            col = df[base_name]
            parsed: List[List[float]] = []

            for v in col:
                if v is None or (isinstance(v, float) and np.isnan(v)):
                    parsed.append([np.nan, np.nan, np.nan, np.nan])
                    continue

                if isinstance(v, (tuple, list, np.ndarray)):
                    vals = list(v)
                else:
                    s = str(v).strip()
                    if s.lower() == "none" or s == "":
                        parsed.append([np.nan, np.nan, np.nan, np.nan])
                        continue
                    if s.startswith(("(", "[")):
                        s = s[1:]
                    if s.endswith((")", "]")):
                        s = s[:-1]
                    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
                    vals = []
                    for p in parts:
                        pc = p.strip()
                        if pc.lower() == "none" or pc == "":
                            vals.append(np.nan)
                        else:
                            try:
                                vals.append(float(pc))
                            except ValueError:
                                vals.append(np.nan)

                if len(vals) < 4:
                    vals = vals + [np.nan] * (4 - len(vals))
                elif len(vals) > 4:
                    vals = vals[:4]

                parsed.append(vals)

            Q_raw = np.asarray(parsed, dtype=float)

            mask_all_nan = np.isnan(Q_raw).all(axis=1)
            Q_raw[mask_all_nan, :] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
            Q_raw[np.isnan(Q_raw)] = 0.0

            idx_w = quat_order.index("w")
            idx_x = quat_order.index("x")
            idx_y = quat_order.index("y")
            idx_z = quat_order.index("z")

            Q_canon = np.stack(
                [Q_raw[:, idx_w], Q_raw[:, idx_x], Q_raw[:, idx_y], Q_raw[:, idx_z]],
                axis=1,
            )

        else:
            raise KeyError(
                f"Could not find quaternion data for base '{base_name}'. "
                f"Looked for columns {list(comp_cols.values())} or '{base_name}'. "
                f"Available columns: {list(df.columns)}"
            )

        norms = np.linalg.norm(Q_canon, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Q_canon = Q_canon / norms

        # Enforce sign continuity over time to avoid q ↔ -q flips
        # (q and -q represent the same rotation, but sign flips can
        #  introduce artificial jumps in downstream angles.)
        for i in range(1, Q_canon.shape[0]):
            if np.dot(Q_canon[i], Q_canon[i - 1]) < 0.0:
                Q_canon[i] *= -1.0

        return Q_canon

    def _imu_quat_to_rotmat(self, Q: np.ndarray) -> np.ndarray:
        """
        Convert an array of quaternions Q of shape (N, 4) in [w, x, y, z] form
        into an array of rotation matrices of shape (N, 3, 3).
        """
        w = Q[:, 0]
        x = Q[:, 1]
        y = Q[:, 2]
        z = Q[:, 3]

        ww = w * w
        xx = x * x
        yy = y * y
        zz = z * z

        wx = w * x
        wy = w * y
        wz = w * z
        xy = x * y
        xz = x * z
        yz = y * z

        R = np.empty((Q.shape[0], 3, 3), dtype=float)

        R[:, 0, 0] = ww + xx - yy - zz
        R[:, 0, 1] = 2 * (xy - wz)
        R[:, 0, 2] = 2 * (xz + wy)

        R[:, 1, 0] = 2 * (xy + wz)
        R[:, 1, 1] = ww - xx + yy - zz
        R[:, 1, 2] = 2 * (yz - wx)

        R[:, 2, 0] = 2 * (xz - wy)
        R[:, 2, 1] = 2 * (yz + wx)
        R[:, 2, 2] = ww - xx - yy + zz

        return R

    def _imu_ensure_time_column(
        self,
        df: pd.DataFrame,
        time_col: str = "t_sec",
        trial_len_sec: float | None = None,
    ) -> None:
        """
        Ensure df has a time column.
        If trial_len_sec is provided, create linspace [0, trial_len_sec).
        Otherwise use integer index as time.
        """
        if time_col in df.columns:
            return

        n = len(df)
        if n == 0:
            df[time_col] = []
            return

        if trial_len_sec is not None:
            t = np.linspace(0.0, trial_len_sec, n, endpoint=False)
        else:
            t = np.arange(n, dtype=float)
        df[time_col] = t

    # ------------------------------------------------------------------
    # IMU relative-angle pipeline (pure quaternion math)
    # ------------------------------------------------------------------
    def compute_imu_relative_angles_by_trial(
        self,
        imu_trials: List[Optional[pd.DataFrame]],
        *,
        set_label: str,
        quat_cols: Tuple[str, str] = ("euler1", "euler2"),
        fixed_axis: str = "y",
        moving_axis: str = "y",
        quat_order: str = "wxyz",
        out_col: str = "imu_bend_deg",
        time_col: str = "t_sec",
        trial_len_sec: float | None = None,
    ) -> Tuple[List[Optional[pd.DataFrame]], pd.DataFrame]:
        """
        Compute bend angle between a chosen axis on IMU1 (fixed) and a chosen
        axis on IMU2 (moving), *as seen in the fixed IMU's frame*, using
        principal (unsigned) angle in [0, 180] degrees.
        """
        if len(quat_cols) != 2:
            raise ValueError(f"quat_cols must have length 2, got {quat_cols}")

        fixed_base, moving_base = quat_cols
        e_fix = self._imu_axis_vector(fixed_axis)
        e_move = self._imu_axis_vector(moving_axis)

        tall_list: List[pd.DataFrame] = []
        aug_trials: List[Optional[pd.DataFrame]] = []

        for trial_idx, df in enumerate(imu_trials):
            if df is None or df.empty:
                aug_trials.append(df)
                continue

            self._imu_ensure_time_column(df, time_col=time_col, trial_len_sec=trial_len_sec)

            Q_fix = self._imu_quat_from_cols(df, fixed_base, quat_order=quat_order)
            Q_move = self._imu_quat_from_cols(df, moving_base, quat_order=quat_order)

            R_fix = self._imu_quat_to_rotmat(Q_fix)
            R_move = self._imu_quat_to_rotmat(Q_move)

            R_fix_T = np.transpose(R_fix, (0, 2, 1))
            R_rel = np.einsum("nij,njk->nik", R_fix_T, R_move)

            v = np.einsum("nij,j->ni", R_rel, e_move)

            dot = (v * e_fix).sum(axis=1)
            dot = np.clip(dot, -1.0, 1.0)
            theta_rad = np.arccos(dot)
            theta_deg = np.degrees(theta_rad)

            df = df.copy()
            df[out_col] = theta_deg

            sub = df[[time_col, out_col]].copy()
            sub["trial_index"] = trial_idx
            sub["set_label"] = set_label
            tall_list.append(sub)
            aug_trials.append(df)

        if tall_list:
            tall = pd.concat(tall_list, axis=0, ignore_index=True)
        else:
            tall = pd.DataFrame(columns=[time_col, out_col, "trial_index", "set_label"])

        return aug_trials, tall

    def compute_imu_relative_angles_first(
        self,
        *,
        quat_cols: Tuple[str, str] = ("euler1", "euler2"),
        fixed_axis: str = "y",
        moving_axis: str = "y",
        quat_order: str = "wxyz",
        out_col: str = "imu_bend_deg",
        time_col: str = "t_sec",
        trial_len_sec: float | None = None,
    ) -> Tuple[List[Optional[pd.DataFrame]], pd.DataFrame]:
        """Convenience wrapper for first set."""
        imu_trials_first = self.extract_imu_dfs_first()
        return self.compute_imu_relative_angles_by_trial(
            imu_trials_first,
            set_label="first_cam",
            quat_cols=quat_cols,
            fixed_axis=fixed_axis,
            moving_axis=moving_axis,
            quat_order=quat_order,
            out_col=out_col,
            time_col=time_col,
            trial_len_sec=trial_len_sec,
        )

    def compute_imu_relative_angles_second(
        self,
        *,
        quat_cols: Tuple[str, str] = ("euler1", "euler2"),
        fixed_axis: str = "y",
        moving_axis: str = "y",
        quat_order: str = "wxyz",
        out_col: str = "imu_bend_deg",
        time_col: str = "t_sec",
        trial_len_sec: float | None = None,
    ) -> Tuple[List[Optional[pd.DataFrame]], pd.DataFrame]:
        """Convenience wrapper for second set."""
        imu_trials_second = self.extract_imu_dfs_second()
        return self.compute_imu_relative_angles_by_trial(
            imu_trials_second,
            set_label="second_cam",
            quat_cols=quat_cols,
            fixed_axis=fixed_axis,
            moving_axis=moving_axis,
            quat_order=quat_order,
            out_col=out_col,
            time_col=time_col,
            trial_len_sec=trial_len_sec,
        )

    # ------------------------------------------------------------------
    # IMU joint-angle pipeline via DLC3DBendAngles (old notebook)
    # ------------------------------------------------------------------
    def compute_imu_joint_angles_by_trial(
        self,
        imu_trials: List[Optional[pd.DataFrame]],
        *,
        set_label: str,
        trial_len_sec: float = 10.0,
        quat_cols: Tuple[str, str] = ("euler1", "euler2"),
        fixed_axis: str = "y",
        moving_axis: str = "y",
        quat_order: str = "wxyz",
        out_col: str = "imu_joint_deg_rx_py",
    ) -> Tuple[List[Optional[pd.DataFrame]], pd.DataFrame]:
        """
        Compute IMU joint angles for each trial using the SAME logic as the
        original BallBearingData.imu_augment_trials_inplace / imu_collect_tall,
        via DLC3DBendAngles.compute_joint_angle_trials.
        """
        dlc = DLC3DBendAngles(pd.DataFrame({"_": []}))

        valid_indices = [i for i, df in enumerate(imu_trials) if df is not None and not df.empty]
        valid_trials = [imu_trials[i] for i in valid_indices]

        if not valid_trials:
            empty_tall = pd.DataFrame(columns=["time_s", out_col, "trial_index", "set_label"])
            return imu_trials, empty_tall

        augmented_valid, tall = dlc.compute_joint_angle_trials(
            valid_trials,
            set_label=set_label,
            trial_len_sec=trial_len_sec,
            quat_cols=quat_cols,
            fixed_axis=fixed_axis,
            moving_axis=moving_axis,
            quat_order=quat_order,
        )

        col_src = "imu_joint_deg_rx_py"
        augmented_out: List[Optional[pd.DataFrame]] = [None] * len(imu_trials)

        for local_idx, global_idx in enumerate(valid_indices):
            df_aug = augmented_valid[local_idx]
            if df_aug is None or df_aug.empty:
                augmented_out[global_idx] = imu_trials[global_idx]
                continue

            df_aug = df_aug.copy()
            if col_src in df_aug.columns and out_col != col_src:
                df_aug[out_col] = df_aug[col_src]

            augmented_out[global_idx] = df_aug

        if not tall.empty and col_src in tall.columns and out_col != col_src:
            tall = tall.copy()
            tall[out_col] = tall[col_src]

        return augmented_out, tall

    def compute_imu_joint_angles_first(
        self,
        *,
        trial_len_sec: float = 10.0,
        quat_cols: Tuple[str, str] = ("euler1", "euler2"),
        fixed_axis: str = "y",
        moving_axis: str = "y",
        quat_order: str = "wxyz",
        out_col: str = "imu_joint_deg_rx_py",
    ) -> Tuple[List[Optional[pd.DataFrame]], pd.DataFrame]:
        """Compute IMU joint angles for the first set using DLC3DBendAngles."""
        imu_trials_first = self.extract_imu_dfs_first()
        return self.compute_imu_joint_angles_by_trial(
            imu_trials_first,
            set_label="first_cam",
            trial_len_sec=trial_len_sec,
            quat_cols=quat_cols,
            fixed_axis=fixed_axis,
            moving_axis=moving_axis,
            quat_order=quat_order,
            out_col=out_col,
        )

    def compute_imu_joint_angles_second(
        self,
        *,
        trial_len_sec: float = 10.0,
        quat_cols: Tuple[str, str] = ("euler1", "euler2"),
        fixed_axis: str = "y",
        moving_axis: str = "y",
        quat_order: str = "wxyz",
        out_col: str = "imu_joint_deg_rx_py",
    ) -> Tuple[List[Optional[pd.DataFrame]], pd.DataFrame]:
        """Compute IMU joint angles for the second set using DLC3DBendAngles."""
        imu_trials_second = self.extract_imu_dfs_second()
        return self.compute_imu_joint_angles_by_trial(
            imu_trials_second,
            set_label="second_cam",
            trial_len_sec=trial_len_sec,
            quat_cols=quat_cols,
            fixed_axis=fixed_axis,
            moving_axis=moving_axis,
            quat_order=quat_order,
            out_col=out_col,
        )

    # ------------------------------------------------------------------
    # Simple two-set IMU error boxplot
    # ------------------------------------------------------------------
    def plot_abs_error_box_two_sets_imu(
        self,
        *,
        refined_first: List[pd.DataFrame],
        summary_first: pd.DataFrame,
        refined_second: List[pd.DataFrame],
        summary_second: pd.DataFrame,
        dlc_col=("metric", "mcp_bend_deg", "deg"),
        imu_angle_col: str = "imu_joint_deg_rx_py_imu",
        trial_indices_first: Optional[Sequence[int]] = None,
        trial_indices_second: Optional[Sequence[int]] = None,
        n_best_first: int = 2,
        n_best_second: int = 2,
        label_first: str = "FIRST",
        label_second: str = "SECOND",
    ) -> None:
        """
        Boxplot of |DLC - IMU| absolute error for two sets (FIRST vs SECOND).
        """
        errs_first = self.collect_abs_error_for_set(
            refined_trials=refined_first,
            summary=summary_first,
            dlc_col=dlc_col,
            adc_rmse_col=imu_angle_col,
            trial_indices=trial_indices_first,
            n_best=n_best_first,
        )

        errs_second = self.collect_abs_error_for_set(
            refined_trials=refined_second,
            summary=summary_second,
            dlc_col=dlc_col,
            adc_rmse_col=imu_angle_col,
            trial_indices=trial_indices_second,
            n_best=n_best_second,
        )

        if errs_first.size == 0:
            print("[plot_abs_error_box_two_sets_imu] WARNING: no errors for FIRST set.")
        if errs_second.size == 0:
            print("[plot_abs_error_box_two_sets_imu] WARNING: no errors for SECOND set.")

        med_first = float(np.median(errs_first)) if errs_first.size > 0 else np.nan
        med_second = float(np.median(errs_second)) if errs_second.size > 0 else np.nan

        data = [errs_first, errs_second]
        labels = [label_first, label_second]

        fig, ax = plt.subplots(figsize=(6, 4))
        box = ax.boxplot(
            data,
            labels=labels,
            showfliers=False,
            patch_artist=True,
        )

        colors = ["tab:blue", "tab:orange"]
        for patch, c in zip(box["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.5)

        ax.set_ylabel("|DLC – IMU| error (deg)")
        ax.set_title("Absolute error between DLC and IMU bend angles")

        for i, med in enumerate([med_first, med_second], start=1):
            if not np.isnan(med):
                ax.text(
                    i,
                    med,
                    f"{med:.1f}°",
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    fontweight="bold",
                    color="black",
                )

        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Zero-baseline helper (used in pipeline)
    # ------------------------------------------------------------------
    @staticmethod
    def zero_baseline_trials(trials, angle_col: str):
        """
        Subtract the first non-NaN angle from each trial's time series
        so that each trial starts at 0 deg.
        """
        out = []
        for df in trials:
            if df is None or df.empty or angle_col not in df.columns:
                out.append(df)
                continue

            series = df[angle_col]
            valid = series.dropna()
            if valid.empty:
                out.append(df)
                continue

            offset = valid.iloc[0]

            df2 = df.copy()
            df2[angle_col] = df2[angle_col] - offset
            out.append(df2)

        return out

    @staticmethod
    def run_imu_cam_pipeline_for_participant_speed(
            root_dir: str,
            speed_tag: str,
            *,
            path_to_repo: str,
            time_unit: str = "ns",  # IMU 'timestamp' in ns, to match DLC side
            align_tolerance: str = "200ms",
            max_lag_samples: int = 5,
            trial_indices_first: Optional[Sequence[int]] = None,
            trial_indices_second: Optional[Sequence[int]] = None,
            min_dlc_likelihood: float = 0.9,
            dlc_bodyparts: Sequence[str] = ("MCP", "PIP", "hand"),

            # IMU orientation & angle naming
            imu_quat_cols: Tuple[str, str] = ("euler1", "euler2"),
            imu_fixed_axis: str = "y",
            imu_moving_axis: str = "y",
            imu_quat_order: str = "wxyz",

            # Which scalar IMU angle to use (e.g. 'imu_azimuth_deg' or 'imu_bend_deg')
            imu_angle_col: str = "imu_azimuth_deg",

            # Time column in IMU CSV used for alignment (keep 'timestamp' in ns)
            imu_time_col: str = "timestamp",

            # Windowed zero-baseline options
            imu_zero_window_n: int = 40,
            imu_zero_window_use_median: bool = True,
            imu_zero_window_abs: bool = True,

            # Jump filtering in IMU angle BEFORE alignment (deg). If None, skip.
            imu_jump_max_delta_deg: Optional[float] = 5.0,

            # Hard cap on aligned IMU angle (deg). If None, skip.
            imu_max_angle_deg: Optional[float] = 90.0,
    ) -> Dict[str, Any]:
        """
        IMU↔DLC pipeline for a scalar IMU angle (e.g. wrist azimuth) using the
        bend/pitch/azimuth computation in the wrist frame.

        Steps
        -----
          1. Discover imu1_/imu2_ trials (first/second).
          2. Compute IMU bend/pitch/azimuth via compute_imu_bend_pitch_azimuth_*.
             - The column imu_angle_col (e.g. 'imu_azimuth_deg') is filled.
          3. Optional: filter out big sample-to-sample jumps in imu_angle_col.
          4. Windowed zero-baseline imu_angle_col + optional abs().
          5. DLC → wrist bend angles + trigger timestamps.
          6. Align DLC wrist vs IMU angle using imu_time_col and 'cam_timestamp'.
          7. Optional: drop aligned IMU samples whose angle exceeds imu_max_angle_deg.
          8. refine_alignment_by_rmse_for_set with do_refine=False.
          9. Collect |DLC wrist - IMU angle| distributions.

        Notes
        -----
        - Typically imu_angle_col='imu_azimuth_deg' for azimuth, or 'imu_bend_deg'
          for pure flexion.
        """

        # ----------------------------------------------------------------
        # 1) Build IMU_cam + load trials (imu1_/imu2_)
        # ----------------------------------------------------------------
        cam = IMU_cam(
            root_dir=root_dir,
            path_to_repo=path_to_repo,
            folder_suffix_first=f"imu1_{speed_tag}",
            folder_suffix_second=f"imu2_{speed_tag}",
        )

        first_trials = cam.load_first()
        second_trials = cam.load_second()

        if len(first_trials) == 0 and len(second_trials) == 0:
            print("[run_imu_cam_pipeline_for_participant_speed] No trials found for this speed.")
            return {
                "abs_err_first": np.array([]),
                "abs_err_second": np.array([]),
                "summary_first": pd.DataFrame(),
                "summary_second": pd.DataFrame(),
                "merged_first": [],
                "merged_second": [],
                "refined_first": [],
                "refined_second": [],
            }

        # ----------------------------------------------------------------
        # 2) IMU bend/pitch/azimuth in wrist frame
        # ----------------------------------------------------------------
        TRIAL_LEN_SEC = 10.0

        # We compute all three but store the scalar we care about in imu_angle_col.
        aug_imu_first, imu_tall_first = cam.compute_imu_bend_pitch_azimuth_first(
            quat_cols=imu_quat_cols,
            fixed_axis=imu_fixed_axis,
            moving_axis=imu_moving_axis,
            plane_normal_axis="z",
            quat_order=imu_quat_order,
            bend_col="imu_bend_deg",
            pitch_col="imu_pitch_deg",
            azim_col=imu_angle_col,  # <--- this is the one we'll track
            time_col=imu_time_col,  # <--- keep the IMU's 'timestamp'
            trial_len_sec=TRIAL_LEN_SEC,
            zero_baseline_bend=False,
            zero_baseline_pitch=False,
            zero_baseline_azim=False,
        )

        aug_imu_second, imu_tall_second = cam.compute_imu_bend_pitch_azimuth_second(
            quat_cols=imu_quat_cols,
            fixed_axis=imu_fixed_axis,
            moving_axis=imu_moving_axis,
            plane_normal_axis="z",
            quat_order=imu_quat_order,
            bend_col="imu_bend_deg",
            pitch_col="imu_pitch_deg",
            azim_col=imu_angle_col,
            time_col=imu_time_col,
            trial_len_sec=TRIAL_LEN_SEC,
            zero_baseline_bend=False,
            zero_baseline_pitch=False,
            zero_baseline_azim=False,
        )

        # ----------------------------------------------------------------
        # 2b) Optional: filter out large sample-to-sample jumps in imu_angle_col
        # ----------------------------------------------------------------
        if imu_jump_max_delta_deg is not None:
            aug_imu_first = IMU_cam.filter_joint_angle_jumps_in_trials(
                aug_imu_first,
                angle_col=imu_angle_col,
                max_delta_deg=imu_jump_max_delta_deg,
                verbose=True,
            )
            aug_imu_second = IMU_cam.filter_joint_angle_jumps_in_trials(
                aug_imu_second,
                angle_col=imu_angle_col,
                max_delta_deg=imu_jump_max_delta_deg,
                verbose=True,
            )

        # ----------------------------------------------------------------
        # 2c) Windowed zero-baseline + optional abs()
        # ----------------------------------------------------------------
        aug_imu_first = IMU_cam.zero_baseline_trials_windowed(
            aug_imu_first,
            angle_col=imu_angle_col,
            n_points=imu_zero_window_n,
            use_median=imu_zero_window_use_median,
            abs_values=imu_zero_window_abs,
        )

        aug_imu_second = IMU_cam.zero_baseline_trials_windowed(
            aug_imu_second,
            angle_col=imu_angle_col,
            n_points=imu_zero_window_n,
            use_median=imu_zero_window_use_median,
            abs_values=imu_zero_window_abs,
        )

        # ----------------------------------------------------------------
        # 3) DLC 3D + wrist angles + trigger timestamps
        # ----------------------------------------------------------------
        cam_trials_first = cam.extract_trigger_time_dfs_by_trial(
            first_trials,
            add_labels=True,
            trial_labels=None,
            trial_base=1,
            set_label="first_cam",
            set_labels=None,
            include_path=True,
        )
        cam_trials_second = cam.extract_trigger_time_dfs_by_trial(
            second_trials,
            add_labels=True,
            trial_labels=None,
            trial_base=1,
            set_label="second_cam",
            set_labels=None,
            include_path=True,
        )

        dlc3d_trials_first = cam.extract_dlc3d_dfs_by_trial(
            first_trials,
            add_labels=True,
            trial_base=1,
            set_label="first_cam",
            include_path=True,
        )
        dlc3d_trials_second = cam.extract_dlc3d_dfs_by_trial(
            second_trials,
            add_labels=True,
            trial_base=1,
            set_label="second_cam",
            include_path=True,
        )

        dlc_aug_first, _ = cam.compute_dlc3d_angles_first(
            dlc3d_trials_first,
            signed_in_plane=True,
        )
        dlc_aug_second, _ = cam.compute_dlc3d_angles_second(
            dlc3d_trials_second,
            signed_in_plane=True,
        )

        dlc_angles_first = cam.attach_cam_timestamps_first(
            dlc_aug_first,
            cam_trials=cam_trials_first,
            time_col_name="timestamp",
            new_col_name="cam_timestamp",
        )
        dlc_angles_second = cam.attach_cam_timestamps_second(
            dlc_aug_second,
            cam_trials=cam_trials_second,
            time_col_name="timestamp",
            new_col_name="cam_timestamp",
        )

        dlc_angles_first = cam.filter_angle_trials_by_likelihood(
            dlc_angles_first,
            bodyparts=dlc_bodyparts,
            min_likelihood=min_dlc_likelihood,
        )
        dlc_angles_second = cam.filter_angle_trials_by_likelihood(
            dlc_angles_second,
            bodyparts=dlc_bodyparts,
            min_likelihood=min_dlc_likelihood,
        )

        # ----------------------------------------------------------------
        # 4) ALIGN IMU angle vs DLC wrist angle
        # ----------------------------------------------------------------
        merged_first = cam.align_adc_theta_to_dlc_angles_for_set(
            dlc_angle_trials=dlc_angles_first,
            adc_theta_trials=aug_imu_first,
            dlc_time_col="cam_timestamp",
            adc_time_col=imu_time_col,  # <--- use IMU 'timestamp' in ns
            adc_cols=[imu_angle_col],
            time_unit=time_unit,  # 'ns' to match both sides
            tolerance=align_tolerance,
            direction="nearest",
            suffix="_imu",
            keep_time_delta=True,
            drop_unmatched=True,
        )

        merged_second = cam.align_adc_theta_to_dlc_angles_for_set(
            dlc_angle_trials=dlc_angles_second,
            adc_theta_trials=aug_imu_second,
            dlc_time_col="cam_timestamp",
            adc_time_col=imu_time_col,
            adc_cols=[imu_angle_col],
            time_unit=time_unit,
            tolerance=align_tolerance,
            direction="nearest",
            suffix="_imu",
            keep_time_delta=True,
            drop_unmatched=True,
        )

        imu_aligned_col = f"{imu_angle_col}_imu"  # e.g. "imu_azimuth_deg_imu"

        # ----------------------------------------------------------------
        # 4b) OPTIONAL: drop impossible IMU angles above imu_max_angle_deg
        # ----------------------------------------------------------------
        if imu_max_angle_deg is not None:
            merged_first = IMU_cam.drop_angle_above_threshold_in_trials(
                merged_first,
                angle_col=imu_aligned_col,
                max_angle_deg=imu_max_angle_deg,
                verbose=True,
            )
            merged_second = IMU_cam.drop_angle_above_threshold_in_trials(
                merged_second,
                angle_col=imu_aligned_col,
                max_angle_deg=imu_max_angle_deg,
                verbose=True,
            )

        # ----------------------------------------------------------------
        # 5) RMSE at lag=0 (do_refine=False), WRIST ONLY (DLC)
        # ----------------------------------------------------------------
        wrist_col = ("metric", "wrist_bend_deg", "deg")
        time_col = "cam_timestamp"

        refined_first, summary_first = cam.refine_alignment_by_rmse_for_set(
            merged_trials=merged_first,
            dlc_col=wrist_col,
            adc_col=imu_aligned_col,
            max_lag_samples=max_lag_samples,
            time_col=time_col,
            plot_indices=None,
            set_name=f"{speed_tag.upper()} FIRST (IMU {imu_angle_col})",
            do_refine=False,
        )

        refined_second, summary_second = cam.refine_alignment_by_rmse_for_set(
            merged_trials=merged_second,
            dlc_col=wrist_col,
            adc_col=imu_aligned_col,
            max_lag_samples=max_lag_samples,
            time_col=time_col,
            plot_indices=None,
            set_name=f"{speed_tag.upper()} SECOND (IMU {imu_angle_col})",
            do_refine=False,
        )

        # ----------------------------------------------------------------
        # 6) Collect abs-error distributions
        # ----------------------------------------------------------------
        imu_err_col_for_box = imu_aligned_col

        abs_err_first = cam.collect_abs_error_for_set(
            refined_trials=refined_first,
            summary=summary_first,
            dlc_col=wrist_col,
            adc_rmse_col=imu_err_col_for_box,
            trial_indices=trial_indices_first,
            n_best=2,
        )

        abs_err_second = cam.collect_abs_error_for_set(
            refined_trials=refined_second,
            summary=summary_second,
            dlc_col=wrist_col,
            adc_rmse_col=imu_err_col_for_box,
            trial_indices=trial_indices_second,
            n_best=2,
        )

        return {
            "abs_err_first": abs_err_first,
            "abs_err_second": abs_err_second,
            "summary_first": summary_first,
            "summary_second": summary_second,
            "merged_first": merged_first,
            "merged_second": merged_second,
            "refined_first": refined_first,
            "refined_second": refined_second,
        }

    # ------------------------------------------------------------------
    # New robust IMU bend-angle method using plane normals
    # ------------------------------------------------------------------
    def compute_imu_plane_bend_by_trial(
        self,
        imu_trials: List[pd.DataFrame],
        *,
        set_label: str,
        quat_cols: Tuple[str, str] = ("euler1", "euler2"),
        quat_order: str = "wxyz",
        fixed_axes: Tuple[str, str] = ("x", "y"),
        moving_axes: Tuple[str, str] = ("x", "y"),
        out_col: str = "imu_plane_bend_deg",
        time_col: str = "t_sec",
        trial_len_sec: float | None = 10.0,
    ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        """
        New robust IMU bend-angle method using PLANE NORMALS.
        """
        base_fix, base_move = quat_cols
        a_fix, b_fix = fixed_axes
        a_mov, b_mov = moving_axes

        e = {
            "x": np.array([1.0, 0.0, 0.0]),
            "y": np.array([0.0, 1.0, 0.0]),
            "z": np.array([0.0, 0.0, 1.0]),
        }

        n_fix_body = np.cross(e[a_fix], e[b_fix])
        n_move_body = np.cross(e[a_mov], e[b_mov])

        tall_list: List[pd.DataFrame] = []
        aug_trials: List[pd.DataFrame] = []

        for trial_idx, df in enumerate(imu_trials):
            if df is None or df.empty:
                aug_trials.append(df)
                continue

            self._imu_ensure_time_column(df, time_col=time_col, trial_len_sec=trial_len_sec)

            Q_fix = self._imu_quat_from_cols(df, base_fix, quat_order=quat_order)
            Q_move = self._imu_quat_from_cols(df, base_move, quat_order=quat_order)

            R_fix = self._imu_quat_to_rotmat(Q_fix)
            R_move = self._imu_quat_to_rotmat(Q_move)

            nF = R_fix @ n_fix_body
            nM = R_move @ n_move_body

            R_fix_T = np.transpose(R_fix, (0, 2, 1))
            nM_in_fix = np.einsum("nij,nj->ni", R_fix_T, nM)

            dot = np.sum(nF * nM_in_fix, axis=1)
            dot = np.clip(dot, -1.0, 1.0)
            theta = np.degrees(np.arccos(dot))

            theta = theta - theta[0]

            df2 = df.copy()
            df2[out_col] = theta
            aug_trials.append(df2)

            sub = df2[[time_col, out_col]].copy()
            sub["trial_index"] = trial_idx
            sub["set_label"] = set_label
            tall_list.append(sub)

        if tall_list:
            tall = pd.concat(tall_list, axis=0, ignore_index=True)
        else:
            tall = pd.DataFrame(columns=[time_col, out_col, "trial_index", "set_label"])

        return aug_trials, tall

    # ------------------------------------------------------------------
    # Low-level IMU helper: relative pitch (wrist→palm) from quaternions
    # ------------------------------------------------------------------
    def _imu_relative_pitch_from_quats(
        self,
        Q_fix: np.ndarray,
        Q_move: np.ndarray,
    ) -> np.ndarray:
        """
        Compute relative *pitch* angle (in degrees) between two IMUs given
        their world-frame quaternions.
        """
        if Q_fix.shape != Q_move.shape:
            raise ValueError(
                f"Q_fix and Q_move must have same shape, got {Q_fix.shape} vs {Q_move.shape}"
            )
        if Q_fix.shape[1] != 4:
            raise ValueError(
                f"Quaternions must have 4 components [w,x,y,z], got shape {Q_fix.shape}"
            )

        Q_fix_conj = Q_fix.copy()
        Q_fix_conj[:, 1:] *= -1.0  # (w, -x, -y, -z)

        w1, x1, y1, z1 = Q_fix_conj.T
        w2, x2, y2, z2 = Q_move.T

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        Q_rel = np.column_stack([w, x, y, z])
        norms = np.linalg.norm(Q_rel, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        Q_rel /= norms

        w, x, y, z = Q_rel.T

        # Yaw–Pitch–Roll (Z–Y–X) pitch extraction: pitch = asin(2(wy - zx))
        sinp = 2.0 * (w * y - z * x)
        sinp = np.clip(sinp, -1.0, 1.0)
        pitch_rad = np.arcsin(sinp)
        pitch_deg = np.degrees(pitch_rad)

        # Or yaw-based mapping (you’d pick one; here we follow your last edit):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        yaw_rad = np.arctan2(siny_cosp, cosy_cosp)
        pitch_deg = np.degrees(yaw_rad)

        return -1 * pitch_deg

    # ------------------------------------------------------------------
    # IMU relative *pitch* pipeline (wrist vs palm, pure quaternion math)
    # ------------------------------------------------------------------
    def compute_imu_relative_pitch_by_trial(
        self,
        imu_trials: List[Optional[pd.DataFrame]],
        *,
        set_label: str,
        quat_cols: Tuple[str, str] = ("euler1", "euler2"),
        quat_order: str = "wxyz",
        out_col: str = "imu_rel_pitch_deg",
        time_col: str = "t_sec",
        trial_len_sec: float | None = None,
        zero_baseline: bool = False,
    ) -> Tuple[List[Optional[pd.DataFrame]], pd.DataFrame]:
        """
        Compute palm flexion angle as the *relative pitch* between IMU1 (wrist)
        and IMU2 (palm) using only quaternions.
        """
        if len(quat_cols) != 2:
            raise ValueError(f"quat_cols must have length 2, got {quat_cols}")

        base_fix, base_move = quat_cols

        tall_list: List[pd.DataFrame] = []
        aug_trials: List[Optional[pd.DataFrame]] = []

        for trial_idx, df in enumerate(imu_trials):
            if df is None or df.empty:
                aug_trials.append(df)
                continue

            self._imu_ensure_time_column(df, time_col=time_col, trial_len_sec=trial_len_sec)

            Q_fix = self._imu_quat_from_cols(df, base_fix, quat_order=quat_order)
            Q_move = self._imu_quat_from_cols(df, base_move, quat_order=quat_order)

            pitch_deg = self._imu_relative_pitch_from_quats(Q_fix, Q_move)

            if zero_baseline:
                valid = np.isfinite(pitch_deg)
                if np.any(valid):
                    pitch_deg = pitch_deg - pitch_deg[valid][0]

            df2 = df.copy()
            df2[out_col] = pitch_deg
            aug_trials.append(df2)

            sub = df2[[time_col, out_col]].copy()
            sub["trial_index"] = trial_idx
            sub["set_label"] = set_label
            tall_list.append(sub)

        if tall_list:
            tall = pd.concat(tall_list, axis=0, ignore_index=True)
        else:
            tall = pd.DataFrame(columns=[time_col, out_col, "trial_index", "set_label"])

        return aug_trials, tall

    def compute_imu_relative_pitch_first(
        self,
        *,
        quat_cols: Tuple[str, str] = ("euler1", "euler2"),
        quat_order: str = "wxyz",
        out_col: str = "imu_rel_pitch_deg",
        time_col: str = "t_sec",
        trial_len_sec: float | None = None,
        zero_baseline: bool = False,
    ) -> Tuple[List[Optional[pd.DataFrame]], pd.DataFrame]:
        """Relative pitch for first (e.g. imu1_*) set."""
        imu_first = self.extract_imu_dfs_first()
        return self.compute_imu_relative_pitch_by_trial(
            imu_first,
            set_label="first_cam",
            quat_cols=quat_cols,
            quat_order=quat_order,
            out_col=out_col,
            time_col=time_col,
            trial_len_sec=trial_len_sec,
            zero_baseline=zero_baseline,
        )

    def compute_imu_relative_pitch_second(
        self,
        *,
        quat_cols: Tuple[str, str] = ("euler1", "euler2"),
        quat_order: str = "wxyz",
        out_col: str = "imu_rel_pitch_deg",
        time_col: str = "t_sec",
        trial_len_sec: float | None = None,
        zero_baseline: bool = False,
    ) -> Tuple[List[Optional[pd.DataFrame]], pd.DataFrame]:
        """Relative pitch for second (e.g. imu2_*) set."""
        imu_second = self.extract_imu_dfs_second()
        return self.compute_imu_relative_pitch_by_trial(
            imu_second,
            set_label="second_cam",
            quat_cols=quat_cols,
            quat_order=quat_order,
            out_col=out_col,
            time_col=time_col,
            trial_len_sec=trial_len_sec,
            zero_baseline=zero_baseline,
        )

    # ------------------------------------------------------------------
    # Trigger-time CSV convenience wrappers
    # ------------------------------------------------------------------
    def extract_trigger_time_dfs_first(
        self,
        *,
        add_labels: bool = True,
        trial_base: int = 1,
        set_label: str = "first_cam",
        include_path: bool = True,
    ) -> List[pd.DataFrame]:
        """Load first-application trials and return trigger-time CSV DataFrames per trial."""
        trials_first = self.load_first()
        return self.extract_trigger_time_dfs_by_trial(
            trials_first,
            add_labels=add_labels,
            trial_labels=None,
            trial_base=trial_base,
            set_label=set_label,
            set_labels=None,
            include_path=include_path,
        )

    def extract_trigger_time_dfs_second(
        self,
        *,
        add_labels: bool = True,
        trial_base: int = 1,
        set_label: str = "second_cam",
        include_path: bool = True,
    ) -> List[pd.DataFrame]:
        """Load second-application trials and return trigger-time CSV DataFrames per trial."""
        trials_second = self.load_second()
        return self.extract_trigger_time_dfs_by_trial(
            trials_second,
            add_labels=add_labels,
            trial_labels=None,
            trial_base=trial_base,
            set_label=set_label,
            set_labels=None,
            include_path=include_path,
        )

    # ------------------------------------------------------------------
    # IMU_cam version of refine_alignment_by_rmse_for_set
    # ------------------------------------------------------------------
    def refine_alignment_by_rmse_for_set(
        self,
        merged_trials: List[pd.DataFrame],
        *,
        dlc_col=("metric", "mcp_bend_deg", "deg"),
        adc_col: str = "theta_cam_cal_adc",
        max_lag_samples: int = 5,
        time_col=("_t_cam_td", "", ""),  # or "cam_timestamp"
        plot_indices: Optional[Sequence[int]] = None,
        set_name: str = "set",
        do_refine: bool = True,
    ) -> Tuple[List[pd.DataFrame], pd.DataFrame]:
        """
        Refines ADC/IMU vs DLC alignment by searching over integer lags
        to minimize RMSE between `dlc_col` and `adc_col`, or,
        if do_refine=False, just computes RMSE at lag=0.
        """
        effective_max_lag = max_lag_samples if do_refine else 0

        refined: List[pd.DataFrame] = []
        records: List[Dict[str, Any]] = []

        for i, df in enumerate(merged_trials):
            if df is None or df.empty:
                refined.append(pd.DataFrame())
                records.append(
                    dict(
                        trial_index=i,
                        best_lag_samples=0,
                        rmse_deg=np.nan,
                        n_points=0,
                        set_name=set_name,
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
                    set_name=set_name,
                )
            )

        summary = pd.DataFrame.from_records(records)

        if plot_indices is not None:
            for idx in plot_indices:
                if idx < 0 or idx >= len(refined):
                    continue
                df_pl = refined[idx]
                if df_pl is None or df_pl.empty:
                    continue

                if isinstance(time_col, tuple) and time_col in df_pl.columns:
                    t_raw = df_pl[time_col]
                    if hasattr(t_raw, "dt"):
                        t = (t_raw - t_raw.iloc[0]).dt.total_seconds().to_numpy()
                    else:
                        t = t_raw.to_numpy(dtype=float)
                        t = t - t[0]
                elif isinstance(time_col, str) and time_col in df_pl.columns:
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

                if len(t) > 1 and np.nanmax(t) > np.nanmin(t):
                    t_plot = (t - np.nanmin(t)) / (np.nanmax(t) - np.nanmin(t)) * 10.0
                else:
                    t_plot = np.zeros_like(t, dtype=float)

                theta_dlc = df_pl[dlc_col]
                theta_adc_rmse = df_pl[adc_col + "_rmse"]

                dlc_arr = theta_dlc.to_numpy(dtype=float)
                adc_arr = theta_adc_rmse.to_numpy(dtype=float)
                both_valid = np.isfinite(dlc_arr) & np.isfinite(adc_arr)

                idx_max = None
                if both_valid.sum() > 0:
                    abs_diff = np.full_like(dlc_arr, np.nan, dtype=float)
                    abs_diff[both_valid] = np.abs(
                        dlc_arr[both_valid] - adc_arr[both_valid]
                    )
                    idx_max = int(np.nanargmax(abs_diff))

                plt.figure(figsize=(10, 4))
                plt.plot(t_plot, theta_dlc, label="DLC angle")
                plt.plot(
                    t_plot,
                    theta_adc_rmse,
                    label="IMU angle (RMSE-aligned)",
                )

                if (
                    idx_max is not None
                    and np.isfinite(dlc_arr[idx_max])
                    and np.isfinite(adc_arr[idx_max])
                ):
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

                plt.xlabel("Scaled time (0–10 s)")
                plt.ylabel("Angle (deg)")
                plt.title(
                    f"{set_name} trial {idx + 1}: DLC vs IMU (RMSE column, do_refine={do_refine})"
                )
                plt.xlim(0.0, 10.0)
                plt.legend()
                plt.grid(True)
                plt.tight_layout()

        return refined, summary

    # ------------------------------------------------------------------
    # Row filter: iteratively drop samples with large jumps in angle_col
    # ------------------------------------------------------------------
    @staticmethod
    def filter_joint_angle_jumps_in_trials(
            trials: List[Optional[pd.DataFrame]],
            angle_col: str,
            max_delta_deg: float = 60.0,
            verbose: bool = True,
            max_iters: int = 20,
    ) -> List[Optional[pd.DataFrame]]:
        """
        For each trial, iteratively drop rows where the step-to-step change
        in `angle_col` exceeds `max_delta_deg`.

        This fixes plateau-style glitches where a long run of bad values
        would otherwise leave a large jump after a single-pass filter.

        Algorithm (per trial)
        ---------------------
        1) s = angle series
        2) while True:
              - compute d = diff(s)
              - mark the *later* sample of any |d| > max_delta_deg as bad
              - if no bad samples or we reach max_iters: break
              - drop bad rows and repeat on the shortened series

        Parameters
        ----------
        trials : list[DataFrame | None]
            List of per-trial DataFrames (e.g. aug_imu_first).
        angle_col : str
            Name of the joint-angle column to clean
            (e.g. 'imu_joint_deg_rx_py').
        max_delta_deg : float, default 60.0
            Threshold for |Δangle|; if the jump into row i is larger than
            this, row i is removed.
        verbose : bool, default True
            If True, print how many rows were removed per trial and iteration.
        max_iters : int, default 20
            Safety cap to avoid infinite loops in pathological cases.

        Returns
        -------
        cleaned_trials : list[DataFrame | None]
            Same structure as input but with bad rows removed.
        """
        cleaned: List[Optional[pd.DataFrame]] = []

        for trial_idx, df in enumerate(trials):
            if df is None or df.empty or angle_col not in df.columns:
                cleaned.append(df)
                continue

            df_clean = df.copy()

            for it in range(max_iters):
                s = df_clean[angle_col].to_numpy(dtype=float)

                if len(s) < 2:
                    break

                d = np.diff(s)
                bad = np.insert(np.abs(d) > max_delta_deg, 0, False)
                n_bad = int(bad.sum())

                if n_bad == 0:
                    if verbose:
                        print(
                            f"[filter_joint_angle_jumps_in_trials] "
                            f"trial {trial_idx}: converged after {it} iterations."
                        )
                    break

                if verbose:
                    print(
                        f"[filter_joint_angle_jumps_in_trials] "
                        f"trial {trial_idx}, iter {it}: "
                        f"dropping {n_bad} rows (|Δ{angle_col}| > {max_delta_deg}°)"
                    )

                df_clean = df_clean.loc[~bad].reset_index(drop=True)

            cleaned.append(df_clean)

        return cleaned

    # ------------------------------------------------------------------
    # Row filter: drop samples where angle_col exceeds a hard threshold
    # ------------------------------------------------------------------
    @staticmethod
    def drop_angle_above_threshold_in_trials(
        trials: List[Optional[pd.DataFrame]],
        angle_col: str,
        max_angle_deg: float = 90.0,
        verbose: bool = True,
    ) -> List[Optional[pd.DataFrame]]:
        """
        For each trial DataFrame in `trials`, drop rows where `angle_col`
        exceeds `max_angle_deg`.

        Intended use: after DLC↔IMU alignment, clean up obviously
        impossible IMU angles (e.g., > 90°).

        Parameters
        ----------
        trials : list[DataFrame | None]
            List of per-trial DataFrames (e.g. merged_first / merged_second).
        angle_col : str
            Name of the angle column to threshold
            (e.g. 'imu_joint_deg_rx_py_imu').
        max_angle_deg : float, default 90.0
            Rows with angle_col > max_angle_deg are removed.
        verbose : bool, default True
            If True, print how many rows were dropped per trial.

        Returns
        -------
        cleaned_trials : list[DataFrame | None]
            Same as input but with rows above the threshold removed.
        """
        cleaned: List[Optional[pd.DataFrame]] = []

        for trial_idx, df in enumerate(trials):
            if df is None or df.empty or angle_col not in df.columns:
                cleaned.append(df)
                continue

            df = df.copy()
            mask_keep = (df[angle_col].isna()) | (df[angle_col] <= max_angle_deg)
            n_drop = int((~mask_keep).sum())

            if verbose and n_drop > 0:
                print(
                    f"[drop_angle_above_threshold_in_trials] "
                    f"trial {trial_idx}: dropped {n_drop} rows "
                    f"({angle_col} > {max_angle_deg}°)"
                )

            df_clean = df.loc[mask_keep].reset_index(drop=True)
            cleaned.append(df_clean)

        return cleaned


    def compute_imu_bend_pitch_azimuth_first(
        self,
        *,
        quat_cols: Tuple[str, str] = ("euler1", "euler2"),
        fixed_axis: str = "y",
        moving_axis: str = "y",
        plane_normal_axis: str = "z",
        quat_order: str = "wxyz",
        bend_col: str = "imu_bend_deg",
        pitch_col: str = "imu_pitch_deg",
        azim_col: str = "imu_azimuth_deg",
        time_col: str = "t_sec",
        trial_len_sec: float | None = None,
        zero_baseline_bend: bool = False,
        zero_baseline_pitch: bool = False,
        zero_baseline_azim: bool = False,
    ) -> Tuple[List[Optional[pd.DataFrame]], pd.DataFrame]:
        """Convenience wrapper: bend + pitch + azimuth for FIRST (imu1_*) set."""
        imu_first = self.extract_imu_dfs_first()
        return self.compute_imu_bend_pitch_azimuth_by_trial(
            imu_first,
            set_label="first_cam",
            quat_cols=quat_cols,
            fixed_axis=fixed_axis,
            moving_axis=moving_axis,
            plane_normal_axis=plane_normal_axis,
            quat_order=quat_order,
            bend_col=bend_col,
            pitch_col=pitch_col,
            azim_col=azim_col,
            time_col=time_col,
            trial_len_sec=trial_len_sec,
            zero_baseline_bend=zero_baseline_bend,
            zero_baseline_pitch=zero_baseline_pitch,
            zero_baseline_azim=zero_baseline_azim,
        )

    def compute_imu_bend_pitch_azimuth_second(
        self,
        *,
        quat_cols: Tuple[str, str] = ("euler1", "euler2"),
        fixed_axis: str = "y",
        moving_axis: str = "y",
        plane_normal_axis: str = "z",
        quat_order: str = "wxyz",
        bend_col: str = "imu_bend_deg",
        pitch_col: str = "imu_pitch_deg",
        azim_col: str = "imu_azimuth_deg",
        time_col: str = "t_sec",
        trial_len_sec: float | None = None,
        zero_baseline_bend: bool = False,
        zero_baseline_pitch: bool = False,
        zero_baseline_azim: bool = False,
    ) -> Tuple[List[Optional[pd.DataFrame]], pd.DataFrame]:
        """Convenience wrapper: bend + pitch + azimuth for SECOND (imu2_*) set."""
        imu_second = self.extract_imu_dfs_second()
        return self.compute_imu_bend_pitch_azimuth_by_trial(
            imu_second,
            set_label="second_cam",
            quat_cols=quat_cols,
            fixed_axis=fixed_axis,
            moving_axis=moving_axis,
            plane_normal_axis=plane_normal_axis,
            quat_order=quat_order,
            bend_col=bend_col,
            pitch_col=pitch_col,
            azim_col=azim_col,
            time_col=time_col,
            trial_len_sec=trial_len_sec,
            zero_baseline_bend=zero_baseline_bend,
            zero_baseline_pitch=zero_baseline_pitch,
            zero_baseline_azim=zero_baseline_azim,
        )

    # ------------------------------------------------------------------
    # Relative bend (your current method) + pitch + azimuth in wrist frame
    # ------------------------------------------------------------------
    def compute_imu_bend_pitch_azimuth_by_trial(
        self,
        imu_trials: List[Optional[pd.DataFrame]],
        *,
        set_label: str,
        quat_cols: Tuple[str, str] = ("euler1", "euler2"),
        fixed_axis: str = "y",
        moving_axis: str = "y",
        plane_normal_axis: str = "z",
        quat_order: str = "wxyz",
        bend_col: str = "imu_bend_deg",
        pitch_col: str = "imu_pitch_deg",
        azim_col: str = "imu_azimuth_deg",
        time_col: str = "t_sec",
        trial_len_sec: float | None = None,
        zero_baseline_bend: bool = False,
        zero_baseline_pitch: bool = False,
        zero_baseline_azim: bool = False,
    ) -> Tuple[List[Optional[pd.DataFrame]], pd.DataFrame]:
        """
        Compute three wrist-frame angles for each sample:

          1) bend (same scalar as your current method)
          2) pitch  = flexion in the chosen bending plane (wrist frame)
          3) azimuth = tilt out of that plane (signed)

        Construction
        ------------
        - Let e_fix be the fixed (wrist) IMU axis, and e_move the palm IMU axis
          you currently use.
        - Form R_fix, R_move from quats, then:

              R_rel = R_fix^T @ R_move

        - Express the moving axis in the wrist frame:

              v = R_rel @ e_move   (shape (N,3))

        - Total bend (your existing scalar):

              bend = arccos( dot( v_hat, e_fix ) )

        - Choose a wrist-frame plane normal n_plane (e.g. 'z') representing the
          "out-of-plane" direction for bending.
        - Decompose v_hat into in-plane + out-of-plane:

              v_hat     = v / ||v||
              n_plane   = unit vector for plane_normal_axis
              v_out     = (v_hat · n_plane) n_plane
              v_plane   = v_hat - v_out

        - Pitch = angle between e_fix and v_plane (within bending plane)
        - Azimuth = signed out-of-plane angle:

              azim = arcsin( v_hat · n_plane )

        Parameters
        ----------
        imu_trials : list[DataFrame | None]
            Per-trial IMU CSVs (e.g. from extract_imu_dfs_by_trial).
        set_label : str
            Label for tall-format output ('first_cam', 'second_cam', etc.).
        quat_cols : (str, str), default ('euler1', 'euler2')
            Base names for wrist and palm quaternions.
        fixed_axis, moving_axis : {'x','y','z'}
            IMU body axes for the wrist and palm unit vectors.
        plane_normal_axis : {'x','y','z'}, default 'z'
            Wrist-frame axis that defines the *normal* to your bending plane.
            For example, if bending is mostly in the y–z plane, choose 'x'
            as the normal; if bending is in x–y, choose 'z', etc.
        quat_order : str, default 'wxyz'
            Order of components in your CSV.
        bend_col, pitch_col, azim_col : str
            Output column names for bend, pitch, and azimuth (deg).
        time_col : str, default 't_sec'
            Time column to ensure/create.
        trial_len_sec : float or None, default None
            If provided and time_col missing, create linspace [0, trial_len_sec).
        zero_baseline_* : bool
            If True, subtract the first non-NaN value per trial for that angle.

        Returns
        -------
        aug_trials : list[DataFrame | None]
            Trials with new columns [bend_col, pitch_col, azim_col] added.
        tall : DataFrame
            Tall-format table with [time_col, bend_col, pitch_col, azim_col,
            trial_index, set_label].
        """
        if len(quat_cols) != 2:
            raise ValueError(f"quat_cols must have length 2, got {quat_cols}")

        base_fix, base_move = quat_cols  # wrist (fixed), palm (moving)

        # Wrist-frame basis vectors
        e_fix = self._imu_axis_vector(fixed_axis)
        e_move = self._imu_axis_vector(moving_axis)
        n_plane = self._imu_axis_vector(plane_normal_axis)

        # Ensure plane normal is unit
        n_plane = n_plane / np.linalg.norm(n_plane)

        tall_list: List[pd.DataFrame] = []
        aug_trials: List[Optional[pd.DataFrame]] = []

        for trial_idx, df in enumerate(imu_trials):
            if df is None or df.empty:
                aug_trials.append(df)
                continue

            # Ensure time column
            self._imu_ensure_time_column(df, time_col=time_col, trial_len_sec=trial_len_sec)

            # Quaternions in canonical [w,x,y,z]
            Q_fix = self._imu_quat_from_cols(df, base_fix, quat_order=quat_order)
            Q_move = self._imu_quat_from_cols(df, base_move, quat_order=quat_order)

            # Rotation matrices
            R_fix = self._imu_quat_to_rotmat(Q_fix)
            R_move = self._imu_quat_to_rotmat(Q_move)

            # Relative rotation: palm in wrist frame
            R_fix_T = np.transpose(R_fix, (0, 2, 1))
            R_rel = np.einsum("nij,njk->nik", R_fix_T, R_move)

            # Moving axis expressed in wrist frame
            v = np.einsum("nij,j->ni", R_rel, e_move)  # shape (N,3)

            # Normalize v
            v_norm = np.linalg.norm(v, axis=1, keepdims=True)
            valid = v_norm.squeeze() > 0
            v_hat = np.zeros_like(v)
            v_hat[valid] = v[valid] / v_norm[valid]

            # 1) Total bend (your existing scalar)
            dot_bend = np.einsum("ni,i->n", v_hat, e_fix)
            dot_bend = np.clip(dot_bend, -1.0, 1.0)
            bend_rad = np.arccos(dot_bend)
            bend_deg = np.degrees(bend_rad)

            # 2) Decompose into in-plane + out-of-plane
            dot_out = np.einsum("ni,i->n", v_hat, n_plane)  # component along plane normal
            v_out = dot_out[:, None] * n_plane[None, :]
            v_plane = v_hat - v_out

            plane_norm = np.linalg.norm(v_plane, axis=1)
            pitch_deg = np.full_like(bend_deg, np.nan, dtype=float)
            azim_deg = np.full_like(bend_deg, np.nan, dtype=float)

            # For rows where in-plane component is usable
            good_plane = plane_norm > 1e-8
            if np.any(good_plane):
                v_plane_hat = np.zeros_like(v_plane)
                v_plane_hat[good_plane] = v_plane[good_plane] / plane_norm[good_plane, None]

                # Pitch: angle between e_fix and v_plane (within bending plane)
                dot_pitch = np.einsum("ni,i->n", v_plane_hat, e_fix)
                dot_pitch = np.clip(dot_pitch, -1.0, 1.0)
                pitch_rad = np.arccos(dot_pitch)
                pitch_deg[good_plane] = np.degrees(pitch_rad)[good_plane]

            # Azimuth: signed out-of-plane tilt (relative to bending plane)
            # v_hat·n_plane is the sine of out-of-plane tilt if bend is moderate.
            dot_out_clipped = np.clip(dot_out, -1.0, 1.0)
            azim_rad = np.arcsin(dot_out_clipped)
            azim_deg = np.degrees(azim_rad)

            # Optional zero-baselining
            def _zero_baseline(arr: np.ndarray) -> np.ndarray:
                arr2 = arr.copy()
                valid2 = np.isfinite(arr2)
                if np.any(valid2):
                    arr2 -= arr2[valid2][0]
                return arr2

            if zero_baseline_bend:
                bend_deg = _zero_baseline(bend_deg)
            if zero_baseline_pitch:
                pitch_deg = _zero_baseline(pitch_deg)
            if zero_baseline_azim:
                azim_deg = _zero_baseline(azim_deg)

            # Attach to DataFrame
            df2 = df.copy()
            df2[bend_col] = bend_deg
            df2[pitch_col] = pitch_deg
            df2[azim_col] = azim_deg
            aug_trials.append(df2)

            # Tall-format for analysis
            sub = df2[[time_col, bend_col, pitch_col, azim_col]].copy()
            sub["trial_index"] = trial_idx
            sub["set_label"] = set_label
            tall_list.append(sub)

        if tall_list:
            tall = pd.concat(tall_list, axis=0, ignore_index=True)
        else:
            tall = pd.DataFrame(
                columns=[time_col, bend_col, pitch_col, azim_col, "trial_index", "set_label"]
            )

        return aug_trials, tall

    # ------------------------------------------------------------------
    # Zero-baseline using the average/median of the first N samples
    # with optional absolute value
    # ------------------------------------------------------------------
    @staticmethod
    def zero_baseline_trials_windowed(
            trials: List[Optional[pd.DataFrame]],
            angle_col: str,
            n_points: int = 10,
            use_median: bool = False,
            abs_values: bool = False,
    ) -> List[Optional[pd.DataFrame]]:
        """
        Subtract a baseline defined as the mean (or median) of the first
        N non-NaN samples in `angle_col` for each trial.

        Optionally convert the entire corrected series to absolute value.

        Parameters
        ----------
        trials : list[DataFrame | None]
            Per-trial DataFrames (e.g. aug_imu_first).
        angle_col : str
            Name of the angle column to baseline (e.g. 'imu_joint_deg_rx_py').
        n_points : int, default 10
            Number of initial valid samples to use for the baseline.
        use_median : bool, default False
            If True, use the median of the first N samples instead of the mean.
        abs_values : bool, default False
            If True, take np.abs() of the entire angle_col after baselining.

        Returns
        -------
        out : list[DataFrame | None]
            Same structure as input, but with `angle_col` shifted so that
            its first N samples (on average) are ~0, and optionally abs().
        """
        out: List[Optional[pd.DataFrame]] = []

        for df in trials:
            if df is None or df.empty or angle_col not in df.columns:
                out.append(df)
                continue

            series = df[angle_col].astype(float)

            # First N non-NaN samples
            valid = series.dropna()
            if valid.empty:
                out.append(df)
                continue

            window = valid.iloc[:n_points]
            if window.empty:
                out.append(df)
                continue

            baseline = window.median() if use_median else window.mean()

            df2 = df.copy()
            df2[angle_col] = df2[angle_col] - baseline

            if abs_values:
                df2[angle_col] = df2[angle_col].abs()

            out.append(df2)

        return out

    @staticmethod
    def plot_imu_abs_error_summary_grid(
            results_imu: dict,
            *,
            speed_titles: dict | None = None,
            speed_order: list[str] | None = None,
            imu_angle_col: str = "imu_azimuth_deg",
            example1_participant: str = "P4",
            example1_speed: str = "slow",
            example2_participant: str = "P3",
            example2_speed: str = "vfas",
            example_trial_idx: int = 0,
            example_col_start: int = 1,
            figsize: tuple[float, float] = (20, 12),
            title_fontsize: int = 14,
            label_fontsize: int = 12,
            tick_fontsize: int = 10,
            title_weight: str = "bold",
            label_weight: str = "bold",
            # global y-limit (fallback)
            abs_err_ylim: tuple[float, float] | None = None,
            # separate tuners
            abs_err_ylim_examples: tuple[float, float] | None = None,
            abs_err_ylim_summary: tuple[float, float] | None = None,
            # horizontal spacing for whole grid (used for all columns initially)
            boxplot_wspace: float = 0.02,
            # scale gap between TS and boxplot in first two rows
            example_col_gap_scale: float = 1.0,
            # fixed gap between AFAP boxplot and summary bar plots (rows 3–4)
            bar_gap_from_afap: float = 0.33,
            # NEW: drop specific participant–speed combos from summary
            exclude_participant_speed_for_summary: list[tuple[str, str]] | None = None,
    ):
        """
        Seaborn summary figure for IMU abs-error results.

        Layout
        ------
        Row 1:
          [0, example_col_start + 1] : DLC vs IMU (0–10 s) for example trial 1
          [0, example_col_start + 2] : boxplot of |DLC–IMU| for that slow trial

        Row 2:
          [1, example_col_start + 1] : DLC vs IMU (0–10 s) for example trial 2
          [1, example_col_start + 2] : boxplot of |DLC–IMU| for that vfas trial

        Row 3 (6 plots):
          FIRST set – per-speed boxplots by participant in cols 0–4,
          plus bar-summary of mean error vs speed in col 5.

          - First boxplot (col 0) keeps y-axis label/ticks.
          - Boxplots in cols 1–4 have NO y-axis at all.

        Row 4 (6 plots):
          Same as row 3, but SECOND set.

        Y-range controls
        ----------------
        - `abs_err_ylim_examples`: y-limits for example boxplots (rows 1–2).
        - `abs_err_ylim_summary`: y-limits for summary boxplots (rows 3–4).
        - `abs_err_ylim`: fallback applied to both if specific ones are not given.

        Spacing
        -------
        - `boxplot_wspace`: base horizontal spacing between subplot columns.
        - `example_col_gap_scale`: rescales the gap between TS and example
           boxplot in rows 0–1 (<1 = closer, >1 = farther).
        - `bar_gap_from_afap`: absolute horizontal gap (figure coords) between
           the rightmost boxplot (AFAP, col 4) and the summary bar plots (col 5)
           in rows 2–3.
        """
        # -----------------------------
        # 1) Build tall DataFrame of abs errors
        # -----------------------------
        records = []

        for (pname, speed), res in results_imu.items():
            if res is None:
                continue

            for set_label, key_err in [("FIRST", "abs_err_first"),
                                       ("SECOND", "abs_err_second")]:
                if key_err not in res:
                    continue

                arr = res.get(key_err)
                if arr is None:
                    continue

                vals = np.asarray(arr).ravel()
                vals = vals[~np.isnan(vals)]
                for v in vals:
                    records.append(
                        dict(
                            participant=pname,
                            speed=speed,
                            set=set_label,
                            abs_err_deg=float(v),
                        )
                    )

        if not records:
            print("[plot_imu_abs_error_summary_grid] No error data found.")
            return

        # -----------------------------
        # 2) DataFrame + speed labels
        # -----------------------------
        df = pd.DataFrame.from_records(records)

        if speed_titles is not None:
            df["speed_label"] = df["speed"].map(
                lambda s: speed_titles.get(s, s)
            )
        else:
            df["speed_label"] = df["speed"]

        # ------------------------------------------------------
        # Optional: drop specific participant–speed combinations
        # from the summary (boxplots + bar plots)
        # ------------------------------------------------------
        if exclude_participant_speed_for_summary:
            mask_exclude = pd.Series(False, index=df.index)
            for pname, spd in exclude_participant_speed_for_summary:
                mask_exclude |= (
                    (df["participant"] == pname)
                    & (df["speed"] == spd)
                    & df["set"].isin(["FIRST", "SECOND"])
                )
            df = df.loc[~mask_exclude].copy()


        if speed_order is None:
            speed_order = sorted(df["speed"].unique())

        # -----------------------------
        # 3) Seaborn theme & figure
        # -----------------------------
        sns.set_theme(style="ticks", context="talk")

        nrows, ncols = 4, 6
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

        def _style_ax(ax):
            """Base styling: remove top/right spines, no grid, set tick font size."""
            sns.despine(ax=ax, top=True, right=True)
            ax.grid(False)
            ax.tick_params(labelsize=tick_fontsize)

        def _hide_y_axis(ax):
            """Completely hide y-axis (ticks, labels, and vertical spines)."""
            ax.yaxis.set_visible(False)
            ax.tick_params(axis="y", left=False, labelleft=False)
            for side in ("left", "right"):
                if side in ax.spines:
                    ax.spines[side].set_visible(False)

        # -----------------------------
        # Helper: get example trial series and 0–10 s time axis
        # -----------------------------
        def _get_example_trial(participant: str, speed: str, set_key="refined_first"):
            key = (participant, speed)
            imu_aligned_col = f"{imu_angle_col}_imu"
            dlc_col = ("metric", "wrist_bend_deg", "deg")

            if key not in results_imu:
                return None

            res = results_imu[key]
            trials = res.get(set_key) or res.get("merged_first")
            if not trials:
                return None
            if example_trial_idx >= len(trials):
                return None

            df_trial = trials[example_trial_idx]
            if df_trial is None or df_trial.empty:
                return None
            if dlc_col not in df_trial.columns or imu_aligned_col not in df_trial.columns:
                return None

            dlc_angle = df_trial[dlc_col].to_numpy(dtype=float)
            imu_angle = df_trial[imu_aligned_col].to_numpy(dtype=float)

            # Time axis
            if "cam_timestamp" in df_trial.columns:
                t_raw = df_trial["cam_timestamp"].to_numpy(dtype=float)
                t_plot = (t_raw - t_raw[0]) / 1e9  # ns → s
            elif "_t_cam_td" in df_trial.columns:
                t_td = df_trial["_t_cam_td"]
                t_plot = t_td.dt.total_seconds().to_numpy()
                t_plot = t_plot - t_plot[0]
            else:
                n = len(df_trial)
                t_plot = np.linspace(0.0, 10.0, n, endpoint=False)

            t_min, t_max = np.nanmin(t_plot), np.nanmax(t_plot)
            if t_max > t_min:
                t_plot = (t_plot - t_min) / (t_max - t_min) * 10.0
            else:
                t_plot = np.zeros_like(t_plot)

            abs_err = np.abs(dlc_angle - imu_angle)
            return t_plot, dlc_angle, imu_angle, abs_err

        def _plot_example_timeseries(ax, participant, speed, title_prefix):
            res_ex = _get_example_trial(participant, speed)
            if res_ex is None:
                ax.text(0.5, 0.5, f"No data for {participant}, {speed}",
                        ha="center", va="center", fontsize=10)
                _style_ax(ax)
                return

            t_plot, dlc_angle, imu_angle, _ = res_ex

            sns.lineplot(x=t_plot, y=dlc_angle, ax=ax, label="DLC wrist", linewidth=2)
            sns.lineplot(x=t_plot, y=imu_angle, ax=ax, label="IMU", linewidth=2)

            ax.set_xlabel("Time (s)", fontsize=label_fontsize, fontweight=label_weight)
            ax.set_ylabel("Angle (deg)", fontsize=label_fontsize, fontweight=label_weight)
            ax.set_xlim(0.0, 10.0)
            ax.set_xticks(np.arange(0, 10.01, 2.0))

            ax.set_title(
                f"{title_prefix}: {participant} {speed}, trial {example_trial_idx + 1}",
                fontsize=title_fontsize,
                fontweight=title_weight,
            )
            _style_ax(ax)

            ax.legend(
                fontsize=tick_fontsize,
                frameon=False,
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
            )

        def _plot_example_box(ax, participant, speed, title_prefix):
            res_ex = _get_example_trial(participant, speed)
            if res_ex is None:
                ax.text(0.5, 0.5, f"No data for {participant}, {speed}",
                        ha="center", va="center", fontsize=10)
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
            )
            ax.set_xlabel("", fontsize=label_fontsize, fontweight=label_weight)
            ax.set_ylabel("|Error| (deg)", fontsize=label_fontsize, fontweight=label_weight)
            ax.set_title(
                f"{title_prefix}: |DLC–IMU|",
                fontsize=title_fontsize,
                fontweight=title_weight,
            )

            ylim_ex = abs_err_ylim_examples or abs_err_ylim
            if ylim_ex is not None:
                ax.set_ylim(ylim_ex)

            _style_ax(ax)

        # -----------------------------
        # 4) Column positions (shift first two rows right)
        # -----------------------------
        if example_col_start < 0:
            example_col_start = 0
        if example_col_start > ncols - 3:
            example_col_start = ncols - 3

        ts_col = example_col_start + 1
        box_col = example_col_start + 2

        # -----------------------------
        # 5) Row 1 – slow example
        # -----------------------------
        for j in range(ncols):
            if j not in (ts_col, box_col):
                axes[0, j].axis("off")

        _plot_example_timeseries(
            axes[0, ts_col],
            participant=example1_participant,
            speed=example1_speed,
            title_prefix="Example (FIRST)",
        )

        _plot_example_box(
            axes[0, box_col],
            participant=example1_participant,
            speed=example1_speed,
            title_prefix="Slow trial",
        )

        # -----------------------------
        # 6) Row 2 – vfas example
        # -----------------------------
        for j in range(ncols):
            if j not in (ts_col, box_col):
                axes[1, j].axis("off")

        _plot_example_timeseries(
            axes[1, ts_col],
            participant=example2_participant,
            speed=example2_speed,
            title_prefix="Example (FIRST)",
        )

        _plot_example_box(
            axes[1, box_col],
            participant=example2_participant,
            speed=example2_speed,
            title_prefix="VFast trial",
        )

        # -----------------------------
        # 7) Row 3 – FIRST summary
        # -----------------------------
        df_first = df[df["set"] == "FIRST"]
        ylim_sum = abs_err_ylim_summary or abs_err_ylim

        for i, spd in enumerate(speed_order[:5]):
            ax = axes[2, i]
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
            )

            label = speed_titles.get(spd, spd) if speed_titles else spd
            ax.set_title(f"FIRST – {label}",
                         fontsize=title_fontsize,
                         fontweight=title_weight)
            ax.set_xlabel("Participant", fontsize=label_fontsize, fontweight=label_weight)

            if ylim_sum is not None:
                ax.set_ylim(ylim_sum)

            if i == 0:
                ax.set_ylabel("|Error| (deg)",
                              fontsize=label_fontsize,
                              fontweight=label_weight)
                _style_ax(ax)
            else:
                ax.set_ylabel("")
                _style_ax(ax)
                _hide_y_axis(ax)

        ax_bar_first = axes[2, ncols - 1]
        means_first = []
        labels_first = []
        for spd in speed_order:
            sub = df_first[df_first["speed"] == spd]["abs_err_deg"]
            means_first.append(sub.mean() if not sub.empty else np.nan)
            labels_first.append(speed_titles.get(spd, spd) if speed_titles else spd)

        sns.barplot(
            x=labels_first,
            y=means_first,
            ax=ax_bar_first,
            color="C0",
            errorbar=None,
        )
        ax_bar_first.set_title("FIRST – mean |error| vs speed",
                               fontsize=title_fontsize,
                               fontweight=title_weight)
        ax_bar_first.set_xlabel("Speed", fontsize=label_fontsize, fontweight=label_weight)
        ax_bar_first.set_ylabel("Mean |Error| (deg)", fontsize=label_fontsize, fontweight=label_weight)
        ax_bar_first.tick_params(axis="x", rotation=30)
        _style_ax(ax_bar_first)

        # -----------------------------
        # 8) Row 4 – SECOND summary
        # -----------------------------
        df_second = df[df["set"] == "SECOND"]

        for i, spd in enumerate(speed_order[:5]):
            ax = axes[3, i]
            sub = df_second[df_second["speed"] == spd]
            if sub.empty:
                ax.axis("off")
                continue

            sns.boxplot(
                data=sub,
                x="participant",
                y="abs_err_deg",
                ax=ax,
                showfliers=False,
            )

            label = speed_titles.get(spd, spd) if speed_titles else spd
            ax.set_title(f"SECOND – {label}",
                         fontsize=title_fontsize,
                         fontweight=title_weight)
            ax.set_xlabel("Participant", fontsize=label_fontsize, fontweight=label_weight)

            if ylim_sum is not None:
                ax.set_ylim(ylim_sum)

            if i == 0:
                ax.set_ylabel("|Error| (deg)",
                              fontsize=label_fontsize,
                              fontweight=label_weight)
                _style_ax(ax)
            else:
                ax.set_ylabel("")
                _style_ax(ax)
                _hide_y_axis(ax)

        ax_bar_second = axes[3, ncols - 1]
        means_second = []
        labels_second = []
        for spd in speed_order:
            sub = df_second[df_second["speed"] == spd]["abs_err_deg"]
            means_second.append(sub.mean() if not sub.empty else np.nan)
            labels_second.append(speed_titles.get(spd, spd) if speed_titles else spd)

        sns.barplot(
            x=labels_second,
            y=means_second,
            ax=ax_bar_second,
            color="C0",
            errorbar=None,
        )
        ax_bar_second.set_title("SECOND – mean |error| vs speed",
                                fontsize=title_fontsize,
                                fontweight=title_weight)
        ax_bar_second.set_xlabel("Speed", fontsize=label_fontsize, fontweight=label_weight)
        ax_bar_second.set_ylabel("Mean |Error| (deg)", fontsize=label_fontsize, fontweight=label_weight)
        ax_bar_second.tick_params(axis="x", rotation=30)
        _style_ax(ax_bar_second)

        # -----------------------------
        # 9) Layout + base spacing
        # -----------------------------
        fig.tight_layout()
        fig.subplots_adjust(wspace=boxplot_wspace, hspace=1.0)

        # -----------------------------
        # 10) Extra gap scaling for first two rows (TS vs boxplot)
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
        # 11) Force bar plots to be bar_gap_from_afap away from AFAP boxplots
        # -----------------------------
        # AFAP boxplot is in column ncols-2 (index 4), bar plots in ncols-1 (index 5)
        for row_idx in (2, 3):
            afap_ax = axes[row_idx, ncols - 2]
            bar_ax = axes[row_idx, ncols - 1]

            afap_pos = afap_ax.get_position()
            bar_pos = bar_ax.get_position()

            afap_right = afap_pos.x0 + afap_pos.width
            current_gap = bar_pos.x0 - afap_right
            # shift so gap becomes exactly bar_gap_from_afap
            delta = (afap_right + bar_gap_from_afap) - bar_pos.x0

            new_x0 = bar_pos.x0 + delta
            bar_ax.set_position([
                new_x0,
                bar_pos.y0,
                bar_pos.width,
                bar_pos.height,
            ])

        return fig, axes







