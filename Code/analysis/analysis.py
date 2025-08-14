import pandas as pd
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

import numpy as np
from sklearn.model_selection import KFold, train_test_split
from config import path_to_repository
from scipy.interpolate import interp1d

from typing import Dict, Tuple, Optional, List
from scipy.io import loadmat


class DLC3DBendAngles:
    """
    DLC 3D (MultiIndex header) bend-angle calculator.

    angle_type:
      - 'mcp'   : angle(hand→MCP, MCP→PIP)
      - 'wrist' : angle(forearm→hand, hand→MCP)
    """

    def __init__(self, csv_paths, bodyparts: Optional[Dict[str, str]] = None):
        """
        Initialize from one or two DLC 3D CSV files.
        """
        # Store for reference
        self.csv_path = csv_paths
        self._match_map: Optional[pd.DataFrame] = None  # holds ['cam_index','enc_index','time_delta']

        # Handle single CSV
        if isinstance(csv_paths, str):
            df = pd.read_csv(csv_paths, header=[0, 1, 2])
            if not isinstance(df.columns, pd.MultiIndex):
                raise ValueError("Expected DLC 3D CSV with a 3-row MultiIndex header.")
            self.df = df

        # Handle two CSVs
        elif isinstance(csv_paths, (list, tuple)) and len(csv_paths) == 2:
            dfs = []
            for path in csv_paths:
                df = pd.read_csv(path, header=[0, 1, 2])
                if not isinstance(df.columns, pd.MultiIndex):
                    raise ValueError(f"{path} does not have a 3-row MultiIndex header.")
                dfs.append(df)
            if len(dfs[0]) != len(dfs[1]):
                raise ValueError(f"Row count mismatch: {len(dfs[0])} vs {len(dfs[1])}")
            self.df = pd.concat(dfs, axis=1)
        else:
            raise ValueError("csv_paths must be a string or a list/tuple of two CSV paths.")

        # Default bodyparts mapping
        self.bp = {
            "forearm": "forearm",
            "hand": "hand",
            "MCP": "MCP",
            "PIP": "PIP",
        }
        if bodyparts:
            self.bp.update(bodyparts)

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

    @staticmethod
    def _resolve_col_key(df: pd.DataFrame, col_spec):
        """
        Resolve a column given:
          - exact name present in df.columns
          - a tuple (true MultiIndex key)
          - a substring (returns first match)
        """
        # exact (including tuple)
        if col_spec in df.columns:
            return col_spec
        # substring search on stringified column names
        candidates = [c for c in df.columns if str(col_spec) in str(c)]
        if len(candidates) == 0:
            raise KeyError(f"Column matching '{col_spec}' not found.")
        # pick first match (you can tighten this if needed)
        return candidates[0]

    @staticmethod
    def _ensure_float_series(s: pd.Series) -> pd.Series:
        """Coerce to float64 for numeric asof matching."""
        if not np.issubdtype(s.dtype, np.number):
            s = pd.to_numeric(s, errors="coerce")
        return s.astype("float64")

    # ---------------- MAT loader ----------------
    def load_mat_as_df(self, mat_path: str, prefix: str = None) -> pd.DataFrame:
        from scipy.io import loadmat
        mat_data = loadmat(mat_path)
        keys = [k for k in mat_data.keys() if not k.startswith("__")]
        if prefix:
            keys = [k for k in keys if k.startswith(prefix)]

        dfs = []
        for k in keys:
            val = mat_data[k]
            if isinstance(val, np.ndarray):
                if val.ndim == 1:
                    df_k = pd.DataFrame({k: val})
                elif val.ndim == 2:
                    col_names = [f"{k}_{i}" for i in range(val.shape[1])]
                    df_k = pd.DataFrame(val, columns=col_names)
                else:
                    flat = val.reshape(val.shape[0], -1)
                    col_names = [f"{k}_{i}" for i in range(flat.shape[1])]
                    df_k = pd.DataFrame(flat, columns=col_names)
                dfs.append(df_k)
            else:
                print(f"Skipping non-array variable: {k}")

        if dfs:
            return pd.concat(dfs, axis=1)
        else:
            raise ValueError(f"No array variables found in {mat_path} matching prefix='{prefix}'")

    @staticmethod
    def compare_row_counts(df1: pd.DataFrame, df2: pd.DataFrame) -> Tuple[int, int]:
        rows_df1 = len(df1)
        rows_df2 = len(df2)
        print(f"DataFrame 1: {rows_df1} rows")
        print(f"DataFrame 2: {rows_df2} rows")
        return rows_df1, rows_df2


    def add_dataframe(self, other_df: pd.DataFrame, inplace: bool = True) -> pd.DataFrame:
        if len(other_df) != len(self.df):
            raise ValueError(
                f"Row count mismatch: self.df has {len(self.df)} rows, other_df has {len(other_df)} rows")
        if inplace:
            for col in other_df.columns:
                self.df[col] = other_df[col].values
            return self.df
        else:
            return pd.concat([self.df, other_df], axis=1)

    @staticmethod
    def _series_time_of_day_to_timedelta(s: pd.Series) -> pd.Series:
        """
        Convert various time formats to Timedelta since midnight.
        Special handling for HHMMSSffffff strings/ints from strftime("%H%M%S%f").
        """
        s = pd.Series(s, copy=False)

        # Detect HHMMSSffffff pattern (int or str)
        def parse_hhmmssfff(val):
            if pd.isna(val):
                return pd.NaT
            val_str = str(int(val)).zfill(12)  # ensure zero-padded
            try:
                t = datetime.strptime(val_str, "%H%M%S%f").time()
                return (pd.to_timedelta(t.hour, unit="h") +
                        pd.to_timedelta(t.minute, unit="m") +
                        pd.to_timedelta(t.second, unit="s") +
                        pd.to_timedelta(t.microsecond, unit="us"))
            except Exception:
                return pd.NaT

        if np.issubdtype(s.dtype, np.number) or s.dtype == object:
            # Check if all values look like HHMMSSffffff
            if s.dropna().astype(str).str.fullmatch(r"\d{9,12}").all():
                return s.apply(parse_hhmmssfff)

        # Fallback to pandas parser
        try:
            return pd.to_timedelta(s, errors="coerce")
        except Exception:
            return pd.Series(pd.NaT, index=s.index)

    @staticmethod
    def _coerce_tolerance_to_timedelta(tolerance) -> pd.Timedelta:
        """
        Tolerance interpretation:
          - int   -> microseconds
          - float -> seconds
          - str   -> parsed by pandas ('10ms', '500us', '1s', etc.)
        """
        if isinstance(tolerance, (np.integer, int)):
            return pd.to_timedelta(int(tolerance), unit="us")
        if isinstance(tolerance, (np.floating, float)):
            return pd.to_timedelta(float(tolerance), unit="s")
        return pd.to_timedelta(tolerance)

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
        Accepts µs integers, datetimes, or strings; stores time_delta in **milliseconds (float)**.
        NOTE: int tolerance = microseconds, float = seconds, str via pandas ('10ms','500us',...).
        """
        if not isinstance(self.df, pd.DataFrame):
            raise ValueError("self.df must be a pandas DataFrame.")

        cam_col = self._resolve_col_key(self.df, cam_time_col)
        enc_col = self._resolve_col_key(encoder_df, enc_time_col)

        # Convert to Timedelta since midnight / start
        cam_td = self._series_time_of_day_to_timedelta(self.df[cam_col]).rename("t_cam_td")
        enc_td = self._series_time_of_day_to_timedelta(encoder_df[enc_col]).rename("t_enc_td")

        cam_small = pd.DataFrame({"t_cam_td": cam_td, "cam_index": self.df.index})
        enc_small = pd.DataFrame({"t_enc_td": enc_td, "enc_index": encoder_df.index})

        # Debug counts
        cam_nan_count = cam_small["t_cam_td"].isna().sum()
        enc_nan_count = enc_small["t_enc_td"].isna().sum()
        print(f"[find_matching_indices] Dropping {cam_nan_count} camera rows with NaT timestamps.")
        print(f"[find_matching_indices] Dropping {enc_nan_count} encoder rows with NaT timestamps.")

        cam_small = cam_small.dropna(subset=["t_cam_td"]).sort_values("t_cam_td")
        enc_small = enc_small.dropna(subset=["t_enc_td"]).sort_values("t_enc_td")

        if cam_small.empty or enc_small.empty:
            self._match_map = pd.DataFrame(columns=["cam_index", "enc_index", "time_delta"])
            return self._match_map

        # Merge on int64 nanoseconds for speed/safety
        cam_small["_t_cam_ns"] = cam_small["t_cam_td"].view("i8")
        enc_small["_t_enc_ns"] = enc_small["t_enc_td"].view("i8")

        # ===== DEBUG: ranges and rough gap stats =====
        print("[debug] cam range:", cam_small["t_cam_td"].min(), "→", cam_small["t_cam_td"].max())
        print("[debug] enc range:", enc_small["t_enc_td"].min(), "→", enc_small["t_enc_td"].max())

        probe = pd.merge_asof(
            cam_small[["t_cam_td", "_t_cam_ns"]].iloc[::max(1, len(cam_small) // 20)].sort_values("_t_cam_ns"),
            enc_small[["t_enc_td", "_t_enc_ns"]].sort_values("_t_enc_ns"),
            left_on="_t_cam_ns",
            right_on="_t_enc_ns",
            direction="nearest"
        )
        probe["delta"] = pd.to_timedelta(probe["_t_enc_ns"] - probe["_t_cam_ns"], unit="ns")
        probe["delta_ms"] = probe["delta"].dt.total_seconds() * 1000.0
        print("[debug] probe |delta| (ms) stats:", probe["delta_ms"].abs().describe())

        # =================================================

        merged = pd.merge_asof(
            cam_small,
            enc_small,
            left_on="_t_cam_ns",
            right_on="_t_enc_ns",
            direction=direction,
            allow_exact_matches=True,
        )

        # Timedelta difference (ns → Timedelta → ms)
        merged["time_delta_td"] = pd.to_timedelta(merged["_t_enc_ns"] - merged["_t_cam_ns"], unit="ns")
        merged["time_delta_ms"] = merged["time_delta_td"].dt.total_seconds() * 1000.0

        # Tolerance check still in Timedelta space
        tol_td = self._coerce_tolerance_to_timedelta(tolerance)
        keep = merged["time_delta_td"].abs() <= tol_td

        # Output with time_delta as milliseconds (numeric)
        out = (merged.loc[keep, ["cam_index", "enc_index", "time_delta_ms"]]
               .rename(columns={"time_delta_ms": "time_delta"})
               .drop_duplicates(subset=["cam_index"], keep="first")
               .reset_index(drop=True))

        self._match_map = out
        return out

    def attach_encoder_using_match(
            self,
            encoder_df: pd.DataFrame,
            columns: Optional[List] = None,
            suffix: str = "_renc",
            keep_time_delta: bool = True,
            drop_unmatched: bool = True,
    ) -> pd.DataFrame:
        """
        Use self._match_map to (a) optionally drop unmatched rows in self.df
        and (b) attach chosen encoder columns (flattened + suffixed).
        Note: 'time_delta' is a Timedelta. Set drop_unmatched=False to preserve all rows.
        """
        if self._match_map is None or self._match_map.empty:
            raise RuntimeError("No matches stored. Run find_matching_indices(...) first.")

        m = self._match_map
        cam_to_enc = dict(zip(m["cam_index"].tolist(), m["enc_index"].tolist()))

        if drop_unmatched:
            keep_idx = m["cam_index"].unique()
            self.df = self.df.loc[keep_idx]
        self.df = self.df.sort_index()

        # Choose encoder columns
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

    # ---------------- One-shot aligner (kept, now using helpers) ----------------
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

        left = left.dropna(subset=["_t"]).sort_values("_t").reset_index(drop=False)  # keep original idx
        right = right.dropna(subset=["_t_enc"]).sort_values("_t_enc").reset_index(drop=False)

        # Suffix encoder columns (skip key)
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

        if drop_unmatched:
            matched = merged[merged["_t_enc"].notna()].copy()
        else:
            matched = merged

        if verbose:
            print(f"[align_and_attach_encoder] camera rows in: {len(left)}, kept after match: {len(matched)}")

        # Restore original index for rows we kept
        matched = matched.set_index("index").sort_index()
        if inplace:
            self.df = matched
            return self.df
        return matched


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

    def get_points(self, bodypart_key: str) -> np.ndarray:
        """Return Nx3 array for the bodypart."""
        xyz_cols = self.get_xyz(bodypart_key)
        return self.df[list(xyz_cols)].to_numpy(dtype=float)

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

    def compute_vectors(self, angle_type: str = "mcp") -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (v1, v2) for the requested angle.
          - 'mcp'   : v1 = hand→MCP, v2 = MCP→PIP
          - 'wrist' : v1 = forearm→hand, v2 = hand→MCP
        """
        angle_type = angle_type.lower()
        hand = self.get_points("hand")
        mcp  = self.get_points("MCP")

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

    # ---------- Output to DataFrame ----------
    def add_angle_column(
        self,
        angle_type: str = "mcp",
        out_key: Optional[Tuple[str, str, str]] = None,
        inplace: bool = False
    ) -> pd.DataFrame:
        """
        Append angle column to a DataFrame copy (or in place if inplace=True).

        out_key: MultiIndex tuple for the column; default ('metric', f'{angle_type}_bend_deg', 'deg')
        """
        angles = self.compute_bend_angle(angle_type)
        target = self.df if inplace else self.df.copy()
        if out_key is None:
            out_key = ("metric", f"{angle_type}_bend_deg", "deg")
        target[out_key] = angles
        return target

    def add_all_angles(self, inplace: bool = False) -> pd.DataFrame:
        """Convenience: add both wrist and mcp angles."""
        target = self.df if inplace else self.df.copy()
        for ang in ("wrist", "mcp"):
            target[("metric", f"{ang}_bend_deg", "deg")] = self.compute_bend_angle(ang)
        return target

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

