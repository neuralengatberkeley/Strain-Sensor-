# Load CSVs and attach computed columns used in plots/stats.

import pandas as pd
from .utils import mmss_to_seconds

def load_app_times(path):
    """Load APP times.
    expected columns:
        participant, method, app_label, app_index, time_str
    adds:
        time_sec
    """
    df = pd.read_csv(path)
    df = df.copy()
    df["time_sec"] = df["time_str"].apply(mmss_to_seconds)
    return df

def load_rem_times(path):
    """Load REM times.
    expected columns:
        participant, method, rem_label, rem_index, time_str
    adds:
        time_sec
    """
    df = pd.read_csv(path)
    df = df.copy()
    df["time_sec"] = df["time_str"].apply(mmss_to_seconds)
    return df

def load_ratings(path):
    """Load survey ratings (1-10).
    expected columns:
        participant, method, app_label, app_index, metric, rating
    """
    df = pd.read_csv(path)
    return df.copy()
