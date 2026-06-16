# Color + marker utilities. Scales to more participants if needed.

BASE_COLORS = [
    "#AED5F3", "#BDAFF3", "#FF99C5", "#FBCC9E", "#66a61e",
    "#e6ab02", "#a6761d", "#666666", "#1f78b4", "#b2df8a",
]

APP_MARKERS = ["o", "s", "^"]  # fixed

def participant_color_map(participants):
    """Return dict participant -> color (deterministic, sorted by ID)."""
    parts = sorted(participants)
    return {p: BASE_COLORS[i % len(BASE_COLORS)] for i, p in enumerate(parts)}

def app_marker_map(app_indices):
    """Return dict app_index -> marker. Works with [1,2,3] or [0,1,2]."""
    uniq = sorted(set(app_indices))
    return {idx: APP_MARKERS[i % len(APP_MARKERS)] for i, idx in enumerate(uniq)}
