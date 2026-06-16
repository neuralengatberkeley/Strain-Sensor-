# Simple time conversion helpers used across plotting + stats.

def mmss_to_seconds(mmss):
    """Convert 'M:SS' to total seconds (int). Example: '8:06' -> 486"""
    m, s = mmss.split(":")
    return int(m) * 60 + int(s)

def sec_to_mmss(sec):
    """Format seconds as 'M:SS' (unsigned)."""
    sec_i = int(round(abs(sec)))
    m, s = divmod(sec_i, 60)
    return f"{m}:{s:02d}"

def sec_to_mmss_signed(sec):
    """Format possibly-negative seconds as '±M:SS' (useful for TT-DT differences)."""
    sign = "-" if sec < 0 else ""
    sec_i = int(round(abs(sec)))
    m, s = divmod(sec_i, 60)
    return f"{sign}{m}:{s:02d}"
