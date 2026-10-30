"""
Calculates the number of overlapping smart tagging phases favoring each team.
- Offensive codes favor the tagged team.
- Defensive codes favor the opposing team.
- 'SET PIECES' and unknown ('U') teams are ignored.
- Overlapping phases are strictly additive.

Strict constraints applied:
- Half-open intervals: [start_frame, end_frame) without '+ 1'.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import params

# Sets defined at the top for easy review and alignment with OVERVIEW.md
OFFENSIVE_CODES = {
    "BUILD UP", "PROGRESSION", "CREATING CHANCES", "ATTACKING TRANSITION", 
    "LONG BALL", "BALL IN FINAL THIRD", "BALL IN THE BOX", "PLAYERS IN THE BOX", "GOALS"
}

DEFENSIVE_CODES = {
    "DEFENDING IN ATTACKING THIRD", "DEFENDING IN MIDDLE THIRD",
    "DEFENDING IN DEFENSIVE THIRD", "DEFENSIVE TRANSITION"
}

def build_phase_features(ev: pd.DataFrame, n_frames: int = params.N_FRAMES) -> pd.DataFrame:
    """    
    Args:
        ev: Event DataFrame from prep_smart_tagging.py
        n_frames: Total number of frames in the video/tracking data.
        
    Returns:
        DataFrame with columns 'n_tags_favor_home' and 'n_tags_favor_away' (int16).
    """
    favor_home = np.zeros(n_frames, dtype=np.int16)
    favor_away = np.zeros(n_frames, dtype=np.int16)
    
    for _, r in ev.iterrows():
        code = str(r["code"]).strip().upper()
        team = str(r["team"]).strip().upper()
        
        if code == "SET PIECES" or team not in ("H", "A"):
            continue
            
        a, b = int(r["start_frame"]), int(r["end_frame"])  # [a, b) semi-open
        
        if code in OFFENSIVE_CODES:
            if team == "H":
                favor_home[a:b] += 1
            else:
                favor_away[a:b] += 1
        elif code in DEFENSIVE_CODES:
            if team == "H":
                favor_away[a:b] += 1  # Defensive phase favors the opponent
            else:
                favor_home[a:b] += 1  # Defensive phase favors the opponent

    return pd.DataFrame({
        "n_tags_favor_home": favor_home,
        "n_tags_favor_away": favor_away
    })