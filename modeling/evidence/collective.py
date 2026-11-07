"""Collective evidence for ball gaps."""
from __future__ import annotations
import numpy as np
import pandas as pd

def _centered_diff(s: pd.Series, half: pd.Series, window: int = 25, fps: float = 25.0) -> pd.Series:
    dt = (window * 2) / fps
    forward = s.groupby(half).shift(-window)
    backward = s.groupby(half).shift(window)
    return (forward - backward) / dt

def build_collective_score(master: pd.DataFrame) -> pd.Series:
    required = ["centroid_home_x", "centroid_away_x", "spread_home_x", 
                "spread_away_x", "spread_home_y", "spread_away_y", 
                "mean_speed_home", "mean_speed_away"]
    
    # Se for um teste antigo sem as colunas, devolve 0
    if not all(c in master.columns for c in required):
        return pd.Series(0.0, index=master.index, dtype=np.float32)
        
    half = master.get("half", pd.Series(1, index=master.index))
    
    adv_home = _centered_diff(master["centroid_home_x"], half)
    adv_away = -_centered_diff(master["centroid_away_x"], half)
    
    rel_adv = adv_home - adv_away
    rel_spread_x = master["spread_home_x"] - master["spread_away_x"]
    rel_spread_y = master["spread_home_y"] - master["spread_away_y"]
    rel_speed = master["mean_speed_home"] - master["mean_speed_away"]
    
    coll_score = rel_adv + rel_spread_x + rel_spread_y + rel_speed
    coll_score = coll_score.fillna(0.0)
    coll_score = np.where(half == 0, 0.0, coll_score)
    
    return pd.Series(coll_score, index=master.index, dtype=np.float32)