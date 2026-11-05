"""Slot-by-slot agreement between the XML team and the half occupied at kickoff.

Uses already-oriented tracking (Home attacking toward increasing x), so at kickoff Home
should be at x < midfield and Away at x > midfield, in both halves.
Output: counts and proportions only.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from preprocessing.params import PITCH_L
from preprocessing.prep_tracking import _kickoff_frames

WINDOW = 25        # frames on each side of kickoff (~1 s each side)
MARGIN = 3.0       # ignore slots within 3 m of the halfway line
MID = PITCH_L / 2

from games import DATA_DIR  # noqa: E402  (honours MTAH_DATA_DIR, else ./data)

PROCESSED = Path(DATA_DIR) / "processed"
data = pd.read_parquet(PROCESSED / "tracking_clean.parquet")
home = pd.read_parquet(PROCESSED / "slot_teams.parquet").is_home.to_numpy()
kickoffs = _kickoff_frames()  # kickoffs.json if present; otherwise the published fallback

print("slots Home no XML:", int(home.sum()), "| Away:", int((~home).sum()))

for half in (1, 2):
    k = kickoffs[half]
    w = data[(data.video_frame >= k - WINDOW) & (data.video_frame <= k + WINDOW)]
    x = np.array([w[f"s{i:02d}_x"].median() for i in range(1, len(home) + 1)])

    observed = np.isfinite(x)
    usable = observed & (np.abs(x - MID) > MARGIN)
    in_low_half = x < MID                      # lower-x half
    agree = in_low_half == home                # Home should be on the lower-x side

    n = int(usable.sum())
    print(f"\ntempo {half}")
    print("  frames na janela:", len(w))
    print("  slots observados:", int(observed.sum()), "| avaliáveis (fora da margem):", n)
    if n:
        print("  concordância geral:", round(float(agree[usable].mean()), 3))
        for name, mask in (("Home", home), ("Away", ~home)):
            m = usable & mask
            if m.sum():
                print(f"  concordância {name}: {round(float(agree[m].mean()), 3)} (n={int(m.sum())})")