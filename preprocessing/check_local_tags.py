"""
Local verification script for tagging rules.
Analyzes overlaps, coexistence, set pieces, goals, and transitions.
Prints conclusions to the standard output.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from preprocessing import params
from preprocessing.prep_smart_tagging import load_tagging_events
from preprocessing.prep_phases import OFFENSIVE_CODES, DEFENSIVE_CODES

def run_checks(ev: pd.DataFrame, n_frames: int = params.N_FRAMES) -> None:
    """
    Executes all local sanity checks on the event dataframe and prints the results.
    """
    print("\n--- tagging local checks ---")
    
    # 1. Phase overlap
    h_off = np.zeros(n_frames, dtype=int)
    h_def = np.zeros(n_frames, dtype=int)
    a_off = np.zeros(n_frames, dtype=int)
    a_def = np.zeros(n_frames, dtype=int)
    total_phases = np.zeros(n_frames, dtype=int)
    
    for _, r in ev.iterrows():
        code = str(r["code"]).strip().upper()
        team = str(r["team"]).strip().upper()
        a, b = int(r["start_frame"]), int(r["end_frame"])
        
        if code == "SET PIECES" or team not in ("H", "A"):
            continue
            
        total_phases[a:b] += 1
        
        if code in OFFENSIVE_CODES:
            if team == "H":
                h_off[a:b] += 1
            else:
                a_off[a:b] += 1
        elif code in DEFENSIVE_CODES:
            if team == "H":
                h_def[a:b] += 1
            else:
                a_def[a:b] += 1

    max_overlap = np.max(total_phases)
    avg_overlap = np.mean(total_phases[total_phases > 0])
    print(f"\n1. Phase overlap:")
    print(f"   Maximum active phases at the same time: {max_overlap}")
    print(f"   Average overlapping phases (when active): {avg_overlap:.2f}")

    # 2. Coexistence of offensive and defensive phases for the same team
    h_coexist = np.sum((h_off > 0) & (h_def > 0))
    a_coexist = np.sum((a_off > 0) & (a_def > 0))
    print(f"\n2. Coexistence of offensive and defensive phases for the same team:")
    print(f"   Home team (H) simultaneous off/def frames: {h_coexist}")
    print(f"   Away team (A) simultaneous off/def frames: {a_coexist}")

    # 3. Set pieces overview
    sp = ev[ev["code"] == "SET PIECES"]
    team_counts = sp["team"].value_counts().to_dict()
    print(f"\n3. Set pieces summary:")
    print(f"   Team assignment counts: {team_counts}")
    print("   First 3 throw-ins for visual video check (ball out vs kick):")
    throw_ins = sp[sp["type"] == "throw-in"].head(3)
    for _, r in throw_ins.iterrows():
        length_s = (r["end_frame"] - r["start_frame"]) / params.FPS
        print(f"   - Frames [{r['start_frame']} : {r['end_frame']}) | Duration: {length_s:.1f}s")

    # 4. Goals vs kick off (after goal)
    goals = ev[ev["code"] == "GOALS"].sort_values("start_frame")
    ko = ev[ev["type"] == "kick off (after goal)"].sort_values("start_frame")
    print(f"\n4. Goals tracking:")
    print(f"   Found {len(goals)} GOALS tags and {len(ko)} kick off (after goal) tags.")
    for i, (g_idx, g_row) in enumerate(goals.iterrows()):
        g_start, g_end = g_row["start_frame"], g_row["end_frame"]
        ko_start = ko.iloc[i]["start_frame"] if i < len(ko) else "N/A"
        gap = (ko_start - g_end) / params.FPS if ko_start != "N/A" else 0
        print(f"   Goal {i+1}: Tag [{g_start} : {g_end}) -> Kickoff starts at {ko_start}. Dead ball gap: {gap:.1f}s")

    # 5. Exact team values and defensive transition polarity
    exact_teams = ev["team_raw"].unique()
    print(f"\n5. Team values and defensive transitions:")
    print(f"   Exact 'Team' raw strings found: {list(exact_teams)}")
    
    def_trans = ev[ev["code"] == "DEFENSIVE TRANSITION"]
    polarity_matches = 0
    for _, r in def_trans.iterrows():
        a, b = r["start_frame"], r["end_frame"]
        team = r["team"]
        opponent = "A" if team == "H" else "H"
        # check if opponent has an offensive phase at the exact same time
        opp_off_active = a_off[a:b] if opponent == "A" else h_off[a:b]
        if np.any(opp_off_active > 0):
            polarity_matches += 1
            
    print(f"   Total defensive transitions: {len(def_trans)}")
    print(f"   Times the opponent was simultaneously in an offensive phase: {polarity_matches}")
    print("   (If matches are high, defensive transition implies the tagged team lost the ball).")
    print("\n---------------------------\n")

if __name__ == "__main__":
    events = load_tagging_events()
    run_checks(events)