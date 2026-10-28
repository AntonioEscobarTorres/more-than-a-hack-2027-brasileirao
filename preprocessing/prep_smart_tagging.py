"""
Smart Tagging Reader: XML SportsCode -> Event DataFrame by video frame.

Output of load_tagging_events() (one row per XML instance, deterministically sorted):

    event_id     0..n-1 after sorting (for traceability)
    start_frame  starting video frame = round(seconds * 25)
    end_frame    ending video frame   = round(seconds * 25) (EXCLUSIVE)
    code         uppercase code, stripped (e.g., 'SET PIECES')
    team         'H' (Barcelona), 'A' (Atletic Club) or 'U' (N/A or missing)
    team_raw     original Team tag text ('Barcelona', 'Atletic Club', 'N/A', '')
    type         lowercase Type tag ('throw-in', 'kick off (start)', ...; '' if missing)
    half         1 or 2 (Half tag); 0 if missing

Conventions:
  * XML time is already based on the tactical video: DO NOT add any offset.
  * Half-open interval: end_frame is the first frame AFTER the phase. 
    Use arr[start_frame:end_frame] in frame arrays.
  * Rounding: int(round(seconds * fps)), halves rounded to the nearest even number.
  * Text comparisons use normalized lowercase/uppercase strings.
  * 'U' is used for unknown teams ('NA' or 'N/A' are parsed as NaN by pandas read_csv).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

FPS = 25                 
N_FRAMES = 199_319
GAME = "BAR-ATH"

# ---- Expected counts from DATA.md (used for data parsing validation)
EXPECTED_TOTAL = 638
EXPECTED_CODES = {
    "PLAYERS IN THE BOX": 97, "BALL IN FINAL THIRD": 93, "BALL IN THE BOX": 72,
    "PROGRESSION": 66, "DEFENDING IN MIDDLE THIRD": 66, "SET PIECES": 59,
    "CREATING CHANCES": 30, "DEFENDING IN DEFENSIVE THIRD": 30, "LONG BALL": 29,
    "ATTACKING TRANSITION": 24, "DEFENSIVE TRANSITION": 24, "BUILD UP": 22,
    "DEFENDING IN ATTACKING THIRD": 22, "GOALS": 4,
}
EXPECTED_TEAMS = {"Barcelona": 314, "Atletic Club": 288, "N/A": 36}   # 'Atletic': single 't' intentional
EXPECTED_SP_TYPES = {
    "throw-in": 26, "corner kick": 12, "goal kick": 10, "free kick": 5,
    "kick off (after goal)": 4, "kick off (start)": 2,
}

TEAM_MAP = {"barcelona": "H", "atletic club": "A"}       # Any other value -> 'U'
HALF_MAP = {"1st half": 1, "2nd half": 2}                # Half tag is a string, not numeric
REQUIRED = ("timestamp", "end_timestamp", "code")
EVENT_COLUMNS = ["event_id", "start_frame", "end_frame", "code", "team", "team_raw", "type", "half"]


def _text(value):
    """Cleans text; NaN/None become empty strings."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value).strip()


def _build(code, start_s, end_s, team_raw, type_raw, half_raw, fps, n_frames):
    """
    Core builder for both readers. 
    Receives seconds (float) and aligned Series (index 0..n-1).

    Note: Seconds are processed directly without pd.to_timedelta. pd.to_timedelta(float_list) 
    has a known bug in some pandas versions where if the first value is an integer, 
    the rest are evaluated as nanoseconds (e.g., 10.0 becomes 10 ns).
    """
    start = np.rint(np.asarray(start_s, dtype=float) * fps).astype(int)
    end = np.rint(np.asarray(end_s, dtype=float) * fps).astype(int)
    
    if (end < start).any():
        raise ValueError(f"{int((end < start).sum())} events have end times before start times.")
        
    out = pd.DataFrame({
        "start_frame": np.clip(start, 0, n_frames - 1),
        "end_frame": np.clip(end, 0, n_frames),               # Exclusive: can equal n_frames
        "code": code.map(lambda v: _text(v).upper()),
        "team": team_raw.str.lower().map(TEAM_MAP).fillna("U"),
        "team_raw": team_raw,
        "type": type_raw.str.lower(),
        "half": half_raw.str.lower().map(HALF_MAP).fillna(0).astype(int),
    })
    
    # Deterministic sorting
    out = out.sort_values(["start_frame", "end_frame", "code", "team_raw", "type"],
                          kind="mergesort").reset_index(drop=True) 
                          
    for col in ("code", "team", "team_raw", "type"):
        out[col] = out[col].astype(object)
        
    out.insert(0, "event_id", np.arange(len(out)))
    return out[EVENT_COLUMNS]


def events_from_frame(raw, fps=FPS, n_frames=N_FRAMES):
    """
    Converts raw kloppy DataFrame (sportscode.load(...).to_df()) to an Event DataFrame.
    Independent of file structures, allowing for synthetic data testing.
    """
    lower = {str(c).lower(): c for c in raw.columns}          # Tolerates Team/team, Type/type
    missing = [c for c in REQUIRED if c not in lower]
    
    if missing:
        raise ValueError(f"Missing columns: {missing}; DataFrame contains: {list(raw.columns)}")

    def tag(name):
        if name in lower:
            return raw[lower[name]].map(_text).reset_index(drop=True)
        return pd.Series([""] * len(raw))

    return _build(raw[lower["code"]].reset_index(drop=True),
                  raw[lower["timestamp"]].dt.total_seconds().to_numpy(),
                  raw[lower["end_timestamp"]].dt.total_seconds().to_numpy(),
                  tag("team"), tag("type"), tag("half"), fps, n_frames)


def events_from_records(records, fps=FPS, n_frames=N_FRAMES):
    """Converts output from starter.load_smart_tagging.load() (list of dicts) to an Event DataFrame."""
    def tag(name):
        return pd.Series([_text(r["tags"].get(name)) for r in records], dtype=object)

    return _build(pd.Series([r["code"] for r in records], dtype=object),
                  [r["start_s"] for r in records], [r["end_s"] for r in records],
                  tag("Team"), tag("Type"), tag("Half"), fps, n_frames)


def load_tagging_events(data_dir=None, reader="starter", fps=FPS, n_frames=N_FRAMES):
    """
    Reads Smart Tagging from the tactical video and returns the events DataFrame.

    Args:
        reader="starter": Uses standard library (reference for the challenge, no kloppy required).
        reader="kloppy": Uses kloppy.sportscode (for cross-checking).
    """
    if reader == "starter":
        from starter import load_smart_tagging as ref
        return events_from_records(ref.load(GAME, data_dir, fps), fps=fps, n_frames=n_frames)
    if reader == "kloppy":
        from kloppy import sportscode
        from games import require
        (path,) = require(["smart_tagging"], GAME, data_dir)
        return events_from_frame(sportscode.load(path).to_df(), fps=fps, n_frames=n_frames)
        
    raise ValueError(f"Reader must be 'starter' or 'kloppy', got {reader!r}")


def check_counts(ev):
    """Validates read counts against expected counts from DATA.md. Returns an empty list if matching."""
    problems = []

    def cmp(label, got, expected):
        if got != expected:
            problems.append(f"{label}: parsed {got} / expected {expected}")

    cmp("Total instances", len(ev), EXPECTED_TOTAL)
    codes = ev["code"].value_counts().to_dict()
    
    for code, exp in EXPECTED_CODES.items():
        cmp("Code " + code, codes.get(code, 0), exp)
        
    for code in sorted(set(codes) - set(EXPECTED_CODES)):
        problems.append(f"Unexpected code: {code!r} ({codes[code]} instances)")
        
    teams = ev["team_raw"].value_counts().to_dict()
    for team, exp in EXPECTED_TEAMS.items():
        cmp("Team " + team, teams.get(team, 0), exp)
        
    for team in sorted(set(teams) - set(EXPECTED_TEAMS)):
        problems.append(f"Unexpected Team value: {team!r} ({teams[team]} instances)")
        
    sp = ev[ev["code"] == "SET PIECES"]["type"].value_counts().to_dict()
    cmp("SET PIECES", int(sum(sp.values())), EXPECTED_CODES["SET PIECES"])
    
    for typ, exp in EXPECTED_SP_TYPES.items():
        cmp("Type " + typ, sp.get(typ, 0), exp)
        
    for typ in sorted(set(sp) - set(EXPECTED_SP_TYPES)):
        problems.append(f"Unexpected Type in SET PIECES: {typ!r}")
        
    return problems


def check_half_consistency(ev, blocks=None):
    """Ensures the 'Half' tag matches the video frame blocks timeframe."""
    if blocks is None:
        from games import half_video_blocks
        blocks = half_video_blocks(GAME)
        
    problems = []
    for half, first, last in blocks:
        inside = ev[(ev["start_frame"] >= first) & (ev["start_frame"] <= last)]
        bad = int((inside["half"] != half).sum())
        if bad:
            problems.append(f"{bad} events in time block {half} have a mismatched 'Half' tag")
            
    return problems


def compare_events(a, b):
    """Compares two Event DataFrames. Returns aggregated metric numbers only."""
    keys = ["code", "team_raw", "type", "half", "start_frame", "end_frame"]
    a = a.sort_values(keys, kind="mergesort").reset_index(drop=True)
    b = b.sort_values(keys, kind="mergesort").reset_index(drop=True)
    
    report = {"rows": (len(a), len(b))}
    if len(a) != len(b):
        return report
        
    for col in ("start_frame", "end_frame"):
        diff = (b[col] - a[col]).to_numpy()
        report[col] = {
            "divergences": int((diff != 0).sum()),
            "distinct_differences": sorted(set(int(d) for d in diff))
        }
        
    for col in ("code", "team_raw", "type", "half"):
        report[col] = int((a[col] != b[col]).sum())
        
    return report


def cross_check(data_dir=None):
    """starter vs kloppy: asserts that both readers output the same events."""
    report = compare_events(
        load_tagging_events(data_dir, reader="starter"),
        load_tagging_events(data_dir, reader="kloppy")
    )
    
    print("Rows (starter, kloppy):", report["rows"])
    
    for col in ("start_frame", "end_frame"):
        if col in report:
            print(f"{col}: divergences {report[col]['divergences']} | distinct differences {report[col]['distinct_differences']}")
            
    for col in ("code", "team_raw", "type", "half"):
        if col in report:
            print(f"{col}: divergences {report[col]}")
            
    ok = (report["rows"][0] == report["rows"][1]
          and all(report[c]["divergences"] == 0 for c in ("start_frame", "end_frame"))
          and all(report[c] == 0 for c in ("code", "team_raw", "type", "half")))
          
    print("OK: Both readers agree" if ok else "WARNING: Readers diverge")
    return 0 if ok else 1


def main(argv=None):
    import sys
    argv = sys.argv[1:] if argv is None else argv
    
    if "--cross-check" in argv:
        return cross_check()
        
    ev = load_tagging_events()
    problems = check_counts(ev) + check_half_consistency(ev)
    
    print(f"{len(ev)} events read successfully.")
    print("OK: Counts and times match DATA.md specs." if not problems else "\n".join(problems))
    
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())