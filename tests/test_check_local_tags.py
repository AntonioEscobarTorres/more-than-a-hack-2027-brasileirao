"""
Unit tests for the tagging verification script.
Uses synthetic data to validate the analysis logic without real data leakage.
"""
import pandas as pd
from preprocessing import check_local_tags as clt
from preprocessing import params

def test_run_checks_synthetic_data(capsys):
    ev = pd.DataFrame([
        # overlapping phases
        {"start_frame": 10, "end_frame": 50, "code": "BUILD UP", "team": "H", "team_raw": "Barcelona", "type": "", "half": 1},
        {"start_frame": 30, "end_frame": 70, "code": "DEFENSIVE TRANSITION", "team": "H", "team_raw": "Barcelona", "type": "", "half": 1},
        
        # set pieces
        {"start_frame": 100, "end_frame": 150, "code": "SET PIECES", "team": "U", "team_raw": "N/A", "type": "throw-in", "half": 1},
        {"start_frame": 200, "end_frame": 250, "code": "SET PIECES", "team": "A", "team_raw": "Athletic Club", "type": "corner kick", "half": 1},
        
        # goals
        {"start_frame": 300, "end_frame": 320, "code": "GOALS", "team": "H", "team_raw": "Barcelona", "type": "", "half": 1},
        {"start_frame": 500, "end_frame": 550, "code": "SET PIECES", "team": "A", "team_raw": "Athletic Club", "type": "kick off (after goal)", "half": 1}
    ])
    
    clt.run_checks(ev, n_frames=1000)
    
    captured = capsys.readouterr()
    output = captured.out
    
    assert "tagging local checks" in output
    assert "Maximum active phases at the same time:" in output
    assert "Goals tracking:" in output
    assert "Barcelona" in output
    assert "Athletic Club" in output