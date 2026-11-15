"""Unit tests for tools/compare_external_realism.py (F4-04)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from evaluation import possession_metrics as pm
from tools import compare_external_realism as cer


def _make_master(
    n_frames: int = 300,
    *,
    half: np.ndarray | None = None,
    speed_home: float | np.ndarray = 1.5,
    speed_away: float | np.ndarray = 1.5,
    ball_speed: float | np.ndarray = 4.0,
    ball_observed: np.ndarray | None = None,
    ball_x: np.ndarray | None = None,
    ball_y: np.ndarray | None = None,
) -> pd.DataFrame:
    """Create a minimal synthetic master DataFrame with the columns used by realism checks."""
    if half is None:
        half = np.ones(n_frames, dtype=np.int64)
    if ball_observed is None:
        ball_observed = np.ones(n_frames, dtype=np.int64)
    if ball_x is None:
        ball_x = np.linspace(0.0, 60.0, n_frames, dtype=np.float64)
    if ball_y is None:
        ball_y = np.zeros(n_frames, dtype=np.float64)

    return pd.DataFrame(
        {
            "frame": np.arange(n_frames, dtype=np.int64),
            "half": half,
            "mean_speed_home": np.broadcast_to(
                np.asarray(speed_home, dtype=np.float64), (n_frames,)
            ).copy(),
            "mean_speed_away": np.broadcast_to(
                np.asarray(speed_away, dtype=np.float64), (n_frames,)
            ).copy(),
            "ball_speed": np.broadcast_to(
                np.asarray(ball_speed, dtype=np.float64), (n_frames,)
            ).copy(),
            "ball_observed": np.asarray(ball_observed, dtype=np.int64),
            "ball_x": np.asarray(ball_x, dtype=np.float64),
            "ball_y": np.asarray(ball_y, dtype=np.float64),
        }
    )


def _two_half_master(n_per_half: int = 300, **kwargs) -> pd.DataFrame:
    """Create a synthetic two-half master DataFrame (half 1 and half 2)."""
    half = np.array([1] * n_per_half + [2] * n_per_half, dtype=np.int64)
    return _make_master(n_frames=2 * n_per_half, half=half, **kwargs)


def test_cli_default_outputs_only_pass_fail_and_verbose_adds_numbers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Default CLI prints only PASS/FAIL without numbers; --verbose adds diagnostic figures."""
    baseline_cfg = tmp_path / "baseline.yaml"
    baseline_cfg.write_text("dummy: true\n", encoding="utf-8")

    ref_df = _two_half_master(n_per_half=300, speed_home=1.637, speed_away=1.637, ball_speed=4.219)
    ext_df = _two_half_master(n_per_half=300, speed_home=1.641, speed_away=1.641, ball_speed=4.223)

    monkeypatch.setattr(cer, "_n_frames", lambda: len(ref_df))
    monkeypatch.setattr(cer.io, "load_master", lambda *_args, **_kwargs: ref_df)
    monkeypatch.setattr(cer, "load_external", lambda _match, _degrade: (ext_df, None))
    monkeypatch.setattr(
        cer,
        "predict_labels",
        lambda df, _cfg: np.full(len(df), pm.H, dtype=np.int64),
    )

    rc = cer.main(
        [
            "--match",
            "J03WMX",
            "--degrade",
            "default",
            "--baseline-config",
            str(baseline_cfg),
        ]
    )
    out = capsys.readouterr().out

    assert rc == 0
    assert "PASS" in out
    assert "1.637" not in out
    assert "4.219" not in out
    assert "BAR-ATH" not in out

    rc_verbose = cer.main(
        [
            "--match",
            "J03WMX",
            "--degrade",
            "default",
            "--baseline-config",
            str(baseline_cfg),
            "--verbose",
        ]
    )
    out_verbose = capsys.readouterr().out

    assert rc_verbose == 0
    assert "BAR-ATH" in out_verbose
    assert "1.6370" in out_verbose
    assert "4.2190" in out_verbose


def test_statistics_ignore_half_zero_and_unobserved_ball(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stats (a)-(e) ignore half == 0 frames, and (b) ignores ball_observed == 0."""
    half = np.array([0] * 100 + [1] * 250 + [2] * 250, dtype=np.int64)
    obs = np.ones(600, dtype=np.int64)
    # Mark first 50 frames of half 1 as unobserved
    obs[100:150] = 0

    df = _make_master(
        n_frames=600,
        half=half,
        speed_home=1.5,
        speed_away=1.7,
        ball_speed=3.5,
        ball_observed=obs,
    )
    # Corrupt half == 0 rows
    df.loc[df["half"] == 0, "mean_speed_home"] = 99.0
    df.loc[df["half"] == 0, "mean_speed_away"] = 99.0
    df.loc[df["half"] == 0, "ball_speed"] = 99.0
    # Corrupt unobserved rows in half 1
    df.loc[100:149, "ball_speed"] = 77.0

    # Alternate H and A every 25 frames (1s) inside each half -> 10 possessions per 10s = 60/min
    codes = np.zeros(600, dtype=np.int64)
    for start in range(100, 600, 25):
        codes[start : start + 25] = pm.H if ((start - 100) // 25) % 2 == 0 else pm.A
    monkeypatch.setattr(cer, "predict_labels", lambda _df, _cfg: codes)

    stats = cer.master_statistics(df, {})

    assert np.isclose(stats["a"], 1.6)
    assert np.isclose(stats["b"], 3.5)
    assert np.isclose(stats["d"][1], 60.0)
    assert np.isclose(stats["d"][2], 60.0)


def test_stat_c_stationary_ball_respects_half_boundaries_and_visibility() -> None:
    """Stat (c) never pairs t-12 and t+12 across halves and skips unobserved endpoints."""
    # Two halves of 40 frames each; stationary within each half, huge jump between halves
    half = np.array([1] * 40 + [2] * 40, dtype=np.int64)
    ball_x = np.array([10.0] * 40 + [80.0] * 40, dtype=np.float64)
    ball_y = np.full(80, 20.0, dtype=np.float64)

    obs = np.ones(80, dtype=np.int64)
    # Mask frame 20 in half 1: invalidates pairs (8, 32) -> index 20 is t-12 for t=32 and t+12 for t=8
    obs[20] = 0

    df = _make_master(
        n_frames=80,
        half=half,
        ball_observed=obs,
        ball_x=ball_x,
        ball_y=ball_y,
    )

    # Half 1 has 16 interior indices (12..27); two (t=12 uses 0 & 24, valid; t=20 uses 8 & 32 -> 20 is masked
    # only as endpoint for t=8 and t=32, where t=8 and t=32 are outside 12..27! Wait: 20 is t+12 for t=8
    # and t-12 for t=32, neither of which is in [12..27], so all 16 pairs in half 1 remain valid).
    # Now mask frame 12 so t=24 (which uses 12 and 36) becomes invalid: 15 valid in half 1 + 16 in half 2 = 31.
    obs[12] = 0
    df["ball_observed"] = obs

    frac = cer.stat_stationary_fraction(df)
    assert np.isclose(frac, 1.0)

    # Make half 2 move > 0.5m over 24 frames so half 2 contributes 0 stationary out of 16 valid
    df.loc[40:, "ball_x"] = np.linspace(0.0, 20.0, 40)
    frac_mixed = cer.stat_stationary_fraction(df)
    assert np.isclose(frac_mixed, 15.0 / 31.0)


def test_stat_e_counts_only_gaps_strictly_longer_than_125_frames() -> None:
    """Stat (e) counts frames inside gaps > 125 frames within each half, ignoring <= 125."""
    # Half 1: 400 frames with a 125-frame gap (not > 125) and a 150-frame gap (> 125)
    obs_h1 = np.ones(400, dtype=np.int64)
    obs_h1[10:135] = 0  # length 125 -> excluded
    obs_h1[200:350] = 0  # length 150 -> included

    # Half 2: 200 frames, gap of 100 at start (even touching half 1's end, must not merge)
    obs_h2 = np.ones(200, dtype=np.int64)
    obs_h2[:100] = 0  # length 100 -> excluded

    half = np.array([1] * 400 + [2] * 200, dtype=np.int64)
    obs = np.concatenate([obs_h1, obs_h2])
    df = _make_master(n_frames=600, half=half, ball_observed=obs)

    assert np.isclose(cer.stat_long_gap_fraction(df), 150.0 / 600.0)


def test_stat_d_requires_both_halves_to_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stat (d) fails if half 1 passes but half 2 exceeds +/-20% tolerance."""
    ref_df = _two_half_master(n_per_half=300)
    ext_df = _two_half_master(n_per_half=300)

    def _fake_predict(df: pd.DataFrame, _cfg: dict) -> np.ndarray:
        codes = np.full(len(df), pm.H, dtype=np.int64)
        # Half 1: 4 possessions in both ref and ext
        for i in range(4):
            codes[i * 50 : i * 50 + 25] = pm.A
        # Half 2: 4 possessions in ref, 10 possessions in ext (>20% diff)
        n_h2 = 4 if df is ref_df else 10
        for i in range(n_h2):
            codes[300 + i * 25 : 300 + i * 25 + 12] = pm.A
        return codes

    monkeypatch.setattr(cer, "predict_labels", _fake_predict)

    ref_stats = cer.master_statistics(ref_df, {})
    ext_stats = cer.master_statistics(ext_df, {})
    results = {r.key: r for r in cer.run_checks(ref_stats, [ext_stats])}

    assert not results["d"].passed


def test_stat_f_bin_threshold_small_bin_reporting_and_gating_on_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Stat (f) checks <= 0.10 diff on bins >= 2%, ignores < 2% bins, and fails if (a) fails."""
    monkeypatch.setattr(
        cer,
        "predict_labels",
        lambda df, _cfg: np.full(len(df), pm.H, dtype=np.int64),
    )

    n = 1000
    # 500 frames slow (0.5 m/s), 490 medium (1.5 m/s), 10 fast (2.5 m/s -> 1.0% < 2% share)
    speeds = np.array([0.5] * 500 + [1.5] * 490 + [2.5] * 10, dtype=np.float64)

    ref_obs = np.zeros(n, dtype=np.int64)
    ref_obs[:300] = 1  # slow: 0.60
    ref_obs[500:745] = 1  # medium: 0.50
    ref_obs[990:1000] = 1  # fast (<2%): 1.00

    ext_obs = np.zeros(n, dtype=np.int64)
    ext_obs[:345] = 1  # slow: 0.69 (diff 0.09 <= 0.10 -> PASS)
    ext_obs[500:745] = 1  # medium: 0.50 (diff 0.00 -> PASS)
    # fast (<2%): 0.00 (diff 1.00, but share is 1% < 2% so reported only, must NOT fail)

    ref_df = _make_master(n_frames=n, speed_home=speeds, speed_away=speeds, ball_observed=ref_obs)
    ext_df = _make_master(n_frames=n, speed_home=speeds, speed_away=speeds, ball_observed=ext_obs)

    ref_stats = cer.master_statistics(ref_df, {})
    ext_stats = cer.master_statistics(ext_df, {})

    results_ok = {r.key: r for r in cer.run_checks(ref_stats, [ext_stats])}
    assert results_ok["a"].passed
    assert results_ok["f"].passed
    assert "reported only" in results_ok["f"].detail

    # Now make (a) fail -> (f) must also fail and state that (a) failed
    ext_bad_a = _make_master(n_frames=n, speed_home=5.0, speed_away=5.0, ball_observed=ext_obs)
    results_gated = {r.key: r for r in cer.run_checks(ref_stats, [cer.master_statistics(ext_bad_a, {})])}
    assert not results_gated["a"].passed
    assert not results_gated["f"].passed
    assert "(a) failed" in results_gated["f"].detail