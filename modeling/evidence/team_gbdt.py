"""GBDT ``team_logit`` (F1-07): features, team swap and inference.

The model ``g(x)`` is a ``HistGradientBoostingClassifier`` whose ``decision_function`` is the
log-odds of H. The evidence is antisymmetric by construction::

    t(m) = clip((g(X(m)) - g(S X(m))) / 2, +-TEAM_LOGIT_MAX)

where ``S`` is the team swap written on the feature matrix: every home/away pair of columns is
exchanged, every odd column changes sign and every even column stays. ``S`` is a permutation
with signs, so ``S S X == X`` bit for bit and ``t`` computed on ``S X`` is exactly ``-t``. ``S X(m)``
equals ``X(swap(m))`` (the features of the master with the teams swapped, the spec's definition,
decision D-023) up to float rounding in the derived features; ``tests/test_modeling_team_gbdt.py``
checks that equality on a synthetic master.

Feature groups (:func:`gbdt_feature_layout`):

- odd: the F1-05 candidate features ``f1``..``f9`` (``f9`` = touch with the calibrated
  thresholds), the pass bridge ``f10_<tau>``/``f11_<tau>`` when ``include_bridge``, the three
  collective features and ``d_diff``;
- pairs: the raw home/away master columns of :data:`RAW_PAIRS`;
- even: :data:`EVEN_COLUMNS` (ball confidence, ball speed, distances to the lines, gap timing);
- windows: :mod:`modeling.evidence.windows`.

``f_params["exclude_features"]`` removes features (a home/away pair as a whole when either side
is named); D-023 removes the ones that fail the BAR-ATH range check of
``tools/gbdt_feature_shift.py``.

Missing values stay NaN: the model routes them natively (no imputation, rule 4).
Ball-position columns (``ball_x``, ``ball_vx``...) are left out: their meaning depends on the
attacking direction, which changes at half time.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .. import paths
from ..contracts import TEAM_LOGIT_MAX
from .control import build_control_features
from .windows import build_window_features, window_feature_roles, window_sizes_from

RAW_PAIRS = (
    ("d_near_home", "d_near_away"),
    ("n_home_r2", "n_away_r2"),
    ("n_home_r3", "n_away_r3"),
    ("n_home_r5", "n_away_r5"),
    ("pressure_home", "pressure_away"),
    ("speed_rel_home", "speed_rel_away"),
    ("speed_radial_home", "speed_radial_away"),
    ("mean_speed_home", "mean_speed_away"),
    ("n_still_home", "n_still_away"),
    ("n_tracked_home", "n_tracked_away"),
)
EVEN_COLUMNS = ("ev_ball_conf", "ball_speed", "d_touchline", "d_goalline", "gap_length",
                "since_last_obs", "until_next_obs")
COLLECTIVE_ODD = ("ev_coll_adv", "ev_coll_spread", "ev_coll_speed")
BASE_ODD = ("f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9")
GAP_MODELS = ("gbdt", "logistic")


def _tau_label(tau) -> str:
    return str(int(tau)) if float(tau).is_integer() else str(tau)


def gbdt_feature_layout(f_params: Mapping) -> dict:
    """Feature names and team-swap roles: ``{"features", "pairs", "odd", "even"}``."""
    odd = list(BASE_ODD)
    if f_params.get("include_bridge", True):
        for tau in f_params["bridge_tau_frames"]:
            odd.extend(["f10_" + _tau_label(tau), "f11_" + _tau_label(tau)])
    odd.extend(COLLECTIVE_ODD)
    odd.append("d_diff")
    pairs = [tuple(p) for p in RAW_PAIRS]
    even = list(EVEN_COLUMNS)
    win = window_feature_roles(window_sizes_from(f_params))
    pairs.extend(win["pairs"])
    odd.extend(win["odd"])
    even.extend(win["even"])
    excluded = set(f_params.get("exclude_features", ()))
    unknown = excluded - set(odd) - set(even) - {c for p in pairs for c in p}
    if unknown:
        raise ValueError("unknown GBDT features to exclude: {}".format(sorted(unknown)))
    # A pair is excluded as a whole: dropping one side would break the swap.
    pairs = [p for p in pairs if not (set(p) & excluded)]
    odd = [c for c in odd if c not in excluded]
    even = [c for c in even if c not in excluded]
    features = list(odd) + [c for p in pairs for c in p] + list(even)
    if len(set(features)) != len(features):
        raise ValueError("duplicate GBDT feature names")
    return {"features": features, "pairs": pairs, "odd": odd, "even": even}


def swap_permutation(layout: Mapping) -> tuple[np.ndarray, np.ndarray]:
    """``(perm, sign)`` with ``S X = X[:, perm] * sign`` for the feature order of ``layout``."""
    names = list(layout["features"])
    pos = {name: i for i, name in enumerate(names)}
    perm = np.arange(len(names))
    sign = np.ones(len(names), dtype=np.float64)
    for a, b in layout["pairs"]:
        perm[pos[a]], perm[pos[b]] = pos[b], pos[a]
    for name in layout["odd"]:
        sign[pos[name]] = -1.0
    return perm, sign


def swap_matrix(X: np.ndarray, perm: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Team swap of a feature matrix (exact involution)."""
    return X[:, perm] * sign


def build_gbdt_features(master: pd.DataFrame, params: Mapping, f_params: Mapping
                        ) -> tuple[pd.DataFrame, np.ndarray]:
    """Feature frame (float64, NaN kept, column order of the layout) and ``ev_ball_valid``.

    ``params`` is the team section's ``params``; ``f_params`` are the feature parameters of the
    calibration file (they override ``params``).
    """
    # Imported here: modeling.evidence.team imports this module lazily as well.
    from .team import build_team_calibration_features

    layout = gbdt_feature_layout(f_params)
    p = {**dict(params), **dict(f_params)}
    taus = list(f_params["bridge_tau_frames"])
    base = build_team_calibration_features(
        master, p, include_bridge=bool(f_params.get("include_bridge", True)),
        include_touch=True, taus=taus)
    ev = build_control_features(master, p)
    win = build_window_features(master, p)

    half = master["half"].to_numpy()
    cols: dict[str, np.ndarray] = {}
    for name in layout["features"]:
        if name in base.columns:
            src = base[name]
        elif name in ev.columns:
            src = ev[name]
        elif name in win.columns:
            src = win[name]
        elif name in master.columns:
            src = master[name]
        else:
            raise KeyError("GBDT feature {!r} has no source".format(name))
        cols[name] = src.to_numpy(dtype=np.float64)
    feats = pd.DataFrame(cols, index=master.index)
    feats.loc[half == 0, :] = np.nan
    valid = base["ev_ball_valid"].to_numpy(dtype=bool) & (half != 0)
    return feats, valid


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else paths.repo_root() / p


def fold_model_path(calib: Mapping, holdout: str) -> Path:
    """Fold model of ``holdout``: ``<fold_dir>/<holdout>.joblib``, ``fold_dir`` relative to
    ``ext_dir()`` unless absolute (fold models are not committed)."""
    fold_dir = Path(calib["fold_dir"])
    if not fold_dir.is_absolute():
        fold_dir = paths.ext_dir() / fold_dir
    return fold_dir / "{}.joblib".format(holdout)


@lru_cache(maxsize=16)
def _load_yaml(path_str: str, mtime_ns: int) -> dict:
    del mtime_ns
    with open(path_str, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or data.get("variant") != "gbdt":
        raise ValueError("invalid GBDT team calibration file: {}".format(path_str))
    for key in ("sklearn_version", "gap_model", "model_file", "fold_dir", "feature_params",
                "features"):
        if key not in data:
            raise ValueError("GBDT team calibration file {} misses {!r}".format(path_str, key))
    if data["gap_model"] not in GAP_MODELS:
        raise ValueError("gap_model must be one of {}, got {!r}".format(GAP_MODELS,
                                                                       data["gap_model"]))
    if data["gap_model"] == "logistic" and not data.get("logistic_file"):
        raise ValueError("gap_model 'logistic' needs 'logistic_file'")
    return data


def load_gbdt_calibration(path: str | Path) -> dict:
    p = resolve_repo_path(path)
    if not p.is_file():
        raise FileNotFoundError("GBDT team calibration file not found: {}".format(p))
    return _load_yaml(str(p.resolve()), p.stat().st_mtime_ns)


@lru_cache(maxsize=16)
def _load_bundle(path_str: str, mtime_ns: int) -> dict:
    del mtime_ns
    import joblib
    return joblib.load(path_str)


def load_gbdt_bundle(path: str | Path, *, sklearn_version: str,
                     features: Sequence[str]) -> dict:
    """Load a ``{"sklearn_version", "features", "ball", "gap"}`` joblib bundle.

    Refuses a bundle saved with another scikit-learn version, a bundle whose feature list
    differs from the calibration file and a running scikit-learn other than ``sklearn_version``.
    """
    import sklearn

    if sklearn.__version__ != sklearn_version:
        raise RuntimeError("GBDT models need scikit-learn {}, found {}".format(
            sklearn_version, sklearn.__version__))
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(
            "GBDT model not found: {} (run python -m modeling.calibration.team_gbdt)".format(p))
    bundle = _load_bundle(str(p.resolve()), p.stat().st_mtime_ns)
    if bundle.get("sklearn_version") != sklearn_version:
        raise RuntimeError("GBDT model {} was saved with scikit-learn {}, expected {}".format(
            p, bundle.get("sklearn_version"), sklearn_version))
    if list(bundle.get("features", [])) != list(features):
        raise ValueError("GBDT model {} has a different feature list".format(p))
    return bundle


def antisymmetric_logit(model, X: np.ndarray, perm: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """``(g(X) - g(S X)) / 2`` in float64, single-threaded (rule 6)."""
    from threadpoolctl import threadpool_limits

    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    Xs = swap_matrix(X, perm, sign)
    with threadpool_limits(limits=1):
        g = model.decision_function(X).astype(np.float64)
        gs = model.decision_function(Xs).astype(np.float64)
    return 0.5 * (g - gs)


def team_logit_gbdt_impl(master: pd.DataFrame, params: Mapping, *, holdout=None) -> np.ndarray:
    """Body of the registered ``gbdt`` team kind (see :func:`modeling.evidence.team.team_logit_gbdt`)."""
    from .team import (_score_linear_section, build_team_calibration_features,
                       load_team_calibration, team_tag_prior)

    if not params.get("calibrated_file"):
        raise ValueError("gbdt team_logit requires params['calibrated_file']")
    calib = load_gbdt_calibration(params["calibrated_file"])
    f_params = dict(calib["feature_params"])
    layout = gbdt_feature_layout(f_params)
    if layout["features"] != list(calib["features"]):
        raise ValueError("feature list of {} does not match the code".format(
            params["calibrated_file"]))
    model_path = (resolve_repo_path(calib["model_file"]) if holdout is None
                  else fold_model_path(calib, holdout))
    bundle = load_gbdt_bundle(model_path, sklearn_version=str(calib["sklearn_version"]),
                              features=calib["features"])

    feats, valid = build_gbdt_features(master, params, f_params)
    perm, sign = swap_permutation(layout)
    X = feats.to_numpy(dtype=np.float64)

    t = np.zeros(len(master), dtype=np.float64)
    t[valid] = antisymmetric_logit(bundle["ball"], X[valid], perm, sign)
    half = master["half"].to_numpy()
    gap = (~valid) & (half != 0)
    if calib["gap_model"] == "gbdt":
        t[gap] = antisymmetric_logit(bundle["gap"], X[gap], perm, sign)
    else:
        logistic = load_team_calibration(resolve_repo_path(calib["logistic_file"]))
        lp = {**dict(params), **logistic.get("feature_params", {})}
        lfeats = build_team_calibration_features(master, lp, include_bridge=any(
            n.startswith(("f10_", "f11_")) for n in logistic["gap"]["features"]),
            include_touch=False, taus=logistic.get("feature_params", {}).get("bridge_tau_frames"))
        t[gap] = _score_linear_section(lfeats, logistic["gap"], holdout)[gap]

    t = t + team_tag_prior(master, {**dict(params), **f_params})
    t = np.clip(t, -TEAM_LOGIT_MAX, TEAM_LOGIT_MAX)
    t = np.where((half != 0) & np.isfinite(t), t, 0.0) + 0.0
    return t.astype(np.float32)


__all__ = ["BASE_ODD", "COLLECTIVE_ODD", "EVEN_COLUMNS", "GAP_MODELS", "RAW_PAIRS",
           "antisymmetric_logit", "build_gbdt_features", "fold_model_path",
           "gbdt_feature_layout", "load_gbdt_bundle", "load_gbdt_calibration",
           "resolve_repo_path", "swap_matrix", "swap_permutation", "team_logit_gbdt_impl"]
