"""GBDT ``dead_logit`` (F3-08): features, model files and inference of ``d_fit``.

The model ``g(x)`` is a ``HistGradientBoostingClassifier`` (dead ball vs live ball) whose
``decision_function`` is the log-odds of D, prior included. The likelihood ratio is

    d_fit = g(x) - logit(pi_D)

with ``pi_D`` the share of D in the training rows of that model (the fold's, or the one of
``all``), so the share of D is not counted twice (once here, once in the HMM transitions). The
registered ``gbdt`` dead kind (:mod:`modeling.deadball.dead`) adds the fixed Smart Tagging prior
(``dead_tag_prior``, the external matches have no tags) and clips to +-``DEAD_LOGIT_MAX``, as the
``calibrated`` kind of F3-04 does.

Features (:func:`feature_names`), float64, NaN kept (the model routes missing values natively; no
imputation, no indicator, rule 4):

- per frame (:data:`FRAME_FEATURES`): ``out_of_pitch``, ``db_gap_kind``, ``db_still_frac``,
  ``db_mean_speed``, ``db_fast_in_pitch``, ``gap_length``, ``d_touchline``, ``d_goalline``;
- windows (:mod:`modeling.evidence.windows`, per ``half`` block), for each ``W`` of
  ``window_sizes``:

  ======================================  ==================================================
  column                                  meaning
  ======================================  ==================================================
  ``still_past_mean_W``                   nanmean of ``db_still_frac`` over ``[t - W, t]``
  ``still_fut_mean_W``                    nanmean of ``db_still_frac`` over ``[t, t + W]``
  ``still_fut_minus_past_W``              future minus past (players stopping after a whistle)
  ``speed_past_mean_W``                   same three for ``db_mean_speed`` (players
  ``speed_fut_mean_W``                    speeding up at the restart)
  ``speed_fut_minus_past_W``
  ``d_touchline_cent_min_W``              nanmin over ``[t - W, t + W]``: the ball was near a
  ``d_goalline_cent_min_W``               line
  ``out_of_pitch_cent_max_W``             nanmax of ``out_of_pitch`` over ``[t - W, t + W]``
  ======================================  ==================================================

``db_ball_rest`` and ``ball_speed`` are never features (F3-04, update of 11/11, item 3: the ball
freezes during dead balls on the external matches, so they leak the label), and neither is the
windowed share of ``ball_observed`` (F3-08 update of 11/11, item 2: it is the "ball vanished => D"
channel, the most uncertain part of the degradation).

Every feature is NaN where ``half == 0``; ``d_fit`` is exactly 0 there.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from .. import paths
from ..evidence.windows import rolling_window_stats
from .features import build_deadball_features

FORMAT_VERSION = 1
VARIANT = "gbdt"
DEFAULT_WINDOW_SIZES = (25, 50, 75)

# Raw per-frame features, in matrix order. Master columns or F3-01 dead-ball features.
FRAME_FEATURES = ("out_of_pitch", "db_gap_kind", "db_still_frac", "db_mean_speed",
                  "db_fast_in_pitch", "gap_length", "d_touchline", "d_goalline")
MASTER_COLUMNS = ("half", "out_of_pitch", "gap_length", "d_touchline", "d_goalline")
# Never features (see the module docstring).
FORBIDDEN_FEATURES = ("db_ball_rest", "ball_speed", "ball_observed")

# Keys of a calibration file, checked by :func:`validate_calibration`.
REQUIRED_KEYS = ("format_version", "variant", "sklearn_version", "model_file", "fold_dir",
                 "window_sizes", "feature_params", "features", "pi_d")
BUNDLE_KEYS = ("sklearn_version", "features", "model", "pi_d")


def _check_window_sizes(window_sizes: Sequence) -> tuple[int, ...]:
    sizes = []
    for w in window_sizes:
        w_int = int(w)
        if w_int != w or w_int < 1:
            raise ValueError("window sizes must be positive integers, got {!r}".format(w))
        sizes.append(w_int)
    if not sizes or len(set(sizes)) != len(sizes):
        raise ValueError("window_sizes must be a non-empty list of distinct sizes")
    return tuple(sizes)


def window_feature_names(window_sizes: Sequence[int] = DEFAULT_WINDOW_SIZES) -> list[str]:
    """Names of the window columns, grouped by ``W`` in the order of ``window_sizes``."""
    names = []
    for w in _check_window_sizes(window_sizes):
        for short in ("still", "speed"):
            names += ["{}_past_mean_{}".format(short, w), "{}_fut_mean_{}".format(short, w),
                      "{}_fut_minus_past_{}".format(short, w)]
        names += ["d_touchline_cent_min_{}".format(w), "d_goalline_cent_min_{}".format(w),
                  "out_of_pitch_cent_max_{}".format(w)]
    return names


def feature_names(window_sizes: Sequence[int] = DEFAULT_WINDOW_SIZES) -> list[str]:
    """Column order of the feature matrix: :data:`FRAME_FEATURES`, then the window columns."""
    return list(FRAME_FEATURES) + window_feature_names(window_sizes)


def build_features(master: pd.DataFrame, params: Mapping,
                   window_sizes: Sequence[int] = DEFAULT_WINDOW_SIZES) -> pd.DataFrame:
    """Feature frame (float64, NaN kept, columns of :func:`feature_names`, index of ``master``).

    ``params`` is the ``dead`` section's ``params`` (the F3-01 feature parameters).
    """
    missing = [c for c in MASTER_COLUMNS if c not in master.columns]
    if missing:
        raise ValueError("master is missing columns {}".format(missing))
    sizes = _check_window_sizes(window_sizes)
    db = build_deadball_features(master, dict(params))
    half = master["half"].to_numpy()

    def master_col(name):
        return master[name].to_numpy(dtype=np.float64)

    def db_col(name):
        return db[name].to_numpy(dtype=np.float64)

    cols: dict[str, np.ndarray] = {
        "out_of_pitch": master_col("out_of_pitch"),
        "db_gap_kind": db_col("db_gap_kind"),
        "db_still_frac": db_col("db_still_frac"),
        "db_mean_speed": db_col("db_mean_speed"),
        "db_fast_in_pitch": db_col("db_fast_in_pitch"),
        "gap_length": master_col("gap_length"),
        "d_touchline": master_col("d_touchline"),
        "d_goalline": master_col("d_goalline"),
    }
    for w in sizes:
        for short, source in (("still", cols["db_still_frac"]), ("speed", cols["db_mean_speed"])):
            r = rolling_window_stats(source, half, w, ("past_mean", "fut_mean"))
            cols["{}_past_mean_{}".format(short, w)] = r["past_mean"]
            cols["{}_fut_mean_{}".format(short, w)] = r["fut_mean"]
            cols["{}_fut_minus_past_{}".format(short, w)] = r["fut_mean"] - r["past_mean"]
        for line in ("d_touchline", "d_goalline"):
            cols["{}_cent_min_{}".format(line, w)] = rolling_window_stats(
                cols[line], half, w, ("cent_min",))["cent_min"]
        cols["out_of_pitch_cent_max_{}".format(w)] = rolling_window_stats(
            cols["out_of_pitch"], half, w, ("cent_max",))["cent_max"]

    names = feature_names(sizes)
    out = pd.DataFrame({name: cols[name] for name in names}, index=master.index)
    out.loc[half == 0, :] = np.nan
    return out


# -- model ---------------------------------------------------------------------------------------
def prior_logit(pi_d: float) -> float:
    """``logit(pi_D)``; ``pi_D`` must be in (0, 1)."""
    pi = float(pi_d)
    if not 0.0 < pi < 1.0:
        raise ValueError("pi_D must be in (0, 1), got {}".format(pi))
    return float(np.log(pi) - np.log1p(-pi))


def model_logit(model, X: np.ndarray) -> np.ndarray:
    """``g(X)``: log-odds of D with the prior, float64, single-threaded (rule 6)."""
    from threadpoolctl import threadpool_limits

    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    with threadpool_limits(limits=1):
        return np.asarray(model.decision_function(X), dtype=np.float64)


def d_fit_values(model, X: np.ndarray, pi_d: float) -> np.ndarray:
    """Likelihood ratio ``g(X) - logit(pi_D)``, float64."""
    return model_logit(model, X) - prior_logit(pi_d)


# -- files ---------------------------------------------------------------------------------------
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


def validate_calibration(doc, source: str = "calibration") -> dict:
    """Check the structure of a GBDT dead calibration file; returns it unchanged."""
    if not isinstance(doc, dict):
        raise ValueError("{}: expected a mapping".format(source))
    missing = [k for k in REQUIRED_KEYS if k not in doc]
    if missing:
        raise ValueError("{}: missing keys {}".format(source, missing))
    if doc["format_version"] != FORMAT_VERSION or doc["variant"] != VARIANT:
        raise ValueError("{}: not a GBDT dead calibration file (format {}, variant {!r})".format(
            source, doc["format_version"], doc["variant"]))
    expected = feature_names(doc["window_sizes"])
    if list(doc["features"]) != expected:
        raise ValueError("{}: feature list does not match the code".format(source))
    pi = doc["pi_d"]
    if not isinstance(pi, dict) or "all" not in pi or not isinstance(pi.get("folds"), dict):
        raise ValueError("{}: pi_d needs 'all' and 'folds'".format(source))
    for value in [pi["all"], *pi["folds"].values()]:
        prior_logit(value)                            # raises outside (0, 1)
    return doc


@lru_cache(maxsize=16)
def _load_yaml(path_str: str, mtime_ns: int, size: int) -> dict:
    del mtime_ns, size
    with open(path_str, encoding="utf-8") as handle:
        return validate_calibration(yaml.safe_load(handle), path_str)


def load_calibration(path: str | Path) -> dict:
    """Read and validate a GBDT dead calibration file (cached by path, mtime and size)."""
    p = resolve_repo_path(path)
    if not p.is_file():
        raise FileNotFoundError("GBDT dead calibration file not found: {}".format(p))
    stat = p.stat()
    return _load_yaml(str(p.resolve()), stat.st_mtime_ns, stat.st_size)


def save_bundle(path: str | Path, model, features: Sequence[str], pi_d: float) -> Path:
    """``joblib.dump({"sklearn_version", "features", "model", "pi_d"}, compress=3)``."""
    import joblib
    import sklearn

    prior_logit(pi_d)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"sklearn_version": sklearn.__version__, "features": list(features),
              "model": model, "pi_d": float(pi_d)}
    joblib.dump(bundle, path, compress=3)
    return path


@lru_cache(maxsize=16)
def _load_bundle(path_str: str, mtime_ns: int, size: int) -> dict:
    del mtime_ns, size
    import joblib
    return joblib.load(path_str)


def load_bundle(path: str | Path, *, sklearn_version: str, features: Sequence[str]) -> dict:
    """Load a model bundle saved by :func:`save_bundle`.

    Refuses a running scikit-learn other than ``sklearn_version``, a bundle saved with another
    version and a bundle whose feature list differs from ``features``.
    """
    import sklearn

    if sklearn.__version__ != sklearn_version:
        raise RuntimeError("GBDT dead models need scikit-learn {}, found {}".format(
            sklearn_version, sklearn.__version__))
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError("GBDT dead model not found: {} (run python -m "
                                "modeling.calibration.dead_gbdt)".format(p))
    stat = p.stat()
    bundle = _load_bundle(str(p.resolve()), stat.st_mtime_ns, stat.st_size)
    if not isinstance(bundle, dict) or any(k not in bundle for k in BUNDLE_KEYS):
        raise ValueError("GBDT dead model {} needs keys {}".format(p, BUNDLE_KEYS))
    if bundle["sklearn_version"] != sklearn_version:
        raise RuntimeError("GBDT dead model {} was saved with scikit-learn {}, expected {}".format(
            p, bundle["sklearn_version"], sklearn_version))
    if list(bundle["features"]) != list(features):
        raise ValueError("GBDT dead model {} has a different feature list".format(p))
    return bundle


def select_bundle(calib: Mapping, holdout: str | None) -> dict:
    """The bundle of ``all`` (``holdout`` None) or of the fold trained without ``holdout``.

    Its ``pi_d`` must be the one the calibration file records for that model.
    """
    if holdout is None:
        path, expected = resolve_repo_path(calib["model_file"]), calib["pi_d"]["all"]
    else:
        if holdout not in calib["pi_d"]["folds"]:
            raise ValueError("no fold for holdout {!r}; folds are {}".format(
                holdout, sorted(calib["pi_d"]["folds"])))
        path, expected = fold_model_path(calib, holdout), calib["pi_d"]["folds"][holdout]
    bundle = load_bundle(path, sklearn_version=str(calib["sklearn_version"]),
                         features=calib["features"])
    if float(bundle["pi_d"]) != float(expected):
        raise ValueError("GBDT dead model {} has pi_D {} but the calibration file records {}; "
                         "regenerate both together".format(path, bundle["pi_d"], expected))
    return bundle


def feature_params(params: Mapping) -> dict:
    """The ``dead.params`` that define the features: everything but the weights (as F3-04)."""
    return {k: v for k, v in params.items() if k not in ("weights", "calibrated_file")}


def check_feature_params(calib: Mapping, params: Mapping) -> None:
    """The features at inference must be built exactly as in the fit."""
    fitted, used = calib["feature_params"], feature_params(params)
    diff = sorted(k for k in set(fitted) | set(used) if fitted.get(k) != used.get(k))
    if diff:
        raise ValueError("dead.params differ from the GBDT calibration on {}".format(diff))


def gbdt_d_fit(master: pd.DataFrame, params: Mapping, holdout: str | None = None) -> np.ndarray:
    """``d_fit`` of every master frame (float64), exactly 0 where ``half == 0``.

    ``params["calibrated_file"]`` is the calibration file; ``holdout`` selects the fold model.
    """
    path = params.get("calibrated_file")
    if path is None:
        raise ValueError("dead kind 'gbdt' needs params['calibrated_file']")
    return calibrated_gbdt_d_fit(master, params, load_calibration(path), holdout)


def calibrated_gbdt_d_fit(master: pd.DataFrame, params: Mapping, calib: Mapping,
                          holdout: str | None = None) -> np.ndarray:
    """:func:`gbdt_d_fit` with the calibration file already loaded (``calib``)."""
    check_feature_params(calib, params)
    bundle = select_bundle(calib, holdout)
    X = build_features(master, params, calib["window_sizes"]).to_numpy(dtype=np.float64)
    half = master["half"].to_numpy()
    out = np.zeros(len(master), dtype=np.float64)
    inside = half != 0
    out[inside] = d_fit_values(bundle["model"], X[inside], bundle["pi_d"])
    return out + 0.0


__all__ = ["DEFAULT_WINDOW_SIZES", "FORBIDDEN_FEATURES", "FORMAT_VERSION", "FRAME_FEATURES",
           "VARIANT", "build_features", "calibrated_gbdt_d_fit", "check_feature_params",
           "d_fit_values", "feature_names", "feature_params", "fold_model_path", "gbdt_d_fit", "load_bundle", "load_calibration",
           "model_logit", "prior_logit", "resolve_repo_path", "save_bundle", "select_bundle",
           "validate_calibration", "window_feature_names"]
