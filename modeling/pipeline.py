"""Contract C8: the single function that turns a master table into labels.

Used for BAR-ATH (``make_submission``) and for the external IDSSE matches alike, so a variant is
always evaluated exactly as it would be submitted.
"""
from __future__ import annotations

import importlib
import logging
from pathlib import Path

import numpy as np
import pandas as pd

import games

from . import paths, variants
from .baseline import baseline_labels
from .contracts import LABEL_DTYPE, validate_labels, validate_logits
from .deadball.force import build_force_d
from .decode.emission import logits_to_logscores
from .decode.params import DecoderParams
from .decode.viterbi import decode_match

log = logging.getLogger(__name__)

# Post-processing (F2-04). Missing module → identity stand-in with final imposition only.
POSTPROCESS_MODULE = "modeling.decode.postprocess"
POSTPROCESS_RULES = ("midpoint_neutral_switch", "boundary_shift_file", "d_bridge_frames",
                     "d_cross_frames", "min_spell_frames")


def _rules_enabled(cfg_post: dict) -> list[str]:
    """Names of the post-processing rules that ``cfg_post`` switches on."""
    on = []
    if cfg_post["midpoint_neutral_switch"]:
        on.append("midpoint_neutral_switch")
    if cfg_post["boundary_shift_file"] is not None:
        on.append("boundary_shift_file")
    if cfg_post["d_bridge_frames"] > 0:
        on.append("d_bridge_frames")
    if cfg_post.get("d_cross_frames", 0) > 0:
        on.append("d_cross_frames")
    if cfg_post["min_spell_frames"] > 0:
        on.append("min_spell_frames")
    return on


def final_imposition(labels: np.ndarray, half: np.ndarray, force_d: np.ndarray):
    """F2-04 rule 5 (always applied): ``force_d`` -> D and ``half == 0`` -> X.

    Returns ``(labels, n_changed)``; the input array is not modified.
    """
    out = np.array(labels, dtype=LABEL_DTYPE, copy=True)
    out[force_d] = "D"
    out[half == 0] = "X"
    if (out[half != 0] == "X").any():
        raise ValueError("label X inside a half after the final imposition")
    return out, int(np.count_nonzero(out != labels))


def _identity_postprocess(labels, master, team_logit, dead_logit, force_d, cfg_post, *,
                          holdout=None):
    """Stand-in for F2-04 ``postprocess``: every rule off, then the final imposition."""
    enabled = _rules_enabled(cfg_post)
    if enabled:
        raise NotImplementedError(
            "postprocess rules {} need {} (F2-04), which does not exist yet".format(
                enabled, POSTPROCESS_MODULE))
    out, n_final = final_imposition(labels, master["half"].to_numpy(), force_d)
    stats = {rule: 0 for rule in POSTPROCESS_RULES}
    stats["final_imposition"] = n_final
    return out, stats


def _postprocess_fn():
    """``postprocess`` from F2-04 if the module exists, else the identity stand-in."""
    try:
        module = importlib.import_module(POSTPROCESS_MODULE)
    except ModuleNotFoundError as exc:
        # Only a missing module is tolerated; a broken import inside it is not.
        if exc.name != POSTPROCESS_MODULE:
            raise
        return _identity_postprocess
    return module.postprocess


def variant_params(section: dict) -> dict:
    """The ``params`` a variant receives for one evidence section (``cfg["team"]`` or ``cfg["dead"]``).

    ``params`` itself, unchanged, when the section has no ``calibrated_file``. Otherwise a copy of
    ``params`` with the extra key ``calibrated_file``: the path of the fitted file (``calibrated`` and
    ``gbdt`` kinds), absolute, with a relative path read against the repository root so the result
    does not depend on the working directory. ``cfg`` itself is never modified.
    """
    params = section["params"]
    calibrated_file = section["calibrated_file"]
    if calibrated_file is None:
        return params
    if "calibrated_file" in params:
        raise ValueError("params must not contain 'calibrated_file'; it belongs to the section")
    path = Path(calibrated_file)
    if not path.is_absolute():
        path = paths.repo_root() / path
    return {**params, "calibrated_file": str(path)}


def hmm_labels(master: pd.DataFrame, cfg: dict, *, holdout: str | None = None) -> np.ndarray:
    """HMM pipeline (F5-04): evidence -> emission -> per-half Viterbi -> post-processing.

    Each variant receives its ``params`` section (``cfg["team"]["params"]``,
    ``cfg["dead"]["params"]``), the convention of ``team_logit_rules_v1`` and
    ``dead_logit_rules_v1`` (decision D-008), plus ``params["calibrated_file"]`` when the section
    sets one (:func:`variant_params`, decision D-011).
    """
    variants.load_kinds()
    team_kind, dead_kind = cfg["team"]["kind"], cfg["dead"]["kind"]
    if team_kind not in variants.TEAM_KINDS:
        raise NotImplementedError("team.kind {!r} is not registered (known: {})".format(
            team_kind, sorted(variants.TEAM_KINDS)))
    if dead_kind not in variants.DEAD_KINDS:
        raise NotImplementedError("dead.kind {!r} is not registered (known: {})".format(
            dead_kind, sorted(variants.DEAD_KINDS)))

    n = len(master)
    half = master["half"].to_numpy()
    team = variants.TEAM_KINDS[team_kind](master, variant_params(cfg["team"]), holdout=holdout)
    dead = variants.DEAD_KINDS[dead_kind](master, variant_params(cfg["dead"]), holdout=holdout)
    validate_logits(team, dead, n, half=half)

    force_d = build_force_d(master)
    params = DecoderParams.from_config(cfg["decoder"], fps=games.GAMES[games.GAME]["fps"])
    scores = logits_to_logscores(team, dead, params.lam_team, params.lam_dead, force_d,
                                 neg=params.neg)
    labels = decode_match(scores, half, force_d, params)

    labels, post_stats = _postprocess_fn()(labels, master, team, dead, force_d,
                                           cfg["postprocess"], holdout=holdout)
    log.info("postprocess (%s, holdout=%s): %s", cfg.get("name"), holdout,
             ", ".join("{}={}".format(k, post_stats[k]) for k in sorted(post_stats)))
    return labels


def predict_labels(master: pd.DataFrame, cfg: dict, *, holdout: str | None = None) -> np.ndarray:
    """C8: labels (``<U1``, one per master row) for ``master`` under the config ``cfg``.

    ``holdout`` is the id of the external match being evaluated: calibrated components then use
    the parameters fitted without that match. The baseline has no fitted parameters and ignores it.
    The result always goes through ``validate_labels`` before it is returned.
    """
    pipeline = cfg.get("pipeline")
    if pipeline == "baseline":
        labels = baseline_labels(master, cfg["baseline"])
    elif pipeline == "hmm":
        labels = hmm_labels(master, cfg, holdout=holdout)
    else:
        raise ValueError("unknown pipeline {!r}".format(pipeline))
    validate_labels(labels, master["half"].to_numpy())
    return labels


__all__ = ["final_imposition", "hmm_labels", "predict_labels", "variant_params"]
