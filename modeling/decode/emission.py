"""C3 emission: convert team/dead logits into per-frame log-scores for (H, A, D).

See docs/contracts.md (C3, C4) and docs/tasks/F2-01.md.
"""
from __future__ import annotations

import numpy as np

DEFAULT_NEG = -1.0e6


def logits_to_logscores(
    team_logit,
    dead_logit,
    lam_team: float,
    lam_dead: float,
    force_d,
    neg: float = DEFAULT_NEG,
) -> np.ndarray:
    """Map per-frame logits to float64 log-scores of shape ``(n, 3)`` in order (H, A, D).

    Exactly the C3 formulae::

        logσ(x) = −logaddexp(0, −x)
        S_H = lam_team · (log 2 + logσ(t))
        S_A = lam_team · (log 2 + logσ(−t))
        S_D = lam_dead · d

    Where ``force_d`` is True: ``S_D = 0`` and ``S_H = S_A = neg`` (C4).
    """
    t = np.asarray(team_logit, dtype=np.float64)
    d = np.asarray(dead_logit, dtype=np.float64)
    if t.ndim != 1 or d.ndim != 1 or t.shape != d.shape:
        raise ValueError(
            "team_logit and dead_logit must be 1-D with the same shape, got "
            "{} and {}".format(getattr(t, "shape", None), getattr(d, "shape", None))
        )
    mask = np.asarray(force_d, dtype=bool)
    if mask.shape != t.shape:
        raise ValueError(
            "force_d must have shape {}, got {}".format(t.shape, mask.shape)
        )

    # logσ(x) = −logaddexp(0, −x); at x=0 this is −log(2), so log2 + logσ(0) = 0.
    log_sigma_t = -np.logaddexp(0.0, -t)
    log_sigma_neg_t = -np.logaddexp(0.0, t)
    lam_team = float(lam_team)
    lam_dead = float(lam_dead)
    s_h = lam_team * (np.log(2.0) + log_sigma_t)
    s_a = lam_team * (np.log(2.0) + log_sigma_neg_t)
    s_d = lam_dead * d

    scores = np.empty((t.shape[0], 3), dtype=np.float64)
    scores[:, 0] = s_h
    scores[:, 1] = s_a
    scores[:, 2] = s_d

    if mask.any():
        neg_f = float(neg)
        scores[mask, 0] = neg_f
        scores[mask, 1] = neg_f
        scores[mask, 2] = 0.0
    return scores
