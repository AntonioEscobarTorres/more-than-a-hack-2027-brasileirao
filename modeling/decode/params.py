"""Immutable decoder parameters (task F2-01) and hazard-modulated team exits (task F2-07).

``DecoderParams.from_config`` converts the ``decoder`` section of an HMM config into
float64 arrays used by the Viterbi decoder, including ``p_stay`` derived from the
mean duration and minimum dwell. :func:`team_exit_logs` gives the frame-dependent
transition row of the terminal H and A sub-states under an exit logit (IO-HMM, D-024).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from modeling.contracts import STATES
from modeling.decode.emission import DEFAULT_NEG

_STATE_INDEX = {name: i for i, name in enumerate(STATES)}
H, A, D = (_STATE_INDEX[name] for name in ("H", "A", "D"))
TEAM_STATES = (H, A)


def _p_stay_from_mean(mean_frames: float, min_dwell: int) -> float:
    """Stay probability so the expected segment length equals ``mean_frames``.

    After reaching the terminal sub-state, each step stays with ``p`` or exits with
    ``1 − p``. Expected length = ``min_dwell − 1 + 1/(1 − p)``, which equals
    ``mean_frames`` when ``p = 1 − 1/(mean_frames − min_dwell + 1)``.
    """
    mean_frames = float(mean_frames)
    min_dwell = int(min_dwell)
    if not mean_frames > min_dwell:
        raise ValueError(
            "mean_duration_frames ({}) must exceed min_dwell ({})".format(
                mean_frames, min_dwell
            )
        )
    return 1.0 - 1.0 / (mean_frames - min_dwell + 1.0)


@dataclass(frozen=True)
class DecoderParams:
    """Frozen float64 parameters for emission scaling and the expanded-state HMM."""

    lam_team: float
    lam_dead: float
    mean_duration_frames: np.ndarray  # (3,) float64
    exit_split: np.ndarray  # (3, 3) float64, zero diagonal, rows sum to 1
    min_dwell: np.ndarray  # (3,) int64, each >= 1
    neg: float
    p_stay: np.ndarray  # (3,) float64
    log_p_stay: np.ndarray  # (3,) float64
    log_exit: np.ndarray  # (3, 3) float64; log((1-p_stay[s]) * exit_split[s,s'])

    def __post_init__(self) -> None:
        # Frozen dataclass: use object.__setattr__ for array copies / casts.
        object.__setattr__(
            self, "mean_duration_frames",
            np.asarray(self.mean_duration_frames, dtype=np.float64).reshape(3).copy())
        object.__setattr__(
            self, "exit_split",
            np.asarray(self.exit_split, dtype=np.float64).reshape(3, 3).copy())
        object.__setattr__(
            self, "min_dwell",
            np.asarray(self.min_dwell, dtype=np.int64).reshape(3).copy())
        object.__setattr__(
            self, "p_stay",
            np.asarray(self.p_stay, dtype=np.float64).reshape(3).copy())
        object.__setattr__(
            self, "log_p_stay",
            np.asarray(self.log_p_stay, dtype=np.float64).reshape(3).copy())
        object.__setattr__(
            self, "log_exit",
            np.asarray(self.log_exit, dtype=np.float64).reshape(3, 3).copy())
        # Frozen means frozen: the arrays are read-only too, so no caller can edit the
        # transition table in place after the params were validated.
        for name in ("mean_duration_frames", "exit_split", "min_dwell", "p_stay",
                     "log_p_stay", "log_exit"):
            getattr(self, name).setflags(write=False)
        object.__setattr__(self, "lam_team", float(self.lam_team))
        object.__setattr__(self, "lam_dead", float(self.lam_dead))
        object.__setattr__(self, "neg", float(self.neg))

    @classmethod
    def from_config(cls, cfg_decoder: Mapping, fps: float = 25) -> "DecoderParams":
        """Build params from a validated ``cfg["decoder"]`` mapping.

        ``mean_duration_s`` is converted with ``fps``; ``p_stay`` follows the
        closed-form that makes the expected segment length equal
        ``mean_duration_frames``.
        """
        fps = float(fps)
        if not fps > 0:
            raise ValueError("fps must be positive, got {!r}".format(fps))

        lam_team = float(cfg_decoder["lam_team"])
        lam_dead = float(cfg_decoder["lam_dead"])
        neg = float(cfg_decoder.get("neg", DEFAULT_NEG))

        mean_duration_frames = np.empty(3, dtype=np.float64)
        min_dwell = np.empty(3, dtype=np.int64)
        for name, idx in _STATE_INDEX.items():
            mean_duration_frames[idx] = float(cfg_decoder["mean_duration_s"][name]) * fps
            min_dwell[idx] = int(cfg_decoder["min_dwell_frames"][name])
            if min_dwell[idx] < 1:
                raise ValueError("min_dwell[{}] must be >= 1, got {}".format(
                    name, min_dwell[idx]))
            if not mean_duration_frames[idx] > min_dwell[idx]:
                raise ValueError(
                    "mean_duration_frames[{}] ({}) must exceed min_dwell[{}] ({})".format(
                        name, mean_duration_frames[idx], name, min_dwell[idx]))

        exit_split = np.zeros((3, 3), dtype=np.float64)
        for src, si in _STATE_INDEX.items():
            row = cfg_decoder["exit_split"][src]
            for dst, di in _STATE_INDEX.items():
                if dst == src:
                    continue
                value = float(row[dst])
                if not 0.0 <= value <= 1.0:
                    raise ValueError("exit_split[{}][{}] must be in [0, 1], got {}".format(
                        src, dst, value))
                exit_split[si, di] = value
            total = float(exit_split[si].sum())
            if abs(total - 1.0) > 1e-9:
                raise ValueError(
                    "exit_split[{}] must sum to 1, got {}".format(src, total))

        p_stay = np.empty(3, dtype=np.float64)
        for s in range(3):
            p_stay[s] = _p_stay_from_mean(mean_duration_frames[s], int(min_dwell[s]))

        log_p_stay = np.log(p_stay)
        one_minus = 1.0 - p_stay
        log_exit = np.full((3, 3), -np.inf, dtype=np.float64)
        for s in range(3):
            for sp in range(3):
                if s == sp:
                    continue
                if exit_split[s, sp] > 0.0 and one_minus[s] > 0.0:
                    log_exit[s, sp] = np.log(one_minus[s]) + np.log(exit_split[s, sp])

        return cls(
            lam_team=lam_team,
            lam_dead=lam_dead,
            mean_duration_frames=mean_duration_frames,
            exit_split=exit_split,
            min_dwell=min_dwell,
            neg=neg,
            p_stay=p_stay,
            log_p_stay=log_p_stay,
            log_exit=log_exit,
        )


def team_exit_logs(params: DecoderParams, exit_logit) -> dict[int, tuple[np.ndarray, ...]]:
    """Transition row of the terminal sub-state of H and of A, per frame, under ``exit_logit``.

    For a team state ``s`` with stay ``p``, exit to the other team ``q_o`` and exit to D ``q_d``
    (the constant row of ``params``), frame ``t`` with ``x_t = exit_logit[t]`` uses::

        q_o' = q_o e^{x_t} / Z_t,   p' = p / Z_t,   q_d' = q_d / Z_t,   Z_t = p + q_d + q_o e^{x_t}

    computed in log space with ``logaddexp`` in float64 (``log Z_t = logaddexp(log(p + q_d),
    log q_o + x_t)``). An edge with probability 0 (``-inf``) stays ``-inf``. Returns
    ``{s: (log_stay, log_other, log_d)}`` for ``s`` in (H, A), each float64 ``(T,)``.
    """
    x = np.asarray(exit_logit, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("exit_logit must be one-dimensional, got shape {}".format(x.shape))
    if not np.isfinite(x).all():
        raise ValueError("exit_logit must be finite")
    out = {}
    for s, other in ((H, A), (A, H)):
        log_p = float(params.log_p_stay[s])
        log_qo = float(params.log_exit[s, other])
        log_qd = float(params.log_exit[s, D])
        log_keep = np.logaddexp(log_p, log_qd)          # log(p + q_d), the unmodulated mass
        log_other = log_qo + x                          # log(q_o e^x); -inf stays -inf
        log_z = np.logaddexp(log_keep, log_other)
        out[s] = (log_p - log_z, log_other - log_z, log_qd - log_z)
    return out
