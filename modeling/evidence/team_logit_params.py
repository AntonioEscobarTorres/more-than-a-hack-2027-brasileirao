"""Parameters of team_logit v1: the control parameters plus the three of the logit itself.

Same rules as params.py: no defaults, unknown or missing keys are errors, values come from
configs/<variant>.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass

from .params import ControlParams

@dataclass(frozen=True)
class TeamLogitParams(ControlParams):
    kappa: float              # logit bound |t| <= kappa
    sigma_d: float            # m, scale of tanh(d_diff / sigma_d)
    contest_factor: float     # factor applied when both teams have a player within 2 m
    w_coll: float = 0.0       # weight for collective evidence (default 0.0 until calibrated)

    def __post_init__(self):
        super().__post_init__()
        for name in ("kappa", "sigma_d"):
            if getattr(self, name) <= 0:
                raise ValueError("{} must be > 0, got {!r}".format(name, getattr(self, name)))
        if not 0.0 <= self.contest_factor <= 1.0:
            raise ValueError("contest_factor must be in [0, 1]")