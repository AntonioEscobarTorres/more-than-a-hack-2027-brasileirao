"""Parameters of the control evidence. No defaults on purpose.
"""
from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass(frozen=True)
class ControlParams:
    r_ctrl: float             # m, nearest-player distance with full proximity weight
    l_prox: float             # m, decay length of the proximity weight beyond r_ctrl
    v_rel: float              # m/s, relative speed at which the weight drops to 0.5
    ball_conf_filled: float   # confidence of an interpolated (filled) ball position
    rel_w_missing: float      # relative-speed weight when the speed is not available

    def __post_init__(self):
        for name in ("l_prox", "v_rel"):
            if not getattr(self, name) > 0:
                raise ValueError("{} must be > 0, got {!r}".format(name, getattr(self, name)))
        if self.r_ctrl < 0:
            raise ValueError("r_ctrl must be >= 0")
        for name in ("ball_conf_filled", "rel_w_missing"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError("{} must be in [0, 1], got {!r}".format(name, getattr(self, name)))

    @classmethod
    def from_dict(cls, values):
        names = {f.name for f in fields(cls)}
        unknown = sorted(set(values) - names)
        missing = sorted(names - set(values))
        if unknown or missing:
            raise ValueError("{}: unknown keys {}, missing keys {}".format(cls.__name__, unknown, missing))
        return cls(**{k: float(v) for k, v in values.items()})
