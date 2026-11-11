"""Contract C4: the mask of frames that must be labelled D (the pre-kickoff segment)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..contracts import validate_force_d

REQUIRED_COLUMNS = ("half", "pre_kickoff")


def build_force_d(master: pd.DataFrame) -> np.ndarray:
    """C4: bool[n] mask, True on the graded frames that come before the kickoff of their half.

    It is ``master["pre_kickoff"]`` as a bool array, checked with ``validate_force_d``: nothing is
    inferred from the tracking here. The emission (C3) then pins those frames to D.

    * The first half starts at the kickoff itself, so it has no pre-kickoff frames and the mask
      is False over the whole half.
    * In the second half the graded block opens before the kickoff, and the mask covers the
      2,934 frames between the start of the block and the kickoff (published in FAQ.md).
    * On the external IDSSE masters ``pre_kickoff`` is False everywhere, so the mask is all False.

    Raises ``ValueError`` if a column is missing, if ``pre_kickoff`` is not boolean, or if the
    mask is True where ``half == 0``.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in master.columns]
    if missing:
        raise ValueError("master is missing columns for force_d: {}".format(missing))
    pre_kickoff = master["pre_kickoff"]
    if not pd.api.types.is_bool_dtype(pre_kickoff.dtype):
        raise ValueError("pre_kickoff must be boolean, got dtype {}".format(pre_kickoff.dtype))
    if pre_kickoff.isna().any():
        raise ValueError("pre_kickoff has missing values")
    mask = pre_kickoff.to_numpy(dtype=bool, copy=True)
    validate_force_d(mask, master["half"].to_numpy())
    return mask


__all__ = ["REQUIRED_COLUMNS", "build_force_d"]
