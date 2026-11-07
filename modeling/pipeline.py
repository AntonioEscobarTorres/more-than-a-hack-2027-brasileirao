"""Contract C8: the single function that turns a master table into labels.

Used for BAR-ATH (``make_submission``) and for the external IDSSE matches alike, so a variant is
always evaluated exactly as it would be submitted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .baseline import baseline_labels
from .contracts import validate_labels


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
        raise NotImplementedError("pipeline 'hmm' is not wired into predict_labels yet (F5-04)")
    else:
        raise ValueError("unknown pipeline {!r}".format(pipeline))
    validate_labels(labels, master["half"].to_numpy())
    return labels


__all__ = ["predict_labels"]
