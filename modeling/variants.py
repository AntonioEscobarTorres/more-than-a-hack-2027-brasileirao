"""Registries of evidence variants: ``team.kind`` -> TEAM_KINDS, ``dead.kind`` -> DEAD_KINDS,
``hazard.kind`` -> HAZARD_KINDS (F1-08; the config section arrives with F2-07).

Every variant has the signature::

    fn(master: pd.DataFrame, params: dict, *, holdout: str | None) -> np.ndarray  # float32[n]

and registers itself with ``@register_team("name")``, ``@register_dead("name")`` or
``@register_hazard("name")`` in its own
module. ``params`` is the section's ``params``; when the section sets ``calibrated_file`` (the
``calibrated`` and ``gbdt`` kinds), it carries the extra key ``params["calibrated_file"]``, an
absolute path (``modeling.pipeline.variant_params``, decision D-011). :func:`load_kinds` imports those modules so the registries are populated.
"""
from __future__ import annotations

import importlib
from typing import Callable, Dict

import numpy as np
import pandas as pd

VariantFn = Callable[..., np.ndarray]

TEAM_KINDS: Dict[str, VariantFn] = {}
DEAD_KINDS: Dict[str, VariantFn] = {}
HAZARD_KINDS: Dict[str, VariantFn] = {}

# Modules that register variants; imported by load_kinds() when they exist.
KIND_MODULES = ("modeling.evidence.team", "modeling.deadball.dead", "modeling.evidence.hazard",
                "modeling.evidence.hazard_gbdt")


def _register(registry: Dict[str, VariantFn], label: str, name: str):
    if not isinstance(name, str) or not name:
        raise ValueError("{} kind name must be a non-empty string, got {!r}".format(label, name))

    def decorator(fn: VariantFn) -> VariantFn:
        current = registry.get(name)
        if current is not None and current is not fn:
            raise ValueError("{} kind {!r} is already registered by {}.{}".format(
                label, name, current.__module__, current.__qualname__))
        registry[name] = fn
        return fn

    return decorator


def register_team(name: str):
    """Decorator: register ``fn`` as the team-evidence variant ``name``."""
    return _register(TEAM_KINDS, "team", name)


def register_dead(name: str):
    """Decorator: register ``fn`` as the dead-ball-evidence variant ``name``."""
    return _register(DEAD_KINDS, "dead", name)


def register_hazard(name: str):
    """Decorator: register ``fn`` as the possession-change hazard variant ``name``."""
    return _register(HAZARD_KINDS, "hazard", name)


def _zeros(master: pd.DataFrame, params: dict, *, holdout: str | None = None) -> np.ndarray:
    """No evidence: 0 on every frame."""
    return np.zeros(len(master), dtype=np.float32)


register_team("none")(_zeros)
register_dead("none")(_zeros)
register_hazard("none")(_zeros)


def load_kinds() -> None:
    """Import the modules that register variants, skipping the ones not written yet."""
    for module in KIND_MODULES:
        try:
            importlib.import_module(module)
        except ModuleNotFoundError as exc:
            # Only a missing variant module is tolerated; a broken import inside it is not.
            if exc.name != module:
                raise


__all__ = ["DEAD_KINDS", "HAZARD_KINDS", "KIND_MODULES", "TEAM_KINDS", "load_kinds",
           "register_dead", "register_hazard", "register_team"]
