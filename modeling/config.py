"""Contract C7: strict YAML config reader and config hash (schema: plan, Appendix A.7).

Every key is required and no other key is allowed; a duplicated key, a wrong type or a value
out of range is an error. Numbers declared as floats are normalised to ``float`` (so ``15`` and
``15.0`` give the same config and the same hash); booleans are never accepted as numbers.
"""
from __future__ import annotations

import hashlib
import json
import math
import copy
from pathlib import Path

import yaml

import games

PIPELINES = ("baseline", "hmm")
BALL_SOURCES = ("tactical", "fused")
TEAM_KIND_NAMES = ("none", "rules_v1", "calibrated", "gbdt")
DEAD_KIND_NAMES = ("none", "rules_v1", "calibrated", "gbdt")
KINDS_WITH_FILE = ("calibrated", "gbdt")
HMM_SECTIONS = ("team", "dead", "decoder", "postprocess")
EXIT_SPLIT_TOL = 1e-9


class ConfigError(ValueError):
    """A config file does not follow the schema."""


# -- field types --------------------------------------------------------------------------------

class _Field:
    def check(self, value, where):  # pragma: no cover - interface
        raise NotImplementedError


class Float(_Field):
    """A real number (int accepted, normalised to float) with optional bounds."""

    def __init__(self, gt=None, ge=None, le=None):
        self.gt, self.ge, self.le = gt, ge, le

    def check(self, value, where):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError("{}: expected a number, got {!r}".format(where, value))
        value = float(value)
        if not math.isfinite(value):
            raise ConfigError("{}: must be finite, got {!r}".format(where, value))
        if self.gt is not None and not value > self.gt:
            raise ConfigError("{}: must be > {}, got {}".format(where, self.gt, value))
        if self.ge is not None and not value >= self.ge:
            raise ConfigError("{}: must be >= {}, got {}".format(where, self.ge, value))
        if self.le is not None and not value <= self.le:
            raise ConfigError("{}: must be <= {}, got {}".format(where, self.le, value))
        return value


class Int(_Field):
    """An integer (bools and floats rejected) with an optional lower bound."""

    def __init__(self, ge=None):
        self.ge = ge

    def check(self, value, where):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError("{}: expected an integer, got {!r}".format(where, value))
        if self.ge is not None and value < self.ge:
            raise ConfigError("{}: must be >= {}, got {}".format(where, self.ge, value))
        return value


class Bool(_Field):
    def check(self, value, where):
        if not isinstance(value, bool):
            raise ConfigError("{}: expected true/false, got {!r}".format(where, value))
        return value


class Str(_Field):
    """A non-empty string, optionally restricted to ``choices``; ``nullable`` allows null."""

    def __init__(self, choices=None, nullable=False):
        self.choices, self.nullable = choices, nullable

    def check(self, value, where):
        if value is None and self.nullable:
            return None
        if not isinstance(value, str) or not value:
            raise ConfigError("{}: expected a non-empty string, got {!r}".format(where, value))
        if self.choices is not None and value not in self.choices:
            raise ConfigError("{}: must be one of {}, got {!r}".format(
                where, list(self.choices), value))
        return value

class FloatList(_Field):
    """A non-empty list of numbers, each checked by ``item``."""

    def __init__(self, item):
        self.item = item

    def check(self, value, where):
        if not isinstance(value, list) or not value:
            raise ConfigError("{}: expected a non-empty list, got {!r}".format(where, value))
        return [self.item.check(v, "{}[{}]".format(where, i)) for i, v in enumerate(value)]


class Opt(_Field):
    """Optional field or sub-mapping with a default when omitted from the YAML."""

    def __init__(self, inner, default=None):
        self.inner = inner
        self.default = default

    def check(self, value, where):
        if isinstance(self.inner, dict):
            return _check_mapping(self.inner, value, where)
        return self.inner.check(value, where)


def _check_mapping(schema: dict, value, where: str) -> dict:
    """Check that value has the required keys of schema and validate each one."""
    if not isinstance(value, dict):
        raise ConfigError("{}: expected a mapping, got {!r}".format(where or "<root>", value))
    unknown = sorted(set(value) - set(schema), key=str)
    if unknown:
        raise ConfigError("{}: unknown keys {}".format(where or "<root>", unknown))
    missing = [k for k, spec in schema.items() if not isinstance(spec, Opt) and k not in value]
    if missing:
        raise ConfigError("{}: missing keys {}".format(where or "<root>", missing))
    out = {}
    for key, spec in schema.items():
        if isinstance(spec, Opt) and key not in value:
            if spec.default is not None:
                out[key] = copy.deepcopy(spec.default)
            continue
        sub = "{}.{}".format(where, key) if where else key
        out[key] = _check_mapping(spec, value[key], sub) if isinstance(spec, dict) \
            else spec.check(value[key], sub)
    return out


# -- schema (Appendix A.7) ----------------------------------------------------------------------

_POS = Float(gt=0.0)
_UNIT = Float(ge=0.0, le=1.0)
_ANY = Float()

BASELINE_SCHEMA = {
    "d_gap_frames": Int(ge=1),
    "slow_speed_mps": _POS,
    "min_spell_frames": Int(ge=0),
}

TEAM_SCHEMA = {
    "kind": Str(TEAM_KIND_NAMES),
    "params": {
        "pitch_l_m": _POS,
        "pitch_w_m": _POS,
        "kappa": _POS,
        "sigma_d": _POS,
        "r_ctrl": _POS,
        "l_prox": _POS,
        "v_rel": _POS,
        "ball_conf_filled": _UNIT,
        "contest_factor": _UNIT,
        "rel_w_when_nan": _UNIT,
        "collective": {
            "win_frames": Int(ge=1),
            "w_adv": _ANY,
            "w_spread": _ANY,
            "w_speed": _ANY,
        },
        "bridge_median_frames": Opt(Int(ge=1)),
        "bridge_tau_frames": Opt(FloatList(_POS)),
        "r_touch_m": Opt(_POS),
        "touch_min_frames": Opt(Int(ge=1)),
        "touch_win_frames": Opt(Int(ge=1)),
        "touch_dir_change_rad": Opt(_POS),
        "touch_acc_mps2": Opt(_POS),
        "w_tag_phase": Opt(_ANY),
        "w_sp_team": Opt(_ANY),
        "tau_sp_team": Opt(_POS),
    },
    "calibrated_file": Str(nullable=True),
}

DEAD_WEIGHT_KEYS = ("out_of_pitch", "gap_exit", "gap_air", "gap_long", "still", "speed_high",
                    "speed_low", "ball_rest", "fast_in_pitch", "tag_set_piece", "tag_sp_prox",
                    "tag_sp_after", "tag_goal_window")

DEAD_SCHEMA = {
    "kind": Str(DEAD_KIND_NAMES),
    "params": {
        "long_gap_frames": Int(ge=1),
        "exit_line_m": _POS,
        "exit_speed_mps": _POS,
        "restart_line_m": _POS,
        "goal_area_depth_m": _POS,
        "goal_area_half_width_m": _POS,
        "pitch_w_m": _POS,
        "air_speed_mps": _POS,
        "air_max_frames": Int(ge=1),
        "smooth_frames": Int(ge=1),
        "rest_speed_mps": _POS,
        "rest_min_frames": Int(ge=1),
        "fast_speed_mps": _POS,
        "sp_prox_tau_frames": _POS,
        "sp_after_tau_frames": _POS,
        "goal_window_frames": Int(ge=1),
        "weights": {k: _ANY for k in DEAD_WEIGHT_KEYS},
    },
    "calibrated_file": Str(nullable=True),
}

_STATE_KEYS = ("H", "A", "D")

DECODER_SCHEMA = {
    "lam_team": _POS,
    "lam_dead": _POS,
    "mean_duration_s": {s: _POS for s in _STATE_KEYS},
    "exit_split": {s: {t: _UNIT for t in _STATE_KEYS if t != s} for s in _STATE_KEYS},
    "min_dwell_frames": {s: Int(ge=1) for s in _STATE_KEYS},
}

POSTPROCESS_SCHEMA = {
    "midpoint_neutral_switch": Bool(),
    "boundary_shift_file": Str(nullable=True),
    "d_bridge_frames": Int(ge=0),
    "d_cross_frames": Int(ge=0),
    "min_spell_frames": Int(ge=0),
}

TOP_COMMON = {
    "name": Str(),
    "pipeline": Str(PIPELINES),
    "ball_source": Str(BALL_SOURCES),
}

PIPELINE_SCHEMAS = {
    "baseline": {**TOP_COMMON, "baseline": BASELINE_SCHEMA},
    "hmm": {**TOP_COMMON, "team": TEAM_SCHEMA, "dead": DEAD_SCHEMA,
            "decoder": DECODER_SCHEMA, "postprocess": POSTPROCESS_SCHEMA},
}


BROADCAST_SCHEMA = {
    "tactical_n_slots": Int(ge=1),
    "broadcast_n_slots": Int(ge=1),
    "source_fps": Int(ge=1),
    "coarse_fps": Int(ge=1),
    "block_seconds": _POS,
    "block_frames": Int(ge=1),
    "min_visible_players": Int(ge=1),
    "refine_search_radius_frames": Int(ge=1),
    "align_max_cost_m": _POS,
    "second_best_min_offset_frames": Int(ge=0),
    "second_best_cost_ratio": Float(ge=1.0),
    "segment_max_offset_diff_frames": Int(ge=0),
    "segment_min_blocks": Int(ge=1),
    "min_coverage_ratio": _UNIT,
    "max_player_median_cost_m": _POS,
    "min_ball_recovery_ratio": _UNIT,
    "max_ball_median_dist_m": _POS,
}


# -- cross-field rules --------------------------------------------------------------------------

def _check_hmm(cfg: dict) -> None:
    for section in ("team", "dead"):
        if cfg[section]["kind"] in KINDS_WITH_FILE and cfg[section]["calibrated_file"] is None:
            raise ConfigError("{}.calibrated_file is required when {}.kind is {!r}".format(
                section, section, cfg[section]["kind"]))

    dec = cfg["decoder"]
    for state, row in dec["exit_split"].items():
        total = math.fsum(row.values())
        if abs(total - 1.0) > EXIT_SPLIT_TOL:
            raise ConfigError("decoder.exit_split.{} must sum to 1, got {!r}".format(state, total))

    fps = games.GAMES[games.GAME]["fps"]
    for state in _STATE_KEYS:
        mean_frames = dec["mean_duration_s"][state] * fps
        dwell = dec["min_dwell_frames"][state]
        if not mean_frames > dwell:
            raise ConfigError(
                "decoder.mean_duration_s.{} ({} s = {} frames) must exceed "
                "decoder.min_dwell_frames.{} ({})".format(
                    state, dec["mean_duration_s"][state], mean_frames, state, dwell))


def validate_config(raw: dict, expected_name: str | None = None) -> dict:
    """Validate a parsed config and return a normalised copy."""
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping, got {!r}".format(type(raw).__name__))
    if "pipeline" not in raw:
        raise ConfigError("<root>: missing keys ['pipeline']")
    pipeline = Str(PIPELINES).check(raw.get("pipeline"), "pipeline")
    cfg = _check_mapping(PIPELINE_SCHEMAS[pipeline], raw, "")
    if expected_name is not None and cfg["name"] != expected_name:
        raise ConfigError("name {!r} must match the file name {!r}".format(
            cfg["name"], expected_name))
    if pipeline == "hmm":
        _check_hmm(cfg)
    return cfg


def validate_broadcast_config(raw: dict) -> dict:
    """Validate a broadcast-feasibility config (F4-05) and return a normalised copy."""
    if not isinstance(raw, dict):
        raise ConfigError("config must be a mapping, got {!r}".format(type(raw).__name__))
    cfg = _check_mapping(BROADCAST_SCHEMA, raw, "")
    if cfg["source_fps"] != games.GAMES[games.GAME]["fps"]:
        raise ConfigError("source_fps ({}) must match the game fps ({})".format(
            cfg["source_fps"], games.GAMES[games.GAME]["fps"]))
    if cfg["source_fps"] % cfg["coarse_fps"] != 0:
        raise ConfigError("coarse_fps ({}) must divide source_fps ({})".format(
            cfg["coarse_fps"], cfg["source_fps"]))
    expected_frames = cfg["block_seconds"] * cfg["source_fps"]
    if abs(cfg["block_frames"] - expected_frames) > EXIT_SPLIT_TOL:
        raise ConfigError("block_frames ({}) must equal block_seconds * source_fps ({})".format(
            cfg["block_frames"], expected_frames))
    return cfg


class _StrictLoader(yaml.SafeLoader):
    """SafeLoader that rejects duplicated keys instead of keeping the last one."""


def _construct_mapping(loader, node, deep=False):
    keys = set()
    for key_node, _ in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in keys:
            raise ConfigError("duplicated key {!r} (line {})".format(
                key, key_node.start_mark.line + 1))
        keys.add(key)
    return loader.construct_mapping(node, deep=deep)


_StrictLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_mapping)


def load_config(path) -> dict:
    """Read and validate a strict pipeline or external degradation config."""
    path = Path(path)
    with open(path, encoding="utf-8") as handle:
        raw = yaml.load(handle, Loader=_StrictLoader)  # noqa: S506 - strict SafeLoader subclass
    if path.name.startswith("degrade_"):
        from .external.degrade import validate_degrade_config
        try:
            cfg = validate_degrade_config(raw)
            if cfg["name"] != path.stem.removeprefix("degrade_"):
                raise ConfigError("name {!r} must match the file name {!r}".format(
                    cfg["name"], path.stem.removeprefix("degrade_")))
            return cfg
        except (ConfigError, ValueError) as exc:
            raise ConfigError("{}: {}".format(path, exc)) from None
    if path.name.startswith("broadcast"):
        try:
            return validate_broadcast_config(raw)
        except ConfigError as exc:
            raise ConfigError("{}: {}".format(path, exc)) from None
    try:
        return validate_config(raw, expected_name=path.stem)
    except ConfigError as exc:
        raise ConfigError("{}: {}".format(path, exc)) from None


def config_hash(cfg: dict) -> str:
    """C7: SHA-256 of the canonical JSON dump of ``cfg``."""
    payload = json.dumps(cfg, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = ["BALL_SOURCES", "ConfigError", "DEAD_KIND_NAMES", "PIPELINES", "TEAM_KIND_NAMES",
           "config_hash", "load_config", "validate_broadcast_config", "validate_config"]