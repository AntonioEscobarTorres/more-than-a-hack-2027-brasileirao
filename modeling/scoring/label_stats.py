"""IDSSE label statistics: durations, exit_split and plausibility bands (task F2-03).

Reads the seven public ``.npz`` arrays under ``$MTAH_EXT_DIR/arrays``, aggregates
per-half segments (X frames are skipped and never form a segment), and writes:

- ``configs/decoder_idsse.yaml`` — decoder section with IDSSE durations / exit_split and
  Appendix A.4 ``lam_*`` / ``min_dwell_frames``;
- ``configs/plausibility.yaml`` — min/max over the 14 halves of possessions/min,
  mean possession duration and D fraction;
- ``reports/idsse_label_stats.md`` — short human-readable summary.

Match (BAR-ATH) data is never read.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import yaml

from modeling import paths
from modeling.contracts import LABEL_CODES, STATES
from modeling.external.files import IDSSE_MATCHES, ExternalMatch, arrays_path

# Appendix A.4 (25 fps). Kept here as the fixed non-IDSSE decoder knobs for F2-03.
FPS = 25.0
A4_LAM_TEAM = 0.2
A4_LAM_DEAD = 0.2
A4_MIN_DWELL = {"H": 12, "A": 12, "D": 25}

_STATE_INDEX = {name: i for i, name in enumerate(STATES)}
_CODE_TO_STATE = {LABEL_CODES[s]: s for s in STATES}
_X = LABEL_CODES["X"]


@dataclass
class Segment:
    """One contiguous run of a non-X label inside a single half."""

    state: str
    start: int
    stop: int  # exclusive

    @property
    def length(self) -> int:
        return self.stop - self.start


@dataclass
class HalfStats:
    match_id: str
    half: int
    n_frames: int
    segments: list[Segment]
    mean_duration_s: dict  # H, A, D, HA
    transition_counts: np.ndarray  # (3, 3) int64
    possessions_per_min: float
    mean_possession_duration_s: float
    d_fraction: float


@dataclass
class IdsseLabelStats:
    halves: list[HalfStats]
    mean_duration_s: dict  # pooled over all segments
    exit_split: dict  # {src: {dst: p}}
    transition_counts: np.ndarray
    plausibility: dict  # min/max over halves
    fps: float = FPS


def segments_from_labels(labels: np.ndarray, half: np.ndarray, half_id: int) -> list[Segment]:
    """Build segments for one half; X frames are ignored and never form a segment.

    An X inside the half breaks the current run (the next non-X starts a new segment
    even if the label matches the one before the X).
    """
    labels = np.asarray(labels)
    half = np.asarray(half)
    if labels.shape != half.shape:
        raise ValueError("labels and half must share shape, got {} vs {}".format(
            labels.shape, half.shape))
    idx = np.flatnonzero(half == half_id)
    if idx.size == 0:
        return []

    segments: list[Segment] = []
    run_state: str | None = None
    run_start: int | None = None
    run_last: int | None = None

    def close():
        nonlocal run_state, run_start, run_last
        if run_state is not None and run_start is not None and run_last is not None:
            segments.append(Segment(run_state, run_start, run_last + 1))
        run_state = None
        run_start = None
        run_last = None

    for t in idx:
        code = int(labels[t])
        if code == _X:
            close()
            continue
        state = _CODE_TO_STATE.get(code)
        if state is None:
            raise ValueError("unknown label code {} at frame {}".format(code, int(t)))
        if run_state is None:
            run_state = state
            run_start = int(t)
            run_last = int(t)
        elif state == run_state and int(t) == run_last + 1:
            run_last = int(t)
        else:
            close()
            run_state = state
            run_start = int(t)
            run_last = int(t)
    close()
    return segments


def _mean_duration_by_state(segments: Sequence[Segment], fps: float) -> dict:
    buckets = {s: [] for s in STATES}
    buckets["HA"] = []
    for seg in segments:
        dur = seg.length / fps
        buckets[seg.state].append(dur)
        if seg.state in ("H", "A"):
            buckets["HA"].append(dur)
    out = {}
    for key, vals in buckets.items():
        out[key] = float(np.mean(vals)) if vals else float("nan")
    return out


def _transition_counts(segments: Sequence[Segment]) -> np.ndarray:
    counts = np.zeros((3, 3), dtype=np.int64)
    for prev, cur in zip(segments, segments[1:]):
        if prev.state == cur.state:
            continue
        counts[_STATE_INDEX[prev.state], _STATE_INDEX[cur.state]] += 1
    return counts


def _exit_split_from_counts(counts: np.ndarray) -> dict:
    out = {}
    for s, s_name in enumerate(STATES):
        row = counts[s].astype(np.float64)
        row[s] = 0.0
        total = float(row.sum())
        dest = {}
        for sp, sp_name in enumerate(STATES):
            if sp == s:
                continue
            dest[sp_name] = float(row[sp] / total) if total > 0.0 else 0.0
        # If a row is empty, fall back to uniform over the two exits (A.4 prior).
        if total <= 0.0:
            for sp_name in dest:
                dest[sp_name] = 0.5
        out[s_name] = dest
    return out


def half_stats(
    labels: np.ndarray,
    half: np.ndarray,
    half_id: int,
    *,
    match_id: str = "",
    fps: float = FPS,
) -> HalfStats:
    """Aggregate segment / transition / possession stats for one half."""
    fps = float(fps)
    mask = half == half_id
    n_frames = int(np.count_nonzero(mask))
    segs = segments_from_labels(labels, half, half_id)
    mean_dur = _mean_duration_by_state(segs, fps)
    counts = _transition_counts(segs)

    possession_segs = [s for s in segs if s.state in ("H", "A")]
    minutes = n_frames / (fps * 60.0) if n_frames else 0.0
    poss_per_min = (len(possession_segs) / minutes) if minutes > 0.0 else float("nan")
    mean_poss = (
        float(np.mean([s.length / fps for s in possession_segs]))
        if possession_segs else float("nan")
    )
    # D fraction over non-X frames in the half (X is ignored).
    in_half = np.asarray(labels)[mask]
    usable = in_half != _X
    n_usable = int(np.count_nonzero(usable))
    d_frac = (
        float(np.count_nonzero(in_half[usable] == LABEL_CODES["D"]) / n_usable)
        if n_usable else float("nan")
    )
    return HalfStats(
        match_id=match_id,
        half=int(half_id),
        n_frames=n_frames,
        segments=segs,
        mean_duration_s=mean_dur,
        transition_counts=counts,
        possessions_per_min=poss_per_min,
        mean_possession_duration_s=mean_poss,
        d_fraction=d_frac,
    )


def aggregate_halves(halves: Sequence[HalfStats], *, fps: float = FPS) -> IdsseLabelStats:
    """Pool segments / transitions and compute plausibility min/max over halves."""
    all_segs: list[Segment] = []
    counts = np.zeros((3, 3), dtype=np.int64)
    for h in halves:
        all_segs.extend(h.segments)
        counts += h.transition_counts

    mean_duration_s = _mean_duration_by_state(all_segs, fps)
    # Drop the HA helper from the decoder-facing dict; keep it for the report via all_segs.
    exit_split = _exit_split_from_counts(counts)

    def _band(values: Iterable[float]) -> dict:
        arr = np.asarray(list(values), dtype=np.float64)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return {"min": float("nan"), "max": float("nan")}
        return {"min": float(arr.min()), "max": float(arr.max())}

    plausibility = {
        "possessions_per_min": _band(h.possessions_per_min for h in halves),
        "mean_possession_duration_s": _band(h.mean_possession_duration_s for h in halves),
        "d_fraction": _band(h.d_fraction for h in halves),
    }
    return IdsseLabelStats(
        halves=list(halves),
        mean_duration_s={s: mean_duration_s[s] for s in STATES},
        exit_split=exit_split,
        transition_counts=counts,
        plausibility=plausibility,
        fps=float(fps),
    )


def stats_from_match(match: ExternalMatch, *, fps: float = FPS) -> list[HalfStats]:
    return [
        half_stats(match.labels, match.half, h, match_id=match.match_id, fps=fps)
        for h in (1, 2)
    ]


def labels_and_half_from_xml_map(
    xml_labels: Mapping[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild C0 ``labels`` / ``half`` vectors from ``parse_ball_labels`` output.

    Frames are ordered by ``frame_id`` within each half; the artificial C9 gap is
    inserted between halves so the layout matches ``ExternalMatch``.
    """
    from modeling.external.files import GAP_FRAMES

    halves_frames: dict[int, list[int]] = {1: [], 2: []}
    for (h, frame_id), _code in xml_labels.items():
        if h not in halves_frames:
            raise ValueError("unexpected half id {} in XML labels".format(h))
        halves_frames[h].append(int(frame_id))
    for h in (1, 2):
        if not halves_frames[h]:
            raise ValueError("XML labels missing half {}".format(h))
        halves_frames[h] = sorted(set(halves_frames[h]))

    parts_labels = []
    parts_half = []
    for h in (1, 2):
        frame_ids = halves_frames[h]
        # Dense time axis in sorted frame_id order (IDSSE reports 0 missing frames;
        # we do not require the integer IDs themselves to be gap-free).
        codes = np.asarray([xml_labels[(h, fid)] for fid in frame_ids], dtype=np.int8)
        parts_labels.append(codes)
        parts_half.append(np.full(codes.shape[0], h, dtype=np.int8))
        if h == 1:
            parts_labels.append(np.full(GAP_FRAMES, _X, dtype=np.int8))
            parts_half.append(np.zeros(GAP_FRAMES, dtype=np.int8))
    return np.concatenate(parts_labels), np.concatenate(parts_half)


def stats_from_xml_labels(
    xml_labels: Mapping[tuple[int, int], int],
    *,
    match_id: str = "",
    fps: float = FPS,
) -> list[HalfStats]:
    labels, half = labels_and_half_from_xml_map(xml_labels)
    return [
        half_stats(labels, half, h, match_id=match_id, fps=fps)
        for h in (1, 2)
    ]


def compute_idsse_stats(
    match_ids: Sequence[str] | None = None,
    *,
    fps: float = FPS,
    source: str = "npz",
) -> IdsseLabelStats:
    """Aggregate label statistics from IDSSE ``.npz`` arrays or tracking XMLs.

    ``source="npz"`` (default) loads ``$MTAH_EXT_DIR/arrays/{match}.npz``.
    ``source="xml"`` downloads/parses only the public tracking XML ball labels — enough
    for F2-03 and much cheaper than a full C9 conversion when arrays are missing.
    """
    ids = tuple(match_ids) if match_ids is not None else IDSSE_MATCHES
    source = str(source).lower()
    if source not in ("npz", "xml"):
        raise ValueError("source must be 'npz' or 'xml', got {!r}".format(source))

    halves: list[HalfStats] = []
    if source == "npz":
        for match_id in ids:
            path = arrays_path(match_id)
            if not path.is_file():
                raise FileNotFoundError(
                    "IDSSE array not found: {} (run "
                    "`python -m modeling.external.idsse --matches all --delete-raw`, "
                    "or pass --source xml)".format(path)
                )
            match = ExternalMatch.load(path)
            halves.extend(stats_from_match(match, fps=fps))
    else:
        from modeling.external.idsse import download, parse_ball_labels

        for match_id in ids:
            _meta, tracking_path = download(match_id)
            xml_labels = parse_ball_labels(tracking_path)
            halves.extend(stats_from_xml_labels(xml_labels, match_id=match_id, fps=fps))

    if len(halves) != 2 * len(ids):
        raise RuntimeError("expected {} halves, got {}".format(2 * len(ids), len(halves)))
    return aggregate_halves(halves, fps=fps)


def decoder_idsse_config(stats: IdsseLabelStats) -> dict:
    """Decoder section: IDSSE durations / exit_split + Appendix A.4 lam / min_dwell."""
    return {
        "lam_team": A4_LAM_TEAM,
        "lam_dead": A4_LAM_DEAD,
        "mean_duration_s": {
            s: float(stats.mean_duration_s[s]) for s in STATES
        },
        "exit_split": {
            s: {t: float(stats.exit_split[s][t]) for t in STATES if t != s}
            for s in STATES
        },
        "min_dwell_frames": dict(A4_MIN_DWELL),
    }


def _round_float(x: float, ndigits: int = 6) -> float:
    if not np.isfinite(x):
        return float("nan")
    return float(round(float(x), ndigits))


def write_decoder_idsse_yaml(stats: IdsseLabelStats, path: Path | None = None) -> Path:
    path = Path(path) if path is not None else paths.repo_root() / "configs" / "decoder_idsse.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    dec = decoder_idsse_config(stats)
    # Stable, human-editable floats.
    dec["mean_duration_s"] = {k: _round_float(v, 4) for k, v in dec["mean_duration_s"].items()}
    dec["exit_split"] = {
        s: {t: _round_float(p, 6) for t, p in row.items()}
        for s, row in dec["exit_split"].items()
    }
    payload = {
        "decoder": dec,
    }
    text = (
        "# Generated by modeling.scoring.label_stats (F2-03).\n"
        "# Durations and exit_split from IDSSE labels; lam_* and min_dwell from Appendix A.4.\n"
        + yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    )
    path.write_text(text, encoding="utf-8")
    return path


def write_plausibility_yaml(stats: IdsseLabelStats, path: Path | None = None) -> Path:
    path = Path(path) if path is not None else paths.repo_root() / "configs" / "plausibility.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    band = {
        key: {
            "min": _round_float(stats.plausibility[key]["min"], 6),
            "max": _round_float(stats.plausibility[key]["max"], 6),
        }
        for key in ("possessions_per_min", "mean_possession_duration_s", "d_fraction")
    }
    text = (
        "# Generated by modeling.scoring.label_stats (F2-03).\n"
        "# Min/max over the 14 IDSSE halves (7 matches × 2).\n"
        + yaml.safe_dump(band, sort_keys=False, default_flow_style=False)
    )
    path.write_text(text, encoding="utf-8")
    return path


def write_report(stats: IdsseLabelStats, path: Path | None = None) -> Path:
    path = Path(path) if path is not None else paths.repo_root() / "reports" / "idsse_label_stats.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# IDSSE label statistics",
        "",
        "Generated by `modeling.scoring.label_stats` (F2-03) from the public IDSSE matches.",
        "X frames are ignored and never counted as segments.",
        "",
        "## Pooled mean segment duration (s)",
        "",
        "| State | Mean duration (s) |",
        "|---|---:|",
    ]
    for s in STATES:
        lines.append("| {} | {:.4f} |".format(s, stats.mean_duration_s[s]))
    ha = _mean_duration_by_state(
        [seg for h in stats.halves for seg in h.segments], stats.fps
    )["HA"]
    lines.append("| H+A (possessions) | {:.4f} |".format(ha))
    lines.extend([
        "",
        "## Exit split (row-normalised transition counts)",
        "",
        "| From \\ To | H | A | D |",
        "|---|---:|---:|---:|",
    ])
    for s in STATES:
        row = []
        for t in STATES:
            if t == s:
                row.append("—")
            else:
                row.append("{:.4f}".format(stats.exit_split[s][t]))
        lines.append("| {} | {} |".format(s, " | ".join(row)))
    lines.extend([
        "",
        "## Plausibility bands (min / max over 14 halves)",
        "",
        "| Metric | Min | Max |",
        "|---|---:|---:|",
    ])
    for key, label in (
        ("possessions_per_min", "Possessions / min"),
        ("mean_possession_duration_s", "Mean possession duration (s)"),
        ("d_fraction", "D fraction"),
    ):
        band = stats.plausibility[key]
        lines.append("| {} | {:.4f} | {:.4f} |".format(label, band["min"], band["max"]))
    lines.extend(["", "## Per-half summary", "",
                  "| Match | Half | Frames | Poss/min | Mean poss (s) | D frac |",
                  "|---|---:|---:|---:|---:|---:|"])
    for h in stats.halves:
        lines.append(
            "| {} | {} | {} | {:.3f} | {:.3f} | {:.3f} |".format(
                h.match_id, h.half, h.n_frames,
                h.possessions_per_min, h.mean_possession_duration_s, h.d_fraction,
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def write_all(stats: IdsseLabelStats, *, root: Path | None = None) -> dict:
    root = Path(root) if root is not None else paths.repo_root()
    return {
        "decoder_idsse": write_decoder_idsse_yaml(
            stats, root / "configs" / "decoder_idsse.yaml"),
        "plausibility": write_plausibility_yaml(
            stats, root / "configs" / "plausibility.yaml"),
        "report": write_report(stats, root / "reports" / "idsse_label_stats.md"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--matches", default="all",
        help="all or comma-separated IDSSE match IDs (default: all)",
    )
    parser.add_argument(
        "--source", choices=("npz", "xml"), default="npz",
        help="read C9 arrays (npz) or ball labels from tracking XML (xml)",
    )
    parser.add_argument(
        "--output-root", default=None,
        help="repository root for configs/ and reports/ (default: repo root)",
    )
    args = parser.parse_args(argv)
    match_ids = (
        IDSSE_MATCHES if args.matches == "all"
        else tuple(x.strip() for x in args.matches.split(",") if x.strip())
    )
    stats = compute_idsse_stats(match_ids, source=args.source)
    written = write_all(stats, root=Path(args.output_root) if args.output_root else None)
    for kind, path in written.items():
        print("{}: {}".format(kind, path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
