"""Repository hygiene (task F5-03): no match data, no notebook outputs, no large files in git.

The checks read the git index (``git ls-files``), so they see exactly what a commit would publish,
staged files included. They read file names, sizes and notebook JSON only; they never open a data
file. Skipped when git or the repository is not available.
"""
from __future__ import annotations

import json
import shutil
import subprocess
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath

import pytest

REPO = Path(__file__).resolve().parents[1]

DATA_EXTENSIONS = (".parquet", ".npy", ".npz", ".pkl", ".mp4", ".xml")
FIXTURES_PREFIX = "tests/fixtures/"
DATA_PREFIX = "data/"
DATA_ALLOWED = ("data/README.md",)
MAX_BYTES = 1_000_000                       # 1 MB
MODEL_MAX_BYTES = 5_000_000                 # 5 MB, for models/**/*.joblib
# (pattern, size limit or None for no limit); patterns match whole paths, "*" never crosses "/".
SIZE_ALLOWLIST = (
    ("evaluation/gt/*", None),
    ("evaluation/results/baseline_*.csv", None),
    ("models/**/*.joblib", MODEL_MAX_BYTES),
)
GITLINK_MODE = "160000"                     # submodule entry: no blob to measure

# Paths .gitignore must ignore, and paths it must not (task F5-03, item 1).
MUST_IGNORE = ("data/master.parquet", "data/raw/tracking.csv", "outputs/v1.csv",
               "x.parquet", "sub/x.npy", "x.npz", "x.pkl", "x.mp4")
MUST_NOT_IGNORE = ("data/README.md", "configs/v1.yaml", "configs/calib/team.yaml",
                   "reports/external/v1__default.md")


# -- reading the index --------------------------------------------------------------------------

def _git(repo: Path, *args: str, stdin: bytes | None = None) -> bytes:
    return subprocess.run(["git", "-C", str(repo), *args], input=stdin, check=True,
                          capture_output=True).stdout


def _git_available(repo: Path) -> bool:
    if shutil.which("git") is None:
        return False
    try:
        return _git(repo, "rev-parse", "--is-inside-work-tree").strip() == b"true"
    except (subprocess.CalledProcessError, OSError):
        return False


def index_entries(repo: Path) -> list[dict]:
    """Every file in the git index: ``{"path", "mode", "sha", "size"}`` (size in bytes)."""
    raw = _git(repo, "ls-files", "--stage", "-z")
    entries = []
    for record in raw.split(b"\0"):
        if not record:
            continue
        meta, path = record.split(b"\t", 1)
        mode, sha, _stage = meta.decode().split()
        entries.append({"path": path.decode("utf-8", "surrogateescape"), "mode": mode,
                        "sha": sha, "size": None})
    blobs = [e for e in entries if e["mode"] != GITLINK_MODE]
    if blobs:
        query = "".join(e["sha"] + "\n" for e in blobs).encode()
        sizes = _git(repo, "cat-file", "--batch-check=%(objectsize)", stdin=query).split()
        for entry, size in zip(blobs, sizes, strict=True):
            entry["size"] = int(size)
    return entries


def read_blob(repo: Path, sha: str) -> bytes:
    return _git(repo, "cat-file", "blob", sha)


# -- rules (pure functions of a path, a size, a notebook) ---------------------------------------

def _match(path: str, pattern: str) -> bool:
    """Whole-path glob where ``*`` stays inside one directory and ``**`` spans any number."""
    if "**" not in pattern:
        p_parts, g_parts = path.split("/"), pattern.split("/")
        return len(p_parts) == len(g_parts) and all(
            fnmatchcase(p, g) for p, g in zip(p_parts, g_parts))
    head, tail = pattern.split("/**/", 1)
    if not path.startswith(head + "/"):
        return False
    rest = PurePosixPath(path[len(head) + 1:])
    return fnmatchcase(rest.name, tail) and "/" not in tail


def is_data_file(path: str) -> bool:
    return PurePosixPath(path).suffix.lower() in DATA_EXTENSIONS


def data_file_violation(path: str) -> str | None:
    if is_data_file(path) and not path.startswith(FIXTURES_PREFIX):
        return "{}: data file outside {}".format(path, FIXTURES_PREFIX)
    return None


def data_dir_violation(path: str) -> str | None:
    if path.startswith(DATA_PREFIX) and path not in DATA_ALLOWED:
        return "{}: only {} may be committed under {}".format(path, DATA_ALLOWED, DATA_PREFIX)
    return None


def size_violation(path: str, size: int | None) -> str | None:
    if size is None:
        return None
    for pattern, limit in SIZE_ALLOWLIST:
        if _match(path, pattern):
            if limit is not None and size > limit:
                return "{}: {} bytes > {} allowed for {}".format(path, size, limit, pattern)
            return None
    if size > MAX_BYTES:
        return "{}: {} bytes > {} (not in the size allowlist)".format(path, size, MAX_BYTES)
    return None


def notebook_violation(path: str, content: bytes) -> str | None:
    """A committed notebook must have no outputs in any cell."""
    if PurePosixPath(path).suffix.lower() != ".ipynb":
        return None
    try:
        nb = json.loads(content.decode("utf-8"))
        with_outputs = sum(1 for cell in nb.get("cells", []) if cell.get("outputs"))
    except (UnicodeDecodeError, ValueError, AttributeError, TypeError):
        return "{}: not a valid notebook JSON".format(path)
    if with_outputs:
        return "{}: {} cells with outputs (clear them before committing)".format(
            path, with_outputs)
    return None


def find_violations(repo: Path) -> list[str]:
    """Every hygiene violation in the index of ``repo``, one message per problem."""
    found = []
    for entry in index_entries(repo):
        path, size = entry["path"], entry["size"]
        for message in (data_file_violation(path), data_dir_violation(path),
                        size_violation(path, size)):
            if message:
                found.append(message)
        if PurePosixPath(path).suffix.lower() == ".ipynb" and entry["mode"] != GITLINK_MODE:
            message = notebook_violation(path, read_blob(repo, entry["sha"]))
            if message:
                found.append(message)
    return found


# -- the repository itself -----------------------------------------------------------------------

needs_git = pytest.mark.skipif(not _git_available(REPO), reason="git or the repository is "
                                                                 "not available")


@needs_git
def test_repository_is_clean():
    violations = find_violations(REPO)
    assert not violations, "repository hygiene:\n" + "\n".join(violations)


@needs_git
@pytest.mark.parametrize("path", MUST_IGNORE)
def test_gitignore_ignores_data_and_outputs(path):
    proc = subprocess.run(["git", "-C", str(REPO), "check-ignore", "--no-index", "-q", path])
    assert proc.returncode == 0, "{} must be ignored by .gitignore".format(path)


@needs_git
@pytest.mark.parametrize("path", MUST_NOT_IGNORE)
def test_gitignore_keeps_configs_reports_and_data_readme(path):
    proc = subprocess.run(["git", "-C", str(REPO), "check-ignore", "--no-index", "-q", path])
    assert proc.returncode == 1, "{} must not be ignored by .gitignore".format(path)


# -- the scanner on a throwaway repository (proves each rule fires) ------------------------------

@pytest.fixture
def scratch_repo(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("git is not available")
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    (repo / "README.md").write_text("ok\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    return repo


def _add(repo: Path, path: str, content: bytes) -> None:
    target = repo / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)
    _git(repo, "add", "-f", path)          # -f: the point is to bypass .gitignore


def test_scanner_clean_repo(scratch_repo):
    _add(scratch_repo, "data/README.md", b"readme\n")
    _add(scratch_repo, "tests/fixtures/mini.parquet", b"PAR1")
    _add(scratch_repo, "tests/fixtures/sub/mini.xml", b"<a/>")
    assert find_violations(scratch_repo) == []


def test_scanner_flags_a_staged_parquet(scratch_repo):
    _add(scratch_repo, "master_features.parquet", b"PAR1")
    assert find_violations(scratch_repo) == [
        "master_features.parquet: data file outside tests/fixtures/"]


@pytest.mark.parametrize("ext", DATA_EXTENSIONS + (".PARQUET",))
def test_scanner_flags_every_data_extension(scratch_repo, ext):
    _add(scratch_repo, "modeling/x" + ext, b"0")
    assert find_violations(scratch_repo) == ["modeling/x{}: data file outside "
                                             "tests/fixtures/".format(ext)]


def test_scanner_flags_files_under_data(scratch_repo):
    _add(scratch_repo, "data/processed/notes.txt", b"0")
    _add(scratch_repo, "data/sub/README.md", b"0")
    found = find_violations(scratch_repo)
    assert len(found) == 2 and all("only" in m for m in found)


def test_scanner_flags_notebook_outputs(scratch_repo):
    clean = {"cells": [{"cell_type": "code", "outputs": [], "source": []},
                       {"cell_type": "markdown", "source": []}], "nbformat": 4}
    dirty = {"cells": [{"cell_type": "code", "source": [],
                        "outputs": [{"output_type": "stream", "text": ["x"]}]}], "nbformat": 4}
    _add(scratch_repo, "clean.ipynb", json.dumps(clean).encode())
    _add(scratch_repo, "nb/dirty.ipynb", json.dumps(dirty).encode())
    _add(scratch_repo, "broken.ipynb", b"{not json")
    assert find_violations(scratch_repo) == [
        "broken.ipynb: not a valid notebook JSON",
        "nb/dirty.ipynb: 1 cells with outputs (clear them before committing)"]


def test_scanner_reads_the_index_not_the_working_tree(scratch_repo):
    dirty = {"cells": [{"cell_type": "code", "source": [], "outputs": [{"x": 1}]}]}
    _add(scratch_repo, "a.ipynb", json.dumps(dirty).encode())
    (scratch_repo / "a.ipynb").write_text(json.dumps({"cells": []}), encoding="utf-8")
    assert len(find_violations(scratch_repo)) == 1         # the staged version has outputs


def test_scanner_size_limits(scratch_repo):
    big = b"0" * (MAX_BYTES + 1)
    _add(scratch_repo, "ok_limit.csv", b"0" * MAX_BYTES)
    _add(scratch_repo, "too_big.csv", big)
    _add(scratch_repo, "evaluation/gt/X_possession_gt_public.csv", big)
    _add(scratch_repo, "evaluation/results/baseline_raw.csv", big)
    _add(scratch_repo, "evaluation/results/other.csv", big)
    _add(scratch_repo, "evaluation/gt/sub/deep.csv", big)        # "*" does not cross "/"
    _add(scratch_repo, "models/team_gbdt/all.joblib", b"0" * MODEL_MAX_BYTES)
    _add(scratch_repo, "models/dead_gbdt/v2/all.joblib", b"0" * (MODEL_MAX_BYTES + 1))
    found = find_violations(scratch_repo)
    flagged = sorted(m.split(":")[0] for m in found)
    assert flagged == ["evaluation/gt/sub/deep.csv", "evaluation/results/other.csv",
                       "models/dead_gbdt/v2/all.joblib", "too_big.csv"]


@pytest.mark.parametrize("path, pattern, expected", [
    ("evaluation/gt/a.csv", "evaluation/gt/*", True),
    ("evaluation/gt/a/b.csv", "evaluation/gt/*", False),
    ("evaluation/results/baseline_x.csv", "evaluation/results/baseline_*.csv", True),
    ("evaluation/results/baseline_x.csv.bak", "evaluation/results/baseline_*.csv", False),
    ("models/a.joblib", "models/**/*.joblib", True),
    ("models/a/b/c.joblib", "models/**/*.joblib", True),
    ("models/a/b/c.pkl", "models/**/*.joblib", False),
    ("other/models/a.joblib", "models/**/*.joblib", False),
])
def test_glob_matching(path, pattern, expected):
    assert _match(path, pattern) is expected
