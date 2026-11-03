"""Domain lint — the policies upstream's `ale lint` cannot know about.

`ale lint` checks what the engine owns: `task.toml` parses against the strict schema, the
stage entries exist, every base image names an exact version. It has no opinion about whether
a metric can be compared across contributors' hardware, whether an anchor's provenance is
recorded, or whether a practice grader quietly imports the anchor. Those are the robotics
domain's rules, and without this file they erode task by task.

Seven batteries:

1. **Hardware-invariant metrics.** `tests/anchor.json` names the metric (`metric`) and its
   `direction` (higher | lower). A metric naming a wall-clock quantity measures the machine,
   not the method — two contributors doing identical work would score differently.
2. **Positive metric scale.** A non-positive anchor inverts the ratio, so a better result would
   score lower, silently and permanently.
3. **Visible-surface isolation.** Nothing in `environment/setup/payload/` (the agent-visible
   channel) may read the anchor or the grader, and the practice grader reports the raw metric
   only — never a ratio or a score.
4. **Image hygiene.** No privileged flags, and no per-task image (`environment/Dockerfile`, or
   a separate verifier's `tests/Dockerfile`) may bake in the hidden surface (`tests/`, the
   anchor, `solution/`) or a task's answers.
5. **Anchor provenance.** Every anchor records where its number came from, and the number is
   never a paper's printed figure — the reference implementation's own measured value only.
6. **Completeness.** `task.toml` carries `schema_version = "1.4+ale.1"` and a `[task].name`
   of the form `org/slug`, and the folder holds `instruction.md`, an image to run in
   (`environment/Dockerfile`, or a prebuilt `[environment].docker_image`), `tests/test.sh`,
   `solution/solve.sh` and `tests/anchor.json` — cheap static forms of what the engine would
   reject anyway, caught before a container is ever started.
7. **Identity, platform and placement.** The org part of `[task].name` is the benchmark's,
   `grounded-intelligence-initiative` (`TASK_ORG`); `metadata.platform` (`[metadata].platform`
   in `task.toml`) is one of the benchmark's platforms (`tasks/platforms.json`, owned by this
   repository); and a landed task sits exactly at `tasks/<platform>/<slug>/`: the folder name
   is the slug (the part of `[task].name` after `/`), the parent folder is
   `metadata.platform`. A contribution's pull request puts the folder there directly, so this
   is the check that keeps the tree and the registry in agreement.

A `task.yaml` with no `task.toml` in its folder or above it is a task in the retired core/v1
form. The engine finds tasks by `task.toml` alone and would pass over such a folder without a
word, so the lint reports it.

Run: `uv run python scripts/lint_domain.py [paths...]` (default: this repository)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
#: The repository root holds the task collections side by side (flat layout).
DEFAULT_DOMAIN = REPO_ROOT
#: The collections the lint walks. Never the repository root itself: a bare rglob would
#: descend into vendor/ale and the venvs.
COLLECTION_DIRS = ("tasks", "templates")
#: The slug alphabet a folder name and `metadata.platform` must share (they are path
#: segments of the landing directory).
_SEGMENT = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
#: A task id, `org/slug`, each part in the slug alphabet (the engine's `TaskId`).
_TASK_ID = re.compile(r"([a-z0-9][a-z0-9_-]*)/([a-z0-9][a-z0-9_-]*)")

#: The manifest the engine reads, and the one schema revision it accepts.
MANIFEST = "task.toml"
SCHEMA_VERSION = "1.4+ale.1"
#: The manifest of the retired core/v1 form (before the Harbor layout).
RETIRED_MANIFEST = "task.yaml"
#: The org part of every task id in this benchmark: `[task].name` is
#: "grounded-intelligence-initiative/<slug>", the GitHub and Hugging Face organization that
#: owns the benchmark.
TASK_ORG = "grounded-intelligence-initiative"

#: What the batteries read, relative to the task folder (Harbor layout).
ANCHOR = "tests/anchor.json"
GRADER_CONFIG = "tests/grader_config.json"
PAYLOAD = "environment/setup/payload"
#: The images a task builds: the solver's, and a separate verifier's (build context tests/).
DOCKERFILES = ("environment/Dockerfile", "tests/Dockerfile")
#: The folders those Dockerfiles sit in, so a finding names the task, not the folder.
_IMAGE_DIRS = frozenset(path.split("/")[0] for path in DOCKERFILES)
#: The files every task needs besides task.toml, an image and the anchor (battery 6).
REQUIRED_FILES = (
    ("instruction.md", "the agent is told nothing"),
    ("tests/test.sh", "nothing scores the artifact"),
    ("solution/solve.sh", "`ale validate` has no reference solution to run"),
)

#: The benchmark's platform vocabulary: the only values `metadata.platform` may take, and so
#: the only folders directly under tasks/. This repository owns it; the registry export copies
#: it into the document it publishes.
PLATFORMS_FILE = REPO_ROOT / "tasks" / "platforms.json"


def platform_keys(path: Path = PLATFORMS_FILE) -> tuple[str, ...]:
    """The registered platform keys, in file order."""
    return tuple(str(entry["key"]) for entry in json.loads(path.read_text(encoding="utf-8")))


def read_manifest(task_dir: Path) -> dict:
    """The task's task.toml as a dict. Raises OSError or ValueError (TOMLDecodeError,
    UnicodeDecodeError) when it cannot be read."""
    with open(task_dir / MANIFEST, "rb") as fh:
        return tomllib.load(fh)


def _table(manifest: dict, key: str) -> dict:
    """A table of the manifest, or an empty one when it is absent or not a table."""
    value = manifest.get(key)
    return value if isinstance(value, dict) else {}


def split_name(name: object) -> tuple[str, str] | None:
    """(org, slug) of a valid `[task].name`, or None when it is not `org/slug`."""
    match = _TASK_ID.fullmatch(name) if isinstance(name, str) else None
    return (match.group(1), match.group(2)) if match else None

# The kit's default ratio cap (shared/robotics_grader/scoring.py) when a task's
# tests/grader_config.json declares no positive `scoring_cap`.
DEFAULT_SCORING_CAP = 1.5


def _scoring_cap(grader_config: Path) -> float:
    """`full_at` is bounded by the task's own cap, read exactly as the kit reads it
    (`tests/verify.py`: `config.get("scoring_cap", 1.5)`), so a task its own grader accepts
    is never refused here over the cap. A malformed config is the engine lint's business;
    here it just falls back."""
    try:
        data = json.loads(grader_config.read_text())
    except (OSError, ValueError):
        return DEFAULT_SCORING_CAP
    cap = data.get("scoring_cap") if isinstance(data, dict) else None
    if isinstance(cap, bool) or not isinstance(cap, (int, float)) or cap <= 0:
        return DEFAULT_SCORING_CAP
    return float(cap)

# A metric whose name carries one of these TOKENS measures the hardware. Matching is on
# whole tokens — the underscore-, dash- or camelCase-separated alphabetic segments of the
# name — never on substrings: `planning_time_ms`, `control_frequency_hz`, `fps`,
# `throughput_fps`, `inferenceLatency` are refused; `rms_error`, `rmse`, `items_total`,
# `timesteps_to_goal` pass (the old substring regex swept up `rms` via `ms` and
# `timesteps` via `time`). `runtime`/`walltime`/`wallclock` are listed explicitly because
# the substring rule used to catch them through `time`.
_HW_TOKENS = frozenset((
    "time", "runtime", "walltime", "wallclock", "latency",
    "ms", "msec", "millis", "milliseconds",
    "fps", "throughput", "frequency", "freq", "hz", "khz", "mhz", "ghz",
))
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_ALPHA_RUN = re.compile(r"[A-Za-z]+")


def hw_metric_token(name: str) -> str | None:
    """The first hardware-dependent token in a metric name, or None when it has none."""
    for token in _ALPHA_RUN.findall(_CAMEL_BOUNDARY.sub("_", name)):
        if token.lower() in _HW_TOKENS:
            return token
    return None

# An anchor `source` that cites a paper's number is refused — unless the sentence is the
# disclaimer we ask for ("not the paper's figure", "never a published figure", ...).
_PAPER_MENTION = re.compile(r"paper|reported|published", re.IGNORECASE)
_PAPER_DISCLAIMER = re.compile(r"(not|never)( the| a| any)? (paper|published|printed|reported)", re.IGNORECASE)

# Applies to Dockerfile RUN/CMD text: a task image that can escalate is not a sandbox.
_FORBIDDEN = (
    (re.compile(r"--privileged", re.IGNORECASE), "privileged mode"),
    (re.compile(r"\bcap[_-]add\b", re.IGNORECASE), "adding Linux capabilities"),
    (re.compile(r"--security-opt", re.IGNORECASE), "overriding security options"),
    (re.compile(r"--network[=\s]+host", re.IGNORECASE), "host networking"),
)

# Applies to COPY/ADD only. An image is environment-only; the hidden surface arrives at verify
# time or not at all, and an image layer is readable by anyone who can run the image. tests/
# and solution/ lie outside the build context (environment/, less setup/), so only a copy
# placed inside it could be brought in. The names of the retired core/v1 form (verify,
# oracle, task.yaml) stay on the list.
_HIDDEN_IN_IMAGE = (
    (re.compile(r"\btests\b", re.IGNORECASE), "the tests stage (grader + anchor)"),
    (re.compile(r"\bverify\b", re.IGNORECASE), "the verify stage (grader + anchor)"),
    (re.compile(r"\banchor\.json\b", re.IGNORECASE), "the metric anchor"),
    (re.compile(r"\bsolution\b", re.IGNORECASE), "the solution (the reference solution)"),
    (re.compile(r"\boracle\b", re.IGNORECASE), "the oracle (the reference solution)"),
    (re.compile(r"\btask\.(toml|yaml)\b", re.IGNORECASE), "the manifest"),
)

# Applies to anything under environment/setup/payload/ — the agent-visible channel. The tests
# stage is matched only where a path can mean nothing else: relative to the setup stage
# (../tests: /opt/ale/setup and /opt/ale/tests are siblings), at its staged location, and by
# the grader's own files. A bare `tests/` is too common a folder name (a library's own test
# suite is not the grader). verify/ is the stage's name in the retired core/v1 form.
_VISIBLE_FORBIDDEN = (
    (re.compile(r"anchor", re.IGNORECASE), "the anchor"),
    (re.compile(r"\.\./tests\b", re.IGNORECASE), "the tests stage (../tests)"),
    (re.compile(r"/opt/ale/tests\b", re.IGNORECASE), "the tests stage (/opt/ale/tests)"),
    (re.compile(r"(?<!\w)tests/(verify|grader_config|test\.sh)\b", re.IGNORECASE),
     "the grader (tests/)"),
    (re.compile(r"\.\./verify", re.IGNORECASE), "the verify stage (../verify)"),
    (re.compile(r"(?<!\w)verify/", re.IGNORECASE), "the verify stage (verify/)"),
    (re.compile(r"robotics_grader", re.IGNORECASE), "the grading kit"),
)

# A practice grader may print the raw metric. A ratio or a score would leak the anchor by
# division: one run and one arithmetic step recovers the hidden number.
_VISIBLE_LEAKY_OUTPUT = (
    (re.compile(r'"ratio"'), "a ratio"),
    (re.compile(r'"score"'), "a score"),
    (re.compile(r'"reward"\s*:'), "a framework reward"),
)


def _strip_comments(text: str) -> str:
    """Drop full-line comments — prose mentioning the anchor is not an instruction."""
    return "\n".join(line for line in text.splitlines()
                     if not line.lstrip().startswith("#"))


def _copy_targets(dockerfile: str) -> list[str]:
    out = []
    for line in _strip_comments(dockerfile).splitlines():
        stripped = line.strip()
        if re.match(r"^(COPY|ADD)\b", stripped, re.IGNORECASE):
            out.append(stripped)
    return out


# --------------------------------------------------------------------------- #
# task batteries
# --------------------------------------------------------------------------- #
def lint_task(task_dir: Path) -> list[str]:
    rel = task_dir.name
    problems: list[str] = []

    try:
        manifest = read_manifest(task_dir)
    except (OSError, ValueError) as exc:
        return [f"{rel}: cannot read {MANIFEST}: {exc}"]

    if manifest.get("schema_version") != SCHEMA_VERSION:
        problems.append(f"{rel}: {MANIFEST} has no `schema_version = \"{SCHEMA_VERSION}\"` — "
                        f"the engine's task.toml schema requires exactly that value")
    task_name = _table(manifest, "task").get("name")
    parts = split_name(task_name)
    if parts is None:
        problems.append(f"{rel}: {MANIFEST} [task].name {task_name!r} is not 'org/slug' (each "
                        f"part ^[a-z0-9][a-z0-9_-]*$) — the stable identity the engine demands")
    elif parts[0] != TASK_ORG:
        problems.append(f"{rel}: {MANIFEST} [task].name {task_name!r} is outside the benchmark's "
                        f"org: every task here is '{TASK_ORG}/<slug>'")
    environment = _table(manifest, "environment")
    if not (task_dir / "environment" / "Dockerfile").is_file() \
            and not environment.get("docker_image") and not environment.get("vm_image"):
        problems.append(f"{rel}: no environment/Dockerfile and no [environment].docker_image — "
                        f"the task has no environment to run in")
    for entry, why in REQUIRED_FILES:
        if not (task_dir / entry).is_file():
            problems.append(f"{rel}: no {entry} — {why}")

    meta = _table(manifest, "metadata")
    platforms = platform_keys()
    if meta.get("platform") not in platforms:
        problems.append(f"{rel}: metadata.platform {meta.get('platform')!r} is not one of the "
                        f"benchmark's platforms ({' | '.join(platforms)}; tasks/platforms.json)")

    # Retired metadata keys: metadata.genre / metadata.paper (design 20260902 §8: one kind of
    # task, an agentic robotics task) and metadata.direction / metadata.metric (2026-10-15:
    # platform is the only taxonomy field, and the metric and its direction live in
    # tests/anchor.json, where the grader reads them). Nothing here reads or requires them;
    # a task that still carries them is not wrong, merely stale.

    if "validate" in manifest:
        problems.append(f"{rel}: {MANIFEST} carries a `validate` table — the engine has none "
                        f"(validate is the untouched-zero / oracle-one double pass), and "
                        f"task.toml is strict: unknown keys fail. The seed-variance tolerance "
                        f"is `full_at` in {ANCHOR}")

    problems += _lint_anchor(task_dir, rel)
    problems += _lint_visible(task_dir, rel)
    return problems


def _lint_anchor(task_dir: Path, rel: str) -> list[str]:
    path = task_dir / ANCHOR
    if not path.is_file():
        return [f"{rel}: no {ANCHOR} — the ratio has no denominator"]
    try:
        anchor = json.loads(path.read_text())
    except (OSError, ValueError) as exc:
        return [f"{rel}: cannot read {ANCHOR}: {exc}"]

    problems = []
    metric = str(anchor.get("metric") or "")
    if not metric or metric == "TODO":
        problems.append(f"{rel}: {ANCHOR} names no `metric` — a task with no named "
                        f"metric cannot be compared to anything")
    elif hw_metric_token(metric):
        problems.append(
            f"{rel}: metric {metric!r} is hardware-dependent (token {hw_metric_token(metric)!r}). "
            f"It measures the machine, not the method. Use an outcome metric, or recast a "
            f"timing headline as a pass/fail real-time threshold")
    if anchor.get("direction") not in ("higher", "lower"):
        problems.append(f"{rel}: {ANCHOR} `direction` must be 'higher' or 'lower'")

    value = anchor.get("value")
    if value is None:
        problems.append(
            f"{rel}: anchor value is null — run the gate once and record the measured value. "
            f"(Expected while authoring; never in a landed task.)")
    else:
        try:
            if float(value) <= 0:
                problems.append(
                    f"{rel}: anchor value {value} is not positive. The ratio inverts, so a "
                    f"better result would score lower. Report the positive cost and set the "
                    f"anchor's `direction` to lower, or shift the metric so 0 is the floor")
        except (TypeError, ValueError):
            problems.append(f"{rel}: anchor value {value!r} is not a number")

    full_at = anchor.get("full_at")
    if full_at is None:
        problems.append(
            f"{rel}: anchor has no `full_at` — the reward saturates at this ratio, and "
            f"without it an honest oracle cannot land on the exact 1.0 that `ale validate` "
            f"requires (`oracle_not_full`). Size it from the metric's measured "
            f"seed-to-seed spread")
    else:
        cap = _scoring_cap(task_dir / GRADER_CONFIG)
        try:
            if not 0 < float(full_at) <= cap:
                problems.append(f"{rel}: anchor full_at {full_at} must lie in (0, {cap:g}] "
                                f"(scoring_cap from {GRADER_CONFIG}, "
                                f"{DEFAULT_SCORING_CAP:g} by default)")
        except (TypeError, ValueError):
            problems.append(f"{rel}: anchor full_at {full_at!r} is not a number")

    source = anchor.get("source")
    source = source.strip() if isinstance(source, str) else ""
    if not source or source.startswith("TODO"):
        problems.append(f"{rel}: anchor has no `source` — an unattributed denominator cannot "
                        f"be audited")
    elif _PAPER_MENTION.search(source) and not _PAPER_DISCLAIMER.search(source):
        problems.append(f"{rel}: anchor source mentions the paper's number; the anchor must be "
                        f"the reference implementation's own measured value")

    return problems


def _lint_visible(task_dir: Path, rel: str) -> list[str]:
    """Everything under environment/setup/payload/ is handed to the agent. Nothing there may
    reach the grader. (`files/` was the old engine's name for this channel; the payload
    convention keeps the same guarantee in the Harbor layout, where
    environment/setup/setup.sh stages it into the agent home.)"""
    problems = []
    files_dir = task_dir / PAYLOAD
    if not files_dir.is_dir():
        return problems
    for path in sorted(files_dir.rglob("*")):
        if not path.is_file() or path.suffix not in (".py", ".sh", ".json", ".md", ".yaml"):
            continue
        text = _strip_comments(path.read_text(errors="replace"))
        where = path.relative_to(task_dir)
        for pattern, what in _VISIBLE_FORBIDDEN:
            if pattern.search(text):
                problems.append(f"{rel}: agent-visible {where} references {what}")
        if "practice_grader" in str(where):
            for pattern, what in _VISIBLE_LEAKY_OUTPUT:
                if pattern.search(text):
                    problems.append(
                        f"{rel}: practice grader {where} reports {what} — raw metric only, or "
                        f"one division recovers the hidden anchor")
    return problems


# --------------------------------------------------------------------------- #
# placement battery
# --------------------------------------------------------------------------- #
def tasks_root_of(task_dir: Path) -> Path | None:
    """The nearest ancestor named ``tasks``, or None when the folder is not under one (a
    template, a scratch copy). Placement is checked only for folders that live under a
    ``tasks/`` collection, because only those are landed tasks."""
    for parent in task_dir.resolve().parents:
        if parent.name == "tasks":
            return parent
    return None


def lint_placement(task_dir: Path, tasks_root: Path) -> list[str]:
    """Battery 7: ``tasks/<platform>/<slug>/`` with slug == the part of ``[task].name`` after
    ``/`` and platform == ``metadata.platform``. Returns the problems; an unreadable manifest
    or a malformed name is battery 6's."""
    task_dir = task_dir.resolve()
    rel = task_dir.relative_to(tasks_root.resolve())
    shown = f"tasks/{rel.as_posix()}/"
    problems: list[str] = []
    if len(rel.parts) != 2:
        problems.append(f"{task_dir.name}: task folder must sit exactly at "
                        f"tasks/<platform>/<slug>/ (found {shown})")
    try:
        manifest = read_manifest(task_dir)
    except (OSError, ValueError):
        return problems
    name = _table(manifest, "task").get("name")
    parts = split_name(name)
    if parts is not None and parts[1] != task_dir.name:
        problems.append(f"{task_dir.name}: folder name {task_dir.name!r} != {parts[1]!r}, the "
                        f"slug of {MANIFEST} [task].name {name!r} — the engine keys the task by "
                        f"its name, the tree by folder; rename one so they agree")
    platform = _table(manifest, "metadata").get("platform")
    if len(rel.parts) == 2 and isinstance(platform, str) and _SEGMENT.match(platform) \
            and platform != rel.parts[0]:
        problems.append(f"{task_dir.name}: folder {shown} does not match metadata.platform "
                        f"{platform!r} — move the folder to tasks/{platform}/{task_dir.name}/ "
                        f"or fix {MANIFEST}")
    return problems


# --------------------------------------------------------------------------- #
# image battery
# --------------------------------------------------------------------------- #
def lint_image(dockerfile: Path) -> list[str]:
    # The task folder the Dockerfile serves: <task>/environment/Dockerfile, or a separate
    # verifier's <task>/tests/Dockerfile.
    in_task = dockerfile.parent.name in _IMAGE_DIRS
    rel = dockerfile.parent.parent.name if in_task else dockerfile.parent.name
    shown = f"{dockerfile.parent.name}/{dockerfile.name}" if in_task else dockerfile.name
    try:
        text = dockerfile.read_text()
    except OSError as exc:
        return [f"{rel}: cannot read {shown}: {exc}"]

    problems = []
    body = _strip_comments(text)
    for pattern, what in _FORBIDDEN:
        if pattern.search(body):
            problems.append(f"{rel}: {shown} requests {what}")
    for line in _copy_targets(text):
        for pattern, what in _HIDDEN_IN_IMAGE:
            if pattern.search(line):
                problems.append(
                    f"{rel}: COPY/ADD in {shown} brings in {what}; an image layer is readable "
                    f"by anyone who can run the image: {line[:80]}")
    if re.search(r"^\s*ENTRYPOINT\b", body, re.MULTILINE):
        problems.append(f"{rel}: {shown} declares an ENTRYPOINT; the image contract forbids it "
                        f"(the framework needs to run its own commands)")
    return problems


# --------------------------------------------------------------------------- #
# retired form
# --------------------------------------------------------------------------- #
def _inside_a_task(folder: Path, root: Path) -> bool:
    """Whether ``folder`` or a folder above it, up to ``root``, holds a task.toml."""
    for candidate in (folder, *folder.parents):
        if (candidate / MANIFEST).is_file():
            return True
        if candidate == root:
            break
    return False


def retired_manifests(root: Path) -> list[str]:
    """Task folders below ``root`` still in the core/v1 form: a task.yaml with no task.toml
    in its folder or above it (a task.yaml inside a converted task, a vendored library's
    file, is not one). The engine finds tasks by task.toml alone and skips such a folder
    without a word."""
    root = root.resolve()
    problems = []
    for old in sorted(root.rglob(RETIRED_MANIFEST)):
        if not _inside_a_task(old.parent, root):
            problems.append(
                f"{old.parent.name}: {RETIRED_MANIFEST} without {MANIFEST} is the retired "
                f"core/v1 form; the engine finds tasks by {MANIFEST} only, so neither "
                f"`ale lint` nor a run sees this folder. Convert it to the Harbor layout "
                f"({MANIFEST}, environment/, tests/, solution/)")
    return problems


# --------------------------------------------------------------------------- #
def _lint_task_folder(task_dir: Path) -> list[str]:
    """Every task battery plus placement when the folder lives under a ``tasks/`` root."""
    problems = lint_task(task_dir)
    root = tasks_root_of(task_dir)
    if root is not None:
        problems += lint_placement(task_dir, root)
    return problems


def lint_domain(domain: Path) -> list[str]:
    """Policy-lint the repository: every task under tasks/ (task batteries + placement),
    plus every per-task image Dockerfile (environment/Dockerfile, a separate verifier's
    tests/Dockerfile) under tasks/ and templates/ (a template image that bakes in the hidden
    surface would replicate the leak into every instantiated task), plus any folder of
    either collection still in the retired core/v1 form. Only the collections are walked,
    never the repository root."""
    problems = []
    tasks_dir = domain / "tasks"
    if tasks_dir.is_dir():
        for manifest in sorted(tasks_dir.rglob(MANIFEST)):
            problems += _lint_task_folder(manifest.parent)
    for collection in COLLECTION_DIRS:
        root = domain / collection
        if not root.is_dir():
            continue
        for manifest in sorted(root.rglob(MANIFEST)):
            for dockerfile in DOCKERFILES:
                path = manifest.parent / dockerfile
                if path.is_file():
                    problems += lint_image(path)
        problems += retired_manifests(root)
    return problems


def lint_paths(paths: list[Path]) -> list[str]:
    """Lint whatever the caller points at: the repository (or any root holding a
    `tasks/` or `templates/` collection), a single task folder, or a tree holding several
    task folders.

    The single-task form is what a maintainer runs on one contribution
    (`tasks/<platform>/<slug>/`) before merging; placement is checked whenever the folder
    lives under a `tasks/` root.
    """
    problems: list[str] = []
    for path in paths:
        path = path.resolve()
        if (path / MANIFEST).is_file():
            problems += _lint_task_folder(path)          # a single task folder
            continue
        if any((path / name).is_dir() for name in COLLECTION_DIRS):
            problems += lint_domain(path)                # the repository (or a like root)
            continue
        # A container of task folders (e.g. the tasks/ collection itself, or one platform
        # folder), or one folder still in the retired form.
        found = sorted(path.rglob(MANIFEST))
        retired = retired_manifests(path)
        if not found and not retired:
            problems.append(f"{path}: no {MANIFEST} and no tasks/ or templates/ — "
                            f"nothing to lint")
            continue
        for manifest in found:
            problems += _lint_task_folder(manifest.parent)
        problems += retired
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("paths", type=Path, nargs="*",
                        help="task folders, repository roots, or a tree of either "
                             "(default: this repository)")
    parser.add_argument("--repo", type=Path, default=None,
                        help="deprecated alias for a positional repository root")
    args = parser.parse_args(argv)

    targets = list(args.paths)
    if args.repo is not None:
        targets.append(args.repo)
    if not targets:
        targets = [DEFAULT_DOMAIN]

    problems = lint_paths(targets)
    for problem in problems:
        print(problem, file=sys.stderr)
    if problems:
        print(f"{len(problems)} problem(s)", file=sys.stderr)
        return 1
    print("ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
