"""ale-onboard — build an ALE-shaped robotics task and gate it on the real engine.

The orchestrator behind the onboarding skills, rebuilt on upstream ALE. It is deliberately
a thin, importable core (`onboard(...)` returns a structured `OnboardResult`) so an agent
and a contributor's CLI drive exactly the same code.

What it does:

1. checks feasibility from the spec alone, and gives up honestly if the task cannot be
   scored fairly (a hardware-dependent metric, an unknown metric direction);
2. instantiates `templates/task/` into `tasks/<platform>/<task>/`, filling in
   what the spec knows and leaving TODOs where a human or an agent must still write physics;
3. runs the REAL gates — `ale lint` on the task collections, then `ale validate` on the task (its
   untouched-zero / oracle-one double pass), both from the pinned engine in `vendor/ale`;
4. reports one of four honest outcomes.

What it never does: invent a score, invent an anchor, or call something `built` that a real
`ale validate` did not produce. `built` requires the engine's double pass to hold: the
untouched episode completed with `reward` 0.0 and the oracle episode completed with `reward`
1.0. `reward` is the only value the engine reads, after rounding it to two decimals, and the
engine enforces both (`untouched_not_completed`, `untouched_nonzero`, `oracle_not_completed`,
`oracle_not_full`; exit 2). This tool still classifies from the run records, not from the
exit status, because the bootstrap case fails the engine's gate on purpose:

A task whose anchor is still null runs, measures, and scores 0 by design, so the engine
reports `oracle_not_full`. Here that comes back as `needs_anchor` carrying the measured
value (the `metrics` of the oracle episode's `verification.json`, where the kit records it),
which is the one number the author needs and cannot get any other way.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tomllib
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
#: The repository root is the domain root: it holds the task collections `tasks/` and
#: `templates/` side by side (flat layout since the 2026-09 rebrand).
DOMAIN_ROOT = REPO_ROOT
TEMPLATE_DIR = REPO_ROOT / "templates" / "task"
#: The benchmark's platform vocabulary (also read by scripts/lint_domain.py and the export).
PLATFORMS_FILE = REPO_ROOT / "tasks" / "platforms.json"
ENGINE_DIR = REPO_ROOT / "vendor" / "ale"
#: The task collections the engine lints and the name-collision scan walks. Never the
#: repository root itself: that would descend into vendor/ale and the venvs.
COLLECTION_DIRS = ("tasks", "templates")
#: Every task folder's definition file (Harbor's task.toml, schema "1.4+ale.1").
MANIFEST = "task.toml"
#: The org part of every task id, `[task].name = "<org>/<slug>"`: the GitHub and Hugging Face
#: organization that owns the benchmark. The slug is the spec's `task` and the folder name.
TASK_ORG = "grounded-intelligence-initiative"
#: The stage entry scripts, relative to the task folder.
STAGE_ENTRIES = ("environment/setup/setup.sh", "tests/test.sh", "solution/solve.sh")

#: A metric naming a wall-clock quantity measures the machine, not the method — two
#: contributors on different hardware would get different scores for identical work.
# Whole-token rule, mirrored from scripts/lint_domain.py (kept in sync by tests): a metric
# named after wall-clock or throughput measures the machine, not the method. Tokens are the
# alphabetic runs of the name after splitting camelCase, so `rms_error` and `timesteps`
# pass while `planning_time_ms` and `controlFrequencyHz` do not.
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

#: Each part of upstream's TaskId (ale.core.ids: `org/name`, both parts this shape). The
#: spec's `task` becomes the name part, which is also the folder name and the registry's
#: slug. The engine rejects a collection with two tasks of one id and the registry export
#: two tasks of one slug, so both the shape and the uniqueness are checked here, before a
#: container is ever started.
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass
class OnboardResult:
    """The outcome of one onboarding attempt, JSON-serializable for the web layer."""

    status: str  # "built" | "needs_anchor" | "needs_input" | "gave_up" | "error"
    task: str
    task_dir: str | None = None
    reward: float | None = None
    ratio: float | None = None
    measured: float | None = None
    metric: str | None = None
    anchor_recorded: bool = False
    match_lock_ok: bool | None = None
    untouched_zero: bool | None = None
    engine_failures: list[str] = field(default_factory=list)
    giveup_reason: str | None = None
    detail: str | None = None
    next_steps: list[str] = field(default_factory=list)
    log_tail: str | None = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------- #
# give-up detection — an honest abort beats a task that cannot be scored
# --------------------------------------------------------------------------- #
def check_feasibility(spec: dict[str, Any]) -> str | None:
    """Return a give-up reason if the spec cannot become a fair task, else None.

    Spec-level checks only. Deeper infeasibility — a reference implementation that does
    not run, a checkpoint too large to host, a licence that forbids redistribution —
    surfaces during the build and is reported from there. Cheap refusals belong here so
    nobody spends an hour on a task that was never scoreable. Spec keys this tool does
    not know (the retired `genre` / `paper` block among them) are ignored, not refused.
    """
    metric = spec.get("metric") or {}
    name = str(metric.get("name", ""))
    if hw_metric_token(name):
        return (
            f"metric {name!r} is hardware-dependent: it measures the machine, not the "
            f"method, so two contributors would score differently for identical work. "
            f"Recast it as a pass/fail real-time threshold, or pick an outcome metric "
            f"(success_rate / reward / SPL / ATE / collision_rate)."
        )
    if str(metric.get("direction", "higher")) not in ("higher", "lower"):
        return f"metric direction {metric.get('direction')!r} must be 'higher' or 'lower'."
    return None


# --------------------------------------------------------------------------- #
# instantiate the template
# --------------------------------------------------------------------------- #
def _substitute(text: str, spec: dict[str, Any]) -> str:
    """Fill the template manifest's TODOs with what the spec knows.

    Only matters of record are filled: the task's id (`[task].name`), its platform
    (`[metadata].platform`) and a prebuilt image (`[environment].docker_image`); the metric
    and its direction go into tests/anchor.json, see `_record_metric`. Physics, the
    simulation and the grader body are left as they are, because inventing them is exactly
    the failure mode this pipeline exists to prevent — an agent that "finishes" a task by
    writing a plausible grader has produced a task that scores nothing.
    """
    text = _set_toml_string(text, "task", "name", f"{TASK_ORG}/{spec['task']}")
    text = _set_toml_string(text, "metadata", "platform", str(spec.get("platform") or "TODO"))
    if spec.get("image"):
        # A fully-qualified prebuilt image ref. The engine runs an image named here in place
        # of building environment/Dockerfile, so instantiate() removes the template's
        # Dockerfile (see there).
        text = _set_toml_string(text, "environment", "docker_image", str(spec["image"]))
    return text


#: A table header, `[name]` or `[[name]]` (an array of tables), with an optional comment.
_TOML_HEADER = re.compile(r"^\s*(\[\[?)\s*([^\[\]#]+?)\s*\]\]?\s*(?:#.*)?$")


def _set_toml_string(text: str, table: str, key: str, value: str) -> str:
    """Set `key = "value"` in `[table]`, leaving every other line of the document as written.

    The template's comments are its documentation, so the manifest is edited line by line
    instead of being re-serialized. The key's line is replaced where the table has one;
    otherwise the line goes right below the table's header, and a table the document lacks
    is appended. `_filled_manifest` parses the result and refuses it unless every value
    landed where it belongs.
    """
    lines = text.splitlines(keepends=True)
    entry = f"{key} = {_toml_string(value)}\n"
    assignment = re.compile(rf"^\s*{re.escape(key)}\s*=")
    header: int | None = None
    for index, line in enumerate(lines):
        match = _TOML_HEADER.match(line)
        if match:
            if header is not None:
                break                                   # the table ended without the key
            if match.group(1) == "[" and match.group(2) == table:
                header = index
        elif header is not None and assignment.match(line):
            lines[index] = entry
            return "".join(lines)
    if header is None:
        body = "".join(lines)
        if body and not body.endswith("\n"):
            body += "\n"
        return f"{body}\n[{table}]\n{entry}"
    lines.insert(header + 1, entry)
    return "".join(lines)


def _toml_string(value: str) -> str:
    """`value` as a TOML basic string."""
    escaped = []
    for char in value:
        if char in '"\\':
            escaped.append("\\" + char)
        elif (ord(char) < 0x20 and char != "\t") or ord(char) == 0x7F:
            escaped.append(f"\\u{ord(char):04x}")
        else:
            escaped.append(char)
    return '"' + "".join(escaped) + '"'


def _filled_manifest(spec: dict[str, Any]) -> str:
    """The template's task.toml with the spec filled in, checked before anything is written.

    A template whose text the line editor cannot fill (a renamed table, an inline table)
    must fail here, not produce a task that silently keeps the template's id or platform.
    """
    source = TEMPLATE_DIR / MANIFEST
    text = _substitute(source.read_text(encoding="utf-8"), spec)
    try:
        config = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{source}: the filled-in manifest does not parse: {exc}") from exc
    expected = {
        ("task", "name"): f"{TASK_ORG}/{spec['task']}",
        ("metadata", "platform"): str(spec.get("platform") or "TODO"),
    }
    if spec.get("image"):
        expected[("environment", "docker_image")] = str(spec["image"])
    for (table, key), value in expected.items():
        section = config.get(table)
        landed = section.get(key) if isinstance(section, dict) else None
        if landed != value:
            raise ValueError(
                f"{source}: could not set [{table}].{key} to {value!r} (it reads {landed!r})"
            )
    return text


def platform_keys() -> tuple[str, ...]:
    """The benchmark's platform keys (tasks/platforms.json), in file order."""
    return tuple(str(entry["key"]) for entry in json.loads(PLATFORMS_FILE.read_text(encoding="utf-8")))


def _record_metric(anchor_path: Path, spec: dict[str, Any]) -> None:
    """Write the spec's metric name and direction into the copied tests/anchor.json.

    The anchor is the one place a task declares its metric: the grader reads `direction`
    there, and the registry export publishes both. The contributor's
    grader must report that metric. `value` stays null: only a real `ale validate` fills it.
    """
    metric = spec.get("metric") or {}
    if not metric.get("name") and not metric.get("direction"):
        return
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    if metric.get("name"):
        # A named metric without a direction is higher-is-better, as check_feasibility reads
        # it; never inherit the template's `lower`, which belongs to its cost metric.
        anchor["metric"] = str(metric["name"])
        anchor["direction"] = str(metric.get("direction", "higher"))
    else:
        anchor["direction"] = str(metric["direction"])
    anchor_path.write_text(json.dumps(anchor, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _name_collision(repo: Path, task: str, target: Path) -> Path | None:
    """The folder of a DIFFERENT task already claiming this name, if any.

    The name is the part of `[task].name` after the `/`, whatever the org: the engine
    refuses a collection with two tasks of one id, and the registry export refuses two
    tasks of one slug. The template itself is discovered like any task, so the scan covers
    every collection, templates included.
    """
    for manifest in _manifests(repo):
        if manifest.parent == target:
            continue
        try:
            with manifest.open("rb") as handle:
                existing = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError):
            continue
        table = existing.get("task")
        name = table.get("name") if isinstance(table, dict) else None
        if isinstance(name, str) and name.rpartition("/")[2] == task:
            return manifest.parent
    return None


def collection_roots(repo: Path) -> list[Path]:
    """The task collections under ``repo`` (``tasks/`` and ``templates/``), or ``repo`` itself
    when it is a bare collection with neither (a throwaway test tree, a single folder)."""
    roots = [repo / name for name in COLLECTION_DIRS if (repo / name).is_dir()]
    return roots or [repo]


def _manifests(repo: Path) -> list[Path]:
    manifests: list[Path] = []
    for root in collection_roots(repo):
        manifests += sorted(root.rglob(MANIFEST))
    return manifests


def instantiate(spec: dict[str, Any], *, repo: Path, force: bool = False) -> Path:
    """Copy the template into `<repo>/tasks/<platform>/<task>/` and fill in what the spec knows.

    The spec's `task` becomes the name part of the task's id,
    `[task].name = "grounded-intelligence-initiative/<task>"`, so it must be a valid slug and
    unique across every task.toml in the repo — checked here, because the engine's own
    rejection would otherwise surface only at gate time with a cryptic message. Its
    `platform` must be one of the benchmark's platforms (tasks/platforms.json): it names the
    folder the task lands in.
    """
    task = str(spec["task"])
    if not _SLUG.match(task):
        raise ValueError(
            f"task name {task!r} is not a valid slug (^[a-z0-9][a-z0-9_-]*$)"
        )
    platform = str(spec.get("platform") or "")
    if platform not in (keys := platform_keys()):
        raise ValueError(
            f"platform {platform!r} is not one of the benchmark's platforms "
            f"({' | '.join(keys)}; tasks/platforms.json)"
        )
    target = repo / "tasks" / platform / task
    if (claimed := _name_collision(repo, task, target)) is not None:
        raise ValueError(f"task name {task!r} is already claimed by {claimed}")
    manifest = _filled_manifest(spec)
    if target.exists():
        if not force:
            raise FileExistsError(
                f"{target} already exists; pass force=True to overwrite it"
            )
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(TEMPLATE_DIR, target,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".ale-cache"))

    (target / MANIFEST).write_text(manifest, encoding="utf-8")
    _record_metric(target / "tests" / "anchor.json", spec)
    if spec.get("image"):
        # The manifest now names a prebuilt image, which the engine runs in place of a
        # build: the template's Dockerfile would describe an image nobody runs, so it goes.
        dockerfile = target / "environment" / "Dockerfile"
        if dockerfile.is_file():
            dockerfile.unlink()
    # The engine runs each entry through `bash`, whatever its mode; the bit is for running one
    # by hand. copytree preserves modes, but a template copied through a zip or a Windows
    # checkout may not have them.
    for entry in STAGE_ENTRIES:
        path = target / entry
        if path.is_file():
            path.chmod(path.stat().st_mode | 0o111)
    return target


# --------------------------------------------------------------------------- #
# the real gates
# --------------------------------------------------------------------------- #
def _engine(argv: list[str], *, timeout: int) -> subprocess.CompletedProcess[str]:
    """Run one `ale` subcommand from the pinned engine.

    `VIRTUAL_ENV` is cleared because this harness runs inside its own venv and uv would
    otherwise refuse to use the engine's; the docker socket is pinned because this machine
    has two contexts and a build that lands on the other daemon is invisible to the run.
    """
    env = dict(os.environ)
    env["VIRTUAL_ENV"] = ""
    env.setdefault("DOCKER_HOST", "unix:///var/run/docker.sock")
    return subprocess.run(  # noqa: S603 — fixed argv, no shell
        ["uv", "run", "ale", *argv],
        cwd=str(ENGINE_DIR), env=env, capture_output=True, text=True, timeout=timeout,
    )


def lint(repo: Path, *, timeout: int = 120) -> subprocess.CompletedProcess[str]:
    """The cheap gate: everything answerable from the files alone.

    Runs `ale lint` over each task collection under ``repo`` (``tasks/``, ``templates/``)
    and stops at the first failing one; the repository root is never linted as a whole,
    because the engine would walk into ``vendor/ale`` and the venvs."""
    result: subprocess.CompletedProcess[str] | None = None
    for root in collection_roots(repo):
        result = _engine(["lint", str(root)], timeout=timeout)
        if result.returncode != 0:
            return result
    assert result is not None
    return result


@dataclass
class ValidatePasses:
    """The two episodes one `ale validate` runs per task, read from its run records.

    `untouched` is the NopHarness pass (nobody touched the sandbox; it must complete with
    `reward` 0.0) and `oracle` is the OracleHarness pass (the reference solution ran; it
    must complete with `reward` 1.0). Both maps are the episode's `result.json` `rewards`,
    which hold only `reward`, rounded to two decimals: the one value the engine reads. The
    `*_status` fields are the same records' `status`. The engine fails validation on any
    miss (`untouched_not_completed`, `untouched_nonzero`, `oracle_not_completed`,
    `oracle_not_full`); `engine_failures` carries those codes as read from the run's
    `validation.json`. `metrics` are the oracle pass's diagnostics, the `metrics` of its
    `verification.json` — the raw metric, the capped ratio, the lock outcomes and the kit's
    unrounded `reward_exact` — which the engine records but never gates on; `*_diagnostic`
    is the same file's `diagnostic`, the kit's reason for a zero.
    """

    untouched: dict[str, float] | None = None
    oracle: dict[str, float] | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    untouched_failure: str | None = None
    oracle_failure: str | None = None
    untouched_status: str | None = None
    oracle_status: str | None = None
    untouched_diagnostic: str | None = None
    oracle_diagnostic: str | None = None
    engine_failures: list[str] = field(default_factory=list)


def validate(task_dir: Path, *, runs_dir: Path, timeout: int = 14400
             ) -> tuple[subprocess.CompletedProcess[str] | None, ValidatePasses | None]:
    """The real run: `ale validate`'s untouched + oracle double pass on the pinned engine.

    The engine's exit status is not the verdict: it exits 2 on any validation failure,
    including the intended null-anchor bootstrap run (`oracle_not_full`), so `run_gates`
    classifies from the run records instead. Returns `(process, passes)`. `passes` is None
    when the run produced no records at all — a broken image, a crash before any episode —
    which is a real outcome and is reported as such, never smoothed into a zero. Only a run
    directory this call created is read: an engine that stops before its first episode
    (an image that does not build) creates none, and an earlier run's records must not
    stand in for it.
    """
    before = {path.name for path in runs_dir.glob("validate-*")}
    try:
        proc = _engine(["validate", str(task_dir), "--runs-dir", str(runs_dir)],
                       timeout=timeout)
    except subprocess.TimeoutExpired:
        return None, None
    return proc, _read_passes(runs_dir, task_id=_task_id(task_dir), ignore=frozenset(before))


def _task_id(task_dir: Path) -> str | None:
    """`[task].name` of the folder's task.toml, or None when it cannot be read."""
    try:
        with (task_dir / MANIFEST).open("rb") as handle:
            table = tomllib.load(handle).get("task")
    except (OSError, tomllib.TOMLDecodeError):
        return None
    name = table.get("name") if isinstance(table, dict) else None
    return name if isinstance(name, str) else None


#: The two passes, as `validate` names their episodes and `validation.json` its fields.
_PASSES = ("untouched", "oracle")
#: An episode directory of a validate run: `<org>.<name>-<pass>-<8 hex>` (the engine's
#: `task_component` of the task id, the pass, a random suffix).
_EPISODE_DIR = re.compile(r"^(?P<task>.+)-(?P<pass>untouched|oracle)-[0-9a-f]{8}$")


def _read_passes(runs_dir: Path, task_id: str | None = None,
                 ignore: frozenset[str] = frozenset()) -> ValidatePasses | None:
    """Read both validation episodes out of the newest validate run directory.

    The verdict comes from the framework's own records, not from anything the task printed:
    a number a task reports about itself is not evidence. Per pass, `result.json` gives the
    status, the rounded `reward` and a failure's message; `verification.json` (what the
    verify stage wrote, kept by the engine) gives the oracle's `metrics` and either pass's
    `diagnostic`. With `task_id` (task.toml's `[task].name`) only that task's records count:
    the newest run holding them is read, and a run of other tasks never answers for this
    one. Without it, the newest run and its first task are read. Run directories named in
    `ignore` are skipped.
    """
    runs = sorted((path for path in runs_dir.glob("validate-*")
                   if path.is_dir() and path.name not in ignore),
                  key=lambda p: p.stat().st_mtime, reverse=True)
    for run in runs:
        entry = _validation_entry(run / "validation.json", task_id)
        episodes = _episode_dirs(run, entry, task_id)
        if episodes or task_id is None:
            break
    else:
        return None
    passes = ValidatePasses(engine_failures=_failure_codes(entry))
    found = False
    for name, episode in episodes.items():
        record = _read_json(episode / "result.json")
        if not isinstance(record, dict):
            continue
        found = True
        rewards = record.get("rewards")
        rewards = ({k: float(v) for k, v in rewards.items()}
                   if isinstance(rewards, dict) else None)
        failure = record.get("failure")
        failure = failure.get("message") if isinstance(failure, dict) else None
        status = record.get("status")
        status = str(status) if status is not None else None
        verification = _read_json(episode / "verification.json")
        verification = verification if isinstance(verification, dict) else {}
        diagnostic = verification.get("diagnostic")
        diagnostic = diagnostic if isinstance(diagnostic, str) and diagnostic else None
        if name == "untouched":
            passes.untouched, passes.untouched_failure = rewards, failure
            passes.untouched_status, passes.untouched_diagnostic = status, diagnostic
        else:
            passes.oracle, passes.oracle_failure = rewards, failure
            passes.oracle_status, passes.oracle_diagnostic = status, diagnostic
            passes.metrics = _numbers(verification.get("metrics"))
    return passes if found else None


def _read_json(path: Path) -> Any:
    """The parsed file, or None when it is missing or not JSON."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _numbers(value: Any) -> dict[str, float]:
    """The numeric entries of a `metrics` object; `verification.json` may hold any JSON."""
    if not isinstance(value, dict):
        return {}
    return {str(k): float(v) for k, v in value.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)}


def _validation_entry(path: Path, task_id: str | None) -> dict[str, Any] | None:
    """The task's entry in the run's `validation.json` (`tasks[]`), or None without one.

    The entry whose `name` is `task_id`; without a `task_id`, the first (every run this tool
    starts validates exactly one task).
    """
    document = _read_json(path)
    tasks = document.get("tasks") if isinstance(document, dict) else None
    if not isinstance(tasks, list):
        return None
    entries = [entry for entry in tasks if isinstance(entry, dict)]
    if task_id is None:
        return entries[0] if entries else None
    return next((entry for entry in entries if entry.get("name") == task_id), None)


def _failure_codes(entry: dict[str, Any] | None) -> list[str]:
    """The engine's failure codes for the task (`validation.json`, `tasks[].failures[].code`).

    Diagnostic only: `oracle_not_full` is the expected reading of a null-anchor run, so
    a code never decides the outcome by itself; the episode records do.
    """
    notices = entry.get("failures") if entry else None
    codes: list[str] = []
    for notice in notices if isinstance(notices, list) else []:
        code = notice.get("code") if isinstance(notice, dict) else None
        if code:
            codes.append(str(code))
    return codes


def _episode_dirs(run: Path, entry: dict[str, Any] | None,
                  task_id: str | None) -> dict[str, Path]:
    """`{pass: episode directory}` for the task: as `validation.json` names them, else by name.

    The engine writes `validation.json` only once both passes are over, so a run that
    stopped early leaves just the episode directories, `<org>.<name>-<pass>-<suffix>`. With
    a `task_id` only that task's directories count; without one, the first of each pass.
    """
    found: dict[str, Path] = {}
    for name in _PASSES:
        attempt = entry.get(name) if entry else None
        result_path = attempt.get("result_path") if isinstance(attempt, dict) else None
        if isinstance(result_path, str) and result_path:
            found[name] = (run / result_path).parent
    component = task_id.replace("/", ".") if task_id else None
    for episode in sorted(run.iterdir()):
        match = _EPISODE_DIR.match(episode.name)
        if (match and episode.is_dir() and match.group("pass") not in found
                and (component is None or match.group("task") == component)):
            found[match.group("pass")] = episode
    return found


# --------------------------------------------------------------------------- #
# orchestrator
# --------------------------------------------------------------------------- #
#: What follows a green gate: validation proves the task runs, not that it is hard.
_DIFFICULTY_STEP = (
    "measure difficulty: run frontier models with `ale run <task_dir> --agent <harness> "
    "--model <model>` and read each verdict's `reward`; the target is roughly below 0.5. "
    "Raise the engineering or perception challenge if a model saturates it, and rule out the "
    "pinned claude-code harness's two defects first (docs/contributing-a-task.md section 7)."
)
_NEXT_STEPS = [
    "Write environment/Dockerfile: the simulator and every stable dependency, final stage "
    "FROM an official ALE base at an exact version (lint rejects `latest`), python3 >= 3.12 "
    "world-traversable — or name a prebuilt image in task.toml "
    "(`[environment].docker_image`) and delete environment/Dockerfile.",
    "Replace the template's reach environment with your simulator, imported from the image.",
    "Write solution/solve.sh: the reference method, trained/run inside the agent's own "
    "budget, leaving a self-contained /home/user/submission/.",
    "Adapt tests/verify.py to your physics, keeping the abort handling, the match lock and "
    "the scoring untouched; its `metric_name=` must be tests/anchor.json's `metric`.",
    "Record the anchor: run `ale validate`, take the measured metric off the `metrics` of "
    "the oracle episode's verification.json, write it into tests/anchor.json (with a "
    "`full_at` sized from the metric's measured seed-to-seed spread), re-run — `ale "
    "validate` fails with `oracle_not_full` until the oracle lands on reward 1.0.",
]


def onboard(spec: dict[str, Any] | Path, *, repo: Path | None = None, force: bool = False,
            gate: bool = True, runs_dir: Path | None = None,
            validate_timeout: int = 14400) -> OnboardResult:
    """Instantiate a task from a spec and run the real gates.

    `gate=False` stops after instantiation — for a caller that wants the folder to hand to
    an agent, without paying for a container.
    """
    if isinstance(spec, Path):
        import yaml

        try:
            spec = yaml.safe_load(spec.read_text()) or {}
        except (OSError, Exception) as exc:  # noqa: BLE001 — a bad spec file is an outcome
            return OnboardResult(status="error", task="?",
                                 detail=f"cannot read the spec: {exc}")
    if not isinstance(spec, dict) or not spec.get("task"):
        return OnboardResult(status="error", task="?",
                             detail="the spec has no `task` name")

    task = str(spec["task"])
    repo = repo or DOMAIN_ROOT
    runs_dir = runs_dir or (REPO_ROOT / "runs")

    if reason := check_feasibility(spec):
        return OnboardResult(status="gave_up", task=task, giveup_reason=reason)

    try:
        task_dir = instantiate(spec, repo=repo, force=force)
    except (OSError, FileExistsError, KeyError, ValueError) as exc:
        return OnboardResult(status="error", task=task, detail=str(exc))

    if not gate:
        return OnboardResult(
            status="needs_input", task=task, task_dir=str(task_dir),
            next_steps=list(_NEXT_STEPS),
            detail="instantiated from the template; the environment, the oracle and the "
                   "grader's physics still have to be written.",
        )

    return run_gates(task=task, task_dir=task_dir, repo=repo, runs_dir=runs_dir,
                     validate_timeout=validate_timeout)


def run_gates(*, task: str, task_dir: Path, repo: Path, runs_dir: Path,
              validate_timeout: int = 14400) -> OnboardResult:
    """Lint, validate — and classify the honest outcome.

    Separate from `onboard` because it is also the whole job when a task folder already
    exists: an agent that has just finished writing one asks exactly this question.

    `built` means the double pass held: the untouched episode completed with `reward` 0.0
    and the oracle episode completed with `reward` 1.0, both as the engine rounds `reward`
    (two decimals) and both the engine's own gates (`untouched_not_completed`,
    `untouched_nonzero`, `oracle_not_completed`, `oracle_not_full`). This function reads the
    records rather than the exit status because the null-anchor bootstrap run fails the
    engine's gate on purpose and must come back as `needs_anchor`, not as an error. The
    tolerance that lets an honest oracle land on 1.0 under fresh seeds is the task's
    `full_at` (tests/anchor.json), not a bar declared here.
    """
    linted = lint(repo)
    if linted.returncode != 0:
        return OnboardResult(
            status="needs_input", task=task, task_dir=str(task_dir),
            detail="`ale lint` rejected the task; fix these before anything is scored.",
            log_tail=_tail(linted.stderr or linted.stdout),
            next_steps=list(_NEXT_STEPS),
        )

    proc, passes = validate(task_dir, runs_dir=runs_dir, timeout=validate_timeout)

    if proc is None:
        return OnboardResult(
            status="needs_input", task=task, task_dir=str(task_dir),
            detail=f"`ale validate` exceeded its {validate_timeout}s deadline. If the "
                   f"oracle trains, raise task.toml's `[agent].timeout_sec` and the deadline; "
                   f"if it hangs, the solver is not answering the wire.",
            next_steps=list(_NEXT_STEPS),
        )
    if passes is None or (passes.oracle is None and passes.oracle_failure is None):
        return OnboardResult(
            status="needs_input", task=task, task_dir=str(task_dir),
            detail="`ale validate` produced no verdict — the image did not build, the "
                   "oracle did not run, or the verify stage crashed. The log tail says "
                   "which.",
            log_tail=_tail((proc.stderr or "") + (proc.stdout or "")),
            next_steps=list(_NEXT_STEPS),
        )

    metrics = passes.metrics
    oracle = passes.oracle or {}
    reward = oracle.get("reward")
    declared = _declared_metric(task_dir)
    if declared is not None and declared in metrics:
        metric, measured = declared, metrics[declared]
    else:
        metric, measured = _primary_metric(metrics)
    anchored = metrics.get("anchor_recorded", 1.0) >= 1.0
    lock_ok = metrics.get("match_lock_ok")
    untouched_zero = (passes.untouched_status == "completed"
                      and passes.untouched is not None
                      and passes.untouched.get("reward") == 0.0)
    oracle_full = (passes.oracle_status == "completed"
                   and passes.oracle is not None
                   and passes.oracle.get("reward") == 1.0)
    common = dict(
        task=task, task_dir=str(task_dir), reward=reward, ratio=metrics.get("ratio"),
        measured=measured, metric=metric, anchor_recorded=anchored,
        match_lock_ok=None if lock_ok is None else bool(lock_ok >= 1.0),
        untouched_zero=untouched_zero,
        engine_failures=list(passes.engine_failures),
    )

    if metrics and declared is not None and declared not in metrics:
        reported = repr(metric) if metric else "no raw metric"
        if not metric and passes.oracle_diagnostic:
            # a zero verdict reports no raw metric on purpose; its reason says why
            reported += f" (diagnostic: {passes.oracle_diagnostic})"
        return OnboardResult(
            status="needs_input",
            detail=(f"tests/anchor.json names the metric {declared!r}, but the grader "
                    f"reported {reported}. Set `metric_name=` in tests/verify.py to the "
                    f"anchor's metric, or fix the anchor: the anchor's name is what the "
                    f"registry export publishes for this task."),
            next_steps=list(_NEXT_STEPS),
            **common,
        )
    if not anchored:
        name_it = "" if declared else f" (as `value`, with `metric`: {metric!r})"
        return OnboardResult(
            status="needs_anchor",
            detail=(f"the run measured {metric} = {measured}. Record it in "
                    f"tests/anchor.json{name_it} and re-validate; until then the oracle scores 0 "
                    f"by design and `ale validate` fails with `oracle_not_full`, because "
                    f"a task nobody has measured cannot score anybody."),
            next_steps=[_NEXT_STEPS[-1]],
            **common,
        )
    broken = [f"the {name} pass ended with env_error ({failure})"
              for name, status, failure in (
                  ("untouched", passes.untouched_status, passes.untouched_failure),
                  ("oracle", passes.oracle_status, passes.oracle_failure))
              if status == "env_error"]
    if broken:
        return OnboardResult(
            status="needs_input",
            detail=("; ".join(broken) + ". The sandbox side failed (provisioning, guest service "
                    "or transport), so the run says nothing about the task yet."),
            log_tail=_tail(proc.stderr or ""),
            next_steps=["Fix the host side the failure names, then re-run `ale validate`."],
            **common,
        )
    if not untouched_zero:
        untouched = _pass_summary(passes.untouched, passes.untouched_status,
                                  passes.untouched_failure, passes.untouched_diagnostic)
        return OnboardResult(
            status="needs_input",
            detail=(f"the untouched pass did not complete with reward 0.0 ({untouched}). "
                    "The verify stage must produce a real 0 verdict when nobody has touched "
                    "the sandbox — catch the missing submission and write `zero_verdict(...)` "
                    "instead of failing."),
            log_tail=_tail(proc.stderr or ""),
            next_steps=list(_NEXT_STEPS),
            **common,
        )
    if oracle_full:
        return OnboardResult(
            status="built",
            detail=f"verified: the untouched pass scored reward 0.0 and the oracle reward 1.0 "
                   f"({metric} = {measured}, ratio = {metrics.get('ratio')}).",
            next_steps=[_DIFFICULTY_STEP],
            **common,
        )
    code = "oracle_not_full" if passes.oracle_status == "completed" else "oracle_not_completed"
    oracle_summary = _pass_summary(passes.oracle, passes.oracle_status,
                                   passes.oracle_failure, passes.oracle_diagnostic,
                                   reward_exact=metrics.get("reward_exact"))
    return OnboardResult(
        status="needs_input",
        detail=(f"the oracle pass did not complete with reward 1.0 ({oracle_summary}); "
                f"`ale validate` fails this as `{code}`. A task its own reference "
                f"implementation cannot pass is broken — either the oracle is not really "
                f"solving it, the anchor came from a different setup, or `full_at` "
                f"(tests/anchor.json) leaves no room for seed-to-seed variance."),
        log_tail=_tail(proc.stderr or ""),
        next_steps=list(_NEXT_STEPS),
        **common,
    )


def _pass_summary(rewards: dict[str, float] | None, status: str | None,
                  failure: str | None, diagnostic: str | None,
                  reward_exact: float | None = None) -> str:
    """One pass as its records state it: the failure or the rewards, then what explains them."""
    parts = [failure or str(rewards)]
    if reward_exact is not None:
        parts.append(f"reward_exact {reward_exact}")
    if status not in (None, "completed"):
        parts.append(f"status {status}")
    if diagnostic:
        parts.append(f"diagnostic: {diagnostic}")
    return "; ".join(parts)


def _declared_metric(task_dir: Path) -> str | None:
    """The metric tests/anchor.json declares, or None when it names none (the template's `TODO`)."""
    try:
        metric = json.loads((task_dir / "tests" / "anchor.json").read_text(encoding="utf-8")).get("metric")
    except (OSError, ValueError, AttributeError):
        return None
    return metric if isinstance(metric, str) and metric not in ("", "TODO") else None


def _primary_metric(metrics: dict[str, float]) -> tuple[str | None, float | None]:
    """The task's raw metric — whichever metrics key is not framework bookkeeping."""
    reserved = {"ratio", "anchor_recorded", "match_lock_ok", "reward_exact",
                "probes_total", "probes_passed", "episodes_aborted"}
    for name, value in metrics.items():
        if name not in reserved:
            return name, value
    return None, None


def _tail(text: str, limit: int = 2000) -> str | None:
    text = (text or "").strip()
    return text[-limit:] if text else None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        prog="ale-onboard",
        description="Build an ALE-shaped robotics task and gate it on `ale validate`.",
    )
    parser.add_argument("spec", type=Path, nargs="?",
                        help="spec YAML (task, platform, metric, optional image)")
    parser.add_argument("--gate-only", type=Path, metavar="TASK_DIR",
                        help="skip instantiation; lint + validate an existing task folder")
    parser.add_argument("--repo", type=Path, default=DOMAIN_ROOT,
                        help="the repository root holding tasks/ and templates/ "
                             "(default: this repository)")
    parser.add_argument("--runs-dir", type=Path, default=None)
    parser.add_argument("--force", action="store_true",
                        help="overwrite an existing task folder")
    parser.add_argument("--no-gate", action="store_true",
                        help="instantiate only; do not lint or validate")
    parser.add_argument("--validate-timeout", type=int, default=14400)
    parser.add_argument("--json", action="store_true", help="print the result as JSON")
    args = parser.parse_args(argv)

    runs_dir = args.runs_dir or (REPO_ROOT / "runs")
    if args.gate_only:
        task_dir = args.gate_only.resolve()
        result = run_gates(task=task_dir.name, task_dir=task_dir, repo=args.repo.resolve(),
                           runs_dir=runs_dir, validate_timeout=args.validate_timeout)
    elif args.spec:
        result = onboard(args.spec, repo=args.repo.resolve(), force=args.force,
                         gate=not args.no_gate, runs_dir=runs_dir,
                         validate_timeout=args.validate_timeout)
    else:
        parser.error("give a spec, or --gate-only <task_dir>")

    if args.json:
        print(result.to_json())
    else:
        print(f"[ale-onboard] {result.status}: {result.task}"
              + (f"  reward={result.reward}" if result.reward is not None else ""))
        if result.giveup_reason:
            print(f"  gave up: {result.giveup_reason}")
        if result.detail:
            print(f"  {result.detail}")
        for step in result.next_steps:
            print(f"  - {step}")
        if result.log_tail:
            print(f"  --- log tail ---\n{result.log_tail}")
    # built / needs_anchor / needs_input / gave_up are all real answers; only a crash is a
    # failure of this tool.
    return 1 if result.status == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
