"""ale-onboard on the ALE base: honest outcomes, real gates, no fabrication.

Docker is not exercised here — the real gates need images and minutes, and they are covered
by actual `ale validate` runs (see docs/evidence/). What these cover is the decision
logic, which is where a fabricated `built` would come from: feasibility refusals, template
instantiation, verdict classification, the bootstrap-anchor path, and reading the records a
real run leaves (engine 1cdbc02: `validation.json`, per episode `result.json` and
`verification.json`).
"""

from __future__ import annotations

import json
import os
import subprocess
import tomllib
from pathlib import Path

import pytest

from harness import ale_onboard as ob

REPO_ROOT = Path(__file__).resolve().parent.parent
ENGINE_PYTHON = REPO_ROOT / "vendor" / "ale" / ".venv" / "bin" / "python"
ORG = "grounded-intelligence-initiative"


def _manifest(task_dir: Path) -> dict:
    with (task_dir / "task.toml").open("rb") as handle:
        return tomllib.load(handle)


# --------------------------------------------------------------------------- #
# feasibility — refuse early rather than build something unscoreable
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", [
    "planning_time_ms", "control_frequency_hz", "throughput_fps", "inference_latency",
    "steps_per_second_fps",
])
def test_hardware_metrics_are_refused(name):
    reason = ob.check_feasibility({"metric": {"name": name, "direction": "higher"}})
    assert reason and "hardware-dependent" in reason


@pytest.mark.parametrize("name", [
    "success_rate", "collision_rate", "mean_episode_reward", "ATE", "SPL",
    "mean_goal_distance",
])
def test_outcome_metrics_are_accepted(name):
    assert ob.check_feasibility({"metric": {"name": name, "direction": "higher"}}) is None


def test_bad_metric_direction_is_refused():
    reason = ob.check_feasibility({"metric": {"name": "success_rate", "direction": "up"}})
    assert reason and "direction" in reason


def test_retired_genre_and_paper_keys_are_ignored_not_refused():
    """One kind of task (design 20260902 §8): a spec still carrying the retired genre /
    paper block is stale, not infeasible — and a missing paper block is never a reason
    to give up, because feasibility is about the metric, not about provenance prose."""
    assert ob.check_feasibility({
        "genre": "reproduction",
        "metric": {"name": "success_rate", "direction": "higher"},
        "paper": {"title": "x", "reference_repo": "  ", "year": 2009},
    }) is None
    assert ob.check_feasibility({
        "metric": {"name": "reward", "direction": "higher"},
    }) is None


# --------------------------------------------------------------------------- #
# instantiation — the ALE folder shape, metadata filled, physics left alone
# --------------------------------------------------------------------------- #
SPEC = {
    "task": "probe_task",
    "platform": "arm",
    "metric": {"name": "success_rate", "direction": "lower"},
}


def test_instantiate_writes_the_harbor_shape(tmp_path):
    task_dir = ob.instantiate(SPEC, repo=tmp_path)
    assert task_dir == tmp_path / "tasks" / "arm" / "probe_task"
    for required in ("task.toml", "instruction.md", "environment/Dockerfile",
                     "environment/setup/setup.sh", "tests/test.sh", "tests/verify.py",
                     "tests/anchor.json", "solution/solve.sh"):
        assert (task_dir / required).is_file(), required
    # the vendored verify-stage machinery travels with the task
    assert (task_dir / "tests" / "robotics_grader" / "__init__.py").is_file()
    # only the entries the engine's lint admits at the top of a task folder
    assert {entry.name for entry in task_dir.iterdir()} <= {
        "task.toml", "instruction.md", "environment", "tests", "solution"}
    # nothing from a retired shape survives
    for gone in ("task.yaml", "image", "setup", "verify", "oracle", "tools", "Dockerfile",
                 "description.md", "grader", "example", "practice_grader", "files", "kits"):
        assert not (task_dir / gone).exists(), gone


@pytest.mark.parametrize("direction", ["higher", "lower"])
def test_instantiate_sets_the_declared_metric_direction(tmp_path, direction):
    """The template's anchor ships `direction: lower` (its grader scores a cost); the spec's
    value must land in tests/anchor.json either way, never the template's default."""
    spec = dict(SPEC, metric={"name": "success_rate", "direction": direction})
    task_dir = ob.instantiate(spec, repo=tmp_path)
    anchor = json.loads((task_dir / "tests" / "anchor.json").read_text())
    assert (anchor["metric"], anchor["direction"]) == ("success_rate", direction)


def test_instantiate_fills_metadata_and_keeps_todos(tmp_path):
    task_dir = ob.instantiate(SPEC, repo=tmp_path)
    manifest = _manifest(task_dir)
    assert manifest["schema_version"] == "1.4+ale.1"
    # the stable identity, stamped: the engine's org/name, the name being the spec's task
    assert manifest["task"]["name"] == f"{ORG}/probe_task"
    assert "docker_image" not in manifest.get("environment", {})   # the template builds its image
    meta = manifest["metadata"]
    assert meta == {"platform": "arm"}          # platform is the only metadata field
    # One kind of task: the retired genre / paper block is not written.
    assert "genre" not in meta and "paper" not in meta
    # The metric and its direction go into the anchor. Its value is NOT filled: it can only
    # come from a real measurement.
    anchor = json.loads((task_dir / "tests" / "anchor.json").read_text())
    assert (anchor["metric"], anchor["direction"]) == ("success_rate", "lower")
    assert anchor["value"] is None


def test_instantiate_keeps_the_template_manifest_as_written(tmp_path):
    """The template's comments are its documentation: the manifest is edited line by line,
    so every other line, and every value the spec does not name, survives."""
    template = (ob.TEMPLATE_DIR / "task.toml").read_text(encoding="utf-8")
    task_dir = ob.instantiate(SPEC, repo=tmp_path)
    written = (task_dir / "task.toml").read_text(encoding="utf-8")
    kept = [line for line in template.splitlines()
            if not line.startswith(("name = ", "platform = "))]
    assert [line for line in written.splitlines()
            if not line.startswith(("name = ", "platform = "))] == kept
    original, filled = tomllib.loads(template), _manifest(task_dir)
    for table in ("environment", "agent", "verifier"):
        assert filled.get(table) == original.get(table), table
    assert filled.get("artifacts") == original.get("artifacts")


def test_a_named_metric_without_direction_is_higher_is_better(tmp_path):
    """Never inherit the template anchor's `lower`: it belongs to the template's cost metric."""
    task_dir = ob.instantiate(dict(SPEC, metric={"name": "success_rate"}), repo=tmp_path)
    anchor = json.loads((task_dir / "tests" / "anchor.json").read_text())
    assert (anchor["metric"], anchor["direction"]) == ("success_rate", "higher")


def test_a_direction_without_a_metric_name_leaves_the_anchor_unnamed(tmp_path):
    """The template's anchor names no metric (`TODO`), so the lint keeps asking for one."""
    task_dir = ob.instantiate(dict(SPEC, metric={"direction": "higher"}), repo=tmp_path)
    anchor = json.loads((task_dir / "tests" / "anchor.json").read_text())
    assert (anchor["metric"], anchor["direction"]) == ("TODO", "higher")


def test_instantiate_refuses_an_unregistered_platform(tmp_path):
    with pytest.raises(ValueError, match="tasks/platforms.json"):
        ob.instantiate(dict(SPEC, platform="spacecraft"), repo=tmp_path)
    assert not (tmp_path / "tasks").exists()        # nothing written


def test_instantiate_ignores_a_stale_genre_in_the_spec(tmp_path):
    """An old spec file that still says `genre:` instantiates cleanly and the manifest
    does not grow a genre — the template has no slot for it any more."""
    stale = dict(SPEC, genre="reproduction",
                 paper={"title": "A: study", "reference_repo": "https://x/y"})
    task_dir = ob.instantiate(stale, repo=tmp_path)
    manifest = _manifest(task_dir)
    assert "genre" not in manifest["metadata"]
    assert "paper" not in manifest["metadata"]
    assert "genre" not in (task_dir / "task.toml").read_text()


def test_instantiate_with_an_image_ref_drops_the_dockerfile(tmp_path):
    task_dir = ob.instantiate(dict(SPEC, image="ghcr.io/example/env:1"), repo=tmp_path)
    environment = _manifest(task_dir)["environment"]
    assert environment["docker_image"] == "ghcr.io/example/env:1"
    # the rest of [environment] is the template's
    template = tomllib.loads((ob.TEMPLATE_DIR / "task.toml").read_text(encoding="utf-8"))
    assert {k: v for k, v in environment.items() if k != "docker_image"} == template["environment"]
    # an image named in task.toml wins over a build, so a local Dockerfile would only mislead
    assert not (task_dir / "environment" / "Dockerfile").exists()


def test_instantiate_rejects_a_bad_slug(tmp_path):
    with pytest.raises(ValueError, match="slug"):
        ob.instantiate(dict(SPEC, task="Probe Task"), repo=tmp_path)


def test_instantiate_rejects_a_claimed_name(tmp_path):
    ob.instantiate(SPEC, repo=tmp_path)
    other = dict(SPEC, platform="uav")               # different folder, same name
    with pytest.raises(ValueError, match="already claimed"):
        ob.instantiate(other, repo=tmp_path)
    # force-overwriting the SAME folder is not a collision
    assert ob.instantiate(SPEC, repo=tmp_path, force=True).is_dir()


def test_a_name_claimed_under_another_org_is_still_claimed(tmp_path):
    """`someone-else/probe_task` is another id to the engine but the same slug to the registry."""
    elsewhere = tmp_path / "tasks" / "uav" / "imported"
    elsewhere.mkdir(parents=True)
    (elsewhere / "task.toml").write_text(
        'schema_version = "1.4+ale.1"\n\n[task]\nname = "someone-else/probe_task"\n')
    with pytest.raises(ValueError, match="already claimed"):
        ob.instantiate(SPEC, repo=tmp_path)
    assert not (tmp_path / "tasks" / "arm").exists()        # nothing written


def test_instantiate_marks_the_stage_scripts_executable(tmp_path):
    task_dir = ob.instantiate(SPEC, repo=tmp_path)
    for entry in ("environment/setup/setup.sh", "tests/test.sh", "solution/solve.sh"):
        assert (task_dir / entry).stat().st_mode & 0o111, entry


def test_instantiate_refuses_to_clobber(tmp_path):
    ob.instantiate(SPEC, repo=tmp_path)
    with pytest.raises(FileExistsError):
        ob.instantiate(SPEC, repo=tmp_path)
    # force replaces it
    assert ob.instantiate(SPEC, repo=tmp_path, force=True).is_dir()


def test_a_template_the_editor_cannot_fill_is_refused_before_anything_is_written(
        tmp_path, monkeypatch):
    """An inline `task = {...}` table has no `[task]` header to edit: refuse, never write a
    task that keeps the template's id."""
    template = tmp_path / "template"
    template.mkdir()
    (template / "task.toml").write_text(
        'schema_version = "1.4+ale.1"\ntask = { name = "grounded-intelligence-initiative/x" }\n'
        '\n[metadata]\nplatform = "TODO"\n')
    monkeypatch.setattr(ob, "TEMPLATE_DIR", template)
    with pytest.raises(ValueError, match="task.toml"):
        ob.instantiate(SPEC, repo=tmp_path / "repo")
    assert not (tmp_path / "repo" / "tasks").exists()


def test_set_toml_string_edits_only_its_table():
    text = ('# a comment\nschema_version = "1.4+ale.1"\n\n[task]\nname = "old/x"  # trailing\n\n'
            '[[environment.mcp_servers]]\nname = "lab"\n')
    edited = ob._set_toml_string(text, "task", "name", "org/new")
    parsed = tomllib.loads(edited)
    assert parsed["task"]["name"] == "org/new"
    assert parsed["environment"]["mcp_servers"] == [{"name": "lab"}]
    assert edited.startswith("# a comment\n")
    # a key the table lacks goes below its header; a missing table is appended
    added = tomllib.loads(ob._set_toml_string(edited, "environment", "docker_image", "img:1"))
    assert added["environment"]["docker_image"] == "img:1"
    assert added["environment"]["mcp_servers"] == [{"name": "lab"}]
    appended = tomllib.loads(ob._set_toml_string(text, "metadata", "platform", "arm"))
    assert appended["metadata"] == {"platform": "arm"} and appended["task"]["name"] == "old/x"


@pytest.mark.parametrize("value", ['quote " and back\\slash', "tab\tand newline\n", "ünïcode",
                                   "del\x7fchar"])
def test_toml_string_round_trips(value):
    assert tomllib.loads(f"key = {ob._toml_string(value)}\n")["key"] == value


@pytest.mark.skipif(not ENGINE_PYTHON.is_file(),
                    reason="engine venv missing: run `uv sync` in vendor/ale")
@pytest.mark.parametrize("image", [None, "ghcr.io/example/env:1"])
def test_an_instantiated_task_loads_in_the_engine(tmp_path, image):
    """The pinned engine's own loader reads the filled-in manifest: the id, the platform,
    and the declared image in place of the build."""
    spec = dict(SPEC, image=image) if image else SPEC
    task_dir = ob.instantiate(spec, repo=tmp_path)
    program = ("import json, sys; from pathlib import Path; "
               "from ale.run.tasksets.manifest import load_task_folder; "
               "t = load_task_folder(Path(sys.argv[1])); "
               "print(json.dumps([str(t.spec.name), t.spec.metadata, t.spec.image.ref]))")
    probe = subprocess.run(
        [str(ENGINE_PYTHON), "-c", program, str(task_dir)],
        capture_output=True, text=True, check=False, env={**os.environ, "VIRTUAL_ENV": ""},
    )
    assert probe.returncode == 0, probe.stderr
    assert json.loads(probe.stdout) == [f"{ORG}/probe_task", {"platform": "arm"}, image]


def test_onboard_gives_up_before_touching_the_disk(tmp_path):
    spec = dict(SPEC, task="hw", metric={"name": "planning_time_ms", "direction": "lower"})
    result = ob.onboard(spec, repo=tmp_path, gate=False)
    assert result.status == "gave_up"
    assert result.giveup_reason and "hardware-dependent" in result.giveup_reason
    assert not (tmp_path / "tasks").exists()      # nothing written
    assert result.reward is None and result.anchor_recorded is False


def test_onboard_without_gate_reports_the_work_left(tmp_path):
    result = ob.onboard(SPEC, repo=tmp_path, gate=False)
    assert result.status == "needs_input"
    assert result.next_steps
    assert result.reward is None                  # no score without a real run
    assert Path(result.task_dir).is_dir()
    # the steps name the task's real files, not the retired layout
    steps = " ".join(result.next_steps)
    for path in ("environment/Dockerfile", "solution/solve.sh", "tests/verify.py",
                 "tests/anchor.json", "verification.json"):
        assert path in steps, path
    for retired in ("image/Dockerfile", "oracle/run.sh", "verify/", "result.json"):
        assert retired not in steps, retired


def test_onboard_rejects_a_spec_without_a_task_name(tmp_path):
    assert ob.onboard({"platform": "uav"}, repo=tmp_path, gate=False).status == "error"


def test_onboard_reports_an_unreadable_spec_file(tmp_path):
    assert ob.onboard(tmp_path / "nope.yaml", repo=tmp_path, gate=False).status == "error"


# --------------------------------------------------------------------------- #
# verdict classification — the only place a fake `built` could come from
# --------------------------------------------------------------------------- #
def _task(tmp_path, metric=None):
    # the template's grader reports mean_goal_distance, so that is the anchor's metric here
    return ob.instantiate(dict(SPEC, task="cls",
                               metric=metric or {"name": "mean_goal_distance", "direction": "lower"}),
                          repo=tmp_path)


_CLEAN_ZERO = object()  # sentinel: "the untouched pass scored an honest zero"


def _passes(oracle=None, untouched=_CLEAN_ZERO, metrics=None,
            oracle_failure=None, untouched_failure=None, engine_failures=(),
            oracle_status=None, untouched_status=None, oracle_diagnostic=None):
    """Both passes as `_read_passes` returns them; a pass with rewards completed by default."""
    untouched = {"reward": 0.0} if untouched is _CLEAN_ZERO else untouched
    return ob.ValidatePasses(
        untouched=untouched,
        oracle=oracle,
        metrics=metrics or {},
        oracle_failure=oracle_failure,
        untouched_failure=untouched_failure,
        untouched_status=untouched_status or ("completed" if untouched is not None else "task_error"),
        oracle_status=oracle_status or ("completed" if oracle is not None else "task_error"),
        oracle_diagnostic=oracle_diagnostic,
        engine_failures=list(engine_failures),
    )


def _gate(monkeypatch, tmp_path, passes, *, lint_rc=0, proc=True, proc_rc=0, metric=None):
    task_dir = _task(tmp_path, metric)

    class FakeLint:
        returncode = lint_rc
        stdout = "lint said so"
        stderr = "stderr tail"

    class FakeValidate:
        returncode = proc_rc            # 2 = the engine's "some validation failed"
        stdout = ""
        stderr = "stderr tail"

    monkeypatch.setattr(ob, "lint", lambda *a, **k: FakeLint())
    monkeypatch.setattr(ob, "validate",
                        lambda *a, **k: ((FakeValidate() if proc else None), passes))
    return ob.run_gates(task="cls", task_dir=task_dir, repo=tmp_path,
                        runs_dir=tmp_path / "runs")


def test_a_passing_oracle_is_built(monkeypatch, tmp_path):
    result = _gate(monkeypatch, tmp_path, _passes(
        oracle={"reward": 1.0},
        metrics={"ratio": 1.13, "mean_goal_distance": 0.094,
                 "anchor_recorded": 1.0, "match_lock_ok": 1.0, "reward_exact": 1.0},
    ))
    assert result.status == "built"
    assert result.reward == 1.0 and result.ratio == 1.13
    assert result.metric == "mean_goal_distance" and result.measured == 0.094
    assert result.match_lock_ok is True
    assert result.untouched_zero is True
    # a green gate proves the task runs, not that it is hard: the next step is the
    # frontier-model difficulty test
    assert any("ale run" in step and "0.5" in step for step in result.next_steps)


def test_a_null_anchor_is_needs_anchor_with_the_measurement(monkeypatch, tmp_path):
    """A null-anchor run fails with `oracle_not_full` (exit 2). The records still carry the
    measurement, and that, not the exit status, decides."""
    result = _gate(monkeypatch, tmp_path, _passes(
        oracle={"reward": 0.0},
        metrics={"ratio": 0.0, "mean_goal_distance": 0.1024,
                 "anchor_recorded": 0.0, "match_lock_ok": 1.0, "reward_exact": 0.0},
        engine_failures=["oracle_not_full"],
    ), lint_rc=0, proc_rc=2)
    assert result.status == "needs_anchor"
    assert result.measured == 0.1024                   # the number the author needs
    assert result.anchor_recorded is False
    assert result.engine_failures == ["oracle_not_full"]
    assert "tests/anchor.json" in result.detail and "oracle_not_full" in result.detail
    assert "`metric`" not in result.detail              # the anchor already names it


def test_an_anchor_still_naming_todo_asks_for_the_metric_name(monkeypatch, tmp_path):
    """`TODO` is the template's placeholder, not a metric: no mismatch, but ask for the name."""
    result = _gate(monkeypatch, tmp_path, _passes(
        oracle={"reward": 0.0},
        metrics={"ratio": 0.0, "mean_goal_distance": 0.1024,
                 "anchor_recorded": 0.0, "match_lock_ok": 1.0, "reward_exact": 0.0},
        engine_failures=["oracle_not_full"],
    ), proc_rc=2, metric={"direction": "lower"})
    assert result.status == "needs_anchor"
    assert "`metric`: 'mean_goal_distance'" in result.detail


def test_an_oracle_below_one_is_not_built(monkeypatch, tmp_path):
    result = _gate(monkeypatch, tmp_path, _passes(
        oracle={"reward": 0.62},
        metrics={"ratio": 0.5, "mean_goal_distance": 0.4,
                 "anchor_recorded": 1.0, "match_lock_ok": 1.0, "reward_exact": 0.6213},
    ))
    assert result.status == "needs_input"
    assert result.reward == 0.62
    assert "reward 1.0" in result.detail and "full_at" in result.detail
    assert "reward_exact 0.6213" in result.detail       # the unrounded value, for the author
    # The 1.0 oracle is the engine's own gate (`oracle_not_full`); say so, and never
    # call it a warning or a domain rule.
    assert "oracle_not_full" in result.detail
    assert "partial_oracle" not in result.detail and "domain policy" not in result.detail


def test_an_oracle_rounded_up_to_one_is_built(monkeypatch, tmp_path):
    """The engine rounds `reward` half up to two decimals: 0.995 exact is 1.0, and passes."""
    result = _gate(monkeypatch, tmp_path, _passes(
        oracle={"reward": 1.0},
        metrics={"ratio": 0.796, "mean_goal_distance": 0.11,
                 "anchor_recorded": 1.0, "match_lock_ok": 1.0, "reward_exact": 0.995},
    ))
    assert result.status == "built" and result.reward == 1.0


def test_an_oracle_that_did_not_complete_is_not_built_even_at_reward_one(monkeypatch, tmp_path):
    """Rewards can exist without a completed episode (the verifier ran after a failed
    solve.sh); the engine fails that as `oracle_not_completed`, and so does this gate."""
    result = _gate(monkeypatch, tmp_path, _passes(
        oracle={"reward": 1.0}, oracle_status="agent_error",
        oracle_failure="solution exited with status 1",
        metrics={"ratio": 1.0, "mean_goal_distance": 0.09,
                 "anchor_recorded": 1.0, "match_lock_ok": 1.0},
        engine_failures=["oracle_not_completed"],
    ), proc_rc=2)
    assert result.status == "needs_input"
    assert "oracle_not_completed" in result.detail
    assert "solution exited with status 1" in result.detail and "agent_error" in result.detail


@pytest.mark.parametrize("metrics", [
    # a zero verdict: no raw metric at all (the kit never fabricates one)
    {"ratio": 0.0, "anchor_recorded": 1.0, "reward_exact": 0.0},
    # a measured run that still scored 0 (a failed lock zeroes the reward)
    {"ratio": 0.0, "mean_goal_distance": 0.73, "anchor_recorded": 1.0, "match_lock_ok": 0.0,
     "reward_exact": 0.0},
])
def test_the_kit_s_reason_for_an_oracle_zero_is_reported(monkeypatch, tmp_path, metrics):
    reason = "scoring 0: no submission at /home/user/submission/run.sh"
    result = _gate(monkeypatch, tmp_path, _passes(
        oracle={"reward": 0.0}, metrics=metrics, oracle_diagnostic=reason,
        engine_failures=["oracle_not_full"],
    ), proc_rc=2)
    assert result.status == "needs_input"
    assert reason in result.detail


def test_a_grader_reporting_another_metric_is_needs_input(monkeypatch, tmp_path):
    """The anchor's metric is what the registry publishes for the task; the grader must report it."""
    result = _gate(monkeypatch, tmp_path, _passes(
        oracle={"reward": 1.0},
        metrics={"ratio": 1.1, "success_rate": 0.9, "anchor_recorded": 1.0, "match_lock_ok": 1.0},
    ))
    assert result.status == "needs_input"
    assert "mean_goal_distance" in result.detail and "success_rate" in result.detail
    assert "metric_name" in result.detail and "tests/verify.py" in result.detail


def test_an_untouched_pass_that_crashed_is_not_built(monkeypatch, tmp_path):
    result = _gate(monkeypatch, tmp_path, _passes(
        oracle={"reward": 1.0},
        untouched=None, untouched_failure="verify failed with exit code 1",
        metrics={"ratio": 1.1, "mean_goal_distance": 0.09,
                 "anchor_recorded": 1.0, "match_lock_ok": 1.0},
    ))
    assert result.status == "needs_input"
    assert result.untouched_zero is False
    assert "zero_verdict" in result.detail and "exit code 1" in result.detail


def test_an_infrastructure_failure_is_not_blamed_on_the_task(monkeypatch, tmp_path):
    """`env_error` is the sandbox side (here the shape of a real 1cdbc02 run whose provider
    could not start): no verdict about the task, and no advice to rewrite its grader."""
    failure = ("could not route the sandbox's access to the host: Unable to find image "
               "'ghcr.io/agentslastexam/ale-qemu-runner:0.1.1' locally")
    result = _gate(monkeypatch, tmp_path, _passes(
        oracle=None, untouched=None, oracle_status="env_error", untouched_status="env_error",
        oracle_failure=failure, untouched_failure=failure,
        engine_failures=["untouched_not_completed", "oracle_not_completed"],
    ), proc_rc=2)
    assert result.status == "needs_input"
    assert result.reward is None and result.untouched_zero is False
    assert "env_error" in result.detail and "ale-qemu-runner" in result.detail
    assert "zero_verdict" not in result.detail and "reference implementation" not in result.detail


def test_an_untouched_pass_above_zero_is_not_built(monkeypatch, tmp_path):
    result = _gate(monkeypatch, tmp_path, _passes(
        oracle={"reward": 1.0}, untouched={"reward": 0.01},
        metrics={"ratio": 1.1, "mean_goal_distance": 0.09,
                 "anchor_recorded": 1.0, "match_lock_ok": 1.0},
        engine_failures=["untouched_nonzero"],
    ), proc_rc=2)
    assert result.status == "needs_input"
    assert result.untouched_zero is False


def test_a_failed_match_lock_is_not_built(monkeypatch, tmp_path):
    result = _gate(monkeypatch, tmp_path, _passes(
        oracle={"reward": 0.0},
        metrics={"ratio": 0.0, "mean_goal_distance": 0.73,
                 "anchor_recorded": 1.0, "match_lock_ok": 0.0},
    ))
    assert result.status == "needs_input"
    assert result.match_lock_ok is False


def test_a_failing_lint_never_reaches_validate(monkeypatch, tmp_path):
    called = []
    task_dir = _task(tmp_path)

    class Bad:
        returncode = 1
        stdout = ""
        stderr = "task.toml: something is wrong"

    monkeypatch.setattr(ob, "lint", lambda *a, **k: Bad())
    monkeypatch.setattr(ob, "validate", lambda *a, **k: called.append(1) or (None, None))
    result = ob.run_gates(task="cls", task_dir=task_dir, repo=tmp_path,
                          runs_dir=tmp_path / "runs")
    assert result.status == "needs_input"
    assert not called                                   # no container was started
    assert "something is wrong" in (result.log_tail or "")


def test_no_verdict_is_reported_not_scored_as_zero(monkeypatch, tmp_path):
    result = _gate(monkeypatch, tmp_path, None)
    assert result.status == "needs_input"
    assert result.reward is None                        # NOT 0.0 — there was no measurement
    assert "no verdict" in result.detail


def test_a_timeout_is_its_own_message(monkeypatch, tmp_path):
    result = _gate(monkeypatch, tmp_path, None, proc=False)
    assert result.status == "needs_input"
    assert result.reward is None
    assert "deadline" in result.detail and "[agent].timeout_sec" in result.detail


def test_result_is_json_serializable(monkeypatch, tmp_path):
    result = _gate(monkeypatch, tmp_path, _passes(
        oracle={"reward": 1.0},
        metrics={"ratio": 1.08, "mean_goal_distance": 1.0, "anchor_recorded": 1.0},
    ))
    assert json.loads(result.to_json())["status"] == "built"


# --------------------------------------------------------------------------- #
# reading both validation passes out of a real run-directory shape (engine 1cdbc02)
# --------------------------------------------------------------------------- #
TASK_ID = f"{ORG}/cls"
COMPONENT = f"{ORG}.cls"           # the engine's task_component: the id with "." for "/"


def _write_episode(run, name, rewards, metrics=None, *, failure=None, diagnostic=None,
                   status=None, engine_metrics=None, verification=True):
    """One episode directory as the engine leaves it.

    `result.json` (schema 2): the status, `rewards` with only the rounded `reward`, the
    engine's own `metrics` (empty for a validate pass). `verification.json`: what the verify
    stage wrote through the kit — `reward`, `metrics` (the raw metric, the ratio, the locks,
    `reward_exact`) and the `diagnostic` of a zero.
    """
    episode = run / name
    episode.mkdir(parents=True)
    record = {"schema_version": 2, "episode_id": name,
              "status": status or ("completed" if rewards is not None else "task_error"),
              "rewards": rewards, "metrics": engine_metrics or {},
              "started_at": "2026-11-03T20:23:23Z", "finished_at": "2026-11-03T20:23:28Z"}
    if failure:
        record["failure"] = {"error_type": "VerifierOutputError", "message": failure,
                             "phase": "verify"}
    (episode / "result.json").write_text(json.dumps(record))
    if verification and rewards is not None:
        written = {"reward": rewards["reward"], "metrics": dict(metrics or {})}
        if diagnostic:
            written["diagnostic"] = diagnostic
        (episode / "verification.json").write_text(json.dumps(written))
    return episode


def _write_validation(run, untouched, oracle, failures=(), name=TASK_ID):
    """The run's `validation.json`, as `ale validate` writes it once both passes are over."""
    def attempt(episode):
        record = json.loads((episode / "result.json").read_text())
        return {"episode_id": episode.name, "status": record["status"],
                "rewards": record["rewards"] if record["status"] == "completed" else None,
                "lock_path": f"{episode.name}/lock.json", "lock_digest": "sha256:" + "0" * 64,
                "result_path": f"{episode.name}/result.json"}

    (run / "validation.json").write_text(json.dumps({
        "schema_version": 1, "run_id": run.name,
        "engine": {"commit": "1cdbc02a1f3390e704747b33946304911f3105ad", "version": "0.1.1"},
        "tasks": [{"name": name, "spec_hash": "sha256:" + "1" * 64, "passed": not failures,
                   "reward_names": ["reward"], "warnings": [],
                   "failures": [{"code": code, "message": code} for code in failures],
                   "untouched": attempt(untouched), "oracle": attempt(oracle)}],
    }))


def test_passes_are_read_from_the_framework_records(tmp_path):
    run = tmp_path / "validate-deadbeef"
    untouched = _write_episode(run, f"{COMPONENT}-untouched-11111111", {"reward": 0.0},
                               {"ratio": 0.0, "anchor_recorded": 1.0, "reward_exact": 0.0},
                               diagnostic="scoring 0: no submission")
    oracle = _write_episode(run, f"{COMPONENT}-oracle-22222222", {"reward": 1.0},
                            {"ratio": 1.13, "mean_goal_distance": 0.094,
                             "anchor_recorded": 1.0, "reward_exact": 1.0})
    _write_validation(run, untouched, oracle)
    passes = ob._read_passes(tmp_path, task_id=TASK_ID)
    assert passes.untouched == {"reward": 0.0} and passes.untouched_status == "completed"
    assert passes.oracle == {"reward": 1.0} and passes.oracle_status == "completed"
    assert passes.metrics["ratio"] == 1.13 and passes.metrics["reward_exact"] == 1.0
    assert passes.untouched_diagnostic == "scoring 0: no submission"
    assert passes.engine_failures == []


def test_the_measurement_comes_from_verification_json_not_result_json(tmp_path):
    """`result.json` `metrics` are the engine's own diagnostics now; the task's are in
    `verification.json`, and only those can carry the measurement or `anchor_recorded`."""
    run = tmp_path / "validate-deadbeef"
    _write_episode(run, f"{COMPONENT}-untouched-11111111", {"reward": 0.0}, {"ratio": 0.0})
    _write_episode(run, f"{COMPONENT}-oracle-22222222", {"reward": 0.0},
                   {"ratio": 0.0, "mean_goal_distance": 0.1024, "anchor_recorded": 0.0,
                    "notes": "not a number", "flag": True},
                   engine_metrics={"agent_turns": 3.0})
    passes = ob._read_passes(tmp_path)
    assert passes.metrics == {"ratio": 0.0, "mean_goal_distance": 0.1024,
                              "anchor_recorded": 0.0}


def test_a_task_error_pass_reads_as_failure_not_rewards(tmp_path):
    run = tmp_path / "validate-deadbeef"
    _write_episode(run, f"{COMPONENT}-untouched-11111111", None,
                   failure="verify failed with exit code 1")
    _write_episode(run, f"{COMPONENT}-oracle-22222222", {"reward": 1.0}, {"ratio": 1.1})
    passes = ob._read_passes(tmp_path)
    assert passes.untouched is None
    assert passes.untouched_status == "task_error"
    assert "exit code 1" in passes.untouched_failure


def test_the_newest_validate_run_wins(tmp_path):
    old = tmp_path / "validate-00000000"
    _write_episode(old, f"{COMPONENT}-oracle-11111111", {"reward": 0.0}, {"ratio": 0.0})
    new = tmp_path / "validate-11111111"
    _write_episode(new, f"{COMPONENT}-oracle-22222222", {"reward": 1.0}, {"ratio": 1.0})
    os.utime(old, (1, 1))
    passes = ob._read_passes(tmp_path)
    assert passes.oracle == {"reward": 1.0}


def test_a_missing_runs_dir_reads_as_none(tmp_path):
    assert ob._read_passes(tmp_path / "absent") is None


def test_engine_failure_codes_are_read_from_validation_json(tmp_path):
    """The engine's own verdict codes (`validation.json`) ride along as diagnostics."""
    run = tmp_path / "validate-deadbeef"
    untouched = _write_episode(run, f"{COMPONENT}-untouched-11111111", {"reward": 0.0},
                               {"ratio": 0.0, "anchor_recorded": 0.0})
    oracle = _write_episode(run, f"{COMPONENT}-oracle-22222222", {"reward": 0.0},
                            {"ratio": 0.0, "mean_goal_distance": 0.1, "anchor_recorded": 0.0})
    _write_validation(run, untouched, oracle, failures=["oracle_not_full"])
    passes = ob._read_passes(tmp_path, task_id=TASK_ID)
    assert passes.engine_failures == ["oracle_not_full"]
    assert passes.oracle == {"reward": 0.0}            # the records are still the verdict


def test_a_missing_or_broken_validation_json_is_no_failure_code(tmp_path):
    run = tmp_path / "validate-deadbeef"
    _write_episode(run, f"{COMPONENT}-oracle-22222222", {"reward": 1.0}, {"ratio": 1.0})
    assert ob._read_passes(tmp_path).engine_failures == []
    (run / "validation.json").write_text("{not json")
    assert ob._read_passes(tmp_path).engine_failures == []


def test_validation_json_names_the_episodes_and_picks_the_task(tmp_path):
    """`result_path` decides which directory is which pass; another task's episodes in the
    same run (a collection was validated) do not stand in for this task's."""
    run = tmp_path / "validate-deadbeef"
    other = f"{ORG}.other_task"
    _write_episode(run, f"{other}-untouched-33333333", {"reward": 0.0}, {"ratio": 0.0})
    _write_episode(run, f"{other}-oracle-44444444", {"reward": 0.0}, {"ratio": 0.0})
    untouched = _write_episode(run, f"{COMPONENT}-untouched-11111111", {"reward": 0.0},
                               {"ratio": 0.0})
    oracle = _write_episode(run, f"{COMPONENT}-oracle-22222222", {"reward": 1.0},
                            {"ratio": 1.02})
    _write_validation(run, untouched, oracle)
    passes = ob._read_passes(tmp_path, task_id=TASK_ID)
    assert passes.oracle == {"reward": 1.0} and passes.metrics == {"ratio": 1.02}
    # without validation.json the directory names still find this task's episodes
    (run / "validation.json").unlink()
    passes = ob._read_passes(tmp_path, task_id=TASK_ID)
    assert passes.oracle == {"reward": 1.0} and passes.metrics == {"ratio": 1.02}


def test_a_run_of_another_task_never_answers_for_this_one(tmp_path):
    """A newer run of a different task (another `ale validate` sharing the runs directory)
    is passed over for the newest run that holds this task; with none, there is no verdict."""
    ours = tmp_path / "validate-00000000"
    _write_episode(ours, f"{COMPONENT}-oracle-22222222", {"reward": 1.0}, {"ratio": 1.02})
    theirs = tmp_path / "validate-11111111"
    other_untouched = _write_episode(theirs, f"{ORG}.other_task-untouched-33333333",
                                     {"reward": 0.0}, {"ratio": 0.0})
    other_oracle = _write_episode(theirs, f"{ORG}.other_task-oracle-44444444",
                                  {"reward": 0.0}, {"ratio": 0.0})
    _write_validation(theirs, other_untouched, other_oracle, failures=["oracle_not_full"],
                      name=f"{ORG}/other_task")
    os.utime(ours, (1, 1))
    passes = ob._read_passes(tmp_path, task_id=TASK_ID)
    assert passes.oracle == {"reward": 1.0} and passes.engine_failures == []
    assert ob._read_passes(tmp_path, task_id=TASK_ID, ignore=frozenset({ours.name})) is None
    # without a task id the newest run is read, whatever it holds
    assert ob._read_passes(tmp_path).engine_failures == ["oracle_not_full"]


# --------------------------------------------------------------------------- #
# `validate` + the classification over records shaped like real runs
# --------------------------------------------------------------------------- #
def _fake_validate_run(monkeypatch, write_run):
    """Replace the engine call: `write_run(runs_dir)` leaves what `ale validate` would."""
    calls = []

    def fake_engine(argv, *, timeout):
        calls.append(argv)
        if argv[0] == "validate":
            write_run(Path(argv[argv.index("--runs-dir") + 1]))
        return subprocess.CompletedProcess(argv, 2 if argv[0] == "validate" else 0, "", "")

    monkeypatch.setattr(ob, "_engine", fake_engine)
    return calls


def test_a_run_shaped_like_validate_44ad164a_is_built(monkeypatch, tmp_path):
    """The records of the docking task's real 1cdbc02 run (untouched 0.0, oracle 1.0, ratio
    1.02), reduced to what the gate reads."""
    task_dir = _task(tmp_path, {"name": "mean_docking_quality", "direction": "higher"})

    def write_run(runs_dir):
        run = runs_dir / "validate-44ad164a"
        untouched = _write_episode(
            run, f"{COMPONENT}-untouched-a03bd197", {"reward": 0.0},
            {"ratio": 0.0, "anchor_recorded": 1.0, "reward_exact": 0.0},
            diagnostic="scoring 0: no submission at /home/user/submission/run.sh")
        oracle = _write_episode(
            run, f"{COMPONENT}-oracle-96b77314", {"reward": 1.0},
            {"ratio": 1.020742606065326, "anchor_recorded": 1.0,
             "mean_docking_quality": 0.7594008484285272, "probes_total": 690.0,
             "probes_passed": 410.0, "match_lock_ok": 1.0, "episodes_completed": 48.0,
             "episodes_aborted": 0.0, "docking_successes": 37.0, "target_collisions": 4.0,
             "reward_exact": 1.0})
        _write_validation(run, untouched, oracle)

    calls = _fake_validate_run(monkeypatch, write_run)
    result = ob.run_gates(task="cls", task_dir=task_dir, repo=tmp_path,
                          runs_dir=tmp_path / "runs")
    assert [argv[0] for argv in calls] == ["lint", "validate"]
    assert result.status == "built", result.detail
    assert result.reward == 1.0 and result.ratio == pytest.approx(1.0207, abs=1e-4)
    assert result.metric == "mean_docking_quality"
    assert result.measured == pytest.approx(0.7594, abs=1e-4)
    assert result.match_lock_ok is True and result.untouched_zero is True


def test_a_null_anchor_run_reads_as_needs_anchor(monkeypatch, tmp_path):
    task_dir = _task(tmp_path)

    def write_run(runs_dir):
        run = runs_dir / "validate-0a0a0a0a"
        untouched = _write_episode(
            run, f"{COMPONENT}-untouched-11111111", {"reward": 0.0},
            {"ratio": 0.0, "anchor_recorded": 0.0, "reward_exact": 0.0},
            diagnostic="scoring 0: no submission at /home/user/submission/run.sh")
        oracle = _write_episode(
            run, f"{COMPONENT}-oracle-22222222", {"reward": 0.0},
            {"ratio": 0.0, "anchor_recorded": 0.0, "mean_goal_distance": 0.1024,
             "match_lock_ok": 1.0, "reward_exact": 0.0})
        _write_validation(run, untouched, oracle, failures=["oracle_not_full"])

    _fake_validate_run(monkeypatch, write_run)
    result = ob.run_gates(task="cls", task_dir=task_dir, repo=tmp_path,
                          runs_dir=tmp_path / "runs")
    assert result.status == "needs_anchor"
    assert result.measured == 0.1024 and result.metric == "mean_goal_distance"
    assert result.engine_failures == ["oracle_not_full"]


def test_validate_ignores_a_run_that_was_there_before(monkeypatch, tmp_path):
    """An engine that stops before its first episode (an image that does not build) writes
    no run directory: an earlier run's `built` records must not answer for it."""
    task_dir = _task(tmp_path)
    runs = tmp_path / "runs"
    old = runs / "validate-00000000"
    untouched = _write_episode(old, f"{COMPONENT}-untouched-11111111", {"reward": 0.0},
                               {"ratio": 0.0, "anchor_recorded": 1.0})
    oracle = _write_episode(old, f"{COMPONENT}-oracle-22222222", {"reward": 1.0},
                            {"ratio": 1.1, "mean_goal_distance": 0.09, "anchor_recorded": 1.0})
    _write_validation(old, untouched, oracle)
    _fake_validate_run(monkeypatch, lambda runs_dir: None)      # nothing written this time
    result = ob.run_gates(task="cls", task_dir=task_dir, repo=tmp_path, runs_dir=runs)
    assert result.status == "needs_input"
    assert result.reward is None and "no verdict" in result.detail
