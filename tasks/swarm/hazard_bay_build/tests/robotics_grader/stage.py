"""Verify-stage basics: where things are, who runs them, and how a run ends.

The framework tells a stage script where it is (`ALE_STAGE_DIR`) and whose home it works
on (`ALE_HOME`) through the environment, and the result goes to Harbor's verifier directory;
this module is the only place that knows them, so a task never hardcodes a path the
framework owns.
"""

import json
import os
import subprocess
import sys

#: Harbor's verifier directory: `reward.json` (the score the engine reads) and
#: `verification.json` (the complete result) go here.
VERIFIER_DIR = "/logs/verifier"

__all__ = [
    "VERIFIER_DIR",
    "agent_user",
    "deprivileged",
    "fail",
    "require",
    "stage_dir",
    "verdict_path",
    "write_verdict",
]


def stage_dir():
    """This stage's folder inside the sandbox (`/opt/ale/tests`: root-only, like all of /opt/ale).

    Assets that must not exist while the agent works — the anchor, grader config, hidden
    reference data — live here, because a stage folder is copied in when its stage runs.
    """
    return os.environ.get("ALE_STAGE_DIR", os.path.dirname(os.path.abspath(sys.argv[0])))


def verdict_path():
    """Harbor's reward file, where the engine reads the score back.

    The agent's UID cannot pre-create or overwrite it: the engine empties `/logs/verifier`
    and gives it to the framework alone before every verification attempt. Write it through
    `write_verdict`, never directly: the file holds a flat object of numbers, not the
    kit's `{"rewards": ..., "metrics": ...}` envelope.
    """
    return os.path.join(VERIFIER_DIR, "reward.json")


def agent_home():
    """The unprivileged account's home — the workspace, per the image's declared user."""
    return os.environ.get("ALE_HOME", "/home/user")


def agent_user():
    """The account the agent ran as, derived from its home.

    task.toml's `[agent].user` names it (default `user`) and the framework passes its home;
    taking the leaf of the home keeps the two consistent without a second source of truth.
    """
    return os.path.basename(agent_home().rstrip("/")) or "user"


def fail(reason):
    """End the run as a TASK ERROR, not as a zero.

    The distinction is load-bearing: a zero says the agent did not solve the task, and a
    task error says the task is broken. Conflating them lets a defective grader read as a
    stream of legitimately failing agents.
    """
    print("[grader] TASK ERROR: {}".format(reason), file=sys.stderr, flush=True)
    raise SystemExit(1)


def require(condition, reason):
    """`fail` unless the condition holds — for the grader's own preconditions."""
    if not condition:
        fail(reason)


def write_verdict(verdict, path=None, diagnostic=None):
    """Write the verdict the framework reads back.

    `verdict` is `{"rewards": {...}, "metrics": {...}}`. The split is load-bearing under
    `ale validate`: `reward` must be exactly 0 on an untouched sandbox and exactly 1 for the
    oracle (two decimals, as the engine rounds it), so `rewards` carries only the gated
    score, the single key `reward` in [0, 1], and everything worth keeping but not gating
    (the raw metric, the capped ratio, lock outcomes, abort counts) goes in `metrics`.

    With no `path`, inside the engine's verify stage (`ALE_VERIFY_PYTHON` is set), the result
    goes through the engine's `ale_verify.write`, which it stages into the verify-stage
    interpreter. It writes Harbor's `/logs/verifier/reward.json` (`{"reward": ...}`, the only
    value the engine reads) and `verification.json` beside it: the complete result, with
    `metrics`, the exact unrounded reward as `metrics.reward_exact`, and the grader's
    `diagnostic` text when there is one (the argument, or the verdict's own `diagnostic`).

    With a `path`, the plain envelope is written there instead, for runs outside a sandbox
    (the kit's selftest, a host-side dry run). A grader run by hand outside the engine, such
    as a maintainer re-measuring an anchor inside the task image, gets the same through the
    `ROBOTICS_GRADER_VERDICT_PATH` environment variable without changing the grader.
    """
    rewards = verdict.get("rewards") or {}
    metrics = verdict.get("metrics") or {}
    require("reward" in rewards, "rewards map has no primary 'reward' key")
    require(all(0.0 <= float(v) <= 1.0 for v in rewards.values()),
            "every gated reward must lie in [0, 1]; report raw values under metrics")
    rewards = {k: float(v) for k, v in rewards.items()}
    metrics = {k: float(v) for k, v in metrics.items()}
    diagnostic = diagnostic if diagnostic is not None else verdict.get("diagnostic")

    if path is None and not os.environ.get("ALE_VERIFY_PYTHON"):
        path = os.environ.get("ROBOTICS_GRADER_VERDICT_PATH") or None
    if path is None:
        try:
            from ale_verify import write
        except ImportError:
            fail("ale_verify is not importable: the engine stages it into the verify-stage "
                 "interpreter (3.12+, named by ALE_VERIFY_PYTHON); outside the engine pass a "
                 "path or set ROBOTICS_GRADER_VERDICT_PATH")
        result = {"reward": rewards["reward"],
                  "metrics": dict(metrics, reward_exact=rewards["reward"])}
        if len(rewards) > 1:
            result["rewards"] = rewards
        if diagnostic:
            result["diagnostic"] = str(diagnostic)
        write(result)   # reward.json and verification.json, atomically, reward rounded
        return

    envelope = {"rewards": rewards, "metrics": metrics}
    if diagnostic:
        envelope["diagnostic"] = str(diagnostic)
    with open(path, "w") as handle:
        json.dump(envelope, handle)


def deprivileged(argv, **kwargs):
    """Run `argv` as the agent account instead of as root.

    Two callers need this and both are security-critical:

    * a live solver the agent wrote (closed_loop) — at root it could read the grader's
      memory, environment and `/opt/ale`;
    * deserializing an artifact the agent produced (artifact_rollout) — `torch.load` and
      `pickle` execute arbitrary code, so doing it as root hands over the grader.

    Yama's default `ptrace_scope` blocks tracing across UIDs, so the boundary holds even
    though both processes share a sandbox.
    """
    return subprocess.Popen(  # noqa: S603 — caller-fixed argv, no shell
        ["runuser", "-u", agent_user(), "--"] + list(argv), **kwargs
    )
