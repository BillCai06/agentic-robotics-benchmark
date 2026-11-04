"""Hidden grader for fault-adaptive spacecraft docking."""

import math
import os
import pwd
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import FaultAdaptiveDocking  # noqa: E402
from robotics_grader import (  # noqa: E402
    ProbeLock,
    Solver,
    SolverAborted,
    SolverMissing,
    agent_user,
    ratio_rewards,
    read_anchor,
    seed_rng,
    stage_config,
    write_verdict,
    zero_verdict,
)

SUBMISSION = os.path.join(os.environ.get("ALE_HOME", "/home/user"), "submission")


def own_submission(path):
    """Hand the restored artifact back to the agent account.

    In the separate verifier the engine restores /home/user/submission as root (every
    artifact write is the framework's), so its files land root-owned with their recorded
    modes. The kit Solver runs `bash run.sh` AS THE AGENT: the agent must be able to read
    its own controller (a file it had left non-world-readable would otherwise be
    unreadable) and write scratch beside it, exactly as it could while it worked. Walk the
    tree without following links — a freshly restored tree has none, but lchown is the safe
    habit — and give every entry, then the folder itself, to the agent. A no-op when the
    path is missing or is a link; the Solver then scores a real zero through SolverMissing.
    Runs as root, which the verify stage is.
    """
    if not os.path.isdir(path) or os.path.islink(path):
        return
    try:
        account = pwd.getpwnam(agent_user())
    except KeyError:
        return
    uid, gid = account.pw_uid, account.pw_gid
    for dirpath, dirnames, filenames in os.walk(path, followlinks=False):
        for name in dirnames + filenames:
            try:
                os.lchown(os.path.join(dirpath, name), uid, gid)
            except OSError:
                pass
    os.lchown(path, uid, gid)


def perturbed_observation(obs, rng):
    changed = list(obs)
    changed[0] += rng.uniform(-0.28, 0.28)
    changed[1] += rng.uniform(-0.28, 0.28)
    yaw = math.atan2(changed[3], changed[2]) + rng.uniform(-0.18, 0.18)
    changed[2], changed[3] = math.cos(yaw), math.sin(yaw)
    return changed


def run_episodes(config, rng):
    episodes = int(config.get("episodes", 48))
    lock = ProbeLock(
        rng,
        rate=config.get("probe_rate", 0.06),
        pass_fraction=config.get("probe_pass_fraction", 0.35),
        tolerance=1e-4,
    )
    qualities = []
    successes = collisions = aborted = 0
    # Raises SolverMissing (no run.sh) or SolverAborted (it never comes up); main() scores both 0.
    solver = Solver(step_timeout=config.get("step_timeout_s", 30))
    try:
        for episode in range(episodes):
            env = FaultAdaptiveDocking()
            quality = 0.0
            info = {"success": False, "collision": False, "failure_reason": "not_started"}
            try:
                # 64-bit: the agent holds this simulator's source, so a 31-bit seed could
                # be found by brute force from the first observations.
                obs = env.reset(rng.seed(bits=64))
                solver.reset()
                for t in range(env.max_steps):
                    if lock.due():
                        action = lock.probe(
                            solver, t, obs, perturbed=perturbed_observation(obs, rng)
                        )
                    else:
                        action = solver.act(t, obs)
                    obs, quality, done, info = env.step(action)
                    if done:
                        break
            except (SolverAborted, ValueError, TypeError, OverflowError) as exc:
                aborted += 1
                quality = 0.0
                info = {"success": False, "collision": False,
                        "failure_reason": "aborted: {}".format(exc)}
            qualities.append(float(quality))
            successes += int(bool(info.get("success")))
            collisions += int(bool(info.get("collision")))
            print(
                "[grader] episode {}: quality={:.4f} success={} reason={}".format(
                    episode, quality, bool(info.get("success")),
                    info.get("failure_reason", "")
                ),
                file=sys.stderr,
                flush=True,
            )
    finally:
        solver.close()
    return qualities, successes, collisions, aborted, lock


def main():
    config = stage_config()
    rng = seed_rng()
    anchor, anchor_meta = read_anchor()
    # Separate verifier: the restored submission is root-owned; hand it to the agent before
    # the kit Solver runs `bash run.sh` as the agent.
    own_submission(SUBMISSION)
    try:
        qualities, successes, collisions, aborted, lock = run_episodes(config, rng)
    except (SolverMissing, SolverAborted) as exc:
        # A missing submission, or one that never comes up, is a real zero, not a broken task.
        write_verdict(zero_verdict(reason=str(exc), anchored=anchor is not None))
        return

    measured = sum(qualities) / len(qualities) if qualities else 0.0
    verdict = ratio_rewards(
        measured,
        anchor,
        cap=config.get("scoring_cap", 1.5),
        higher_is_better=True,
        metric_name="mean_docking_quality",
        locks_ok=lock.ok,
        extra=dict(
            lock.rewards(),
            episodes_completed=len(qualities) - aborted,
            episodes_aborted=aborted,
            docking_successes=successes,
            target_collisions=collisions,
        ),
        full_at=anchor_meta.get("full_at", 1.0),
    )
    write_verdict(verdict)
    print(
        "[grader] measured={:.5f} anchor={} ratio={:.3f} reward={:.3f} "
        "lock={} successes={}/{} aborted={}".format(
            measured, anchor, verdict["metrics"]["ratio"],
            verdict["rewards"]["reward"], lock.ok, successes, len(qualities), aborted
        ),
        file=sys.stderr,
        flush=True,
    )


if __name__ == "__main__":
    main()
