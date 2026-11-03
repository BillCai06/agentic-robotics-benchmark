"""Practice grader — a self-test you can run while you work.

Drives your submission over the same wire the real grader uses — the same request shape,
the same deadlines (READY within 120 s of launch, each reply within 30 s), and the same kind
of repeated-`t` queries — on VISIBLE practice seeds, and prints the raw metric it observes.
It deliberately tells you less than the real grader:

* practice seeds, not the hidden ones the real run samples;
* the raw metric only — no reference value, no relative result;
* no import of anything on the grading side, because none of it is here.

So: "my policy works and roughly how well it does", not "what I will be scored".

    python3 /home/user/practice_grader/grade.py [--episodes 3]
"""

import argparse
import json
import os
import queue
import random
import subprocess
import sys
import threading

sys.path.insert(0, "/home/user")

from template_env import TemplateReach  # noqa: E402

PRACTICE_SEEDS = [11, 22, 33, 44, 55]

# The real grader's deadlines: READY within four step deadlines of launch, then one step
# deadline per reply. Miss either there and the episode is aborted.
STEP_TIMEOUT_S = 30.0
READY_TIMEOUT_S = 4 * STEP_TIMEOUT_S
MAX_PRE_READY_LINES = 200

# Steps asked twice with the same `t` — once with the real observation, once with the
# position components shifted, in random order. Only the real reply is applied. The real
# grader does the same at hidden steps and expects the two answers to differ.
REPEAT_STEPS = (3, 77, 150)
PERTURB = 0.4


class WireError(Exception):
    """Your solver broke the protocol (deadline, non-JSON, no action)."""


class Wire(object):
    """Your `run.sh`, spoken to over stdio exactly as the real grader does."""

    def __init__(self, submission):
        self.proc = subprocess.Popen(["bash", "run.sh"], cwd=submission,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     universal_newlines=True, bufsize=1)
        self._lines = queue.Queue()
        threading.Thread(target=self._pump, daemon=True).start()

    def _pump(self):
        for line in self.proc.stdout:
            self._lines.put(line)
        self._lines.put(None)

    def _readline(self, timeout):
        try:
            line = self._lines.get(timeout=timeout)
        except queue.Empty:
            raise WireError("no reply within {:.0f}s — the real grader aborts the episode "
                            "here".format(timeout))
        if line is None:
            raise WireError("your solver closed its stdout (did it crash? check stderr)")
        return line

    def await_ready(self):
        for _ in range(MAX_PRE_READY_LINES):
            if self._readline(READY_TIMEOUT_S).strip() == "READY":
                return
        raise WireError("your solver never printed READY ({} lines of other output "
                        "seen)".format(MAX_PRE_READY_LINES))

    def request(self, message):
        try:
            self.proc.stdin.write(json.dumps(message) + "\n")
            self.proc.stdin.flush()
        except (IOError, OSError):
            raise WireError("your solver's stdin is closed")
        while True:
            line = self._readline(STEP_TIMEOUT_S).strip()
            if not line:
                continue
            try:
                reply = json.loads(line)
            except ValueError:
                raise WireError("non-JSON on stdout: {!r} — stdout is the wire; send "
                                "diagnostics to stderr".format(line[:120]))
            if not isinstance(reply, dict):
                raise WireError("reply is not a JSON object: {!r}".format(line[:120]))
            return reply

    def act(self, t, obs):
        reply = self.request({"type": "act", "t": t, "obs": obs})
        if "action" not in reply:
            raise WireError("solver error at t={}: {}".format(t, reply.get("error", reply)))
        return reply["action"]

    def close(self):
        try:
            self.proc.stdin.write(json.dumps({"type": "close"}) + "\n")
            self.proc.stdin.flush()
        except (IOError, OSError):
            pass
        try:
            self.proc.wait(timeout=10)
        except Exception:  # noqa: BLE001
            self.proc.kill()


def _differs(a, b, tolerance=1e-4):
    fa = [float(x) for x in a]
    fb = [float(x) for x in b]
    return len(fa) != len(fb) or max([abs(x - y) for x, y in zip(fa, fb)] or [0.0]) > tolerance


def _episode(wire, seed, rng, stats):
    """One episode's mean distance to the goal — the metric, lower is better."""
    env = TemplateReach()
    obs = env.reset(seed)
    wire.request({"type": "reset"})
    total, steps = 0.0, 0
    for t in range(env.max_steps):
        if t in REPEAT_STEPS:
            perturbed = list(obs)
            perturbed[0] += rng.uniform(-PERTURB, PERTURB)
            perturbed[1] += rng.uniform(-PERTURB, PERTURB)
            if rng.random() < 0.5:
                other = wire.act(t, perturbed)
                action = wire.act(t, obs)
            else:
                action = wire.act(t, obs)
                other = wire.act(t, perturbed)
            stats["repeated_t_queries"] += 1
            if _differs(action, other):
                stats["repeated_t_answers_differed"] += 1
        else:
            action = wire.act(t, obs)
        obs, reward, done = env.step(action)   # only the reply to the real obs is applied
        total += -reward                        # reward is the negative distance
        steps += 1
        if done:
            break
    return total / steps if steps else float("inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", default=os.environ.get(
        "SUBMISSION_DIR", "/home/user/submission"))
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    entry = os.path.join(args.submission, "run.sh")
    if not os.path.isfile(entry):
        print("[practice] no run.sh at {}".format(args.submission))
        return 2

    rng = random.Random(2024)
    stats = {"repeated_t_queries": 0, "repeated_t_answers_differed": 0}
    wire = Wire(args.submission)
    try:
        wire.await_ready()
        distances = [_episode(wire, seed, rng, stats)
                     for seed in PRACTICE_SEEDS[:args.episodes]]
    except WireError as exc:
        print("[practice] {}".format(exc))
        return 1
    finally:
        wire.close()

    if stats["repeated_t_queries"] and not stats["repeated_t_answers_differed"]:
        print("[practice] WARNING: your action never changed when the observation was "
              "perturbed at a repeated t. The real grader treats a policy that ignores its "
              "input as a zero regardless of its metric.", file=sys.stderr)

    mean = sum(distances) / len(distances) if distances else float("inf")
    print(json.dumps({
        "metric": "mean_goal_distance",
        "direction": "lower",
        "raw_value": round(mean, 4),
        "per_episode": [round(d, 4) for d in distances],
        "repeated_t_queries": stats["repeated_t_queries"],
        "repeated_t_answers_differed": stats["repeated_t_answers_differed"],
        "note": "practice seeds only; this is not your score",
    }))
    return 0


if __name__ == "__main__":
    sys.exit(main())
