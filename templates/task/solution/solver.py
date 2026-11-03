"""The reference policy, serving the closed_loop wire.

A proportional-derivative controller on the goal error: accelerate toward the goal, damp
with the current velocity. Deliberately simple and genuinely obs-dependent — a policy that
ignored its observation would fail the grader's probe lock, which is exactly the point of
having one. Stateless, so a step asked twice with the same `t` is answered twice from the
observation each request carries — repeated queries never advance anything here.

At grading time a real task's solver only LOADS its trained artifact here; it never trains.
Training happened during the agent phase, under the agent's budget.
"""

import json
import sys

KP = 2.4   # pull toward the goal
KD = 1.6   # damp the approach so it settles instead of orbiting


def act(observation):
    x, y, vx, vy, gx, gy = observation
    return [
        _clip(KP * (gx - x) - KD * vx),
        _clip(KP * (gy - y) - KD * vy),
    ]


def _clip(value, limit=1.0):
    return max(-limit, min(limit, value))


def main():
    # READY first, then one JSON object per line. stdout is the wire; anything else a
    # solver wants to say belongs on stderr.
    print("READY", flush=True)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except ValueError:
            print(json.dumps({"error": "bad json"}), flush=True)
            continue
        kind = request.get("type")
        if kind == "reset":
            # Stateless controller: nothing to clear between episodes.
            print(json.dumps({"ok": True}), flush=True)
        elif kind == "act":
            try:
                print(json.dumps({"action": act(request["obs"])}), flush=True)
            except Exception as exc:  # noqa: BLE001 — report on the wire, keep serving
                print(json.dumps({"error": str(exc)[:200]}), flush=True)
        elif kind == "close":
            print(json.dumps({"ok": True}), flush=True)
            return
        else:
            print(json.dumps({"error": "unknown request"}), flush=True)


if __name__ == "__main__":
    main()
