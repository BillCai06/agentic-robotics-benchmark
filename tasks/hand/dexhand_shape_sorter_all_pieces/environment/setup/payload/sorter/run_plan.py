"""Run one plan on one layout and print the outcome JSON (the practice grader and the verifier use this).

usage: python -m sorter.run_plan <layout.json> <plan.json> [--out outcome.json]
"""
import json
import sys
import time

from . import plan as P
from . import world as W


def main():
    lay = json.load(open(sys.argv[1]))
    with open(sys.argv[2], "rb") as fh:
        wps = P.parse_bytes(fh.read())
    t0 = time.time()
    res = W.run_plan(lay, wps, progress=lambda k: print("[run_plan] waypoint %d/%d  %.0fs" % (k, len(wps), time.time() - t0),
                                                       file=sys.stderr, flush=True))
    res["wall_s"] = round(time.time() - t0, 1)
    res["waypoints"] = len(wps)
    s = json.dumps(res, indent=1)
    if "--out" in sys.argv:
        open(sys.argv[sys.argv.index("--out") + 1], "w").write(s)
    print(s)


if __name__ == "__main__":
    main()
