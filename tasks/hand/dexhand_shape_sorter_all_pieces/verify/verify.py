"""Verify stage (root, separate verification sandbox): re-simulate every submitted plan on its hidden layout
(verify/ref/layouts.json) with exactly the public simulator code (verify/sorter = the payload's sorter package) and
score the outcome with the public rule (sorter/score.py): pieces that passed through their own hole.

Nothing the agent wrote is trusted beyond the plan numbers: each plan file is size-checked, opened without following
symlinks, parsed as plain JSON and validated by sorter/plan.py. The four instances run in parallel worker processes
(single-threaded SuperDex, bit-reproducible), under a wall-clock budget below timeouts.verify.
"""

import json
import math
import multiprocessing as mp
import os
import stat
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from robotics_grader import ratio_rewards, read_anchor, stage_config, write_verdict, zero_verdict  # noqa: E402
from robotics_grader.stage import agent_home  # noqa: E402

INSTANCES = ["inst_a", "inst_b", "inst_c", "inst_d"]
N_PIECES = 11


def log(msg):
    print("[verify] " + msg, file=sys.stderr, flush=True)


def read_plan(sub_dir, inst):
    """Return (waypoints, None) or (None, reason). Never follows symlinks, never raises."""
    from sorter import plan as P
    path = os.path.join(sub_dir, "plans", inst + ".json")
    try:
        for p in (sub_dir, os.path.join(sub_dir, "plans")):
            if not stat.S_ISDIR(os.lstat(p).st_mode):
                return None, "submission path is not a plain directory"
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_NONBLOCK", 0))
    except FileNotFoundError:
        return None, "missing"
    except OSError:
        return None, "unreadable or a symlink"
    try:
        st = os.fstat(fd)
        if not stat.S_ISREG(st.st_mode):
            return None, "not a regular file"
        if st.st_size > P.MAX_BYTES:
            return None, "too large"
        data = b""
        while len(data) <= P.MAX_BYTES:
            chunk = os.read(fd, 1 << 20)
            if not chunk:
                break
            data += chunk
    finally:
        os.close(fd)
    try:
        wps = P.parse_bytes(data)
    except P.PlanError as exc:
        return None, "invalid plan: " + str(exc)[:160]
    except Exception as exc:  # any parser surprise is the agent's invalid input, never a task error
        return None, "invalid plan (%s)" % type(exc).__name__
    return [(list(map(float, p)), list(map(float, q)), list(map(float, f))) for p, q, f in wps], None


def _worker(args):
    inst, layout, wps = args
    import numpy as np
    from sorter import world as W
    t0 = time.time()
    res = W.run_plan(layout, [(np.array(p), np.array(q), np.array(f)) for p, q, f in wps])
    res["wall_s"] = time.time() - t0
    return inst, res


def main():
    t0 = time.time()
    # Refuse to grade on a machine whose numerics differ from the reference: its scores would not be comparable
    # (sorter/numerics.py). Raising here makes the run a task error, never a score.
    from sorter import numerics
    numerics.check()
    log("numerics match the reference (OpenBLAS Haswell kernels, NumPy X86_V3, box SDF, physics probe)")
    config = stage_config()
    anchor, meta = read_anchor()
    budget = float(config.get("grading_budget_s", 4800))
    with open(os.path.join(HERE, "ref", "layouts.json")) as fh:
        layouts = json.load(fh)
    sub_dir = os.path.join(agent_home(), "submission")
    if not os.path.isdir(sub_dir) or os.path.islink(sub_dir):
        write_verdict(zero_verdict(reason="no submission directory", anchored=anchor is not None))
        return
    plans, reasons = {}, {}
    for inst in INSTANCES:  # snapshot every plan before any simulation
        plans[inst], reasons[inst] = read_plan(sub_dir, inst)
        log("%s: %s" % (inst, "valid, %d waypoints" % len(plans[inst]) if plans[inst] is not None else reasons[inst]))
    jobs = [(inst, layouts[inst], plans[inst]) for inst in INSTANCES if plans[inst] is not None]
    results = {}
    if jobs:
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(processes=min(4, len(jobs)), maxtasksperchild=1)
        try:
            pending = [pool.apply_async(_worker, (j,)) for j in jobs]
            for job, fut in zip(jobs, pending):
                left = budget - (time.time() - t0)
                try:
                    inst, res = fut.get(timeout=max(1.0, left))
                    results[inst] = res
                except mp.TimeoutError:
                    log("%s: grading budget exhausted" % job[0])
                except Exception as exc:
                    log("%s: simulation failed (%s)" % (job[0], type(exc).__name__))
        finally:
            pool.terminate()
            pool.join()
    scores, extra = {}, {}
    for inst in INSTANCES:
        r = results.get(inst)
        n = int(r["passed"]) if r is not None else 0
        s = n / float(N_PIECES)
        if not math.isfinite(s) or s < 0:
            s = 0.0
        scores[inst] = s
        extra["%s_passed" % inst] = float(n)
        if r is not None:
            extra["%s_nonfinite" % inst] = 1.0 if r.get("nonfinite") else 0.0
            extra["%s_wall_s" % inst] = round(float(r.get("wall_s", 0.0)), 1)
            log("%s: passed %d/%d %s state %s" % (inst, n, N_PIECES,
                json.dumps({k: (v["passed"], v["entry"]) for k, v in r["pieces"].items()}), r.get("state_sha256", "")[:16]))
    measured = sum(scores.values()) / float(len(INSTANCES))
    extra["plans_valid"] = float(sum(p is not None for p in plans.values()))
    extra["grading_wall_s"] = round(time.time() - t0, 1)
    verdict = ratio_rewards(measured, anchor, cap=config.get("scoring_cap", 1.5), higher_is_better=True,
                            metric_name="fraction_pieces_passed", extra=extra, full_at=meta.get("full_at", 1.0))
    write_verdict(verdict)
    log("measured=%.5f anchor=%s reward=%.4f" % (measured, anchor, verdict["rewards"]["reward"]))


if __name__ == "__main__":
    main()
