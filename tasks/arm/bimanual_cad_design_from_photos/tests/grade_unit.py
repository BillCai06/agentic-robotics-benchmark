"""The verify stage's grading child (bimanual_cad_design_from_photos): the extracted document converted into a unit and
graded against the reference with cad_tree, in a process of its own so that verify.py can bound it with a time limit
(an OCC call that hangs on a broken shape cannot be interrupted from Python). Root, verify stage; reads data only.

    python3 -B grade_unit.py EXTRACT_DIR REF_DIR BOUGHT_DIR CONV_DIR SEED_HI:SEED_LO OUT_JSON [SUBMISSION_DIR]

OUT_JSON: {"ok", "rewards", "stats", "details", "conversion", "error"?}. The thread pools of manifold3d and OCC are
limited first (cad_tree.checks.limit_cpus, ARB_GRADER_CPUS, default 8): they size themselves to the host otherwise.
"""
import json
import os
import sys
import time
import traceback

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main(argv):
    ex, ref_dir, bought, conv_dir, seeds, out = argv[1:7]
    sub_dir = argv[7] if len(argv) > 7 else None
    res = {"ok": False, "rewards": {}, "stats": {}, "details": {}, "conversion": {}}
    t0 = time.time()
    try:
        from cad_tree import checks as C
        C.limit_cpus(int(os.environ.get("ARB_GRADER_CPUS", "8")))
        import numpy as np
        from cad_tree import convert as CV
        from cad_tree import grade as G
        classes = C.load_classes(bought)
        ref = G.load_reference(ref_dir, classes)
        t1 = time.time()
        conv = CV.convert(ex, ref_dir, bought, conv_dir, classes=classes)
        t2 = time.time()
        hi, lo = [int(x) for x in seeds.split(":")]
        blocklist = G.load_blocklist(os.path.join(HERE, "given", "public_blocklist.json"))
        rewards, stats, details = G.grade(conv_dir, ref, np.random.SeedSequence([hi, lo]), conversion=conv,
                                          blocklist=blocklist, submission_dir=sub_dir)
        stats.update(load_s=t1 - t0, convert_s=t2 - t1, grade_total_s=time.time() - t2)
        res.update(ok=True, rewards=rewards, stats=stats, details=details,
                   conversion={k: conv.get(k) for k in ("ok", "errors", "notes", "arms", "stats", "alignment")})
    except Exception:  # noqa: BLE001 - recorded; verify.py scores the submission 0 with the reason
        res["error"] = traceback.format_exc()[-2000:]
    with open(out, "w") as fh:
        json.dump(res, fh, default=float)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
