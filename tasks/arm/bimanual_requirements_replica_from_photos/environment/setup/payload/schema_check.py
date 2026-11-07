#!/usr/bin/env python3
"""Structure check (gate G0) of a submission: the three files parse, validate against schemas/, ids unique, one module
line with count 1 per arm and joint (j1..j7, j8 = the gripper), one PSU line, every part on the quote (price_list.md)
with the quote's role, bus 24 or 48 V, finite numbers, sketch module ids = the BOM's lines. Uses the public grading
modules in grader_public/; reads no ground truth and prints nothing about any instance.

    python3 schema_check.py [SUBMISSION_DIR]
"""
import os
import sys

sys.dont_write_bytecode = True
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "grader_public"))
ROLES = ("module", "bus_board", "psu")


def quote_roles(path=os.path.join(HERE, "price_list.md")):
    """{part_id: role} from the quote's table rows (| part_id | role | ...)."""
    roles = {}
    with open(path) as fh:
        for line in fh:
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 2 and cells[1] in ROLES:
                roles[cells[0]] = cells[1]
    return roles


def main():
    sub = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "submission")
    from arm.req_bimanual import grade as GR
    try:
        docs, problems = GR.load_submission(sub, quote_roles())
    except Exception as e:  # noqa: BLE001  (values the structure rules cannot read fail G0, as in the grader)
        docs, problems = {}, ["the submission cannot be checked: %s: %s" % (type(e).__name__, str(e)[:200])]
    if problems:
        print("G0 FAILS:")
        for p in problems:
            print("  -", p)
        return 1
    print("G0 ok: requirements.yaml (%d items), bom.json (%d lines), kinematic_sketch.json" % (
        len(docs["requirements"]["requirements"]), len(docs["bom"]["parts"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
