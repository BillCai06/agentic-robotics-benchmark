"""Validate a plan file against the grader's rules without simulating. usage: python3 -m sorter.check_plan plan.json"""
import sys

from . import plan as P


def main():
    with open(sys.argv[1], "rb") as fh:
        try:
            wps = P.parse_bytes(fh.read())
        except P.PlanError as exc:
            print("INVALID:", exc)
            sys.exit(1)
    print("valid: %d waypoints (%.2f s of motion)" % (len(wps), len(wps) / 25.0))


if __name__ == "__main__":
    main()
