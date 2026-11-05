"""Maintainers' check: the agent's copy of the grader's structure check is the grader's, byte for byte
(environment/setup/payload/given/check_structure.py = tests/scan.py).

    python3 tests/review/check_given_copies.py
"""

import filecmp
import os
import sys

TASK = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GIVEN = os.path.join(TASK, "environment", "setup", "payload", "given")
PAIRS = (("tests/scan.py", "check_structure.py"),)

bad = [(t, g) for t, g in PAIRS if not filecmp.cmp(os.path.join(TASK, t), os.path.join(GIVEN, g), shallow=False)]
for t, g in bad:
    print(f"DIFFERS: {t} vs given/{g}")
print("OK" if not bad else "PROBLEMS")
sys.exit(1 if bad else 0)
