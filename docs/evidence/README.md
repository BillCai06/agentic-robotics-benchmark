# evidence/

Records of real runs. All of them were made on the engine before the Harbor migration
(2026-07-28 to 2026-09-06), with that engine's task layout (`task.yaml`, `verify/`,
`oracle/`) and record format, and they are kept as they were written.

| file | what it is |
|---|---|
| [`isolation-audit.md`](isolation-audit.md) | the isolation audit: what the tested agent can reach (the template, 2026-07-28) |
| `template-validate-untouched-result.json`, `template-validate-oracle-result.json` | the two `result.json` records of an `ale validate` run of the template (2026-09-06) |
| `drone_hover-validate-untouched-result.json`, `drone_hover-validate-oracle-result.json` | the same for the dummy `drone_hover` task (2026-08-22; the task was removed on 2026-10-15) |
| `drone_hover-validate-trace.jsonl` | the engine trace of a `drone_hover` oracle episode (2026-07-28) |

**2026-11-03 note.** None of these runs was repeated on the Harbor-based engine (`vendor/ale`
`1cdbc02`). What the audit shows was re-checked in that engine's code instead; the note at
the top of `isolation-audit.md` lists each check and where it now lives. The record format
changed in three ways that matter when reading these files next to a new run: an episode id
starts with the organization (`grounded-intelligence-initiative.<slug>-oracle-<hex8>` under
`ale validate`); `result.json`'s `reward` is rounded half up to two decimals; and the
grader's numbers (the raw metric, `ratio`, `match_lock_ok`, probe counts) are in the
episode's `verification.json` under `metrics`, no longer in `result.json`. The trace file
shows an older format still, in which every number the grader wrote was a reward key.
