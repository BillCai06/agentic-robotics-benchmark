# Scoring options (maintainers only; the grader never reads this file)

`grader_config.json` chooses how `acceptance_score` is made from the six sections:

- `"combine"`: `"weighted"` (the weighted mean by `sections`), `"mission_gate"` (the weighted mean of the five scene
  sections by their `sections` weights, times the mission section) or the name of another weights block,
  `sections_<name>` (today `"mission_0.8"`: the mission at 0.8, the five scene sections at 0.2 in their proportions);
- `"progress_gate"`: `true` or `false`. With it, each run's four safety checks (parts not dropped, no losses, no
  collisions, separation) are multiplied by that run's progress, (steam released + fire out + station cleared) / 3,
  and consistency by the two runs' mean progress.

Every verdict records all six combinations in its metrics (`acceptance.<combine>` and
`acceptance.<combine>.progress_gate`) and both mission sections (`section.control.progress_gate`,
`section.control.no_progress_gate`), so a switch needs no new measurement of the agents' runs. It does need two
edits besides the config line:

1. **`tests/anchor.json`**: the value and `full_at` are measured under one combination. Take the new value from the
   reference's full verifies under the new choice (the anchor's source note lists their verdicts; each records all six
   numbers) and size `full_at` again from the spread, as the note says.
2. **The agent's text** must describe the scoring that is used. The sentences to swap:

## `instruction.md`, "How it is graded"

- `mission_gate` (today):
  > ``acceptance_score`` is the scene's score times the mission's: the scene's score is the weighted mean of the first
  > five sections below (the weights in brackets), the mission's is the last section's score.

  and the mission's bullet starts "**The mission (it multiplies the scene's score)**:".
- `weighted`:
  > ``acceptance_score`` is a weighted mean (the weights in brackets):

  and the mission's bullet starts "**The mission (0.54)**:".
- `mission_0.8`:
  > ``acceptance_score`` is a weighted mean (the weights in brackets):

  with the brackets Layout (0.035), Physics (0.017), The steam line and the hazards (0.017), The rig (0.087), The
  fleet's design (0.043), and the mission's bullet starting "**The mission (0.80)**:".

## `environment/setup/payload/given/control.md`, "How it is scored"

- The second sentence of the section: `mission_gate`: "The mission's score (the checks' weighted mean) multiplies the
  scene's (`instruction.md`)."; `weighted`: "The section's score is the checks' weighted mean; it is 0.54 of
  `acceptance_score` (`instruction.md`)."; `mission_0.8`: the same with 0.80.
- `progress_gate: true` (today) keeps the sentence "The five checks marked * count in proportion to what each run got
  done: ..." and the five `*` marks in the table (parts not dropped, no losses, no collisions, separation,
  consistency); `progress_gate: false` removes both.
