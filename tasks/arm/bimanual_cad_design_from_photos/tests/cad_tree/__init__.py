"""cad_tree: grading a CAD document of a mechanism with a kinematic tree and closed loops (first user:
bimanual_cad_design_from_photos; contracts in that task's development record). Vendored byte-identically into the task's
grader folder (tests/), so its modules import only the standard library, numpy / scipy / trimesh / manifold3d /
cadquery and each other (relative imports), never the dev packages.
  kinematics  forward kinematics of a tree with four-bar loops (poses from configuration Z)
  extract     the FreeCAD-side extraction of a submitted document (runs under freecadcmd; FreeCAD + stdlib only)
  convert     the extraction -> a unit folder in the reference's conventions (parts, bought classes, links, joints)
  checks      geometry and score helpers (tessellation, ICP, class registry, CPU limits)
  features    cut-out and spacer scoring
  grade       per-arm criteria, floors, gates and the design score (rules.json holds every number)
  synth, tests/
              dev only: the synthetic oracle and mutants, the test suite (not vendored)
"""
