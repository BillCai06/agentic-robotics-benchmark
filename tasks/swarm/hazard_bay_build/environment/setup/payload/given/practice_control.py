"""Practice runner for the bay mission: your scene and your controller through the same simulation, the same random
draws and the same measurements the grader uses (`control_sim.py` is the grader's copy), in your own session, one
episode, raw numbers. The day's bay comes from `--seed` as the grader's does from its own seeds: your parts placed at
random on your tables, your vehicle moved off its authored pose (its odometry starting from the authored one), your
lever partly turned, your pan moved. It knows only your scene: it runs in your bay, with your ropes, and measures
against your scene's plume, crate, gauge and tables. The grader runs the mission in the real bay instead
(`control.md`, The real bay).

    /opt/venv/bin/python3 ~/given/practice_control.py [--scene /home/user/submission/scene.usda] [--submission /home/user/submission] [--seed 0]

The grader draws its own seeds, runs your controller as a separate account behind the same wire, and scores the
numbers this prints (`control.md` says how); this runner prints them and scores nothing.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
ap = argparse.ArgumentParser()
ap.add_argument("--scene", default="/home/user/submission/scene.usda")
ap.add_argument("--submission", default="/home/user/submission")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--time-limit", type=float, default=None, help="stop the mission earlier (seconds of simulation)")
ap.add_argument("--catalog", default=os.path.join(HERE, "catalog", "catalog.json"))
args = ap.parse_args()

from isaacsim import SimulationApp  # noqa: E402

app = SimulationApp({"headless": True})
import omni.kit.app  # noqa: E402

for ext in ("isaacsim.core.nodes", "isaacsim.sensors.physics"):
    omni.kit.app.get_app().get_extension_manager().set_extension_enabled_immediate(ext, True)
import omni.physics.tensors as pt  # noqa: E402
import omni.usd  # noqa: E402
from isaacsim.core.api import SimulationContext  # noqa: E402

import control_sim  # noqa: E402

DT = 1.0 / 120.0


class Scene:
    def __init__(self, path, hidden):
        self.ctx = omni.usd.get_context()
        self.ctx.open_stage(path)
        for _ in range(3):
            app.update()
        self.stage = self.ctx.get_stage()
        control_sim.prepare(self.stage, hidden)  # (contact reports on the robots' bodies, the day's pan and vehicle, before the start)
        self.sim = SimulationContext(physics_dt=DT, rendering_dt=DT, stage_units_in_meters=1.0)
        self.sim.reset()
        self.view = pt.create_simulation_view("numpy", self.ctx.get_stage_id())

    def step(self):
        self.sim.step(render=False)

    def flag(self, path, name):
        a = self.stage.GetPrimAtPath(path).GetAttribute(name)
        v = a.Get() if a and a.IsValid() else None
        return None if v is None else bool(v)


def main():
    catalog = json.load(open(args.catalog))
    hidden = control_sim.draw_hidden(args.seed)
    scene = Scene(args.scene, hidden)
    sim = control_sim.ControlSim(scene, catalog, {"time_limit_s": args.time_limit} if args.time_limit else None, seed=args.seed,
                                 log=lambda *a: print("[practice]", *a, flush=True), hidden=hidden)
    print("[practice] the day's bay:", json.dumps(sim.hidden_record()), flush=True)
    print("[practice] mission:", json.dumps(sim.mission()), flush=True)
    solver = control_sim.LocalSolver(args.submission)
    try:
        m = sim.run(lambda t, obs: solver.act(t, obs))
    finally:
        solver.close()
    m["hidden"] = sim.hidden_record()
    print("[practice] result:", json.dumps(m, indent=1), flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
