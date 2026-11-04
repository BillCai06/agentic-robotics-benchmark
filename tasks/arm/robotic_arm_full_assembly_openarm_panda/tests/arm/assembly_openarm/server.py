"""The OpenArm pair's assembly cell behind the camera-only API (`OpenArmServer`).

The physics is `cell.OpenArmCell`: the pair's base (the gallows with both J1 motors) on the bench, the 18 loose units of
the two arms resting on the bench, two Pandas (the left one builds the left arm, the right one the right arm, either may
help the other), the coupling state machine (`coupling.py`). The arm's J1-J3 motors hold the angle they were fastened
at while the arm is built (a PD hold, the motor's torque limit); J4-J7 are unpowered and back-drive against their
friction. The observation side: a top RGB-D camera over the bench and one RGB-D camera on each Panda's hand, rendered
by MuJoCo's Filament in a baked factory hall (`agent_render.py`), uncalibrated (mounts, intrinsics and lens distortion
drawn per episode), the noisy flange F/T sensors, the Pandas' joint states; no part poses, no mate events.

Per episode the bench layout is drawn from the episode seed (`layout.jitter_layout`). With the oracle token
(privileged mode) and `oracle_nominal_layout` the layout is the bench plan itself, and `reference_assemble()` runs the
privileged program (`reference.Assembler`) inside the cell, through the same control ticks and safety watch.

Score of an episode (`finalize`): the fastened mates' weight / the total (a flange 1.5, the J2 motor 1.0, a cable
holder 0.5, per arm; 22.0 for the pair), a flange counted when it is bolted in its design rotation.
"""
import base64
import json
import math
import time

import numpy as np

__all__ = ["OpenArmServer", "CELL_VERSION", "DEFAULT_CONFIG", "PUBLIC_METHODS", "PRIVILEGED_METHODS", "ROBOTS"]

CELL_VERSION = "openarm-pair-assembly-cell/1.0.0"
PUBLIC_METHODS = ("describe", "get_state", "set_control", "command", "step", "get_camera", "finish")
PRIVILEGED_METHODS = ("get_state_privileged", "describe_privileged", "reference_assemble")
ROBOTS = {"left": "pl", "right": "pr"}           # the API's robot names -> the model's prefixes
CAMERAS = ("top", "wrist_left", "wrist_right")
PANDAS = [{"name": "pl", "pos": [-0.275, -0.476, 0.0], "yaw_deg": 60.0, "arm": "left"},
          {"name": "pr", "pos": [-0.275, 0.476, 0.0], "yaw_deg": -60.0, "arm": "right"}]
BENCH_XY = (1.6, 1.4)
# the arms hanging over the bench and the gallows' foot; each Panda's base
KEEPOUTS = [(-0.25, 0.32, -0.30, 0.30), (-0.405, -0.145, -0.606, -0.346), (-0.405, -0.145, 0.346, 0.606)]

DEFAULT_CONFIG = {
    "sim_cap_s": 1800.0,
    "wall_cap_s": 3000.0,
    "silence_s": 120.0,
    "max_ticks_per_step": 500,
    "camera_budget_frames": 900,
    "bench_jitter": {"xy_m": 0.03, "yaw_deg": 30.0, "gap_m": 0.006},
    "fixture_jitter_m": 0.01,
    "oracle_nominal_layout": True,
    "privileged_wall_cap_s": 7200.0,
    "agent_renderer": "filament",
    "safety": {"panda_force_N": 30.0, "part_speed_m_s": 3.0},
    "damage": {"part_force_N": 150.0, "ticks": 20},
    "held": {"force_N": 1.0, "ticks": 5},
    "drop": {"speed_m_s": 0.25, "watch_ticks": 50},
    "lost_z_m": -0.05,
    # J1-J3 of each arm hold their fastened angle while the arm is built: kp N m/rad, kd N m s/rad, max N m
    "power_hold": {"j1": [40.0, 1.5, 10.0], "j2": [40.0, 1.5, 10.0], "j3": [25.0, 1.0, 5.0]},
    "info_layout": {},
    "photo": {},
}


def _deep_update(base, upd):
    for k, v in (upd or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = v
    return base


def _mat_quat(R):
    import mujoco
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, float).reshape(9))
    return q


class OpenArmServer(object):
    """`kit_dir` = the pair's kit (tests/gt), `panda_dir` = the given Panda model, `bench_plan` = the hidden bench plan
    ({"rest", "grasps", "risers"}), `config` = grader_config.json."""

    def __init__(self, kit_dir, panda_dir, bench_plan, config=None, privileged=False, render=True, log=None):
        import mujoco
        from .cell import load_kit
        from ..assembly_info.cell_info import DEFAULT_INFO_CONFIG
        self.mujoco = mujoco
        self.kit_dir, self.panda_dir = kit_dir, panda_dir
        self.kit = load_kit(kit_dir)
        cfg = json.loads(json.dumps(DEFAULT_CONFIG))
        cfg["photo"] = json.loads(json.dumps(DEFAULT_INFO_CONFIG["photo"]))
        self.cfg = _deep_update(cfg, config or {})
        self.bench_plan = bench_plan
        self.privileged = bool(privileged)
        self.render_enabled = bool(render)
        self.log = log or (lambda s: None)
        self.fa = None
        self.session = None
        self.done = True
        self._outlines = None

    # ------------------------------------------------------------------ episode setup
    def reset(self, seed):
        from ..assembly_info import layout_info as LI
        from .cell import OpenArmCell
        from .layout import jitter_layout, unit_outlines
        from . import reference as ref
        self._close_session()
        seed = int(seed)
        self.ep_seed = seed
        self.episode = LI.draw_episode(seed, self.cfg.get("info_layout"), strict=False)
        c = LI.merge_config(self.cfg.get("info_layout"))
        # each Panda's wrist camera is its own unit: its own mount error
        self.wrist_draw = {"left": self.episode["cameras"].get("wrist"),
                           "right": LI.draw_cameras((seed ^ 0x2B7E151628AED2A6) & ((1 << 63) - 1), c, strict=False).get("wrist")}
        if self.render_enabled and (self.episode["cameras"].get("top") is None or None in self.wrist_draw.values()):
            raise LI.MissingMount("the cell's config lacks a top or wrist camera mount")
        nominal = bool(self.privileged and self.cfg.get("oracle_nominal_layout", True))
        if self._outlines is None:
            self._outlines = unit_outlines(self.kit)
        bj = self.cfg["bench_jitter"]
        amp = 0.0 if nominal else 1.0
        lay = jitter_layout(self.kit, self.bench_plan, seed, xy_m=amp * float(bj["xy_m"]), yaw_deg=amp * float(bj["yaw_deg"]),
                            gap_m=float(bj["gap_m"]), bench_xy=BENCH_XY, keepouts=KEEPOUTS, outlines=self._outlines)
        rng = np.random.default_rng([seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF, 0xF1C7])
        fj = 0.0 if nominal else float(self.cfg.get("fixture_jitter_m", 0.01))
        fixture = {"xy": [float(rng.uniform(-fj, fj)), float(rng.uniform(-fj, fj))], "yaw_deg": 180.0}
        t0 = time.time()
        cell_cfg = {"pandas": PANDAS, "bench": {"size_xy": list(BENCH_XY)}}
        self.fa = fa = OpenArmCell(self.kit_dir, self.panda_dir, {"fixture": fixture, "parts": lay["parts"], "risers": lay["risers"]},
                                   settle_s=1.5, config=cell_cfg, panda_q=ref.PARK, log=self.log)
        fa.recording = False
        self.log("[cell] OpenArm pair cell for seed %d built and settled in %.0f s (layout: %d tries)" % (seed & 0xFFFFFFFF, time.time() - t0, lay["tries"]))
        self.model, self.data, self.sm = fa.m, fa.d, fa.sm
        self.layout = {"fixture": fixture, "parts": lay["parts"], "risers": lay["risers"], "draw": lay["draw"], "nominal": nominal}
        self.tick_s = fa.tick_s
        self._init_state(seed)
        raw = fa.tick_one
        self._raw_tick = raw

        def routed():
            self.physics_tick()
        fa.tick_one = routed          # every control tick goes through the watch, whoever drives it
        return {"ok": True}

    def _init_state(self, seed):
        m, fa = self.model, self.fa
        self.done = False
        self.done_reason = None
        self.tick_count = 0
        self.events_log = []
        self.violations = []
        self.illegal_calls = 0
        self.calls = {}
        self.step_calls = 0
        self.wall_used = 0.0
        self.frames_used = 0
        self.frames_by_camera = {n: 0 for n in CAMERAS}
        self.render_wall_s = 0.0
        self.renderer = None
        self.ctrls = {api: fa.ctrls[pn] for api, pn in ROBOTS.items()}
        pc = {p["name"]: p for p in fa.pandas}
        self.sensors = {api: (m.sensor(pc[pn]["force"]).id, m.sensor(pc[pn]["torque"]).id) for api, pn in ROBOTS.items()}
        self.wrench_bias = {}
        wn = self.episode["config"]["wrench_noise"]
        for i, api in enumerate(ROBOTS):
            r = np.random.default_rng([seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF, 0xB1A5 + i])
            self.wrench_bias[api] = np.array(list(r.uniform(-wn["bias_f_N"], wn["bias_f_N"], 3)) + list(r.uniform(-wn["bias_t_Nm"], wn["bias_t_Nm"], 3)))
        self._wrench_rng = np.random.default_rng([seed & 0xFFFFFFFF, (seed >> 32) & 0xFFFFFFFF, 0x3E17C4])
        # contact classes: 1 a Panda link (not a pad), 2 a pad, 4 a loose unit (or a riser), 5 the static environment
        self.geom_class = np.zeros(m.ngeom, dtype=np.int8)
        self.geom_owner = {}               # geom -> the Panda (API name) / the unit
        pg = fa.meta["pad_geoms"]
        pads = set(n for v in pg.values() for n in v) if isinstance(pg, dict) else set(pg)
        pn_of = {pn: api for api, pn in ROBOTS.items()}
        loose_of = dict(fa.unit_of_body)
        risers = set(b for b in range(m.nbody) if m.body(b).name.startswith("riser_"))
        for g in range(m.ngeom):
            if not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue
            b = int(m.geom_bodyid[g])
            nm = m.body(b).name
            pfx = nm.split("/")[0] if "/" in nm else None
            if pfx in pn_of:
                self.geom_class[g] = 2 if m.geom(g).name in pads else 1
                self.geom_owner[g] = pn_of[pfx]
            elif b in loose_of and loose_of[b] in fa.loose:
                self.geom_class[g] = 4
                self.geom_owner[g] = loose_of[b]
            elif b in risers:
                self.geom_class[g] = 4
                self.geom_owner[g] = nm
            else:
                self.geom_class[g] = 5
                if loose_of.get(b) == "base":
                    self.geom_owner[g] = "base"     # the gallows and the J1 motors (the bench: no owner)
        self.part_vadr = {u: m.jnt_dofadr[m.joint("free_" + u).id] for u in fa.loose}
        self.part_qadr = {u: m.jnt_qposadr[m.joint("free_" + u).id] for u in fa.loose}
        self._f6 = np.zeros(6)
        self.held = {api: None for api in ROBOTS}
        self._held_cand = {api: (None, 0) for api in ROBOTS}
        self._drop_watch = {}
        self._lost, self._dropped = set(), set()
        self._part_hist = []
        self.dmg = {"max_link_force_N": 0.0, "max_pad_force_N": 0.0, "max_part_force_N": 0.0, "max_part_force_sustained_N": 0.0}
        # J1-J3's PD hold: the hinge of each, its angle once fastened
        self.hold = {}
        self.powered = True
        self._hold_keys = {jn: jn.split("_")[-1] for jn in self.sm.joint_info if jn.split("_")[-1] in self.cfg["power_hold"]}

    # ------------------------------------------------------------------ physics
    def _power_hold(self):
        m, d = self.model, self.data
        for jn, key in self._hold_keys.items():
            if jn not in self.sm.limits_set:
                continue
            j = self.sm.joint_info[jn]["hinge"]
            dof = m.jnt_dofadr[j]
            if jn not in self.hold:
                self.hold[jn] = float(d.qpos[m.jnt_qposadr[j]])
            if not self.powered:
                d.qfrc_applied[dof] = 0.0
                continue
            kp, kd, tmax = self.cfg["power_hold"][key]
            tau = kp * (self.hold[jn] - float(d.qpos[m.jnt_qposadr[j]])) - kd * float(d.qvel[dof])
            d.qfrc_applied[dof] = float(np.clip(tau, -tmax, tmax))

    def physics_tick(self, track=True):
        if self.done:
            return []
        n0 = len(self.sm.events)
        self._power_hold()
        self._raw_tick()
        self.tick_count += 1
        events = ["%s" % e for _t, e in self.sm.events[n0:]]
        events += self._watch(self.data)
        for e in events:
            self.events_log.append((float(self.data.time), e))
        if self.data.time >= float(self.cfg["sim_cap_s"]) - 1e-9:
            self._end("sim_cap")
        return events

    def _watch(self, d):
        """safety: a Panda link other than its pads pressing on the bench, the fixture, a unit or the other Panda above
        `panda_force_N`; a unit faster than `part_speed_m_s`; a unit pressed on the environment or another unit above
        `part_force_N` for `ticks` consecutive ticks. Bookkeeping: held (both pads of a Panda on a unit), dropped, lost."""
        m = self.model
        events = []
        thr = float(self.cfg["safety"]["panda_force_N"])
        pads = {api: {} for api in ROBOTS}
        part_sum = {}
        for i in range(d.ncon):
            con = d.contact[i]
            g1, g2 = con.geom1, con.geom2
            c1, c2 = int(self.geom_class[g1]), int(self.geom_class[g2])
            if c1 == 0 and c2 == 0:
                continue
            self.mujoco.mj_contactForce(m, d, i, self._f6)
            fn = abs(float(self._f6[0]))
            for ca, ga, cb, gb in ((c1, g1, c2, g2), (c2, g2, c1, g1)):
                if ca == 1 and (cb in (4, 5) or (cb in (1, 2) and self.geom_owner.get(gb) != self.geom_owner.get(ga))):
                    self.dmg["max_link_force_N"] = max(self.dmg["max_link_force_N"], fn)
                    if fn > thr:
                        what = "panda_contact:%s:%s:%.0fN" % (m.body(m.geom_bodyid[ga]).name, m.body(m.geom_bodyid[gb]).name, fn)
                        last = self.violations[-1] if self.violations else None
                        if last is None or last["t"] < d.time - 0.5 or last["what"].split(":")[1:3] != what.split(":")[1:3]:
                            self.violations.append({"t": float(d.time), "what": what, "force_N": fn})
                            events.append("safety:" + what)
                if ca == 2 and cb == 4:
                    self.dmg["max_pad_force_N"] = max(self.dmg["max_pad_force_N"], fn)
                    api, u = self.geom_owner[ga], self.geom_owner[gb]
                    side = "L" if "pad_left" in m.geom(ga).name else "R"
                    pads[api].setdefault(u, {"L": 0.0, "R": 0.0})[side] += fn
            if (c1 == 4 and c2 in (4, 5)) or (c2 == 4 and c1 in (4, 5)):
                gp, go = (g1, g2) if c1 == 4 else (g2, g1)
                if self._bolted(self.geom_owner[gp], self.geom_owner.get(go)):
                    continue                 # (two units joined at their interface: not a unit pressed on another)
                key = (self.geom_owner[gp], int(m.geom_bodyid[go]))
                part_sum[key] = part_sum.get(key, 0.0) + fn
        pm = max(part_sum.values()) if part_sum else 0.0
        if pm > float(self.cfg["damage"]["part_force_N"]):
            k_ = max(part_sum, key=part_sum.get)
            self._part_pair = "%s:%s" % (k_[0], m.body(k_[1]).name)
        self.dmg["max_part_force_N"] = max(self.dmg["max_part_force_N"], pm)
        ticks = int(self.cfg["damage"]["ticks"])
        self._part_hist.append(pm)
        if len(self._part_hist) > ticks:
            del self._part_hist[0]
        if len(self._part_hist) == ticks:
            sus = min(self._part_hist)
            self.dmg["max_part_force_sustained_N"] = max(self.dmg["max_part_force_sustained_N"], sus)
            if sus > float(self.cfg["damage"]["part_force_N"]) and not any(v["what"].startswith("part_force") and v["t"] > d.time - 1.0 for v in self.violations):
                self.violations.append({"t": float(d.time), "what": "part_force:%s:%.0fN" % (getattr(self, "_part_pair", "?"), sus), "force_N": sus})
                events.append("safety:part_force:%.0fN" % sus)
        # held / released per Panda
        hf, ht = float(self.cfg["held"]["force_N"]), int(self.cfg["held"]["ticks"])
        for api in ROBOTS:
            cand = None
            best = 0.0
            for u, f in pads[api].items():
                s = min(f["L"], f["R"])
                if s >= hf and s > best:
                    cand, best = u, s
            pc, n = self._held_cand[api]
            n = n + 1 if (cand is not None and cand == pc) else (1 if cand is not None else 0)
            self._held_cand[api] = (cand, n)
            new = cand if n >= ht else None
            if new != self.held[api]:
                if self.held[api] is not None and self.held[api] in self.part_vadr:
                    self._drop_watch[self.held[api]] = {"t0": self.tick_count, "vmax": 0.0}
                self.held[api] = new
        thr_v = float(self.cfg["safety"]["part_speed_m_s"])
        for u, va in self.part_vadr.items():
            v = float(np.linalg.norm(d.qvel[va:va + 3]))
            if v > thr_v and not any(x["what"].startswith("part_speed:" + u) for x in self.violations[-3:]):
                self.violations.append({"t": float(d.time), "what": "part_speed:%s:%.1fm/s" % (u, v), "speed": v})
                events.append("safety:part_speed:%s" % u)
            if float(d.qpos[self.part_qadr[u] + 2]) < float(self.cfg["lost_z_m"]) and u not in self._lost:
                self._lost.add(u)
            w = self._drop_watch.get(u)
            if w is not None:
                w["vmax"] = max(w["vmax"], v)
                if self.tick_count - w["t0"] >= int(self.cfg["drop"]["watch_ticks"]):
                    del self._drop_watch[u]
                    if w["vmax"] > float(self.cfg["drop"]["speed_m_s"]) and not any(self.sm.rec[mid]["fastened"] for mid in self.sm.rec
                                                                                 if self.fa.mates[mid]["unit"] == u):
                        self._dropped.add(u)
        return events

    def _bolted(self, a, b):
        """True when units a and b (or the base: the gallows with the J1 motors) are joined through mates fastened or on their
        feature"""
        if a is None or b is None:
            return False
        n = self.tick_count
        if getattr(self, "_groups_n", -1) != n:
            parent = {}

            def find(x):
                while parent.get(x, x) != x:
                    x = parent[x]
                return x
            for mid, r in self.sm.rec.items():
                # fastened, or on its feature (held by the cell: pressed into its seat it fights its carrier's pieces --
                # 6280 N between the forearm and the upper arm while the forearm went from ENGAGED to LOCKED)
                if r["fastened"] or self.sm.state.get(mid) in ("ENGAGED", "HOLDING"):
                    mm = self.fa.mates[mid]
                    ua = self.fa.unit_of_body.get(self.model.body(mm["carrier"]).id, "base")
                    ra, rb = find(mm["unit"]), find(ua)
                    if ra != rb:
                        parent[ra] = rb
            self._groups = {x: find(x) for x in list(self.fa.loose) + ["base"]}
            self._groups_n = n
        return a in self._groups and b in self._groups and self._groups[a] == self._groups[b]

    # ------------------------------------------------------------------ the API
    def _robot(self, robot):
        r = str(robot)
        if r not in ROBOTS:
            raise ValueError("robot must be one of %s" % ", ".join(ROBOTS))
        return r

    def describe(self):
        m = self.model
        cams = {}
        for n in CAMERAS:
            c = self.episode["config"]["cameras"]["top" if n == "top" else "wrist"]
            cams[n] = {"resolution": list(c["size"]), "rgb": True, "depth": bool(c.get("depth", True)), "nominal_hfov_deg": round(float(c["hfov_deg"])),
                       "mount": ("overhead, high above the middle of the bench, looking straight down at the whole bench" if n == "top" else
                                 "on the %s Panda's hand, a few centimetres off the flange axis, tilted towards the fingertips" % n.split("_")[1])}
        robots = {}
        for api, pn in ROBOTS.items():
            ct = self.ctrls[api]
            p = [x for x in PANDAS if x["name"] == pn][0]
            robots[api] = {"model": "7-DoF Panda (given model: panda/)", "base_pos": list(p["pos"]), "base_yaw_deg": float(p["yaw_deg"]),
                           "builds": "the %s arm" % api, "joints": ["%s/joint%d" % (pn, i + 1) for i in range(7)],
                           "joint_range": [[float(m.jnt_range[j][0]), float(m.jnt_range[j][1])] for j in ct.joints],
                           "home_q": [float(v) for v in ct.q_cmd], "tcp_offset_in_hand_m": [0.0, 0.0, 0.1034],
                           "gripper": {"max_width_m": ct.max_width, "rate_m_s": 0.05, "pad_size_m": [0.017, 0.0165],
                                       "force_model": "2500 N/m x (object width - target width) / 4 per finger, capped at 70 N per finger"},
                           "gravity_compensated": True,
                           "ft_sensor": "6-axis at the hand flange (hand frame); noise ~0.3 N / 0.02 N m rms, an unknown per-episode bias up to 1 N / 0.05 N m"}
        return {
            "cell_version": CELL_VERSION, "units": {"length": "m", "angle": "rad", "quaternion": "wxyz"},
            "task": "assemble both arms of the pair from the base: J1's two motors are on the gallows; mount every unit as the picture manual shows",
            "robots": robots,
            "control": {"tick_s": self.tick_s, "joint_rate_limit_rad_s": 1.5, "cartesian_speed_limit_m_s": 0.5, "cartesian_angular_limit_rad_s": 1.5,
                        "stiffness_scale_range": [0.2, 1.0], "modes": ["joint", "cartesian"], "per_robot": True},
            "cameras": cams, "camera_budget_frames": int(self.cfg["camera_budget_frames"]),
            "bench": {"size_xy_m": list(BENCH_XY), "top_z_m": 0.0, "fixture": "the gallows at about (0, 0) m, turned 180 deg (+-10 mm)"},
            "budgets": {"sim_time_cap_s": float(self.cfg["sim_cap_s"]), "wall_clock_cap_s": float(self.cfg["wall_cap_s"]),
                        "silence_abort_s": float(self.cfg["silence_s"]), "max_ticks_per_step": int(self.cfg["max_ticks_per_step"]),
                        "camera_frames": int(self.cfg["camera_budget_frames"])},
            "safety": {"panda_force_N": float(self.cfg["safety"]["panda_force_N"]), "part_speed_m_s": float(self.cfg["safety"]["part_speed_m_s"]),
                       "part_force_N": float(self.cfg["damage"]["part_force_N"]), "part_force_window_s": self.tick_s * int(self.cfg["damage"]["ticks"])},
        }

    def get_state(self):
        d = self.data
        out = {"sim_time": float(d.time), "done": bool(self.done), "robots": {}}
        wn = self.episode["config"]["wrench_noise"]
        sig = np.array([wn["sigma_f_N"]] * 3 + [wn["sigma_t_Nm"]] * 3)
        for api, ct in self.ctrls.items():
            p, R = ct.tcp_actual()
            fs, ts = self.sensors[api]
            adr = self.model.sensor_adr
            w = np.array(list(d.sensordata[adr[fs]:adr[fs] + 3]) + list(d.sensordata[adr[ts]:adr[ts] + 3]))
            w = w + self.wrench_bias[api] + self._wrench_rng.normal(0.0, 1.0, 6) * sig
            out["robots"][api] = {"q": [float(v) for v in ct.q()], "dq": [float(v) for v in ct.dq()], "cmd_q": [float(v) for v in ct.q_cmd],
                                  "gripper_width": float(ct.width()), "tcp_pos": [float(v) for v in p], "tcp_quat": [float(v) for v in _mat_quat(R)],
                                  "wrench": [float(v) for v in w], "mode": ct.mode, "stiffness_scale": float(ct.stiffness_scale)}
        return out

    def set_control(self, robot="left", mode="joint", stiffness_scale=1.0):
        ct = self.ctrls[self._robot(robot)]
        ct.set_mode(str(mode))
        ct.set_stiffness(float(stiffness_scale))
        return {"ok": True, "robot": str(robot), "mode": ct.mode, "stiffness_scale": float(ct.stiffness_scale)}

    def command(self, robot="left", q=None, tcp_pos=None, tcp_quat=None, width=None):
        if self.done:
            return {"ok": False, "error": "episode_over"}
        ct = self.ctrls[self._robot(robot)]
        if q is not None:
            if ct.mode != "joint":
                return {"ok": False, "error": "q target needs joint mode"}
            ct.command_joint(q)
        if tcp_pos is not None or tcp_quat is not None:
            if ct.mode != "cartesian":
                return {"ok": False, "error": "tcp target needs cartesian mode"}
            if tcp_pos is None or tcp_quat is None:
                return {"ok": False, "error": "tcp_pos and tcp_quat go together"}
            ct.command_cartesian(tcp_pos, tcp_quat)
        if width is not None:
            ct.command_width(width)
        return {"ok": True}

    def step(self, n_ticks=1):
        n = int(n_ticks)
        if n < 1:
            return {"ok": False, "error": "n_ticks must be >= 1"}
        n = min(n, int(self.cfg["max_ticks_per_step"]))
        self.step_calls += 1
        ev = []
        for _ in range(n):
            if self.done:
                break
            ev += [e for e in self.physics_tick() if e.startswith("safety:")]
        if self.done:
            ev.append("episode_over")
        return {"sim_time": float(self.data.time), "done": bool(self.done), "events": ev}

    def finish(self):
        self._end("finish")
        return {"ok": True, "sim_time": float(self.data.time)}

    def _end(self, reason):
        if not self.done:
            self.done = True
            self.done_reason = reason

    # ------------------------------------------------------------------ cameras
    def _close_session(self):
        if self.session is not None:
            try:
                self.session.close()
            except Exception:  # noqa: BLE001
                pass
        self.session = None

    def _session(self):
        if self.session is None:
            from . import agent_render
            t = time.time()
            want = str(self.cfg.get("agent_renderer", "filament"))
            ok, why = agent_render.available() if want == "filament" else (False, "configured")
            if want == "filament" and not ok:
                raise RuntimeError("agent renderer filament unavailable: %s" % why)
            self.session = agent_render.open_session(self, filament=ok)
            self.renderer = "filament" if ok else "mujoco"
            self.render_wall_s += time.time() - t
            self.log("[cell] %s render session opened in %.1f s" % (self.renderer, time.time() - t))
        return self.session

    def camera_spec(self, name):
        from render.spec import CameraSpec
        if name == "top":
            c = self.episode["cameras"]["top"]
            W = np.asarray(c["world_T_cam"], float)
        else:
            api = name.split("_")[1]
            c = self.wrist_draw[api]
            b = self.model.body(ROBOTS[api] + "/hand").id
            T = np.eye(4)
            T[:3, :3] = self.data.xmat[b].reshape(3, 3)
            T[:3, 3] = self.data.xpos[b]
            W = T @ np.asarray(c["hand_T_cam"], float)
        intr = c["intrinsics"]
        return CameraSpec(name=name, width=intr["width"], height=intr["height"], fx=intr["fx"], fy=intr["fy"], cx=intr["cx"], cy=intr["cy"],
                          world_T_cam=W.tolist(), k1=intr["k1"], k2=intr["k2"], p1=intr["p1"], p2=intr["p2"])

    def photo_spec(self, name):
        """the photo pipeline (noise, blur, vignetting, colour jitter, the depth sensor); the colour jitter drawn once per
        camera unit and episode, the sensor noise per frame"""
        from render.spec import BlurSpec, ColourJitterSpec, DepthSensorSpec, NoiseSpec, PhotoSpec, derive_seed
        ph = self.cfg["photo"]
        dk = dict(ph["depth_common"])
        dk.update(ph["depth_near"] if name.startswith("wrist") else ph["depth_fixed"])
        sensor = DepthSensorSpec(unit_seed=derive_seed(self.ep_seed, "depth_unit", name), **dk)
        cj = dict(ph["jitter"])
        return PhotoSpec(noise=NoiseSpec(**ph["noise"]), blur=BlurSpec(defocus_px=float(ph["defocus_px"])), vignetting=float(ph["vignetting"]),
                         colour_jitter=ColourJitterSpec(seed=derive_seed(self.ep_seed, "colour_unit", name), **cj), image_format="png",
                         depth_sensor=sensor, png_compress_level=int(ph.get("png_compress_level", 1)))

    def get_camera(self, name="top", depth=True):
        from render.spec import derive_seed
        name = str(name)
        if name not in CAMERAS:
            return {"error": "unknown camera (cameras: %s)" % ", ".join(CAMERAS)}
        if not self.render_enabled:
            return {"error": "camera_unavailable"}
        if self.frames_used >= int(self.cfg["camera_budget_frames"]):
            return {"error": "camera_budget"}
        k = self.frames_used
        self.frames_used += 1
        self.frames_by_camera[name] += 1
        t = time.time()
        s = self._session()
        s.sync_from(self.model, self.data)
        outputs = ("rgb", "sensor_depth") if depth else ("rgb",)
        try:
            r = s.render([self.camera_spec(name)], photo=self.photo_spec(name), seed=derive_seed(self.ep_seed, "frame", k), outputs=outputs)
        except Exception as e:  # noqa: BLE001 - a renderer failure never takes the cell down
            self.log("[cell] %s frame %d not rendered: %r" % (name, k, e))
            self.render_wall_s += time.time() - t
            return {"error": "camera_unavailable"}
        v = r.agent_view()
        out = {"ok": True, "camera": name, "frame_index": int(k), "width": int(v["rgb"][0].shape[1]), "height": int(v["rgb"][0].shape[0]),
               "format": "png", "rgb_png_b64": base64.b64encode(v["encoded"][0]).decode("ascii")}
        if depth:
            out["sensor_depth_u16_mm_b64"] = base64.b64encode(v["sensor_depth_encoded"][0]).decode("ascii")
            out["depth_unit_m"] = 0.001
            out["depth_invalid_value"] = 0
        self.render_wall_s += time.time() - t
        return out

    # ------------------------------------------------------------------ privileged
    def describe_privileged(self):
        out = {"layout": self.layout, "mates": list(self.kit["mates"]), "states": dict(self.sm.state), "score": self.sm.score()}
        return json.loads(json.dumps(out, default=lambda x: x.tolist() if hasattr(x, "tolist") else str(x)))

    def get_state_privileged(self):
        d = self.data
        parts = {u: {"pos": [float(v) for v in d.xpos[self.model.body(u).id]], "quat": [float(v) for v in d.xquat[self.model.body(u).id]]}
                 for u in self.fa.loose}
        return {"sim_time": float(d.time), "parts": parts, "states": dict(self.sm.state), "score": self.sm.score()}

    def reference_assemble(self):
        """the privileged program on this episode (oracle token only): the cell's own J1-J3 hold stands in for its hook"""
        from . import reference as ref
        A = ref.Assembler(self.fa, log=self.log)
        if A.power_hold in self.fa.hooks:
            self.fa.hooks.remove(A.power_hold)
        seq = ref.sequence(self.kit, self.fa.pandas)
        report = A.run(seq, plan=self.bench_plan)
        return {"ok": True, "score": float(self.sm.score()), "placed_by_panda": int(report.get("placed_by_panda", 0)),
                "unplanned_contacts": int(report.get("unplanned_contacts_total", 0))}

    # ------------------------------------------------------------------ dispatch
    def handle(self, method, params):
        self.calls[method] = self.calls.get(method, 0) + 1
        if method in PRIVILEGED_METHODS:
            if not self.privileged:
                self.illegal_calls += 1
                return {"error": "forbidden"}
            return getattr(self, method)()
        if method not in PUBLIC_METHODS:
            self.illegal_calls += 1
            return {"error": "forbidden"}
        try:
            return getattr(self, method)(**(params or {}))
        except (TypeError, ValueError, KeyError, IndexError) as e:
            return {"error": "bad_params: %s" % str(e)[:200]}

    # ------------------------------------------------------------------ end of episode
    def stand_check(self, seconds=3.0):
        """the motors off, the arms left to stand: the largest change of any fastened joint (deg)"""
        m, d = self.model, self.data
        qa = {jn: m.jnt_qposadr[self.sm.joint_info[jn]["hinge"]] for jn in self.sm.limits_set}
        q0 = {jn: float(d.qpos[a]) for jn, a in qa.items()}
        self.powered = False
        done, self.done = self.done, False
        for _ in range(int(round(seconds / self.tick_s))):
            self._power_hold()
            self._raw_tick()
        self.done = done
        return {jn: round(math.degrees(float(d.qpos[qa[jn]]) - q0[jn]), 3) for jn in qa}

    def finalize(self, functional_rng=None, run_functional_check=False):
        self._close_session()
        if not self.done:
            self._end("solve_returned")
        rec = self.sm.record()
        weights = {r["id"]: float(r["weight"]) for r in rec}
        W = sum(weights.values())
        achieved = sum(weights[r["id"]] for r in rec if r["achieved"])
        stand = self.stand_check()
        return {
            "cell_version": CELL_VERSION, "seed": int(self.ep_seed), "done_reason": self.done_reason,
            "sim_time_s": float(self.data.time), "ticks": int(self.tick_count),
            "mates": rec, "states": dict(self.sm.state), "weights": weights, "stage_score": achieved / W if W else 0.0,
            "n_achieved": int(sum(1 for r in rec if r["achieved"])), "n_fastened": int(sum(1 for r in rec if r["fastened"])),
            "n_mates": len(rec), "all_mates": bool(all(r["achieved"] for r in rec)),
            "violations": list(self.violations), "safety_ok": len(self.violations) == 0,
            "illegal_call_count": int(self.illegal_calls), "calls": dict(self.calls), "step_calls": int(self.step_calls),
            "lost_parts": sorted(self._lost), "dropped_parts": sorted(self._dropped),
            "events": [[t, e] for t, e in self.events_log][-400:], "coupling_events": [list(e) for e in self.sm.events],
            "layout": self.layout, "damage": dict(self.dmg), "stand_after_power_off_deg": stand,
            "camera_frames": dict(self.frames_by_camera), "camera_frames_total": int(self.frames_used),
            "render_wall_s": float(self.render_wall_s), "renderer": self.renderer, "privileged_mode": bool(self.privileged),
        }
