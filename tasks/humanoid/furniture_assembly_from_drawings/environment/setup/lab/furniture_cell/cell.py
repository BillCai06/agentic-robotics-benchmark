"""One episode of the furniture cell: model, controllers, physical rules, cameras, budgets and the episode's score.

The public surface (what a program can learn) is `describe`, `get_state`, `command`, `step`, `get_camera`, `finish`;
everything else here is the grader's (the true part poses, the joints' states, the score).
"""

from __future__ import annotations

import math
import time

import mujoco
import numpy as np

from . import furniture as F
from . import layout as LAYOUT
from . import room as RM
from .control import TICK, V_MAX, RobotController
from .physics import GRIP_MAX_FORCE, GRIP_MAX_TORQUE, Bodies, Coupling, Grips, Safety
from .render import CameraRig, camera_pose_world, encode_frame, intrinsics
from .robot import (ARM_HOME, ARM_JOINTS, D435_PITCH, D435_POS, GRASP_SITE_POS, HAND_JOINTS, HEAD_CAM, PELVIS_HEIGHT,
                    SIDES, STAND_RANGE, WAIST_JOINTS, robot_index)
from .scene import ROBOTS, build
from .walk_core import LegWalker, support_from_geom

MM = 1e-3
SUBSTEPS = 5                               # 5 x 2 ms physics per 10 ms control tick
LOC_NOISE = (0.003, math.radians(0.2))     # localisation noise (m, rad), per reading
FT_NOISE = 0.4                             # N, wrist force noise per reading
PUBLIC_CAMERAS = ("ceiling_west", "ceiling_east", "a/head", "b/head")


def _cam_mjname(name: str) -> str:
    return name.replace("/head", "/head_cam") if name.endswith("/head") else name


def _drawn_supports(m: mujoco.MjModel) -> list:
    """What the drawn feet stand on rather than in: the rug as the cameras draw it (the decor's flat mesh over the
    surveyed rug; without the decor, the room's own rug box)."""
    G = mujoco.mjtObj.mjOBJ_GEOM
    (cx, cy), (sx, sy) = RM.RUG["center"], RM.RUG["size"]
    d = mujoco.MjData(m)
    mujoco.mj_kinematics(m, d)
    out = []
    for g in range(m.ngeom):
        if not (mujoco.mj_id2name(m, G, g) or "").startswith("decor_") or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        mid = int(m.geom_dataid[g])
        V = m.mesh_vert[m.mesh_vertadr[mid]:m.mesh_vertadr[mid] + m.mesh_vertnum[mid]].astype(np.float64)
        V = V @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g]
        c = V[:, :2].mean(axis=0)
        if V[:, 2].min() > -0.005 and V[:, 2].max() < 0.03 and abs(c[0] - cx) < sx / 2 and abs(c[1] - cy) < sy / 2:
            out.append(support_from_geom(m, g))
    if not out:
        g = mujoco.mj_name2id(m, G, "rug")
        if g >= 0:
            out.append(support_from_geom(m, g))
    return out


class Cell:
    def __init__(self, seed: int, *, nominal: bool = False, render: bool = True, config: dict | None = None):
        cfg = config or {}
        self.seed = int(seed)
        self.cfg = cfg
        self.layout = LAYOUT.draw(self.seed, nominal=nominal)
        self.spec = build(self.layout)
        self.m = self.spec.compile()
        self.d = mujoco.MjData(self.m)
        self.robots = {n: robot_index(self.m, n) for n in ROBOTS}
        self.ctrl = {n: RobotController(self.m, r) for n, r in self.robots.items()}
        self.bodies = Bodies(self.m, self.robots)
        self.grips = Grips(self.m, self.d, self.bodies)
        self.coupling = Coupling(self.m, self.d, self.bodies)
        self.safety = Safety(self.m, self.d, self.bodies)
        self.rng = np.random.default_rng((self.seed * 2654435761 + 97) & ((1 << 64) - 1))
        sizes = {n: RM.CEILING_CAMERAS[n]["size"] for n in RM.CEILING_CAMERAS}
        sizes.update({f"{n}/head_cam": HEAD_CAM["size"] for n in ROBOTS})
        self.cams = CameraRig(self.m, sizes, np.random.default_rng(self.rng.integers(1 << 62))) if render else None
        # the legs the cameras draw: stepping while a robot moves (walk_core; drawn only, the physics keeps its posed
        # legs). Stepped every control tick, so the drawn legs follow the motion whatever the cameras do.
        sup = _drawn_supports(self.m) if render else []
        self.walkers = {n: LegWalker(self.m, f"{n}/", supports=sup) for n in ROBOTS} if render else {}
        self._q_kin = None                         # the qpos the physics state's kinematics belong to
        self._rd = None                            # the state the cameras draw
        self.sim_cap = float(cfg.get("sim_cap_s", 1500.0))
        self.frame_budget = int(cfg.get("camera_budget_frames", 900))
        self.max_ticks = int(cfg.get("max_ticks_per_step", 500))
        self.frames = {}
        self.frames_total = 0
        self.render_wall = 0.0
        # the cell's own work, not charged to the program's time budget: the simulation inside step() and the frame
        # production inside get_camera() (rendering, encoding); both are bounded by the sim cap and the frame budget
        self.work_wall = 0.0
        self.ticks = 0
        self.done = False
        self.done_reason = None
        self.calls = {}
        self.illegal = 0
        self.notes = []
        self._reset()

    # ------------------------------------------------------------------------------------------------ setup
    def _reset(self):
        m, d = self.m, self.d
        mujoco.mj_resetData(m, d)
        for n, r in self.robots.items():
            x, y, yaw = self.layout["robots"][n]
            d.qpos[r.base_q[:3]] = (x, y, yaw)
            d.qpos[r.base_q[3]] = 0.0
            for s in SIDES:
                d.qpos[r.arm_q[s]] = ARM_HOME[s]
        for key, (pos, quat) in self.layout["parts"].items():
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"part_{key}_free")
            a = m.jnt_qposadr[jid]
            d.qpos[a:a + 3] = pos
            d.qpos[a + 3:a + 7] = quat
        mujoco.mj_forward(m, d)
        if self.walkers:
            self._q_kin = d.qpos.copy()
        for n, c in self.ctrl.items():
            c.reset_targets(d)
        # settle: let the parts come to rest on their supports (not part of the episode's time)
        for _ in range(int(float(self.cfg.get("settle_s", 0.5)) / TICK)):
            self._tick(record=False)
        self.ticks = 0
        self.t0 = float(d.time)
        self.safety.violations.clear()
        self.safety.count.clear()
        for k in self.safety.peaks:
            self.safety.peaks[k] = 0.0
        self.safety.peak_who.clear()
        for w in self.walkers.values():
            w.reset(self.sim_time, d.qpos[w.base_adr])

    @property
    def sim_time(self) -> float:
        return self.ticks * TICK

    def _tick(self, record: bool = True):
        m, d = self.m, self.d
        for c in self.ctrl.values():
            c.tick(d)
        for i in range(SUBSTEPS):
            if i == SUBSTEPS - 1 and self.walkers:
                self._q_kin[:] = d.qpos            # mj_step leaves d's kinematics at the qpos it started from
            mujoco.mj_step(m, d)
        t = self.sim_time
        self.grips.tick(t)
        held = {st["part"] for st in self.grips.state.values() if st["part"]}
        self.coupling.tick(t, held)
        if record:
            self.safety.tick(t)
            self.ticks += 1
            for w in self.walkers.values():
                w.step(self.sim_time, d.qpos[w.base_adr])

    # ------------------------------------------------------------------------------------------------ public API
    def describe(self) -> dict:
        r0 = self.robots["a"]
        joints = {"waist": [f"{w}_joint" for w in WAIST_JOINTS],
                  "left_arm": [f"left_{w}_joint" for w in ARM_JOINTS],
                  "right_arm": [f"right_{w}_joint" for w in ARM_JOINTS],
                  "left_hand": [f"left_hand_{w}_joint" for w in HAND_JOINTS],
                  "right_hand": [f"right_hand_{w}_joint" for w in HAND_JOINTS]}
        ranges = {}
        for group, names in joints.items():
            for jn in names:
                j = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_JOINT, "a/" + jn)
                ranges[jn] = [float(self.m.jnt_range[j, 0]), float(self.m.jnt_range[j, 1])]
        cams = {}
        for n, c in RM.CEILING_CAMERAS.items():
            cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, n)
            p, R = camera_pose_world(self.m, self.d, cid)
            cams[n] = {"intrinsics": intrinsics(c["size"][0], c["size"][1], c["fovy"]),
                       "pose_in_room": {"position": [float(v) for v in p], "rotation": R.tolist()},
                       "depth_min_m": 0.40}
        for n in ROBOTS:
            cams[f"{n}/head"] = {"intrinsics": intrinsics(HEAD_CAM["size"][0], HEAD_CAM["size"][1], HEAD_CAM["fovy"]),
                                 "mount": {"body": "torso_link", "position": list(D435_POS),
                                           "pitch_down_rad": D435_PITCH, "optical_axis": "body +x pitched down"},
                                 "depth_min_m": 0.28,
                                 "note": "every head frame carries the camera's pose in the room at capture time"}
        _ = r0
        return {
            "api": "furniture-cell/1.0",
            "robots": {n: {"model": "unitree_g1_29dof_dex3 (cell variant, ~/given/robot/g1_cell.xml)",
                           "start_hint": "near the south wall",
                           "legs_in_images": "the cameras draw the legs stepping while the base moves (an animation, "
                                             "drawn only: the legs' colliders stay as posed, the drawn legs collide "
                                             "with nothing)"} for n in ROBOTS},
            "joints": joints, "joint_ranges": ranges,
            "control": {"tick_s": TICK, "max_ticks_per_step": self.max_ticks,
                        "base_velocity_max": list(V_MAX), "stand_height_range": list(STAND_RANGE),
                        "stand_height_nominal": PELVIS_HEIGHT,
                        "arm_joint_rate_rad_s": 1.5, "waist_rate_rad_s": 0.8,
                        "grasp_site": {"body": "<side>_wrist_yaw_link", "position": {s: list(GRASP_SITE_POS[s]) for s in SIDES},
                                       "axes": "x = approach (along the fingers), y = pinch axis, z = x cross y"}},
            "grip": {"max_force_N": GRIP_MAX_FORCE, "max_torque_Nm": GRIP_MAX_TORQUE,
                     "rule": "closing a hand whose grasp site straddles a board's edge (thickness along the pinch "
                             "axis within 35 deg, the edge at most 45 mm inside the hand, the grasp site within 15 mm "
                             "of the board's slab) holds that board until the hand opens or the hold exceeds a limit"},
            "cameras": cams,
            "camera_budget_frames": self.frame_budget,
            "room": {"size": [RM.ROOM_X, RM.ROOM_Y, RM.WALL_H], "door_x": list(RM.DOOR_X),
                     "window_y": list(RM.WINDOW_Y),
                     "sideboard": {"center": list(RM.SIDEBOARD["center"]), "size": list(RM.SIDEBOARD["size"])},
                     "kitting_table": {"center": list(RM.TABLE["center"]), "size": list(RM.TABLE["size"]),
                                       "height": RM.TABLE["height"], "battens": {"height": RM.TABLE["batten_h"],
                                                                                 "width": RM.TABLE["batten_w"],
                                                                                 "offsets_y": [-RM.TABLE["batten_dy"],
                                                                                               RM.TABLE["batten_dy"]]}},
                     "couch": {"footprint_xy": list(RM.COUCH["footprint"]),
                               "boxes": {k: list(v) for k, v in RM.COUCH["boxes"].items()},
                               "legs": {"centers": [list(c) for c in RM.COUCH["legs"]["centers"]],
                                        "side": RM.COUCH["legs"]["side"], "height": RM.COUCH["legs"]["height"]},
                               "note": "fixed obstacle against the east wall, seat towards the west; the boxes are "
                                       "its collision envelope, [x0, y0, z0, x1, y1, z1], legs square, standing on "
                                       "the floor; the drawn couch lies within the footprint to 2 mm"},
                     "plant": {k: {"center": list(v["center"]), "radius": v["radius"], "z": list(v["z"])}
                               for k, v in RM.PLANT.items()} | {"note": "fixed obstacle in the north-west corner: "
                                                                        "two upright cylinders, its collision "
                                                                        "envelope (the drawn leaves reach up to 8 cm "
                                                                        "beyond it)"},
                     "rug": {"center": list(RM.RUG["center"]), "size": list(RM.RUG["size"]),
                             "thickness": RM.RUG["thickness"],
                             "note": "flat on the floor in front of the couch, drawn only: no obstacle, no support"}},
            "items": "the bookcase's parts as unpacked: the tall boards stand upright in the panel cart's grooves, the "
                     "other boards lie flat on the kitting table's battens; nothing else in the room is loose (the "
                     "couch, the plant and the rug are the customer's and stay where they are)",
            "budgets": {"sim_cap_s": self.sim_cap,
                        "program_time_cap_s": float(self.cfg.get("program_time_cap_s", 3000.0)),
                        "program_time": "the episode's wall clock minus the cell's own time inside step (simulation) "
                                        "and get_camera (rendering)",
                        "silence_s": float(self.cfg.get("silence_s", 120.0))},
            "safety": {"robot_contact_N": 150.0, "robot_robot_N": 80.0, "part_speed_m_s": 3.0,
                       "part_force_N": 400.0},
        }

    def _localised(self, r):
        x, y, yaw = (float(v) for v in self.d.qpos[r.base_q[:3]])
        return (x + float(self.rng.normal(0, LOC_NOISE[0])), y + float(self.rng.normal(0, LOC_NOISE[0])),
                yaw + float(self.rng.normal(0, LOC_NOISE[1])))

    def _site_pose(self, sid):
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, self.d.site_xmat[sid])
        return [float(v) for v in self.d.site_xpos[sid]] + [float(v) for v in q]

    def get_state(self) -> dict:
        d = self.d
        out = {"sim_time": round(self.sim_time, 4), "done": self.done, "robots": {}}
        for n, r in self.robots.items():
            c = self.ctrl[n]
            q = {}
            for s in SIDES:
                for jn, qa in zip(ARM_JOINTS, r.arm_q[s]):
                    q[f"{s}_{jn}_joint"] = float(d.qpos[qa])
                for jn, qa in zip(HAND_JOINTS, r.hand_q[s]):
                    q[f"{s}_hand_{jn}_joint"] = float(d.qpos[qa])
            for jn, qa in zip(WAIST_JOINTS, r.waist_q):
                q[f"{jn}_joint"] = float(d.qpos[qa])
            hands = {}
            for s in SIDES:
                hands[s] = {"closed": c.hand_cmd[s] == "close", "tactile_N": round(self.grips.tactile(n, s), 2),
                            "grasp_pose_in_robot": self._grasp_in_base(r, s),
                            "ik_residual": [round(c.arm_ik_err[s][0], 5), round(c.arm_ik_err[s][1], 5)]
                            if c.arm_mode[s] == "pose" else None}
            out["robots"][n] = {
                "base_pose": [round(v, 5) for v in self._localised(r)],
                "stand_height": round(PELVIS_HEIGHT + float(d.qpos[r.base_q[3]]), 4),
                "base_moving": bool(np.linalg.norm(c.vel) > 1e-4),
                "q": {k: round(v, 5) for k, v in q.items()},
                "hands": hands,
            }
        return out

    def _grasp_in_base(self, r, side):
        """The grasp site's pose in the robot's base frame (exact kinematics): x, y, z, qw, qx, qy, qz."""
        d = self.d
        x, y, yaw = (float(v) for v in d.qpos[r.base_q[:3]])
        c, s = math.cos(yaw), math.sin(yaw)
        Rb = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        sid = r.grasp_site[side]
        p = Rb.T @ (d.site_xpos[sid] - np.array([x, y, 0.0]))
        Rs = Rb.T @ d.site_xmat[sid].reshape(3, 3)
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, Rs.reshape(-1))
        return [round(float(v), 5) for v in p] + [round(float(v), 6) for v in q]

    def command(self, robot: str, cmd: dict) -> dict:
        if robot not in self.robots:
            return {"error": "bad_params"}
        notes = self.ctrl[robot].command(cmd)
        for s in SIDES:
            h = cmd.get(f"{s}_hand")
            if h in ("open", "close"):
                self.grips.command(robot, s, h == "close", self.sim_time)
        return {"ok": True, "notes": notes}

    def step(self, n_ticks: int) -> dict:
        events = []
        n_viol = len(self.safety.violations)
        t = time.time()
        try:
            for _ in range(int(n_ticks)):
                if self.done:
                    break
                self._tick()
                if self.sim_time >= self.sim_cap - 1e-9:
                    self._end("sim_cap")
                    events.append("episode_over")
                    break
        finally:
            self.work_wall += time.time() - t
        for v in self.safety.violations[n_viol:]:
            events.append("safety:" + v["what"])
        return {"sim_time": round(self.sim_time, 4), "done": self.done, "events": events}

    def get_camera(self, name: str, depth: bool = True) -> dict:
        if name not in PUBLIC_CAMERAS:
            return {"error": "bad_params"}
        if self.frames_total >= self.frame_budget:
            return {"error": "camera_budget"}
        self.frames_total += 1
        self.frames[name] = self.frames.get(name, 0) + 1
        t_frame = time.time()
        try:
            return self._frame(name, depth)
        finally:
            self.work_wall += time.time() - t_frame

    def _frame(self, name: str, depth: bool) -> dict:
        t = time.time()
        mjname = _cam_mjname(name)
        kind = "head" if "head" in name else "ceiling"
        rgb, dep = self.cams.render(self.render_state(), mjname, depth=depth, kind=kind)
        self.render_wall += time.time() - t
        if "head" in name:
            w, h = HEAD_CAM["size"]
            intr = intrinsics(w, h, HEAD_CAM["fovy"])
        else:
            c = RM.CEILING_CAMERAS[name]
            intr = intrinsics(c["size"][0], c["size"][1], c["fovy"])
        out = encode_frame(name, self.frames[name] - 1, rgb, dep, intr)
        if "head" in name:
            r = self.robots[name.split("/")[0]]
            cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, mjname)
            p, R = camera_pose_world(self.m, self.d, cid)
            # composed with the localised base pose (the same noise model as get_state)
            x, y, yaw = (float(v) for v in self.d.qpos[r.base_q[:3]])
            lx, ly, lyaw = self._localised(r)
            dyaw = lyaw - yaw
            cz, sz = math.cos(dyaw), math.sin(dyaw)
            Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
            p2 = Rz @ (p - np.array([x, y, 0.0])) + np.array([lx, ly, 0.0])
            out["pose_in_room"] = {"position": [float(v) for v in p2], "rotation": (Rz @ R).tolist()}
        else:
            cid = mujoco.mj_name2id(self.m, mujoco.mjtObj.mjOBJ_CAMERA, name)
            p, R = camera_pose_world(self.m, self.d, cid)
            out["pose_in_room"] = {"position": [float(v) for v in p], "rotation": R.tolist()}
        out["sim_time"] = round(self.sim_time, 4)
        return out

    def finish(self) -> dict:
        if not self.done:
            self._end("finish")
        return {"ok": True}

    def render_state(self) -> mujoco.MjData:
        """The state every camera image is drawn from (colour, depth): the physics state's kinematics (of the qpos
        its last substep started from) with each robot's 12 leg joints as its walker draws them now. Everything above
        the hips, the parts and the room are the physics' own, bit for bit; the physics state is not touched."""
        if not self.walkers:
            return self.d
        if self._rd is None:
            self._rd = mujoco.MjData(self.m)
        rd = self._rd
        rd.qpos[:] = self._q_kin
        rd.time = self.d.time
        for w in self.walkers.values():
            rd.qpos[w.qpos_adr] = w.pose_at(self.sim_time, self._q_kin[w.base_adr])
        mujoco.mj_kinematics(self.m, rd)
        mujoco.mj_comPos(self.m, rd)
        mujoco.mj_camlight(self.m, rd)
        return rd

    def _end(self, reason: str):
        self.done = True
        self.done_reason = reason

    # ------------------------------------------------------------------------------------------------ grader side
    def part_pose(self, key: str):
        b = self.bodies.part[key]
        return self.d.xpos[b].copy(), self.d.xmat[b].reshape(3, 3).copy()

    def score(self) -> dict:
        """The episode's assembly score and its record (grader only)."""
        from .scoring import score_cell
        return score_cell(self)
