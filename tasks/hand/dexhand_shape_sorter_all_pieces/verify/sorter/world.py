"""The shape-sorter world on public SuperDex 1.0.0 (double precision, single-threaded).

World(layout) builds: a static table (top at z = 0), the static shape_box body + lid (public prefab geometry, box
collision SDF re-baked at 0.5 mm from the public collision mesh), the 11 movable pieces at the layout poses, and the
public Wuji Hand 2 (beta1, right) hand with a floating, gravity-compensated wrist driven by an implicit PD controller.

The grader runs exactly:   w = World(layout); w.settle(); for wp in plan: w.step_waypoint(wp); w.hold(); w.result()
"""
from __future__ import annotations

import json
import math
import os

os.environ.setdefault("SUPERDEX_PRECISION", "double")
import numpy as np  # noqa: E402

import superdex.physics as sp  # noqa: E402
import superdex.robotics as sr  # noqa: E402

from . import config as C  # noqa: E402
from . import plan as P  # noqa: E402
from . import quat as Q  # noqa: E402
from .score import Tracker  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
GEOMETRY = json.load(open(os.path.join(HERE, "geometry.json")))

if not sp.USE_DOUBLE_PRECISION:
    raise RuntimeError("SUPERDEX_PRECISION must be 'double' before superdex is imported")


def assets_root():
    return os.environ.get("SUPERDEX_ASSETS_PATH", "/opt/superdex_assets")


def data_root():
    return os.environ.get("SORTER_DATA", "/opt/sorter_data")


def piece_shape_file(name):
    return os.path.join(assets_root(), "prefabs", "shape_box", "collision", name + ".mochi.h5")


def box_shape_file(name):
    """Fine (0.5 mm) SDF re-baked from the public collision mesh (see bake_box_sdf.py)."""
    return os.path.join(data_root(), name + "_fine.mochi.h5")


def _init():
    if not sp.is_initialized():
        sp.initialize(0)  # single-threaded: results are bit-identical across processes


def _tf(p, q=(0.0, 0.0, 0.0, 1.0)):
    return sp.TransformRT(rotation=[float(x) for x in q], translation=[float(x) for x in p])


def box_pose(layout):
    """Cube placement of a scene: nominal BOX_POS plus the scene's fixed offset (dx, dy in metres, yaw in radians
    about +z), given in the start state as {"box": {"dx": .., "dy": .., "yaw": ..}}; zero if absent.
    Returns (position of the body frame origin, quaternion xyzw)."""
    b = layout.get("box", {}) or {}
    dx, dy, yaw = float(b.get("dx", 0.0)), float(b.get("dy", 0.0)), float(b.get("yaw", 0.0))
    p = np.array(C.BOX_POS, float) + np.array([dx, dy, 0.0])
    q = np.array([0.0, 0.0, math.sin(0.5 * yaw), math.cos(0.5 * yaw)])
    return p, q


def _contact(mu, prefab=True):
    c = sp.ContactParams()
    c.coulomb_friction_coefficient = float(mu)
    c.friction_falloff_vel = C.FRICTION_FALLOFF
    if prefab:
        c.penalty_coefficient = C.PREFAB_CONTACT["penalty"]
        c.penalty_smoothing_half_distance = C.PREFAB_CONTACT["smooth"]
        c.penalty_threshold_default = C.PREFAB_CONTACT["thresh"]
        c.penalty_threshold_extra_padding = C.PREFAB_CONTACT["pad"]
    return c


def physics_params(layout):
    """Per-scene physics of a start state: {"physics": {...}} with any of the keys of C.PHYSICS_NOMINAL; missing keys
    (and a missing "physics" entry) take the nominal values. The shipped scenes have no "physics" entry, so the
    simulator you get is the nominal one; the graded hidden scenes use values inside C.PHYSICS_BOUNDS."""
    out = dict(C.PHYSICS_NOMINAL)
    for k, v in (layout.get("physics", {}) or {}).items():
        if k not in out:
            raise ValueError("unknown physics parameter %r" % k)
        out[k] = float(v)
    return out


def calibration_error(layout):
    """Per-scene hand-eye calibration error of a start state: {"calibration": {"dp": [dx, dy, dz], "q": [qx, qy, qz, qw]}}
    (metres, world axes; unit quaternion of a small rotation about the wrist origin, world axes). When the plan commands
    the wrist pose (p, q), the controller drives the wrist towards (p + dp, q_err * q). Missing entry = no error. The
    shipped scenes have no "calibration" entry; the graded hidden scenes use values inside C.CALIB_BOUNDS.
    Returns (dp, q_err, active)."""
    c = layout.get("calibration")
    if not c:
        return np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), False
    for k in c:
        if k not in ("dp", "q"):
            raise ValueError("unknown calibration entry %r" % k)
    dp = np.array(c.get("dp", [0.0, 0.0, 0.0]), float)
    q = Q.normalize(np.array(c.get("q", [0.0, 0.0, 0.0, 1.0]), float))
    if dp.shape != (3,) or q.shape != (4,) or not (np.all(np.isfinite(dp)) and np.all(np.isfinite(q))):
        raise ValueError("calibration entry must be dp[3], q[4]")
    return dp, q, True


class World:
    def __init__(self, layout):
        _init()
        self.physics = ph = physics_params(layout)
        self.calib_dp, self.calib_q, self.calib_active = calibration_error(layout)
        self.scene = s = sp.create_scene("shape_sorter")
        s.set_gravity(list(C.GRAVITY))
        p = s.get_solver_params()
        p.integration_method = sp.IntegrationMethod.BACKWARD_EULER
        nl = p.non_linear_solver
        nl.max_iter = C.NEWTON_ITERS
        nl.abs_tol = C.ABS_TOL
        p.non_linear_solver = nl
        ee = p.experimental_eval
        ee.fade_friction = False
        p.experimental_eval = ee
        s.set_solver_params(p)
        # table: a static box mesh, top face at z = 0
        hx, hy, hz = (0.5 * v for v in C.TABLE_SIZE)
        verts = [[sx * hx, sy * hy, sz * hz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
        faces = [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
                 [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
        shape = sp.create_tri_mesh_shape([float(v) for r in verts for v in r], [int(i) for f in faces for i in f])
        s.create_rigid_actor(name="table", shape=shape, collider_type=sp.ColliderType.BOX, is_static=True,
                             contact=_contact(C.MU_TABLE, prefab=False), world_from_local=_tf([0, 0, -hz]))
        bp, bq = box_pose(layout)
        self.box_p, self.box_q = bp, bq
        self.box_body = s.create_rigid_actor(name="shapebox_body", shape=sp.load_shape_from_file(box_shape_file("shape_box_body")),
                             collider_type=sp.ColliderType.SDF, is_static=True, contact=_contact(ph["mu_cube"]),
                             world_from_local=_tf(bp, bq))
        self.box_lid = s.create_rigid_actor(name="shapebox_lid", shape=sp.load_shape_from_file(box_shape_file("shape_box_lid")),
                             collider_type=sp.ColliderType.SDF, is_static=True, contact=_contact(ph["mu_cube"]),
                             world_from_local=_tf(bp + Q.apply(bq, np.array(C.LID_OFFSET)), bq))
        self.pieces = {}
        for name in C.PIECES:
            pose = layout["pieces"][name]
            a = s.create_rigid_actor(name=name, shape=sp.load_shape_from_file(piece_shape_file(name)),
                                     collider_type=sp.ColliderType.SDF, is_static=False, density=C.PIECE_DENSITY * ph["mass_scale"],
                                     contact=_contact(ph["mu_piece"]), world_from_local=_tf(pose["pos"], pose["quat"]))
            sp.experimental.enable_newton_euler_inertia(a, True)
            self.pieces[name] = a
        self.centroid_local = {n: np.array(GEOMETRY["pieces"][n]["centroid"], float) for n in C.PIECES}
        # hand
        prefab = sr.load_bot_prefab_from_file(os.path.join(assets_root(), C.HAND_BOT))
        for lk in prefab.links:
            lk.has_gravity = False  # gravity-compensated hand (like a real arm+hand controller)
            c = lk.contact          # asset contact settings, except a small friction regularisation velocity:
            c.friction_falloff_vel = C.HAND_FRICTION_FALLOFF   # the asset's 0.01 m/s lets pinched pieces creep
            if ph["finger_mu_scale"] != 1.0:
                c.coulomb_friction_coefficient = c.coulomb_friction_coefficient * ph["finger_mu_scale"]
            lk.contact = c
        for j in prefab.joints:
            if str(j.type).upper().endswith("REVOLUTE") and j.name.endswith(C.NO_HYPEREXTENSION_SUFFIXES):
                mn = list(j.min_limit)
                mn[int(np.argmax(np.abs(list(j.axis))))] = 0.0
                j.min_limit = mn
        prefab.world_from_root = _tf(C.PARK_POS, C.PARK_QUAT)
        self.prefab = prefab
        self.bot = sr.create_bot(s, prefab, sr.create_context())
        self.hand = h = self.bot.get_articulated_actor()
        self.nl, self.nd = len(prefab.links), h.get_num_dofs()
        lim = sp.DynamicArrayReal2()
        lim.resize(self.nd)
        h.get_articulated_dof_limits(lim)
        self.finger_lo = np.array([lim[i][0] for i in range(6, self.nd)])
        self.finger_hi = np.array([lim[i][1] for i in range(6, self.nd)])
        params = sp.PoseControllerParams(self.nl)
        jt, lp, lr = list(params.joint_tracking), list(params.link_pos_tracking), list(params.link_rot_tracking)
        for i, j in enumerate(prefab.joints):
            if str(j.type).upper().endswith(("HARD", "FREE")):
                jt[i].stiffness, jt[i].damping = 0.0, 0.0
            else:
                jt[i].stiffness, jt[i].damping = C.FINGER_KP * ph["finger_gain"], C.FINGER_KD * ph["finger_gain"]
        lp[0].stiffness, lp[0].damping = C.WRIST_KP, C.WRIST_KD
        lr[0].stiffness, lr[0].damping = C.WRIST_KR, C.WRIST_KRD
        params.joint_tracking, params.link_pos_tracking, params.link_rot_tracking = jt, lp, lr
        h.add_articulated_pose_controller(params)
        sp.experimental.enable_newton_euler_inertia(h, True)
        q = np.zeros(self.nd)
        h.get_articulated_pose(q)
        self.finger_default = q[6:].copy()
        self._p0 = np.array(C.PARK_POS, float)
        self._q0 = np.array(C.PARK_QUAT, float)
        self._buf = sp.DynamicArrayTransformRT()
        self._buf.resize(self.nl)
        # current command (the previous waypoint): park pose, default fingers
        self.cmd = (self._p0.copy(), self._q0.copy(), self.finger_default.copy())
        self.t = 0.0
        self.nonfinite = False
        # wrist speed limit state: the last wrist set-point (pos, quat), starting at the park pose (so that the settle
        # also moves a calibration-offset set-point at the limited rate), and the last actual wrist pose
        self._sp = (self._p0.copy(), self._q0.copy())
        self._wprev = None
        self.tracker = Tracker(GEOMETRY, bp, bq)
        self.last_poses = self.piece_poses()
        self.tracker.reset(self.piece_centroids(self.last_poses))

    def close(self):
        """Free the simulator scene (SuperDex keeps scenes alive until destroy_scene)."""
        if getattr(self, "scene", None) is not None:
            sp.destroy_scene(self.scene)
            self.scene = None

    # ------------------------------------------------------------------ state
    def wrist_pose(self):
        self.hand.get_articulated_link_transforms(self._buf)
        t = self._buf[0]
        return np.array(list(t.translation), float), np.array(list(t.rotation), float)

    def hand_link_poses(self):
        self.hand.get_articulated_link_transforms(self._buf)
        return [(np.array(list(self._buf[i].translation)), np.array(list(self._buf[i].rotation))) for i in range(self.nl)]

    def hand_q(self):
        q = np.zeros(self.nd)
        self.hand.get_articulated_pose(q)
        return q

    def piece_poses(self):
        out = {}
        for n, a in self.pieces.items():
            t = a.get_root_transform()
            out[n] = (np.array(list(t.translation), float), np.array(list(t.rotation), float))
        return out

    def piece_centroids(self, poses=None):
        poses = self.piece_poses() if poses is None else poses
        return {n: Q.apply(q, self.centroid_local[n]) + p for n, (p, q) in poses.items()}

    def capture(self):
        """Exact simulator state + controller/tracker state (restore() continues bit-identically)."""
        buf = sp.DynamicArrayUint8()
        self.scene.capture_state_to_bytes(buf)
        return dict(scene=np.array(buf, dtype=np.uint8), cmd=tuple(np.copy(x) for x in self.cmd), t=self.t,
                    nonfinite=self.nonfinite, tracker=self.tracker.state(),
                    sp=None if self._sp is None else tuple(np.copy(x) for x in self._sp),
                    wprev=None if self._wprev is None else tuple(np.copy(x) for x in self._wprev),
                    last_poses={k: (v[0].copy(), v[1].copy()) for k, v in self.last_poses.items()})

    def restore(self, snap):
        self.scene.restore_state_from_bytes(snap["scene"])
        self.cmd = tuple(np.copy(x) for x in snap["cmd"])
        self.t = snap["t"]
        self.nonfinite = snap["nonfinite"]
        self._sp = None if snap["sp"] is None else tuple(np.copy(x) for x in snap["sp"])
        self._wprev = None if snap["wprev"] is None else tuple(np.copy(x) for x in snap["wprev"])
        self.tracker.set_state(snap["tracker"])
        self.last_poses = {k: (v[0].copy(), v[1].copy()) for k, v in snap["last_poses"].items()}

    # ------------------------------------------------------------------ control
    def _set_hand_target(self, pos, quat, fingers):
        # hand-eye calibration error of the scene: commanded pose -> actual set-point (no-op without an error)
        if self.calib_active:
            pos = pos + self.calib_dp
            quat = Q.normalize(Q.mul(self.calib_q, quat))
        wp, wq = self.wrist_pose()
        sp = self._sp   # previous set-point (the park pose before the first physics step)
        if sp is not None:
            # wrist speed limit, part 1: the set-point moves at most SP_MAX_*_SPEED * SIM_DT per physics step
            d = pos - sp[0]
            n = math.sqrt(float(d @ d))
            lim = C.SP_MAX_LIN_SPEED * C.SIM_DT
            if n > lim:
                pos = sp[0] + d * (lim / n)
            if abs(float(sp[1] @ quat)) < C.COS_HALF_SP_ROT_STEP:
                quat = Q.clamp_towards_cos(sp[1], quat, C.COS_HALF_SP_ROT_STEP)
        # wrist speed limit, part 2: actual wrist velocity over the last physics step (zero before the first one)
        if self._wprev is not None:
            vel = (wp - self._wprev[0]) / C.SIM_DT
            dq = Q.mul(wq, Q.conj(self._wprev[1]))
            om = dq[:3] * ((2.0 if dq[3] >= 0.0 else -2.0) / C.SIM_DT)   # small-angle angular velocity, world frame
        else:
            vel = om = None
        self._wprev = (wp, wq)
        # clamp the set-point to stay within SETPOINT_CLAMP_* of the actual wrist (bounded wrist force); the allowed
        # lead shrinks with the actual wrist speed towards the set-point (zero lead at WRIST_V0_*)
        d = pos - wp
        n = math.sqrt(float(d @ d))
        vt = float(vel @ d) / n if (vel is not None and n > 0.0) else 0.0
        if vt > 0.0:
            lim = C.SETPOINT_CLAMP_POS * max(0.0, 1.0 - vt / C.WRIST_V0_LIN)
            if n > lim:
                pos = wp + d * (lim / n)
        elif n > C.SETPOINT_CLAMP_POS:
            pos = wp + d * (C.SETPOINT_CLAMP_POS / n)
        quat = P.orientation_limit_project(quat)   # palm angle and finger pitch limits (config.WRIST_MAX_PALM_DEG,
        #                                           config.FINGER_MAX_PITCH_DEG); no-op inside them
        r = Q.mul(quat, Q.conj(wq))           # rotation from the actual wrist to the set-point
        rn = math.sqrt(float(r[:3] @ r[:3]))
        wt = float(om @ r[:3]) / rn * (1.0 if r[3] >= 0.0 else -1.0) if (om is not None and rn > 0.0) else 0.0
        if wt > 0.0:
            s = C.SIN_HALF_CLAMP_ROT * max(0.0, 1.0 - wt / C.WRIST_V0_ANG)
            quat = Q.clamp_towards_cos(wq, quat, math.sqrt(1.0 - s * s))
        else:
            quat = Q.clamp_towards(wq, quat, C.SETPOINT_CLAMP_ROT)
        self._sp = (np.array(pos, float), np.array(quat, float))
        q = np.empty(self.nd)
        q[0:3] = Q.apply(Q.conj(self._q0), pos - self._p0)          # FREE root translation: park frame
        q[3:6] = Q.to_rotvec(Q.mul(Q.conj(self._q0), quat))          # FREE root rotation: R = R_park exp(r)
        q[6:] = fingers
        self.hand.set_articulated_target_pose(q)

    def _physics_step(self):
        self.scene.step(C.SIM_DT)
        self.t += C.SIM_DT
        poses = self.piece_poses()
        c = self.piece_centroids(poses)
        if not all(np.all(np.isfinite(v)) for v in c.values()) or not np.all(np.isfinite(self.wrist_pose()[0])):
            self.nonfinite = True  # the episode stops here; a non-finite scene scores 0 (score.py)
            return False
        self.last_poses = poses
        self.tracker.update(c)
        return True

    def settle(self, seconds=C.SETTLE_PRE_S):
        # (with a calibration error the hand starts at the park pose and this pre-plan settle moves it to the park pose
        # plus the error)
        p, q, f = self.cmd
        for _ in range(int(round(seconds / C.SIM_DT))):
            if self.nonfinite:
                return
            self._set_hand_target(p, q, f)
            self._physics_step()

    def step_waypoint(self, wp):
        """One control period: the wrist / finger set-points move linearly (nlerp for the orientation) from the
        previous waypoint to `wp` over SUBSTEPS physics steps. wp = (pos[3], quat_xyzw[4], fingers[20])."""
        if self.nonfinite:
            return
        p0, q0, f0 = self.cmd
        p1, q1, f1 = np.asarray(wp[0], float), Q.normalize(np.asarray(wp[1], float)), np.asarray(wp[2], float)
        if float(q0 @ q1) < 0.0:
            q1 = -q1
        for j in range(1, C.SUBSTEPS + 1):
            a = j / C.SUBSTEPS
            self._set_hand_target(p0 + a * (p1 - p0), Q.normalize(q0 + a * (q1 - q0)), f0 + a * (f1 - f0))
            if not self._physics_step():
                break
        self.cmd = (p1, q1, f1)

    def hold(self, seconds=C.SETTLE_POST_S):
        self.settle(seconds)

    def state_digest(self):
        """sha256 over the exact final piece poses and hand joint state (bit-identity checks)."""
        import hashlib
        h = hashlib.sha256()
        for n in C.PIECES:
            p, q = self.last_poses[n]
            h.update(np.asarray(p, np.float64).tobytes())
            h.update(np.asarray(q, np.float64).tobytes())
        h.update(np.asarray(self.hand_q(), np.float64).tobytes())
        return h.hexdigest()

    def result(self):
        """Per-piece outcome (scoring rule in score.py)."""
        r = self.tracker.result(self.last_poses, nonfinite=self.nonfinite, sim_time=self.t)
        r["state_sha256"] = self.state_digest()
        return r


def run_plan(layout, waypoints, progress=None):
    """Grader entry: settle, execute every waypoint, hold, score. `waypoints` = validated list (plan.py).
    The grader runs each scene in its own fresh process with one World."""
    w = World(layout)
    w.settle()
    for k, wp in enumerate(waypoints):
        w.step_waypoint(wp)
        if w.nonfinite:
            break
        if progress is not None and k % 250 == 0:
            progress(k)
    w.hold()
    return w.result()
