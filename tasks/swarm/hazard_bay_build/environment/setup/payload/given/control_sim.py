"""The bay mission: the agent's controller drives the whole fleet through the plant's job: the ground vehicle closes the
leaking line's valve and clears the assembly station into the crate, the three flying robots lift the bucket on its
ropes, carry it to the burning pan and pour. The grader's `control` probe runs it in its own stage of the real bay with
the agent's four robots in it (`control.md`, The real bay); the agent's practice runner (`~/given/practice_control.py`)
runs this same simulation, with the same random draws, on the agent's own scene. Only the stage, the controller
process, the seeds and the account differ (the grader measures the outcomes in the real bay; the practice runner, which
knows only the agent's scene, against that scene).

What varies per run (`draw_hidden`, from the run's seed): the water and the gust (as always), and the bay of the day:
the six parts placed at random on the two tables (2 to 4 on each, upright, turned to any heading), the vehicle up to
0.15 m and 8 degrees off its parking bay's nominal pose (its odometry starting from the nominal pose), the valve's lever
left partly turned toward closed, and the pan moved up to 0.30 m from its nominal place.

What the controller does is its own business. The grader only steps the physics, applies what the controller commands
(thrust vectors for the flying robots; joint velocity and position targets and a suction grip for the vehicle), keeps
the bay's rules itself (the steam, the fire, the hazards), and measures what happened: the steam released, the fire
out, the parts in the crate, the time, the lift and the swing and the water, the losses, the collisions.

Flying robots: attitude held level by a perfect inner loop (the body's rotation is reset every step); position is pure
dynamics (thrust, gravity, rope, wind). Water: the bucket holds V litres (drawn per run), its mass follows the water;
the free surface stands normal to the apparent gravity; water over the rim leaves at a rate proportional to the
excess; water that leaves over the pan is poured, elsewhere it is spilt.
"""
import json
import math
import random

import numpy as np
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

UAVS = ("UAV_1", "UAV_2", "UAV_3")
ROBOTS = UAVS + ("UGV",)
FAC, FLEET = "/World/Facility", "/World/Fleet"
G = 9.81

DEFAULT = dict(
    dt_physics=1.0 / 120.0, substeps=2, time_limit_s=600.0, cruise_z=2.6,
    pan_center=[2.60, 1.90], pour_r=0.6, pour_zmax=2.5, rope_length=1.5,
    bucket=dict(mass_empty=1.2, radius=0.12, height=0.25, water_min_l=8.5, water_max_l=10.0, spill_rate_l_per_s_m=15.0),
    tilt_max_deg=40.0, wind=dict(t_min=8.0, t_max=40.0, duration_s=2.5, speed_mps=7.0, area_uav=0.12, area_bucket=0.06),
    noise=dict(vio_walk_m_sqrt_s=0.02, vio_white_m=0.01, vel_m_s=0.03, load_n=0.4, lidar_m=0.02, lidar_rel=0.01,
               cam_m=0.02, cam_rel=0.02, tilt_deg=1.0, odo_walk_m_sqrt_m=0.02, odo_yaw_walk_deg_sqrt_rad=1.0, gyro_walk_deg_sqrt_s=0.05,
               odo_white_m=0.005,
               joint_rad=0.001, det_m=0.01, det_rel=0.01, gauge_bar=0.1),
    lidar=dict(**{"LIDAR-2D": dict(range_m=25.0, vfov_deg=0.0, band_m=0.25), "LIDAR-3D": dict(range_m=30.0, vfov_deg=90.0, band_m=None)}),
    separation=dict(full_m=1.2, collide_m=0.9), crash_z=0.30, done_hold_s=3.0,
    ugv=dict(vel_max_rad_s=40.0, grip_reach_m=0.04, see_range_m=8.0, still_mps=0.1),
    rules=dict(steam_stop_s=3.0, pressure_bar=8.0, lever_closed_deg=85.0, plume_s=1.0, fire_r=2.0, fire_s=1.0, fire_out_l=1.0),
    # where the robots are (the real bay: the agent's fleet mounted elsewhere, the plant's ropes; practice: the scene's)
    fleet_root=FLEET, rope_root=None, item_alias=None, table_tops=None, pan_nominal=None,
    # the bay of the day: what draw_hidden draws (control.md, What varies per run)
    hidden=dict(per_table=[2, 4], edge_m=0.06, gap_m=0.05, tries=400, ugv_xy_m=0.15, ugv_yaw_deg=8.0, lever_deg=[0.0, 30.0],
                pan_m=0.30, preroll_s=1.0),
)
# what a robot may touch without it counting as a collision: the floor and the hall, the parts (the vehicle handles
# them), the valve's lever (the vehicle works it), the markings
FREE = ("/World/Facility/Floor", "/World/Facility/Items", "/World/Facility/SteamLine/Handle", "/World/Facility/Pads",
        "/World/Facility/CalibMat")


def draw_hidden(seed, cfg=None):
    """The bay of the day for one run, from its own stream of the run's seed (the noise stream, random.Random(seed), is
    another): the vehicle's offset from its nominal pose (dx, dy [m], dpsi [deg]), the lever's opening [deg], the pan's
    offset (dx, dy [m]) and the seed the parts' places are drawn from."""
    h = dict(DEFAULT["hidden"], **(((cfg or {}).get("hidden")) or {}))
    rng = random.Random(f"bay-hidden:{int(seed)}")
    return dict(spec=h, ugv=[rng.uniform(-h["ugv_xy_m"], h["ugv_xy_m"]), rng.uniform(-h["ugv_xy_m"], h["ugv_xy_m"]),
                             rng.uniform(-h["ugv_yaw_deg"], h["ugv_yaw_deg"])],
                lever_deg=rng.uniform(*h["lever_deg"]), pan=[rng.uniform(-h["pan_m"], h["pan_m"]) for _ in range(2)],
                parts_seed=rng.getrandbits(63))


def rope_roots(stage, fleet_root=FLEET, rope_root=None):
    """Each flying robot's rope: the prim its links live under."""
    return {n: (rope_root or {}).get(n) or f"{fleet_root}/{n}/Rope" for n in UAVS}


def _under(path, root):
    return path == root or path.startswith(root + "/")


def move_in_world(stage, path, world_row, tag):
    """Move a prim and everything below it by a world-space transform (a Gf row-vector matrix): one op at the head of
    its op order, the move turned into its parent's frame (a reset of the transform stack is kept)."""
    prim = stage.GetPrimAtPath(path)
    if not prim.IsA(UsdGeom.Xformable):
        prim.SetTypeName("Xform")
    xf = UsdGeom.Xformable(prim)
    reset = xf.GetResetXformStack()
    parent = Gf.Matrix4d(1.0) if reset else UsdGeom.XformCache().GetParentToWorldTransform(prim)
    head = parent * world_row * parent.GetInverse()
    old = xf.GetOrderedXformOps()
    names = {o.GetOpName() for o in old}
    suffix = tag
    while f"xformOp:transform:{suffix}" in names:
        suffix += "_"
    op = xf.AddTransformOp(UsdGeom.XformOp.PrecisionDouble, suffix)
    op.Set(head)
    xf.SetXformOpOrder([op] + old, reset)


def prepare(stage, hidden=None, fleet_root=FLEET, rope_root=None):
    """Before the simulation starts: contact reports on every body of the four robots (not their ropes), every
    contact at any force with anything (a threshold or a pair list the scene authored is overridden); with the bay of
    the day, the pan moved off its nominal place and the vehicle off its nominal pose (both in USD, so that everything
    they carry moves with them)."""
    ropes = rope_roots(stage, fleet_root, rope_root)
    for name in ROBOTS:
        root = stage.GetPrimAtPath(f"{fleet_root}/{name}")
        if not root.IsValid():
            continue
        for p in Usd.PrimRange(root):
            if p.HasAPI(UsdPhysics.RigidBodyAPI) and not any(_under(p.GetPath().pathString, r) for r in ropes.values()):
                cr = PhysxSchema.PhysxContactReportAPI.Apply(p)
                cr.CreateThresholdAttr(0.0)
                cr.CreateReportPairsRel().SetTargets([])  # (a pair list would report only the contacts with those)
    if not hidden:
        return
    pan = stage.GetPrimAtPath(f"{FAC}/FirePan")
    if pan.IsValid():
        move_in_world(stage, pan.GetPath().pathString, Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(hidden["pan"][0], hidden["pan"][1], 0.0)), "arb_day")
    else:
        hidden["pan"] = [0.0, 0.0]
    root = f"{fleet_root}/UGV"
    aroot = art_root(stage, root)
    if aroot:
        p0 = usd_pose(stage, aroot)[0]
        dx, dy, dpsi = hidden["ugv"]
        w = (Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(*(-p0))) * Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(Gf.Vec3d(0, 0, 1), dpsi))
             * Gf.Matrix4d(1.0).SetTranslate(Gf.Vec3d(p0[0] + dx, p0[1] + dy, p0[2])))
        move_in_world(stage, root, w, "arb_day")
    else:
        hidden["ugv"] = [0.0, 0.0, 0.0]


# --------------------------------------------------------------------------------------------- math
def qinv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array([w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                     w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2])


def rot(q, v):
    w, x, y, z = q
    r = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return r @ np.asarray(v, dtype=float)


def gfq(qd):
    return np.array([qd.GetReal(), *qd.GetImaginary()])


def quat_z(a):
    return np.array([math.cos(a / 2), 0.0, 0.0, math.sin(a / 2)])


def convex_hull(pts):
    """The convex hull of 2D points, counter-clockwise (monotone chain)."""
    p = sorted({(round(float(x), 9), round(float(y), 9)) for x, y in pts})
    if len(p) < 3:
        return np.array(p, dtype=float)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 0:
            lower.pop()
        lower.append(q)
    for q in reversed(p):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 0:
            upper.pop()
        upper.append(q)
    return np.array(lower[:-1] + upper[:-1], dtype=float)


def sat_gap(a, b):
    """How far apart two convex polygons are at least: the largest gap between their projections on any edge normal
    (negative: they overlap)."""
    best = -1e9
    for poly in (a, b):
        for k in range(len(poly)):
            e = poly[(k + 1) % len(poly)] - poly[k]
            nrm = np.array([-e[1], e[0]])
            ln = float(np.linalg.norm(nrm))
            if ln < 1e-12:
                continue
            nrm = nrm / ln
            pa, pb = a @ nrm, b @ nrm
            best = max(best, float(pb.min() - pa.max()), float(pa.min() - pb.max()))
    return best


def tf_pq(tf):
    return np.array(tf[:3], dtype=float), np.array([tf[6], tf[3], tf[4], tf[5]], dtype=float)


def pq_tf(p, q):
    return np.array([p[0], p[1], p[2], q[1], q[2], q[3], q[0]], dtype=np.float32)


def r3(v):
    return [round(float(x), 4) for x in v]


def fin(x):
    return float(x) if x is not None and np.isfinite(x) else None


# --------------------------------------------------------------------------------------------- the scene as authored
def usd_pose(stage, path):
    m = UsdGeom.Xformable(stage.GetPrimAtPath(path)).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    return np.array(m.ExtractTranslation()), gfq(m.ExtractRotationQuat())


def art_root(stage, root):
    p = stage.GetPrimAtPath(root)
    if not p.IsValid():
        return None
    arts = [q for q in Usd.PrimRange(p) if q.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if not arts:
        return None
    a = arts[0]
    if a.HasAPI(UsdPhysics.RigidBodyAPI):
        return a.GetPath().pathString
    bodies = [q for q in Usd.PrimRange(a) if q.HasAPI(UsdPhysics.RigidBodyAPI)]
    return bodies[0].GetPath().pathString if bodies else a.GetPath().pathString


def rigid_bodies(stage, root):
    p = stage.GetPrimAtPath(root)
    if not p.IsValid():
        return []
    return [q.GetPath().pathString for q in Usd.PrimRange(p) if q.HasAPI(UsdPhysics.RigidBodyAPI)]


def sensors_of(stage, robot_root, catalog):
    """The robot's declared sensors (prims with arb:sensor_model naming a catalog entry): cameras with their pose
    relative to the articulation root at rest and their fields of view, lidars with their model."""
    root_p, root_q = usd_pose(stage, art_root(stage, robot_root))
    out = []
    for p in Usd.PrimRange(stage.GetPrimAtPath(robot_root)):
        a = p.GetAttribute("arb:sensor_model")
        if not a or not a.IsValid() or not a.Get() or str(a.Get()) not in catalog:
            continue
        model = str(a.Get())
        entry = dict(path=p.GetPath().pathString, model=model, type=catalog[model]["type"])
        if p.IsA(UsdGeom.Camera):
            c = UsdGeom.Camera(p)
            f, h, v = c.GetFocalLengthAttr().Get(), c.GetHorizontalApertureAttr().Get(), c.GetVerticalApertureAttr().Get()
            cat = catalog[model]
            honest = (f and h and v and cat.get("focal") and abs(f - cat["focal"]) <= 0.02 * cat["focal"]
                      and abs(h - cat["hap"]) <= 0.02 * cat["hap"] and abs(v - cat["vap"]) <= 0.02 * cat["vap"])
            if not honest:
                continue  # (a camera with made-up intrinsics does not count)
            cp, cq = usd_pose(stage, entry["path"])
            entry.update(camera=True, pos_in_root=rot(qinv(root_q), cp - root_p), look_in_root=rot(qinv(root_q), rot(cq, (0, 0, -1))),
                         up_in_root=rot(qinv(root_q), rot(cq, (0, 1, 0))),
                         hfov=math.degrees(2 * math.atan(h / 2 / f)), vfov=math.degrees(2 * math.atan(v / 2 / f)))
        else:
            entry.update(camera=False)
        out.append(entry)
    return out


def in_view(cam, root_p, root_q, target):
    """Is `target` inside this camera's field of view, the camera carried by a level robot at `root_p`?"""
    cp = root_p + rot(root_q, cam["pos_in_root"])
    look, up = rot(root_q, cam["look_in_root"]), rot(root_q, cam["up_in_root"])
    right = np.cross(look, up)
    d = np.asarray(target) - cp
    dist = float(np.linalg.norm(d))
    if dist < 1e-6:
        return False
    dn = d / dist
    if float(np.dot(dn, look)) <= 0:
        return False
    ang_h = math.degrees(math.atan2(float(np.dot(dn, right)), float(np.dot(dn, look))))
    ang_v = math.degrees(math.atan2(float(np.dot(dn, up)), float(np.dot(dn, look))))
    return abs(ang_h) <= cam["hfov"] / 2 and abs(ang_v) <= cam["vfov"] / 2


def hook_of(stage, robot_root, rope_links):
    """The hook: the joint between a body of the robot and a rope link; its attachment point in the articulation root's frame."""
    links = set(rope_links)
    aroot = art_root(stage, robot_root)
    root_p, root_q = usd_pose(stage, aroot)
    for j in stage.Traverse():
        if not (j.IsA(UsdPhysics.Joint) and j.IsActive()):
            continue
        jj = UsdPhysics.Joint(j)
        b0 = [t.pathString for t in jj.GetBody0Rel().GetTargets()]
        b1 = [t.pathString for t in jj.GetBody1Rel().GetTargets()]
        ends = b0 + b1
        link_side = [e for e in ends if e in links]
        robot_side = [e for e in ends if e not in links and e.startswith(robot_root + "/")]
        if len(link_side) == 1 and len(robot_side) == 1:
            side = "0" if robot_side[0] in b0 else "1"
            lp = np.array(j.GetAttribute(f"physics:localPos{side}").Get() or (0, 0, 0), dtype=float)
            bp, bq = usd_pose(stage, robot_side[0])
            hook_w = bp + rot(bq, lp)
            return dict(body=robot_side[0], world=hook_w, in_root=rot(qinv(root_q), hook_w - root_p))
    return None


# --------------------------------------------------------------------------------------------- the bay as authored
def world_box(stage, path, any_visibility=False):
    prim = stage.GetPrimAtPath(path)
    if not prim.IsValid():
        return None
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False,
                              ignoreVisibility=any_visibility)
    r = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    return None if r.IsEmpty() else (np.array(r.GetMin()), np.array(r.GetMax()))


def own_box(stage, path):
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_, UsdGeom.Tokens.render], useExtentsHint=False)
    r = cache.ComputeUntransformedBound(stage.GetPrimAtPath(path)).ComputeAlignedRange()
    return None if r.IsEmpty() else (np.array(r.GetMin()), np.array(r.GetMax()))


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def table_tops(stage):
    """The two tables' tops from a scene as authored (the practice runner's stand-in for the real bay's): each Table_X
    prim's own box carried by its pose, the box's axis nearest the vertical taken as its up; the top is the face at the
    world box's top."""
    out = []
    for name in ("Table_A", "Table_B"):
        path = f"{FAC}/Tables/{name}"
        prim = stage.GetPrimAtPath(path)
        b, wb = own_box(stage, path) if prim.IsValid() else None, world_box(stage, path) if prim.IsValid() else None
        if b is None or wb is None:
            continue
        m = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        axes = [np.array(m.TransformDir(Gf.Vec3d(*e))) for e in ((1, 0, 0), (0, 1, 0), (0, 0, 1))]
        up = int(np.argmax([abs(a[2]) for a in axes]))
        a1, a2 = [k for k in range(3) if k != up]
        lo, hi = b
        c = np.array(m.Transform(Gf.Vec3d(*((lo + hi) / 2))))
        e1, e2 = axes[a1] * (hi[a1] - lo[a1]), axes[a2] * (hi[a2] - lo[a2])
        out.append(dict(name=name, center=[float(c[0]), float(c[1])], size=[float(np.linalg.norm(e1[:2])), float(np.linalg.norm(e2[:2]))],
                        yaw_deg=math.degrees(math.atan2(e1[1], e1[0])), height=float(wb[1][2])))
    return out


def bay_facts(stage):
    """The bay's facts from a scene as authored (the practice runner's stand-in for the real bay's)."""
    out = {}
    for key, path, vis in (("plume", f"{FAC}/SteamLine/Plume", True), ("crate", f"{FAC}/Crate", False), ("dial", f"{FAC}/SteamLine/Gauge/Dial", False),
                           ("table_a", f"{FAC}/Tables/Table_A", False), ("table_b", f"{FAC}/Tables/Table_B", False)):
        b = world_box(stage, path, vis)
        out[key] = None if b is None else [[float(v) for v in b[0]], [float(v) for v in b[1]]]
    return out


class Lever:
    """The valve's handle on its revolute joint, as the grader reads it: the angle turned from its open stop (where it
    rests as authored; unwrapped). The valve's packing holds the lever wherever it is left (the nameplate's breakaway
    torque, 4 N m, is far more than the lever's own weight): while no robot touches it, it stays where it is; while one
    pushes it, it turns, its own weight cancelled about the joint's axis."""

    def __init__(self, stage, view):
        handle = f"{FAC}/SteamLine/Handle"
        self.root = art_root(stage, f"{FAC}/SteamLine")
        self.art = view.create_articulation_view(self.root) if self.root else None
        self.dof = None
        self.grip_local = None
        self.link, self.mass = None, 0.0
        self.latched = None
        if self.art is None:
            return
        dof_names = list(self.art.shared_metatype.dof_names)
        for j in stage.Traverse():
            if not (j.IsA(UsdPhysics.RevoluteJoint) and j.IsActive()):
                continue
            jj = UsdPhysics.Joint(j)
            ends = [t.pathString for t in jj.GetBody0Rel().GetTargets()] + [t.pathString for t in jj.GetBody1Rel().GetTargets()]
            if handle in ends and j.GetName() in dof_names:
                self.dof = dof_names.index(j.GetName())
                self._axis_of(stage, j, handle)
                break
        self.last = float(self.art.get_dof_positions()[0][self.dof]) if self.dof is not None else 0.0
        self.acc = 0.0
        names = list(self.art.shared_metatype.link_names)
        self.link = names.index(handle.split("/")[-1]) if handle.split("/")[-1] in names else None
        self.mass, self.com_local = 0.0, np.zeros(3)
        if self.link is not None and self.dof is not None:
            self.mass = float(self.art.get_masses()[0][self.link])
            try:
                self.com_local = np.array(self.art.get_coms()[0][self.link][:3], dtype=float)
            except Exception:  # noqa: BLE001 - (no centre-of-mass query: the link's origin)
                self.com_local = np.zeros(3)

    def _axis_of(self, stage, j, handle):
        """The joint's pivot and axis in the world, from the side that does not turn (the valve's body)."""
        jj = UsdPhysics.Joint(j)
        b0 = [t.pathString for t in jj.GetBody0Rel().GetTargets()]
        b1 = [t.pathString for t in jj.GetBody1Rel().GetTargets()]
        side, other = ("0", b0) if handle in b1 else ("1", b1)
        if not other:  # (joined to the world: the handle's own side, at rest)
            side, other = ("1", b1) if side == "0" else ("0", b0)
        lp = np.array(j.GetAttribute(f"physics:localPos{side}").Get() or (0, 0, 0), dtype=float)
        lr = j.GetAttribute(f"physics:localRot{side}").Get()
        lr = gfq(lr) if lr is not None else np.array([1.0, 0, 0, 0])
        ax = {"X": (1, 0, 0), "Y": (0, 1, 0), "Z": (0, 0, 1)}[str(j.GetAttribute("physics:axis").Get() or "X")]
        bp, bq = usd_pose(stage, other[0]) if other else (np.zeros(3), np.array([1.0, 0, 0, 0]))
        self.pivot = bp + rot(bq, lp)
        self.axis = rot(bq, rot(lr, ax))

    def hold(self, touched=False):
        """The packing's hold for this step: untouched, the lever stays where it was; touched, a torque about the joint's
        axis cancels its weight's."""
        if self.dof is None:
            return
        if not touched:
            q = np.array(self.art.get_dof_positions(), dtype=np.float32)
            if self.latched is None:
                self.latched = float(q[0, self.dof])
            elif abs(float(q[0, self.dof]) - self.latched) > 1e-6:
                q[0, self.dof] = self.latched
                v = np.array(self.art.get_dof_velocities(), dtype=np.float32)
                v[0, self.dof] = 0.0
                idx = np.array([0], dtype=np.int32)
                self.art.set_dof_positions(q, idx)
                self.art.set_dof_velocities(v, idx)
        else:
            self.latched = None
        if self.link is None or self.mass <= 0.0:
            return
        p, q = tf_pq(self.art.get_link_transforms()[0][self.link])
        r = p + rot(q, self.com_local) - self.pivot
        tau = np.cross(r, np.array([0.0, 0.0, -self.mass * G]))
        n = self.art.max_links
        t = np.zeros((1, n, 3), dtype=np.float32)
        t[0, self.link] = -float(np.dot(tau, self.axis)) * self.axis
        self.art.apply_forces_and_torques_at_position(np.zeros((1, n, 3), dtype=np.float32), t, np.zeros((1, n, 3), dtype=np.float32),
                                                      np.array([0], dtype=np.int32), True)

    def angle(self):
        if self.dof is None:
            return 0.0
        q = float(self.art.get_dof_positions()[0][self.dof])
        self.acc += wrap(q - self.last)
        self.last = q
        return math.degrees(self.acc)

    def open_by(self, deg):
        """Leave the lever turned `deg` from its open stop toward its closed one (the side with more travel)."""
        if self.dof is None or not deg:
            return 0.0
        lo, hi = [float(v) for v in self.art.get_dof_limits()[0][self.dof]]
        lo, hi = (lo if math.isfinite(lo) else -1e3), (hi if math.isfinite(hi) else 1e3)
        sign = 1.0 if hi - self.last >= self.last - lo else -1.0
        q = np.array(self.art.get_dof_positions(), dtype=np.float32)
        q[0, self.dof] = self.last + sign * math.radians(deg)
        idx = np.array([0], dtype=np.int32)
        self.art.set_dof_positions(q, idx)
        self.art.set_dof_velocities(np.zeros_like(q), idx)
        self.last = float(q[0, self.dof])
        self.acc = sign * math.radians(deg)
        self.latched = self.last
        return sign


# --------------------------------------------------------------------------------------------- the mission
class ControlSim:
    """One episode of the bay mission on an open scene (`prepare(stage, hidden, ...)` applied before the simulation
    started). `scene` offers: stage, view (a numpy simulation view, started), step() (one physics step).
    `catalog` is the grader's catalog; `bay` the real bay's facts (None: the scene's own, for practice); `hidden` the
    run's bay of the day (`draw_hidden`; None: the scene as authored)."""

    def __init__(self, scene, catalog, cfg=None, seed=0, log=None, bay=None, hidden=None):
        self.s = scene
        self.cfg = dict(DEFAULT, **(cfg or {}))
        for k in ("bucket", "wind", "noise", "ugv", "rules", "separation", "hidden"):
            self.cfg[k] = dict(DEFAULT[k], **((cfg or {}).get(k) or {}))
        self.cat = catalog
        self.rng = random.Random(seed)
        self.log = log or (lambda *a: None)
        st = scene.stage
        self.st = st
        self.bay = bay or bay_facts(st)
        self.fleet = self.cfg["fleet_root"] or FLEET
        self.rope_roots = rope_roots(st, self.fleet, self.cfg["rope_root"])
        self.hidden = hidden
        self.dt = self.cfg["dt_physics"] * self.cfg["substeps"]
        self.robots = {}
        for name in UAVS:
            root = f"{self.fleet}/{name}"
            aroot = art_root(st, root)
            if not aroot:
                raise RuntimeError(f"{root}: no articulation")
            links = [b for b in rigid_bodies(st, self.rope_roots[name])]
            hook = hook_of(st, root, links)
            if not links or hook is None:
                raise RuntimeError(f"{root}: no rope chain hooked to the robot")
            airframe = None
            for p in Usd.PrimRange(st.GetPrimAtPath(root)):
                a = p.GetAttribute("arb:airframe_model")
                if a and a.IsValid() and a.Get():
                    airframe = str(a.Get())
            air = self.cat["airframes"].get(airframe or "") or {}
            self.robots[name] = dict(root=root, aroot=aroot, links=links, hook=hook, sensors=sensors_of(st, root, self.cat["sensors"]),
                                     thrust_max=float(air.get("thrust_max", 0.0)))
        # the pan: its nominal place (the real bay's, or the scene's own) and where it stands today
        off = np.array((hidden or {}).get("pan") or [0.0, 0.0], dtype=float)
        if self.cfg["pan_nominal"] is not None:
            self.pan_nominal = np.array(self.cfg["pan_nominal"][:2], dtype=float)
        else:
            pb = world_box(st, f"{FAC}/FirePan")
            self.pan_nominal = ((pb[0][:2] + pb[1][:2]) / 2 - off) if pb is not None else np.array(self.cfg["pan_center"], dtype=float)
        self.pan = self.pan_nominal + off
        self.bucket_path = f"{FAC}/Bucket/Body"
        self.view = scene.view
        self.av = {n: self.view.create_articulation_view(r["aroot"]) for n, r in self.robots.items()}
        self.lv = {n: self.view.create_rigid_body_view(r["links"]) for n, r in self.robots.items()}
        self.bv = self.view.create_rigid_body_view(self.bucket_path)
        self.idx = np.array([0], dtype=np.int32)
        for n, av in self.av.items():
            names = list(av.shared_metatype.link_names)
            rname = self.robots[n]["aroot"].split("/")[-1]
            self.robots[n]["root_link"] = names.index(rname) if rname in names else 0
            self.robots[n]["mass"] = float(np.sum(av.get_masses()[0]))
            p, q = tf_pq(av.get_root_transforms()[0])
            self.robots[n]["q0"] = q
            self.robots[n]["p0"] = p
            self.robots[n]["rope_mass"] = float(np.sum(self.lv[n].get_masses()))
        self._init_ugv()
        self._init_items()
        self.lever = Lever(st, self.view)
        # the water
        b = self.cfg["bucket"]
        self.water = self.rng.uniform(b["water_min_l"], b["water_max_l"])
        self.water0 = self.water
        self.bv.set_masses(np.array([b["mass_empty"] + self.water], dtype=np.float32), self.idx)
        self.spilt, self.poured = 0.0, 0.0
        # the wind
        w = self.cfg["wind"]
        self.wind_t0 = self.rng.uniform(w["t_min"], w["t_max"])
        ang = self.rng.uniform(0, 2 * math.pi)
        self.wind_dir = np.array([math.cos(ang), math.sin(ang), 0.0])
        # odometry drift per robot
        self.drift = {n: np.zeros(3) for n in ROBOTS}
        self.yaw_drift = 0.0
        self.t = 0.0
        self.k = 0
        self.thrust = {n: np.zeros(3) for n in UAVS}
        self.prev_vel_uav = {n: np.zeros(3) for n in UAVS}
        self.prev_vel_bucket = np.zeros(3)
        self.acc_bucket = np.zeros(3)
        # the bay's rules, kept by the grader
        self.closed_t = None
        self.steam = True
        self.fire = True
        self.lost = {}
        self.exposure = {n: 0.0 for n in ROBOTS}
        self.contacts_seen = {}
        self.lever_touch_t = -1e9  # (when a robot last touched the valve's lever)
        self.metrics = dict(liftoff_t=None, liftoff_tilt_max=None, swing_max=0.0, min_sep=99.0, crash=[], lost={}, collisions=[],
                            dropped=False, fire_out_t=None, steam_off_t=None, lever_closed_t=None, pour_start_t=None, done_t=None,
                            end=None, steps=0, abort=None)
        from omni.physx import get_physx_simulation_interface
        self.phx = get_physx_simulation_interface()
        self.hidden_rec = None
        if hidden:
            self._apply_hidden()
        self.bucket_start = tf_pq(self.bv.get_transforms()[0])[0].copy()

    # -- the bay of the day ---------------------------------------------------------------------------------------
    def _apply_hidden(self):
        """The run's bay of the day on the started simulation: the lever left partly turned, the parts placed at random
        on the tables; then a pre-roll with nothing commanded (the parts and the ropes settle), after which the clock,
        the contacts and the robots' starts are taken anew: at t = 0 everything is at rest. (The pan and the vehicle
        were moved in USD by `prepare`.)"""
        h = self.hidden
        spec = dict(DEFAULT["hidden"], **(h.get("spec") or {}))
        rec = dict(ugv=[round(float(v), 4) for v in h["ugv"]], lever_deg=round(float(h["lever_deg"]), 3),
                   pan=[round(float(v), 4) for v in h["pan"]], pan_center=r3(self.pan), parts={})
        rec["lever_sign"] = self.lever.open_by(float(h["lever_deg"]))
        self.physics_step()  # (the scene queries see the stage after one step)
        rec["parts"] = self._place_parts(random.Random(int(h["parts_seed"])), spec)
        starts = {n: self.uav_state(n)[0].copy() for n in UAVS}
        u0 = self.ugv_pose()[0].copy() if self.ugv else None
        for _ in range(max(0, int(round(spec["preroll_s"] / self.cfg["dt_physics"])))):
            self.physics_step()
        moved = {n: float(np.linalg.norm(self.uav_state(n)[0] - starts[n])) for n in UAVS}
        if self.ugv:
            moved["UGV"] = float(np.linalg.norm(self.ugv_pose()[0] - u0))
        rec["start_moved"] = {n: round(v, 4) for n, v in moved.items()}
        self.metrics["collisions"].clear()
        self.contacts_seen.clear()
        self.t = 0.0
        for n in UAVS:
            p, q, v = self.uav_state(n)
            self.robots[n]["p0"] = p
            self.prev_vel_uav[n] = v.copy()
        self.prev_vel_bucket = self.bucket_state()[2].copy()
        if self.ugv:  # the odometry reads the nominal pose at t = 0, wherever the vehicle really stands
            p, q, yaw = self.ugv_pose()
            self.drift["UGV"] = np.array([self.ugv["p0"][0] - p[0], self.ugv["p0"][1] - p[1], 0.0])
            self.yaw_drift = -yaw
            self._odo_last = None
            rec["ugv_true"] = [round(float(p[0]), 4), round(float(p[1]), 4), round(math.degrees(wrap(yaw)), 3)]
        rec["lever_start_deg"] = round(abs(self.lever.angle()), 3)
        self.hidden_rec = rec

    def _tops(self):
        tops = self.cfg["table_tops"] or table_tops(self.st)
        return [dict(t, path=f"{FAC}/Tables/{t['name']}") for t in tops]

    def _place_parts(self, rng, spec):
        """Each part on Table A or B (2 to 4 on each), upright as authored, turned to any heading, anywhere on its top at
        least `edge_m` inside the edges and `gap_m` from any other part; checked by raycasts that the table's top is
        what lies under it. A part with no valid place keeps its authored pose (recorded as `fallback`)."""
        if self.iv is None or not self.item_names:
            return {}
        tops = self._tops()
        if not tops:
            return {}
        from omni.physx import get_physx_scene_query_interface
        sq = get_physx_scene_query_interface()
        tfs = np.array(self.iv.get_transforms(), dtype=np.float32)
        parts = []
        for i, name in enumerate(self.item_names):
            pq = usd_pose(self.st, self.items[name]["path"])
            b = self.items[name]["box"]
            if b is None:
                continue
            lo, hi = b
            corners = np.array([(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
            fp = [rot(pq[1], c) for c in corners]
            area = (max(c[0] for c in fp) - min(c[0] for c in fp)) * (max(c[1] for c in fp) - min(c[1] for c in fp))
            parts.append(dict(i=i, name=name, q=pq[1], corners=corners, area=area))
        parts.sort(key=lambda d: -d["area"])
        n = len(parts)
        lo_n, hi_n = spec["per_table"]
        ok_counts = [k for k in range(n + 1) if lo_n <= k <= hi_n and lo_n <= n - k <= hi_n] or list(range(n + 1))
        out = {}
        for attempt in range(20):
            for _ in range(1000):
                on_a = [rng.random() < 0.5 for _ in parts]
                if sum(on_a) in ok_counts:
                    break
            placed, poses, failed = [], {}, None
            for d, a_side in zip(parts, on_a):
                top = tops[0] if a_side or len(tops) == 1 else tops[1]
                got = self._place_one(rng, d, top, placed, spec, sq)
                if got is None:
                    failed = d["name"]
                    break
                poses[d["name"]] = got
                placed.append(got["hull"])
            if failed is None:
                break
        for d in parts:
            name = d["name"]
            if name not in poses:
                out[self.wire[name]] = dict(fallback=True)
                continue
            g = poses[name]
            tfs[d["i"]] = pq_tf(g["pos"], g["quat"])
            out[self.wire[name]] = dict(table=g["table"], xy=r3(g["xy"]), heading_deg=round(g["heading"], 2))
        self.iv.set_transforms(tfs, np.arange(len(self.item_names), dtype=np.int32))
        self.iv.set_velocities(np.zeros((len(self.item_names), 6), dtype=np.float32), np.arange(len(self.item_names), dtype=np.int32))
        return out

    def _place_one(self, rng, d, top, placed, spec, sq):
        c0, (w, dep), yaw = np.array(top["center"], dtype=float), top["size"], math.radians(top["yaw_deg"])
        ca, sa = math.cos(yaw), math.sin(yaw)
        ax, ay = np.array([ca, sa]), np.array([-sa, ca])
        hw, hd = w / 2 - spec["edge_m"], dep / 2 - spec["edge_m"]
        if hw <= 0 or hd <= 0:
            return None
        for _ in range(int(spec["tries"])):
            heading = rng.uniform(0.0, 360.0)
            q = qmul(quat_z(math.radians(heading)), d["q"])
            cw = np.array([rot(q, c) for c in d["corners"]])
            hull = convex_hull(cw[:, :2])
            cen = hull.mean(axis=0)
            u, v = rng.uniform(-hw, hw), rng.uniform(-hd, hd)
            pt = c0 + u * ax + v * ay
            hull = hull - cen + pt
            rel = hull - c0
            if np.any(np.abs(rel @ ax) > hw) or np.any(np.abs(rel @ ay) > hd):
                continue
            if any(sat_gap(hull, other) < spec["gap_m"] for other in placed):
                continue
            zs = []
            for xy in [pt] + list(hull):
                hz = self._top_under(sq, xy, top)
                if hz is None:
                    break
                zs.append(hz)
            if len(zs) < 1 + len(hull):
                continue
            z_top = max(zs)
            origin = np.array([pt[0] - (cen[0] - 0.0), pt[1] - (cen[1] - 0.0), z_top - float(cw[:, 2].min()) + 0.001])
            return dict(pos=origin, quat=q, hull=hull, table=top["name"], xy=pt, heading=heading)
        return None

    def _top_under(self, sq, xy, top):
        """The height of the table's top under (x, y): the first hit of a downward ray that is not a part must be the
        table, within 1 cm of its top."""
        hits = []

        def report(hit):
            hits.append((float(hit.distance), str(hit.collision), str(hit.rigid_body)))
            return True
        z0 = float(top["height"]) + 0.3
        sq.raycast_all((float(xy[0]), float(xy[1]), z0), (0.0, 0.0, -1.0), 0.6, report)
        for dist_, col, body in sorted(hits):
            if _under(col, f"{FAC}/Items") or _under(body, f"{FAC}/Items"):
                continue
            z = z0 - dist_
            return z if _under(col, top["path"]) and abs(z - float(top["height"])) <= 0.01 else None
        return None

    def hidden_record(self):
        """The run's draws, for the record (written after the run)."""
        return self.hidden_rec

    # -- the vehicle ----------------------------------------------------------------------------------------------
    def _init_ugv(self):
        st = self.st
        root = f"{self.fleet}/UGV"
        aroot = art_root(st, root)
        self.ugv = None
        if not aroot:
            return
        gv = self.view.create_articulation_view(aroot)
        p0, q0 = tf_pq(gv.get_root_transforms()[0])
        if self.hidden:  # (prepare moved it off its nominal pose: p0, q0 are that pose, the odometry's origin)
            dx, dy, dpsi = self.hidden["ugv"]
            p0 = p0 - np.array([dx, dy, 0.0])
            q0 = qmul(quat_z(-math.radians(dpsi)), q0)
        meta = gv.shared_metatype
        dof_names = list(meta.dof_names)
        link_names = list(meta.link_names)
        lims = gv.get_dof_limits()[0]
        # the grip point: the last link of the arm (the bodies under the prim that declares arb:arm_model, in link order)
        arm_bodies = set()
        for p in Usd.PrimRange(st.GetPrimAtPath(root)):
            a = p.GetAttribute("arb:arm_model")
            if a and a.IsValid() and a.Get():
                arm_bodies |= {q.GetName() for q in Usd.PrimRange(p) if q.HasAPI(UsdPhysics.RigidBodyAPI)}
        grip_link = max((i for i, n in enumerate(link_names) if n in arm_bodies), default=len(link_names) - 1)
        self.ugv = dict(root=root, aroot=aroot, gv=gv, p0=p0, q0=q0, dofs=dof_names, links=link_names, lims=lims, grip_link=grip_link,
                        grip_link_name=link_names[grip_link], cams=[c for c in sensors_of(st, root, self.cat["sensors"]) if c.get("camera")],
                        held=None, held_rel=None, grip=False)
        self.ugv_pos_tgt = np.array(gv.get_dof_positions(), dtype=np.float32)
        self.ugv_vel_tgt = np.zeros_like(self.ugv_pos_tgt)

    def _init_items(self):
        st = self.st
        ip = st.GetPrimAtPath(f"{FAC}/Items")
        self.items = {}
        for c in (ip.GetChildren() if ip.IsValid() else []):
            if c.HasAPI(UsdPhysics.RigidBodyAPI):
                b = own_box(st, c.GetPath().pathString)
                self.items[c.GetName()] = dict(path=c.GetPath().pathString, box=b)
        self.item_names = list(self.items)
        # the names the wire uses: in the real bay, the agent's own names for its parts (matched by shape)
        alias, self.wire = self.cfg["item_alias"] or {}, {}
        used = {str(alias[n]) for n in self.item_names if n in alias}
        for n in self.item_names:
            if n in alias:
                self.wire[n] = str(alias[n])
                continue
            w = n
            while w in used:
                w += "_2"
            used.add(w)
            self.wire[n] = w
        self.iv = self.view.create_rigid_body_view([self.items[n]["path"] for n in self.item_names]) if self.item_names else None

    def item_box_world(self, name, tfs=None):
        """The item's own box carried by its pose: centre and an axis-aligned box around its corners."""
        i = self.item_names.index(name)
        tf = (tfs if tfs is not None else self.iv.get_transforms())[i]
        p, q = tf_pq(tf)
        b = self.items[name]["box"]
        if b is None:
            return p, p - 0.02, p + 0.02
        lo, hi = b
        cs = np.array([p + rot(q, (x, y, z)) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
        return p + rot(q, (lo + hi) / 2), cs.min(axis=0), cs.max(axis=0)

    def ugv_pose(self):
        tf = self.ugv["gv"].get_root_transforms()[0]
        p, q = tf_pq(tf)
        fwd = rot(q, rot(qinv(self.ugv["q0"]), (0.0, 1.0, 0.0)))  # (its forward as parked: +y)
        return p, q, math.atan2(fwd[1], fwd[0]) - math.pi / 2

    def grip_pose(self):
        tf = self.ugv["gv"].get_link_transforms()[0][self.ugv["grip_link"]]
        return tf_pq(tf)

    def apply_ugv(self, cmd):
        """The vehicle's command: joint velocity targets, joint position targets (by joint name), the grip."""
        u = self.ugv
        if u is None:
            return
        if "UGV" in self.lost:
            cmd = {"vel": {n: 0.0 for n in u["dofs"]}}
        vmax = self.cfg["ugv"]["vel_max_rad_s"]
        for name, v in ((cmd or {}).get("vel") or {}).items():
            if name in u["dofs"] and isinstance(v, (int, float)) and math.isfinite(v):
                self.ugv_vel_tgt[0, u["dofs"].index(name)] = max(-vmax, min(vmax, float(v)))
        for name, v in ((cmd or {}).get("pos") or {}).items():
            if name in u["dofs"] and isinstance(v, (int, float)) and math.isfinite(v):
                i = u["dofs"].index(name)
                lo, hi = u["lims"][i]
                lo = lo if np.isfinite(lo) else -1e3
                hi = hi if np.isfinite(hi) else 1e3
                self.ugv_pos_tgt[0, i] = max(lo, min(hi, float(v)))
        u["gv"].set_dof_velocity_targets(self.ugv_vel_tgt, self.idx)
        u["gv"].set_dof_position_targets(self.ugv_pos_tgt, self.idx)
        want = bool((cmd or {}).get("grip", u["grip"])) and "UGV" not in self.lost
        if want and u["held"] is None:  # (suction: while on, it takes the first part that comes within reach)
            gp, gq = self.grip_pose()
            tfs = self.iv.get_transforms() if self.iv is not None else None
            best = None
            for name in self.item_names:
                c, lo, hi = self.item_box_world(name, tfs)
                d = float(np.linalg.norm(np.maximum(0.0, np.maximum(lo - gp, gp - hi))))
                if d <= self.cfg["ugv"]["grip_reach_m"] and (best is None or d < best[0]):
                    best = (d, name)
            if best is not None:
                ip, iq = tf_pq(tfs[self.item_names.index(best[1])])
                u["held"] = best[1]
                u["held_rel"] = (rot(qinv(gq), ip - gp), qmul(qinv(gq), iq))
        if not want and u["held"] is not None:
            i = self.item_names.index(u["held"])
            vel = np.array(self.iv.get_velocities(), dtype=np.float32)
            vel[i, :3] = u["gv"].get_link_velocities()[0][u["grip_link"]][:3]
            vel[i, 3:] = 0.0
            self.iv.set_velocities(vel, np.array([i], dtype=np.int32))
            u["held"], u["held_rel"] = None, None
        u["grip"] = want

    def hold_item(self):
        u = self.ugv
        if u is None or u["held"] is None:
            return
        gp, gq = self.grip_pose()
        rp, rq = u["held_rel"]
        i = self.item_names.index(u["held"])
        tf = np.array(self.iv.get_transforms(), dtype=np.float32)
        tf[i] = pq_tf(gp + rot(gq, rp), qmul(gq, rq))
        idx = np.array([i], dtype=np.int32)
        self.iv.set_transforms(tf, idx)
        vel = np.array(self.iv.get_velocities(), dtype=np.float32)
        vel[i] = 0.0
        self.iv.set_velocities(vel, idx)

    # -- state ----------------------------------------------------------------------------------------------------
    def mission(self):
        b = self.cfg["bucket"]
        m = dict(dt=self.dt, time_limit=self.cfg["time_limit_s"], cruise_z=self.cfg["cruise_z"],
                 pan=[float(self.pan_nominal[0]), float(self.pan_nominal[1])], pour_r=self.cfg["pour_r"], pour_zmax=self.cfg["pour_zmax"],
                 rope_length=self.cfg["rope_length"], tilt_max_deg=self.cfg["tilt_max_deg"],
                 bucket=dict(mass_full=b["mass_empty"] + b["water_max_l"], radius=b["radius"], height=b["height"]),
                 uav={n: dict(mass=round(r["mass"], 4), rope_mass=round(r["rope_mass"], 4), thrust_max=r["thrust_max"],
                              hook=r3(r["hook"]["in_root"]), start=r3(r["p0"])) for n, r in self.robots.items()},
                 items=[self.wire[n] for n in self.item_names])
        if self.ugv:
            u = self.ugv
            m["ugv"] = dict(start=r3(u["p0"]), joints=u["dofs"], grip_link=u["grip_link_name"],
                            limits={n: [fin(u["lims"][i][0]), fin(u["lims"][i][1])] for i, n in enumerate(u["dofs"])})
        return m

    def uav_state(self, n):
        p, q = tf_pq(self.av[n].get_root_transforms()[0])
        v = np.array(self.av[n].get_root_velocities()[0][:3], dtype=float)
        return p, q, v

    def bucket_state(self):
        p, q = tf_pq(self.bv.get_transforms()[0])
        v = np.array(self.bv.get_velocities()[0][:3], dtype=float)
        up = rot(q, (0, 0, 1))
        tilt = math.degrees(math.acos(max(-1.0, min(1.0, float(up[2])))))
        return p, q, v, up, tilt

    def hooks(self):
        return {n: self.uav_state(n)[0] + rot(self.robots[n]["q0"], self.robots[n]["hook"]["in_root"]) for n in UAVS}

    def disabled(self, n):
        """Lost by the bay's rules (the scene's own logic does not come into the mission)."""
        return n in self.lost

    def pressure(self):
        r = self.cfg["rules"]
        if self.closed_t is None:
            return r["pressure_bar"]
        return r["pressure_bar"] * max(0.0, 1.0 - (self.t - self.closed_t) / r["steam_stop_s"])

    # -- sensing ----------------------------------------------------------------------------------------------
    def detect(self, cam, root_p, root_q, target, size=0.0):
        """A camera's view of a target: (in view, pixels across)."""
        if not in_view(cam, root_p, root_q, target):
            return False, 0.0
        cp = root_p + rot(root_q, cam["pos_in_root"])
        dist = float(np.linalg.norm(np.asarray(target) - cp))
        if dist > self.cfg["ugv"]["see_range_m"]:
            return False, 0.0
        res = (self.cat["sensors"].get(cam["model"]) or {}).get("res") or [0, 0]
        px = res[0] * math.degrees(2 * math.atan(size / 2 / max(dist, 1e-6))) / cam["hfov"] if size else 0.0
        return True, px

    def observe_ugv(self, nz):
        u = self.ugv
        p, q, yaw = self.ugv_pose()
        gv = u["gv"]
        v6 = gv.get_root_velocities()[0]
        dt = self.dt
        # wheel odometry drifts with the distance driven and the turning done (not while the vehicle stands still);
        # the gyro adds a slow drift of its own
        last = getattr(self, "_odo_last", None)
        self._odo_last = (p.copy(), yaw)
        if last is not None:
            ds = float(np.linalg.norm((p - last[0])[:2]))
            dpsi = abs(wrap(yaw - last[1]))
            self.drift["UGV"] = self.drift["UGV"] + np.array([self.rng.gauss(0, nz["odo_walk_m_sqrt_m"] * math.sqrt(ds)) for _ in range(2)] + [0.0])
            self.yaw_drift += self.rng.gauss(0, math.radians(nz["odo_yaw_walk_deg_sqrt_rad"]) * math.sqrt(dpsi))
        self.yaw_drift += self.rng.gauss(0, math.radians(nz["gyro_walk_deg_sqrt_s"]) * math.sqrt(dt))
        pos = p + self.drift["UGV"] + np.array([self.rng.gauss(0, nz["odo_white_m"]) for _ in range(2)] + [0.0])
        qs, qd = gv.get_dof_positions()[0], gv.get_dof_velocities()[0]
        joints = {n: [round(float(qs[i]) + self.rng.gauss(0, nz["joint_rad"]), 5), round(float(qd[i]), 4)] for i, n in enumerate(u["dofs"])}
        gp, gq = self.grip_pose()
        grip_rel = rot(qinv(q), gp - p)  # (forward kinematics: the grip point in the vehicle's own frame)
        sees = []
        targets = []
        tfs = self.iv.get_transforms() if self.iv is not None else None
        for name in self.item_names:
            if name == u["held"]:
                continue
            c, lo, hi = self.item_box_world(name, tfs)
            targets.append((f"item:{self.wire[name]}", c, 0.0))
        h = self.handle_grip()
        if h is not None:
            targets.append(("lever", h, 0.0))
        for key, label in (("crate", "crate"), ("dial", "dial")):
            b = self.bay.get(key)
            if b:
                lo, hi = np.array(b[0]), np.array(b[1])
                c = (lo + hi) / 2
                if key == "crate":
                    c = np.array([c[0], c[1], hi[2]])
                targets.append((label, c, float(max(hi - lo)) if key == "dial" else 0.0))
        targets.append(("pan", np.array([self.pan[0], self.pan[1], 0.05]), 0.0))
        bp = self.bucket_state()[0]
        targets.append(("bucket", bp, 0.0))
        gauge = None
        for what, c, size in targets:
            for cam in u["cams"]:
                ok, px = self.detect(cam, p, q, c, size)
                if not ok:
                    continue
                d = np.asarray(c) - p
                sig = nz["det_m"] + nz["det_rel"] * float(np.linalg.norm(d))
                sees.append(dict(what=what, rel=r3(d + np.array([self.rng.gauss(0, sig) for _ in range(3)]))))
                if what == "dial" and px >= 40.0 and cam["pos_in_root"] is not None:
                    gauge = round(max(0.0, self.pressure() + self.rng.gauss(0, nz["gauge_bar"])), 2)
                break
        return dict(pose=[round(float(pos[0]), 4), round(float(pos[1]), 4), round(math.degrees(wrap(yaw + self.yaw_drift)), 2)],
                    vel=r3(v6[:3]), yaw_rate=round(math.degrees(float(v6[5])), 2), joints=joints, grip_point=r3(grip_rel),
                    holding=self.wire[u["held"]] if u["held"] is not None else None, sees=sees, gauge=gauge, disabled=bool("UGV" in self.lost))

    def handle_grip(self):
        """The lever's grip: the far end of the handle from its joint (as the fleet probe takes it)."""
        if self.lever.art is None:
            return None
        hb = world_box(self.st, f"{FAC}/SteamLine/Handle")
        if hb is None:
            return None
        names = list(self.lever.art.shared_metatype.link_names)
        if "Handle" not in names:
            return (hb[0] + hb[1]) / 2
        tf = self.lever.art.get_link_transforms()[0][names.index("Handle")]
        hp, hq = tf_pq(tf)
        if not hasattr(self, "_handle_tip"):
            hub = usd_pose(self.st, f"{FAC}/SteamLine/Handle")
            c = (hb[0] + hb[1]) / 2
            tip_w = hub[0] + 2.0 * (c - hub[0]) * 0.85
            self._handle_tip = rot(qinv(hub[1]), tip_w - hub[0])
        return hp + rot(hq, self._handle_tip)

    def observe(self):
        nz = self.cfg["noise"]
        dt = self.dt
        bp, bq, bvel, bup, btilt = self.bucket_state()
        states = {n: self.uav_state(n) for n in UAVS}
        obs = dict(t=self.k, time=round(self.t, 4), mission=self.mission(), uav={})
        for n in UAVS:
            p, q, v = states[n]
            r = self.robots[n]
            self.drift[n] = self.drift[n] + np.array([self.rng.gauss(0, nz["vio_walk_m_sqrt_s"] * math.sqrt(dt)) for _ in range(3)])
            pos = p + self.drift[n] + np.array([self.rng.gauss(0, nz["vio_white_m"]) for _ in range(3)])
            vel = v + np.array([self.rng.gauss(0, nz["vel_m_s"]) for _ in range(3)])
            a = (v - self.prev_vel_uav[n]) / dt
            self.prev_vel_uav[n] = v.copy()
            load = r["mass"] * a - self.thrust[n] - r["mass"] * np.array([0, 0, -G])
            load = load + np.array([self.rng.gauss(0, nz["load_n"]) for _ in range(3)])
            neighbors = []
            lid = next((self.cfg["lidar"].get(sn["model"]) for sn in r["sensors"] if sn["type"] == "lidar" and sn["model"] in self.cfg["lidar"]), None)
            if lid:
                for m in UAVS:
                    if m == n:
                        continue
                    d = states[m][0] - p
                    dist = float(np.linalg.norm(d))
                    if dist > lid["range_m"]:
                        continue
                    if lid["band_m"] is not None and abs(d[2]) > lid["band_m"]:
                        continue
                    if lid["band_m"] is None and abs(math.degrees(math.atan2(d[2], math.hypot(d[0], d[1])))) > lid["vfov_deg"] / 2:
                        continue
                    sig = nz["lidar_m"] + nz["lidar_rel"] * dist
                    neighbors.append(r3(d + np.array([self.rng.gauss(0, sig) for _ in range(3)])))
            bucket_rel, tilt_est, pan_rel, fire = None, None, None, None
            for cam in r["sensors"]:
                if not cam.get("camera"):
                    continue
                if bucket_rel is None and in_view(cam, p, q, bp):
                    d = bp - p
                    sig = nz["cam_m"] + nz["cam_rel"] * float(np.linalg.norm(d))
                    bucket_rel = r3(d + np.array([self.rng.gauss(0, sig) for _ in range(3)]))
                    tilt_est = round(btilt + self.rng.gauss(0, nz["tilt_deg"]), 2)
                if cam["type"] == "thermal" and pan_rel is None and in_view(cam, p, q, np.array([self.pan[0], self.pan[1], 0.05])):
                    d = np.array([self.pan[0], self.pan[1], 0.05]) - p
                    sig = nz["cam_m"] + nz["cam_rel"] * float(np.linalg.norm(d))
                    pan_rel = r3(d + np.array([self.rng.gauss(0, sig) for _ in range(3)]))
                    fire = bool(self.fire)
            obs["uav"][n] = dict(pos=r3(pos), vel=r3(vel), load=r3(load), neighbors=neighbors, bucket=bucket_rel, bucket_tilt=tilt_est,
                                 pan=pan_rel, fire=fire, disabled=bool(n in self.lost))
        if self.ugv:
            obs["ugv"] = self.observe_ugv(nz)
        return obs

    # -- acting -----------------------------------------------------------------------------------------------
    def clamp(self, n, f):
        f = np.array(f, dtype=float).reshape(3) if isinstance(f, (list, tuple)) and len(f) == 3 else np.zeros(3)
        if not np.all(np.isfinite(f)):
            f = np.zeros(3)
        tmax = self.robots[n]["thrust_max"]
        fz = max(0.0, f[2])
        horiz = math.hypot(f[0], f[1])
        lim = fz * math.tan(math.radians(self.cfg["tilt_max_deg"]))
        if horiz > lim and horiz > 1e-9:
            f[0], f[1] = f[0] * lim / horiz, f[1] * lim / horiz
        f[2] = fz
        mag = float(np.linalg.norm(f))
        if mag > tmax and mag > 1e-9:
            f = f * tmax / mag
        return f

    def wind_force(self, area):
        w = self.cfg["wind"]
        if self.wind_t0 <= self.t < self.wind_t0 + w["duration_s"]:
            return self.wind_dir * (0.5 * 1.2 * 1.0 * area * w["speed_mps"] ** 2)
        return np.zeros(3)

    def physics_step(self):
        """One physics step: the thrusts (and the wind) on the airframes, the wind on the bucket, attitude held level,
        the held part carried; the contacts read."""
        for n in UAVS:
            av = self.av[n]
            f = np.zeros((1, av.max_links, 3), dtype=np.float32)
            off = n in self.lost
            f[0, self.robots[n]["root_link"]] = (np.zeros(3) if off else self.thrust[n]) + self.wind_force(self.cfg["wind"]["area_uav"])
            av.apply_forces_and_torques_at_position(f, None, None, self.idx, True)
        wb = self.wind_force(self.cfg["wind"]["area_bucket"])
        if np.any(wb):
            self.bv.apply_forces_and_torques_at_position(wb.reshape(1, 3).astype(np.float32), None, None, self.idx, True)
        self.lever.hold(touched=self.t - self.lever_touch_t < 0.5)
        self.hold_item()
        self.s.step()
        self.t += self.cfg["dt_physics"]
        for n in UAVS:
            av = self.av[n]
            tf = av.get_root_transforms()[0]
            av.set_root_transforms(pq_tf(tf[:3], self.robots[n]["q0"])[None], self.idx)
            vel = np.array(av.get_root_velocities()[0], dtype=np.float32)
            vel[3:] = 0.0
            av.set_root_velocities(vel[None], self.idx)
        self.read_contacts()

    def read_contacts(self):
        """Every contact of a robot's body with something it must not touch (another robot or its rope, the bay's
        furniture and fixtures, the bucket), counted once per onset (a contact gone for 0.5 s starts anew)."""
        try:
            from pxr import PhysicsSchemaTools
            hdrs, _ = self.phx.get_contact_report()
        except Exception:  # noqa: BLE001
            return
        fleet = self.fleet
        for h in hdrs:
            a = [str(PhysicsSchemaTools.intToSdfPath(h.actor0)), str(PhysicsSchemaTools.intToSdfPath(h.actor1))]
            c = [str(PhysicsSchemaTools.intToSdfPath(h.collider0)), str(PhysicsSchemaTools.intToSdfPath(h.collider1))]
            for side in (0, 1):  # (a robot's root prim may be a body itself: "under" includes the root)
                me = next((r for r in ROBOTS if _under(a[side], f"{fleet}/{r}")), None)
                if me is None or any(_under(a[side], rr) for rr in self.rope_roots.values()):
                    continue
                other = c[1 - side] or a[1 - side]
                if _under(other, f"{FAC}/SteamLine/Handle") or _under(a[1 - side], f"{FAC}/SteamLine/Handle"):
                    self.lever_touch_t = self.t
                if _under(other, f"{fleet}/{me}") or _under(other, self.rope_roots.get(me, "//")):
                    continue
                owner = next((r for r, rr in self.rope_roots.items() if _under(other, rr)), None)
                if owner is not None:  # (another robot's rope)
                    what = owner
                elif other.startswith(fleet + "/"):
                    what = other[len(fleet) + 1:].split("/")[0]
                elif other.startswith(FAC + "/") and not any(other == f or other.startswith(f + "/") for f in FREE):
                    what = other.split("/")[3] if len(other.split("/")) > 3 else other
                    if what in ("Tables", "Pads") and len(other.split("/")) > 4:
                        what = other.split("/")[4]
                else:
                    continue
                key = (me, what)
                last = self.contacts_seen.get(key)
                if last is None or self.t - last > 0.5:
                    if len(self.metrics["collisions"]) < 50:
                        self.metrics["collisions"].append(dict(t=round(self.t, 2), robot=me, with_=what, body=a[side].split("/")[-1]))
                        self.metrics["collisions"][-1]["with"] = self.metrics["collisions"][-1].pop("with_")
                self.contacts_seen[key] = self.t

    def water_step(self):
        b = self.cfg["bucket"]
        dt = self.dt
        bp, bq, bvel, bup, btilt = self.bucket_state()
        a = (bvel - self.prev_vel_bucket) / dt
        self.prev_vel_bucket = bvel.copy()
        self.acc_bucket = 0.7 * self.acc_bucket + 0.3 * a
        g_eff = np.array([0, 0, -G]) - self.acc_bucket
        water_up = -g_eff / (np.linalg.norm(g_eff) + 1e-9)
        theta = math.degrees(math.acos(max(-1.0, min(1.0, float(np.dot(bup, water_up))))))
        freeboard = b["height"] - self.water / (math.pi * b["radius"] ** 2 * 1000.0)
        if theta >= 89.0:
            rate = 20.0
        else:
            rate = b["spill_rate_l_per_s_m"] * max(0.0, b["radius"] * math.tan(math.radians(theta)) - freeboard)
        lost = min(self.water, rate * dt)
        if lost > 0:
            self.water -= lost
            over_pan = (math.hypot(bp[0] - self.pan[0], bp[1] - self.pan[1]) <= self.cfg["pour_r"] and bp[2] <= self.cfg["pour_zmax"])
            if over_pan:
                self.poured += lost
                if self.metrics["pour_start_t"] is None:
                    self.metrics["pour_start_t"] = round(self.t, 3)
            else:
                self.spilt += lost
            self.bv.set_masses(np.array([b["mass_empty"] + self.water], dtype=np.float32), self.idx)
        return theta

    def rules(self, states):
        """The bay's rules as the grader keeps them: the lever closed releases the pressure (gone 3 s later, back at
        once if it is opened); water poured on the pan puts the fire out; a flying robot in the live plume, the vehicle
        near the burning pan, is lost (its commands are cut from then on)."""
        r, m = self.cfg["rules"], self.metrics
        ang = abs(self.lever.angle())
        if ang >= r["lever_closed_deg"]:
            if self.closed_t is None:
                self.closed_t = self.t
                m["lever_closed_t"] = m["lever_closed_t"] or round(self.t, 2)
        elif ang < r["lever_closed_deg"] - 10.0 and self.closed_t is not None:
            self.closed_t = None
        self.steam = self.pressure() > 0.0
        if not self.steam and m["steam_off_t"] is None:
            m["steam_off_t"] = round(self.t, 2)
        if self.steam and m["steam_off_t"] is not None:
            m["steam_off_t"] = None
        if self.fire and self.poured >= r["fire_out_l"]:
            self.fire = False
            m["fire_out_t"] = round(self.t, 2)
        plume = self.bay.get("plume")
        for n in UAVS:
            p = states[n][0]
            inside = plume is not None and all(plume[0][k] <= p[k] <= plume[1][k] for k in range(3))
            self.exposure[n] = self.exposure[n] + self.dt if (inside and self.steam) else 0.0
            if self.exposure[n] >= r["plume_s"] and n not in self.lost:
                self.lost[n] = dict(t=round(self.t, 2), why="in the live steam plume")
        if self.ugv:
            p = self.ugv_pose()[0]
            near = math.hypot(p[0] - self.pan[0], p[1] - self.pan[1]) <= r["fire_r"]
            self.exposure["UGV"] = self.exposure["UGV"] + self.dt if (near and self.fire) else 0.0
            if self.exposure["UGV"] >= r["fire_s"] and "UGV" not in self.lost:
                self.lost["UGV"] = dict(t=round(self.t, 2), why="near the burning pan")

    def parts(self):
        """Where the parts are: in the crate, on a table, or elsewhere (dropped)."""
        out, n_in, n_drop = {}, 0, 0
        crate = self.bay.get("crate")
        tables = [self.bay.get(k) for k in ("table_a", "table_b") if self.bay.get(k)]
        tfs = self.iv.get_transforms() if self.iv is not None else None
        for name in self.item_names:
            c = self.item_box_world(name, tfs)[0]
            w = self.wire[name]
            if not np.all(np.isfinite(c)):
                out[w] = "lost"
                n_drop += 1
                continue
            if self.ugv and self.ugv["held"] == name:
                out[w] = "held"
                continue
            if crate and crate[0][0] <= c[0] <= crate[1][0] and crate[0][1] <= c[1] <= crate[1][1] and c[2] <= crate[1][2] + 0.10:
                out[w] = "crate"
                n_in += 1
            elif any(t[0][0] <= c[0] <= t[1][0] and t[0][1] <= c[1] <= t[1][1] and c[2] >= t[1][2] - 0.02 for t in tables):
                out[w] = "table"
            else:
                out[w] = "dropped"
                n_drop += 1
        return out, n_in, n_drop

    def measure(self):
        m = self.metrics
        bp, bq, bvel, bup, btilt = self.bucket_state()
        lo = bp[2]
        hooks = self.hooks()
        cen = np.mean(list(hooks.values()), axis=0)
        states = {n: self.uav_state(n) for n in UAVS}
        self.rules(states)
        if m["liftoff_t"] is None and lo > 0.08 and all(float(np.linalg.norm(hooks[n] - (bp + rot(bq, (0, 0, self.cfg["bucket"]["height"]))))) < self.cfg["rope_length"] + 0.2 for n in UAVS):
            m["liftoff_t"] = round(self.t, 3)
            m["liftoff_tilt_max"] = 0.0
        if m["liftoff_t"] is not None:
            if self.t <= m["liftoff_t"] + 1.5:
                m["liftoff_tilt_max"] = max(m["liftoff_tilt_max"], btilt)
            in_pan_zone = math.hypot(bp[0] - self.pan[0], bp[1] - self.pan[1]) <= self.cfg["pour_r"]
            if self.t > m["liftoff_t"] + 2.0 and not in_pan_zone and m["pour_start_t"] is None:
                m["swing_max"] = max(m["swing_max"], float(np.linalg.norm((bp - cen)[:2])))
            if lo < 0.03 and m["pour_start_t"] is None and self.t > m["liftoff_t"] + 1.0:
                m["dropped"] = True
        for i, a in enumerate(UAVS):
            for b in UAVS[i + 1:]:
                m["min_sep"] = min(m["min_sep"], float(np.linalg.norm(states[a][0] - states[b][0])))
        for n in UAVS:
            p = states[n][0]
            far_from_pad = float(np.linalg.norm((p - self.robots[n]["p0"])[:2])) > 0.6
            if p[2] < self.cfg["crash_z"] and far_from_pad and n not in m["crash"]:
                m["crash"].append(n)
        return bp, btilt, states

    def still(self, states, btilt, bp):
        uav_still = all(float(np.linalg.norm(states[n][2])) < 0.3 for n in UAVS)
        bucket_ok = btilt < 8.0
        ugv_ok = True
        if self.ugv:
            ugv_ok = float(np.linalg.norm(self.ugv["gv"].get_root_velocities()[0][:3])) < self.cfg["ugv"]["still_mps"]
        return uav_still and bucket_ok and ugv_ok

    # -- the episode ----------------------------------------------------------------------------------------------
    def run(self, policy):
        """Run the mission with `policy(t, obs) -> action` (None: the controller aborted). Returns the measurements."""
        m = self.metrics
        steady_since = None
        while self.t < self.cfg["time_limit_s"]:
            obs = self.observe()
            action = policy(self.k, obs)
            if action is None:
                m["abort"] = m.get("abort") or "controller aborted"
                break
            cmd = (action.get("thrust") if isinstance(action, dict) else None) or {}
            for n in UAVS:
                self.thrust[n] = self.clamp(n, cmd.get(n, [0, 0, 0]))
            self.apply_ugv((action.get("ugv") if isinstance(action, dict) else None) or {})
            for _ in range(self.cfg["substeps"]):
                self.physics_step()
            self.water_step()
            bp, btilt, states = self.measure()
            self.k += 1
            m["steps"] = self.k
            if self.k % 300 == 0:
                where, n_in, n_drop = self.parts()
                self.log(f"t={self.t:6.1f} steam={self.steam} fire={self.fire} parts={n_in}/{len(self.item_names)} dropped={n_drop} "
                         f"bucket z={bp[2]:.2f} tilt={btilt:4.1f} water={self.water:.2f} spilt={self.spilt:.2f} poured={self.poured:.2f} "
                         f"sep={m['min_sep']:.2f} lost={list(self.lost)} hits={len(m['collisions'])}")
            where, n_in, n_drop = self.parts() if self.k % 30 == 0 else (None, None, None)
            tasks_done = (not self.steam) and (not self.fire) and (n_in is not None and n_in == len(self.item_names))
            if n_in is not None:
                if tasks_done and self.still(states, btilt, bp):
                    steady_since = self.t if steady_since is None else steady_since
                    if self.t - steady_since >= self.cfg["done_hold_s"]:
                        m["done_t"] = round(steady_since, 2)
                        m["end"] = "done"
                        break
                else:
                    steady_since = None
            if len(self.lost) == len(ROBOTS):
                m["end"] = "lost"
                break
        if m["end"] is None:
            m["end"] = "aborted" if m["abort"] else "time"
        bp, bq, bvel, bup, btilt = self.bucket_state()
        where, n_in, n_drop = self.parts()
        m["lost"] = self.lost
        m.update(water0_l=round(self.water0, 3), water_l=round(self.water, 3), spilt_l=round(self.spilt, 3), poured_l=round(self.poured, 3),
                 t_end=round(self.t, 3), end_tilt=round(btilt, 2), end_bucket_z=round(float(bp[2]), 3),
                 end_speed=round(max(float(np.linalg.norm(self.uav_state(n)[2])) for n in UAVS), 3),
                 wind_t0=round(self.wind_t0, 2), min_sep=round(m["min_sep"], 3), swing_max=round(m["swing_max"], 3),
                 steam_end=bool(self.steam), fire_end=bool(self.fire), lever_deg=round(abs(self.lever.angle()), 1),
                 parts=where, parts_in_crate=n_in, parts_dropped=n_drop, n_parts=len(self.item_names),
                 ugv_end=r3(self.ugv_pose()[0]) if self.ugv else None, n_collisions=len(m["collisions"]))
        return m


# --------------------------------------------------------------------------------------------- a local solver (the practice runner)
class LocalSolver:
    """The agent's `run.sh` spoken to over the same wire as the grader's, in the agent's own session (no account change,
    no probe lock). For practice only."""

    def __init__(self, submission, step_timeout=30.0):
        import os
        import queue
        import subprocess
        import threading
        self.proc = subprocess.Popen(["bash", "run.sh"], cwd=submission, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=None, universal_newlines=True, bufsize=1, start_new_session=True)
        self.q = queue.Queue()
        self.timeout = step_timeout
        self._queue_mod = queue
        self._os = os

        def pump():
            for line in self.proc.stdout:
                self.q.put(line)
            self.q.put(None)

        threading.Thread(target=pump, daemon=True).start()
        for _ in range(200):
            line = self.q.get(timeout=step_timeout * 4)
            if line is None:
                raise RuntimeError("the controller exited before READY")
            if line.strip() == "READY":
                break
        else:
            raise RuntimeError("the controller never printed READY")
        self.request({"type": "reset"})

    def request(self, msg):
        self.proc.stdin.write(json.dumps(msg) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.q.get(timeout=self.timeout)
            if line is None:
                raise RuntimeError("the controller closed its stdout")
            line = line.strip()
            if line:
                return json.loads(line)

    def act(self, t, obs):
        return self.request({"type": "act", "t": t, "obs": obs}).get("action")

    def close(self):
        import signal
        try:
            self.request({"type": "close"})
        except Exception:  # noqa: BLE001
            pass
        try:
            self._os.killpg(self.proc.pid, signal.SIGKILL)
        except Exception:  # noqa: BLE001
            pass
