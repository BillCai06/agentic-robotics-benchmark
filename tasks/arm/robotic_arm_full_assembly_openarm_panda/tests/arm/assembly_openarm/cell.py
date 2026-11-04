"""The OpenArm assembly cell: the arm's base (pedestal, J1 bracket, J1 motor) on the bench, the seven loose units of the
kit resting on the bench where the Panda reaches them, the Panda, the coupling state machine (`coupling.py`).

`OpenArmCell(kit_dir, panda_dir, layout, ...)` compiles from the kit (`kit.json`, meshes in body frames):
  * the base: a fixed body at the fixture pose (the pedestal's base plate on the bench, the arm frame 0.697 m above it,
    turned by the fixture yaw), with J1's rotor on an unpowered hinge;
  * every loose unit a free body with its children (the motors' rotors on unpowered hinges, the wrist's J6 hinge);
    the hinges back-drive against the motor's friction (kit "hinge"), the J6 hinge within its range, a rotor's range
    set when its flange fastens;
  * one weld per mate (inactive until the coupling captures / fastens), contact exclusions only between the two bodies
    of a mate (the bolted interface);
  * the Panda (the information tier's model, position actuators, tcp, flange F/T, finger pads, wrist camera).
`targets()` / `live_target(mid)` give a mate's unit pose for the design / for the carrier as it is; `tick()` advances
one 10 ms control tick (controller, physics, coupling) and records a frame every 4 ticks.
"""
import copy
import json
import math
import os
import time

import numpy as np

from .coupling import CouplingStateMachine, _inv, body_pose, quat_wxyz

CELL_CONFIG = {
    "timestep": 0.001, "tick_steps": 10, "impratio": 10.0, "noslip_iterations": 10, "arena_memory_bytes": 512 << 20,
    "bench": {"size_xy": [1.6, 1.0], "thickness": 0.03, "floor_z": -0.994, "friction": 0.8},
    "geom_defaults": {"condim": 4, "friction": [0.8, 0.005, 0.0001], "solref": [0.004, 1.0], "solimp": [0.95, 0.99, 0.001, 0.5, 2.0]},
    # the finger pads: rubber, a 17 x 17 mm patch: torsional friction ~ mu x the patch's radius
    "pad_friction": [1.0, 0.01, 0.0001],
    "weld_solref": [0.002, 1.0], "weld_solimp": [0.99, 0.999, 0.001, 0.5, 2.0],
    # the gripper's actuator drives the fingers' tendon (half the width): each finger presses with half its force;
    # 140 N on the tendon = 70 N per finger, the Franka Hand's continuous grasping force
    "gripper": {"kp": 2500.0, "kv": 50.0, "max_force_N": 140.0, "max_width_m": 0.08},
    "panda_pos": [-0.45, 0.0, 0.0], "panda_gravcomp": 1.0, "tcp_offset_m": 0.1034,
    "wrist_camera": {"pos": [0.0, -0.06, 0.03], "fovy": 70},
    "materials": {"alu": [0.74, 0.75, 0.77, 1.0], "steel": [0.55, 0.56, 0.58, 1.0], "dark": [0.13, 0.13, 0.14, 1.0],
                  "rotor": [0.62, 0.63, 0.66, 1.0], "mark": [0.95, 0.80, 0.10, 1.0]},
    "fixture": {"xy": [0.10, 0.0], "yaw_deg": 120.0},
    # a riser: two aluminium blocks on a common frame (a bridge) under a unit's two ends
    "riser": {"block_m": [0.010, 0.04, 0.035], "mass_kg": 0.04, "rgba": [0.55, 0.57, 0.60, 1.0]},
}
PANDA_HOME = [0.0, -0.4, 0.0, -2.3, 0.0, 1.9, 0.785]


def panda_list(cfg):
    """the cell's Pandas: cfg["pandas"] = [{name, pos, yaw_deg, arm}] (several: each with its own prefix and sites), else the
    one Panda at cfg["panda_pos"] with the information tier's names"""
    ps = cfg.get("pandas")
    if not ps:
        return [{"name": "panda", "pos": list(cfg["panda_pos"]), "yaw_deg": 0.0, "arm": None, "tcp": "tcp", "ft_site": "hand_ft",
                 "wrist": "wrist", "force": "hand_force", "torque": "hand_torque"}]
    out = []
    for p in ps:
        n = p["name"]
        out.append(dict({"yaw_deg": 0.0, "arm": None}, **p, tcp=n + "/tcp", ft_site=n + "/hand_ft", wrist=n + "/wrist",
                        force=n + "/hand_force", torque=n + "/hand_torque"))
    return out


def load_kit(kit_dir):
    kit = json.load(open(os.path.join(kit_dir, "kit.json")))
    kit["dir"] = kit_dir
    return kit


def rotz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def T_world_arm(fixture):
    """the arm frame in the world: the pedestal's base plate on the bench at fixture xy, turned by the yaw"""
    T = np.eye(4)
    T[:3, :3] = rotz(math.radians(float(fixture.get("yaw_deg", 0.0))))
    T[:3, 3] = [float(fixture["xy"][0]), float(fixture["xy"][1]), 0.697 + 0.001]
    return T


def design_world(kit, fixture):
    """{body: world pose at the design zero} for the given fixture"""
    Ta = T_world_arm(fixture)
    return {b: Ta @ np.array(bb["T_design_arm"], float) for b, bb in kit["bodies"].items()}


def build_spec(kit, panda_dir, layout, cfg):
    import mujoco
    spec = mujoco.MjSpec()
    spec.modelname = "openarm_assembly_cell"
    spec.memory = int(cfg["arena_memory_bytes"])
    spec.compiler.degree = False
    spec.compiler.autolimits = True
    spec.option.timestep = float(cfg["timestep"])
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.noslip_iterations = int(cfg["noslip_iterations"])
    spec.option.impratio = float(cfg["impratio"])
    spec.option.gravity = [0.0, 0.0, -9.81]
    spec.option.disableflags |= mujoco.mjtDisableBit.mjDSBL_MULTICCD
    gd = spec.default.geom
    gd.condim = int(cfg["geom_defaults"]["condim"])
    fr = list(cfg["geom_defaults"]["friction"]); fr[0] = float(cfg["bench"]["friction"])
    gd.friction = fr
    gd.solref = cfg["geom_defaults"]["solref"]
    gd.solimp = cfg["geom_defaults"]["solimp"]
    spec.visual.global_.offwidth, spec.visual.global_.offheight = 1920, 1080
    spec.visual.headlight.diffuse = [0.6, 0.6, 0.6]
    spec.visual.headlight.ambient = [0.3, 0.3, 0.3]
    spec.stat.extent, spec.stat.center = 1.5, [0.0, 0.0, 0.35]
    w = spec.worldbody
    w.add_light(name="sun", pos=[0.3, -0.3, 2.5], dir=[-0.15, 0.15, -1.0], type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL, diffuse=[0.7, 0.7, 0.7])
    w.add_light(name="fill", pos=[-0.8, 0.8, 1.6], dir=[0.4, -0.6, -0.8], diffuse=[0.35, 0.35, 0.35])
    for name, rgba in cfg["materials"].items():
        spec.add_material(name="mat_" + name, rgba=list(rgba))
    spec.add_material(name="mat_bench", rgba=[0.80, 0.74, 0.62, 1.0])
    spec.add_material(name="mat_riser", rgba=list(cfg["riser"]["rgba"]))
    b = cfg["bench"]
    w.add_geom(name="table_top", type=mujoco.mjtGeom.mjGEOM_BOX, size=[b["size_xy"][0] / 2, b["size_xy"][1] / 2, b["thickness"] / 2],
               pos=[0.0, 0.0, -b["thickness"] / 2], material="mat_bench")
    w.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4.0, 4.0, 0.1], pos=[0.0, 0.0, b["floor_z"]], rgba=[0.3, 0.3, 0.3, 1.0])
    kd = kit["dir"]
    bodies = kit["bodies"]
    children = {}
    for bn, bb in bodies.items():
        if bb.get("parent"):
            children.setdefault(bb["parent"], []).append(bn)

    def add_geoms(body, bn):
        bb = bodies[bn]
        for mat, f in bb.get("visual", {}).items():
            spec.add_mesh(name="vis_" + bn + "_" + mat, file=os.path.join(kd, f))
            body.add_geom(name="vis_%s_%s" % (bn, mat), type=mujoco.mjtGeom.mjGEOM_MESH, meshname="vis_" + bn + "_" + mat, contype=0,
                          conaffinity=0, group=2, material="mat_" + mat, mass=0.0)
        for i, f in enumerate(bb.get("collision", [])):
            spec.add_mesh(name="col_%s_%03d" % (bn, i), file=os.path.join(kd, f))
            body.add_geom(name="col_%s_%03d" % (bn, i), type=mujoco.mjtGeom.mjGEOM_MESH, meshname="col_%s_%03d" % (bn, i), group=3,
                          rgba=[0.9, 0.3, 0.3, 0.3], mass=0.0)
        disc = bb.get("disc")
        if disc:
            n = np.asarray(disc["normal"], float)
            x = np.cross([0, 0, 1.0], n) if abs(n[2]) < 0.9 else np.cross(n, [1.0, 0, 0])
            x /= np.linalg.norm(x); y = np.cross(n, x)
            R = np.column_stack([x, y, n])
            c = np.asarray(disc["centre_m"], float)
            q = quat_wxyz(R)
            r, th = float(disc["radius_m"]), float(disc["thickness_m"])
            # the flange disc is drawn only: it lies 0.3 mm inside the motor case's face, which collides
            body.add_geom(name="vis_%s_disc" % bn, type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, th / 2, 0], pos=c.tolist(), quat=q.tolist(),
                          contype=0, conaffinity=0, group=2, material="mat_rotor", mass=0.0)
            # the index mark: a yellow tick on the flange's rim, at mark_deg about the axis from the flange frame's x
            mk = math.radians(float(bb.get("mark_deg", 0.0)))
            u = math.cos(mk) * x + math.sin(mk) * y
            p = c + u * (0.78 * r) + n * (th / 2 + 0.0004)
            Rm = np.column_stack([u, np.cross(n, u), n])
            body.add_geom(name="vis_%s_mark" % bn, type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2 * r, 0.0025, 0.0005], pos=p.tolist(),
                          quat=quat_wxyz(Rm).tolist(), contype=0, conaffinity=0, group=2, material="mat_mark", mass=0.0)
        if bb.get("mass_kg", 0.0) > 0 and bn != "base":
            body.mass = float(bb["mass_kg"])
            body.ipos = list(bb["com_m"])
            body.iquat = list(bb["inertia_quat"])
            body.inertia = [max(float(v), 1e-7) for v in bb["inertia_diag"]]
            body.explicitinertial = True

    def add_children(parent_body, pn):
        for cn in children.get(pn, []):
            cb = bodies[cn]
            off = np.array(cb["T_design_arm"])[:3, 3] - np.array(bodies[pn]["T_design_arm"])[:3, 3]
            body = parent_body.add_body(name=cn, pos=off.tolist())
            js = cb["joint"]
            if js is None:
                # a fixed part of the unit (a motor's case, a bearing): its own body only for the contact exclusions
                add_geoms(body, cn)
                add_children(body, cn)
                continue
            jk = kit["joints"].get(js["name"]) or kit["bearings"][js["name"]]
            hp = np.asarray(js["anchor_arm"], float) - np.array(cb["T_design_arm"])[:3, 3]
            jnt = body.add_joint(name="hinge_" + cn, type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(js["axis"]), pos=hp.tolist(),
                                 frictionloss=float(jk["hinge"]["frictionloss"]), damping=float(jk["hinge"]["damping"]),
                                 armature=float(jk["hinge"]["armature"]))
            if cb.get("rotor_of") is None and jk.get("range") is None:
                # a plate on a bearing (J4_D in J4_B): free, its travel stopped by contact
                jnt.limited = mujoco.mjtLimited.mjLIMITED_FALSE
            elif cb.get("rotor_of") is None:
                # a hinge inside a module (J6 of the wrist): its design range
                jnt.range = list(jk["range"])
                jnt.limited = mujoco.mjtLimited.mjLIMITED_TRUE
            else:
                jnt.limited = mujoco.mjtLimited.mjLIMITED_FALSE
            add_geoms(body, cn)
            add_children(body, cn)

    Ta = T_world_arm(layout["fixture"])
    Tb = Ta @ np.array(bodies["base"]["T_design_arm"])
    base = w.add_body(name="base", pos=Tb[:3, 3].tolist(), quat=quat_wxyz(Tb[:3, :3]).tolist())
    add_geoms(base, "base")
    add_children(base, "base")
    for u in kit["loose"]:
        pos, quat = layout["parts"][u]
        body = w.add_body(name=u, pos=list(pos), quat=list(quat))
        body.add_freejoint(name="free_" + u)
        add_geoms(body, u)
        add_children(body, u)
    # risers: a unit lying flat on the bench leaves the fingers no room under its edges (the cable holders rest on
    # their rims): it rests on two loose blocks under its ends, `span` apart along the riser's x (a loose pair, one body)
    R = cfg["riser"]
    for name, v in (layout.get("risers") or {}).items():
        pos, quat, span = v[0], v[1], (v[2] if len(v) > 2 else 0.07)
        body = w.add_body(name=name, pos=list(pos), quat=list(quat))
        body.add_freejoint(name="free_" + name)
        for i, sx in enumerate((-0.5, 0.5)):
            body.add_geom(name="col_%s_%d" % (name, i), type=mujoco.mjtGeom.mjGEOM_BOX, size=[x / 2 for x in R["block_m"]],
                          pos=[sx * span, 0.0, 0.0], material="mat_riser", mass=R["mass_kg"])

    # the Pandas (the information tier's: position actuators, tcp, flange F/T, finger pads, its own gripper actuator);
    # one Panda keeps the information tier's names (panda/..., tcp, wrist, hand_force); several get their own prefixes
    pads_all = {}
    for pc in panda_list(cfg):
        P = pc["name"] + "/"
        panda = mujoco.MjSpec.from_file(os.path.join(panda_dir, "panda.xml"))
        for k in list(panda.keys):
            panda.delete(k)
        frame = w.add_frame(pos=list(pc["pos"]), quat=quat_wxyz(rotz(math.radians(float(pc.get("yaw_deg", 0.0))))).tolist())
        frame.attach_body(panda.worldbody.first_body(), P, "")
        for bd in spec.bodies:
            if bd.name.startswith(P) and bd.name != P + "link0":
                bd.gravcomp = float(cfg["panda_gravcomp"])
        hand = spec.body(P + "hand")
        hand.add_site(name=pc["tcp"], pos=[0.0, 0.0, cfg["tcp_offset_m"]], size=[0.004, 0.004, 0.004], group=4, rgba=[0.2, 0.9, 0.2, 0.6])
        hand.add_site(name=pc["ft_site"], pos=[0.0, 0.0, 0.0], size=[0.003, 0.003, 0.003], group=4)
        wc = cfg["wrist_camera"]
        hand.add_camera(name=pc["wrist"], pos=wc["pos"], quat=quat_wxyz(np.column_stack([[1.0, 0.0, 0.0], [0.0, 0.55, 0.835], np.cross([1.0, 0.0, 0.0], [0.0, 0.55, 0.835])])).tolist(),
                        fovy=wc["fovy"])
        spec.add_sensor(name=pc["force"], type=mujoco.mjtSensor.mjSENS_FORCE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=pc["ft_site"])
        spec.add_sensor(name=pc["torque"], type=mujoco.mjtSensor.mjSENS_TORQUE, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=pc["ft_site"])
        pads = []
        for g in spec.geoms:
            pb = g.parent
            if pb is not None and pb.name in (P + "left_finger", P + "right_finger") and g.type == mujoco.mjtGeom.mjGEOM_BOX:
                side = "left" if "left" in pb.name else "right"
                g.name = P + "pad_%s_%d" % (side, len([q for q in pads if side in q]))
                g.friction = cfg["pad_friction"]
                pads.append(g.name)
        pads_all[pc["name"]] = pads
        gr = cfg["gripper"]
        for act in spec.actuators:
            if act.name == P + "actuator8":
                act.name = P + "gripper"
                act.gainprm = [gr["kp"], 0, 0, 0, 0, 0, 0, 0, 0, 0]
                act.biasprm = [0, -gr["kp"], -gr["kv"], 0, 0, 0, 0, 0, 0, 0]
                act.ctrlrange = [0.0, gr["max_width_m"] / 2]
                act.forcerange = [-gr["max_force_N"], gr["max_force_N"]]
    pad_geoms = [g for v in pads_all.values() for g in v]

    # the welds of the mates and the contact exclusions at the bolted interfaces
    for mm in kit["mates"]:
        eq = spec.add_equality(name="weld_" + mm["id"], type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                               name1=mm["carrier"], name2=mm["part"])
        eq.active = False
        eq.solref = list(cfg["weld_solref"])
        eq.solimp = list(cfg["weld_solimp"])
        if mm["kind"] == "flange":
            # the capture of a flange: two soft point constraints on the axis (the face's centre and 20 mm along the
            # axis into the part): the part held on the axis and the face, its rotation about the axis free
            Fc, Fp = np.array(mm["frame_in_carrier"], float), np.array(mm["frame_in_part"], float)
            for i, h in enumerate((0.0, 0.02)):
                pc = (Fc @ np.array([0, 0, h, 1.0]))[:3]
                pp = (Fp @ np.array([0, 0, h, 1.0]))[:3]
                eq = spec.add_equality(name="cap_%s_%d" % (mm["id"], i), type=mujoco.mjtEq.mjEQ_CONNECT, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                       name1=mm["carrier"], name2=mm["part"])
                ed = np.zeros(len(eq.data))           # (11 values in MuJoCo 3.14, 12 in 3.17)
                ed[:6] = list(pc) + list(pp)
                ed[-1] = 1.0
                eq.data = ed
                eq.active = False
                eq.solref = [0.04, 1.0]
                eq.solimp = list(cfg["weld_solimp"])
        for cw in mm.get("co_welds", []):
            eq = spec.add_equality(name="weld_%s__%s" % (mm["id"], cw["part"]), type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                                   name1=cw["carrier"], name2=cw["part"])
            eq.active = False
            eq.solref = list(cfg["weld_solref"])
            eq.solimp = list(cfg["weld_solimp"])
        if mm.get("exclude_pair", True):
            ex = spec.add_exclude()
            ex.bodyname1, ex.bodyname2 = mm["carrier"], mm["part"]
    for b1, b2 in kit.get("extra_excludes", []):
        ex = spec.add_exclude()
        ex.bodyname1, ex.bodyname2 = b1, b2
    return spec, pad_geoms


class OpenArmCell(object):
    def __init__(self, kit_dir, panda_dir, layout, cache_dir=None, config=None, log=print, panda_q=None, settle_s=1.5, use_cache=True,
                 spec_hook=None):
        import mujoco
        from ..assembly_cell.controller import PandaController
        self.mj, self.np, self.log = mujoco, np, log
        self.kit = load_kit(kit_dir)
        self.cfg = copy.deepcopy(CELL_CONFIG)
        for k, v in (config or {}).items():
            if isinstance(v, dict) and isinstance(self.cfg.get(k), dict):
                self.cfg[k].update(v)
            else:
                self.cfg[k] = v
        self.layout = layout
        t0 = time.time()
        spec, pads = build_spec(self.kit, panda_dir, layout, self.cfg)
        if spec_hook is not None:
            spec_hook(spec)            # e.g. the floating check's jigs (dev only)
        m = spec.compile()
        self.spec, self.m = spec, m
        d = mujoco.MjData(m)
        self.d = d
        self.loose = list(self.kit["loose"])
        self.meta = {"loose": self.loose, "pad_geoms": pads, "root": "base", "servos": [], "config": {"tick_steps": self.cfg["tick_steps"]}}
        self.tick_steps = int(self.cfg["tick_steps"])
        self.tick_s = m.opt.timestep * self.tick_steps
        self.pandas = panda_list(self.cfg)
        q0 = list(panda_q or PANDA_HOME)
        for pc in self.pandas:
            for i in range(7):
                d.qpos[m.jnt_qposadr[m.joint(pc["name"] + "/joint%d" % (i + 1)).id]] = q0[i]
            for fn in ("finger_joint1", "finger_joint2"):
                d.qpos[m.jnt_qposadr[m.joint(pc["name"] + "/" + fn).id]] = self.cfg["gripper"]["max_width_m"] / 2
        self.pj = [m.joint(self.pandas[0]["name"] + "/joint%d" % (i + 1)).id for i in range(7)]
        # the rotors at the episode's angles (an unpowered motor's flange rests wherever it was turned)
        for rb, ang in (layout.get("rotor_rad") or {}).items():
            d.qpos[m.jnt_qposadr[m.joint("hinge_" + rb).id]] = float(ang)
        mujoco.mj_forward(m, d)
        self.ctrls = {}
        for pc in self.pandas:
            ct = PandaController(m, d, tick_s=self.tick_s, prefix=pc["name"] + "/", tcp_site=pc["tcp"])
            ct.q_cmd = np.array(q0, float)
            ct.q_goal = ct.q_cmd.copy()
            ct.width_cmd = ct.width_goal = ct.width()
            ct.write_ctrl()
            self.ctrls[pc["name"]] = ct
        self.ctrl = self.ctrls[self.pandas[0]["name"]]
        self.sm = CouplingStateMachine(m, d, self.kit, insertion={mm["id"]: mm.get("insertion_in_carrier") for mm in self.kit["mates"]})
        self.mates = {mm["id"]: mm for mm in self.kit["mates"]}
        self.geoms_of = {u: [] for u in self.loose}
        unit_of = {bn: bb["unit"] for bn, bb in self.kit["bodies"].items()}
        self.unit_of_body = {}
        for b in range(m.nbody):
            nm = m.body(b).name
            if nm in unit_of:
                self.unit_of_body[b] = unit_of[nm]
        for g in range(m.ngeom):
            u = self.unit_of_body.get(m.geom_bodyid[g])
            if u in self.geoms_of and (m.geom_contype[g] or m.geom_conaffinity[g]):
                self.geoms_of[u].append(g)
        self.panda_bodies_of = {pc["name"]: set(b for b in range(m.nbody) if m.body(b).name.startswith(pc["name"] + "/")) for pc in self.pandas}
        self.panda_bodies = set().union(*self.panda_bodies_of.values())
        self.rec_t, self.rec_q, self.rec_s = [], [], []
        self.label = {"step": 0, "text": "start", "part": None, "mate": None}
        self.n_tick = 0
        self.hooks = []
        self.recording = True
        self.jig = {}
        for _ in range(int(round(settle_s / self.tick_s))):
            for ct in self.ctrls.values():
                ct.tick()
            for _s in range(self.tick_steps):
                mujoco.mj_step(m, d)
        mujoco.mj_forward(m, d)
        d.time = 0.0
        self.initial = {u: self.body_T(u) for u in self.loose}
        log("[cell] OpenArm cell built and settled in %.1f s (%d geoms)" % (time.time() - t0, m.ngeom))

    # ------------------------------------------------------------------ helpers
    def body_T(self, name):
        return body_pose(self.d, self.m.body(name).id)

    def record(self):
        self.rec_t.append(float(self.d.time))
        self.rec_q.append(np.array(self.d.qpos, np.float32))
        self.rec_s.append({"states": dict(self.sm.state), "score": self.sm.score(), "label": dict(self.label)})

    def tick(self, n=1):
        for _ in range(n):
            self.tick_one()

    def tick_one(self):
        m, d = self.m, self.d
        for ct in self.ctrls.values():
            ct.tick()
        for name, (k, T) in self.jig.items():
            d.mocap_pos[k] = T[:3, 3]
            d.mocap_quat[k] = quat_wxyz(T[:3, :3])
        for _s in range(self.tick_steps):
            self.mj.mj_step(m, d)
        self.mj.mj_forward(m, d)
        self.sm.tick(float(d.time))
        self.n_tick += 1
        for h in self.hooks:
            h()
        if self.recording and self.n_tick % 4 == 0:
            self.record()

    def lowest_point(self):
        """the lowest point of any loose unit's collision geometry (world z, m)"""
        m, d = self.m, self.d
        z = 1e9
        for u in self.loose:
            for g in self.geoms_of[u]:
                R = d.geom_xmat[g].reshape(3, 3)
                k = m.geom_dataid[g]
                if m.geom_type[g] == self.mj.mjtGeom.mjGEOM_MESH and k >= 0:
                    v = m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k] + m.mesh_vertnum[k]]
                    z = min(z, float(np.min(v @ R[2, :]) + d.geom_xpos[g][2]))
        return z

    # ------------------------------------------------------------------ targets
    def unit_pose_for(self, mid, T_part):
        """the pose of the mate's moved unit's root body when its mate body (`part`) is at T_part (current hinge angles)"""
        mm = self.mates[mid]
        u = mm["unit"]
        if mm["part"] == u:
            return T_part
        return T_part @ _inv(_inv(self.body_T(u)) @ self.body_T(mm["part"]))

    def live_target(self, mid):
        """{"unit", "T" (the unit's root pose for the mate at the design, from its carrier as it is now), "T_part"}"""
        mm = self.mates[mid]
        Tc = self.body_T(mm["carrier"])
        Tp = Tc @ np.array(mm["T_part_in_carrier"], float)
        return {"unit": mm["unit"], "T": self.unit_pose_for(mid, Tp), "T_part": Tp, "carrier": mm["carrier"]}
