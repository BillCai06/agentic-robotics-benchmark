"""The acceptance procedure of brief B-14 on ANY bimanual S-R-S pair model, with an INJECTED catalogue and lot (public
code: every catalogue, lot, price and layer number comes from the data passed in).

    pair = PairModel.from_files(xml_path, pair_model_json_path)   the dynamics model + its arm data; refuses a model that
                                                                  is not an exact S-R-S pair (0.1 mm)
    acc = Acceptance(pair, layer)            targets, closed-form IK, moves, Cartesian samples, the mass regressor
    res = acc.evaluate(bom, lot, cat, brief) the 19 items + utilisations + peak current + cost
    der = acc.derived(bom, lot, cat, brief)  the 47 derived items
    ik(pair, arm, P, R, psi)                 B-14 b on one arm;  currents(cap, tau, qd, qdd)  one module's bus current

`layer` = layer.acc_layer(layer.json), `brief` = layer.brief_constants(layer.json), `cat` = catalog.Catalogue, `lot` = a
lot truth (lotmodel). `bom` = {"bus_voltage_v", "modules": {arm: {"j1".."j8": part_id}}, "boards": [part_id, ..] (with
repeats), "psu": part_id}; bom_from_json / bom_to_json convert bom.json (bom-bimanual/1).

Frames and angles: base frame +x forward, +y towards the left arm, z up; joint k's angle is the right-hand rotation about
its zero-pose axis (AXES; the right arm's axes are the left's mirrored in the x-z plane and negated, so mirrored poses have
equal angles); at the zero pose each arm hangs straight down and the gripper points along -z.

Procedure:
 1. Targets: the tray reference point at x in 6 equal steps over [x_min, x_max], y in 5 over [y_min, y_max], z in
    {z_lo, (z_lo + z_hi) / 2, z_hi}, grid order (x, y, z); left TCP = (x, y + w/2, z), right TCP = (x, y - w/2, z),
    w = handle_spacing_m; the TCP orientation R_T (relative to the zero pose) turns the gripper axis about y to the
    approach pitch (the elevation of the pointing direction; negative = below the horizontal, pointing forward), the
    fingers closing along y, the same R_T on both arms.
 2. IK, closed form per arm: wrist centre W = TCP - wrist_to_tcp x (pointing direction); j4 = acos from |SW| (j4 >= 0);
    the elbow on its circle about SW at the swivel angle from "down" (the projection of -z onto the plane normal to SW),
    positive towards +y for the left arm; j2 = asin (principal branch), j1 = atan2, j3 from the forearm direction; j5, j6
    (principal branch), j7 from the remaining rotation. The right arm is solved as the mirror image of a left arm with the
    right arm's own geometry. A target angle outside the stops is shifted by +-2 pi when that brings it inside. Every
    solution is checked by the model's forward kinematics (TCP error and orientation error < 1e-9).
 3. Moves: for each joint k = 1..7 the 8 not yet selected target pairs with the largest max over the two arms of
    |delta q_k| (ties, to 1e-9 rad: the lowest (i, j)); then the box corners' 4 space diagonals, 4 edges along x and 4
    along y, each pair once; every pair in both directions.
 4. Samples: the tray reference point on the straight segment, s(t) the trapezoid with equal thirds over T, samples at
    t = 0 .. T (round(T / dt) + 1 of them, dt = 0.005 s); q(s) by the IK (continuous along the move, starting at the
    start target's angles); qd = q'(s) s_dot, qdd = q''(s) s_dot^2 + q'(s) s_ddot, q' and q'' by central differences of
    the IK at s +- 1e-4, one-sided second-order where s -+ 1e-4 leaves [0, 1].
 5. Loads: mj_inverse (gravity 9.81, no contacts, limits, armature or passive forces) of the structure plus the BOM's
    modules (a point mass with the per-kg inertia of the POCKET's housing class at each slot) plus the payload m_hi / 2
    per arm as a point mass rigid on a_l7 at TCP + (0, 0, -com_drop) at R_T. The loads are linear in the masses: one
    inverse-dynamics pass per mass term (base, the slots of a joint on both arms, the payloads); a joint whose subtree
    does not hold a term's body gets an exact zero for it (a static slot, a_j1 on the pedestal, loads no joint). Holds:
    every target, static, payload m_hi / 2; proof holds: proof x m_hi / 2.
 6. Joint item (B-14 f), per arm and joint: every move sample |tau| + J_out |qdd| <= the available torque at |qd|; every
    hold and proof hold |tau| <= tau_s; the duty RMS sqrt(s_move max_moves mean (tau + J_out qdd)^2 + s_hold
    max_targets tau_hold^2) <= tau_c, s_move = T / (T + H); the bus voltage inside the module's range; housing = pocket.
 7. Gripper item (B-14 g): tau_c >= tau_grip = slip_sf (m_hi / 2) (g + a_z,max) finger_lever / mu, a_z,max the largest
    |vertical TCP acceleration| of the moves; w0 >= finger_open_rad / close_s; voltage; housing.
 8. PSU (B-14 h): output = bus voltage, every module's range contains it, rated current >= the peak over the move samples
    of the 16 modules' summed bus current I = idle + (|tau_m qd| + copper_loss_rated (tau_m / tau_c)^2) / V_s,
    tau_m = tau + J_out qdd, tau_c = the unit's continuous torque (a gripper: its holding current, w = 0,
    tau_m = tau_grip).
 9. Boards (B-14 i): catalog.boards_cover. 10. Budget (B-14 j): list price <= ceiling.
 11. Accepted iff all 19 items pass (16 modules, psu, boards, budget).
"""
import copy
import hashlib
import json
import math
import os

import numpy as np

from . import lotmodel as LM

ARMS = ("left", "right")
JOINTS = tuple("j%d" % k for k in range(1, 8))
MODULES = JOINTS + ("j8",)
ITEMS = tuple("%s_%s" % (a, m) for a in ARMS for m in MODULES) + ("psu", "boards", "budget")
SITES = ("shoulder", "elbow", "wrist", "tcp", "fingertip")
SITE_LINK = {"shoulder": 2, "elbow": 4, "wrist": 6, "tcp": 7, "fingertip": 7}
MIRROR = np.array([1.0, -1.0, 1.0])
_LEFT = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, -1], [0, -1, 0], [0, 0, -1], [0, -1, 0], [1, 0, 0]], float)
AXES = {"left": _LEFT, "right": -_LEFT * MIRROR}
G = 9.81
DT = 0.005
DS = 1e-4
N_X, N_Y = 6, 5
PAIRS_PER_JOINT = 8
SRS_TOL = 1e-4
FK_TOL = 1e-9
TIE_DECIMALS = 9
TWO_PI = 2.0 * math.pi


def _mujoco():
    import mujoco
    return mujoco


def wrap(a):
    return (np.asarray(a) + math.pi) % TWO_PI - math.pi


# ------------------------------------------------------------------------------------------------ rotations, closed form
def _rot(axis, a):
    a = np.asarray(a, float)
    c, s = np.cos(a), np.sin(a)
    R = np.zeros(a.shape + (3, 3))
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R[..., axis, axis] = 1.0
    R[..., i, i] = c
    R[..., j, j] = c
    R[..., i, j] = -s
    R[..., j, i] = s
    return R


def rx(a):
    return _rot(0, a)


def ry(a):
    return _rot(1, a)


def rz(a):
    return _rot(2, a)


def tcp_orientation(pitch_deg):
    """R_T: the gripper's pointing direction (-z at the zero pose) turned about y to the elevation `pitch_deg` (negative =
    below the horizontal), pointing forward; the fingers keep closing along y."""
    return ry(-(0.5 * math.pi + math.radians(float(pitch_deg))))


def ik_srs(P, R, psi, S, L1, L2, D):
    """Closed-form IK of a LEFT-convention S-R-S arm (shoulder centre S, upper arm L1, forearm L2, wrist-to-TCP D) for TCP
    positions P (n, 3) at the orientation R (relative to the zero pose) and swivel psi: (n, 7) principal-branch angles,
    NaN where the wrist centre is out of reach."""
    P = np.atleast_2d(np.asarray(P, float))
    W = P + D * R[:, 2]
    sw = W - S
    d = np.linalg.norm(sw, axis=1)
    c4 = (d * d - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
    q4 = np.arccos(np.clip(c4, -1.0, 1.0))
    n = sw / d[:, None]
    a = (L1 * L1 - L2 * L2 + d * d) / (2.0 * d)
    r = np.sqrt(np.maximum(L1 * L1 - a * a, 0.0))
    e1 = n[:, 2:3] * n
    e1[:, 2] -= 1.0
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(n, e1)
    e2 *= np.where(e2[:, 1] < 0.0, -1.0, 1.0)[:, None]
    E = S + a[:, None] * n + r[:, None] * (math.cos(psi) * e1 + math.sin(psi) * e2)
    u = (E - S) / L1
    q2 = np.arcsin(np.clip(-u[:, 1], -1.0, 1.0))
    q1 = np.arctan2(-u[:, 0], -u[:, 2])
    R02 = ry(q1) @ rx(-q2)
    f = np.einsum("nji,nj->ni", R02, (W - E) / L2)
    q3 = np.arctan2(-f[:, 1], f[:, 0])
    M = np.einsum("nji,jk->nik", R02 @ rz(-q3) @ ry(-q4), R)
    q6 = np.arcsin(np.clip(M[:, 2, 0], -1.0, 1.0))
    q5 = -np.arctan2(M[:, 1, 0], M[:, 0, 0])
    q7 = np.arctan2(M[:, 2, 1], M[:, 2, 2])
    Q = np.stack([q1, q2, q3, q4, q5, q6, q7], axis=1)
    Q[np.abs(c4) > 1.0] = np.nan
    return Q


def fk_srs(Q, S, L1, L2, D):
    """Closed-form FK of a LEFT-convention S-R-S arm: (elbow, wrist, TCP (n, 3), orientation (n, 3, 3))."""
    Q = np.atleast_2d(np.asarray(Q, float))
    R02 = ry(Q[:, 0]) @ rx(-Q[:, 1])
    R04 = R02 @ rz(-Q[:, 2]) @ ry(-Q[:, 3])
    R07 = R04 @ rz(-Q[:, 4]) @ ry(-Q[:, 5]) @ rx(Q[:, 6])
    E = S - L1 * R02[:, :, 2]
    W = E - L2 * R04[:, :, 2]
    return E, W, W - D * R07[:, :, 2], R07


def into_stops(Q, lo, hi):
    """A target angle outside the stops is shifted by +-2 pi when that brings it inside (B-14 b)."""
    Q = np.array(Q, float)
    for sgn in (1.0, -1.0):
        Qs = Q + sgn * TWO_PI
        Q = np.where(((Q < lo) | (Q > hi)) & (Qs >= lo) & (Qs <= hi), Qs, Q)
    return Q


def ik(pair, arm, P, R, psi):
    """B-14 b on one arm of a PairModel (the right arm as the mirror image of a left arm with its own geometry): (n, 7)
    principal-branch angles before the shift into the stops, NaN where out of reach."""
    return pair.ik(arm, P, R, psi)


def currents(cap, tau, qd, qdd):
    """Bus current (A) of one module at the samples (CONTRACTS 6, no credit for regeneration):
    idle + (|tau_m qd| + copper_loss_rated (tau_m / tau_c)^2) / V_s with tau_m = tau + J_out qdd and tau_c the unit's
    continuous torque (the copper loss follows the unit's own torque constant); a holding gripper: tau = tau_grip,
    qd = qdd = 0."""
    tm = tau + cap["J_out"] * qdd
    return cap["idle_a"] + (np.abs(tm * qd) + cap["cu_loss_w"] * (tm / cap["tau_c"]) ** 2) / cap["v_servo"]


def _line_dist(p, a, u):
    v = np.asarray(p, float) - a
    return float(np.linalg.norm(v - np.dot(v, u) * u))


# ------------------------------------------------------------------------------------------------ the pair model
class PairModel:
    """pair_dynamics.xml (structure-only inertials, both arms on the static pedestal) + pair_model.json
    (bimanual-pair-model/1: per arm joints, sites, slots, stops, pocket classes, finger lever and open angle; the structure
    moving mass; the housing classes). Raises ValueError for a model that is not an S-R-S pair in the conventions.

    pair_dynamics.xml: a static body "pedestal"; per arm a the chain a_l1 .. a_l7 (a_lk moved by the hinge a_jk, ref 0,
    its axis at the zero pose as in AXES), exactly 14 dofs; sites shoulder (on a_l2), elbow (a_l4), wrist (a_l6), tcp
    and fingertip (a_l7). pair_model.json:
      {"arms": {a: {"joints": [a_j1 .. a_j7 names], "sites": {"shoulder" | "elbow" | "wrist" | "tcp" | "fingertip": name},
                    "slots": {"j1".."j8": {"body": name (static, or on the arm's own chain), "pos": [x, y, z] (m, in that
                              body's frame), "quat": [w, x, y, z] (local z = the joint axis at the zero pose)}},
                    "stops": {"j1".."j8": [lo, hi] rad, j8 = [0, open angle]}, "pocket_class": {"j1".."j8": class},
                    "finger_lever_m", "finger_open_rad"}},
       "structure_moving_mass_kg": {a: kg turning with a_j1, no module},
       "housing": {class: {"shape": "cylinder", "dims_m": [diameter, length]} | {"shape": "box", "dims_m": [L, W, H],
                   H along the axis}}}"""

    def __init__(self, xml, model, xml_path=None):
        self.xml = xml
        self.xml_path = xml_path
        self.model_json = copy.deepcopy(model)
        try:
            self.arms = {a: self._arm(a, model) for a in ARMS}
            self.housing = {c: (h["shape"], [float(x) for x in h["dims_m"]]) for c, h in model["housing"].items()}
        except KeyError as e:
            raise ValueError("pair model json lacks %s" % e) from None
        self._measure()

    @classmethod
    def from_files(cls, xml_path, model_path):
        with open(xml_path) as fh:
            xml = fh.read()
        with open(model_path) as fh:
            model = json.load(fh)
        return cls(xml, model, xml_path=os.path.abspath(xml_path))

    @staticmethod
    def _arm(a, model):
        am = model["arms"][a]
        sites = am.get("sites") or {}
        return {"joints": list(am.get("joints") or ["%s_%s" % (a, j) for j in JOINTS]),
                "bodies": ["%s_l%d" % (a, k) for k in range(1, 8)],
                "sites": {r: sites[r] if isinstance(sites.get(r), str) else "%s_%s" % (a, r) for r in SITES},
                "slots": {m: (am["slots"][m]["body"], np.array(am["slots"][m]["pos"], float),
                              np.array(am["slots"][m]["quat"], float)) for m in MODULES},
                "stops": {m: (float(am["stops"][m][0]), float(am["stops"][m][1])) for m in MODULES},
                "pocket_class": {m: am["pocket_class"][m] for m in MODULES},
                "finger_lever_m": float(am["finger_lever_m"]), "finger_open_rad": float(am["finger_open_rad"]),
                "structure_moving_mass_kg": float(model["structure_moving_mass_kg"][a])}

    def spec(self):
        mj = _mujoco()
        return mj.MjSpec.from_file(self.xml_path) if self.xml_path else mj.MjSpec.from_string(self.xml)

    def digest(self):
        return hashlib.sha256(self.xml.encode() + json.dumps(self.model_json, sort_keys=True).encode()).hexdigest()

    def inertia_per_kg(self, cls):
        """Inertia per kg about the housing centre, slot frame (z = the housing's axis): CONTRACTS 2.4. dims_m: a cylinder
        [diameter, length], a box [L, W, H] with H along the axis."""
        shape, dims = self.housing[cls]
        if shape == "cylinder":
            r, h = 0.5 * dims[0], dims[1]
            return np.array([(3 * r * r + h * h) / 12.0, (3 * r * r + h * h) / 12.0, r * r / 2.0])
        L, W, H = dims
        return np.array([(W * W + H * H) / 12.0, (L * L + H * H) / 12.0, (L * L + W * W) / 12.0])

    def _measure(self):
        """Zero-pose geometry + the CONTRACTS 2.2 checks; collects every problem before raising."""
        mj = _mujoco()
        m = self.spec().compile()
        d = mj.MjData(m)
        d.qpos[:] = 0.0
        mj.mj_kinematics(m, d)
        bad = []
        name = lambda kind, n: mj.mj_name2id(m, kind, n)  # noqa: E731
        ped = name(mj.mjtObj.mjOBJ_BODY, "pedestal")
        if ped < 0 or m.body_weldid[ped] != 0:
            bad.append("no static body 'pedestal'")
        if m.nv != 2 * len(JOINTS):
            bad.append("the model has %d dofs; it must have exactly the 14 arm joints" % m.nv)
        self.geo = {}
        for a in ARMS:
            arm = self.arms[a]
            jid = [name(mj.mjtObj.mjOBJ_JOINT, n) for n in arm["joints"]]
            bid = [name(mj.mjtObj.mjOBJ_BODY, n) for n in arm["bodies"]]
            sid = {r: name(mj.mjtObj.mjOBJ_SITE, n) for r, n in arm["sites"].items()}
            miss = [n for n, i in zip(arm["joints"] + arm["bodies"], jid + bid) if i < 0]
            miss += [arm["sites"][r] for r, i in sid.items() if i < 0]
            if miss:
                bad.append("%s arm: missing %s" % (a, ", ".join(miss)))
                continue
            for k in range(7):
                if m.jnt_type[jid[k]] != mj.mjtJoint.mjJNT_HINGE or m.jnt_bodyid[jid[k]] != bid[k]:
                    bad.append("%s: not a hinge of %s" % (arm["joints"][k], arm["bodies"][k]))
                if abs(m.qpos0[m.jnt_qposadr[jid[k]]]) > 0.0:
                    bad.append("%s: ref must be 0 (the zero pose)" % arm["joints"][k])
                if k and m.body_parentid[bid[k]] != bid[k - 1]:
                    bad.append("%s is not the child of %s" % (arm["bodies"][k], arm["bodies"][k - 1]))
            if m.body_weldid[m.body_parentid[bid[0]]] != 0:
                bad.append("%s must hang from a static body" % arm["bodies"][0])
            for r, i in sid.items():
                if m.site_bodyid[i] != bid[SITE_LINK[r] - 1]:
                    bad.append("site %s is not on %s" % (arm["sites"][r], arm["bodies"][SITE_LINK[r] - 1]))
            ax = np.array([d.xaxis[j] for j in jid])
            pts = np.array([d.xpos[m.jnt_bodyid[j]] + d.xmat[m.jnt_bodyid[j]].reshape(3, 3) @ m.jnt_pos[j] for j in jid])
            for k in range(7):
                if np.abs(ax[k] - AXES[a][k]).max() > SRS_TOL:
                    bad.append("%s axis %s at the zero pose, the conventions need %s" % (
                        arm["joints"][k], np.round(ax[k], 6).tolist(), AXES[a][k].tolist()))
            P = {r: d.site_xpos[i].copy() for r, i in sid.items()}
            for r, ks in (("shoulder", (0, 1, 2)), ("elbow", (2, 3)), ("wrist", (4, 5, 6))):
                for k in ks:
                    e = _line_dist(P[r], pts[k], AXES[a][k])
                    if e > SRS_TOL:
                        bad.append("%s %s is %.3g mm off the J%d axis" % (a, r, 1e3 * e, k + 1))
            for r in ("elbow", "wrist", "tcp"):
                e = float(np.linalg.norm((P[r] - P["shoulder"])[:2]))
                if e > SRS_TOL:
                    bad.append("%s %s is %.3g mm off the shoulder's vertical at the zero pose" % (a, r, 1e3 * e))
            if not P["shoulder"][2] > P["elbow"][2] > P["wrist"][2] > P["tcp"][2]:
                bad.append("%s arm does not hang down at the zero pose" % a)
            e = abs(float(np.dot(P["tcp"] - P["wrist"], AXES[a][6])))
            if e > SRS_TOL:
                bad.append("%s TCP is %.3g mm off the J7-perpendicular gripper axis" % (a, 1e3 * e))
            for mm in MODULES:
                body, _, quat = arm["slots"][mm]
                b = up = name(mj.mjtObj.mjOBJ_BODY, body)
                while up > 0 and up != bid[0]:      # a moving slot body must ride on this arm's own chain
                    up = m.body_parentid[up]
                if b < 0 or np.linalg.norm(quat) < 1e-9:
                    bad.append("%s slot %s: body %r missing or quat zero" % (a, mm, body))
                elif m.body_weldid[b] != 0 and up != bid[0]:
                    bad.append("%s slot %s: body %r is neither static nor on the %s arm" % (a, mm, body, a))
                if arm["pocket_class"][mm] not in self.housing:
                    bad.append("%s %s: pocket class %r has no housing" % (a, mm, arm["pocket_class"][mm]))
                lo, hi = arm["stops"][mm]
                if not lo <= hi:
                    bad.append("%s %s: stops [%g, %g]" % (a, mm, lo, hi))
            R7 = d.xmat[bid[6]].reshape(3, 3).copy()
            self.geo[a] = {"S": P["shoulder"], "E": P["elbow"], "W": P["wrist"], "T": P["tcp"], "F": P["fingertip"],
                           "L1": float(np.linalg.norm(P["elbow"] - P["shoulder"])),
                           "L2": float(np.linalg.norm(P["wrist"] - P["elbow"])),
                           "D": float(np.linalg.norm(P["tcp"] - P["wrist"])),
                           "R7": R7, "tcp_local": R7.T @ (P["tcp"] - d.xpos[bid[6]]),
                           "R_tcp": d.site_xmat[sid["tcp"]].reshape(3, 3).copy(),
                           "model_moving_mass_kg": float(m.body_subtreemass[bid[0]])}
        if bad:
            raise ValueError("not an S-R-S pair in the CONTRACTS 2 conventions: " + "; ".join(bad))

    def ik(self, arm, P, R, psi):
        """Principal-branch closed-form IK of `arm` (the right arm as the mirror image of a left arm with its own
        geometry): (n, 7) angles, NaN where out of reach."""
        g = self.geo[arm]
        if arm == "left":
            return ik_srs(P, R, psi, g["S"], g["L1"], g["L2"], g["D"])
        Sm = np.diag(MIRROR)
        return ik_srs(np.atleast_2d(np.asarray(P, float)) * MIRROR, Sm @ R @ Sm, psi, g["S"] * MIRROR, g["L1"], g["L2"],
                      g["D"])


# ------------------------------------------------------------------------------------------------ targets and moves
def grid_points(lay):
    """The 90 tray reference points (x, y, z) in the grid order (x, y, z)."""
    xs = np.linspace(lay["x_min"], lay["x_max"], N_X)
    ys = np.linspace(lay["y_min"], lay["y_max"], N_Y)
    zs = (lay["z_lo"], 0.5 * (lay["z_lo"] + lay["z_hi"]), lay["z_hi"])
    return np.array([(x, y, z) for x in xs for y in ys for z in zs], float)


def grid_index(ix, iy, iz):
    return (ix * N_Y + iy) * 3 + iz


def corner_pairs():
    """The corner box x in {min, max} x y in {min, max} x z in {lo, hi}: 4 space diagonals, 4 edges along x, 4 along y."""
    X, Y, Z = (0, N_X - 1), (0, N_Y - 1), (0, 2)
    c = grid_index
    out = [(c(X[0], Y[0], Z[0]), c(X[1], Y[1], Z[1])), (c(X[0], Y[1], Z[0]), c(X[1], Y[0], Z[1])),
           (c(X[1], Y[0], Z[0]), c(X[0], Y[1], Z[1])), (c(X[1], Y[1], Z[0]), c(X[0], Y[0], Z[1]))]
    out += [(c(X[0], y, z), c(X[1], y, z)) for z in Z for y in Y]
    out += [(c(x, Y[0], z), c(x, Y[1], z)) for z in Z for x in X]
    return out


def select_moves(qL, qR):
    """B-14 c: the target pairs (i < j) in selection order."""
    iu, ju = np.triu_indices(len(qL), 1)
    chosen, seen = [], set()
    for k in range(len(JOINTS)):
        d = np.maximum(np.abs(qL[iu, k] - qL[ju, k]), np.abs(qR[iu, k] - qR[ju, k]))
        cnt = 0
        for o in np.lexsort((ju, iu, -np.round(d, TIE_DECIMALS))):
            pr = (int(iu[o]), int(ju[o]))
            if pr in seen:
                continue
            seen.add(pr)
            chosen.append(pr)
            cnt += 1
            if cnt == PAIRS_PER_JOINT:
                break
    for a, b in corner_pairs():
        pr = (min(a, b), max(a, b))
        if pr not in seen:
            seen.add(pr)
            chosen.append(pr)
    return chosen


def trapezoid(t, T):
    """Path fraction s(t), s_dot, s_ddot of the trapezoid with equal thirds over T (peak rate 1.5 / T)."""
    t = np.asarray(t, float)
    ta, v = T / 3.0, 1.5 / T
    acc = v / ta
    s = np.where(t < ta, 0.5 * acc * t * t, np.where(t < 2 * ta, 0.5 * acc * ta * ta + v * (t - ta),
                                                      1.0 - 0.5 * acc * (T - t) ** 2))
    sd = np.where(t < ta, acc * t, np.where(t < 2 * ta, v, acc * (T - t)))
    sdd = np.where(t < ta, acc, np.where(t < 2 * ta, 0.0, -acc))
    return s, sd, sdd


# ------------------------------------------------------------------------------------------------ the procedure
class Acceptance:
    """Targets, IK, moves, samples and the regressor traces of one (pair, layer); the heavy work happens once here."""

    def __init__(self, pair, layer):
        self.pair = pair
        self.layer = dict(layer)
        lay = self.layer
        self.R_T = tcp_orientation(lay["approach_pitch_deg"])
        self.psi = math.radians(lay["swivel_deg"])
        self.T = float(lay["T"])
        self.ref = grid_points(lay)
        half = np.array([0.0, 0.5 * float(lay["handle_spacing_m"]), 0.0])
        self.offset = {"left": half, "right": -half}
        self.lo = {a: np.array([pair.arms[a]["stops"][j][0] for j in JOINTS]) for a in ARMS}
        self.hi = {a: np.array([pair.arms[a]["stops"][j][1] for j in JOINTS]) for a in ARMS}
        self._build_model()
        self.q_grid = {}
        for a in ARMS:
            Q = pair.ik(a, self.ref + self.offset[a], self.R_T, self.psi)
            if np.isnan(Q).any():
                raise ValueError("%s arm: target(s) %s out of reach" % (a, np.where(np.isnan(Q[:, 0]))[0].tolist()))
            self.q_grid[a] = into_stops(Q, self.lo[a], self.hi[a])
        self.targets = self._check_targets()
        self.pairs = select_moves(self.q_grid["left"], self.q_grid["right"])
        self._samples()
        self._regress()

    # -------------------------------------------------------------------------------------------- model
    def _build_model(self):
        mj = _mujoco()
        p = self.pair
        spec = p.spec()
        spec.option.gravity = [0.0, 0.0, -G]
        spec.option.density = 0.0
        spec.option.viscosity = 0.0
        B = mj.mjtDisableBit
        spec.option.disableflags |= int(B.mjDSBL_CONSTRAINT) | int(B.mjDSBL_SPRING) | int(B.mjDSBL_DAMPER) | int(
            B.mjDSBL_ACTUATION)
        for j in spec.joints:
            j.armature = 0.0
        for b in spec.bodies:
            b.gravcomp = 0.0

        def add(parent, nm, pos, quat, inertia):
            b = spec.body(parent).add_body()
            b.name, b.pos, b.quat = nm, list(pos), list(quat)
            b.ipos, b.iquat = [0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]
            b.mass, b.inertia, b.explicitinertial = 1.0, list(inertia), True

        for a in ARMS:
            for m in MODULES:
                body, pos, quat = p.arms[a]["slots"][m]
                add(body, "reqslot_%s_%s" % (a, m), pos, quat / np.linalg.norm(quat),
                    p.inertia_per_kg(p.arms[a]["pocket_class"][m]))
            g = p.geo[a]
            pay = g["tcp_local"] + g["R7"].T @ self.R_T.T @ np.array([0.0, 0.0, -float(self.layer["com_drop"])])
            add(p.arms[a]["bodies"][6], "reqpayload_" + a, pay, (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        self.model = spec.compile()
        self.data = mj.MjData(self.model)
        m = self.model
        idn = lambda kind, n: mj.mj_name2id(m, kind, n)  # noqa: E731
        self.qadr, self.dadr, self.tcp_site, self.terms = {}, {}, {}, {}
        for a in ARMS:
            jid = [idn(mj.mjtObj.mjOBJ_JOINT, n) for n in p.arms[a]["joints"]]
            self.qadr[a] = np.array([m.jnt_qposadr[j] for j in jid])
            self.dadr[a] = np.array([m.jnt_dofadr[j] for j in jid])
            self.tcp_site[a] = idn(mj.mjtObj.mjOBJ_SITE, p.arms[a]["sites"]["tcp"])
        for mm in MODULES:
            self.terms[mm] = [idn(mj.mjtObj.mjOBJ_BODY, "reqslot_%s_%s" % (a, mm)) for a in ARMS]
        self.terms["payload"] = [idn(mj.mjtObj.mjOBJ_BODY, "reqpayload_" + a) for a in ARMS]
        self.unit_inertia = {bid: m.body_inertia[bid].copy() for ids in self.terms.values() for bid in ids}

        def ancestors(b):
            out = set()
            while b > 0:
                out.add(b)
                b = m.body_parentid[b]
            return out

        self.carries = {}                   # {arm: {term: (7,) bool}}: joint k carries the term body (in its subtree)
        for ia, a in enumerate(ARMS):
            links = [idn(mj.mjtObj.mjOBJ_BODY, n) for n in p.arms[a]["bodies"]]
            self.carries[a] = {t: np.array([lk in ancestors(ids[ia]) for lk in links]) for t, ids in self.terms.items()}

    def _qpos(self, qs):
        out = np.zeros((len(next(iter(qs.values()))), self.model.nq))
        for a in ARMS:
            out[:, self.qadr[a]] = qs[a]
        return out

    def _fk(self, QP):
        """Model FK of both TCP sites at the rows of a full qpos array: {arm: (positions, rotation matrices)}."""
        mj = _mujoco()
        m, d = self.model, self.data
        out = {a: (np.empty((len(QP), 3)), np.empty((len(QP), 3, 3))) for a in ARMS}
        for i in range(len(QP)):
            d.qpos[:] = QP[i]
            mj.mj_kinematics(m, d)
            for a in ARMS:
                out[a][0][i] = d.site_xpos[self.tcp_site[a]]
                out[a][1][i] = d.site_xmat[self.tcp_site[a]].reshape(3, 3)
        return out

    def _fk_errors(self, a, pos, rot, target_pos):
        R_ref = self.R_T @ self.pair.geo[a]["R_tcp"]
        return np.linalg.norm(pos - target_pos, axis=1), np.abs(rot - R_ref).reshape(len(rot), -1).max(axis=1)

    def _check_targets(self):
        fk = self._fk(self._qpos(self.q_grid))
        out = {}
        for a in ARMS:
            Q = self.q_grid[a]
            pe, ae = self._fk_errors(a, fk[a][0], fk[a][1], self.ref + self.offset[a])
            out[a] = {"pos_err_m": pe, "ang_err": ae, "margin_rad": np.minimum(Q - self.lo[a], self.hi[a] - Q).min(axis=1)}
        return out

    def grid_ok(self, margin_rad=0.0):
        """Every target of both arms solved inside the stops with `margin_rad` to spare, and every target and move
        sample confirmed by the model's FK (TCP and orientation errors < 1e-9)."""
        fk = all(f["pos_err_m"] < FK_TOL and f["ang_err"] < FK_TOL for f in self.sample_fk.values())
        return fk and all(bool((r["pos_err_m"] < FK_TOL).all() and (r["ang_err"] < FK_TOL).all()
                               and (r["margin_rad"] >= margin_rad - 1e-12).all()) for r in self.targets.values())

    # -------------------------------------------------------------------------------------------- samples
    def _samples(self):
        n = int(round(self.T / DT)) + 1
        s, sd, sdd = trapezoid(np.linspace(0.0, self.T, n), self.T)
        self.moves = [mv for (i, j) in self.pairs for mv in ((i, j), (j, i))]
        nm = len(self.moves)
        self.n_moves, self.samples_per_move = nm, n
        self.move_id = np.repeat(np.arange(nm), n)
        S0, SD, SDD = np.tile(s, nm), np.tile(sd, nm), np.tile(sdd, nm)
        A = self.ref[np.array([mv[0] for mv in self.moves])][self.move_id]
        dX = self.ref[np.array([mv[1] for mv in self.moves])][self.move_id] - A
        self.s, self.s_dot, self.s_ddot = S0, SD, SDD
        self.path = A + S0[:, None] * dX
        self.az_max = float(np.abs(dX[:, 2] * SDD).max())
        fwd = np.where(S0 - DS < 0.0)[0]
        bwd = np.where(S0 + DS > 1.0)[0]
        self.mv_q, self.mv_qd, self.mv_qdd, self.end_mismatch, self.sample_margin = {}, {}, {}, {}, {}
        for a in ARMS:
            def ik(rows, sv):
                return self.pair.ik(a, A[rows] + sv[:, None] * dX[rows] + self.offset[a], self.R_T, self.psi)
            allr = np.arange(len(S0))
            Q0, Qp, Qm = ik(allr, S0), ik(allr, S0 + DS), ik(allr, S0 - DS)
            if np.isnan(Q0).any() or np.isnan(Qp[S0 + DS <= 1.0]).any() or np.isnan(Qm[S0 - DS >= 0.0]).any():
                raise ValueError("%s arm: a move sample is out of reach" % a)
            Dp, Dm = wrap(Qp - Q0), wrap(Qm - Q0)
            d1 = (Dp - Dm) / (2.0 * DS)
            d2 = (Dp + Dm) / (DS * DS)
            for rows, sg, D1 in ((fwd, 1.0, Dp), (bwd, -1.0, Dm)):
                if len(rows):
                    D2 = wrap(ik(rows, S0[rows] + 2 * sg * DS) - Q0[rows])
                    D3 = wrap(ik(rows, S0[rows] + 3 * sg * DS) - Q0[rows])
                    d1[rows] = sg * (4.0 * D1[rows] - D2) / (2.0 * DS)
                    d2[rows] = (-5.0 * D1[rows] + 4.0 * D2 - D3) / (DS * DS)
            if np.isnan(d1).any() or np.isnan(d2).any():
                raise ValueError("%s arm: a move sample's difference stencil is out of reach" % a)
            Q = Q0.copy()
            mism = 0
            for k, (i, j) in enumerate(self.moves):
                r = slice(k * n, (k + 1) * n)
                Qu = np.unwrap(Q0[r], axis=0)
                Qu += TWO_PI * np.round((self.q_grid[a][i] - Qu[0]) / TWO_PI)
                Q[r] = Qu
                mism += int(np.abs(Qu[-1] - self.q_grid[a][j]).max() > 1e-6)
            self.mv_q[a] = Q                                   # (N, 7) arrays column-major: one joint is contiguous
            self.mv_qd[a] = np.asfortranarray(d1 * SD[:, None])
            self.mv_qdd[a] = np.asfortranarray(d2 * (SD * SD)[:, None] + d1 * SDD[:, None])
            self.end_mismatch[a] = mism
            self.sample_margin[a] = float(np.minimum(Q - self.lo[a], self.hi[a] - Q).min())

    # -------------------------------------------------------------------------------------------- regressor
    def _inverse(self, QP, QV, QA, fk=False):
        mj = _mujoco()
        m, d = self.model, self.data
        out = np.empty((len(QP), m.nv))
        rec = {a: (np.empty((len(QP), 3)), np.empty((len(QP), 9))) for a in ARMS} if fk else None
        for i in range(len(QP)):
            d.qpos[:] = QP[i]
            d.qvel[:] = QV[i]
            d.qacc[:] = QA[i]
            mj.mj_inverse(m, d)
            out[i] = d.qfrc_inverse
            if fk:
                for a in ARMS:
                    rec[a][0][i] = d.site_xpos[self.tcp_site[a]]
                    rec[a][1][i] = d.site_xmat[self.tcp_site[a]]
        return out, rec

    def _set_term(self, term, on):
        m = self.model
        for bid in self.terms[term]:
            m.body_mass[bid] = 1.0 if on else 0.0
            m.body_inertia[bid] = self.unit_inertia[bid] if on else 0.0

    def _regress(self):
        """tau = tau0 + sum_k m_k b_k + m_p b_p over the move samples and the holds (exact: inverse dynamics is linear in
        the inertial parameters, and every term body carries the per-kg inertia)."""
        m = self.model
        nv = m.nv
        QP = self._qpos(self.mv_q)
        QV, QA = np.zeros((len(QP), nv)), np.zeros((len(QP), nv))
        for a in ARMS:
            QV[:, self.dadr[a]] = self.mv_qd[a]
            QA[:, self.dadr[a]] = self.mv_qdd[a]
        HP = self._qpos(self.q_grid)
        H0 = np.zeros((len(HP), nv))
        for t in self.terms:
            self._set_term(t, False)
        mv, hd = {}, {}
        mv[None], fk = self._inverse(QP, QV, QA, fk=True)
        hd[None] = self._inverse(HP, H0, H0)[0]
        for t, ids in self.terms.items():
            if all(m.body_weldid[b] == 0 for b in ids):        # a static slot (the pedestal's) loads no joint
                mv[t], hd[t] = mv[None], hd[None]
                continue
            self._set_term(t, True)
            mv[t], hd[t] = self._inverse(QP, QV, QA)[0], self._inverse(HP, H0, H0)[0]
            self._set_term(t, False)
        self.mv0 = {a: np.asfortranarray(mv[None][:, self.dadr[a]]) for a in ARMS}
        self.hd0 = {a: np.asfortranarray(hd[None][:, self.dadr[a]]) for a in ARMS}
        # a joint that does not carry a term body gets an exact zero column (mj_inverse works about the tree root's
        # centre of mass, which every added mass moves, so the raw difference is zero only to roundoff)
        self.mvb = {a: {t: np.asfortranarray(np.where(self.carries[a][t], mv[t][:, self.dadr[a]] - self.mv0[a], 0.0))
                        for t in self.terms} for a in ARMS}
        self.hdb = {a: {t: np.asfortranarray(np.where(self.carries[a][t], hd[t][:, self.dadr[a]] - self.hd0[a], 0.0))
                        for t in self.terms} for a in ARMS}
        self.sample_fk = {}
        for a in ARMS:
            pe, ae = self._fk_errors(a, fk[a][0], fk[a][1].reshape(-1, 3, 3), self.path + self.offset[a])
            self.sample_fk[a] = {"pos_err_m": float(pe.max()), "ang_err": float(ae.max())}

    # -------------------------------------------------------------------------------------------- loads and items
    def loads(self, a, masses, payload):
        """(moves (N, 7), holds (90, 7), proof holds (90, 7)) load torques of arm a: the joint_trace columns."""
        out = [np.empty((len(x), len(JOINTS)), order="F") for x in (self.move_id, self.ref, self.ref)]
        for k in range(len(JOINTS)):
            for o, col in zip(out, self.joint_trace(a, k, masses, payload)):
                o[:, k] = col
        return out

    def joint_trace(self, a, k, masses, payload):
        """(moves (N,), holds (90,), proof holds (90,)) load torque of joint index k of arm a for module masses
        {module: kg} and the payload (kg); a module that is not given carries no mass."""
        mv = self.mv0[a][:, k].copy()
        hd = self.hd0[a][:, k].copy()
        for mm in MODULES:
            mk = float(masses.get(mm, 0.0))
            if mk:
                mv += mk * self.mvb[a][mm][:, k]
                hd += mk * self.hdb[a][mm][:, k]
        pr = hd + float(self.layer["proof"]) * payload * self.hdb[a]["payload"][:, k]
        mv += payload * self.mvb[a]["payload"][:, k]
        hd += payload * self.hdb[a]["payload"][:, k]
        return mv, hd, pr

    def payload(self):
        return 0.5 * float(self.layer["m_hi"])

    def s_move(self):
        return self.T / (self.T + float(self.layer["H"]))

    def _duty_rms(self, motor, hold):
        msq = np.bincount(self.move_id, weights=motor * motor, minlength=self.n_moves) / self.samples_per_move
        sm = self.s_move()
        return math.sqrt(sm * float(msq.max()) + (1.0 - sm) * float((hold * hold).max()))

    def joint_util(self, a, k, cap, mv, hd, pr):
        """Utilisations (> 1 fails) of B-14 f for joint index k of arm a with capability `cap` and the load traces."""
        qd = np.abs(self.mv_qd[a][:, k])
        qdd = self.mv_qdd[a][:, k]
        need = np.abs(mv) + cap["J_out"] * np.abs(qdd)
        avail = LM.available_torque_array(qd, cap)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(avail > 0.0, need / avail, np.where(need > 0.0, np.inf, 0.0))
        u = {"move": float(r.max()), "hold": float(np.abs(hd).max()) / cap["tau_s"],
             "proof": float(np.abs(pr).max()) / cap["tau_s"],
             "rms": self._duty_rms(mv + cap["J_out"] * qdd, hd) / cap["tau_c"]}
        u["worst"] = max(u.values())
        return u

    def grip_torque(self, a):
        L = self.layer
        lever = self.pair.arms[a]["finger_lever_m"]
        return float(L["slip_sf"]) * self.payload() * (G + self.az_max) * lever / float(L["mu"])

    def gripper_util(self, a, cap):
        need_w = self.pair.arms[a]["finger_open_rad"] / float(self.layer["close_s"])
        u = {"hold": self.grip_torque(a) / cap["tau_c"], "close": need_w / cap["w0"] if cap["w0"] > 0 else math.inf}
        u["worst"] = max(u.values())
        return u

    def arm_current(self, a, caps_a, tau):
        """Summed bus current (A) of arm a's 8 modules at every move sample; tau = (N, 7) load torques."""
        I = np.zeros(len(self.move_id))
        for k, j in enumerate(JOINTS):
            I += currents(caps_a[j], tau[:, k], self.mv_qd[a][:, k], self.mv_qdd[a][:, k])
        return I + currents(caps_a["j8"], self.grip_torque(a), 0.0, 0.0)

    def caps_for(self, bom, lot, cat, brief):
        bus = float(bom["bus_voltage_v"])
        return {a: {m: LM.unit_capability(bom["modules"][a][m], bus, lot, cat, brief) for m in MODULES} for a in ARMS}

    def module_item(self, a, m, cap, ok_bus, masses, loads=None):
        """(pass, utilisations) of module m of arm a (B-14 f / g) for its capability, whether its range contains the bus
        voltage, and the arm's module masses (only the masses its joint carries matter) or their loads()."""
        if m == "j8":
            u = self.gripper_util(a, cap)
        else:
            k = JOINTS.index(m)
            tr = self.joint_trace(a, k, masses, self.payload()) if loads is None else [x[:, k] for x in loads]
            u = self.joint_util(a, k, cap, *tr)
        u["fits"] = cap["housing"] == self.pair.arms[a]["pocket_class"][m]
        u["voltage_ok"] = bool(ok_bus)
        return bool(u["fits"] and u["voltage_ok"] and u["worst"] <= 1.0 + 1e-12), u

    def peak_current(self, caps):
        """Peak over the move samples of the 16 modules' summed bus current (left arm + right arm)."""
        tot = None
        for a in ARMS:
            I = self.arm_current(a, caps[a], self.loads(a, {m: caps[a][m]["mass_kg"] for m in MODULES}, self.payload())[0])
            tot = I if tot is None else tot + I
        return float(tot.max())

    def psu_ok(self, bom, cat, peak):
        bus = float(bom["bus_voltage_v"])
        pid = bom.get("psu")
        if not pid or not cat.has(pid) or cat.role(pid) != "psu":
            return False
        p = cat.get(pid)
        return bool(abs(float(p["output_v"]) - bus) < 1e-9 and float(p["rated_current_a"]) >= peak - 1e-12
                    and all(cat.on_bus(x, bus) for arm in bom["modules"].values() for x in arm.values()))

    def budget_ok(self, cents):
        c = float(self.layer["ceiling"])
        return True if math.isinf(c) else cents <= math.floor(100.0 * c + 1e-6)

    def evaluate(self, bom, lot, cat, brief, caps=None):
        """The 19 items of B-14 + utilisations + the peak summed current + the list price."""
        bus = float(bom["bus_voltage_v"])
        caps = caps or self.caps_for(bom, lot, cat, brief)
        items, util, tot = {}, {}, None
        for a in ARMS:
            masses = {m: caps[a][m]["mass_kg"] for m in MODULES}
            tr = self.loads(a, masses, self.payload())
            for m in MODULES:
                nm = "%s_%s" % (a, m)
                items[nm], util[nm] = self.module_item(a, m, caps[a][m], cat.on_bus(bom["modules"][a][m], bus), masses, tr)
            I = self.arm_current(a, caps[a], tr[0])
            tot = I if tot is None else tot + I
        peak = float(tot.max())
        items["psu"] = self.psu_ok(bom, cat, peak)
        items["boards"] = bool(cat.boards_cover(bom.get("boards", []), cat.groups(bom["modules"]), bus))
        cents = cat.bom_cents(bom)
        items["budget"] = self.budget_ok(cents)
        return {"items": items, "util": util, "peak_current_a": peak, "cost": cents / 100.0, "caps": caps,
                "accepted": all(items.values()), "pass_fraction": sum(items.values()) / float(len(ITEMS))}

    def derived(self, bom, lot, cat, brief):
        """The 47 derived items of B-16 for the BOM installed on this pair (the delivered lot's weakest unit)."""
        caps = self.caps_for(bom, lot, cat, brief)
        out, tot = {}, None
        for a in ARMS:
            masses = {m: caps[a][m]["mass_kg"] for m in MODULES}
            tau, hds, _ = self.loads(a, masses, self.payload())
            for k, j in enumerate(JOINTS):
                mv, hd = tau[:, k], hds[:, k]
                nm = "%s_%s" % (a, j)
                out["joint_peak_torque:" + nm] = float(max(np.abs(mv).max(), np.abs(hd).max()))
                out["joint_rms_torque:" + nm] = self._duty_rms(mv, hd)
                out["joint_peak_speed:" + nm] = float(np.abs(self.mv_qd[a][:, k]).max())
            out["grip_torque:" + a] = self.grip_torque(a)
            out["moving_mass:" + a] = self.pair.arms[a]["structure_moving_mass_kg"] + sum(masses[m] for m in MODULES[1:])
            I = self.arm_current(a, caps[a], tau)
            tot = I if tot is None else tot + I
        out["supply_peak_current"] = float(tot.max())
        return out


# ------------------------------------------------------------------------------------------------ bom.json
def bom_from_json(bj):
    """bom.json (bom-bimanual/1) -> the internal BOM dict."""
    modules, boards, psu = {a: {} for a in ARMS}, [], None
    for line in bj["parts"]:
        role = line.get("role")
        if role == "module":
            modules[line["arm"]][line["joint"]] = line["part_id"]
        elif role == "bus_board":
            boards += [line["part_id"]] * int(line.get("count", 1))
        elif role == "psu":
            psu = line["part_id"]
    return {"bus_voltage_v": float(bj["bus_voltage_v"]), "modules": modules, "boards": boards, "psu": psu}


def bom_to_json(bom, cat):
    parts = [{"part_id": bom["modules"][a][m], "count": 1, "role": "module", "arm": a, "joint": m}
             for a in ARMS for m in MODULES]
    counts = {}
    for b in bom.get("boards", []):
        counts[b] = counts.get(b, 0) + 1
    parts += [{"part_id": b, "count": c, "role": "bus_board"}
              for b, c in sorted(counts.items(), key=lambda t: cat.index(t[0]))]
    parts.append({"part_id": bom["psu"], "count": 1, "role": "psu"})
    return {"format": "bom-bimanual/1", "bus_voltage_v": float(bom["bus_voltage_v"]), "parts": parts,
            "total_cost_usd": cat.bom_cost(bom)}
