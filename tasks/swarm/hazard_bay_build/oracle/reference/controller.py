"""The reference hoist controller: three flying robots lift the bucket together, carry it to the burning pan, pour,
and hold. Speaks the task's wire on stdin/stdout (one JSON object per line; `READY` first).

Design, in the world frame of the twin (the robots' headings are fixed):
- one position loop per robot (PID on the estimated position, the rope's measured load fed forward);
- the formation's SHAPE comes from the lidars (every pair's relative vector, measured from both ends, propagated with
  the odometry velocities and corrected by the returns), its PLACE from the mean of the three odometries: the
  odometries drift apart, and the bucket hangs level only if the hooks' heights agree to a centimetre;
- the hooks stand on a circle round the bucket (a symmetric rig: equal tensions mean a level bucket); the robots'
  centres are then 1.35 m apart;
- a take-off in TENSION mode: while the bucket is on the floor the rope is a stiff spring (a centimetre of hook
  height is tens of newtons), so the vertical thrust commands each rope's tension directly (the height is free):
  the three tensions are equal by construction, ramp together, and the bucket leaves the floor level; position
  control takes over once it hangs, with a height trim on the measured loads that keeps the three equal;
- minimum-jerk lift and carry profiles (no acceleration steps: little swing, little slosh), the bucket's swing
  damped from what the cameras see, and the pour driven by what the thermal cameras see.
"""
import copy
import json
import math
import sys

import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # (run with -I: the script's folder is not on the path)
from vehicle import Vehicle  # noqa: E402

G = 9.81
UAVS = ("UAV_1", "UAV_2", "UAV_3")
# the rigging the reference scene has: which ring each robot's rope goes to, as a bearing from the bucket's centre
RING_BEARING = {"UAV_1": 90.0, "UAV_2": 210.0, "UAV_3": 330.0}
RING_R, RING_Z = 0.14, 0.25
BUCKET_START = np.array([0.0, -2.60])
FORMATION_R = 0.78         # the hooks' distance from the bucket's centre, horizontally (the robots' centres then sit
                           # 1.35 m apart: the grader's separation rule wants 1.2 m)
KP, KD, KI = 3.0, 2.6, 1.0  # the position loop [1/s^2], [1/s], [1/s^3] (the integral absorbs what the feed-forward misses)
A_RISE, V_TAUT = 0.3, 0.08  # the take-off move's peak acceleration [m/s^2]; the approach to the taut rope [m/s]
A_LIFT, A_CARRY = 0.35, 0.2  # [m/s^2] the peak accelerations of the minimum-jerk lift and carry
T_PRELOAD = 20.0           # [N] each rope's share of the bucket before the lift (3 x 20 N < the lightest bucket's weight)
KD_T, V_CATCH = 12.0, 0.06  # tension mode: the vertical damping [1/s]; the climb while the rope is still slack [m/s]
T_RAMP, T_LIFT_RAMP = 8.0, 10.0  # [N/s] the preload ramp; the lift-off ramp (all three ropes together)
T_LIFT_CAP = 45.0          # [N] over the rope's weight: the bucket is surely off the floor by then (11.2 kg / 3 = 36.6 N)
K_TRIM = 0.002             # [m/(N s)] the height trim that keeps the three loads equal while the bucket hangs still
POUR_DROP, T_POUR, T_RECOVER = 0.9, 4.0, 4.0  # how far the pouring robot lowers its hook [m]; how long down and back up take [s]
POUR_PRE = 0.10            # [m] when its rope slackens the bucket swings away from the pouring robot by about this much (it
                           # hangs from the other two): the level bucket is aligned that far towards the pouring robot


def minjerk(t, T):
    """The minimum-jerk profile 0 -> 1 over T seconds: (fraction, rate)."""
    if T <= 0 or t >= T:
        return 1.0, 0.0
    if t <= 0:
        return 0.0, 0.0
    x = t / T
    return 10 * x ** 3 - 15 * x ** 4 + 6 * x ** 5, (30 * x ** 2 - 60 * x ** 3 + 30 * x ** 4) / T


def t_minjerk(dist, a_max):
    """The duration that keeps a minimum-jerk move of `dist` under the peak acceleration a_max (peak = 5.77 d / T^2)."""
    return math.sqrt(5.77 * max(abs(dist), 1e-6) / a_max)


STEAM_WAIT = 4.0          # [s] after the lever has been pushed to its stop: the leak stops 3 s after it closes


class Controller:
    def __init__(self):
        self.state = None
        self.snapshot = None
        self.vehicle = Vehicle()

    def reset(self):
        self.state = None

    # -- set-up ---------------------------------------------------------------------------------------------------
    def init(self, obs):
        mis = obs["mission"]
        hooks = {n: np.array(mis["uav"][n]["hook"]) for n in UAVS}
        self.vehicle.setup(mis)
        s = dict(phase="WAIT", t_phase=0.0, last_t=-1, mis=mis, hooks=hooks, ugv=self.vehicle.init_state(),
                 load_f={n: np.zeros(3) for n in UAVS}, ie={n: np.zeros(3) for n in UAVS},
                 taut={n: False for n in UAVS}, z_ref=None, preload_ok=0.0, trim={n: 0.0 for n in UAVS},
                 rel={}, bucket_xy=None, bucket_f=None, off_f=None, off_rate=np.zeros(2), fire_seen_out=0.0, carry=None, align=None,
                 rise=None, tz={n: 0.0 for n in UAVS}, vz_ref={n: 0.0 for n in UAVS}, z_lo=None)
        # formation: each robot's HOOK stands out from the bucket along its own ring's bearing (the hook sits behind
        # the airframe's centre; the headings are fixed, so the root is the hook less that offset)
        s["form"] = {}
        for n in UAVS:
            b = math.radians(RING_BEARING[n])
            s["form"][n] = FORMATION_R * np.array([math.cos(b), math.sin(b)]) - hooks[n][:2]
        L = mis["rope_length"]
        rh = FORMATION_R - RING_R  # hook to ring, horizontally (the same bearing)
        s["z_taut"] = {n: RING_Z + math.sqrt(max(0.0, L * L - rh * rh)) - hooks[n][2] for n in UAVS}  # the root with the rope just taut
        s["z_cruise"] = mis["cruise_z"] - float(np.mean([hooks[n][2] for n in UAVS]))  # the hooks at the cruise height
        return s

    # -- estimation -----------------------------------------------------------------------------------------------
    def estimates(self, obs, s, dt):
        """Positions: the formation's shape from the lidars, its place from the mean of the odometries."""
        pos = {n: np.array(obs["uav"][n]["pos"]) for n in UAVS}
        vel = {n: np.array(obs["uav"][n]["vel"]) for n in UAVS}
        rel = s["rel"]
        for (n, m) in list(rel.keys()):  # propagate every pair with the odometry velocities (no lag while moving)
            rel[(n, m)] = rel[(n, m)] + (vel[m] - vel[n]) * dt
        seen = {}  # (n, m) -> true_m - true_n, as the lidars measured it this step (from either end)
        for n in UAVS:
            for v in obs["uav"][n].get("neighbors") or []:
                v = np.array(v)
                guess = {m: (rel[(n, m)] if (n, m) in rel else pos[m] - pos[n]) for m in UAVS if m != n}
                d, m = min((float(np.linalg.norm(guess[m] - v)), m) for m in guess)
                if d < 1.0:
                    seen.setdefault((n, m), []).append(v)
                    seen.setdefault((m, n), []).append(-v)
        a = min(1.0, dt / 0.4)  # the returns are noisy (a few cm); the propagation carries the motion
        for key, vs in seen.items():
            v = np.mean(vs, axis=0)
            rel[key] = v if key not in rel else (1 - a) * rel[key] + a * v
        c = np.mean([pos[n] for n in UAVS], axis=0)
        est = {}
        for n in UAVS:
            others = [rel.get((n, m)) for m in UAVS if m != n]
            if all(o is not None for o in others):
                est[n] = c - np.sum(others, axis=0) / 3.0  # true_n - centroid = -(1/3) sum_m (true_m - true_n)
            else:
                est[n] = pos[n]  # no lidar picture yet: the odometry alone
        return est, vel

    # -- the robot's loop -----------------------------------------------------------------------------------------
    def thrust(self, n, p, v, p_ref, v_ref, s, obs, dt, tz=None, vz_ref=0.0):
        """The robot's thrust: a PID position loop with the rope's load fed forward; in tension mode (`tz`, newtons)
        the vertical thrust commands the rope's tension instead and the height is left free (damped)."""
        m = s["mis"]["uav"][n]["mass"]
        load = np.array(obs["uav"][n]["load"])
        s["load_f"][n] = 0.7 * s["load_f"][n] + 0.3 * load
        ff = -s["load_f"][n]
        ff[2] = max(0.0, ff[2])  # the rope only pulls down; never feed a ground reaction forward
        ff = np.clip(ff, -60.0, 60.0)
        e = p_ref - p
        ie = np.clip(s["ie"][n] + e * dt, -0.5, 0.5)
        acc = KP * e + KD * (v_ref - v) + KI * ie
        f = m * (np.array([0.0, 0.0, G]) + acc) + ff
        if tz is not None:
            f[2] = m * G + tz - m * KD_T * (v[2] - vz_ref)
            ie[2] = 0.0
        tmax = s["mis"]["uav"][n]["thrust_max"]
        mag = float(np.linalg.norm(f))
        if mag > tmax:
            f = f * tmax / mag
        else:
            s["ie"][n] = ie  # (no wind-up while saturated)
        return [float(x) for x in f]

    def trim(self, s, dt):
        """Keep the three measured loads equal: a robot whose rope pulls more than the mean comes down a little."""
        lz = {n: float(s["load_f"][n][2]) for n in UAVS}
        mean = sum(lz.values()) / 3.0
        for n in UAVS:
            s["trim"][n] = float(np.clip(s["trim"][n] + K_TRIM * (lz[n] - mean) * dt, -0.06, 0.06))

    # -- the mission ----------------------------------------------------------------------------------------------
    def act(self, t, obs):
        if self.state is None or t == 0:
            self.state = self.init(obs)
            self.snapshot = copy.deepcopy(self.state)
        s = self.state
        if t == s["last_t"]:  # a repeated query: start again from the state before this step
            s = self.state = copy.deepcopy(self.snapshot)
        else:
            self.snapshot = copy.deepcopy(s)
        s["last_t"] = t
        mis = s["mis"]
        dt = mis["dt"]
        s["t_phase"] += dt
        pos, vel = self.estimates(obs, s, dt)
        cen_est = np.mean([pos[n] for n in UAVS], axis=0)
        # the bucket and the pan as the cameras see them (world estimates through the position estimates)
        bucket_est, pan_est, fire_seen = [], [], []
        for n in UAVS:
            u = obs["uav"][n]
            if u.get("bucket") is not None:
                bucket_est.append(pos[n] + np.array(u["bucket"]))
            if u.get("pan") is not None:
                pan_est.append(pos[n] + np.array(u["pan"]))
                fire_seen.append(bool(u.get("fire")))
        if bucket_est:
            s["bucket_xy"] = np.mean(bucket_est, axis=0)[:2]
            # a slow filter of it: where the formation centres itself for the take-off (the cameras, not the prior)
            s["bucket_f"] = s["bucket_xy"].copy() if s["bucket_f"] is None else s["bucket_f"] + min(1.0, dt / 1.0) * (s["bucket_xy"] - s["bucket_f"])
        bucket_xy = s["bucket_xy"] if s["bucket_xy"] is not None else BUCKET_START
        cxy = s["bucket_f"] if s["bucket_f"] is not None else BUCKET_START  # the formation's centre until the carry
        # the bucket's horizontal offset from the robots' centroid, filtered twice; its rate damps the swing
        off = bucket_xy - cen_est[:2]
        a = min(1.0, dt / 0.25)
        if s["off_f"] is None:
            s["off_f"] = off.copy()
        prev = s["off_f"]
        s["off_f"] = prev + a * (off - prev)
        s["off_rate"] = s["off_rate"] + a * ((s["off_f"] - prev) / dt - s["off_rate"])
        damp = np.clip(0.6 * s["off_rate"], -0.25, 0.25)

        def form(n, cxy, z):
            return np.array([cxy[0] + s["form"][n][0], cxy[1] + s["form"][n][1], z])

        refs, vrefs = {}, {}
        ugv_cmd = self.vehicle.act(s["ugv"], obs, s["phase"]) if "ugv" in obs else {}
        phase = s["phase"]
        if phase == "WAIT":  # on the pads until the steam is off: the pour hover is inside the plume
            done_t = s["ugv"].get("lever_done_t")
            gauge = (obs.get("ugv") or {}).get("gauge")
            if (done_t is not None and obs["time"] > done_t + STEAM_WAIT) or (gauge is not None and gauge < 0.05):
                s["phase"], s["t_phase"] = "RISE", 0.0
            return {"thrust": {n: [0.0, 0.0, 0.0] for n in UAVS}, "ugv": ugv_cmd}
        if phase == "RISE":  # a minimum-jerk move from the pad to the formation point just below the taut height
            if s["rise"] is None:
                starts = {n: np.array(mis["uav"][n]["start"], dtype=float) for n in UAVS}
                goals = {n: form(n, cxy, s["z_taut"][n] - 0.10) for n in UAVS}
                T = max([2.5] + [t_minjerk(float(np.linalg.norm(goals[n] - starts[n])), A_RISE) for n in UAVS])
                s["rise"] = dict(starts=starts, T=T)
            frac, rate = minjerk(s["t_phase"], s["rise"]["T"])
            for n in UAVS:
                goal = form(n, cxy, s["z_taut"][n] - 0.10)  # (the goal follows the cameras' bucket as it comes into view)
                d = goal - s["rise"]["starts"][n]
                refs[n] = s["rise"]["starts"][n] + frac * d
                vrefs[n] = rate * d
            if s["t_phase"] > s["rise"]["T"] + 0.5 and all(abs(pos[n][2] - (s["z_taut"][n] - 0.10)) < 0.08 for n in UAVS):
                s["phase"], s["t_phase"] = "TENSION", 0.0
        elif phase == "TENSION":  # tension mode: climb gently until each rope catches, then every rope to the same preload
            ok = all(s["taut"].values())
            for n in UAVS:
                rope_w = mis["uav"][n]["rope_mass"] * G
                t_meas = -float(s["load_f"][n][2])
                target = rope_w + T_PRELOAD
                if not s["taut"][n]:
                    s["tz"][n], s["vz_ref"][n] = rope_w + 2.0, V_CATCH  # (the slack rope hangs from the hook: its weight is a load)
                    if t_meas > rope_w + 4.0:
                        s["taut"][n] = True
                        s["tz"][n] = max(rope_w + 2.0, t_meas)
                else:
                    s["vz_ref"][n] = 0.0
                    s["tz"][n] = min(target, s["tz"][n] + T_RAMP * dt)
                    ok = ok and abs(t_meas - target) < 4.0  # (the tensions are equal by command; this only waits for the ramp)
                refs[n] = form(n, cxy, pos[n][2])  # (the height is not controlled here)
                vrefs[n] = np.zeros(3)
            s["preload_ok"] = s["preload_ok"] + dt if ok else 0.0
            if s["preload_ok"] > 0.3 or s["t_phase"] > 12.0:
                s["phase"], s["t_phase"] = "LIFTOFF", 0.0
                s["z_lo"] = float(np.mean([pos[n][2] for n in UAVS]))
        elif phase == "LIFTOFF":  # the three tensions ramp together until the bucket leaves the floor (level: equal pulls)
            for n in UAVS:
                s["tz"][n] += T_LIFT_RAMP * dt
                s["vz_ref"][n] = 0.0
                refs[n] = form(n, cxy, pos[n][2])
                vrefs[n] = np.zeros(3)
            rope_w = mis["uav"]["UAV_1"]["rope_mass"] * G
            if float(np.mean([pos[n][2] for n in UAVS])) - s["z_lo"] > 0.04 or s["tz"]["UAV_1"] - rope_w > T_LIFT_CAP:
                s["phase"], s["t_phase"] = "LIFT", 0.0
                s["lift_xy"] = np.array(cxy, dtype=float)  # (frozen: once the bucket hangs, the formation must not chase it)
                s["lift_z0"] = {n: float(pos[n][2]) for n in UAVS}
                s["T_lift"] = t_minjerk(s["z_cruise"] - float(np.mean(list(s["lift_z0"].values()))), A_LIFT)
        elif phase == "LIFT":  # all three up together, a minimum-jerk climb to the cruise height, then a short hold
            if s["t_phase"] > s["T_lift"]:
                self.trim(s, dt)  # (only once the climb is over: the loads are dynamic while it accelerates)
            frac, rate = minjerk(s["t_phase"], s["T_lift"])
            for n in UAVS:
                h = s["z_cruise"] - s["lift_z0"][n]
                refs[n] = form(n, s["lift_xy"] + damp, s["lift_z0"][n] + frac * h + s["trim"][n])
                vrefs[n] = np.array([0.0, 0.0, rate * h])
            if s["t_phase"] > s["T_lift"] + 1.5:
                s["phase"], s["t_phase"] = "CARRY", 0.0
                p0, p1 = s["lift_xy"].copy(), np.array(mis["pan"], dtype=float)
                dist = float(np.linalg.norm(p1 - p0))
                s["carry"] = dict(p0=p0, p1=p1, dist=dist, dirn=(p1 - p0) / dist, T=t_minjerk(dist, A_CARRY))
        elif phase == "CARRY":  # the formation's centre rides a minimum-jerk move along the straight line to the pan
            self.trim(s, dt)
            c = s["carry"]
            frac, rate = minjerk(s["t_phase"], c["T"])
            cen = c["p0"] + frac * c["dist"] * c["dirn"]
            v = rate * c["dist"] * c["dirn"]
            for n in UAVS:
                refs[n] = form(n, cen + damp, s["z_cruise"] + s["trim"][n])
                vrefs[n] = np.array([v[0], v[1], 0.0])
            if s["t_phase"] > c["T"] + 1.5:
                s["phase"], s["t_phase"] = "ALIGN", 0.0
                s["align"] = c["p1"].copy()
        elif phase == "ALIGN":  # servo the bucket over the pan on what the cameras see (drift-free), gently
            self.trim(s, dt)
            aligned = False
            if pan_est and bucket_est:
                b1 = math.radians(RING_BEARING["UAV_1"])
                err = np.mean(pan_est, axis=0)[:2] + POUR_PRE * np.array([math.cos(b1), math.sin(b1)]) - bucket_xy
                s["align"] = s["align"] + np.clip(0.8 * err * dt, -0.06 * dt, 0.06 * dt)
                aligned = float(np.linalg.norm(err)) < 0.08
            for n in UAVS:
                refs[n] = form(n, s["align"] + damp, s["z_cruise"] + s["trim"][n])
                vrefs[n] = np.zeros(3)
            if (aligned and s["t_phase"] > 1.5) or s["t_phase"] > 8.0:
                s["phase"], s["t_phase"] = "POUR", 0.0
        elif phase == "POUR":  # one robot lowers its hook: the bucket tips; the thermal cameras say when the fire is out
            if pan_est and bucket_est:  # (a gentle correction only: the bucket swings towards the pan as its rope slackens)
                err = np.mean(pan_est, axis=0)[:2] - bucket_xy
                s["align"] = s["align"] + np.clip(0.8 * err * dt, -0.04 * dt, 0.04 * dt)
            frac, rate = minjerk(s["t_phase"], T_POUR)
            for n in UAVS:
                z = s["z_cruise"] + s["trim"][n] - (frac * POUR_DROP if n == "UAV_1" else 0.0)
                refs[n] = form(n, s["align"] + damp, z)
                vrefs[n] = np.array([0.0, 0.0, -rate * POUR_DROP if n == "UAV_1" else 0.0])
            if fire_seen and not any(fire_seen):
                s["fire_seen_out"] += dt
            else:
                s["fire_seen_out"] = 0.0
            if s["fire_seen_out"] > 0.5 or s["t_phase"] > 15.0:
                s["phase"], s["t_phase"] = "RECOVER", 0.0
                s["recover_drop"] = frac * POUR_DROP
        else:  # RECOVER / HOLD: level again and hover (the trim resumes once the lowered robot is back up)
            frac, rate = minjerk(s["t_phase"], T_RECOVER)
            if s["t_phase"] > T_RECOVER + 0.5:
                self.trim(s, dt)
            # still burning (the thermal cameras say so) and water left: another pour, at most three in all
            if s["t_phase"] > T_RECOVER + 3.0 and fire_seen and any(fire_seen) and s.get("pours", 1) < 3:
                s["pours"] = s.get("pours", 1) + 1
                s["phase"], s["t_phase"] = "ALIGN", 0.0
            for n in UAVS:
                z = s["z_cruise"] + s["trim"][n]
                vz = 0.0
                if n == "UAV_1":
                    z -= (1.0 - frac) * s["recover_drop"]
                    vz = rate * s["recover_drop"]
                refs[n] = form(n, s["align"] + damp, z)
                vrefs[n] = np.array([0.0, 0.0, vz])
        tension_mode = s["phase"] in ("TENSION", "LIFTOFF")
        out = {"thrust": {n: self.thrust(n, pos[n], vel[n], refs[n], vrefs[n], s, obs, dt,
                                         tz=(s["tz"][n] if tension_mode else None), vz_ref=s["vz_ref"][n]) for n in UAVS}}
        out["ugv"] = ugv_cmd
        return out


def main():
    ctl = Controller()
    sys.stdout.write("READY\n")
    sys.stdout.flush()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except ValueError:
            continue
        kind = msg.get("type")
        if kind == "reset":
            ctl.reset()
            reply = {"ok": True}
        elif kind == "act":
            try:
                reply = {"action": ctl.act(int(msg["t"]), msg["obs"])}
            except Exception as exc:  # noqa: BLE001
                reply = {"error": f"{type(exc).__name__}: {exc}"[:200]}
        elif kind == "close":
            sys.stdout.write(json.dumps({"ok": True}) + "\n")
            sys.stdout.flush()
            return
        else:
            reply = {"error": "unknown request"}
        sys.stdout.write(json.dumps(reply) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
