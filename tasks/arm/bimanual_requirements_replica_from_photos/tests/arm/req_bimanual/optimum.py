"""The optimum (public code): the cheapest BOM over the GIVEN catalogue that passes all 19 items of B-14 on the given pair
and lot, with the capability of the acceptance verdict (lotmodel.unit_capability: the delivered lot's weakest expected
unit, an uninspected vendor at the B-10 cap), so the optimum belongs to the verdict's own feasible set.

Exact search, per bus voltage of the brief:
  0. per arm and line the admissible SKUs (housing = pocket, voltage) minus those that fail for EVERY admissible mass of
     the modules the joint carries (screen / lower_util: interval bounds on the affine loads; the gripper by its exact
     mass-free check); this removes no feasible SKU, it only tightens the bounds of step 1.
  1. per arm a best-first branch and bound over the module lines distal to proximal (j8, j7, .., j1): a joint is checked
     as soon as every module whose mass it carries is chosen (with the contract's slots: when its own line is chosen),
     its result memoised per (SKU, the carried masses); the lower bound adds the cheapest admissible price of every line
     still open; a node's candidates are tried lazily, in price order, when the search reaches their cost. This yields
     the arm's feasible assignments in increasing cost.
  2. the two arms' streams combined in increasing module cost, best-first over all bus voltages at once; each
     combination gets its cheapest board cover and the cheapest PSU at its bus voltage that carries the summed current
     trace; the search stops when the module cost plus the cheapest board and PSU exceeds the best total. PSUs rated
     below a lower bound of every combination's peak current (current_floor) are dropped first (a bus without a PSU
     left is skipped).
Costs in integer cents; ties: price, then the quote order, line by line (left j8..j1, right j8..j1, boards, PSU).

    best = optimal_bom(acc, lot, cat, brief, buses=None)  -> {"bom", "cost", "bus_voltage_v", "peak_current_a", ..} | None
    table = utilisation_table(acc, lot, cat, brief, bom)   -> {"<arm>_<module>": {part_id: worst utilisation}}
"""
import heapq

import numpy as np

from . import lotmodel as LM
from .procedure import ARMS, JOINTS, MODULES, currents

ORDER = tuple(reversed(MODULES))


def admissible(acc, a, m, bus_v, cat):
    """Module part ids that fit the pocket of module m of arm a and accept the bus voltage, by (price, quote order)."""
    cls = acc.pair.arms[a]["pocket_class"][m]
    out = [(cat.cents(s["part_id"]), cat.index(s["part_id"]), s["part_id"]) for s in cat.modules
           if s["housing"] == cls and cat.on_bus(s["part_id"], bus_v)]
    return [pid for _, _, pid in sorted(out)]


def carried(acc, a):
    """{joint: modules whose mass that joint carries} (the slot body lies in the joint's subtree; a static slot loads
    no joint)."""
    return {j: tuple(m for m in MODULES if acc.carries[a][m][k]) for k, j in enumerate(JOINTS)}


def lower_util(acc, a, k, cap, box):
    """A lower bound of B-14 f's worst utilisation of joint index k of arm a over every mass vector of the modules it
    carries inside `box` = {module: (lo, hi) kg}: the loads are affine in the masses, so at every sample |tau| is at
    least the distance of 0 from the interval the box maps it to."""
    pay = acc.payload()
    mvb, hdb = acc.mvb[a], acc.hdb[a]

    def interval(base, cols):
        lo, hi = base.copy(), base.copy()
        for x, (ml, mh) in box.items():
            b = cols[x][:, k]
            lo += np.minimum(ml * b, mh * b)
            hi += np.maximum(ml * b, mh * b)
        return lo, hi

    def dist0(lo, hi):
        return np.maximum(np.maximum(lo, -hi), 0.0)

    qdd = acc.mv_qdd[a][:, k]
    jq = cap["J_out"] * qdd
    mlo, mhi = interval(acc.mv0[a][:, k] + pay * mvb["payload"][:, k], mvb)
    hlo, hhi = interval(acc.hd0[a][:, k] + pay * hdb["payload"][:, k], hdb)
    plo, phi = interval(acc.hd0[a][:, k] + float(acc.layer["proof"]) * pay * hdb["payload"][:, k], hdb)
    need = dist0(mlo, mhi) + np.abs(jq)
    avail = LM.available_torque_array(np.abs(acc.mv_qd[a][:, k]), cap)
    with np.errstate(divide="ignore", invalid="ignore"):
        move = float(np.where(avail > 0.0, need / avail, np.where(need > 0.0, np.inf, 0.0)).max())
    hold = dist0(hlo, hhi)
    rms = acc._duty_rms(dist0(mlo + jq, mhi + jq), hold)
    return max(move, float(hold.max()) / cap["tau_s"], float(dist0(plo, phi).max()) / cap["tau_s"], rms / cap["tau_c"])


def screen(acc, a, cand, caps):
    """The candidates of arm a's lines that can pass for SOME admissible masses of the modules their joint carries
    (exact: a dropped SKU fails for every mass vector, by lower_util with a margin far above the roundoff; the gripper
    by its own mass-free check). Distal lines first, so a joint's box uses the screened distal candidates."""
    carry = carried(acc, a)
    out = dict(cand)
    out["j8"] = [p for p in cand["j8"] if acc.module_item(a, "j8", caps[p], True, {})[0]]
    for j in reversed(JOINTS):
        if any(not out[x] for x in carry[j] if x != j):
            out[j] = []
            continue
        box = {x: (min(caps[p]["mass_kg"] for p in out[x]), max(caps[p]["mass_kg"] for p in out[x]))
               for x in carry[j] if x != j}
        out[j] = [p for p in cand[j] if lower_util(acc, a, JOINTS.index(j), caps[p],
                                                   dict(box, **({j: (caps[p]["mass_kg"],) * 2} if j in carry[j] else {})))
                  <= 1.0 + 1e-9]
    return out


def current_floor(acc, a, cand, caps):
    """A lower bound, at every move sample, of arm a's summed bus current over every assignment of its candidates: per
    line the least current of any candidate over the interval of its joint's load torque (affine in the carried
    masses; the current is convex in tau_m and least at the point of the interval nearest 0); the grippers exactly."""
    carry, pay = carried(acc, a), acc.payload()
    box = {x: (min(caps[p]["mass_kg"] for p in cand[x]), max(caps[p]["mass_kg"] for p in cand[x])) for x in MODULES}
    tot = np.zeros(len(acc.move_id))
    for k, j in enumerate(JOINTS):
        lo = acc.mv0[a][:, k] + pay * acc.mvb[a]["payload"][:, k]
        hi = lo.copy()
        for x in carry[j]:
            b = acc.mvb[a][x][:, k]
            lo += np.minimum(box[x][0] * b, box[x][1] * b)
            hi += np.maximum(box[x][0] * b, box[x][1] * b)
        qd, qdd = acc.mv_qd[a][:, k], acc.mv_qdd[a][:, k]
        least = None
        for p in cand[j]:
            jq = caps[p]["J_out"] * qdd
            I = currents(caps[p], np.maximum(np.maximum(lo + jq, -(hi + jq)), 0.0), qd, 0.0)
            least = I if least is None else np.minimum(least, I)
        tot += least
    return tot + min(currents(caps[p], acc.grip_torque(a), 0.0, 0.0) for p in cand["j8"])


class _Arm:
    """Best-first stream of one arm's feasible assignments at one bus voltage. A node's children are generated lazily in
    (price, quote order): a heap entry (bound, key, chosen, r) stands for candidate r and the dearer ones of the next line
    after `chosen` (its bound and key are those of candidate r, a lower bound of all of them), so a line is checked only
    when the search reaches its cost; the stream is the same as with eager expansion."""

    def __init__(self, acc, a, cand, caps, cat, memo):
        self.acc, self.a, self.cand, self.caps, self.cat, self.memo = acc, a, cand, caps, cat, memo
        self.carry = carried(acc, a)
        pos = {m: i for i, m in enumerate(ORDER)}
        self.check = [[] for _ in ORDER]
        for m in MODULES:
            need = (m,) + (self.carry[m] if m in self.carry else ())
            self.check[max(pos[x] for x in need)].append(m)
        self.lk = {m: [(cat.cents(p), cat.index(p)) for p in cand[m]] for m in ORDER}   # (cents, index), ascending
        cheapest = [self.lk[m][0][0] for m in ORDER]
        self.rest = [sum(cheapest[i:]) for i in range(len(ORDER) + 1)]
        self.heap = [(self.rest[0], (self.lk[ORDER[0]][0],), (), 0)]
        self.items = []                      # [(cents, key, {module: part_id})]
        self.traces, self.prot = {}, {}

    def _ok(self, m, assign):
        if m == "j8":
            key = (self.a, m, assign[m])
        else:
            key = (self.a, m, assign[m], tuple(self.caps[assign[x]]["mass_kg"] for x in self.carry[m]))
        if key not in self.memo:
            masses = {x: self.caps[assign[x]]["mass_kg"] for x in self.carry.get(m, ())}
            self.memo[key] = self.acc.module_item(self.a, m, self.caps[assign[m]], True, masses)[0]
        return self.memo[key]

    def get(self, i):
        """The i-th feasible assignment (cheapest first, ties by key), or None when there are fewer."""
        while len(self.items) <= i and self.heap:
            f, key, chosen, r = heapq.heappop(self.heap)
            pos = len(chosen)
            if pos == len(ORDER):
                self.items.append((f, key, dict(zip(ORDER, chosen))))
                continue
            m = ORDER[pos]
            lk = self.lk[m]
            g = f - lk[r][0] - self.rest[pos + 1]               # the cents of `chosen`
            if r + 1 < len(lk):                                 # the dearer candidates of this line, still untried
                heapq.heappush(self.heap, (g + lk[r + 1][0] + self.rest[pos + 1], key[:-1] + (lk[r + 1],), chosen, r + 1))
            assign = dict(zip(ORDER, chosen))
            assign[m] = self.cand[m][r]
            if not all(self._ok(x, assign) for x in self.check[pos]):
                continue
            child = chosen + (assign[m],)
            if pos + 1 < len(ORDER):                            # the child's cheapest candidate: the same bound
                heapq.heappush(self.heap, (f, key + (self.lk[ORDER[pos + 1]][0],), child, 0))
            else:
                heapq.heappush(self.heap, (f, key, child, -1))
        return self.items[i] if i < len(self.items) else None

    def current(self, i):
        """(summed bus current trace of the arm's modules, its peak, its floor) of the i-th assignment."""
        if i not in self.traces:
            assign = self.items[i][2]
            caps_a = {m: self.caps[assign[m]] for m in MODULES}
            tau = self.acc.loads(self.a, {m: caps_a[m]["mass_kg"] for m in MODULES}, self.acc.payload())[0]
            I = self.acc.arm_current(self.a, caps_a, tau)
            self.traces[i] = (I, float(I.max()), float(I.min()))
        return self.traces[i]

    def groups(self, i):
        """{(arm, protocol): number of modules} of the i-th assignment."""
        if i not in self.prot:
            self.prot[i] = self.cat.groups({self.a: self.items[i][2]})
        return self.prot[i]


def optimal_bom(acc, lot, cat, brief, buses=None):
    buses = tuple(float(v) for v in (buses if buses is not None else brief["bus_voltages_v"]))
    runs = []
    for bus in buses:
        cand = {a: {m: admissible(acc, a, m, bus, cat) for m in MODULES} for a in ARMS}
        boards = [b for b in cat.boards if cat.on_bus(b["part_id"], bus)]
        psus = sorted((cat.cents(p["part_id"]), cat.index(p["part_id"]), p) for p in cat.psus
                      if abs(float(p["output_v"]) - bus) < 1e-9)
        if not boards or not psus or any(not c for arm in cand.values() for c in arm.values()):
            continue
        caps = {pid: LM.unit_capability(pid, bus, lot, cat, brief) for arm in cand.values() for c in arm.values()
                for pid in c}
        cand = {a: screen(acc, a, cand[a], caps) for a in ARMS}
        if any(not c for arm in cand.values() for c in arm.values()):
            continue
        floor = float(sum(current_floor(acc, a, cand[a], caps) for a in ARMS).max())     # both arms at the same sample
        psus = [t for t in psus if float(t[2]["rated_current_a"]) >= floor - 1e-9]     # no other PSU carries any pair
        if not psus:
            continue
        memo = {}
        L, R = (_Arm(acc, a, cand[a], caps, cat, memo) for a in ARMS)
        if L.get(0) is not None and R.get(0) is not None:
            runs.append({"bus": bus, "L": L, "R": R, "psus": psus, "memo": memo, "seen": {(0, 0)},
                         "extra": min(cat.cents(b["part_id"]) for b in boards) + psus[0][0],
                         "rmax": max(float(p["rated_current_a"]) for _, _, p in psus)})
    # one best-first heap over every bus: (lower bound = module cents + the cheapest board and PSU, run, i, j)
    heap = [(r["extra"] + r["L"].get(0)[0] + r["R"].get(0)[0], k, 0, 0) for k, r in enumerate(runs)]
    heapq.heapify(heap)
    best, n_comb = None, 0
    while heap:
        lb, k, i, j = heapq.heappop(heap)
        if (best is not None and lb > best["cents"]) or not acc.budget_ok(lb):
            break
        n_comb += 1
        r = runs[k]
        L, R, bus = r["L"], r["R"], r["bus"]
        (tot, kl, ml), (tr, kr, mr) = L.get(i), R.get(j)
        modules = {"left": ml, "right": mr}
        cover = cat.cheapest_board_cover({**L.groups(i), **R.groups(j)}, bus)
        (IL, pl, fl), (IR, pr, fr) = L.current(i), R.current(j)
        if cover is not None and max(pl + fr, fl + pr) <= r["rmax"] + 1e-12:     # a lower bound of the summed peak
            peak = float((IL + IR).max())
            psu = next((p for _, _, p in r["psus"] if float(p["rated_current_a"]) >= peak - 1e-12), None)
            if psu is not None:
                cents = tot + tr + cover[0] + cat.cents(psu["part_id"])
                key = (cents, kl, kr, tuple((cat.cents(b), cat.index(b)) for b in cover[1]),
                       (cat.cents(psu["part_id"]), cat.index(psu["part_id"])))
                if acc.budget_ok(cents) and (best is None or key < best["key"]):
                    best = {"key": key, "cents": cents, "peak_current_a": peak,
                            "bom": {"bus_voltage_v": bus, "modules": {a: dict(modules[a]) for a in ARMS},
                                    "boards": list(cover[1]), "psu": psu["part_id"]}}
        for ni, nj in ((i + 1, j), (i, j + 1)):
            if (ni, nj) not in r["seen"] and L.get(ni) is not None and R.get(nj) is not None:
                r["seen"].add((ni, nj))
                heapq.heappush(heap, (r["extra"] + L.get(ni)[0] + R.get(nj)[0], k, ni, nj))
    if best is None:
        return None
    return {"bom": best["bom"], "cost": best["cents"] / 100.0, "bus_voltage_v": best["bom"]["bus_voltage_v"],
            "peak_current_a": best["peak_current_a"], "n_combined": n_comb, "n_memo": sum(len(r["memo"]) for r in runs)}


def utilisation_table(acc, lot, cat, brief, bom):
    """Worst utilisation of EVERY admissible SKU at every module line with the rest of `bom` installed (its masses), at
    the bom's bus voltage."""
    bus = float(bom["bus_voltage_v"])
    out = {}
    for a in ARMS:
        masses = {m: cat.module(bom["modules"][a][m])["mass_kg"] for m in MODULES}
        for m in MODULES:
            row = {}
            for pid in admissible(acc, a, m, bus, cat):
                ms = dict(masses)
                ms[m] = cat.module(pid)["mass_kg"]
                row[pid] = acc.module_item(a, m, LM.unit_capability(pid, bus, lot, cat, brief), True, ms)[1]["worst"]
            out["%s_%s" % (a, m)] = row
    return out
