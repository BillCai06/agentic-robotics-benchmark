"""A joint-module catalogue as DATA (public, number-free code). The grader loads the truth (the generator-side truth of
the datasheet pages, the erratum and the quote); an agent may load its own transcription of the pages
(schemas.catalogue_schema). Format bimanual-req-catalogue/1, SI units:

  vendors: {code: {inspected: bool}}       a vendor without an incoming-inspection report is not inspected (brief B-10)
  modules: [{part_id, vendor, housing (large | medium | small), protocol, voltage_range_v [lo, hi],
             points [{voltage_v, no_load_speed_rad_s, speed_at_peak_rad_s}] (one or two voltage points), peak_torque_nm,
             rated_torque_nm, idle_current_a, copper_loss_rated_w, output_inertia_kgm2, mass_kg, price_usd}]
  boards:  [{part_id, protocols [..], channels, modules_per_channel, input_range_v [lo, hi], price_usd}]
  psus:    [{part_id, output_v, rated_current_a, price_usd}]

List order = the quote's order: ties of the optimum and the repair are broken by price, then by this order. Speeds at a
voltage: linear interpolation / extrapolation between the two points (one point: in proportion to the voltage); torques
do not depend on the voltage. Prices are summed in integer cents. Roles (the quote's and bom.json's): module, bus_board,
psu.

Boards (brief B-14 i): every (arm, protocol) group of a BOM's modules needs its own channel of a board that serves the
protocol, with the bus voltage inside the board's input range and the group's module count <= modules_per_channel; a
board offers `channels` channels, each serving any of its protocols.
"""
import copy
import itertools
import json

ROLES = ("module", "bus_board", "psu")
V_TOL = 1e-9


def _in(v, rng):
    return float(rng[0]) - V_TOL <= float(v) <= float(rng[1]) + V_TOL


class Catalogue:
    def __init__(self, data):
        if isinstance(data, str):
            with open(data) as fh:
                data = json.load(fh)
        self.data = copy.deepcopy(data)
        self.modules = list(self.data.get("modules", []))
        self.boards = list(self.data.get("boards", []))
        self.psus = list(self.data.get("psus", []))
        self.vendors = dict(self.data.get("vendors", {}))
        self._by = {}
        for role, items in zip(ROLES, (self.modules, self.boards, self.psus)):
            for it in items:
                if it["part_id"] in self._by:
                    raise ValueError("part %r listed twice in the catalogue" % it["part_id"])
                d = dict(it)
                d["role"] = role
                self._by[it["part_id"]] = d
        self.order = [x["part_id"] for x in self.modules + self.boards + self.psus]
        self._idx = {p: i for i, p in enumerate(self.order)}
        self._si = {}
        self._cover = {}

    def has(self, pid):
        return pid in self._by

    def get(self, pid):
        return self._by[pid]

    def role(self, pid):
        return self._by[pid]["role"]

    def roles(self):
        return {p: self._by[p]["role"] for p in self.order}

    def price(self, pid):
        return float(self._by[pid]["price_usd"])

    def cents(self, pid):
        return int(round(100.0 * float(self._by[pid]["price_usd"])))

    def index(self, pid):
        return self._idx[pid]

    def inspected(self, vendor):
        return bool(self.vendors.get(vendor, {}).get("inspected", False))

    def module(self, pid):
        """The SI record of a module (cached)."""
        if pid not in self._si:
            s = self._by[pid]
            if s["role"] != "module":
                raise KeyError("%r is a %s, not a module" % (pid, s["role"]))
            pts = sorted((float(p["voltage_v"]), float(p["no_load_speed_rad_s"]), float(p["speed_at_peak_rad_s"]))
                         for p in s["points"])
            self._si[pid] = {"part_id": pid, "vendor": s["vendor"], "housing": s["housing"], "protocol": s["protocol"],
                             "vrange": (float(s["voltage_range_v"][0]), float(s["voltage_range_v"][1])), "points": pts,
                             "peak_nm": float(s["peak_torque_nm"]), "rated_nm": float(s["rated_torque_nm"]),
                             "idle_a": float(s["idle_current_a"]), "cu_loss_w": float(s["copper_loss_rated_w"]),
                             "J_out": float(s["output_inertia_kgm2"]), "mass_kg": float(s["mass_kg"]),
                             "price": float(s["price_usd"]), "inspected": self.inspected(s["vendor"])}
        return self._si[pid]

    @staticmethod
    def speeds_at(m, v):
        """(no-load speed, speed at peak torque) of module record `m` at the module voltage v: linear interpolation and
        extrapolation between the datasheet's voltage points; one point: in proportion to the voltage."""
        pts = m["points"]
        v = float(v)
        if len(pts) == 1:
            v1, w0, wk = pts[0]
            return w0 * v / v1, wk * v / v1
        i = 0
        while i < len(pts) - 2 and v > pts[i + 1][0]:
            i += 1
        (va, w0a, wka), (vb, w0b, wkb) = pts[i], pts[i + 1]
        a = (v - va) / (vb - va)
        return w0a + a * (w0b - w0a), wka + a * (wkb - wka)

    def on_bus(self, pid, bus_v):
        """The module's (or board's) voltage range contains the bus voltage."""
        it = self._by[pid]
        return _in(bus_v, it["voltage_range_v"] if it["role"] == "module" else it["input_range_v"])

    def groups(self, modules):
        """{(arm, protocol): number of modules} of a BOM's module lines {arm: {joint: part_id}}."""
        out = {}
        for arm in sorted(modules):
            for pid in modules[arm].values():
                key = (arm, self.module(pid)["protocol"])
                out[key] = out.get(key, 0) + 1
        return out

    def _channels(self, board_ids, bus_v):
        ch = []
        for bid in board_ids:
            b = self._by.get(bid)
            if b is None or b["role"] != "bus_board" or not _in(bus_v, b["input_range_v"]):
                continue
            ch += [(frozenset(b["protocols"]), int(b["modules_per_channel"]))] * int(b["channels"])
        return ch

    @staticmethod
    def _match(channels, groups):
        """Every group gets its own channel (bipartite matching, augmenting paths)."""
        owner = [-1] * len(channels)

        def place(g, seen):
            p, n = groups[g]
            for c, (prots, cap) in enumerate(channels):
                if c in seen or p not in prots or n > cap:
                    continue
                seen.add(c)
                if owner[c] < 0 or place(owner[c], seen):
                    owner[c] = g
                    return True
            return False

        return len(groups) <= len(channels) and all(place(g, set()) for g in range(len(groups)))

    def boards_cover(self, board_ids, groups, bus_v):
        """Brief B-14 i for the board list (with repeats) and the groups {(arm, protocol): n}."""
        g = [(p, n) for (_, p), n in sorted(groups.items())]
        return self._match(self._channels(board_ids, bus_v), g)

    def cheapest_board_cover(self, groups, bus_v):
        """(cents, [board ids in quote order, with repeats]) of the cheapest board multiset covering `groups` at bus_v
        (ties: the quote order of the boards), or None."""
        key = (tuple(sorted(groups.items())), round(float(bus_v), 9))
        if key in self._cover:
            return self._cover[key]
        ok = [b for b in self.boards if _in(bus_v, b["input_range_v"])]
        best = None
        for counts in itertools.product(range(len(groups) + 1), repeat=len(ok)):
            if not any(counts):
                continue
            ids = [b["part_id"] for b, c in zip(ok, counts) for _ in range(c)]
            if not self.boards_cover(ids, groups, bus_v):
                continue
            k = (sum(self.cents(i) for i in ids), [self.index(i) for i in ids])
            if best is None or k < best[0]:
                best = (k, ids)
        self._cover[key] = None if best is None else (best[0][0], best[1])
        return self._cover[key]

    def bom_cents(self, bom):
        c = sum(self.cents(p) for arm in bom["modules"].values() for p in arm.values() if self.has(p))
        c += sum(self.cents(b) for b in bom.get("boards", []) if self.has(b))
        if bom.get("psu") and self.has(bom["psu"]):
            c += self.cents(bom["psu"])
        return c

    def bom_cost(self, bom):
        return self.bom_cents(bom) / 100.0
