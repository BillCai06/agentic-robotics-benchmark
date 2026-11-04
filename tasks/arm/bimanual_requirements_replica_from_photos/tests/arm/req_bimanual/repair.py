"""The deterministic engineering-change repair and the economy criterion (brief B-13, C5; public code over a GIVEN
catalogue).

While an item fails, replace the failing line by the cheapest line of the catalogue that makes it pass given the rest
(module lines: left j8..j1, then right j8..j1; then the board set; then the PSU; ties by price, then the quote order) and
add the brief's eco cost per replaced line. A board-set replacement counts the board lines it adds or whose count it
changes (at least one). The bus voltage is never changed by a repair: a module line with no passing module at that
voltage, no board cover or no PSU makes the BOM irreparable (J_eff = inf, economy 0). The budget is not a line.

    r = repair(acc, bom, lot, cat, brief)        -> {"bom", "replaced", "J", "J_eff", "log", "repairable", ..}
    economy(J_star, J_eff, ceiling, band_usd)    clamp(1 - (J_eff - J*) / band, 0, 1) x taper(J_eff vs the ceiling);
                                                 band = band_lines x eco (layer.brief_constants)
"""
import copy
import math

from . import lotmodel as LM
from .procedure import ARMS, MODULES

REPAIR_LINES = tuple((a, m) for a in ARMS for m in reversed(MODULES))


def _irreparable(b, replaced, J0, log, why):
    return {"bom": b, "replaced": replaced, "J": J0, "J_eff": math.inf, "log": log + [why], "repairable": False}


def repair(acc, bom, lot, cat, brief, max_steps=64):
    eco_cents = int(round(100.0 * float(brief["eco"])))
    b = copy.deepcopy(bom)
    b.setdefault("boards", [])
    bus = float(b["bus_voltage_v"])
    replaced, log, J0 = 0, [], None
    by_price = [s["part_id"] for s in sorted(cat.modules, key=lambda s: (cat.cents(s["part_id"]), cat.index(s["part_id"])))]
    for _ in range(max_steps):
        res = acc.evaluate(b, lot, cat, brief)
        if J0 is None:
            J0 = res["cost"]
        failing = [(a, m) for a, m in REPAIR_LINES if not res["items"]["%s_%s" % (a, m)]]
        if failing:
            a, m = failing[0]
            masses = {x: cat.module(b["modules"][a][x])["mass_kg"] for x in MODULES}
            for pid in by_price:
                if pid == b["modules"][a][m]:
                    continue
                ms = dict(masses)
                ms[m] = cat.module(pid)["mass_kg"]
                if acc.module_item(a, m, LM.unit_capability(pid, bus, lot, cat, brief), cat.on_bus(pid, bus), ms)[0]:
                    log.append("%s_%s: %s -> %s" % (a, m, b["modules"][a][m], pid))
                    b["modules"][a][m] = pid
                    replaced += 1
                    break
            else:
                return _irreparable(b, replaced, J0, log, "%s_%s: no module passes at %g V" % (a, m, bus))
            continue
        if not res["items"]["boards"]:
            cover = cat.cheapest_board_cover(cat.groups(b["modules"]), bus)
            if cover is None:
                return _irreparable(b, replaced, J0, log, "boards: no cover at %g V" % bus)
            old, new = {}, {}
            for x in b["boards"]:
                old[x] = old.get(x, 0) + 1
            for x in cover[1]:
                new[x] = new.get(x, 0) + 1
            n = max(1, sum(1 for x in new if new[x] != old.get(x, 0)))
            log.append("boards: %s -> %s (%d line%s)" % (b["boards"], cover[1], n, "" if n == 1 else "s"))
            b["boards"] = list(cover[1])
            replaced += n
            continue
        if not res["items"]["psu"]:
            ok = [p["part_id"] for p in sorted(cat.psus, key=lambda p: (cat.cents(p["part_id"]), cat.index(p["part_id"])))
                  if abs(float(p["output_v"]) - bus) < 1e-9 and float(p["rated_current_a"]) >= res["peak_current_a"] - 1e-12]
            if not ok:
                return _irreparable(b, replaced, J0, log, "psu: none carries %.3f A at %g V" % (res["peak_current_a"], bus))
            log.append("psu: %s -> %s" % (b.get("psu"), ok[0]))
            b["psu"] = ok[0]
            replaced += 1
            continue
        break
    res = acc.evaluate(b, lot, cat, brief)
    if not all(res["items"][i] for i in res["items"] if i != "budget"):
        return _irreparable(b, replaced, J0, log, "not repaired within %d steps" % max_steps)
    return {"bom": b, "replaced": replaced, "J": J0, "J_eff": (cat.bom_cents(b) + eco_cents * replaced) / 100.0,
            "log": log, "repairable": True, "final_items": res["items"]}


def economy(J_star, J_eff, ceiling, band_usd):
    if not math.isfinite(J_eff) or J_eff <= 0:
        return 0.0
    over = round(float(J_eff) - float(J_star), 6)
    ratio = min(1.0, max(0.0, 1.0 - over / float(band_usd)))
    if J_eff <= float(ceiling) + 1e-9:
        return ratio
    return ratio * max(0.0, 1.0 - (J_eff - float(ceiling)) / (0.10 * float(ceiling)))
