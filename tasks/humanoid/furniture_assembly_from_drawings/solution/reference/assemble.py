"""Oracle of furniture_assembly_from_drawings: submitted with the oracle token, this program's cells are privileged (the
nominal layout) and it asks the cell to run the privileged reference assembler (furniture_cell.reference: true part
poses, the two G1 humanoids driven through the cell's own controllers, safety watch and physical joints) inside one
call. It first takes one frame of each public camera, so every validation also exercises the agents' camera path (a
camera that cannot render fails the oracle, and with it the validation)."""

CAMERAS = ("ceiling_west", "ceiling_east", "a/head", "b/head")


def solve(cell):
    info = cell.describe()
    print("[oracle] cell %s, cameras %s" % (info.get("api"), sorted(info.get("cameras", {}))), flush=True)
    for name in CAMERAS:
        f = cell.get_camera(name)
        print("[oracle] camera %s: %s" % (name, {k: v for k, v in f.items() if not k.endswith("_b64")}), flush=True)
    r = cell._call("reference_assemble")
    print("[oracle] reference: %s" % (r,), flush=True)
