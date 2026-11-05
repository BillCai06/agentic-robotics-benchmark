# furniture_assembly_from_drawings's environment image: MuJoCo with CPU rendering (the cameras' colour through MuJoCo's
# Filament renderer on Mesa's lavapipe, their depth through OSMesa), the agent's image tools, the Unitree G1 robot model
# (from MuJoCo Menagerie) and the room's decor (from the benchmark's asset dataset).
#
# `environment/` (less its `setup/`) is the sole build context and the final stage derives DIRECTLY from the official
# ALE base, at an exact version. Nothing hidden is here: the furniture cell's code, the grader and the reference program
# live in tests/ and solution/ (absent from the image); the givens and the practice lab are staged by
# environment/setup/setup.sh. The layout follows the arm chain's images (tasks/arm/*/environment/Dockerfile), leaner: no
# CAD or mesh toolchain.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:0.1.1

COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv

# OSMesa for software rendering (no GPU in the task's resources; EGL enumerates no device in the sandbox), Mesa's Vulkan
# drivers (lavapipe: the CPU Vulkan device MuJoCo's Filament renderer draws the cameras' colour on), the GL / X client
# libraries MuJoCo and OpenCV link even when headless, fontconfig / freetype for matplotlib, libgomp for the numerical
# libraries' threads.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libosmesa6 libgl1 libglu1-mesa libglew2.2 libxrender1 libxext6 libsm6 libxi6 \
        mesa-vulkan-drivers libvulkan1 \
        libfontconfig1 libfreetype6 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# World-readable venv at /opt/venv: setup (root), the agent (`user`), the oracle (`user`), the practice lab (`arbcell`)
# and verify (root) all use this one interpreter. UV_PYTHON_INSTALL_DIR must be world-traversable too (the default
# /root/.local/share/uv is not, and the venv then fails for every unprivileged account with a misleading
# ModuleNotFoundError). Python 3.12: the engine stages `ale_verify` into the `python3` it finds and refuses anything
# below 3.12. The ALE base ships a standalone CPython 3.12 at /opt/python, so uv links the venv against it.
ENV UV_PYTHON_INSTALL_DIR=/opt/uv-python
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
COPY requirements.txt /tmp/requirements.txt
# --no-cache: the wheel cache would otherwise stay inside this layer. The base image sets HOME=/home/user for root too:
# every build step that writes per-user files (uv's interpreter cache here, matplotlib's config below) points them away
# from the agent home, which must not hold root-owned entries the agent can neither write nor remove.
RUN export HOME=/root UV_CACHE_DIR=/tmp/uv-cache \
    && uv venv /opt/venv --python 3.12 \
    && uv pip install --no-cache --python /opt/venv -r /tmp/requirements.txt \
    && find /opt/venv -name '__pycache__' -type d -prune -exec rm -rf {} + \
    && chmod -R a+rX /opt/venv \
    && { [ ! -d /opt/uv-python ] || chmod -R a+rX /opt/uv-python; } \
    && rm -rf /tmp/uv-cache /tmp/requirements.txt

# The robot model: MuJoCo Menagerie's Unitree G1 (folder unitree_g1, BSD-3-Clause, LICENSE inside), fetched at the commit
# pinned in g1_assets.sha256 (a sparse, blob-filtered fetch: only that folder's files are downloaded) and checked file
# by file against the sha256 recorded there; a missing, extra or changed file fails the build. The cell compiles its
# robots from this copy (ARB_G1_DIR); environment/setup/setup.sh stages a copy into the agent home
# (~/given/robot/unitree_g1), and the verify stage (tests/test.sh, in its separate verifier sandbox started from this
# image) one into the grader's root-owned copy of the givens (/opt/arb/given/robot/unitree_g1).
COPY g1_assets.sha256 /opt/arb-assets/g1_assets.sha256
RUN set -eu; \
    list=/opt/arb-assets/g1_assets.sha256; \
    repo="$(sed -n 's/^# repo: *//p' "$list")"; \
    commit="$(sed -n 's/^# commit: *//p' "$list")"; \
    folder="$(sed -n 's/^# folder: *//p' "$list")"; \
    test -n "$repo"; test -n "$commit"; test -n "$folder"; \
    work="$(mktemp -d)"; \
    git -C "$work" init -q; \
    git -C "$work" remote add origin "$repo"; \
    git -C "$work" config core.sparseCheckout true; \
    echo "/$folder/" > "$work/.git/info/sparse-checkout"; \
    for try in 1 2 3; do \
        if git -C "$work" fetch -q --depth 1 --filter=blob:none origin "$commit" \
           && git -C "$work" -c advice.detachedHead=false checkout -q FETCH_HEAD; then break; fi; \
        if [ "$try" = 3 ]; then echo "[image] fetching $repo@$commit failed" >&2; exit 1; fi; \
        sleep 15; \
    done; \
    test "$(git -C "$work" rev-parse HEAD)" = "$commit"; \
    rm -rf "/opt/arb-assets/$folder"; \
    cp -r "$work/$folder" "/opt/arb-assets/$folder"; \
    rm -rf "$work"; \
    cd "/opt/arb-assets/$folder"; \
    grep -v '^#' "$list" | sha256sum --check --strict --quiet; \
    test "$(find . -type f | wc -l)" -eq "$(grep -vc '^#' "$list")"; \
    test -z "$(find . ! -type f ! -type d)"; \
    chown -R root:root /opt/arb-assets; \
    chmod -R u+rwX,go+rX,go-w /opt/arb-assets; \
    echo "[image] $folder @ $commit: $(find . -type f | wc -l) files verified"
ENV ARB_G1_DIR=/opt/arb-assets/unitree_g1

# ---- task assets: fetched at the dataset commit pinned in assets.sha256, checked file by file (docs/task-assets.md):
# render/room_decor, the room's decor (what the cameras see of the room: textured meshes of the furnished room of the
# task's demo, made from the owner's Unity packs, base maps at most 512 px, and the image light distilled from a
# path-traced render of it). It is moved to the root-owned /opt/arb-assets/room_decor, where the cell reads it
# (ARB_ROOM_DECOR_DIR, also the cell's default), and is not staged into the agent home; the list holds nothing else,
# so the emptied payload folder is removed.
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && mv /opt/task-assets/payload/render/room_decor /opt/arb-assets/room_decor \
    && rmdir /opt/task-assets/payload/render /opt/task-assets/payload \
    && chown -R root:root /opt/task-assets /opt/arb-assets/room_decor \
    && chmod -R u+rwX,go+rX,go-w /opt/task-assets /opt/arb-assets/room_decor \
    && echo "[image] room decor: $(find /opt/arb-assets/room_decor -type f | wc -l) files in /opt/arb-assets/room_decor"
ENV ARB_ROOM_DECOR_DIR=/opt/arb-assets/room_decor

# The practice lab's account (environment/setup/setup.sh starts the lab server as it): a system account with no login
# shell and no home, so the lab's cell and its state stay out of the agent's reach.
RUN useradd --system --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin arbcell

# The declared artifact path must exist even when nobody works (the engine captures declared artifacts after every
# episode, untouched validation passes included).
RUN mkdir -p /home/user/submission && chown -R user:user /home/user/submission

# Headless defaults: the classic renderer (the cameras' depth) goes through OSMesa; the Filament renderer needs no
# display (furniture_cell/render.py drops DISPLAY before it starts).
ENV MPLBACKEND=Agg
ENV MUJOCO_GL=osmesa
ENV PYOPENGL_PLATFORM=osmesa

# The agent harness starts the agent's shell with a PATH that has lost /opt/venv/bin, and a bare `python3` there is the
# base image's interpreter without the task's packages. Login shells read /etc/profile.d: put the venv first again.
RUN printf '%s\n' \
        '# furniture_assembly_from_drawings: the task venv first on PATH (the agent harness drops it).' \
        'case "$PATH" in /opt/venv/bin:*) ;; *) PATH="/opt/venv/bin:$PATH" ;; esac' \
        'export PATH' \
        > /etc/profile.d/10-arb-venv.sh \
    && chmod 0644 /etc/profile.d/10-arb-venv.sh

# Fail the build early if the toolchain is broken: the imports, an offscreen OSMesa render, the robot model compiles
# from the pinned copy, and the AGENT and LAB accounts can use the venv and read the model (the agent's own run also
# leaves it a ready matplotlib font cache, owned by the agent); finally, nothing in the agent home is root-owned. One
# import line per -c: a continuation line starting with `from` reads as a Dockerfile FROM to the final-stage lint.
RUN export HOME=/root MPLCONFIGDIR=/tmp/mpl-build \
    && python3 -c "import sys; assert sys.version_info >= (3, 12), sys.version; import mujoco, numpy, scipy, PIL, cv2, matplotlib; print('[image] import OK mujoco', mujoco.__version__, 'numpy', numpy.__version__, 'cv2', cv2.__version__, 'matplotlib', matplotlib.__version__)" \
    && python3 -c "import mujoco; m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\"0 0 1\"/><geom type=\"box\" size=\"0.1 0.1 0.1\"/></worldbody></mujoco>'); r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d); px = r.render(); assert int(px.max()) > 0; print('[image] OSMesa render OK', px.shape, int(px.max()))" \
    && python3 -c "import mujoco._render_filament as F, mujoco.experimental.studio.window; c = F.Context(F.ContextConfig(graphics_api=F.GraphicsApi.GRAPHICS_API_VULKAN)); print('[image] Filament on Vulkan (lavapipe) OK')" \
    && python3 -c "import os, mujoco; ms = [mujoco.MjModel.from_xml_path(os.path.join(os.environ['ARB_G1_DIR'], f)) for f in ('g1_with_hands.xml', 'g1_mjlab.xml')]; print('[image] G1 models compile, nbody', [m.nbody for m in ms])" \
    && runuser -u user -- env -u MPLCONFIGDIR /opt/venv/bin/python3 -c "import os, mujoco, numpy, scipy, PIL, cv2, matplotlib; m = mujoco.MjModel.from_xml_path(os.path.join(os.environ['ARB_G1_DIR'], 'g1_with_hands.xml')); print('[image] agent-UID import OK, G1 nbody', m.nbody)" \
    && runuser -u arbcell -- env MPLCONFIGDIR=/tmp/mpl-lab /opt/venv/bin/python3 -c "import os, mujoco, numpy, PIL, matplotlib; m = mujoco.MjModel.from_xml_path(os.path.join(os.environ['ARB_G1_DIR'], 'g1_mjlab.xml')); print('[image] lab-UID import OK, G1 nbody', m.nbody)" \
    && rm -rf /tmp/mpl-build /tmp/mpl-lab \
    && test -z "$(find /home/user -user root -print -quit)"

# The claude-code harness aborts a model stream after 300 s without a byte (Bun's HTTP idle timeout and the CLI's
# stream watchdog); both timers are raised to the gateway's own 1800 s ceiling. ENV is inherited by every process the
# guest service starts, the profile.d file covers the login shell the CLI is launched with. Nothing else reads them.
ENV BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800 CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000
RUN printf '%s\n' 'export BUN_CONFIG_HTTP_IDLE_TIMEOUT=1800' 'export CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000' \
        > /etc/profile.d/99-ale-idle-timeouts.sh \
    && chmod 0644 /etc/profile.d/99-ale-idle-timeouts.sh
