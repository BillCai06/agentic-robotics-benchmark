# Environment image for dexhand_shape_sorter_all_pieces. Final stage derives from the official ALE base.
# Contains the public SuperDex 1.0.0 simulator (double precision), the public SuperDex assets this task uses
# (shape_box prefab, Wuji Hand 2 right hand) and a finer collision SDF of the box baked at build time from the
# public mesh. setup/, verify/, oracle/ and the anchor are never copied into the image.
FROM ghcr.io/agentslastexam/container-ubuntu22-base:latest

ENV VIRTUAL_ENV=/opt/venv
ENV PATH=/opt/venv/bin:$PATH
ENV PYTHONNOUSERSITE=1
COPY requirements.txt superdex-requirements.txt /tmp/
RUN python3 -m venv /opt/venv \
    && /opt/venv/bin/python3 -m pip install --no-cache-dir --retries 30 --timeout 60 --upgrade pip \
    && /opt/venv/bin/python3 -m pip install --no-cache-dir --retries 30 --timeout 60 -r /tmp/requirements.txt \
    && /opt/venv/bin/python3 -m pip install --no-cache-dir --retries 30 --timeout 60 --no-deps -r /tmp/superdex-requirements.txt \
    && rm -f /tmp/requirements.txt /tmp/superdex-requirements.txt \
    && chmod -R a+rX /opt/venv

# ---- task assets: fetched at the dataset commit pinned in assets.sha256, checked file by file (docs/task-assets.md):
# the agent's images (setup/run.sh stages them into the home) and the public SuperDex assets this task uses (subset of
# the SuperDex asset tree, non-internal, see superdex_assets/*/LICENSE), moved to /opt/superdex_assets.
COPY fetch_assets.py assets.sha256 /opt/task-assets/
RUN python3 /opt/task-assets/fetch_assets.py /opt/task-assets/assets.sha256 /opt/task-assets/payload \
    && mv /opt/task-assets/payload/superdex_assets /opt/superdex_assets \
    && chmod -R a+rX /opt/task-assets /opt/superdex_assets
COPY bake_box_sdf.py /opt/sorter_tools/bake_box_sdf.py
# OPENBLAS_CORETYPE and NPY_DISABLE_CPU_FEATURES give every machine the same numerics. Without them OpenBLAS picks its
# kernels by CPU, and the replays of the same plan differed between hosts (sorter/numerics.py in the payload).
ENV SUPERDEX_ASSETS_PATH=/opt/superdex_assets SUPERDEX_PRECISION=double SORTER_DATA=/opt/sorter_data \
    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    OPENBLAS_CORETYPE=Haswell NPY_DISABLE_CPU_FEATURES="X86_V4 AVX512_ICL AVX512_SPR"
RUN python3 -B /opt/sorter_tools/bake_box_sdf.py /opt/sorter_data \
    && chmod -R a+rX /opt/superdex_assets /opt/sorter_data /opt/sorter_tools

# The declared artifact path exists even when nobody works.
RUN mkdir -p /home/user/submission/plans && chown -R user:user /home/user/submission

# Build-time checks: double-precision SuperDex imports for root and for the agent account; no job scheduler.
RUN python3 -c "import os; os.environ['SUPERDEX_PRECISION']='double'; import superdex.physics as sp, superdex.robotics; assert sp.USE_DOUBLE_PRECISION; print('[image] superdex fp64 OK')" \
    && python3 -c "import site, sys; assert sys.prefix != sys.base_prefix" \
    && runuser -u user -- /opt/venv/bin/python3 -c "import superdex.physics, trimesh, h5py, scipy; print('[image] agent-UID import OK')" \
    && ! command -v cron && ! command -v atd && ! command -v crond
# Build-time numerics check: the CPU must run SuperDex's AVX2 + FMA code, and OpenBLAS / NumPy must use the pinned paths
# (the grader's full check, with the reference values, is sorter/numerics.py).
RUN python3 -c "import ctypes, glob, numpy as np; from numpy._core._multiarray_umath import __cpu_dispatch__ as D, __cpu_features__ as F; \
lib = ctypes.CDLL(glob.glob(np.__file__.rsplit('/', 1)[0] + '.libs/libscipy_openblas64_*.so')[0]); \
fn = lib.scipy_openblas_get_corename64_; fn.restype = ctypes.c_char_p; core = fn().decode(); \
on = [k for k in D if F.get(k)]; print('[image] OpenBLAS kernels', core, '- NumPy dispatch', on); \
assert core == 'Haswell' and on == ['X86_V3'], 'numerics differ from the reference: OpenBLAS %s, NumPy %s' % (core, on)"
