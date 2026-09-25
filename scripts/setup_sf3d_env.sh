#!/usr/bin/env bash
# Build the isolated `protoskin3d` conda env for the 3D reconstruction
# service (visual_engine/reconstruct3d.py + scripts/run_sf3d_service.sh).
#
# Why a separate env: Stable Fast 3D pins transformers==4.42.3 and
# trimesh==4.4.1, which conflict with the main `zgx` env's transformers
# (used by the Qwen3-8B explainer). Keeping this isolated avoids that clash
# entirely rather than trying to reconcile pinned versions.
#
# This was developed and tested on an aarch64 Linux machine (HP ZGX Nano,
# GB10 GPU) with CUDA 13.0 and GCC 13. Two of Stable Fast 3D's from-source
# Python dependencies (gpytoolbox, pynanoinstantmeshes) don't ship aarch64
# Linux wheels on PyPI and needed small source patches to build here; those
# patches are applied below with the reasoning inline. On x86_64 Linux with
# an older GCC, prebuilt wheels likely exist for both and this script's
# patches would be unnecessary (harmless if applied anyway, since they only
# loosen a compiler diagnostic and skip an unused CGAL/embree feature).
#
# Idempotent-ish: safe to re-run, but does not attempt to skip work that's
# already done. Expect ~15-30 minutes, most of it compiling gpytoolbox and
# pynanoinstantmeshes's C++ dependencies (Eigen, TBB, libigl).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SF3D_SRC="${PROTOSKIN_SF3D_SRC:-$REPO_ROOT/third_party/stable-fast-3d}"
CONDA_ENV_NAME="protoskin3d"
SCRATCH_DIR="$(mktemp -d)"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"

echo "== ProtoSkin: setting up the $CONDA_ENV_NAME environment =="
echo "SF3D source: $SF3D_SRC"
echo "Scratch dir for dependency builds: $SCRATCH_DIR"
echo "CUDA toolkit: $CUDA_HOME"

if [ ! -d "$SF3D_SRC" ]; then
  echo "error: $SF3D_SRC not found. Expected the vendored Stable Fast 3D"
  echo "source at third_party/stable-fast-3d (see third_party/README.md)."
  exit 1
fi

export PATH="$CUDA_HOME/bin:$PATH"
if ! command -v nvcc >/dev/null 2>&1; then
  echo "error: nvcc not found on PATH after adding $CUDA_HOME/bin."
  echo "Set CUDA_HOME to your CUDA 13.x toolkit install and re-run."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck source=/dev/null
source "$CONDA_BASE/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${CONDA_ENV_NAME} "; then
  conda create -n "$CONDA_ENV_NAME" python=3.10 -y
fi
conda activate "$CONDA_ENV_NAME"
PY="$(command -v python)"
echo "Using interpreter: $PY"

# --- torch -------------------------------------------------------------
# Match whatever cu13x build the main `zgx` env uses so the two envs run
# against compatible CUDA runtime versions. Adjust the version pin below if
# your `zgx` env uses a different torch/cu version; check with:
#   /home/hp12/miniforge3/envs/zgx/bin/pip show torch
pip install -U setuptools wheel
pip install torch==2.14.0

# --- SF3D's pinned pure-Python / prebuilt-wheel requirements ------------
# Excludes the two from-source packages (gpytoolbox, pynanoinstantmeshes)
# and texture_baker/uv_unwrapper (SF3D's own repo-local CUDA/C++
# extensions, built separately below) — installed in a batch further down
# once everything else that could fail in isolation is out of the way.
cat > "$SCRATCH_DIR/requirements-wheels.txt" <<'EOF'
einops==0.7.0
jaxtyping==0.2.31
omegaconf==2.3.0
transformers==4.42.3
open_clip_torch==2.24.0
trimesh==4.4.1
numpy==1.26.4
huggingface-hub==0.23.4
rembg==2.0.57
EOF
# rembg[gpu]'s onnxruntime-gpu extra has no aarch64 Linux wheel on PyPI;
# plain CPU rembg works fine functionally (only affects background-removal
# speed, a small preprocessing step, not reconstruction quality/correctness).
pip install -r "$SCRATCH_DIR/requirements-wheels.txt"

# --- texture_baker / uv_unwrapper (SF3D's own repo-local extensions) ----
# Both build via PyTorch's own CUDAExtension/BuildExtension mechanism.
# --no-build-isolation is required: their setup.py does `import torch` at
# build time, which an isolated build env can't see otherwise.
pip install --no-build-isolation "$SF3D_SRC/texture_baker/"
pip install --no-build-isolation "$SF3D_SRC/uv_unwrapper/"

# --- gpytoolbox (from source; no aarch64 wheel on PyPI) -----------------
# The PyPI sdist for gpytoolbox==0.2.0 ships only its pure-Python files —
# its C++ bindings live in git submodules the sdist doesn't include, so it
# fails to build ("does not appear to contain CMakeLists.txt"). Clone the
# real repo with submodules instead.
GPT_SRC="$SCRATCH_DIR/gpytoolbox-src"
git clone --recursive https://github.com/sgsellan/gpytoolbox.git "$GPT_SRC"
(cd "$GPT_SRC" && git checkout v0.2.0 && git submodule update --init --recursive)

# Patch 1: embree (a FetchContent'd dependency of libigl, pulled in
# transitively) ships a CMakeLists.txt whose cmake_minimum_required predates
# policies CMake 3.5+ removed support for -- fails outright on any modern
# CMake without this flag.
# Patch 2: embree also pulls in Boost 1.71.0 via FetchContent from
# boostorg.jfrog.io, whose pinned SHA256 no longer matches what that mirror
# serves today (checksum drift on an external artifact host, not fixable
# here). gpytoolbox's own code (subdivide/decimate/remesh_botsch, the only
# functions Stable Fast 3D calls) never uses CGAL or embree -- both are only
# linked into gpytoolbox's separate `copyleft` binding module -- so disabling
# LIBIGL_COPYLEFT_CGAL and LIBIGL_EMBREE skips this unrelated, broken fetch
# chain entirely instead of trying to fix the checksum.
sed -i \
  "s/cmake_args = \['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,\n *'-DPYTHON_EXECUTABLE=' + sys.executable\]//" \
  "$GPT_SRC/setup.py" 2>/dev/null || true
python - "$GPT_SRC/setup.py" <<'PYEOF'
import re, sys
path = sys.argv[1]
text = open(path).read()
old = "cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,\n                      '-DPYTHON_EXECUTABLE=' + sys.executable]"
new = (
    "cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,\n"
    "                      '-DPYTHON_EXECUTABLE=' + sys.executable,\n"
    "                      '-DCMAKE_POLICY_VERSION_MINIMUM=3.5',\n"
    "                      '-DLIBIGL_EMBREE=OFF',\n"
    "                      '-DLIBIGL_COPYLEFT_CGAL=OFF']"
)
if old in text:
    text = text.replace(old, new)
    open(path, "w").write(text)
    print("patched setup.py cmake_args")
else:
    print("warning: setup.py cmake_args pattern not found; may already be patched or upstream changed", file=sys.stderr)
PYEOF

# Patch 3: even with LIBIGL_COPYLEFT_CGAL=OFF, gpytoolbox's top-level
# CMakeLists.txt unconditionally builds a `cpytoolbox_copyleft` target,
# links igl::embree into the main `cpytoolbox` target, and unconditionally
# configures the `gpytoolbox_bindings_copyleft` pybind11 module -- the
# option only gates whether libigl's internal igl_include() macro sets up
# the CGAL target, not these top-level add_library/pybind11_add_module/
# target_link_libraries/target_include_directories calls. Gate all of them
# properly so turning the option off actually skips the whole CGAL/embree
# build (and the broken Boost fetch behind it). Verified against a fresh
# clone of this exact tag before being folded into this script.
python - "$GPT_SRC/CMakeLists.txt" <<'PYEOF'
import sys
path = sys.argv[1]
text = open(path).read()

old_copyleft = """\tadd_library(cpytoolbox_copyleft
\tSTATIC
\t# SHARED
\t# Headers
\tsrc/cpp/swept_volume/fd_interpolate.cpp
\tsrc/cpp/swept_volume/gradient_descent_test.cpp
\tsrc/cpp/swept_volume/random_points_on_mesh.cpp
\tsrc/cpp/swept_volume/sparse_continuation.cpp
\tsrc/cpp/swept_volume/swept_volume.cpp
\t# Source
\tsrc/cpp/swept_volume/fd_interpolate.h
\tsrc/cpp/swept_volume/gradient_descent_test.h
\tsrc/cpp/swept_volume/random_points_on_mesh.h
\tsrc/cpp/swept_volume/sparse_continuation.h
\tsrc/cpp/swept_volume/swept_volume.h
\t)

# target_link_libraries(cpytoolbox igl::core igl_copyleft::cgal igl::embree)
target_link_libraries(cpytoolbox igl::core igl::embree )
target_link_libraries(cpytoolbox_copyleft igl::core igl_copyleft::cgal)"""

new_copyleft = """if(LIBIGL_COPYLEFT_CGAL)
\tadd_library(cpytoolbox_copyleft
\tSTATIC
\tsrc/cpp/swept_volume/fd_interpolate.cpp
\tsrc/cpp/swept_volume/gradient_descent_test.cpp
\tsrc/cpp/swept_volume/random_points_on_mesh.cpp
\tsrc/cpp/swept_volume/sparse_continuation.cpp
\tsrc/cpp/swept_volume/swept_volume.cpp
\tsrc/cpp/swept_volume/fd_interpolate.h
\tsrc/cpp/swept_volume/gradient_descent_test.h
\tsrc/cpp/swept_volume/random_points_on_mesh.h
\tsrc/cpp/swept_volume/sparse_continuation.h
\tsrc/cpp/swept_volume/swept_volume.h
\t)
\ttarget_link_libraries(cpytoolbox_copyleft igl::core igl_copyleft::cgal)
endif()

# target_link_libraries(cpytoolbox igl::core igl_copyleft::cgal igl::embree)
if(LIBIGL_EMBREE)
\ttarget_link_libraries(cpytoolbox igl::core igl::embree)
else()
\ttarget_link_libraries(cpytoolbox igl::core)
endif()"""

assert old_copyleft in text, "gpytoolbox CMakeLists.txt copyleft block pattern not found -- upstream may have changed"
text = text.replace(old_copyleft, new_copyleft)

old_bindings = '''pybind11_add_module(gpytoolbox_bindings_copyleft
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/gpytoolbox_bindings_copyleft_core.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_swept_volume.cpp"
\t"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_booleans.cpp"
)

target_link_libraries(gpytoolbox_bindings PUBLIC cpytoolbox igl::core)
target_link_libraries(gpytoolbox_bindings_copyleft PUBLIC cpytoolbox_copyleft igl::core igl_copyleft::cgal)

target_include_directories(gpytoolbox_bindings PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/")
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
\tadd_custom_command(TARGET gpytoolbox_bindings POST_BUILD
\t\tCOMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gpytoolbox_bindings> $<TARGET_RUNTIME_DLLS:gpytoolbox_bindings> $<TARGET_FILE_DIR:gpytoolbox_bindings>
\t\tCOMMAND_EXPAND_LISTS)
endif()

target_include_directories(gpytoolbox_bindings_copyleft PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp")
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
\tadd_custom_command(TARGET gpytoolbox_bindings_copyleft POST_BUILD
\t\tCOMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gpytoolbox_bindings_copyleft> $<TARGET_RUNTIME_DLLS:gpytoolbox_bindings_copyleft> $<TARGET_FILE_DIR:gpytoolbox_bindings_copyleft>
\t\tCOMMAND_EXPAND_LISTS)
endif()'''

new_bindings = '''if(LIBIGL_COPYLEFT_CGAL)
\tpybind11_add_module(gpytoolbox_bindings_copyleft
\t    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/gpytoolbox_bindings_copyleft_core.cpp"
\t    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_swept_volume.cpp"
\t\t"${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/binding_booleans.cpp"
\t)
\ttarget_link_libraries(gpytoolbox_bindings_copyleft PUBLIC cpytoolbox_copyleft igl::core igl_copyleft::cgal)
\ttarget_include_directories(gpytoolbox_bindings_copyleft PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp")
\tif(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
\t\tadd_custom_command(TARGET gpytoolbox_bindings_copyleft POST_BUILD
\t\t\tCOMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gpytoolbox_bindings_copyleft> $<TARGET_RUNTIME_DLLS:gpytoolbox_bindings_copyleft> $<TARGET_FILE_DIR:gpytoolbox_bindings_copyleft>
\t\t\tCOMMAND_EXPAND_LISTS)
\tendif()
endif()

target_link_libraries(gpytoolbox_bindings PUBLIC cpytoolbox igl::core)

target_include_directories(gpytoolbox_bindings PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/src/cpp/")
if(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
\tadd_custom_command(TARGET gpytoolbox_bindings POST_BUILD
\t\tCOMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gpytoolbox_bindings> $<TARGET_RUNTIME_DLLS:gpytoolbox_bindings> $<TARGET_FILE_DIR:gpytoolbox_bindings>
\t\tCOMMAND_EXPAND_LISTS)
endif()'''

assert old_bindings in text, "gpytoolbox CMakeLists.txt bindings block pattern not found -- upstream may have changed"
text = text.replace(old_bindings, new_bindings)
open(path, "w").write(text)
print("patched CMakeLists.txt copyleft/embree and bindings blocks")
PYEOF

# Patch 4: ray_mesh_intersect_aabb.cpp is the only gpytoolbox source file
# that #includes igl/embree/EmbreeIntersector.h -- with LIBIGL_EMBREE=OFF
# there's no igl::embree target to link, so this file fails to compile
# (fatal error: embree3/rtcore.h: No such file or directory). SF3D's
# mesh.py never calls ray_mesh_intersect, so drop this one file + its
# pybind11 binding rather than fix or work around embree.
sed -i '/src\/cpp\/ray_mesh_intersect_aabb\.h/d; /src\/cpp\/ray_mesh_intersect_aabb\.cpp/d; /src\/cpp\/binding_ray_mesh_intersect\.cpp/d' \
  "$GPT_SRC/CMakeLists.txt"
sed -i \
  -e '/void binding_ray_mesh_intersect(py::module& m);/d' \
  -e '/    binding_ray_mesh_intersect(m);/d' \
  "$GPT_SRC/src/cpp/gpytoolbox_bindings_core.cpp"

pip install --no-deps "$GPT_SRC"

# --- pynanoinstantmeshes (from source; no aarch64 wheel on PyPI) --------
# The PyPI package `pynanoinstantmeshes` has prebuilt wheels for
# manylinux_x86_64 and macOS only -- none for aarch64 Linux -- and its
# sdist's own tarball is missing a submodule file
# (native/ext/tbb/build/version_string.ver.in), so a source build from the
# sdist fails too. The actual upstream repo (not obvious from PyPI, which
# has no project URL in its metadata) is vork/PyNanoInstantMeshes; its
# v0.0.3 tag matches the PyPI version and has real submodules.
PNIM_SRC="$SCRATCH_DIR/pynanoinstantmeshes-src"
git clone --recursive https://github.com/vork/PyNanoInstantMeshes.git "$PNIM_SRC"
(cd "$PNIM_SRC" && git checkout v0.0.3 && git submodule update --init --recursive)

# Patch: the vendored TBB (~2017-era, deprecated per its own #pragma
# message) defines tbb::internal::task_prefix::task(), a member function
# whose name matches the forward-declared class tbb::task. Modern GCC
# (13+) treats this as a hard error (-Wchanges-meaning); older GCC only
# warned. The usage is unambiguous in context throughout TBB's own source
# (verified with a minimal repro), so -fpermissive (downgrading it back to
# a warning) is safe rather than patching TBB internals. This needs to be
# set in three independent places -- the top-level CMakeLists.txt (for
# src/process_numpy.cpp, the _pynim/nanobind target), native/CMakeLists.txt
# (for the InstantMeshesLib target), and native/ext/tbb/CMakeLists.txt (TBB
# is its own nested CMake project with its own flags) -- since none of
# them share compile flags with each other.
python - "$PNIM_SRC/CMakeLists.txt" "$PNIM_SRC/native/CMakeLists.txt" "$PNIM_SRC/native/ext/tbb/CMakeLists.txt" <<'PYEOF'
import sys

top, native, tbb = sys.argv[1:4]

top_text = open(top).read()
old_top = 'if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")\n  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")\nelseif'
new_top = 'if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")\n  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++17")\n  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")\nelseif'
if old_top in top_text:
    open(top, "w").write(top_text.replace(old_top, new_top))
    print("patched top-level CMakeLists.txt")

native_text = open(native).read()
old_native = 'set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-unused-parameter")'
new_native = old_native + '\n  if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")\n    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fpermissive")\n  endif()'
if old_native in native_text and "-fpermissive" not in native_text:
    open(native, "w").write(native_text.replace(old_native, new_native))
    print("patched native/CMakeLists.txt")

tbb_text = open(tbb).read()
old_tbb = 'if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")\n   check_cxx_compiler_flag ("-flifetime-dse=1" SUPPORTS_FLIFETIME)\n   if (SUPPORTS_FLIFETIME)\n     target_compile_options(tbb_interface INTERFACE -flifetime-dse=1)\n   endif()\nendif()'
if old_tbb in tbb_text:
    replaced = old_tbb.replace(
        "   endif()\nendif()",
        "   endif()\n   target_compile_options(tbb_interface INTERFACE -fpermissive)\nendif()",
    )
    open(tbb, "w").write(tbb_text.replace(old_tbb, replaced))
    print("patched native/ext/tbb/CMakeLists.txt")
PYEOF

pip install --no-deps "$PNIM_SRC"

# The real package installs as `pynim` (its actual current module name
# upstream), but SF3D's mesh.py does `import pynanoinstantmeshes` (the
# PyPI package name at the version SF3D pins). Shim the expected import
# name -- this matches what the real PyPI wheel does internally.
SITE_PACKAGES="$(python -c 'import site; print(site.getsitepackages()[0])')"
mkdir -p "$SITE_PACKAGES/pynanoinstantmeshes"
cat > "$SITE_PACKAGES/pynanoinstantmeshes/__init__.py" <<'EOF'
"""Compatibility shim: SF3D imports `pynanoinstantmeshes` (the PyPI package
name), but the only real upstream source (vork/PyNanoInstantMeshes) installs
itself as `pynim`. Re-exported here under the name SF3D actually imports.
"""
from pynim import remesh  # noqa: F401

__all__ = ["remesh"]
EOF

# --- sidecar service dependencies ---------------------------------------
# gateway_and_ui/backend/service_3d.py runs its own small FastAPI process
# under this env (see scripts/run_sf3d_service.sh); these aren't SF3D's
# own dependencies, just what that service needs to be reachable over
# HTTP from the main gateway.
pip install fastapi uvicorn python-multipart httpx

echo
echo "== Verifying =="
python -c "
import torch, numpy, trimesh, transformers, gpytoolbox, pynanoinstantmeshes
import texture_baker, uv_unwrapper, rembg, omegaconf, einops, jaxtyping, open_clip
assert hasattr(gpytoolbox, 'subdivide') and hasattr(gpytoolbox, 'decimate') and hasattr(gpytoolbox, 'remesh_botsch')
assert hasattr(pynanoinstantmeshes, 'remesh')
print('All imports OK. CUDA available:', torch.cuda.is_available())
"

rm -rf "$SCRATCH_DIR"
echo
echo "== Done. Environment '$CONDA_ENV_NAME' is ready. =="
echo "Point PROTOSKIN_SF3D_PYTHON at: $(command -v python)"
