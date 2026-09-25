# third_party/

Vendored source for dependencies that couldn't be installed as plain pip
packages. Tracked in git (unlike `models/`) because these are patched copies,
not reproducible downloads.

## stable-fast-3d/

Cloned from [`Stability-AI/stable-fast-3d`](https://github.com/Stability-AI/stable-fast-3d)
(commit at clone time; see `git log -1` inside the directory), used by
`visual_engine/reconstruct3d.py` via the sidecar process described in
`scripts/run_sf3d_service.sh`.

License: Stability AI Community License (free for personal/hackathon use and
orgs under $1M/yr revenue; see `LICENSE.md` inside this directory). The app's
UI must display "Powered by Stability AI" and name the license per its terms
— see the disclaimer footer in `gateway_and_ui/frontend/index.html`.

### Why this is vendored instead of `pip install stable-fast-3d`

The package isn't published to PyPI as an importable library — its own
`setup.py`/`pyproject.toml` don't exist at the repo root in an installable
form; usage is via `python run.py <image>` from a checkout. This vendored
copy is what `reconstruct3d.py` imports `sf3d.system.SF3D` from directly
(with the directory added to `sys.path`).

### `stable-fast-3d` itself is unpatched

This is a plain, unmodified copy of the upstream repo (its own `.git` history
was dropped so it plugs cleanly into this repo instead of nesting a second
git repo). The build issues on this machine (aarch64 Linux, GCC 13, CUDA
13.0) were entirely in two of its from-source Python dependencies —
`gpytoolbox` and `pynanoinstantmeshes` — neither of which ships an aarch64
Linux wheel on PyPI. Those two are built separately (not vendored as source
here; `scripts/setup_sf3d_env.sh` clones and patches them into a scratch
directory as part of building the `protoskin3d` conda env) with small,
documented CMake/compiler-flag patches — see that script for the exact fixes
and the reasoning behind each one.

## Not committing model weights here

`stable-fast-3d/model.safetensors` (and the base SDXL/ControlNet/Qwen
weights) are *not* in this directory or in git. They're downloaded
separately to `PROTOSKIN_SF3D` (see `.env.example`) via Hugging Face, same
as every other model this project uses. Downloading Stable Fast 3D's
weights requires requesting access on its Hugging Face model page first
(auto-approved, but a real account action — see `scripts/download_models.sh`).
