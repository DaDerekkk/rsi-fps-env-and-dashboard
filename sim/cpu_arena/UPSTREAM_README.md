# fpssim: a local simulator for the agx.cool FPS bomb-defusal contest (rsi-fps-5.5)

`fpssim` is a pure-Python and NumPy re-implementation of the contest referee: 5v5 bomb defusal, MR12, 60 Hz physics and 20 Hz decisions.
It is meant for building agents, running offline self-play and RL, and debugging. Every module was checked against the platform's public
omniscient replays; the numbers are in the table below and in `docs/`.

The referee mechanics follow [cs2-web](https://github.com/apich/cs2-web), the browser CS2 game the platform referee is derived from.
Each platform-specific change (60 Hz tick, recoil index, economy values and so on) was found from the replays.

## Quick start
```bash
pip install numpy            # GPU simulator: also torch with CUDA
# data root (see "Data" below): ./fps_data/{data/static, ref/assets, fps_platform/kit/agent}
python3 tools/selfplay.py --map nevaio-r3-v54 --seed 1 --agent fps_data/fps_platform/kit/agent/agent.py
```
```python
from fpssim.arena import Arena
ar = Arena()
obs = ar.reset(map_id='borgo-r3-v54', seed=1)        # 10 observations in the platform observation contract
obs, events, done, info = ar.step(actions)            # 10 action dicts in the platform action contract (one 20 Hz decision)
```
A full match takes about 1 to 2 minutes of wall time in one process (whitebox maps). Dust2 is slower because it uses the exact triangle mesh.

## GPU simulator (`gpusim/`)
A batched CUDA port of the same referee: thousands of matches in parallel on one GPU, built from CUDA C++ kernels (JIT-compiled through
`torch.utils.cpp_extension.load_inline`) and vectorized PyTorch. Every module was parity-tested against `fpssim`, and the integrated
`GpuSim` matches the CPU `Arena` end to end: same seeds and actions, 0 differences over 4 full Dust2 matches (12,000 decisions each) and
shorter runs on Mirage, Borgo and Nevaio.
```python
from gpusim.sim import GpuSim
sim = GpuSim('nevaio-r3-v54', E=16384, device='cuda', seed=1)
sim.reset()
sim.step(actions)              # dict of [E,10] tensors (see gpusim/SPEC.md and docs/gpu_sim.md), one 20 Hz decision = 3 ticks
obs = sim.observe()            # fixed-size observation tensor for training (gpusim/obs.py)
```
Throughput on one workstation GPU (random actions, grenades and player collision on): about 1.06M env-decisions/s on Nevaio and 0.45M
on Dust2 at E=16384, roughly 100x the CPU simulator on 20 cores. Requires PyTorch with CUDA and a CUDA toolkit for the JIT build.
Module docs: `docs/gpu_move.md`, `docs/gpu_grenades.md`, `docs/gpu_vis_obs.md`, `docs/gpu_rounds.md`, `docs/gpu_sim.md`.

## Data (not included)
Paths are relative to `./fps_data` (edit the constants in `fpssim/arena.py` and `fpssim/common.py` to move it):
- `data/static/static_<map_id>.json`: the map description in the public replays (`GET /api/v1/python-submissions/recordings/<job>/static`).
- `ref/assets/map_positions.f32`, `map_penetration-materials.u8`: Dust2 collision mesh and materials. These are the cs2-web assets
  (Awpy-derived CS2 geometry); the platform records the same SHA-256 in the replay's `map.collision`. Not redistributed here.
- `ref/assets/mirage_positions.f32` (plus a copy named `mirage-positions.f32`) and `mirage_penetration-materials.u8`: the Mirage mesh.
  Build it yourself from the public Awpy data:
  ```bash
  python3 tools/mirage_awpy_fetch.py fps_data/ref/awpy de_mirage          # awpy-data release 2000905, geometry.zip (HTTP range reads)
  python3 tools/mirage_build_mesh.py fps_data/ref/awpy/de_mirage.mesh fps_data/ref/assets   # port of cs2-web map-build.py
  ```
  The result is byte-identical to the platform mesh (SHA-256 `3d6ce1fd...`, 73,152 triangles). The platform's Mirage penetration
  materials are not public. `tools/mirage_materials.py` rebuilds them from replay bullets (needs Mirage replays), which matches 99.9% of
  recorded bullets. See `docs/mirage_mesh.md`.
- `fps_platform/kit/agent/agent.py`: the official baseline from the contest kit, used as the default self-play policy.
- `data/rec/<job>/` (events.jsonl, frames.npz): public replays, needed only for the scripts in `validate/`.
  Use `tools/extract_frames.py` to turn `frames.jsonl.gz` into `frames.npz`.

## Fidelity against platform replays
| Module | File | Status | Evidence |
|---|---|---|---|
| Economy and round settlement | economy.py, arena.py | exact | 22,873/22,873 cash transactions |
| Movement (cs2-web hull vs mesh) | meshmove.py, movement.py | near-exact | whitebox maps 97-98% of 3-tick blocks exact; Dust2 98% < 1 cm |
| Weapon state machine | weapons.py | exact | about 20M field comparisons |
| Recoil, aim punch, inaccuracy | recoil.py | tick-exact | 99.6-99.85% |
| Spread | accuracy.py | distribution-exact | KS test on 8,931 shots |
| Hit volumes and damage | combat.py | exact | 4,226/4,226 hits |
| Wall penetration (Dust2) | penetration.py | exact | 1,049/1,049 bullets |
| Grenades | grenades.py | tick-exact | paths, bounces, detonations, smoke, fire, HE, flash |
| Visibility | visibility.py | exact on Dust2/Mirage | per enemy pair 0.99971-0.999999 (Mirage 0.99996) |
| Mirage mesh | tools/mirage_*.py | byte-identical geometry | bullets 9,675/9,675 exact; wallbangs 1,850/1,854 with rebuilt materials |
| Observations and event views | obs.py | near-exact | per-seat event views 0.9965; schema-valid |
| Sounds: delivery into observations | obs.py, arena.py | exact | 137,526/137,526 decisions on our own training.jsonl (12 private matches) |
| Sounds: emitters | obs.AudioEmitter, arena.py | footsteps near-exact; bomb beeps exact | footstep F1 0.67-0.76, 84-89% within +-1 tick (24 Hz-limited) |

## Known gaps
- **Mirage**: the geometry rebuilds byte-identically from Awpy (see Data). The penetration materials are reconstructed from replays rather than
  the platform file, so a few wallbangs differ. Without the rebuilt mesh, the fallback approximation (contest kit contours) is not usable.
- Whitebox maps: the platform has an unpublished "visible-world" mesh with facade trim, so rare line-of-sight and grenade differences remain.
- Assumptions: the whitebox buy zone (spawn-pad circles as on Dust2/Mirage) and a few observation fields such as `teammates.zone` (see `docs/obs.md`).
  The 200 ms minimum grenade prime follows cs2-web `game.js`, which the platform documentation says it copies.
- Sounds: the platform delivers only the decision tick's emitters, never gunshots, mirrors left/right and ignores occlusion (see `docs/obs.md`).
- Single-bullet spread is not reproducible because the platform's seeds are hidden; only the distribution matches.
- Knife, shotguns and SCAR20/SG553 scopes are ported from cs2-web but do not appear in the public replays.

## Layout
`fpssim/` CPU simulator · `gpusim/` GPU simulator and its parity tests · `validate/` alignment scripts (each prints its numbers; they need replays) · `docs/` per-module notes · `tools/` self-play runner and frame extractor.

See NOTICE for attribution.
