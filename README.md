# RSI-FPS 5.5 Simulation Environment & Multi-Account Live Dashboard

A clean, standalone toolkit for the **RSI-FPS 5.5** 5v5 tactical shooter benchmark:

1. **CPU Reference Arena (`sim/cpu_arena/`)** — Deterministic, strictly legal `rsi-fps-5.5` 10-seat Python simulator supporting all 5 maps (`borgo-r3-v54`, `molo-r3-v54`, `nevaio-r3-v54`, `dust2-cs2-v54`, `mirage-cs2-v54`).
2. **CUDA Batched `GpuSim` (`sim/gpusim/`)** — High-throughput parallel PyTorch + CUDA simulator (patched for both `sm_75` Turing/T4 and `sm_80+` Ampere/Ada/Hopper GPUs).
3. **Public-Replay-Calibrated CS2 TriMesh Cache (`sim/meshes/`)** — Exact Dust2 & Mirage triangle meshes with penetration materials calibrated strictly from public replays.
4. **Multi-Account Live Dashboard (`dashboard/`)** — Strictly read-only (`GET`-only), loopback-bound (`127.0.0.1`) web dashboard for monitoring multiple competition accounts, dual-phase (Attack/Defense) K/D/A & ADR stats, leaderboard standings, and 2D replay playback.

> **No proprietary strategy code, trained weights, private accounts, or credentials are included in this repository.**

---

## Repository Layout

```text
.
├── sim/
│   ├── cpu_arena/                  # CPU reference simulator (`fpssim`, legal obs/action contract)
│   │   ├── fpssim/
│   │   ├── NOTICE
│   │   └── UPSTREAM_README.md
│   ├── gpusim/                     # Batched CUDA simulator (`move_kernels.cu`, `sim.py`, unit tests)
│   └── meshes/
│       └── exact_mesh_cache_v2_calibrated.tar.gz   # Calibrated CS2 TriMesh cache (4.1 MiB)
├── dashboard/                      # Read-only multi-account web dashboard & replay viewer
│   ├── accounts.example.json       # Account configuration template (copy to accounts.json)
│   ├── api_client.py               # Strictly GET-only AGX API client & public replay parser
│   ├── server.py                   # Loopback-only HTTP server (127.0.0.1)
│   ├── replay_codec.py             # Public replay chunk decoder
│   ├── index.html / app.js / replay.js / style.css
│   └── test_dashboard_smoke.py     # Offline unit & security smoke tests
├── examples/
│   └── dummy_agent.py              # Minimal placeholder agent & runnable CPU Arena demo
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Environment Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the CPU Reference Simulator

```bash
python3 examples/dummy_agent.py
```

To enable exact CS2 triangle-mesh visibility and calibrated wallbang materials for `dust2-cs2-v54` and `mirage-cs2-v54`, extract the mesh archive once:

```bash
mkdir -p .cache/fps_data
tar -xzf sim/meshes/exact_mesh_cache_v2_calibrated.tar.gz -C .cache/fps_data
export FPS_DATA="$(pwd)/.cache/fps_data"
export FPS_STATIC_DIR="$FPS_DATA/static"
export FPS_ASSET_DIR="$FPS_DATA/assets"
export FPS_KIT_DIR="$(pwd)/sim/meshes/mapdata"
export FPS_FLOOR_DIR="$FPS_DATA/floor"
```

(`examples/dummy_agent.py` performs this extraction automatically on first run.)

### 3. Run the CUDA Batched `GpuSim` (Optional, Requires NVIDIA GPU + PyTorch)

```bash
pip install torch
PYTHONPATH=sim/cpu_arena:sim python3 sim/gpusim/test_sim_smoke.py
```

`sim/gpusim/move_kernels.cu` includes a butterfly `warp_max_u32` fallback so it compiles cleanly on both **Tesla T4 (`sm_75`)** and **L4 / A100 / H100 (`sm_80+`)**.

### 4. Launch the Multi-Account Live Dashboard

The dashboard is **strictly read-only** (HTTP `GET` only; never uploads, activates, or mutates any competition state) and binds exclusively to `127.0.0.1`.

1. Create your local accounts file (git-ignored):

   ```bash
   cp dashboard/accounts.example.json dashboard/accounts.json
   ```

   Edit `dashboard/accounts.json` with your own account aliases and `fps_agent_ids`.

2. Provide API tokens via environment variables (works on Linux, macOS, and Windows):

   ```bash
   export AGX_TOKEN_MAIN_ACCOUNT="your_bearer_token_here"
   export AGX_TOKEN_ALT_ACCOUNT_01="your_secondary_token_here"
   ```

   *(On macOS, if `AGX_TOKEN_<ALIAS>` is unset, `api_client.py` automatically falls back to macOS Keychain under service `FPS_KEYCHAIN_SERVICE`, default `agx.cool.fps.api`.)*

3. Start the server:

   ```bash
   python3 dashboard/server.py --port 8765
   ```

4. Open `http://127.0.0.1:8765/` in your browser.

#### Optional Environment Variables

| Variable | Default | Description |
|---|---|---|
| `FPS_DASHBOARD_ACCOUNTS` | `dashboard/accounts.json` | Path to account registry JSON |
| `FPS_DASHBOARD_CACHE` | `dashboard/.cache` | Local cache directory for snapshots & replays |
| `FPS_KEYCHAIN_SERVICE` | `agx.cool.fps.api` | macOS Keychain service name (fallback only) |
| `FPS_COMPETITION_ID` | `2a97f94f-816a-52b7-b4b8-5925db160245` | Competition UUID |

---

## Verified Simulator Fidelity & Known Sim-to-Real Gaps

Before relying on offline simulator results, note the measured alignment against official `rsi-fps-5.5` replays:

| Dimension | Measured Parity / Status | Notes |
|---|---|---|
| **5-Map LOS Visibility** | `100.00%` (20,000+ replay checks); `4999/5000` in 3-way enriched check | Exact CS2 TriMesh; 1 single-frame edge case on `nevaio` |
| **CPU Arena vs CUDA GpuSim XYZ** | Max error `0` to `9.54e-8` m (120 ticks × 5 maps) | Bit-close kinematic stepping |
| **CS2 Penetration Materials** | Dust2 `95.00%` (228/240), Mirage `95.20%` (218/229) held-out val | Calibrated from public replays (not official server binary); legacy proxy was 59.2% / 80.8% |
| **128D Feature MAE (GpuSim vs CPU)** | `0.0227` overall MAE (18,114 seats) | Largest residual in walkability corridor-depth rays (`105/107/111`) |
| **CS2 `GpuSim` Navigation** | Dust2 low-disp `1.55%`, Mirage low-disp `19.71%` (500-step budget) | Mirage requires longer horizon (1,000 steps) for full engagement |
| **Ballistics / Grenades / RNG** | Not frame-by-frame calibrated | Grenade bounce arcs, exact damage falloff, and spread RNG are approximate |

Offline H2H evaluation is useful for fast screening and regression checks, not a substitute for live competition validation.

---

## Running Verification Tests

```bash
python3 examples/dummy_agent.py
python3 -m unittest dashboard/test_dashboard_smoke.py
```

---

## Security & Privacy Guarantees

- **Zero strategy leakage**: No agent policies, tactical spot tables, lineup libraries, RL trainers, or `.npz`/`.zip` weights are included.
- **Zero credentials**: Tokens are read only at runtime from environment variables or OS Keychain and are never written to disk, logs, or HTTP responses.
- **Loopback-only & GET-only**: `dashboard/server.py` rejects non-loopback `Host` headers, enforces strict CSP headers, and `api_client.py` permits only `GET /api/v1/...` requests.
