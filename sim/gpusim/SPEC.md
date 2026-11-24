# gpusim: batched GPU port of fpssim. Shared specification

Goal: step E independent matches (E = 1k to 16k) on one R6000 GPU, bit-faithful to the CPU reference `fpssim/`
(which is validated against platform recordings). CUDA C++ (torch `load_inline`, `TORCH_EXTENSIONS_DIR=./fps_data/tmp/torch_ext`)
plus vectorized torch. **No Triton.** Each module ships with a parity test against its fpssim counterpart.

## Conventions
- World: metres, z up, forward = (cos p cos y, -cos p sin y, sin p); yaw increases to the right. 60 Hz physics, 20 Hz decisions (3 ticks).
- Seats 0..9, team = seat // 5. `attack_team[E]` tells which team attacks.
- dtype: positions and velocities float64; angles float64; timers int32 (absolute match tick); flags bool; counts int32.
- Batch axes: `[E]` per match, `[E,10]` per player, `[E,10,3]` per weapon slot (0 primary, 1 secondary, 2 knife).
- Weapon ids: index into `gpusim.tables.WEAPON_IDS` (order of fpssim catalog.json `weapons`, -1 = empty). Utility ids: `UTIL_IDS`.
- Map: one map per GpuSim instance (all E matches share the map; run several instances for several maps).
  Geometry: one triangle soup per map (Dust2/Mirage mesh; whitebox maps triangulated like tools/meshmove_wb_validate.py),
  BVH via gpusim/bvh.py, ray queries via gpusim/ray.py.
- RNG: counter-based (Philox via torch.Generator per instance, or curand with (seed, match, tick, seat, purpose)).
  Parity tests inject identical random numbers into both implementations.

## State (all tensors live in `sim.s`, a dict of tensors; module owners add their own keys with a prefix)
Players `[E,10]`: x y z vx vy vz yaw pitch (f64); alive crouch grounded last_jump walk_held (bool); hp armor (f64); helmet kit (bool);
cash (i32); height (f64); step_distance (f64); movement_mode (i8: 0 ground, 1 air, 2 ladder); ladder_id (i16, -1);
blind_until blind_start throw_ready pickup_ready_tick death_tick (i32); has_bomb (bool); objective (i8: 0 none, 1 plant, 2 defuse);
obj_progress (i32).
Weapons `[E,10,3]`: wid ammo reserve shots (i32); shot_ready reload_until reload_commit last_shot (i32, -1 none); reload_committed reload_shell pressed (bool);
inacc (f64); rs `[E,10,3,7]` = (index, pitch, yaw, velPitch, velYaw, viewPitch, viewYaw) f64. Per player: active (i8), switch_until (i32), zoom (i8).
Match `[E]`: tick, round_index, round_start_tick, live_end, post_end (i32); phase (i8: 0 buy, 1 live, 2 post, 3 finished);
round_wins `[E,2]`, loss_level `[E,2]` (i32); attack_team (i8); bomb_* (state i8: 0 held, 1 world, 2 planted, 3 defused, 4 exploded;
carrier i8; pos f64[3]; fuse_end, planted_tick, drop_tick i32; dropper i8; actor i8; duration i32).
Items: world_items `[E,K]` (K=32) kind/pos/ammo/reserve/valid.

## Actions `[E,10]` (held for 3 ticks; decoded from the platform action contract)
move_forward move_right yaw_rate pitch_rate climb (f32 in [-1,1]); jump crouch walk fire fire2 reload use_plant use_defuse (bool);
weapon_slot zoom (i8); buy (i16 catalogue id, -1 none, applied once at the decision boundary); throw_kind (i8), throw_state (i8: idle/prime/release/cancel),
throw_mode (i8); drop (i8); ladder (i8: none/attach/detach).

## Tick order (mirrors fpssim.arena.Arena._tick_inner; keep it identical)
1. tick += 1; buy->live phase switch at round_start+360.
2. Movement for alive players (gpusim.move), then player-vs-player resolution (cs2-web resolveAllPlayerCollisions, 2 passes).
3. Weapons: decay all slots -> reload progress -> switch -> zoom -> reload request -> fire (gpusim.weapons). Shots -> bullets (gpusim.shoot):
   accuracy from the pre-kick punch/penalty, two-disc spread, hit volumes, penetration, damage, kills, kill rewards.
4. Objective/bomb step (plant/defuse progress, pickup), then grenades (live only: handle_throw, then step), then bomb fuse, then win conditions,
   post->next round. Economy events update cash.
5. At decision boundaries: visibility (gpusim.vis) and observation tensors (gpusim.obs).

## Module ownership
- gpusim/bvh.py, ray.py: main session (done).
- gpusim/move*.py: movement agent.
- gpusim/weapons.py, shoot.py, vis.py, sim.py (integration), tables.py: main session.
- gpusim/grenades*.py: grenades agent.
- gpusim/rounds*.py (phase machine, economy, buy/refund, bomb/objective, drops/pickups, win conditions): rounds agent.
- gpusim/obs*.py (observation tensors for training + optional JSON decode for parity): later.
