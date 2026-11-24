"""gpusim.collide.resolve_players vs fpssim.arena.Arena._resolve_players, and movement + collision together.

usage: python -m gpusim.test_collide [--maps dust2,borgo,molo,nevaio] [--scenes 20000] [--rolls 1500] [--ticks 90] [--procs 48]
         [--bench] [--out json]

A. collide only: random crowded scenes (10 seats: players piled within ~1 m, some stacked in height, coincident players, crouched,
   rising, dead, objective-locked, standing next to walls), one call = the 2 pair passes.  CPU = the real
   Arena._resolve_players (called on a stand-in object that has the same .players/.world fields).  CPU worlds:
     whitebox maps  BoxWorld (analytic boxes, the CPU Arena's world there), GPU default wall_check=True on the triangulated boxes;
     dust2          the MeshWorld, which has no `blockers` (so no wall check), GPU default wall_check=False;
     dust2 + wall   CPU MeshWorld + a mesh `blockers` shim (same SAT as the mover, shrunk by BoxWorld.EPS), GPU wall_check=True.
B. movement + collision: 10 players start close together on the ground, random held actions (pressing into each other, jumping,
   crouching, locking), CPU = MeshMover.step for every alive player then _resolve_players each tick (what Arena._tick_inner does with
   the mesh mover); GPU = GpuMover.step + resolve_players on the whole batch, free-running from the same initial state; compared at
   every tick.
"""
import argparse, json, math, os, sys, time, types, zlib
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch
from fpssim import meshmove as MM
from fpssim.arena import Arena
from fpssim.collision import BoxWorld
from fpssim.movement import PlayerState
from gpusim import move as GM
from gpusim import test_move as TM

ROOT = TM.ROOT
FIELDS = ['x', 'y', 'z', 'vx', 'vy', 'vz']
_W = None; _MAP = None; _REF = None


class MeshBlockers:
    """BoxWorld.blockers semantics on a MeshWorld: any triangle overlapping the world box shrunk by 1e-6 - 1e-8 on every side"""
    SHRINK = 1e-6 - 1e-8

    def __init__(self, w):
        self.w = w

    def blockers(self, x0, x1, y0, y1, zlo, zhi):
        e = self.SHRINK
        lo = np.array([x0 + e, y0 + e, zlo + e]); hi = np.array([x1 - e, y1 - e, zhi - e])
        corners = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
        raw = (self.w.Bt @ (corners - self.w.o).T).T
        rlo, rhi = raw.min(0), raw.max(0)
        L = self.w.gather(rlo, rhi)
        if L.n == 0: return np.zeros((0, 6))
        return np.ones((1, 6)) if L.overlap_mask((list(rlo), list(rhi))).any() else np.zeros((0, 6))


class Pl:
    def __init__(self, ps, alive, locked):
        self.ps = ps; self.alive = alive; self.objective = 1 if locked else None


class FakeArena:
    _resolve_players = Arena._resolve_players

    def __init__(self, players, world):
        self.players = players; self.world = world


def cpu_world(name, mode):
    """mode 'cpu': what the CPU Arena has (BoxWorld on whitebox, bare MeshWorld on dust2); 'mesh': MeshWorld + blockers shim"""
    if mode == 'mesh' or not _MAP.get('boxes'):
        if mode == 'mesh': return MeshBlockers(_W)
        return _W
    return BoxWorld(_MAP)


def load_world(name, cache):
    global _W, _MAP
    _MAP = TM.load_static(name)
    _W = GM.make_world(_MAP, f'{ROOT}/ref/assets', cache)


# ---------------------------------------------------------------------------------------------------------------- scenes
def centers(m, rng, n):
    """scene centres: nav nodes, and on whitebox maps also points right next to box faces"""
    nodes = np.array([nd['position'] for nd in m['nav']['nodes']])
    out = []
    boxes = m.get('boxes') or []
    gz = float((m.get('ground') or {}).get('z', 0.0)) - 0.002 + 0.00011
    for _ in range(n):
        if boxes and rng.random() < 0.6:
            b = boxes[rng.integers(len(boxes))]
            side = rng.integers(4); off = rng.uniform(0.05, 0.9); along = rng.uniform(-0.5, 0.5)
            hw, hd = b['w'] / 2, b['d'] / 2
            if side == 0: p = (b['x'] + hw + off, b['y'] + along * b['d'])
            elif side == 1: p = (b['x'] - hw - off, b['y'] + along * b['d'])
            elif side == 2: p = (b['x'] + along * b['w'], b['y'] + hd + off)
            else: p = (b['x'] + along * b['w'], b['y'] - hd - off)
            out.append((p[0], p[1], gz))
        else:
            q = nodes[rng.integers(len(nodes))]
            out.append((q[0] + rng.normal(0, .4), q[1] + rng.normal(0, .4), q[2] + 0.00011))
    return out


def free_at(blk, x, y, z, h):
    return len(blk.blockers(x - .3, x + .3, y - .3, y + .3, z + .46, z + h)) == 0


def make_scene(rng, c, blk=None):
    """10 players: (x,y,z,vx,vy,vz,yaw,crouch,grounded,alive,locked).  With blk (an object with .blockers) the starting hulls are
    required not to overlap world geometry (players cannot start inside walls; the pushes may then end up there)."""
    P = []
    for s in range(10):
        for _try in range(30):
            r = rng.uniform(0, 0.9) if rng.random() < 0.8 else rng.uniform(0.9, 2.5)
            a = rng.uniform(0, 2 * math.pi)
            x, y = c[0] + r * math.cos(a), c[1] + r * math.sin(a)
            u = rng.random()
            z = c[2] if u < 0.55 else (c[2] + rng.uniform(-0.3, 0.45) if u < 0.8 else c[2] + rng.uniform(0.7, 2.0))
            crouch = rng.random() < 0.2
            if blk is None or free_at(blk, x, y, z, 1.1 if crouch else 1.8): ok = True; break
        else: ok = False                  # no free spot found: this seat is dead in the scene
        vz = rng.uniform(-0.5, 0.05) if rng.random() < 0.8 else rng.uniform(0.1, 4.0)
        P.append([x, y, z, rng.normal(0, 2.5), rng.normal(0, 2.5), vz, rng.uniform(-math.pi, math.pi), crouch,
                  rng.random() < 0.7, ok and rng.random() < 0.92, rng.random() < 0.12])
    if rng.random() < 0.08:       # coincident players (centre distance < 1e-5 -> yaw-direction push)
        j, k = rng.choice(10, 2, replace=False)
        P[k][0], P[k][1] = P[j][0] + rng.uniform(0, 5e-6), P[j][1]
    return P


def cpu_resolve(scene, world):
    pl = []
    for (x, y, z, vx, vy, vz, yaw, crouch, grounded, alive, locked) in scene:
        ps = PlayerState(x=x, y=y, z=z, vx=vx, vy=vy, vz=vz, yaw=yaw, crouch=bool(crouch), grounded=bool(grounded))
        pl.append(Pl(ps, bool(alive), bool(locked)))
    FakeArena(pl, world)._resolve_players()
    return [[p.ps.x, p.ps.y, p.ps.z, p.ps.vx, p.ps.vy, p.ps.vz, float(p.ps.grounded)] for p in pl]


def scenes_job(args):
    seed, n, mode = args
    rng = np.random.default_rng(seed)
    w = cpu_world(None, mode)
    cs = centers(_MAP, rng, n)
    blk = w if hasattr(w, 'blockers') else MeshBlockers(_W)
    S = [make_scene(rng, c, blk) for c in cs]
    R = [cpu_resolve(s, w) for s in S]
    return np.array(S, dtype=np.float64), np.array(R, dtype=np.float64)


def gpu_scenes(S, dev):
    S = torch.as_tensor(S, device=dev)
    st = {k: S[:, :, n].contiguous() for n, k in enumerate(['x', 'y', 'z', 'vx', 'vy', 'vz', 'yaw'])}
    st['crouch'] = S[:, :, 7] != 0; st['grounded'] = (S[:, :, 8] != 0).contiguous(); st['alive'] = S[:, :, 9] != 0
    return st, S[:, :, 10] != 0


def part_a(gm, name, a, dev, mode, wall, swept):
    import multiprocessing as mp
    ctx = mp.get_context('fork')
    per = 500
    jobs = [(zlib.crc32(f'{name}{mode}a'.encode()) % 100000 * 1000 + k, per, mode) for k in range(a.scenes // per)]
    t0 = time.time()
    with ctx.Pool(a.procs) as pool:
        res = pool.map(_job_wrap, jobs)
    S = np.concatenate([r[0] for r in res]); R = np.concatenate([r[1] for r in res])
    print(f'  CPU scenes ({mode}): {len(S)} scenes in {time.time() - t0:.1f}s', flush=True)
    st, L = gpu_scenes(S, dev)
    alive = st['alive'].clone()
    gm.reset_errors()
    gm.resolve_players(st, L, wall_check=wall, swept=swept)
    torch.cuda.synchronize()
    G = torch.stack([st[k] for k in FIELDS] + [st['grounded'].double()], -1).cpu().numpy()
    alive = alive.cpu().numpy()
    dead_untouched = bool((G[~alive] == S[~alive][:, [0, 1, 2, 3, 4, 5, 8]]).all())
    err = np.abs(G - R)
    perr = err[:, :, :3].max(2); verr = err[:, :, 3:6].max(2); gerr = err[:, :, 6]
    scene_exact = (err.reshape(len(S), -1) == 0).all(1)
    scene_pos = perr.max(1); scene_vel = verr.max(1); scene_g = gerr.max(1)
    moved = (np.abs(R[:, :, :6] - S[:, :, :6]).max((1, 2)) > 0)          # scenes where the CPU changed anything
    out = dict(mode=mode, wall=wall, swept=swept, scenes=len(S), changed_by_cpu=float(moved.mean()), scene_exact=float(scene_exact.mean()),
               scene_exact_changed=float(scene_exact[moved].mean()),
               pos_p50=float(np.percentile(scene_pos, 50)), pos_p99=float(np.percentile(scene_pos, 99)), pos_max=float(scene_pos.max()),
               vel_p99=float(np.percentile(scene_vel, 99)), vel_max=float(scene_vel.max()),
               frac_scenes_le_1e9=float((scene_pos <= 1e-9).mean() * 1.0), frac_scenes_both_le_1e9=float(((scene_pos <= 1e-9) & (scene_vel <= 1e-9) & (scene_g == 0)).mean()),
               grounded_mismatch=int((scene_g > 0).sum()), dead_untouched=dead_untouched, errors=gm.errors())
    print(f'  collide-only [{name} cpu-world={mode} wall={wall} swept={swept}] scenes={out["scenes"]} (CPU changed something in {out["changed_by_cpu"] * 100:.1f}%): '
          f'bitwise-identical scenes {out["scene_exact"] * 100:.3f}%  (of the changed ones {out["scene_exact_changed"] * 100:.3f}%)', flush=True)
    print(f'    position error  p50 {out["pos_p50"]:.2e}  p99 {out["pos_p99"]:.2e}  max {out["pos_max"]:.2e} m;  velocity p99 {out["vel_p99"]:.2e} '
          f'max {out["vel_max"]:.2e};  scenes with all errors <= 1e-9: {out["frac_scenes_both_le_1e9"] * 100:.3f}%;  grounded mismatches '
          f'{out["grounded_mismatch"]};  dead players untouched: {dead_untouched}', flush=True)
    bad = np.nonzero(~scene_exact)[0][:3]
    for b in bad:
        print('    mismatch scene', b, 'pos', scene_pos[b], 'vel', scene_vel[b], 'grounded', scene_g[b], flush=True)
    return out


def _job_wrap(a): return scenes_job(a)


# ---------------------------------------------------------------------------------------------------------------- part B
def roll_job(args):
    seed, ticks, mode = args
    rng = np.random.default_rng(seed)
    w = _W
    bw = cpu_world(None, mode)
    c = centers(_MAP, rng, 1)[0]
    mv = MM.MeshMover()
    pl = []
    M = [float(TM.SPEEDS[rng.integers(len(TM.SPEEDS))]) for _ in range(10)]
    blk = bw if hasattr(bw, 'blockers') else MeshBlockers(w)
    for s in range(10):
        for _try in range(30):
            r = rng.uniform(0, 0.8); a = rng.uniform(0, 2 * math.pi)
            if free_at(blk, c[0] + r * math.cos(a), c[1] + r * math.sin(a), c[2], 1.8): ok = True; break
        else: ok = False                  # no spot outside the walls found: this seat is dead
        ps = PlayerState(x=c[0] + r * math.cos(a), y=c[1] + r * math.sin(a), z=c[2], yaw=rng.uniform(-math.pi, math.pi))
        ps.jump_buffer = 0.0
        pl.append(Pl(ps, ok and bool(rng.random() < 0.95), False))
    to_c = lambda p: math.atan2(-(c[1] - p.ps.y), c[0] - p.ps.x)            # yaw pointing at the centre (forward = (cos y, -sin y))
    out = dict(init=[], act=[], lock=[], post=[], grounded=[], crouch=[])
    out['init'] = [[p.ps.x, p.ps.y, p.ps.z, p.ps.vx, p.ps.vy, p.ps.vz, p.ps.yaw, p.alive] for p in pl]
    held = [None] * 10; lock = [False] * 10
    fa = FakeArena(pl, bw if (mode == 'mesh' or _MAP.get('boxes')) else w)
    for t in range(ticks):
        if t % 3 == 0:
            for s in range(10):
                press = rng.random() < 0.6
                held[s] = dict(move_forward=1.0 if press else float(rng.choice([-1, 0, 1])), move_right=float(rng.choice([0, 0, 0.5, -0.5, 1, -1])),
                               yaw_rate=float(np.clip(rng.normal(0, 0.5), -1, 1)), pitch_rate=0.0, jump=bool(rng.random() < 0.08),
                               crouch=bool(rng.random() < 0.1), walk=bool(rng.random() < 0.1), ladder='none', climb=0.0)
                if press and rng.random() < 0.5:        # turn towards the pile
                    d = (to_c(pl[s]) - pl[s].ps.yaw + math.pi) % (2 * math.pi) - math.pi
                    held[s]['yaw_rate'] = float(np.clip(d / (3.0 * 3 / 60), -1, 1))
                if rng.random() < 0.04: lock[s] = not lock[s]
        for s, p in enumerate(pl):
            p.objective = 1 if lock[s] else None
            if p.alive: mv.step(p.ps, held[s], M[s], w, (), t, lock[s])
        fa._resolve_players()
        out['act'].append([[h['move_forward'], h['move_right'], h['yaw_rate'], h['jump'], h['crouch'], h['walk'], M[s]] for s, h in enumerate(held)])
        out['lock'].append(list(lock))
        out['post'].append([[p.ps.x, p.ps.y, p.ps.z, p.ps.vx, p.ps.vy, p.ps.vz] for p in pl])
        out['grounded'].append([p.ps.grounded for p in pl]); out['crouch'].append([p.ps.crouch for p in pl])
    return {k: np.array(v, dtype=np.float64) for k, v in out.items()}


def part_b(gm, name, a, dev, mode, wall, swept):
    import multiprocessing as mp
    ctx = mp.get_context('fork')
    t0 = time.time()
    with ctx.Pool(a.procs) as pool:
        rolls = pool.map(roll_job, [(zlib.crc32(f'{name}{mode}b'.encode()) % 100000 * 1000 + k, a.ticks, mode) for k in range(a.rolls)], chunksize=2)
    R = {k: np.stack([r[k] for r in rolls]) for k in rolls[0]}
    n, T = R['post'].shape[:2]
    print(f'  CPU movement+collision rollouts ({mode}): {n} matches x {T} ticks in {time.time() - t0:.1f}s', flush=True)
    I = torch.as_tensor(R['init'], device=dev)
    st = gm.new_state((n, 10))
    for k, c in zip(FIELDS + ['yaw'], range(7)): st[k].copy_(I[:, :, c])
    st['alive'].copy_(I[:, :, 7] != 0)
    maxp = np.zeros(n); maxv = np.zeros(n); first = np.full(n, T); gm_mis = 0
    pos_all = []; vel_all = []
    gm.reset_errors()
    for t in range(T):
        A = torch.as_tensor(R['act'][:, t], device=dev)
        ac = gm.prep_action(dict(move_forward=A[:, :, 0], move_right=A[:, :, 1], yaw_rate=A[:, :, 2], jump=A[:, :, 3] != 0,
                                 crouch=A[:, :, 4] != 0, walk=A[:, :, 5] != 0), (n, 10))
        L = torch.as_tensor(R['lock'][:, t] != 0, device=dev)
        gm.step(st, ac, A[:, :, 6].contiguous(), L, t)
        gm.resolve_players(st, L, wall_check=wall, swept=swept)
        G = torch.stack([st[k] for k in FIELDS], -1).cpu().numpy()
        alive = st['alive'].cpu().numpy()
        err = np.abs(G - R['post'][:, t])
        pe = err[:, :, :3].max((1, 2)); ve = err[:, :, 3:].max((1, 2))
        ex = (err.reshape(n, -1) == 0).all(1)
        first[(~ex) & (first == T)] = t
        maxp = np.maximum(maxp, pe); maxv = np.maximum(maxv, ve)
        gm_mis += int((st['grounded'].cpu().numpy() != (R['grounded'][:, t] != 0)).sum()) + int((st['crouch'].cpu().numpy() != (R['crouch'][:, t] != 0)).sum())
        pos_all.append(pe); vel_all.append(ve)
    pos_all = np.concatenate(pos_all); vel_all = np.concatenate(vel_all)
    out = dict(mode=mode, wall=wall, swept=swept, matches=n, ticks=T, bitwise_whole_rollout=float((first == T).mean()),
               frac_ticks_bitwise=float((pos_all == 0).mean()), tick_pos_p99=float(np.percentile(pos_all, 99)), tick_pos_p999=float(np.percentile(pos_all, 99.9)),
               tick_pos_max=float(pos_all.max()), tick_vel_p99=float(np.percentile(vel_all, 99)), tick_vel_max=float(vel_all.max()),
               frac_rollouts_maxpos_le_1e9=float((maxp <= 1e-9).mean()), frac_rollouts_maxpos_le_1e6=float((maxp <= 1e-6).mean()),
               frac_rollouts_maxpos_le_1e3=float((maxp <= 1e-3).mean()), flag_mismatches=gm_mis, errors=gm.errors())
    print(f'  move+collide free-run [{name} cpu-world={mode} wall={wall} swept={swept}]: {n} matches x {T} ticks: whole rollout bitwise {out["bitwise_whole_rollout"] * 100:.2f}%, '
          f'ticks with position bitwise-equal {out["frac_ticks_bitwise"] * 100:.3f}%', flush=True)
    print(f'    per-tick position error p99 {out["tick_pos_p99"]:.2e} p99.9 {out["tick_pos_p999"]:.2e} max {out["tick_pos_max"]:.2e} m; velocity p99 '
          f'{out["tick_vel_p99"]:.2e} max {out["tick_vel_max"]:.2e}; rollouts with max pos err <= 1e-9: {out["frac_rollouts_maxpos_le_1e9"] * 100:.2f}%, '
          f'<= 1e-6: {out["frac_rollouts_maxpos_le_1e6"] * 100:.2f}%, <= 1e-3: {out["frac_rollouts_maxpos_le_1e3"] * 100:.2f}%; '
          f'grounded/crouch flag mismatches {gm_mis}; GPU errors {gm.errors()}', flush=True)
    return out


def bench(gm, name, m, dev, wall, Es=(1024, 4096, 16384), crowd=False):
    """time of resolve_players on E matches of 10 players: spread over the map, or all piled up (crowd)"""
    nodes = np.array([n['position'] for n in m['nav']['nodes']])
    res = {}
    for E in Es:
        rng = np.random.default_rng(E)
        shape = (E, 10)
        p = nodes[rng.integers(len(nodes), size=E)]
        base = torch.as_tensor(p, device=dev)
        r = (torch.rand(shape, device=dev, dtype=torch.float64) * (0.5 if crowd else 5.0))
        a = torch.rand(shape, device=dev, dtype=torch.float64) * 6.28
        st = dict(x=(base[:, 0:1] + r * torch.cos(a)).contiguous(), y=(base[:, 1:2] + r * torch.sin(a)).contiguous(),
                  z=(base[:, 2:3] + torch.zeros(shape, device=dev, dtype=torch.float64)).contiguous(),
                  vx=torch.randn(shape, device=dev, dtype=torch.float64), vy=torch.randn(shape, device=dev, dtype=torch.float64),
                  vz=torch.zeros(shape, device=dev, dtype=torch.float64), yaw=torch.zeros(shape, device=dev, dtype=torch.float64),
                  crouch=torch.zeros(shape, device=dev, dtype=torch.bool), grounded=torch.ones(shape, device=dev, dtype=torch.bool),
                  alive=torch.ones(shape, device=dev, dtype=torch.bool))
        L = torch.zeros(shape, device=dev, dtype=torch.bool)
        gm.resolve_players(st, L, wall_check=wall); torch.cuda.synchronize(); t0 = time.time()
        for _ in range(20): gm.resolve_players(st, L, wall_check=wall)
        torch.cuda.synchronize(); dt = (time.time() - t0) / 20
        res[E] = dict(ms=dt * 1e3)
        print(f'  collide bench [{name} wall={wall} crowd={crowd}] E={E:6d}: {dt * 1e3:7.3f} ms/call', flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', default='dust2,borgo,molo,nevaio')
    ap.add_argument('--scenes', type=int, default=20000)
    ap.add_argument('--rolls', type=int, default=1500)
    ap.add_argument('--ticks', type=int, default=90)
    ap.add_argument('--procs', type=int, default=48)
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--no-parity', action='store_true')
    ap.add_argument('--out', default=f'{ROOT}/tmp/gpu_move/collide.json')
    ap.add_argument('--cache', default=f'{ROOT}/tmp/meshcache')
    a = ap.parse_args()
    dev = torch.device('cuda')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    report = {}
    for name in a.maps.split(','):
        load_world(name, a.cache)
        m = _MAP
        gm = GM.GpuMover(world=_W, device=dev, whitebox=bool(m.get('boxes')))
        print(f'{name}: {len(_W.tri)} triangles', flush=True)
        rep = {}
        # (mode, GPU wall_check): the CPU-Arena behaviour of the map, and (dust2 only) the wall check on a mesh world
        # (CPU world, GPU wall_check, GPU swept): the CPU Arena as it is (GPU defaults), the same with the path check, and the mesh `blockers` shim
        variants = [('cpu', None, False), ('cpu', None, True), ('mesh', True, False)]
        for mode, wall, swept in ([] if a.no_parity else variants):
            if mode == 'cpu' and not m.get('boxes') and swept: continue            # no wall check on dust2: swept is irrelevant
            key = f'{mode}_swept{int(swept)}'
            rep[key] = dict(collide=part_a(gm, name, a, dev, mode, wall, swept), move_collide=part_b(gm, name, a, dev, mode, wall, swept))
        if a.bench:
            rep['bench'] = dict(spread=bench(gm, name, m, dev, None), crowd=bench(gm, name, m, dev, None, crowd=True),
                                spread_wall=bench(gm, name, m, dev, True), crowd_wall=bench(gm, name, m, dev, True, crowd=True))
        report[name] = rep
        json.dump(report, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
