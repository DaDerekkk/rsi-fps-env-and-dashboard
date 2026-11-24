"""GpuMover vs CPU MeshMover: per-tick parity (teacher forced on CPU rollouts), free-running parity, throughput.

usage: python -m gpusim.test_move [--maps dust2,borgo,molo,nevaio] [--players 2000] [--ticks 150] [--procs 48] [--bench]
         [--out json]
CPU rollouts (MeshMover, others=()) of single players from nav nodes / ladders with random held 20 Hz actions: walking, sprinting
into walls and sliding along them, jumping, crouching, stairs/ramps, perturbed starts (airborne, random velocity, crouched,
objective-locked, buy-phase frozen, dead), ladder attach / climb / detach.  Every recorded tick (pre-state, action) is then
(1) stepped once on the GPU from the CPU pre-state and compared with the CPU post-state (teacher forcing), and
(2) the GPU is run freely from the initial state with the recorded actions and compared with the CPU state at every tick.
"exact" = every float field bitwise equal and all discrete fields equal.
"""
import argparse, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch
from fpssim import meshmove as MM
from fpssim.movement import PlayerState
from gpusim import move as GM

ROOT = './fps_data'
MAPS = {'dust2': 'dust2-cs2', 'mirage': 'mirage-cs2', 'borgo': 'borgo-r3', 'molo': 'molo-r3', 'nevaio': 'nevaio-r3'}
SPEEDS = [6.0, 5.75, 5.5, 5.2, 5.0, 4.5, 4.0, 2.6]
FF = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'yaw', 'pitch', 'mv_jump_buffer', 'step_distance', 'height']
FI = ['crouch', 'grounded', 'last_jump', 'alive', 'mv_stand_blocked', 'movement_mode', 'ladder_id', 'mv_reattach_until']
CPU_NAME = {'mv_jump_buffer': 'jump_buffer', 'mv_stand_blocked': 'stand_blocked', 'mv_reattach_until': 'reattach_until'}
SCEN = ['flat', 'stairs', 'wall', 'jump', 'air', 'land', 'crouch', 'ladder', 'locked', 'frozen', 'dead']

_W = None; _MAP = None


def load_static(name):
    return json.load(open(f'{ROOT}/data/static/static_{MAPS[name]}-v54.json'))['map']


def snap(s, lids):
    f = np.array([float(getattr(s, CPU_NAME.get(k, k), 0.0) or 0.0) for k in FF[:-1]] + [float(s.height)])
    i = np.array([bool(s.crouch), bool(s.grounded), bool(s.last_jump), bool(s.alive), bool(s.stand_blocked),
                  GM.MODES[s.movement_mode], lids.index(s.ladder_id) if s.ladder_id is not None else -1, int(s.reattach_until)])
    return f, i


def scenario(a, s0, s1, trace):
    if not s0['alive']: return 'dead'
    if s0['movement_mode'] == 2 or s1['movement_mode'] == 2 or a['ladder'] != 'none': return 'ladder'
    if a['_locked']: return 'locked'
    if a['_freeze']: return 'frozen'
    wall = any(t['hit'] is not None and any(not c[2] for c in t['hit'][2]) for t in trace)
    if wall: return 'wall'
    if s0['grounded'] and a['jump'] and not s0['last_jump']: return 'jump'
    if not s0['grounded'] or not s1['grounded']: return 'air' if not (s1['grounded'] and not s0['grounded']) else 'land'
    if s0['crouch'] or a['crouch']: return 'crouch'
    if abs(s1['z'] - s0['z']) > 1e-4: return 'stairs'
    return 'flat'


def rollout(args):
    """one player: CPU MeshMover with random held actions"""
    seed, ticks = args
    rng = np.random.default_rng(seed)
    W = _W; m = _MAP
    mv = MM.MeshMover()
    lids = [l['id'] for l in W.ladders]
    nodes = [n['position'] for n in m['nav']['nodes']]
    s = PlayerState()
    u = rng.random()
    kind = 'ladder' if (W.ladders and u < 0.2) else ('perturb' if u < 0.4 else ('sprint' if u < 0.65 else 'walk'))
    if kind == 'ladder':
        l = W.ladders[rng.integers(len(W.ladders))]
        if rng.random() < 0.5:
            s.x, s.y, s.z = l['bottom_exit'][0] + rng.normal(0, .3), l['bottom_exit'][1] + rng.normal(0, .3), l['bottom'] + 0.00011
        else:
            s.x, s.y, s.z = l['rail'][0], l['rail'][1], rng.uniform(l['bottom'], l['top'])
            s.movement_mode = 'ladder'; s.ladder_id = l['id']; s.grounded = False
    else:
        p = nodes[rng.integers(len(nodes))]
        s.x, s.y, s.z = p[0] + rng.normal(0, .5), p[1] + rng.normal(0, .5), p[2] + 0.00011
    s.yaw = rng.uniform(-math.pi, math.pi); s.pitch = rng.uniform(-1, 1)
    s.jump_buffer = 0.0
    if kind == 'perturb':          # airborne / random velocity / crouched start
        s.z += rng.uniform(0, 2.5); s.grounded = rng.random() < 0.3
        s.vx, s.vy, s.vz = rng.normal(0, 3), rng.normal(0, 3), rng.uniform(-8, 8)
        s.crouch = rng.random() < 0.3
    if rng.random() < 0.01: s.alive = False
    M = float(SPEEDS[rng.integers(len(SPEEDS))])
    out = {k: [] for k in ('pre_f', 'pre_i', 'post_f', 'post_i', 'af', 'ab', 'lad', 'scen')}
    a = None; crouch = False; walk = False
    lockp = 0.01 if rng.random() < 0.3 else 0.0
    frzp = 0.01 if rng.random() < 0.1 else 0.0
    sprint_strafe = float(rng.choice([0.0, 0.0, 0.5, -0.5, 1.0, -1.0]))
    for t in range(ticks):
        if t % 3 == 0:
            def axis():
                u = rng.random()
                return float(rng.choice([-1.0, 0.0, 1.0, 1.0])) if u < 0.7 else float(rng.uniform(-1.3, 1.3))
            if rng.random() < (0.05 if kind == 'sprint' else 0.1): crouch = not crouch
            if rng.random() < 0.1: walk = not walk
            lad = 'none'
            if W.ladders:
                u = rng.random()
                lad = 'attach' if u < (0.4 if kind == 'ladder' else 0.02) else ('detach' if u < (0.45 if kind == 'ladder' else 0.03) else 'none')
            if kind == 'sprint':
                a = dict(move_forward=1.0, move_right=sprint_strafe, yaw_rate=0.0, pitch_rate=0.0, jump=bool(rng.random() < 0.04),
                         crouch=crouch, walk=False, ladder=lad, climb=0.0)
            else:
                a = dict(move_forward=axis(), move_right=axis() if rng.random() < 0.6 else 0.0,
                         yaw_rate=float(np.clip(rng.normal(0, 0.6), -1, 1)), pitch_rate=float(np.clip(rng.normal(0, 0.3), -1, 1)),
                         jump=bool(rng.random() < 0.1), crouch=crouch, walk=walk, ladder=lad,
                         climb=float(rng.choice([0.0, 0.0, 1.0, -1.0, rng.uniform(-1, 1)])))
            a['_locked'] = bool(rng.random() < lockp); a['_freeze'] = bool(rng.random() < frzp)
        pre_f, pre_i = snap(s, lids)
        pre = dict(zip(FI, pre_i)); pre.update(zip(FF, pre_f))
        mv.trace = []
        aa = dict(a)
        if a['_freeze']:                      # Arena buy phase
            aa.pop('jump', None); aa['move_forward'] = 0.0; aa['move_right'] = 0.0; s.vx = s.vy = 0.0
        mv.step(s, aa, M, W, (), t, a['_locked'])
        post_f, post_i = snap(s, lids)
        post = dict(zip(FI, post_i)); post.update(zip(FF, post_f))
        out['pre_f'].append(pre_f); out['pre_i'].append(pre_i); out['post_f'].append(post_f); out['post_i'].append(post_i)
        out['af'].append([a['move_forward'], a['move_right'], a['yaw_rate'], a['pitch_rate'], a['climb'], M])
        out['ab'].append([a['jump'], a['crouch'], a['walk'], a['_locked'], a['_freeze']])
        out['lad'].append(GM.LADDER[a['ladder']])
        out['scen'].append(SCEN.index(scenario(a, pre, post, mv.trace)))
    return {k: np.array(v) for k, v in out.items()}


def init_worker(name, cache):
    global _W, _MAP
    _MAP = load_static(name)
    _W = GM.make_world(_MAP, f'{ROOT}/ref/assets', cache)


def to_state(f, i, dev):
    """f [P,11] float64, i [P,8] int64 -> GpuMover state dict (flat [P])"""
    st = {k: torch.as_tensor(np.ascontiguousarray(f[:, n]), device=dev) for n, k in enumerate(FF)}
    st['crouch'] = torch.as_tensor(i[:, 0] != 0, device=dev); st['grounded'] = torch.as_tensor(i[:, 1] != 0, device=dev)
    st['last_jump'] = torch.as_tensor(i[:, 2] != 0, device=dev); st['alive'] = torch.as_tensor(i[:, 3] != 0, device=dev)
    st['mv_stand_blocked'] = torch.as_tensor(i[:, 4] != 0, device=dev)
    st['movement_mode'] = torch.as_tensor(i[:, 5], dtype=torch.int8, device=dev)
    st['ladder_id'] = torch.as_tensor(i[:, 6], dtype=torch.int16, device=dev)
    st['mv_reattach_until'] = torch.as_tensor(i[:, 7], dtype=torch.int32, device=dev)
    return st


def to_action(af, ab, lad, dev):
    ac = {k: torch.as_tensor(np.ascontiguousarray(af[:, n]), device=dev) for n, k in enumerate(GM.ACT_F64)}
    for n, k in enumerate(GM.ACT_BOOL): ac[k] = torch.as_tensor(ab[:, n] != 0, device=dev)
    ac['ladder'] = torch.as_tensor(lad, dtype=torch.int8, device=dev)
    return ac, torch.as_tensor(np.ascontiguousarray(af[:, 5]), device=dev), torch.as_tensor(ab[:, 3] != 0, device=dev), \
        torch.as_tensor(ab[:, 4] != 0, device=dev)


def compare(st, post_f, post_i, dev):
    """per-record: max position error, max velocity error, bitwise exact, discrete mismatch"""
    ref = to_state(post_f, post_i, dev)
    pos = torch.stack([(st[k] - ref[k]).abs() for k in 'xyz']).amax(0)
    vel = torch.stack([(st[k] - ref[k]).abs() for k in ('vx', 'vy', 'vz')]).amax(0)
    other = torch.stack([(st[k] - ref[k]).abs() for k in ('yaw', 'pitch', 'step_distance', 'height', 'mv_jump_buffer')]).amax(0)
    exact = torch.ones_like(pos, dtype=torch.bool)
    for k in FF: exact &= (st[k] == ref[k])
    disc = torch.zeros_like(exact)
    for k in ('crouch', 'grounded', 'last_jump', 'mv_stand_blocked', 'movement_mode', 'ladder_id', 'mv_reattach_until'): disc |= (st[k] != ref[k])
    return [t.cpu().numpy() for t in (pos, vel, other, exact & ~disc, disc)]


def dist(v):
    return dict(p50=float(np.percentile(v, 50)), p90=float(np.percentile(v, 90)), p99=float(np.percentile(v, 99)),
                p999=float(np.percentile(v, 99.9)), max=float(v.max()))


def summarize(pos, vel, other, exact, disc, scen):
    def one(msk):
        n = int(msk.sum())
        if not n: return None
        p = pos[msk]; v = vel[msk]
        return dict(n=n, exact=float(exact[msk].mean()), discrete_mismatch=int(disc[msk].sum()), pos=dist(p), vel=dist(v),
                    other_max=float(other[msk].max()), pos_le_1e12=float((p <= 1e-12).mean()), pos_le_1e9=float((p <= 1e-9).mean()),
                    pos_le_1e6=float((p <= 1e-6).mean()), vel_le_1e9=float((v <= 1e-9).mean()))
    out = {'all': one(np.ones(len(pos), bool))}
    for k, sc in enumerate(SCEN):
        r = one(scen == k)
        if r: out[sc] = r
    return out


def show(title, rep):
    print(f'  {title}', flush=True)
    print('    %-8s %8s %9s %5s %10s %10s %10s %10s %10s' % ('scenario', 'n', 'exact', 'disc', 'pos p99', 'pos max', 'vel p99', 'vel max', '<=1e-9m'), flush=True)
    for sc, v in rep.items():
        print('    %-8s %8d %8.4f%% %5d %10.2e %10.2e %10.2e %10.2e %9.4f%%' % (
            sc, v['n'], v['exact'] * 100, v['discrete_mismatch'], v['pos']['p99'], v['pos']['max'], v['vel']['p99'], v['vel']['max'],
            v['pos_le_1e9'] * 100), flush=True)


def bench(gm, name, m, dev, Es=(1024, 4096, 16384), ticks=60):
    nodes = np.array([n['position'] for n in m['nav']['nodes']])
    res = {}
    g = torch.Generator(device=dev); g.manual_seed(0)
    for E in Es:
        shape = (E, 10)
        st = gm.new_state(shape)
        rng = np.random.default_rng(E)
        p = nodes[rng.integers(len(nodes), size=E * 10)]
        st['x'].copy_(torch.tensor(p[:, 0] + rng.normal(0, .5, E * 10)).view(shape))
        st['y'].copy_(torch.tensor(p[:, 1] + rng.normal(0, .5, E * 10)).view(shape))
        st['z'].copy_(torch.tensor(p[:, 2] + 0.00011).view(shape))
        st['yaw'].copy_(torch.rand(shape, generator=g, device=dev, dtype=torch.float64) * 6.28 - 3.14)
        M = torch.full(shape, 5.5, dtype=torch.float64, device=dev)

        def act():
            r = lambda: torch.rand(shape, generator=g, device=dev)
            return gm.prep_action(dict(move_forward=(r() * 3).floor() - 1, move_right=(r() * 3).floor() - 1,
                                       yaw_rate=r() * 2 - 1, pitch_rate=torch.zeros(shape, device=dev), jump=r() < 0.05,
                                       crouch=r() < 0.15, walk=r() < 0.1), shape)
        acts = [act() for _ in range(ticks // 3 + 1)]
        L = torch.zeros(shape, dtype=torch.bool, device=dev)
        for t in range(30): gm.step(st, acts[(t // 3) % len(acts)], M, L, t)     # warm-up / spread out
        torch.cuda.synchronize(); t0 = time.time()
        for t in range(ticks): gm.step(st, acts[t // 3], M, L, 30 + t)
        torch.cuda.synchronize(); dt = (time.time() - t0) / ticks
        moved = float(torch.sqrt(st['vx'] ** 2 + st['vy'] ** 2).mean())
        n = gm.candidates(st['x'], st['y'], st['z'], st['vx'], st['vy'], st['vz']).double()
        res[E] = dict(ms_per_tick=dt * 1e3, player_ticks_per_s=E * 10 / dt, mean_speed=moved, err=gm.errors(),
                      cand_mean=float(n.mean()), cand_p99=float(n.quantile(0.99)) if n.numel() < 2 ** 24 else None, cand_max=float(n.max()))
        print(f'  {name} E={E:6d} x10: {dt * 1e3:8.2f} ms/tick  {E * 10 / dt / 1e6:7.2f} M player-ticks/s  (mean speed {moved:.2f}, '
              f'candidates mean {res[E]["cand_mean"]:.0f} max {res[E]["cand_max"]:.0f}, err {gm.errors()})', flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', default='dust2,borgo,molo,nevaio')
    ap.add_argument('--players', type=int, default=2000)
    ap.add_argument('--ticks', type=int, default=150)
    ap.add_argument('--procs', type=int, default=48)
    ap.add_argument('--bench', action='store_true')
    ap.add_argument('--no-parity', action='store_true')
    ap.add_argument('--out', default=f'{ROOT}/tmp/gpu_move/parity.json')
    ap.add_argument('--cache', default=f'{ROOT}/tmp/meshcache')
    a = ap.parse_args()
    dev = torch.device('cuda')
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    report = {}
    # helper parity (CPython math.hypot port, CUDA sin/cos vs glibc)
    rng = np.random.default_rng(1)
    xs = np.concatenate([rng.normal(0, 3, 200000), rng.uniform(-1, 1, 100000) * 10.0 ** rng.integers(-8, 3, 100000)])
    ys = np.concatenate([rng.normal(0, 3, 200000), rng.uniform(-1, 1, 100000) * 10.0 ** rng.integers(-8, 3, 100000)])
    hg = GM._ext().dbg_hypot(torch.tensor(xs, device=dev), torch.tensor(ys, device=dev)).cpu().numpy()
    hc = np.array([math.hypot(x, y) for x, y in zip(xs, ys)])
    ang = rng.uniform(-math.pi, math.pi, 300000)
    tg = GM._ext().dbg_trig(torch.tensor(ang, device=dev)).cpu().numpy()
    tc = np.array([(math.sin(v), math.cos(v)) for v in ang])
    report['helpers'] = dict(hypot_bitwise=float((hg == hc).mean()), sincos_bitwise=float((tg == tc).all(1).mean()),
                             sincos_maxabs=float(np.abs(tg - tc).max()))
    print('helpers', report['helpers'], flush=True)
    import multiprocessing as mp
    for name in a.maps.split(','):
        m = load_static(name)
        try:
            init_worker(name, a.cache)
        except FileNotFoundError as e:
            print(name, 'skipped:', e); continue
        t0 = time.time()
        gm = GM.GpuMover(world=_W, device=dev, whitebox=bool(m.get('boxes')))
        print(f'{name}: {len(_W.tri)} triangles, grid {_W.nx}x{_W.nz}, ladders {len(_W.ladders)}, GPU world {gm.tri.numel() * 8 / 1e6:.0f} MB'
              f' ({time.time() - t0:.1f}s)', flush=True)
        rep = {}
        if not a.no_parity:
            t0 = time.time()
            ctx = mp.get_context('fork')
            with ctx.Pool(a.procs) as pool:
                rolls = pool.map(rollout, [(sum(map(ord, name)) * 1000003 + k, a.ticks) for k in range(a.players)], chunksize=4)
            R = {k: np.stack([r[k] for r in rolls]) for k in rolls[0]}           # [players, ticks, ...]
            n, T = R['af'].shape[:2]
            tcpu = time.time() - t0
            print(f'  CPU rollouts: {n * T} ticks in {tcpu:.1f}s', flush=True)
            flat = lambda k: R[k].reshape(n * T, *R[k].shape[2:])
            scen = flat('scen')
            tick = np.tile(np.arange(T, dtype=np.int32), n)
            # (1) teacher-forced single tick
            st = to_state(flat('pre_f'), flat('pre_i'), dev)
            ac, M, L, Fz = to_action(flat('af'), flat('ab'), flat('lad'), dev)
            gm.reset_errors()
            gm.step(st, ac, M, L, torch.as_tensor(tick, device=dev), Fz)
            torch.cuda.synchronize()
            pos, vel, other, exact, disc = compare(st, flat('post_f'), flat('post_i'), dev)
            rep['single_tick'] = summarize(pos, vel, other, exact, disc, scen)
            rep['single_tick']['errors'] = gm.errors()
            cn = gm.candidates(*[to_state(flat('pre_f'), flat('pre_i'), dev)[k] for k in ('x', 'y', 'z', 'vx', 'vy', 'vz')]).cpu().numpy()
            rep['candidates'] = dict(mean=float(cn.mean()), p99=float(np.percentile(cn, 99)), max=int(cn.max()))
            show(f'{name} single tick (teacher forced), n={n * T}, GPU errors {gm.errors()}, candidates mean {cn.mean():.0f} p99 '
                 f'{np.percentile(cn, 99):.0f} max {cn.max()}', {k: v for k, v in rep['single_tick'].items() if k != 'errors'})
            bad = np.nonzero(~exact)[0][:3]
            for b in bad:
                print('    mismatch', SCEN[scen[b]], 'pos', pos[b], 'vel', vel[b], flush=True)
            # (2) free running: GPU steps every player from its initial state with the recorded actions
            st = to_state(R['pre_f'][:, 0], R['pre_i'][:, 0], dev)
            first_div = np.full(n, T); maxpos = np.zeros(n); maxvel = np.zeros(n); disc_n = 0
            tick_t = torch.zeros(n, dtype=torch.int32, device=dev)
            for t in range(T):
                ac, M, L, Fz = to_action(R['af'][:, t], R['ab'][:, t], R['lad'][:, t], dev)
                tick_t.fill_(t)
                gm.step(st, ac, M, L, tick_t, Fz)
                p, v, o, ex, dsc = compare(st, R['post_f'][:, t], R['post_i'][:, t], dev)
                first_div[(~ex) & (first_div == T)] = t
                maxpos = np.maximum(maxpos, p); maxvel = np.maximum(maxvel, v); disc_n += int(dsc.sum())
            rep['free_run'] = dict(players=n, ticks=T, bitwise_whole_rollout=float((first_div == T).mean()), discrete_mismatch=disc_n,
                                   pos_max=float(maxpos.max()), pos_p99=float(np.percentile(maxpos, 99)), vel_max=float(maxvel.max()),
                                   frac_maxpos_le_1e9=float((maxpos <= 1e-9).mean()), frac_maxpos_le_1e6=float((maxpos <= 1e-6).mean()),
                                   frac_maxpos_le_1e3=float((maxpos <= 1e-3).mean()))
            print('  free-run:', rep['free_run'], flush=True)
        if a.bench:
            rep['bench'] = bench(gm, name, m, dev)
        report[name] = rep
        json.dump(report, open(a.out, 'w'), indent=1)
    print('wrote', a.out)


if __name__ == '__main__':
    main()
