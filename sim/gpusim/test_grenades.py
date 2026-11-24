"""Parity gpusim.grenades.GpuGrenades vs fpssim.grenades.GrenadeSystem, lockstep tick by tick.

usage (server, from sim/):  python3 -m gpusim.test_grenades [--map dust2-cs2-v54|nevaio-r3-v54|borgo-r3-v54|molo-r3-v54]
                                [--mode fuzz|rec|bench|all] [--E 48] [--ticks 1400] [--seed 1]

fuzz   E independent matches; ten players per match placed at recorded positions (or clustered), random nades, random throw
       actions (prime / release / cancel / mode / kind, incl. illegal ones), random bullets (smoke carves, aimed at smokes),
       random smoke_blocks queries.  The CPU GrenadeSystem and the GPU module get the identical inputs every tick and the
       whole state is compared every tick: projectiles (pos, vel, rest, bounces), events (throw, bounce, detonate, flash,
       hits, decoy pulses, fire extinguish), smokes + carves, fires + cells, decoys, player hp / armor / alive / blind /
       throw_ready / nades, smoke_blocks booleans.
rec    one match per recorded throw (thrower state of the recording at the throw tick, all other players as recorded): the
       real throws of the platform games re-simulated by both implementations (needs tmp/gren_ext + data/rec).
bench  throughput of handle_throw + step + smoke_blocks at E = 4096 / 16384.
"""
import argparse, collections, glob, json, math, os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch

from fpssim import grenades as CG
from gpusim.grenades import GpuGrenades, KINDS, MODES, STATES

ROOT = os.environ.get('FPS_ROOT', './fps_data')
DEV = 'cuda'
EYE = (1.62, 0.95)


# ----------------------------------------------------------------------------------------------------------- world
def load_world(map_id):
    d = json.load(open(f'{ROOT}/data/static/static_{map_id}.json'))
    st = d.get('map', d)
    mesh = None
    if not st.get('boxes'):
        from fpssim.trimesh import TriMesh
        col = st['collision'][0]
        mesh = TriMesh(f'{ROOT}/ref/assets/map_positions.f32', col['basis'], col['offset'])
    return st, mesh


def positions_pool(map_id, n_frames=600):
    """player positions from recorded frames of that map (frames.npz of a recorded game)."""
    pool = []
    for d in sorted(glob.glob(f'{ROOT}/data/rec/*/frames.npz')):
        job = os.path.dirname(d)
        try:
            sid = json.loads(open(job + '/static.json').readline())['map']['id']
        except Exception:
            continue
        if sid != map_id:
            continue
        z = np.load(d)
        P, Gf, cols = z['P'], z['G'], list(z['cols'])
        c = {k: i for i, k in enumerate(cols)}
        ok = [f for f in range(0, len(P), 11) if Gf[f, 8] == 1]
        for f in ok[:n_frames]:
            pool.append({k: P[f, :, c[k]].astype(float) for k in ('x', 'y', 'z', 'yaw', 'pitch', 'vx', 'vy', 'vz', 'crouch', 'alive')})
        if len(pool) >= n_frames:
            break
    return pool


# ----------------------------------------------------------------------------------------------------------- scenario
def random_nades(rng):
    n = np.zeros(6, np.int32)
    tot = rng.integers(0, 5)
    for _ in range(tot):
        k = rng.integers(0, 6)
        cap = 2 if k == 1 else 1
        if n[k] < cap:
            n[k] += 1
    return n


def make_fuzz_scenarios(pool, E, rng, cluster_frac=0.5):
    scs = []
    for e in range(E):
        fr = pool[rng.integers(len(pool))]
        pl = []
        cluster = rng.random() < cluster_frac
        if cluster:
            c = rng.integers(10)
            cx, cy, cz = fr['x'][c], fr['y'][c], fr['z'][c]
            rad = rng.uniform(1.0, 9.0)
        for s in range(10):
            p = dict(seat=s, team=s // 5, alive=bool(fr['alive'][s] > .5) or cluster, x=float(fr['x'][s]), y=float(fr['y'][s]),
                     z=float(fr['z'][s]), yaw=float(fr['yaw'][s]), pitch=float(fr['pitch'][s]), vx=float(fr['vx'][s]),
                     vy=float(fr['vy'][s]), vz=float(fr['vz'][s]), crouch=bool(fr['crouch'][s] > .5),
                     hp=100.0, armor=float(rng.choice([0.0, 91.7, 100.0])), nades=random_nades(rng), throw_ready=0,
                     blind_until=0, blind_start=0)
            if cluster:
                a = rng.uniform(0, 2 * math.pi); r = rad * math.sqrt(rng.random())
                p['x'], p['y'], p['z'] = cx + r * math.cos(a), cy + r * math.sin(a), cz + rng.uniform(-0.2, 0.2)
                p['yaw'] = float(rng.uniform(-math.pi, math.pi)); p['pitch'] = float(rng.uniform(-0.6, 0.6))
            if rng.random() < 0.1:
                p['hp'] = float(rng.integers(1, 100))
            pl.append(p)
        scs.append(dict(players=pl, fuzz=True, seed=int(rng.integers(1 << 30)), t0=1000 + int(rng.integers(0, 50)) * 3))
    return scs


def make_stack_scenarios(pool, E, rng):
    """smoke first, then HE / molotov / flash from (almost) the same spot: explosion carves, fire vs smoke extinguish, flash + blind,
    HE damage; random bullets aimed at the smokes.  Players clustered around the throw spot."""
    scs = []
    for e in range(E):
        fr = pool[rng.integers(len(pool))]
        o = rng.integers(10)
        cx, cy, cz, yaw, pitch = fr['x'][o], fr['y'][o], fr['z'][o], fr['yaw'][o], fr['pitch'][o]
        pl = []
        for s_ in range(10):
            a = rng.uniform(0, 2 * math.pi); r_ = 6.0 * math.sqrt(rng.random())
            pl.append(dict(seat=s_, team=s_ // 5, alive=True, x=float(cx + r_ * math.cos(a)), y=float(cy + r_ * math.sin(a)), z=float(cz + rng.uniform(-0.1, 0.1)),
                           yaw=float(rng.uniform(-math.pi, math.pi)), pitch=float(rng.uniform(-0.5, 0.5)), vx=0.0, vy=0.0, vz=0.0,
                           crouch=bool(rng.random() < 0.2), hp=100.0, armor=float(rng.choice([0.0, 91.7, 100.0])), nades=np.zeros(6, np.int32),
                           throw_ready=0, blind_until=0, blind_start=0))
        script = {}
        t_s = 3
        # throwers 0..4 stand at the throw spot with the same view; each gets one kind
        for k_, (q, kd) in enumerate(((0, 2), (1, 0), (2, 3), (3, 1), (4, 0))):
            pl[q].update(x=float(cx), y=float(cy), z=float(cz), yaw=float(yaw + rng.normal(0, 0.05)), pitch=float(pitch + rng.normal(0, 0.05)),
                         crouch=False)
            pl[q]['nades'][kd] = 1
            t_th = t_s if k_ == 0 else int(t_s + rng.integers(100, 700))
            md = int(rng.integers(0, 3)) if k_ else 0
            script[t_th - 1] = script.get(t_th - 1, []) + [(q, kd, 1, md)]
            script[t_th] = script.get(t_th, []) + [(q, kd, 2, md)]
        scs.append(dict(players=pl, fuzz=False, bullets=True, seed=int(rng.integers(1 << 30)), t0=2000 + int(rng.integers(0, 50)) * 3, script=script))
    return scs


def make_burst_scenarios(pool, E, rng):
    """all ten players with a full legal inventory (HE, 2 flash, smoke) throw as fast as the 24-tick cooldown allows (latched
    releases), so that the projectile table (P) overflows: tests the cap behaviour, the cooldown and the latched release."""
    scs = []
    for e in range(E):
        fr = pool[rng.integers(len(pool))]
        o = rng.integers(10)
        cx, cy, cz = fr['x'][o], fr['y'][o], fr['z'][o]
        pl = []
        script = {}
        for s_ in range(10):
            a = rng.uniform(0, 2 * math.pi); r_ = 5.0 * math.sqrt(rng.random())
            nd = np.zeros(6, np.int32); nd[0] = 1; nd[1] = 2; nd[2] = 1
            pl.append(dict(seat=s_, team=s_ // 5, alive=True, x=float(cx + r_ * math.cos(a)), y=float(cy + r_ * math.sin(a)), z=float(cz),
                           yaw=float(rng.uniform(-math.pi, math.pi)), pitch=float(rng.uniform(-0.5, 0.5)), vx=0.0, vy=0.0, vz=0.0,
                           crouch=False, hp=100.0, armor=0.0, nades=nd, throw_ready=0, blind_until=0, blind_start=0))
            order = [int(k) for k in rng.permutation([0, 1, 1, 2])]
            t = int(rng.integers(0, 6))
            for kd in order:                                        # prime, release one tick later (latched until throw_ready)
                script.setdefault(t, []).append((s_, kd, 1, int(rng.integers(0, 3))))
                script.setdefault(t + 1, []).append((s_, kd, 2, int(rng.integers(0, 3))))
                t += int(rng.choice([2, 25, 26]))
        scs.append(dict(players=pl, fuzz=False, bullets=False, seed=int(rng.integers(1 << 30)), t0=3000, script=script))
    return scs


# ----------------------------------------------------------------------------------------------------------- engine
class Cmp:
    def __init__(self):
        self.mx = collections.defaultdict(float)
        self.bad = collections.Counter()
        self.n = collections.Counter()
        self.examples = collections.defaultdict(list)
        self.dead_envs = set()

    def num(self, name, a, b, e, t, tol=1e-9):
        d = float(np.max(np.abs(np.asarray(a, float) - np.asarray(b, float)))) if np.size(a) else 0.0
        self.n[name] += 1
        if d > self.mx[name]:
            self.mx[name] = d
        if not d <= tol:
            self.bad[name] += 1
            self.dead_envs.add(e)
            if len(self.examples[name]) < 4:
                self.examples[name].append((e, t, a if np.size(a) < 6 else '...', b if np.size(b) < 6 else '...', d))

    def eq(self, name, a, b, e, t):
        self.n[name] += 1
        if a != b:
            self.bad[name] += 1
            self.dead_envs.add(e)
            if len(self.examples[name]) < 4:
                self.examples[name].append((e, t, a, b))

    def report(self):
        print('%-28s %9s %7s %12s' % ('check', 'n', 'bad', 'max|diff|'))
        for k in sorted(self.n):
            print('%-28s %9d %7d %12.3g' % (k, self.n[k], self.bad[k], self.mx.get(k, 0.0)))
        for k, v in self.examples.items():
            for x in v:
                print('  mismatch', k, x)
        return sum(self.bad.values())


def np_state(scs, dev):
    E = len(scs)
    f = lambda k: torch.tensor([[p[k] for p in sc['players']] for sc in scs], dtype=torch.float64, device=dev)
    b = lambda k: torch.tensor([[p[k] for p in sc['players']] for sc in scs], dtype=torch.bool, device=dev)
    i = lambda k: torch.tensor([[p[k] for p in sc['players']] for sc in scs], dtype=torch.int32, device=dev)
    s = {k: f(k) for k in ('x', 'y', 'z', 'vx', 'vy', 'vz', 'yaw', 'pitch', 'hp', 'armor')}
    s['alive'] = b('alive'); s['crouch'] = b('crouch')
    for k in ('blind_until', 'blind_start', 'throw_ready'):
        s[k] = i(k)
    s['nades'] = torch.tensor(np.stack([np.stack([p['nades'] for p in sc['players']]) for sc in scs]), dtype=torch.int32, device=dev)
    s['death_tick'] = torch.full((E, 10), -1, dtype=torch.int32, device=dev)
    s['tick'] = torch.zeros(E, dtype=torch.int32, device=dev)
    return s


def cpu_players(sc):
    out = []
    for p in sc['players']:
        q = dict(p)
        q['nades'] = {KINDS[k]: int(n) for k, n in enumerate(p['nades']) if n > 0}
        out.append(q)
    return out


def dump_gpu(G, s):
    g = {k: v.detach().cpu().numpy() for k, v in G.g.items()}
    for k in ('x', 'y', 'z', 'hp', 'armor', 'alive', 'blind_until', 'blind_start', 'throw_ready', 'nades', 'death_tick'):
        g['P_' + k] = s[k].cpu().numpy()
    return g


def compare_tick(C, g, ev, th, evn, scs, cgs, cpl, t, lists):
    """compare GPU dump g / event tensors vs the CPU systems.  evn[e]: CPU events of this tick."""
    E = len(scs)
    P = G_dims['P']
    for e in range(E):
        gs = cgs[e]
        # ---- projectiles
        gp = [p for p in range(P) if g['pr_kind'][e, p] >= 0]
        gp.sort(key=lambda p: g['pr_serial'][e, p])
        cpu = gs.projectiles
        C.eq('proj_count', len(cpu), len(gp), e, t)
        if len(cpu) == len(gp):
            for a, p in zip(cpu, gp):
                C.eq('proj_kind', KINDS.index(a.kind), int(g['pr_kind'][e, p]), e, t)
                C.num('proj_pos', (a.x, a.y, a.z), g['pr_pos'][e, p], e, t)
                C.num('proj_vel', (a.vx, a.vy, a.vz), g['pr_vel'][e, p], e, t)
                C.eq('proj_rest_bounces', (bool(a.resting), a.bounces, bool(a.ignite)),
                     (bool(g['pr_rest'][e, p]), int(g['pr_bounces'][e, p]), bool(g['pr_ign'][e, p])), e, t)
        # ---- smokes / carves
        gsm = sorted([s for s in range(g['sm_valid'].shape[1]) if g['sm_valid'][e, s]], key=lambda s: g['sm_serial'][e, s])
        C.eq('smoke_count', len(gs.smokes), len(gsm), e, t)
        if len(gs.smokes) == len(gsm):
            for a, s in zip(gs.smokes, gsm):
                C.num('smoke_pos', a['position'], g['sm_pos'][e, s], e, t)
                C.eq('smoke_times', (a['created'], a['expires']), (int(g['sm_created'][e, s]), int(g['sm_expires'][e, s])), e, t)
                C.eq('carve_count', len(a['carves']), int(g['sm_ncv'][e, s]), e, t)
                if len(a['carves']) == int(g['sm_ncv'][e, s]):
                    for k, c in enumerate(a['carves']):
                        ty = 2 if c['type'] == 'explosion' else 1
                        C.eq('carve_type_created', (ty, c['created']), (int(g['cv_type'][e, s, k]), int(g['cv_created'][e, s, k])), e, t)
                        if ty == 1:
                            C.num('carve_geom', list(c['start']) + list(c['end']) + [c['radius']],
                                  list(g['cv_a'][e, s, k]) + list(g['cv_b'][e, s, k]) + [g['cv_radius'][e, s, k]], e, t)
                        else:
                            C.num('carve_geom', list(c['position']) + [c['radius']], list(g['cv_a'][e, s, k]) + [g['cv_radius'][e, s, k]], e, t)
        # ---- fires
        gfi = sorted([f for f in range(g['fi_valid'].shape[1]) if g['fi_valid'][e, f]], key=lambda f: g['fi_serial'][e, f])
        C.eq('fire_count', len(gs.fires), len(gfi), e, t)
        if len(gs.fires) == len(gfi):
            for a, f in zip(gs.fires, gfi):
                ok = g['fi_cok'][e, f] > 0
                cells = g['fi_cells'][e, f][ok]
                C.eq('fire_ncells', len(a['cells']), len(cells), e, t)
                if len(a['cells']) == len(cells) and len(cells):
                    C.num('fire_cells', np.array(a['cells']), cells, e, t)
                C.eq('fire_times', (a['expires'], a['next_damage'], KINDS.index(a['kind'])),
                     (int(g['fi_expires'][e, f]), int(g['fi_next'][e, f]), int(g['fi_kind'][e, f])), e, t)
        # ---- decoys
        gdc = sorted([d for d in range(g['dc_valid'].shape[1]) if g['dc_valid'][e, d]], key=lambda d: g['dc_serial'][e, d])
        C.eq('decoy_count', len(gs.decoys), len(gdc), e, t)
        if len(gs.decoys) == len(gdc):
            for a, d in zip(gs.decoys, gdc):
                C.eq('decoy_times', (a['expires'], a['next_sound']), (int(g['dc_expires'][e, d]), int(g['dc_next'][e, d])), e, t)
                C.num('decoy_pos', a['position'], g['dc_pos'][e, d], e, t)
        # ---- players (vectorised over the 10 seats)
        pl = cpl[e]
        ca = np.array([[p['hp'], p['armor']] for p in pl]); ga = np.stack([g['P_hp'][e], g['P_armor'][e]], 1)
        C.num('p_hp_armor', ca, ga, e, t, 1e-9)
        ci = np.array([[bool(p['alive']), p['blind_until'], p['blind_start'], p['throw_ready']] for p in pl], np.int64)
        gi = np.stack([g['P_alive'][e].astype(np.int64), g['P_blind_until'][e], g['P_blind_start'][e], g['P_throw_ready'][e]], 1)
        C.eq('p_alive_blind_ready', ci.tolist(), gi.tolist(), e, t)
        # ---- events of the tick
        cev = evn[e]
        cb = sorted((str(x['item_id']), tuple(np.round(x['position'], 6))) for x in cev if x['type'] == 'bounce')
        gb = sorted((str(i_), tuple(np.round(pos, 6))) for i_, pos in lists['bounce'][e])
        C.eq('ev_bounce_n', len(cb), len(gb), e, t)
        for a in cev:
            if a['type'] == 'bounce':
                m = [q for q in lists['bounce'][e] if str(q[0]) == str(a['item_id'])]
                if m:
                    C.num('ev_bounce_pos', a['position'], m[0][1], e, t)
        cd = sorted((str(x['item_id']), x['effect']) for x in cev if x['type'] == 'detonate' and x['effect'] in ('detonate', 'fire_started', 'fire_failed'))
        gd = sorted((str(i_), c) for i_, c in lists['det'][e])
        C.eq('ev_detonate', cd, gd, e, t)
        for a in cev:
            if a['type'] == 'detonate' and a['effect'] == 'detonate':
                m = [q for q in lists['detpos'][e] if str(q[0]) == str(a['item_id'])]
                if m:
                    C.num('ev_detonate_pos', a['position'], m[0][1], e, t)
        cf = collections.defaultdict(lambda: [0, 0])
        for x in cev:
            if x['type'] == 'flash':
                cf[x['target']][0] += 1; cf[x['target']][1] = max(cf[x['target']][1], x['duration'])
        C.eq('ev_flash', sorted((s_, n, d) for s_, (n, d) in cf.items()), sorted(lists['flash'][e]), e, t)
        ch = collections.defaultdict(float); cn = collections.Counter()
        for x in cev:
            if x['type'] == 'hit':
                ch[x['target']] += x['hp_damage']; cn[x['target']] += 1
        gh = {s_: (float(ev['ev_dmg'][e, s_]), int(ev['ev_hits'][e, s_])) for s_ in range(10) if ev['ev_hits'][e, s_]}
        C.eq('ev_hit_targets', sorted((s_, cn[s_]) for s_ in ch), sorted((s_, n) for s_, (d, n) in gh.items()), e, t)
        for s_ in ch:
            if s_ in gh:
                C.num('ev_hit_amount', ch[s_], gh[s_][0], e, t)
        C.eq('ev_decoy_pulse', sum(1 for x in cev if x['type'] == 'decoy_pulse'), int(ev['ev_decoy_pulse'][e]), e, t)
        C.eq('ev_decoy_exp', sum(1 for x in cev if x['type'] == 'detonate' and x['effect'] == 'explosion'), int(ev['ev_decoy_exp'][e]), e, t)
        C.eq('ev_fire_ext', sum(1 for x in cev if x['type'] == 'detonate' and x['effect'] == 'fire_extinguished'), int(ev['ev_fire_ext'][e]), e, t)
        ck = sorted(gs.kills)
        gk = sorted((s_, int(ev['ev_killer'][e, s_]), int(ev['ev_kill_kind'][e, s_])) for s_ in range(10) if ev['ev_killer'][e, s_] >= 0)
        C.eq('ev_kills', ck, gk, e, t)
        gs.kills.clear()


G_dims = {}


def gpu_event_lists(ev, g, E):
    """GPU step event tensors -> per-env python lists (only the non-empty entries)."""
    L = {k: [[] for _ in range(E)] for k in ('bounce', 'det', 'detpos', 'flash')}
    for e, p in np.argwhere(ev['ev_bounce'] > 0):
        L['bounce'][e].append((g['pr_id'][e, p], ev['ev_bounce_pos'][e, p]))
    for e, p in np.argwhere(ev['ev_det'] > 0):
        c = ev['ev_det'][e, p]
        if c == 2:                       # removed without an event (flight cap / non finite)
            continue
        i = ev['ev_det_id'][e, p]
        L['det'][e].append((i, 'detonate')); L['detpos'][e].append((i, ev['ev_det_pos'][e, p]))
        if c == 3:
            L['det'][e].append((i, 'fire_failed'))
        elif c == 4:
            L['det'][e].append((i, 'fire_started'))
    for e, q in np.argwhere(ev['ev_flash_n'] > 0):
        L['flash'][e].append((int(q), int(ev['ev_flash_n'][e, q]), int(ev['ev_flash_dur'][e, q])))
    return L


def run_lockstep(map_id, scs, ticks, seed=1, verbose=True, smoke_q=24, vis_every=0):
    st, mesh = load_world(map_id)
    E = len(scs)
    kw = dict(P=24, S=12, F=12, D=12, NI=int(os.environ.get('GG_NI', 4)))        # GG_NI=0: sequential fire-cell fallback only
    G = GpuGrenades(st, mesh, E, DEV, **kw)
    G_dims.update(kw)
    CG.MAX_PROJECTILES = kw['P']
    contact = CG.MeshContact(mesh) if mesh is not None else CG.make_contact(st)
    if vis_every:
        from gpusim.vis import visible, whitebox_mesh
        from fpssim.visibility import visible_enemies, LOS
        from gpusim.ray import GpuMesh
        vmesh = GpuMesh(mesh.tri, DEV) if mesh is not None else whitebox_mesh(st, DEV)
        vlos = LOS(mesh) if mesh is not None else LOS(CG.BoxContact(st))
    cpl = [cpu_players(sc) for sc in scs]
    cgs = []
    for e in range(E):
        cnt = [0]

        def iid(p, kind, cnt=cnt):
            cnt[0] += 1
            return cnt[0]
        gs = CG.GrenadeSystem(contact, damage_cb=None, item_id_fn=iid)
        kl = []

        def cb(victim, attacker, hp, ab, weapon, kl=kl):
            victim['armor'] = max(0.0, victim['armor'] - ab); victim['hp'] -= hp
            if victim['hp'] <= 1e-9 and victim['alive']:
                victim['hp'] = 0.0; victim['alive'] = False; victim['nades'] = {}
                kl.append((victim['seat'], attacker, KINDS.index(weapon)))
        gs.damage_cb = cb
        gs.kills = kl
        cgs.append(gs)
    s = np_state(scs, DEV)
    C = Cmp()
    base = np.array([sc['t0'] for sc in scs], np.int64)
    rngs = [np.random.default_rng(sc['seed']) for sc in scs]
    tim = collections.Counter()
    nthrow = collections.Counter()
    stat = collections.Counter(); carve_seen = set()
    for k in range(ticks):
        tk = base + k
        s['tick'].copy_(torch.tensor(tk, dtype=torch.int32))
        # ----- kinematics + random actions (identical for both) -----
        kind = np.full((E, 10), -1, np.int32); state = np.zeros((E, 10), np.int32); mode = np.zeros((E, 10), np.int32)
        bullets = []
        for e in range(E):
            r = rngs[e]; pl = cpl[e]; t = int(tk[e]); sc = scs[e]
            for (q, kd, stt, md) in sc.get('script', {}).get(k, ()):
                kind[e, q], state[e, q], mode[e, q] = kd, stt, md
            if not (sc.get('fuzz') or sc.get('bullets')):
                continue
            for q in range(10):
                p = pl[q]
                if not p['alive']:
                    continue
                if sc.get('fuzz'):
                    p['x'] += p['vx'] / 60.0; p['y'] += p['vy'] / 60.0
                    p['yaw'] += float(r.normal(0, 0.03)); p['pitch'] = float(np.clip(p['pitch'] + r.normal(0, 0.02), -1.2, 1.2))
                    u = r.random()
                    nd = p['nades']
                    if u < 0.025:                                            # prime (sometimes an unowned / invalid kind)
                        kind[e, q] = r.integers(0, 6) if r.random() < 0.7 else (r.choice(list(KINDS.index(kk) for kk in nd)) if nd else r.integers(-1, 6))
                        state[e, q] = 1; mode[e, q] = r.integers(0, 3)
                    elif u < 0.075:                                          # release (kind may mismatch / be none)
                        kind[e, q] = -1 if r.random() < 0.3 else r.integers(0, 6); state[e, q] = 2; mode[e, q] = r.integers(0, 3)
                    elif u < 0.08:
                        state[e, q] = 3; kind[e, q] = r.integers(-1, 6)
                    else:
                        state[e, q] = 0
                        kind[e, q] = r.integers(-1, 6) if r.random() < 0.3 else -1
                if r.random() < 0.02:                                    # bullet: from the eye, aimed at a smoke when one exists
                    eye = (p['x'], p['y'], p['z'] + EYE[bool(p['crouch'])])
                    sm = cgs[e].smokes
                    if sm and r.random() < 0.8:
                        c = sm[r.integers(len(sm))]['position']
                        tgt = np.array(c) + r.normal(0, 3.0, 3)
                    else:
                        tgt = np.array(eye) + r.normal(0, 1, 3) * 15
                    dv = tgt - np.array(eye); dv /= max(1e-9, np.linalg.norm(dv))
                    ln = float(r.uniform(5, 45))
                    bullets.append((e, eye, tuple(np.array(eye) + dv * ln)))
        # ----- CPU -----
        tc = time.time()
        evn = [[] for _ in range(E)]
        for (e, a, b) in bullets:
            cgs[e].carve_bullet(a, b, int(tk[e]))
        cth = [[None] * 10 for _ in range(E)]
        for e in range(E):
            gs = cgs[e]; t = int(tk[e])
            for q in range(10):
                p = cpl[e][q]
                if not p['alive']:
                    continue
                act = dict(throw=(KINDS[kind[e, q]] if kind[e, q] >= 0 else 'none'), throw_state=STATES[state[e, q]], throw_mode=MODES[mode[e, q]])
                ev = gs.handle_throw(p, act, t)
                if ev:
                    cth[e][q] = ev[0]; nthrow[KINDS.index(ev[0]['kind'])] += 1
            evn[e] = gs.step(t, cpl[e])
        tim['cpu'] += time.time() - tc
        for e in range(E):
            for x in evn[e]:
                stat['ev:' + x['type'] + ':' + str(x.get('effect') or x.get('kind'))] += 1
            for sm in cgs[e].smokes:
                for c in sm['carves']:
                    key = (e, sm['id'], c['type'], c['created'], c['id'])
                    if key not in carve_seen:
                        carve_seen.add(key); stat['carve_created:' + c['type']] += 1
        # ----- GPU -----
        tg = time.time()
        ttk = torch.tensor(tk, dtype=torch.int32, device=DEV)
        if bullets:
            be = torch.tensor([b[0] for b in bullets], device=DEV)
            b0 = torch.tensor([b[1] for b in bullets], dtype=torch.float64, device=DEV)
            b1 = torch.tensor([b[2] for b in bullets], dtype=torch.float64, device=DEV)
            G.carve_bullet(b0, b1, be, ttk)
        for kk in ('x', 'y', 'z', 'yaw', 'pitch'):
            s[kk].copy_(torch.tensor([[p[kk] for p in cpl[e]] for e in range(E)], dtype=torch.float64))
        act = dict(throw_kind=torch.tensor(kind, device=DEV), throw_state=torch.tensor(state, device=DEV), throw_mode=torch.tensor(mode, device=DEV))
        th = G.handle_throw(s, act, None)
        ev = G.step(s, None, None)
        torch.cuda.synchronize(); tim['gpu'] += time.time() - tg
        thf = th['th_flag'].cpu().numpy(); thk = th['th_kind'].cpu().numpy(); thp = th['th_pos'].cpu().numpy()
        for e in range(E):
            for q in range(10):
                a = cth[e][q]
                C.eq('throw_flag', a is not None, bool(thf[e, q]), e, k)
                if a is not None and thf[e, q]:
                    C.eq('throw_kind', KINDS.index(a['kind']), int(thk[e, q]), e, k)
                    C.num('throw_pos', a['position'], thp[e, q], e, k)
        # ----- compare -----
        g = dump_gpu(G, s)
        evn_np = {kk: v.cpu().numpy() for kk, v in ev.items()}
        lists = gpu_event_lists(evn_np, g, E)
        compare_tick(C, g, evn_np, th, evn, scs, cgs, cpl, k, lists)
        # ----- smoke_blocks queries -----
        qs = []
        for e in range(E):
            if cgs[e].smokes:
                r = rngs[e]
                for _ in range(smoke_q // 4 + 1):
                    sm = cgs[e].smokes[r.integers(len(cgs[e].smokes))]['position']
                    a = np.array(sm) + r.normal(0, 4.0, 3); b = np.array(sm) + r.normal(0, 4.0, 3)
                    qs.append((e, a, b))
        if qs:
            sbm = G.smoke_blocks(torch.tensor(np.stack([q[1] for q in qs]), device=DEV), torch.tensor(np.stack([q[2] for q in qs]), device=DEV),
                                 torch.tensor([q[0] for q in qs], device=DEV), ttk).cpu().numpy()
            for (e, a, b), gb in zip(qs, sbm):
                cb_ = bool(cgs[e].smoke_blocks(a, b, int(tk[e]))); stat['smoke_blocks_true'] += cb_; stat['smoke_blocks_q'] += 1
                C.eq('smoke_blocks', cb_, bool(gb), e, k)
        # ----- visibility with the smokes of the module -----
        if vis_every and k % vis_every == vis_every - 1:
            vs = {kk: s[kk] for kk in ('x', 'y', 'z', 'yaw', 'pitch', 'crouch', 'alive', 'blind_start', 'blind_until')}
            out = visible(vs, vmesh, ttk, G.smoke_blocks).cpu().numpy()
            for e in range(E):
                pl = [dict(p, blind_until=int(p['blind_until']), blind_start=int(p['blind_start'])) for p in cpl[e]]
                cv = visible_enemies(pl, vlos, cgs[e].smoke_blocks, int(tk[e]))
                for o in range(10):
                    cm = {gg: set(pp) for gg, pp in cv[o]}
                    for gg in range(10):
                        if gg // 5 == o // 5:
                            continue
                        gp = {('head', 'chest', 'abdomen')[j] for j in range(3) if out[e, o, gg, j]}
                        stat['vis_pairs_visible'] += bool(cm.get(gg))
                        C.eq('vis_parts', sorted(cm.get(gg, set())), sorted(gp), e, k)
        if verbose and k % 300 == 299:
            print('  tick', k + 1, 'bad so far', sum(C.bad.values()), 'cpu %.1fs gpu %.1fs' % (tim['cpu'], tim['gpu']), flush=True)
    print('throws by kind', {KINDS[kk]: v for kk, v in sorted(nthrow.items())})
    print('coverage', dict(sorted(stat.items())))
    print('dead players at end', sum(not cpl[e][q]['alive'] for e in range(E) for q in range(10)), 'of', E * 10)
    return C, cgs


def rec_scenarios(map_id):
    """one scenario per recorded throw whose throw tick has an exact frame (thrower state after movement) on this map."""
    from validate.grenades import Game, mk_player
    ext = f'{ROOT}/tmp/gren_ext'
    worlds = {}
    scs = []
    for f in sorted(glob.glob(ext + '/*.json')):
        gm = Game(f'{ROOT}/data/rec', f, f'{ROOT}/ref/assets', worlds)
        if gm.map != map_id:
            continue
        for e in gm.events:
            if e['type'] != 'throw' or e['item_id'] not in gm.proj:
                continue
            T = e['match_tick']
            info = gm.proj[e['item_id']]['info']
            ps, exact = gm.players_at(T)               # exact frame, or the thrower state interpolated between 24 Hz frames
            if ps is None:
                continue
            pl = []
            for r in ps:
                q = mk_player(r)
                nd = np.zeros(6, np.int32)
                if q['seat'] == e['player']:
                    nd[KINDS.index(e['kind'])] = 1
                pl.append(dict(seat=q['seat'], team=q['seat'] // 5, alive=bool(q['alive']), x=q['x'], y=q['y'], z=q['z'], yaw=q['yaw'], pitch=q['pitch'],
                               vx=q['vx'] or 0.0, vy=q['vy'] or 0.0, vz=q['vz'] or 0.0, crouch=bool(q['crouch']), hp=float(q['hp'] if q['hp'] is not None else 100.0),
                               armor=float(q['armor'] or 0.0), nades=nd, throw_ready=0, blind_until=0, blind_start=0))
            pl[e['player']]['alive'] = True
            md = MODES.index(info['throw_mode'])
            kd = KINDS.index(e['kind'])
            scs.append(dict(players=pl, fuzz=False, seed=1, t0=T - 1, script={0: [(e['player'], kd, 1, md)], 1: [(e['player'], kd, 2, md)]},
                            tag=(gm.job[:8], e['item_id'], e['kind'], info['throw_mode'], exact)))
    return scs


# ----------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='dust2-cs2-v54')
    ap.add_argument('--mode', default='fuzz')
    ap.add_argument('--E', type=int, default=48)
    ap.add_argument('--ticks', type=int, default=1400)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--vis-every', type=int, default=0)
    ap.add_argument('--rate', type=float, default=1.0)
    ap.add_argument('--kinds', default='0,1,2,3,4,5')
    ap.add_argument('--profile', action='store_true')
    a = ap.parse_args()
    t0 = time.time()
    if a.mode in ('fuzz', 'all'):
        pool = positions_pool(a.map)
        print('map', a.map, 'position pool', len(pool))
        rng = np.random.default_rng(a.seed)
        scs = make_fuzz_scenarios(pool, a.E, rng)
        C, cgs = run_lockstep(a.map, scs, a.ticks, a.seed, vis_every=a.vis_every)
        nbad = C.report()
        print('FUZZ %s: %d mismatching checks, %d diverged matches of %d (%.0fs)' % (a.map, nbad, len(C.dead_envs), a.E, time.time() - t0))
    if a.mode in ('stack', 'all'):
        pool = positions_pool(a.map)
        rng = np.random.default_rng(a.seed + 100)
        scs = make_stack_scenarios(pool, a.E, rng)
        C, cgs = run_lockstep(a.map, scs, a.ticks, a.seed, vis_every=a.vis_every)
        nbad = C.report()
        print('STACK %s: %d mismatching checks, %d diverged matches of %d (%.0fs)' % (a.map, nbad, len(C.dead_envs), a.E, time.time() - t0))
    if a.mode in ('burst', 'all'):
        pool = positions_pool(a.map)
        rng = np.random.default_rng(a.seed + 200)
        scs = make_burst_scenarios(pool, a.E, rng)
        C, cgs = run_lockstep(a.map, scs, a.ticks, a.seed, vis_every=a.vis_every)
        nbad = C.report()
        print('BURST %s: %d mismatching checks, %d diverged matches of %d (%.0fs)' % (a.map, nbad, len(C.dead_envs), a.E, time.time() - t0))
    if a.mode in ('rec', 'all'):
        scs = rec_scenarios(a.map)
        print('map', a.map, 'recorded throws', len(scs), collections.Counter(sc['tag'][2] for sc in scs),
              'exact-frame', sum(sc['tag'][4] for sc in scs))
        if scs:
            C, cgs = run_lockstep(a.map, scs, a.ticks, a.seed, vis_every=a.vis_every)
            nbad = C.report()
            print('REC %s: %d mismatching checks, %d diverged throws of %d (%.0fs)' % (a.map, nbad, len(C.dead_envs), len(scs), time.time() - t0))
    if a.mode == 'bench':
        bench(a.map, ticks=a.ticks if a.ticks != 1400 else 240, rate=a.rate, profile=a.profile, kinds=tuple(int(x) for x in a.kinds.split(',')))


def bench(map_id, Es=(4096, 16384), ticks=240, rate=1.0, profile=False, kinds=(0, 1, 2, 3, 4, 5)):
    """throughput: random primes / releases / bullets on the GPU, E matches, one `tick` = handle_throw + carve_bullet + step."""
    st, mesh = load_world(map_id)
    pool = positions_pool(map_id, 600)
    rng = np.random.default_rng(0)
    for E in Es:
        sel = rng.integers(0, len(pool), E)
        f = lambda k: torch.tensor(np.stack([pool[i][k] for i in sel]), dtype=torch.float64, device=DEV)
        s = dict(x=f('x'), y=f('y'), z=f('z'), vx=f('vx'), vy=f('vy'), vz=f('vz'), yaw=f('yaw'), pitch=f('pitch'))
        s['hp'] = torch.full((E, 10), 100.0, dtype=torch.float64, device=DEV); s['armor'] = torch.zeros(E, 10, dtype=torch.float64, device=DEV)
        s['alive'] = torch.ones(E, 10, dtype=torch.bool, device=DEV); s['crouch'] = f('crouch') > .5
        for k in ('blind_until', 'blind_start', 'throw_ready'):
            s[k] = torch.zeros(E, 10, dtype=torch.int32, device=DEV)
        cap = torch.tensor([1, 2, 1, 1, 1, 1], dtype=torch.int32, device=DEV)
        s['nades'] = (torch.rand(E, 10, 6, device=DEV) * (cap + 1)).int().clamp(max=cap) * (torch.rand(E, 10, 6, device=DEV) < 0.5)
        s['nades'] = s['nades'].int().contiguous()
        s['death_tick'] = torch.full((E, 10), -1, dtype=torch.int32, device=DEV); s['tick'] = torch.zeros(E, dtype=torch.int32, device=DEV)
        G = GpuGrenades(st, mesh, E, DEV)
        gen = torch.Generator(device=DEV); gen.manual_seed(1)
        T = collections.Counter()
        for t in range(1000, 1000 + ticks):
            s['tick'].fill_(t)
            u = torch.rand(E, 10, device=DEV, generator=gen)
            act = dict(throw_kind=torch.tensor(kinds, device=DEV, dtype=torch.int32)[torch.randint(0, len(kinds), (E, 10), device=DEV, generator=gen)],
                       throw_state=torch.where(u < 0.03 * rate, 1, torch.where(u < 0.03 * rate + 0.07, 2, 0)).int(), throw_mode=torch.randint(0, 3, (E, 10), device=DEV, generator=gen, dtype=torch.int32))
            nb = E // 5
            be = torch.randint(0, E, (nb,), device=DEV, generator=gen)
            b0 = torch.stack([s['x'][be, 0], s['y'][be, 0], s['z'][be, 0] + 1.6], 1); b1 = b0 + torch.randn(nb, 3, device=DEV, dtype=torch.float64, generator=gen) * 20
            torch.cuda.synchronize(); t0 = time.time()
            G.carve_bullet(b0, b1, be, t); torch.cuda.synchronize(); t1 = time.time()
            G.handle_throw(s, act, None); torch.cuda.synchronize(); t2 = time.time()
            G.step(s, None, None); torch.cuda.synchronize(); t3 = time.time()
            qe = torch.randint(0, E, (E * 20,), device=DEV, generator=gen)
            q0 = torch.stack([s['x'][qe, 1], s['y'][qe, 1], s['z'][qe, 1] + 1.6], 1); q1 = q0 + torch.randn(len(qe), 3, device=DEV, dtype=torch.float64, generator=gen) * 20
            torch.cuda.synchronize(); t4 = time.time()
            G.smoke_blocks(q0, q1, qe, t); torch.cuda.synchronize(); t5 = time.time()
            if profile and t == 1000 + ticks - 21:
                from torch.profiler import profile as _prof, ProfilerActivity
                _p = _prof(activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU]); _p.__enter__()
            if profile and t == 1000 + ticks - 1:
                _p.__exit__(None, None, None)
                print(_p.key_averages().table(sort_by='cuda_time_total', row_limit=10, max_name_column_width=40))
            if t >= 1000 + ticks // 2:
                T['carve'] += t1 - t0; T['throw'] += t2 - t1; T['step'] += t3 - t2; T['smoke_blocks(20/env)'] += t5 - t4; T['n'] += 1
            if t == 1000 + ticks - 1:
                g = G.g
                print('   state at end: projectiles %d smokes %d fires %d decoys %d' % ((g['pr_kind'] >= 0).sum(), (g['sm_valid'] > 0).sum(), (g['fi_valid'] > 0).sum(), (g['dc_valid'] > 0).sum()))
        print('E=%d: ' % E + ', '.join('%s %.2f ms' % (k, v / T['n'] * 1e3) for k, v in T.items() if k != 'n'), flush=True)


if __name__ == '__main__':
    main()
