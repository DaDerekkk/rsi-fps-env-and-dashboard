"""End-to-end parity: gpusim.GpuSim vs the CPU fpssim Arena, same seeds, same actions, injected random numbers.

Shared harness (also used by gpusim/test_rounds.py):
  RefArena   fpssim Arena with (a) the counter-based RNG of gpusim.rounds (spread / spawn shuffle / bomb carrier / first attack
             side: InjRng below), (b) LOS for bomb pick-up / defuse on the platform mesh (Arena uses the voxel approximation),
             (c) no audio / observations (speed).  The patches live here, fpssim/ is untouched.
  Pair       one RefArena per compared match + the batched GpuSim (match e <-> GPU env e), identical action lists, per-decision
             state comparison (players, cash, weapons, bomb, round machine, event counters).
usage:  python -m gpusim.test_sim [--map dust2-cs2-v54] [--matches 4] [--rounds 3] [--seed 1] [--steps N] [--throws]
Scenario "duels": every round the director shops randomly in the buy phase, then teleports the 10 players (both sims) into
5 facing pairs on the navigation graph so that fights, kills, drops, pick-ups, bomb plant / defuse / explosion, round ends,
half-time and the match end are all exercised; the decision-by-decision comparison reports the first divergence per match.
"""
import sys, os, json, math, time, random, types, argparse, collections
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch

from fpssim import arena as FA
from fpssim.arena import Arena
from gpusim.sim import GpuSim
from gpusim import tables as TB, rounds as RD
from gpusim.rounds import u01_py

UTIL = list(TB.UTIL_IDS)
PHC = dict(buy=0, live=1, post=2)
BOMB = dict(held=0, world=1, planted=2, defused=3, exploded=4)
EVN = [n for n in RD.RC.EVENTS if n not in ('void_shot', 'dead_hit')]
WEAPON_STATE = ('w_ammo', 'w_shots', 'w_shot_ready', 'w_reload_until')


# ====================================================================================================== CPU reference
class InjRng:
    """random.Random stand-in drawing from gpusim.rounds.u01_py (same numbers as the CUDA kernels)."""
    def __init__(self, seed, ar):
        self.seed, self.ar = seed, ar; self.shuf = (-1, 0); self.n = 0; self.ctx = (0, 0)

    def randrange(self, n):
        assert n == 2
        return int(u01_py(self.seed, 4) * 2)

    def shuffle(self, x):
        ri = self.ar.round_index
        if self.shuf[0] != ri: self.shuf = (ri, 0)
        side = self.shuf[1]; self.shuf = (ri, side + 1)
        n = len(x); keys = [u01_py(self.seed, 2, ri, side * 16 + k, 0) for k in range(n)]
        x[:] = sorted(range(n), key=lambda k: (keys[k], k))

    def choice(self, seq):
        assert len(seq) == 5
        return seq[min(int(u01_py(self.seed, 3, self.ar.round_index) * 5), 4)]

    def random(self):
        n = self.n; self.n += 1; mt, seat = self.ctx
        return u01_py(self.seed, 1, mt, seat * 16 + n // 4, n % 4)


class RefArena(Arena):
    _cache = {}

    def __init__(self, match_seed, t0=0, resolve=True):
        super().__init__(); self.keep = False; self.dbg = []
        self.match_seed, self.t0, self.resolve = match_seed, t0, resolve

    def _load_map(self, map_id):
        c = RefArena._cache.get(map_id)
        if c is None:
            super()._load_map(map_id)
            if self.mesh is not None:
                mesh = self.mesh; self.geom.los = lambda a, b, m=mesh: m.los(a, b)
            elif self.static.get('boxes'):
                # whitebox maps: the GPU port only has the MeshMover (on the triangulated boxes), which beats the axis-sweep Mover of the
                # CPU Arena; give the reference the same mover / world so that the comparison is about the round logic
                from tools.meshmove_wb_validate import make_wb_world
                self.world = make_wb_world(self.static, './fps_data/tmp/meshcache')
            RefArena._cache[map_id] = (self.static, self.mesh, self.geom, self.world)
        else:
            self.static, self.mesh, self.geom, self.world = c
        if self.mesh is not None or self.static.get('boxes'):
            from fpssim.meshmove import MeshMover
            self.mover = MeshMover()
        else:
            from fpssim.movement import Mover
            self.mover = Mover()

    def reset(self, map_id=None, seed=0):
        orig = FA.random
        FA.random = types.SimpleNamespace(Random=lambda s: InjRng(self.match_seed, self))
        try:
            super().reset(map_id=map_id, seed=self.match_seed)
        finally:
            FA.random = orig
        self.rng.ar = self

    def observations(self):
        return None

    def _tick(self):
        self._tick_inner()

    def _resolve_players(self):
        if self.resolve: super()._resolve_players()

    def _do_shot(self, p, w, targets):
        self.rng.ctx = (self.tick - self.t0, p.seat); self.rng.n = 0
        if getattr(self, 'keep', False):
            from fpssim import combat as _C, accuracy as _A
            rec = self.dbg; tick = self.tick; seat = p.seat
            orig_hit, orig_dir, orig_ray = _C.ray_hit_player, _A.sample_direction, self.geom.raycast
            def hit(o, d, pos, yaw, crouch, tmax=0):
                r = orig_hit(o, d, pos, yaw, crouch, tmax); rec.append(('hit', tick, seat, tuple(pos), r)); return r
            def sd(*a, **k):
                r = orig_dir(*a, **k); rec.append(('dir', tick, seat, tuple(r))); return r
            def rc(*a, **k):
                r = orig_ray(*a, **k); rec.append(('ray', tick, seat, tuple(map(tuple, a[:2])), r[:2])); return r
            _C.ray_hit_player, _A.sample_direction, self.geom.raycast = hit, sd, rc
            try: return super()._do_shot(p, w, targets)
            finally: _C.ray_hit_player, _A.sample_direction, self.geom.raycast = orig_hit, orig_dir, orig_ray
        return super()._do_shot(p, w, targets)


# ====================================================================================================== state extraction
def cpu_state(ar):
    P = ar.players; b = ar.bomb
    f = lambda fn, dt=float: np.array([fn(p) for p in P], dtype=dt)
    d = {}
    for k in ('x', 'y', 'z'): d[k] = f(lambda p, k=k: getattr(p.ps, k))
    for k in ('vx', 'vy', 'vz', 'yaw', 'pitch'): d[k] = f(lambda p, k=k: getattr(p.ps, k))
    d['hp'] = f(lambda p: p.hp); d['armor'] = f(lambda p: p.armor); d['cash'] = f(lambda p: p.cash, int)
    d['crouch'] = f(lambda p: p.ps.crouch, int); d['grounded'] = f(lambda p: p.ps.grounded, int); d['alive'] = f(lambda p: p.alive, int)
    d['helmet'] = f(lambda p: p.helmet, int); d['kit'] = f(lambda p: p.kit, int); d['has_bomb'] = f(lambda p: p.has_bomb, int)
    d['objective'] = f(lambda p: {None: 0, 'plant': 1, 'defuse': 2}[p.objective], int); d['obj_progress'] = f(lambda p: p.obj_progress, int)
    d['active'] = f(lambda p: p.ws.active, int); d['switch_until'] = f(lambda p: p.ws.switch_until, int)
    d['zoom'] = f(lambda p: p.ws.zoom_level, int); d['kills'] = f(lambda p: p.kills, int); d['deaths'] = f(lambda p: p.deaths, int)
    d['nades'] = np.array([[p.nades.get(k, 0) for k in UTIL] for p in P], int)
    for key, fn, dt in (('w_wid', lambda w: TB.WID[w.id], int), ('w_ammo', lambda w: w.ammo, int), ('w_reserve', lambda w: w.reserve, int),
                        ('w_shots', lambda w: w.shots, int), ('w_shot_ready', lambda w: w.shot_ready, int),
                        ('w_reload_until', lambda w: -1 if w.reload_until is None else w.reload_until, int)):
        d[key] = np.array([[(fn(w) if w is not None else (-1 if key == 'w_wid' else 0)) for w in p.ws.slots] for p in P], dt)
    m = {}
    m['round_index'] = ar.round_index; m['phase'] = PHC[ar.phase]; m['tick'] = ar.tick; m['round_start_tick'] = ar.round_start_tick
    m['round_wins'] = list(ar.round_wins); m['loss_level'] = list(ar.loss_level); m['attack_team'] = ar.attack_team
    m['finished'] = int(ar.finished); m['winner'] = -1 if ar.winner is None else ar.winner
    m['bomb_state'] = BOMB[b['state']]; m['bomb_carrier'] = -1 if b['carrier'] is None else b['carrier']
    pos = b['position'] if b['position'] is not None else (0, 0, 0)
    m['bomb_pos'] = list(pos); m['bomb_site'] = -1 if not b.get('site') else ['A', 'B'].index(b['site'])
    m['bomb_fuse_end'] = -1 if b['fuse_end'] is None else b['fuse_end']
    m['bomb_planted_tick'] = -1 if b['planted_tick'] is None else b['planted_tick']
    m['n_items'] = len(ar.world_items)
    m['items'] = sorted((TB.WID[i['kind']], i['owner']) for i in ar.world_items)
    m['item_states'] = sorted((TB.WID[i['kind']], i['owner'], i['weapon'].ammo, i['weapon'].reserve, i['weapon'].shots, i['weapon'].shot_ready,
                               -1 if i['weapon'].reload_until is None else i['weapon'].reload_until) for i in ar.world_items)
    c = collections.Counter(e['type'] for e in ar.events)
    m['events'] = {n: c.get(n, 0) for n in EVN if n != 'reload_cancel'}
    return d, m


def gpu_state(sim, n):
    s = sim.s
    keys = ['x', 'y', 'z', 'vx', 'vy', 'vz', 'yaw', 'pitch', 'hp', 'armor', 'cash', 'crouch', 'grounded', 'alive', 'helmet', 'kit',
            'has_bomb', 'objective', 'obj_progress', 'active', 'switch_until', 'zoom', 'kills', 'deaths', 'nades', 'w_wid', 'w_ammo',
            'w_reserve', 'w_shots', 'w_shot_ready', 'w_reload_until', 'round_index', 'phase', 'tick', 'round_start_tick',
            'round_wins', 'loss_level', 'attack_team', 'finished', 'winner', 'bomb_state', 'bomb_carrier', 'bomb_pos', 'bomb_site',
            'bomb_fuse_end', 'bomb_planted_tick', 'wi_valid', 'wi_kind', 'wi_owner', 'ev_count']
    keys = keys + ['wi_ammo', 'wi_reserve', 'wi_shots', 'wi_shot_ready', 'wi_reload_until']
    S = {k: s[k][:n].cpu().numpy() for k in keys}
    out = []
    for e in range(n):
        d = {k: S[k][e] for k in keys[:keys.index('round_index')]}
        for k in ('crouch', 'grounded', 'alive', 'helmet', 'kit', 'has_bomb'): d[k] = d[k].astype(int)
        for k in ('cash', 'objective', 'obj_progress', 'active', 'switch_until', 'zoom', 'kills', 'deaths'): d[k] = d[k].astype(int)
        empty = d['w_wid'] < 0
        for k in ('w_ammo', 'w_reserve', 'w_shots', 'w_shot_ready', 'w_reload_until'): d[k] = np.where(empty, 0, d[k])
        m = {k: S[k][e] for k in keys[keys.index('round_index'):]}
        mm = dict(round_index=int(m['round_index']), phase=int(m['phase']), tick=int(m['tick']), round_start_tick=int(m['round_start_tick']),
                  round_wins=m['round_wins'].tolist(), loss_level=m['loss_level'].tolist(), attack_team=int(m['attack_team']),
                  finished=int(m['finished']), winner=int(m['winner']), bomb_state=int(m['bomb_state']), bomb_carrier=int(m['bomb_carrier']),
                  bomb_pos=m['bomb_pos'].tolist(), bomb_site=int(m['bomb_site']), bomb_fuse_end=int(m['bomb_fuse_end']),
                  bomb_planted_tick=int(m['bomb_planted_tick']))
        v = m['wi_valid'] > 0
        mm['item_states'] = sorted((int(S['wi_kind'][e][j]), int(S['wi_owner'][e][j]), int(S['wi_ammo'][e][j]), int(S['wi_reserve'][e][j]),
                                    int(S['wi_shots'][e][j]), int(S['wi_shot_ready'][e][j]), int(S['wi_reload_until'][e][j])) for j in np.where(v)[0])
        mm['n_items'] = int(v.sum()); mm['items'] = sorted((int(a), int(b)) for a, b in zip(m['wi_kind'][v], m['wi_owner'][v]))
        mm['events'] = {n: int(m['ev_count'][i]) for i, n in enumerate(RD.RC.EVENTS) if n in EVN and n != 'reload_cancel'}
        mm['void'] = int(m['ev_count'][RD.EV['void_shot']])
        if mm['finished'] and mm['phase'] == 3: mm['phase'] = 2              # the CPU Arena keeps 'post' after the match end
        out.append((d, mm))
    return out


FLOAT_KEYS = {'x': 1e-6, 'y': 1e-6, 'z': 1e-6, 'vx': 1e-5, 'vy': 1e-5, 'vz': 1e-5, 'yaw': 1e-6, 'pitch': 1e-6, 'hp': 1e-6, 'armor': 1e-6}


def compare(cd, cm, gd, gm, keys=None, ptol=1.0):
    """list of mismatch strings between CPU state (cd, cm) and GPU state (gd, gm)."""
    bad = []
    for k, a in cd.items():
        if keys is not None and k not in keys: continue
        b = gd[k]
        if k in FLOAT_KEYS:
            alive = cd['alive'].astype(bool)
            err = np.abs(a - b) * (1 if k in ('hp', 'armor', 'x', 'y', 'z', 'yaw', 'pitch', 'vx', 'vy', 'vz') else 1)
            if k in ('x', 'y', 'z', 'vx', 'vy', 'vz', 'yaw', 'pitch'): err = err * alive
            for i in np.where(err > FLOAT_KEYS[k] * ptol)[0]: bad.append(f'{k}[{i}] cpu {a[i]:.6g} gpu {b[i]:.6g}')
        else:
            dead_ok = k in ('active', 'switch_until', 'zoom', 'w_wid', 'w_ammo', 'w_reserve', 'w_shots', 'w_shot_ready', 'w_reload_until', 'nades')
            for idx in zip(*np.where(a != b)):
                bad.append(f'{k}{list(idx)} cpu {a[idx]} gpu {b[idx]}')
    for k, a in cm.items():
        if keys is not None and k not in keys: continue
        b = gm[k]
        if k == 'bomb_pos':
            if np.abs(np.array(a) - np.array(b)).max() > 1e-6: bad.append(f'{k} cpu {a} gpu {b}')
        elif k == 'events':
            for n in a:
                if a[n] != b[n]: bad.append(f'event {n} cpu {a[n]} gpu {b[n]}')
        elif a != b: bad.append(f'{k} cpu {a} gpu {b}')
    return bad


def void_tainted(gm):
    return gm.get('void', 0) > 0


# ====================================================================================================== the pair
class Pair:
    def __init__(self, map_id, n, E=None, seed=1, resolve=True, K=64, grenades=True, graph=True):
        E = E or n
        self.n, self.map_id = n, map_id
        self.sim = GpuSim(map_id, E, 'cuda', seed=seed, K=K, grenades=grenades, collide=resolve, graph=graph)
        if self.sim.static.get('boxes'): self.sim.mover.wall_check_default = False     # MeshWorld has no `blockers` (CPU pushes unconditionally)
        self.sim.reset(seed)
        self.cpu = []
        for e in range(n):
            ar = RefArena(self.sim.R.match_seed(e), resolve=resolve); ar.reset(map_id)
            self.cpu.append(ar)
        self.slot_of = [dict() for _ in range(n)]; self.slot_n = [0] * n; self.ev_seen = [0] * n; self.gitems = None
        self.tainted = [False] * n; self.taint = False; self.first_bad = [None] * n; self.nbad = [0] * n; self.steps = 0; self.log = []
        self.scan_events(); self.refresh_items()

    def refresh_items(self):
        s = self.sim.s; n = self.n
        self.gitems = {k: s[k][:n].cpu().numpy() for k in ('wi_valid', 'wi_kind', 'wi_pos', 'wi_ammo')}

    def gpu_slot(self, e, it):
        """GPU world-item slot of the CPU world item `it` (same kind and ammo, nearest position within 1 cm), -1 if none."""
        g = self.gitems
        ok = (g['wi_valid'][e] > 0) & (g['wi_kind'][e] == TB.WID[it['kind']])
        d = np.where(ok, np.abs(g['wi_pos'][e] - np.array(it['position'])).max(1) + 1e-4 * (g['wi_ammo'][e] != it['weapon'].ammo), np.inf)
        j = int(np.argmin(d))
        if d[j] >= 1e-2 and not getattr(self, '_warned', False):
            self._warned = True
            kk = np.where(g['wi_valid'][e] > 0)[0]
            print('  [harness] CPU item', it['id'], it['kind'], 'ammo', it['weapon'].ammo, 'pos', it['position'], 'owner', it['owner'],
                  'not found on GPU; GPU items:', [(int(k), int(g['wi_kind'][e][k]), int(g['wi_ammo'][e][k]), g['wi_pos'][e][k].round(3).tolist()) for k in kk], flush=True)
        return j if d[j] < 1e-2 else -1

    def scan_events(self):
        for e, ar in enumerate(self.cpu):
            for ev in ar.events[self.ev_seen[e]:]:
                if ev['type'] == 'round_start': self.slot_of[e].clear(); self.slot_n[e] = 0
                if ev['type'] == 'drop' and str(ev.get('item_id', '')).startswith('item:'):
                    self.slot_of[e][ev['item_id']] = self.slot_n[e]; self.slot_n[e] += 1
            self.ev_seen[e] = len(ar.events)

    def step(self, acts):
        for ar in self.cpu: ar.dbg.clear()
        self.sim.damage.dbg.clear()
        """acts: list (per compared match) of 10 CPU action dicts.  The other GPU envs repeat the last compared match's actions."""
        E = self.sim.E
        rows = [list(a) for a in acts]
        for e in range(self.n, E):
            rows.append(rows[(e - self.n) % self.n])
        # pickup ids -> GPU world-item slots (looked up from the GPU state of the previous decision)
        grows = []
        for e, row in enumerate(rows):
            g = []
            for a in row:
                a = dict(a)
                if a.get('use') == 'pickup' and isinstance(a.get('pickup_id'), str):
                    ee = e if e < self.n else e % self.n
                    it = next((i for i in self.cpu[ee].world_items if i['id'] == a['pickup_id']), None)
                    a['pickup_id'] = self.gpu_slot(ee, it) if (it is not None and e < self.n) else -1
                g.append(a)
            grows.append(g)
        self.last_acts = rows; self.last_ev = []
        for e, ar in enumerate(self.cpu):
            n0 = len(ar.events); ar.step(rows[e]); self.last_ev.append(ar.events[n0:])
        self.sim.step(self.sim.actions_from_json(grows))
        self.scan_events(); self.refresh_items()
        self.steps += 1

    def dump(self, e):
        """print every weapon field of every seat for match e (CPU vs GPU)"""
        ar = self.cpu[e]; g = self.sim.s
        for q in range(10):
            p = ar.players[q]
            for k in range(3):
                w = p.ws.slots[k]
                gw = [int(g[n][e, q, k]) for n in ('w_wid', 'w_ammo', 'w_reserve', 'w_shots', 'w_shot_ready', 'w_reload_until', 'w_reload_commit', 'w_last_shot')]
                cw = None if w is None else [TB.WID[w.id], w.ammo, w.reserve, w.shots, w.shot_ready, -1 if w.reload_until is None else w.reload_until,
                                             -1 if w.reload_commit is None else w.reload_commit, -1 if w.last_shot is None else w.last_shot]
                if cw is None and gw[0] < 0: continue
                print(f'   seat {q} slot {k} alive {p.alive}/{bool(g["alive"][e, q])} active {p.ws.active}/{int(g["active"][e, q])} cpu {cw} gpu {gw}')

    def check(self, verbose=False, keys=None, ptol=1.0):
        gst = gpu_state(self.sim, self.n)
        res = []
        for e, ar in enumerate(self.cpu):
            cd, cm = cpu_state(ar)
            gd, gm = gst[e]
            kk = keys
            if void_tainted(gm) and self.taint:           # known difference: a shooter killed in the tick it fires keeps the shot in the dropped weapon
                self.tainted[e] = True
            if self.tainted[e]:
                kk = [k for k in (keys or list(cd) + list(cm)) if k not in WEAPON_STATE + ('item_states',)]
            bad = compare(cd, cm, gd, gm, keys=kk, ptol=ptol)
            if bad:
                self.nbad[e] += 1
                if self.first_bad[e] is None:
                    self.first_bad[e] = (self.steps, cm['round_index'], bad[:8])
                    if verbose:
                        print(f'  match {e}: first mismatch at step {self.steps} (tick {cm["tick"]}, round {cm["round_index"]}):', bad[:8], flush=True)
                        print('    actions with content:', [(q, {k: v for k, v in a.items() if k not in ('yaw_rate', 'pitch_rate', 'move_right', 'move_forward')})
                                                           for q, a in enumerate(self.last_acts[e]) if set(a) - {'yaw_rate', 'pitch_rate', 'move_right', 'move_forward'}][:12])
                        print('    CPU events this step:', [(x['type'], x.get('player', x.get('seat', x.get('target')))) for x in self.last_ev[e]
                                                          if x['type'] not in ('cash',)][:30])
                        print('    GPU receipts:', self.sim.s['rcpt'][e].cpu().numpy().reshape(10, 12).tolist())
                        self.dump(e)
                        if self.cpu[e].keep:
                            print('    CPU shot details this step:'); [print('      ', r) for r in self.cpu[e].dbg[-60:]]
                            print('    GPU rays this step (env %d):' % e)
                            for dd in self.sim.damage.dbg[-3:]:
                                m = dd['env'] == e
                                for j in np.where(m)[0]: print('      tick', dd['tick'], 'shooter', dd['shooter'][j], 'org', dd['org'][j], 'dir', dd['dir'][j], 'end', dd['end'][j], 'nhit', dd['nhit'][j], 'seat/dist/part/scale', dd['seat'][j] if False else '', list(zip(*[dd[k][j][:dd['nhit'][j]] for k in ('seat', 'dist', 'part', 'scale')])) if dd['nhit'][j] else '')
            res.append(bad)
        return res


# ====================================================================================================== director
BUYS_T = ['ak47', 'awp', 'pistol', 'elite', 'p250', 'deagle', 'tec9', 'nova', 'xm1014', 'sawedoff', 'mac10', 'mp7', 'galilar', 'sg553',
          'ssg08', 'armor', 'helmet', 'hegrenade', 'flashbang', 'smokegrenade', 'molotov', 'decoy', 'defusekit', 'm4a1', 'c4', 'knife']
BUYS_CT = ['m4a1', 'm4a4', 'awp', 'usp', 'elite', 'p250', 'fiveseven', 'deagle', 'nova', 'mag7', 'mp9', 'mp7', 'bizon', 'scar20', 'ssg08',
           'armor', 'helmet', 'defusekit', 'hegrenade', 'flashbang', 'smokegrenade', 'incgrenade', 'decoy', 'ak47']


class Director:
    """Produces the same random-but-plausible action lists for both sims and teleports players (both sims) between decisions."""

    def __init__(self, pair, seed=0, throws=False, fight_after=125, plant_prob=0.6, defuse_prob=0.5, fights=True, fast_buy=False,
                 buy_activity=0.18):
        self.pair = pair; self.rng = random.Random(seed); self.throws = throws; self.fight_after = fight_after
        self.plant_prob, self.defuse_prob, self.fights, self.fast_buy, self.buy_activity = plant_prob, defuse_prob, fights, fast_buy, buy_activity
        n = pair.n
        nav = pair.cpu[0].static['nav']
        self.nodes = np.array([nd['position'] for nd in nav['nodes']]); idx = {nd['id']: i for i, nd in enumerate(nav['nodes'])}
        self.adj = collections.defaultdict(list)
        for ed in nav['edges']:
            a, b = (ed[0], ed[1]) if isinstance(ed, (list, tuple)) else (ed.get('from', ed.get('a')), ed.get('to', ed.get('b')))
            if a in idx and b in idx: self.adj[idx[a]].append(idx[b]); self.adj[idx[b]].append(idx[a])
        self.last_round = [-1] * n; self.tele_step = [0] * n; self.plan = [dict() for _ in range(n)]

    # ---- helpers
    def teleport(self, e, seat, x, y, z, yaw, pitch=0.0):
        """set a player's pose in the CPU arena and in the GPU env e (and the replicas of e)."""
        ar = self.pair.cpu[e]; ps = ar.players[seat].ps
        ps.x, ps.y, ps.z, ps.yaw, ps.pitch = x, y, z, yaw, pitch
        ps.vx = ps.vy = ps.vz = 0.0; ps.grounded = True
        s = self.pair.sim.s
        for k, v in (('x', x), ('y', y), ('z', z), ('yaw', yaw), ('pitch', pitch), ('vx', 0.0), ('vy', 0.0), ('vz', 0.0)):
            s[k][e, seat] = v
        s['grounded'][e, seat] = True

    def arena_pairs(self, e, alive_t, alive_ct):
        """5 facing pairs on adjacent nav nodes (distance 3-14 m)."""
        rg = self.rng; out = []
        for _ in range(60):
            i = rg.randrange(len(self.nodes))
            nb = [j for j in self.adj[i] if 3.0 < np.linalg.norm(self.nodes[i][:2] - self.nodes[j][:2]) < 14.0]
            if nb: out.append((i, rg.choice(nb)))
            if len(out) >= 5: break
        return out

    def actions(self):
        """one action list per compared match, from the CPU states."""
        pair = self.pair; rg = self.rng; acts = []
        for e, ar in enumerate(pair.cpu):
            row = [dict() for _ in range(10)]
            rs = ar.round_index
            if ar.finished:
                acts.append(row); continue
            rt = ar.tick - ar.round_start_tick
            if rs != self.last_round[e]:
                self.last_round[e] = rs; self.plan[e] = dict(fought=False, planted_try=rg.random() < self.plant_prob, nteleports=0)
                if self.fast_buy:       # shorten the freeze time by 240 ticks in both sims (state injection)
                    ar.round_start_tick -= 240; ar.live_end -= 240
                    for k in ('round_start_tick', 'live_end'): pair.sim.s[k][e] -= 240
                    rt -= 240
            if ar.phase == 'buy' or rt < 1860:
                for p in ar.players:
                    if not p.alive: continue
                    kinds = BUYS_T if ar._side(p) == 'attack' else BUYS_CT
                    if ar.phase == 'buy' and rt < 100 and rg.random() < self.buy_activity:
                        row[p.seat]['buy'] = rg.choice(kinds)
                    if rg.random() < 0.02:
                        row[p.seat]['refund_item_id'] = rg.choice(kinds)
                    if rg.random() < 0.01:
                        row[p.seat]['drop'] = rg.choice(['primary', 'secondary', 'bomb', 'secondary'])
            # live phase: fights
            if ar.phase == 'live':
                self.fight(e, ar, row, rt)
            elif ar.phase == 'post':
                for p in ar.players:
                    if rg.random() < 0.05: row[p.seat]['buy'] = rg.choice(BUYS_T)
            # opportunistic pick-ups of items within reach
            for p in ar.players:
                if p.alive and ar.phase != 'post' and ar.world_items and rg.random() < 0.04:
                    near = [it for it in ar.world_items if math.dist((p.x, p.y, p.z), it['position']) < 2.5 and it['kind'] != 'knife']
                    if near and 'use' not in row[p.seat]:
                        row[p.seat]['use'] = 'pickup'; row[p.seat]['pickup_id'] = rg.choice(near)['id']
            acts.append(row)
        return acts

    def fight(self, e, ar, row, rt):
        rg = self.rng; pl = self.plan[e]
        att = [p for p in ar.players if ar._side(p) == 'attack']; dfn = [p for p in ar.players if ar._side(p) == 'defend']
        alive_a = [p for p in att if p.alive]; alive_d = [p for p in dfn if p.alive]
        b = ar.bomb
        # --- plant: the carrier walks into a plant zone (teleport), holds still and plants
        if pl['planted_try'] and b['state'] == 'held' and rt > 130 and not pl.get('plant_tp'):
            c = ar.players[b['carrier']] if b['carrier'] is not None else None
            if c is not None and c.alive:
                site = rg.choice(['A', 'B']); pz = ar.static['plant_zones'][site]
                poly = np.array(pz['polygon']); cx, cy = poly.mean(0)
                sz = next(sd for sd in ar.static['sites'] if sd['id'] == site)['center'][2] + 0.05
                self.teleport(e, c.seat, float(cx), float(cy), float(sz), c.ps.yaw)
                pl['plant_tp'] = True; pl['plant_t'] = rt
        if b['state'] == 'held' and pl.get('plant_tp') and rt < pl.get('plant_t', 0) + 200:
            c = ar.players[b['carrier']] if b['carrier'] is not None else None
            if c is not None and c.alive: row[c.seat]['use'] = 'plant'
        # --- defuse: after the plant, a defender is teleported to the bomb and defuses (sometimes)
        if b['state'] == 'planted' and alive_d and rg.random() < self.defuse_prob and pl.get('defuse_tp') is None:
            d = rg.choice(alive_d); bp = b['position']
            self.teleport(e, d.seat, bp[0] + 0.5, bp[1], bp[2], d.ps.yaw); ar.players[d.seat].ps.z = bp[2]
            pl['defuse_tp'] = d.seat
        if pl.get('defuse_tp') is not None and b['state'] == 'planted' and rg.random() < 0.97:
            d = ar.players[pl['defuse_tp']]
            if d.alive and rt > 0: row[d.seat]['use'] = 'defuse'
        # --- fights: pair up the living and shoot (teleport every time the pairs dissolve)
        if self.fights and rt > self.fight_after and (pl['nteleports'] == 0 or rt - self.tele_step[e] > 140) and alive_a and alive_d \
                and not (b['state'] == 'planted' and rg.random() < 0.9) and pl['nteleports'] < 12:
            prs = self.arena_pairs(e, alive_a, alive_d)
            for k, (ta, td) in enumerate(zip(alive_a, alive_d)):
                if k >= len(prs): break
                i, j = prs[k]; pa, pd = self.nodes[i], self.nodes[j]
                ya = math.atan2(-(pd[1] - pa[1]), pd[0] - pa[0]); yd = math.atan2(-(pa[1] - pd[1]), pa[0] - pd[0])
                self.teleport(e, ta.seat, pa[0], pa[1], pa[2], ya); self.teleport(e, td.seat, pd[0], pd[1], pd[2], yd)
            pl['nteleports'] += 1; self.tele_step[e] = rt
        if self.fights and rt > 1400 and not pl.get('softened') and alive_a and alive_d and b['state'] != 'planted':
            # make the round end: one random side is left with 1 hp (injected in both sims)
            pl['softened'] = True
            for p in (alive_a if rg.random() < 0.5 else alive_d):
                p.hp = 1.0; self.pair.sim.s['hp'][e, p.seat] = 1.0
        if pl['nteleports'] > 0:
            for p in ar.players:
                if not p.alive: continue
                en = [q for q in ar.players if q.alive and q.team != p.team]
                if not en: continue
                q = min(en, key=lambda q: math.dist((p.x, p.y), (q.x, q.y)))
                dx, dy, dz = q.x - p.x, q.y - p.y, (q.z + 1.2) - (p.z + 1.62)
                tyaw = math.atan2(-dy, dx); err = (tyaw - p.ps.yaw + math.pi) % (2 * math.pi) - math.pi
                dist = math.hypot(dx, dy); tp = math.atan2(dz, dist); perr = tp - p.ps.pitch
                a = row[p.seat]
                a['yaw_rate'] = max(-1.0, min(1.0, err / 0.15)); a['pitch_rate'] = max(-1.0, min(1.0, perr / 0.1))
                if abs(err) < 0.25 and not a.get('use'): a['fire'] = True
                if rg.random() < 0.08: a['reload'] = True
                if rg.random() < 0.05: a['weapon_slot'] = rg.choice([0, 1, 2])
                if rg.random() < 0.1: a['crouch'] = True
                a['move_right'] = rg.choice([-1.0, 0.0, 1.0]); a['move_forward'] = rg.choice([0.0, 0.5, 1.0])
                if rg.random() < 0.03: a['jump'] = True
                if rg.random() < 0.02: a['fire_secondary'] = True
                if self.throws and rg.random() < 0.04:
                    a['throw'] = rg.choice(UTIL); a['throw_state'] = rg.choice(['prime', 'release', 'release'])


# ====================================================================================================== main
def run_duels(map_id, n, rounds, seed, throws, max_steps, verbose, E=None, **kw):
    t0 = time.time()
    taint = kw.pop('taint', False); graph = kw.pop('graph', True)
    pair = Pair(map_id, n, E=E, seed=seed, graph=graph); pair.taint = taint
    if kw.pop('keep', False):
        pair.sim.damage.keep = True
        for ar in pair.cpu: ar.keep = True
    print(f'init {time.time() - t0:.1f}s', flush=True)
    dr = Director(pair, seed=seed, throws=throws, **kw)
    target = rounds
    step = 0
    while step < max_steps:
        if all(ar.finished or ar.round_index >= target for ar in pair.cpu): break
        pair.step(dr.actions()); pair.check(verbose=verbose); step += 1
        if step % 200 == 0:
            print(f'step {step} tick {pair.sim.gtick} rounds {[ar.round_index for ar in pair.cpu]} wins {[ar.round_wins for ar in pair.cpu]} '
                  f'bad {pair.nbad} {time.time() - t0:.0f}s', flush=True)
    print('steps', step, 'matches', n)
    print('GPU-only counters per match (void_shot: actions of players killed earlier in the same tick that the CPU loop never runs; '
          'dead_hit: bullet hits on a player killed earlier in the same tick):',
          [(pair.sim.events(e)['void_shot'], pair.sim.events(e)['dead_hit']) for e in range(n)])
    evsum = collections.Counter()
    for ar in pair.cpu:
        evsum.update(e['type'] for e in ar.events)
    print('CPU event totals', {k: v for k, v in evsum.items() if k in EVN})
    for e in range(n):
        print(f'match {e}: rounds {pair.cpu[e].round_index}, wins {pair.cpu[e].round_wins}, finished {pair.cpu[e].finished}, '
              f'steps with any mismatch {pair.nbad[e]}/{step}, void-shot taint {pair.tainted[e]}, first {pair.first_bad[e]}')
    return pair


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--map', default='dust2-cs2-v54'); ap.add_argument('--matches', type=int, default=4)
    ap.add_argument('--rounds', type=int, default=3); ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--steps', type=int, default=10 ** 9); ap.add_argument('--throws', action='store_true')
    ap.add_argument('--E', type=int, default=None); ap.add_argument('-v', action='store_true')
    ap.add_argument('--taint', action='store_true'); ap.add_argument('--keep', action='store_true'); ap.add_argument('--no-graph', action='store_true'); ap.add_argument('--fast-buy', action='store_true'); ap.add_argument('--no-fights', action='store_true')
    ap.add_argument('--plant', type=float, default=0.6); ap.add_argument('--defuse', type=float, default=0.5)
    a = ap.parse_args()
    run_duels(a.map, a.matches, a.rounds, a.seed, a.throws, a.steps, a.v, E=a.E, fast_buy=a.fast_buy, fights=not a.no_fights,
              plant_prob=a.plant, defuse_prob=a.defuse, taint=a.taint, graph=not a.no_graph, keep=a.keep)
