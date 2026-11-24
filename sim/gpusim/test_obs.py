"""Parity / legality / throughput test of gpusim.obs against fpssim.obs.build_observation driven by the CPU Arena.

usage (server, from sim/):  CUDA_VISIBLE_DEVICES=1 python3 -m gpusim.test_obs [--maps dust2-cs2-v54,nevaio-r3-v54] [--arenas 3] [--steps 330]

Scenario: scripted (omniscient, random) bots drive a real Arena (buy phase purchases, teleport into contact at the start of
the live phase, random movement / shooting / grenades, one planter + defuser scenario per arena).  The GPU emitter buffer is fed
tick by tick from the same Arena (events + player state), snapshots are taken at decision boundaries.
Per snapshot and seat: CPU observation = fpssim.obs.build_observation(view_from_arena(arena) [+ patches below], seat, CPU visible),
GPU observation = decode_obs(build_obs(state tensors, vis_parts)).  Patches of the CPU view (CPU Arena gaps, documented):
vis bitmask (so the held-bomb carrier rule works), planter/plant_progress/defuser/defuse_progress/defuse_duration (the Arena does
not set them), buy_deadline_tick (the CPU obs reports the constant 1860).
vis_parts: gpusim.vis.visible on the exact Dust2 mesh / gpusim.vis.whitebox_mesh, with the CPU smokes of each snapshot as smoke_fn; its difference to the CPU
parts is printed.  Sounds: CPU rules fitted on our training.jsonl (only the decision tick's emitters, no occlusion, no merging, mirrored left/right).
"""
import sys, os, time, json, math, argparse, collections, importlib.util, copy
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import numpy as np
import torch

from fpssim.arena import Arena
from fpssim import obs as CO
from fpssim.grenades import BoxContact
from gpusim import obs as GO
from gpusim.obs import ObsStatic, GpuAudio, build_obs, decode_obs, OBS_DIM
from gpusim.tables import WEAPON_IDS, UTIL_IDS

DEV = 'cuda'
RS_KEYS = ('index', 'pitch', 'yaw', 'velPitch', 'velYaw', 'viewPitch', 'viewYaw')
PHASE = {'buy': 0, 'live': 1, 'post': 2, 'finished': 3}
BSTATE = {'held': 0, 'world': 1, 'planted': 2, 'defused': 3, 'exploded': 4}
MMODE = {'ground': 0, 'air': 1, 'ladder': 2, 'ladder_exit': 3}
K_ITEMS = 32


# ------------------------------------------------------------------------------------------- Arena -> state (SPEC layout)
def to_state(A, ladders):
    P = A.players
    f = lambda fn: np.array([fn(p) for p in P], np.float64)
    b = lambda fn: np.array([bool(fn(p)) for p in P])
    i = lambda fn: np.array([int(fn(p)) for p in P], np.int64)
    s = dict(
        x=f(lambda p: p.ps.x), y=f(lambda p: p.ps.y), z=f(lambda p: p.ps.z), vx=f(lambda p: p.ps.vx), vy=f(lambda p: p.ps.vy),
        vz=f(lambda p: p.ps.vz), yaw=f(lambda p: p.ps.yaw), pitch=f(lambda p: p.ps.pitch),
        alive=b(lambda p: p.alive), crouch=b(lambda p: p.ps.crouch), grounded=b(lambda p: p.ps.grounded),
        walk_held=b(lambda p: p.held.get('walk')), stand_blocked=b(lambda p: p.ps.stand_blocked),
        hp=f(lambda p: p.hp), armor=f(lambda p: p.armor), helmet=b(lambda p: p.helmet), kit=b(lambda p: p.kit),
        cash=i(lambda p: p.cash), step_distance=f(lambda p: p.ps.step_distance),
        movement_mode=i(lambda p: MMODE[p.ps.movement_mode]),
        ladder_id=i(lambda p: -1 if p.ps.ladder_id is None else ladders.setdefault(p.ps.ladder_id, len(ladders))),
        reattach_until=i(lambda p: p.ps.reattach_until), blind_until=i(lambda p: p.blind_until), blind_start=i(lambda p: p.blind_start),
        throw_ready=i(lambda p: p.throw_ready), pickup_ready_tick=i(lambda p: p.pickup_ready_tick),
        has_bomb=b(lambda p: p.has_bomb), objective=i(lambda p: {None: 0, 'plant': 1, 'defuse': 2}[p.objective]),
        obj_progress=i(lambda p: p.obj_progress), active=i(lambda p: p.ws.active), switch_until=i(lambda p: p.ws.switch_until),
        zoom=i(lambda p: p.ws.zoom_level),
        nades=np.array([[p.nades.get(k, 0) for k in UTIL_IDS] for p in P], np.int64))
    W3 = lambda fn, dt, dflt: np.array([[fn(p.ws.slots[k]) if p.ws.slots[k] is not None else dflt for k in range(3)] for p in P], dt)
    s.update(
        w_wid=W3(lambda w: WEAPON_IDS.index(w.id), np.int64, -1), w_ammo=W3(lambda w: w.ammo, np.int64, 0),
        w_reserve=W3(lambda w: w.reserve, np.int64, 0), w_shots=W3(lambda w: w.shots, np.int64, 0),
        w_shot_ready=W3(lambda w: w.shot_ready, np.int64, 0),
        w_reload_until=W3(lambda w: -1 if w.reload_until is None else w.reload_until, np.int64, -1),
        w_reload_commit=W3(lambda w: -1 if w.reload_commit is None else w.reload_commit, np.int64, -1),
        w_last_shot=W3(lambda w: -1 if w.last_shot is None else w.last_shot, np.int64, -1),
        w_reload_committed=W3(lambda w: w.reload_committed, bool, False), w_reload_shell=W3(lambda w: w.reload_shell, bool, False),
        w_pressed=W3(lambda w: w.pressed, bool, False), w_inacc=W3(lambda w: w.inaccuracy, np.float64, 0.0))
    s['w_rs'] = np.array([[[w.rs[k] for k in RS_KEYS] if w is not None else [0.0] * 7 for w in p.ws.slots] for p in P], np.float64)
    bm = A.bomb
    pos = bm.get('position') or (0.0, 0.0, 0.0)
    sites = [x['id'] for x in A.static.get('sites') or []]
    s_e = dict(
        tick=A.tick, round_index=max(0, A.round_index), round_start_tick=A.round_start_tick, live_end=A.live_end,
        post_end=A.post_end or 0, phase=PHASE[A.phase], attack_team=A.attack_team, round_wins=list(A.round_wins),
        loss_level=list(A.loss_level), bomb_state=BSTATE[bm['state']], bomb_carrier=-1 if bm.get('carrier') is None else bm['carrier'],
        bomb_pos=list(pos), bomb_planted_tick=-1 if bm.get('planted_tick') is None else bm['planted_tick'], bomb_fuse_end=-1 if bm.get('fuse_end') is None else bm['fuse_end'], bomb_duration=bm.get('duration') or 0,
        bomb_site=sites.index(bm['site']) if bm.get('site') in sites else -1)
    it = A.world_items
    wi_valid = np.zeros(K_ITEMS, bool); wi_kind = np.zeros(K_ITEMS, np.int64); wi_pos = np.zeros((K_ITEMS, 3))
    for k, w in enumerate(it[:K_ITEMS]):
        wi_valid[k] = True; wi_kind[k] = WEAPON_IDS.index(w['kind']); wi_pos[k] = w['position']
    out = {k: np.asarray(v)[None] for k, v in s.items()}
    out.update({k: np.asarray(v)[None] for k, v in s_e.items()})
    out.update(wi_valid=wi_valid[None], wi_kind=wi_kind[None], wi_pos=wi_pos[None])
    return out


def to_torch(npd, dev=DEV):
    out = {}
    for k, v in npd.items():
        t = torch.as_tensor(v, device=dev)
        out[k] = t
    return out


def cat_states(lst):
    return {k: torch.cat([d[k] for d in lst], 0) for k in lst[0]}


def events_to_masks(evs, dev=DEV):
    """CPU events of one tick -> GpuAudio events (only the audible kinds matter: reload, bounce, fire_started, bomb events)."""
    z = lambda *sh: torch.zeros(*sh, dtype=torch.bool, device=dev)
    ev = dict(reload=z(1, 10), bounce=z(1, GO.NB), bounce_pos=torch.zeros(1, GO.NB, 3, dtype=torch.float64, device=dev),
              fire_started=z(1, GO.NB), fire_started_pos=torch.zeros(1, GO.NB, 3, dtype=torch.float64, device=dev))
    for k in ('plant_complete', 'defuse_complete', 'bomb_explode'):
        ev[k] = z(1)
    nb = nf = 0
    for e in evs:
        t = e.get('type')
        if t == 'reload_start': ev['reload'][0, e['player']] = True
        elif t == 'bounce' and nb < GO.NB:
            ev['bounce'][0, nb] = True; ev['bounce_pos'][0, nb] = torch.tensor(e['position'], dtype=torch.float64, device=dev); nb += 1
        elif t == 'detonate' and e.get('kind') in ('molotov', 'incgrenade') and e.get('position') and nf < GO.NB:
            # Arena.step: every grenade `detonate` event of a fire grenade also adds a fire_started emitter
            ev['fire_started'][0, nf] = True; ev['fire_started_pos'][0, nf] = torch.tensor(e['position'], dtype=torch.float64, device=dev); nf += 1
        elif t in ('plant_complete', 'defuse_complete', 'bomb_explode'): ev[t][0] = True
    return ev


def fires_of(A, dev=DEV):
    F_ = GO.NFIRE
    pos = torch.zeros(1, F_, 3, dtype=torch.float64, device=dev); valid = torch.zeros(1, F_, dtype=torch.bool, device=dev)
    for i, f in enumerate(A.grenades.fires[:F_]):
        pos[0, i] = torch.tensor(f['position'], dtype=torch.float64); valid[0, i] = True
    return pos, valid


# ------------------------------------------------------------------------------------------- scripted scenario
def load_baseline(path):
    spec = importlib.util.spec_from_file_location('agent_', path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


class Scenario:
    def __init__(self, map_id, seed, st, mode='random'):
        self.rng = np.random.default_rng(seed)
        self.A = Arena()
        self.A.observations = lambda: [None] * 10                    # skip the (slow) CPU observations during the run
        self.A.reset(map_id=map_id, seed=seed)
        self.A.first_obs = False
        self.ladders = {}
        self.au = GpuAudio(1, DEV)
        self.cpu_emitters = []
        self.st = st
        self.mode = mode
        self.plant = mode in ('plant', 'explode')
        self.walk = [np.zeros(4) for _ in range(10)]
        self.n_steps = 0; self.audio_diffs = []; self.emit_kinds = collections.Counter()
        A = self.A
        orig = A._tick

        def hooked():
            prev = self.au.snapshot(to_torch(to_state(A, self.ladders)))
            n0 = len(A.events)
            orig()
            cur = to_torch(to_state(A, self.ladders))
            self.au.step(prev, cur, events_to_masks(A.events[n0:]))
        A._tick = hooked
        nodes = A.static['nav']['nodes']
        self.nodes = np.array([n['position'] for n in nodes], float)
        self.wb = bool(A.static.get('boxes'))

    def buy_all(self):
        A, r = self.A, self.rng
        for p in A.players:
            p.cash = int(r.integers(1500, 9000))
        self.buys = {}
        for p in A.players:
            side = 'attack' if p.team == A.attack_team else 'defend'
            lst = ['ak47' if side == 'attack' else 'm4a1', 'mp9', 'awp', 'armor', 'helmet', 'hegrenade', 'flashbang',
                   'smokegrenade', 'molotov' if side == 'attack' else 'incgrenade', 'defusekit', 'deagle', 'decoy']
            r.shuffle(lst)
            self.buys[p.seat] = lst[:4]

    def teleport(self):
        A, r = self.A, self.rng
        piv = self.nodes[r.integers(len(self.nodes))]
        d = np.linalg.norm(self.nodes[:, :2] - piv[:2], axis=1)
        near = np.argsort(d)[:30]
        pick = r.choice(near, size=10, replace=False)
        for p, k in zip(A.players, pick):
            n = self.nodes[k]
            p.ps.x, p.ps.y, p.ps.z = float(n[0]), float(n[1]), float(n[2]) + (0.00011 if self.wb else 0.0)
            p.ps.vx = p.ps.vy = p.ps.vz = 0.0
            tgt = [q for q in A.players if q.team != p.team]
            q = tgt[r.integers(len(tgt))]
            p.ps.yaw = float(math.atan2(-(q.ps.y - n[1]), q.ps.x - n[0])) if q is not p else 0.0
            p.ps.pitch = 0.0

    def actions(self, step):
        A, r = self.A, self.rng
        acts = []
        for p in A.players:
            a = {}
            if not p.alive:
                acts.append(a); continue
            if A.phase == 'buy':
                if step % 6 == 1 and self.buys[p.seat]:
                    a['buy'] = self.buys[p.seat].pop()
                acts.append(a); continue
            w = self.walk[p.seat]
            if step % 6 == 0 or w[0] == 0:                         # persistent behaviour for 6 steps: run (70%) / walk / crouch
                self.walk[p.seat] = w = np.array([r.uniform(0.5, 1.0) if r.random() < 0.85 else r.uniform(-1, -0.3), r.uniform(-.5, .5),
                                                  r.uniform(-.5, .5), r.uniform(-.2, .2), r.choice([0, 0, 0, 0, 0, 0, 0, 1, 1, 2])])
            a.update(move_forward=float(w[0]), move_right=float(w[1]), yaw_rate=float(w[2]), pitch_rate=float(w[3]))
            if r.random() < 0.10: a['jump'] = True
            if w[4] == 1: a['walk'] = True
            if w[4] == 2: a['crouch'] = True
            near = [q for q in A.players if q.team != p.team and q.alive and math.dist((p.x, p.y), (q.x, q.y)) < 35]
            if near and r.random() < 0.5: a['fire'] = True
            if r.random() < 0.04: a['reload'] = True
            if r.random() < 0.15: a['zoom_level'] = int(r.integers(0, 3))
            if r.random() < 0.05: a['weapon_slot'] = int(r.integers(0, 3))
            have = [k for k, n in p.nades.items() if n > 0]
            fire_g = [k for k in have if k in ('molotov', 'incgrenade')]
            if have and r.random() < (0.35 if fire_g else 0.12):
                a['throw'] = fire_g[0] if fire_g and r.random() < 0.7 else have[int(r.integers(len(have)))]
                a['throw_state'] = 'prime' if p.throw_primed is None and step % 2 == 0 else 'release'
                a['throw_mode'] = ['full', 'lob', 'drop'][int(r.integers(3))]
            if self.plant and p.has_bomb and self.phase_step >= 0:
                a = dict(use='plant')
            if self.plant and self.defuser is not None and p.seat == self.defuser:
                a = dict(use='defuse')
            acts.append(a)
        return acts

    def run(self, steps, snap_every, st_plant=40):
        A = self.A
        snaps = []
        self.phase_step = -1; self.defuser = None
        for step in range(steps):
            if step == 0: self.buy_all()
            if A.tick >= 360 and getattr(self, 'tele', False) is False:
                self.tele = True
                self.teleport()
                if self.plant:                                       # carrier to the plant zone of site A
                    c = [p for p in A.players if p.has_bomb][0]
                    site = A.static['sites'][0]['center']
                    c.ps.x, c.ps.y, c.ps.z = float(site[0]), float(site[1]), float(site[2]) + (0.00011 if self.wb else 0.0)
                    self.phase_step = step
            if self.mode == 'explode' and A.bomb['state'] == 'planted' and self.phase_step >= 0:
                if not getattr(self, 'shifted', False):               # jump 1790 ticks ahead on the beep schedule: exercises the fast beeps
                    A.bomb['planted_tick'] -= 1790; A.bomb['fuse_end'] -= 1790; self.shifted = True
                elif A.tick - A.bomb['planted_tick'] >= 1900 and A.bomb['fuse_end'] > A.tick + 24:
                    A.bomb['fuse_end'] = A.tick + 24                  # explode on a decision tick (tick % 3 == 0 at the start of a step)
            if self.mode == 'wipe' and step == 200:
                for q in A.players:
                    if q.team != A.attack_team and q.alive: A._damage(q, A.players[0], 500.0, 0.0, 'ak47')
            if step in (170, 190, 215, 240) and self.mode in ('random', 'wipe') and A.phase == 'live':
                al = [q for q in A.players if q.alive]
                if len(al) > 4:
                    v = al[int(self.rng.integers(len(al)))]; k_ = [q for q in al if q.team != v.team]
                    A._damage(v, k_[int(self.rng.integers(len(k_)))], 500.0, 0.0, 'ak47')
            if self.plant and self.phase_step >= 0 and A.bomb['state'] == 'planted' and self.defuser is None:
                # a defender walks to the planted bomb and defuses
                dd = [p for p in A.players if p.alive and p.team != A.attack_team]
                if dd:
                    q = dd[0]; bp = A.bomb['position']; q.kit = (self.mode == 'plant'); q.kit_item = 'item:kit' if q.kit else None
                    q.ps.x, q.ps.y, q.ps.z = bp[0] + 0.8, bp[1], bp[2]
                    self.defuser = q.seat
            acts = self.actions(step)
            for p, a in zip(A.players, acts):
                if p.alive and a.get('throw_state') == 'prime': p.throw_primed = True
            A.step(acts)
            self.last_em = self.au.flush(fires_of(A))                # GPU emitters of this step = CPU A.audio_events
            self.n_steps += 1
            kg, kc = emitter_keys_gpu(self.last_em, 0), emitter_keys_cpu(A.audio_events)
            for k_ in kc: self.emit_kinds[k_[0]] += 1
            if kg != kc:
                self.audio_diffs.append((step, A.tick, A.bomb['state'], A.bomb.get('planted_tick'), A.bomb.get('fuse_end'), kg, kc))
            rare = {k_[0] for k_ in kc} & RARE_KINDS                  # extra snapshot whenever a rare sound is on the air
            if step % snap_every == snap_every - 1 or rare or (self.plant and A.bomb['state'] == 'planted' and step % 3 == 0):
                snaps.append(self.snapshot())
        return snaps

    def snapshot(self):
        A = self.A
        vis = A.visible_enemies()
        v = dict(CO.view_from_arena(A))
        P = A.players
        vmask = [0] * 10
        for o, lst in vis.items():
            for q, _ in lst: vmask[o] |= 1 << q
        v['vis'] = vmask
        if self.wb:                                                  # one whitebox LOS model for obs and visibility (BoxContact)
            v['los'] = lambda p0, p1: A.vis_los.clear_many(np.asarray(p0, float), [p1])[0]
        b = dict(v['bomb'])
        for p in P:
            if p.objective == 'plant': b['planter'] = p.seat; b['plant_progress'] = p.obj_progress
            if p.objective == 'defuse':
                b['defuser'] = p.seat; b['defuse_progress'] = p.obj_progress; b['defuse_duration'] = A.bomb['duration']
        v['bomb'] = b
        cpu = [CO.build_observation(v, s, vis[s], first=False) for s in range(10)]
        parts = np.zeros((10, 10, 3), bool)
        for o, lst in vis.items():
            for q, pr in lst:
                for j, nme in enumerate(('head', 'chest', 'abdomen')): parts[o, q, j] = nme in pr
        st = to_state(A, self.ladders)
        return dict(cpu=cpu, state=st, parts=parts, emit_cpu=list(A.audio_events), emit_gpu=self.last_em,
                    round_start=A.round_start_tick, tick=A.tick, smokes=copy.deepcopy(A.grenades.smokes))


# ------------------------------------------------------------------------------------------- comparison
def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class Cmp:
    def __init__(self):
        self.n = collections.Counter(); self.bad = collections.Counter(); self.ex = {}

    def eq(self, name, a, b, atol=0.0, rtol=0.0, angle=False):
        self.n[name] += 1
        if a is None or b is None or isinstance(a, (str, bool, list, tuple)) or isinstance(b, (str, bool, list, tuple)):
            ok = a == b
        else:
            d = abs(wrap(a - b)) if angle else abs(a - b)
            ok = d <= atol + rtol * abs(b)
        if not ok:
            self.bad[name] += 1
            self.ex.setdefault(name, (a, b))
        return ok


def compare(c, cpu, gpu, tick, round_start, ladder_none_only=True):
    T = tick
    c.eq('match_tick', gpu['match_tick'], cpu['match_tick'])
    c.eq('round_index', gpu['round_index'], cpu['round_index'])
    c.eq('round_tick', gpu['round_tick'], cpu['round_tick'])
    c.eq('phase', gpu['phase'], cpu['phase'])
    c.eq('phase_remaining_ticks', gpu['phase_remaining_ticks'], cpu['phase_remaining_ticks'])
    for k in ('seat', 'team', 'side', 'role', 'round_wins', 'alive_by_team', 'map_id', 'attack_team', 'period',
              'overtime_number', 'win_target', 'loss_level'):
        c.eq(k, gpu[k], cpu[k])
    c.eq('buy_deadline_tick', gpu['buy_deadline_tick'], round_start + 1860)
    m, g = cpu['me'], gpu['me']
    for k in ('x', 'y', 'z'): c.eq('me.' + k, g[k], m[k], atol=1e-4)
    c.eq('me.yaw', g['yaw'], m['yaw'], atol=2e-6, angle=True)
    c.eq('me.pitch', g['pitch'], m['pitch'], atol=2e-6)
    for k in ('vx', 'vy', 'vz'): c.eq('me.' + k, g[k], m[k], atol=1e-5)
    c.eq('me.hp', g['hp'], m['hp'], atol=1e-3); c.eq('me.armor', g['armor'], m['armor'], atol=1e-3)
    c.eq('me.cash', g['cash'], m['cash'], atol=1e-2)
    for k in ('helmet', 'alive', 'crouch', 'grounded', 'walk', 'quiet', 'stand_blocked', 'active_slot', 'zoom_level',
              'ads_progress', 'movement_mode', 'carrying_bomb', 'has_kit', 'ammo', 'blind_ticks'):
        c.eq('me.' + k, g[k], m[k])
    c.eq('me.ladder_id.none', g['ladder_id'] is None, m['ladder_id'] is None)
    for k in ('reattach_until', 'pickup_ready_tick', 'switch_until', 'throw_ready'):
        c.eq('me.' + k, max(g[k], T), max(m[k], T))
    c.eq('me.shot_ready', max(g['shot_ready'], T), max(m['shot_ready'], T), atol=1e-3)
    c.eq('me.throwables', sorted(g['throwables']), sorted(m['throwables']))
    c.eq('me.throwable_kinds', sorted(g['throwable_kinds']), sorted(m['throwable_kinds']))
    for i in range(3):
        a, b = g['weapons'][i], m['weapons'][i]
        c.eq('w.present', a is None, b is None)
        if a is None or b is None: continue
        c.eq('w.id', a['id'], b['id'])
        for k in ('ammo', 'reserve', 'shots'): c.eq('w.' + k, a[k], b[k], atol=1e-3)
        c.eq('w.shot_ready', max(a['shot_ready'], T), max(b['shot_ready'], T), atol=1e-3)
        c.eq('w.reload_until', a['reload_until'], None if b['reload_until'] is None else int(b['reload_until']))
        ls = b['last_shot']
        if ls is None or T - ls <= GO.SINCE_CLIP:
            c.eq('w.last_shot', a['last_shot'], None if ls is None else int(ls))
        for k in ('recoil_pitch', 'recoil_yaw'): c.eq('w.' + k, a[k], b[k], atol=1e-6)
        for k in RS_KEYS:
            c.eq('w.aim_punch.' + k, a['aim_punch'][k], b['aim_punch'][k], atol=1e-5)
        c.eq('w.inaccuracy', a['inaccuracy'], b['inaccuracy'], atol=1e-5)
        if a['id'] != 'knife':                                       # CPU hides trigger_state.pressed for a knife without last_press
            c.eq('w.pressed', a['trigger_state']['pressed'], b['trigger_state'].get('pressed', False))
        ra, rb = a['reload_state'], b['reload_state']
        c.eq('w.reload_state.none', ra is None, rb is None)
        if ra is not None and rb is not None:
            c.eq('w.reload_state.commit', max(ra['commit'], T), None if rb['commit'] is None else max(int(rb['commit']), T))
            c.eq('w.reload_state.committed', ra['committed'], rb['committed']); c.eq('w.reload_state.shell', ra['shell'], rb['shell'])
    # teammates
    tg = {t['seat']: t for t in gpu['teammates']}
    tc = {t['seat']: t for t in cpu['teammates']}
    c.eq('tm.seats', sorted(tg), sorted(tc))
    for sd in tc:
        if sd not in tg: continue
        a, b = tg[sd], tc[sd]
        c.eq('tm.alive', a['alive'], b['alive']); c.eq('tm.cash', a['cash'], b['cash'], atol=1e-2)
        c.eq('tm.zone', a['zone'], b['zone']); c.eq('tm.equipment', sorted(a['equipment']), sorted(b['equipment']))
    # visible
    vg = {t['seat']: t for t in gpu['visible']}
    vc = {t['seat']: t for t in cpu['visible']}
    c.eq('vis.seats', sorted(vg), sorted(vc))
    for sd in vc:
        if sd not in vg: continue
        a, b = vg[sd], vc[sd]
        for k in ('x', 'y', 'z'): c.eq('vis.' + k, a[k], b[k], atol=1e-4)
        c.eq('vis.yaw', a['yaw'], b['yaw'], atol=2e-6, angle=True); c.eq('vis.pitch', a['pitch'], b['pitch'], atol=2e-6)
        for k in ('crouch', 'weapon', 'movement_mode', 'parts', 'team'): c.eq('vis.' + k, a[k], b[k])
    # order: nearest first
    ds = [math.dist((gpu['me']['x'], gpu['me']['y'], gpu['me']['z']), (t['x'], t['y'], t['z'])) for t in gpu['visible']]
    c.eq('vis.sorted', ds == sorted(ds), True)
    # sounds / items
    sk = lambda l: sorted((x['type'], x['direction'], x['distance_band']) for x in l)
    c.eq('sounds', sk(gpu['sounds']), sk(cpu['sounds']))
    ik = lambda l: sorted((x['kind'], tuple(round(c_, 3) for c_ in x['position'])) for x in l)
    me_pos = (m['x'], m['y'], m['z'])                                   # CPU list -> the NITEM nearest (stable), like the tensor
    near = sorted(cpu['world_items'], key=lambda w: math.dist(me_pos, w['position']))[:GO.NITEM]
    gi, ci = sorted(gpu['world_items'], key=lambda w: (w['kind'], w['position'])), sorted(near, key=lambda w: (w['kind'], w['position']))
    ok_wi = len(gi) == len(ci) and all(a_['kind'] == b_['kind'] and max(abs(x - y) for x, y in zip(a_['position'], b_['position'])) < 2e-4
                                       for a_, b_ in zip(gi, ci))
    c.eq('world_items', ok_wi, True)
    c.eq('world_items.count', gpu['n_visible_items'], len(cpu['world_items']))
    # bomb
    a, b = gpu['bomb'], cpu['bomb']
    c.eq('bomb.state', a['state'], b['state'])
    c.eq('bomb.fuse_end_tick', a['fuse_end_tick'], b['fuse_end_tick'])
    c.eq('bomb.known', 'position' in a, 'position' in b)
    if 'position' in a and 'position' in b:
        c.eq('bomb.id', a['id'], b['id']); c.eq('bomb.carrier', a['carrier'], b['carrier']); c.eq('bomb.site', a['site'], b['site'])
        for i in range(3): c.eq('bomb.position', a['position'][i], b['position'][i], atol=1e-4)
    for k in ('plant_progress', 'defuse_progress', 'defuse_duration'):
        c.eq('bomb.' + k + '.present', k in a, k in b)
        if k in a and k in b: c.eq('bomb.' + k, a[k], b[k], atol=1e-3)


# ------------------------------------------------------------------------------------------- whitebox triangles (test helper)
def wb_mesh(static):
    """whitebox GpuMesh: gpusim.vis.whitebox_mesh (= BoxContact geometry), else the local triangulation below."""
    from gpusim.ray import GpuMesh
    try:
        from gpusim.vis import whitebox_mesh
        return whitebox_mesh(static, DEV)
    except Exception as ex:                                      # pragma: no cover
        print('   (gpusim.vis.whitebox_mesh unavailable: %r; using the local triangulation)' % (ex,))
        return GpuMesh(boxes_tris(static), DEV)


def boxes_tris(static):
    """world-coordinate triangles of the whitebox boxes + stair steps + ground plane (as tools/meshmove_wb_validate.py)."""
    bc = BoxContact(static)
    tris = []
    for lo, hi in zip(bc.lo, bc.hi):
        x0, y0, z0 = lo; x1, y1, z1 = hi
        c = [(x0, y0, z0), (x1, y0, z0), (x1, y1, z0), (x0, y1, z0), (x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)]
        for f in ((0, 1, 2, 3), (4, 5, 6, 7), (0, 1, 5, 4), (3, 2, 6, 7), (0, 3, 7, 4), (1, 2, 6, 5)):
            tris += [(c[f[0]], c[f[1]], c[f[2]]), (c[f[0]], c[f[2]], c[f[3]])]
    bd = static.get('bounds') or {'min': [-200, -200], 'max': [200, 200]}
    g = bc.ground
    g0 = (bd['min'][0] - 20, bd['min'][1] - 20); g1 = (bd['max'][0] + 20, bd['max'][1] + 20)
    tris += [((g0[0], g0[1], g), (g1[0], g0[1], g), (g1[0], g1[1], g)), ((g0[0], g0[1], g), (g1[0], g1[1], g), (g0[0], g1[1], g))]
    return np.array(tris, np.float64)


# ------------------------------------------------------------------------------------------- batching helpers
def cat_audio(lst):
    cap = max(e['itype'].shape[1] for e in lst)
    out = {}
    for k in lst[0]:
        out[k] = torch.cat([e[k] for e in lst], 0)
    return out


AUDIBLE_KINDS = set(GO.SOUND_KINDS)
RARE_KINDS = {'bomb_beep_fast', 'jump', 'land_soft', 'land_hard', 'reload', 'grenade_bounce', 'fire_started', 'bomb_exploded', 'bomb_defused', 'bomb_planted'}


def emitter_keys_gpu(em, e):
    out = []
    for j in range(em['itype'].shape[1]):
        if not bool(em['valid'][e, j]): continue
        it = int(em['itype'][e, j])
        pos = tuple(round(float(x), 4) for x in em['pos'][e, j]) if GO._IT_HASPOS[it] else None
        out.append((GO.SOUND_KINDS[GO._IT_KIND[it]], int(em['src'][e, j]), pos, GO._IT_LOCAL[it], GO._IT_REMOTE[it]))
    return sorted(out, key=str)


def emitter_keys_cpu(lst):
    """CPU audio_events restricted to the kinds that can ever be heard (gunshot / draw / grenade_thrown / plant+defuse start are not
    modelled); every emitter is its own entry (no merging)."""
    out = []
    for a in lst:
        if a['kind'] not in AUDIBLE_KINDS: continue
        pos = tuple(round(float(x), 4) for x in a['position']) if a.get('position') else None
        src = a.get('source')
        out.append((a['kind'], src if isinstance(src, int) else -1, pos, int(bool(a.get('local_only'))), int(bool(a.get('remote_only')))))
    return sorted(out, key=str)


def rng_report(obs):
    mx = obs.abs().amax((0, 1)).cpu().numpy()
    bad = []
    for name, a, n, d in GO.SCHEMA:
        m = float(mx[a:a + n].max())
        if m > 2.0 or not np.isfinite(m): bad.append('%s=%.2f' % (name, m))
    print('   fields with max|value| > 2:', bad if bad else 'none', '| overall max %.2f, NaN %d' % (float(np.nanmax(mx)), int(torch.isnan(obs).sum())))


def sync():
    torch.cuda.synchronize()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--maps', default='dust2-cs2-v54,nevaio-r3-v54')
    ap.add_argument('--arenas', type=int, default=4)
    ap.add_argument('--steps', type=int, default=330)
    ap.add_argument('--snap', type=int, default=4)
    a = ap.parse_args()
    from gpusim.ray import GpuMesh
    from gpusim.vis import visible
    t_start = time.time()
    summary = {}
    bench_state = None
    n_audio = n_audio_bad = 0
    emit_kinds = collections.Counter()
    for map_id in a.maps.split(','):
        static = json.load(open('./fps_data/data/static/static_%s.json' % map_id))
        static = static['map'] if 'map' in static else static
        st = ObsStatic(static, DEV)
        snaps_all = []
        mesh = None
        for k in range(a.arenas):
            sc = Scenario(map_id, 100 + k, st, mode=['random', 'plant', 'explode', 'wipe'][k % 4])
            if mesh is None:
                mesh = GpuMesh(sc.A.mesh.tri, DEV) if sc.A.mesh is not None else wb_mesh(sc.A.static)
            t0 = time.time()
            snaps = sc.run(a.steps, a.snap)
            print('%s arena %d (%s): %d snapshots, %.1fs; per-step emitter parity: %d / %d steps differ' % (
                map_id, k, sc.mode, len(snaps), time.time() - t0, len(sc.audio_diffs), sc.n_steps), flush=True)
            for d_ in sc.audio_diffs[:4]:
                print('     step %d tick %d bomb %s planted %s fuse %s\n       gpu %s\n       cpu %s' % d_)
            n_audio_bad += len(sc.audio_diffs); n_audio += sc.n_steps; emit_kinds.update(sc.emit_kinds)
            snaps_all += snaps
        n = len(snaps_all)
        S = cat_states([to_torch(s['state']) for s in snaps_all])
        parts_cpu = torch.tensor(np.stack([s['parts'] for s in snaps_all]), device=DEV)
        tick = S['tick']
        # ---- visibility for the GPU observation: gpusim.vis with the CPU smokes of each snapshot
        from fpssim.grenades import segment_intersects_smoke

        def smoke_fn(p0, p1, env, tk):
            out = torch.zeros(len(env), dtype=torch.bool)
            ec = env.cpu().numpy()
            if not any(snaps_all[i]['smokes'] for i in set(ec.tolist())):
                return out.to(DEV)
            a0, a1 = p0.cpu().numpy(), p1.cpu().numpy()
            for r_, i in enumerate(ec):
                sm = snaps_all[i]['smokes']; now = snaps_all[i]['tick']
                if sm and any(s_['expires'] > now and segment_intersects_smoke(a0[r_], a1[r_], s_, now) for s_ in sm):
                    out[r_] = True
            return out.to(DEV)
        vis_gpu = visible(S, mesh, tick, smoke_fn)
        diff = (vis_gpu != parts_cpu)
        n_smoke = sum(1 for sn_ in snaps_all if sn_['smokes'])
        print('  gpusim.vis (+ CPU smokes, %d snapshots with smoke) vs CPU parts: %d / %d (obs,tgt,part) triples differ, %d snapshots affected' % (
            n_smoke, int(diff.sum()), diff.numel(), int(diff.flatten(1).any(1).sum())), flush=True)
        n_vis_diff = int(diff.sum())
        vis = vis_gpu
        em = cat_audio([s['emit_gpu'] for s in snaps_all])
        obs = build_obs(S, vis, st, mesh=mesh, audio=em)
        sync()
        assert obs.shape == (n, 10, OBS_DIM) and obs.dtype == torch.float32, obs.shape
        c = Cmp()
        for e in range(n):
            for seat in range(10):
                gpu = decode_obs(obs, e, seat, st)
                compare(c, snaps_all[e]['cpu'][seat], gpu, snaps_all[e]['tick'], snaps_all[e]['round_start'])
            if not c.eq('emitters', emitter_keys_gpu(em, e), emitter_keys_cpu(snaps_all[e]['emit_cpu'])):
                sb = snaps_all[e]['state']
                print('   emitter mismatch: tick %d bomb_state %d planted %d fuse_end %d\n     gpu %s\n     cpu %s' % (
                    sb['tick'][0], sb['bomb_state'][0], sb['bomb_planted_tick'][0], sb['bomb_fuse_end'][0],
                    emitter_keys_gpu(em, e), emitter_keys_cpu(snaps_all[e]['emit_cpu'])))
        tot = sum(c.n.values()); bad = sum(c.bad.values())
        print('== %s: %d snapshots x 10 seats, %d field comparisons, %d mismatches' % (map_id, n, tot, bad))
        for k in sorted(c.bad, key=lambda k: -c.bad[k]):
            print('   MISMATCH %-28s %d / %d  e.g. gpu=%r cpu=%r' % (k, c.bad[k], c.n[k], str(c.ex[k][0])[:90], str(c.ex[k][1])[:90]))
        # coverage of what was exercised
        cov = collections.Counter(); kinds = collections.Counter(); dirs = collections.Counter()
        for e in range(n):
            for seat in range(10):
                o = snaps_all[e]['cpu'][seat]
                cov['visible_entries'] += len(o['visible']); cov['sounds'] += len(o['sounds']); cov['world_items'] += len(o['world_items'])
                cov['bomb_known_defender'] += int(o['side'] == 'defend' and 'position' in o['bomb'])
                cov['reloading'] += sum(1 for w in o['me']['weapons'] if w and w['reload_state'])
                cov['blind'] += int(o['me']['blind_ticks'] > 0); cov['dead_seats'] += int(not o['me']['alive'])
                cov['plant_progress'] += int('plant_progress' in o['bomb']); cov['defuse_progress'] += int('defuse_progress' in o['bomb'])
                cov['throwables'] += len(o['me']['throwables'])
                for x_ in o['sounds']: kinds[x_['type']] += 1; dirs[x_['direction'] + '/' + x_['distance_band']] += 1
        cov['phases'] = sorted({s['cpu'][0]['phase'] for s in snaps_all}); cov['bomb_states'] = sorted({s['cpu'][0]['bomb']['state'] for s in snaps_all} | {s['cpu'][5]['bomb']['state'] for s in snaps_all})
        print('   coverage:', dict(cov))
        print('   heard sound kinds:', dict(kinds))
        print('   heard direction/band:', dict(dirs))
        summary[map_id] = dict(snapshots=n, comparisons=tot, mismatches=bad, vis_diff=n_vis_diff, by_field=dict(c.bad))
        rng_report(obs)
        # ---- legality (perturbation) test
        legality(S, vis, st, mesh, em, obs, map_id)
        if bench_state is None or map_id.startswith('dust2'):
            bench_state = (S, vis, st, mesh, em)
    # ---- throughput
    S, vis, st, mesh, em = bench_state
    bench(S, vis, st, mesh, em)
    print('emitters per kind over all arena steps:', dict(emit_kinds))
    print('per-step emitter parity (every physics step of every arena): %d / %d steps identical' % (n_audio - n_audio_bad, n_audio))
    print('RESULT', json.dumps(summary))
    print('total %.1fs' % (time.time() - t_start))


def legality(S, vis, st, mesh, em, obs0, map_id):
    """Hidden enemy information must not change an observer's tensor."""
    n = S['x'].shape[0]
    g = torch.Generator(device=DEV); g.manual_seed(7)
    rnd = lambda *sh: torch.rand(*sh, generator=g, device=DEV, dtype=torch.float64)
    results = {}
    for team_pert in (0, 1):
        pm = torch.zeros(n, 10, dtype=torch.bool, device=DEV); pm[:, team_pert * 5:team_pert * 5 + 5] = True
        obs_rows = slice((1 - team_pert) * 5, (1 - team_pert) * 5 + 5)           # observers of the other team
        # enemies (team_pert) visible to any observer of the other team: only private fields are perturbed for them
        seen = vis[:, obs_rows].any(-1).any(1)                                       # [n,10] over targets
        seen_full = torch.zeros(n, 10, dtype=torch.bool, device=DEV)
        seen_full[:, :] = seen
        pub = pm & ~seen_full
        S2 = {k: v.clone() for k, v in S.items()}
        dF = lambda k, m, scale: S2.__setitem__(k, torch.where(m, S2[k] + (rnd(*S2[k].shape) - .5) * scale, S2[k]))
        for k, sc in (('x', 60.0), ('y', 60.0), ('z', 6.0), ('vx', 8.0), ('vy', 8.0), ('vz', 4.0), ('yaw', 6.0), ('pitch', 1.0)):
            dF(k, pub, sc)
        for k in ('crouch', 'grounded'):
            S2[k] = torch.where(pub, ~S2[k], S2[k])
        S2['movement_mode'] = torch.where(pub, (S2['movement_mode'] + 1) % 4, S2['movement_mode'])
        S2['active'] = torch.where(pub, (S2['active'] + 1) % 3, S2['active'])
        # private fields of the whole team
        S2['hp'] = torch.where(pm, 1 + rnd(n, 10) * 98, S2['hp'])
        S2['armor'] = torch.where(pm, rnd(n, 10) * 100, S2['armor'])
        S2['helmet'] = torch.where(pm, ~S2['helmet'], S2['helmet'])
        S2['kit'] = torch.where(pm, ~S2['kit'], S2['kit'])
        S2['cash'] = torch.where(pm, (rnd(n, 10) * 16000).long(), S2['cash'])
        S2['nades'] = torch.where(pm[..., None], (rnd(n, 10, 6) * 3).long(), S2['nades'])
        pm3 = pm[..., None]
        tk3 = S['tick'][:, None, None]
        for k in ('w_ammo', 'w_reserve', 'w_shots'):
            S2[k] = torch.where(pm3, (rnd(n, 10, 3) * 50).long(), S2[k])
        S2['w_shot_ready'] = torch.where(pm3, tk3 + (rnd(n, 10, 3) * 30).long(), S2['w_shot_ready'])
        act3 = torch.nn.functional.one_hot(S['active'].long(), 3).bool()
        S2['w_wid'] = torch.where(pm3 & (~act3 | pub[..., None]), (S['w_wid'] + 5) % len(WEAPON_IDS), S['w_wid'])   # hidden: slots of unseen players, inactive slots of seen ones
        S2['w_inacc'] = torch.where(pm3, rnd(n, 10, 3) * .05, S2['w_inacc'])
        S2['w_rs'] = torch.where(pm3[..., None], rnd(n, 10, 3, 7) * .1, S2['w_rs'])
        S2['blind_until'] = torch.where(pm, S['tick'][:, None] + (rnd(n, 10) * 200).long(), S2['blind_until'])
        S2['throw_ready'] = torch.where(pm, S['tick'][:, None] + (rnd(n, 10) * 100).long(), S2['throw_ready'])
        S2['switch_until'] = torch.where(pm, S['tick'][:, None] + (rnd(n, 10) * 100).long(), S2['switch_until'])
        S2['pickup_ready_tick'] = torch.where(pm, S['tick'][:, None] + (rnd(n, 10) * 100).long(), S2['pickup_ready_tick'])
        S2['step_distance'] = torch.where(pm, rnd(n, 10) * 100, S2['step_distance'])
        o2 = build_obs(S2, vis, st, mesh=mesh, audio=em)
        rows = obs0[:, obs_rows] ; rows2 = o2[:, obs_rows]
        # bomb: has_bomb / objective of the perturbed team are not touched (bomb_view decides what is public)
        same = torch.equal(rows, rows2)
        nbad = int((rows != rows2).any(-1).sum())
        print('  legality (perturb hidden state of team %d, compare team %d observers): %s (%d / %d rows changed)' % (
            team_pert, 1 - team_pert, 'OK' if same else 'FAIL', nbad, rows.shape[0] * rows.shape[1]))
        results[team_pert] = same
    # unseen world items and an unseen bomb
    S3 = {k: v.clone() for k, v in S.items()}
    if 'wi_valid' in S3:
        wp = GO.point_visible(S, mesh, S['wi_pos'], S['wi_valid'], None, S['tick'])
        unseen = S['wi_valid'] & ~wp.any(1)
        S3['wi_pos'] = torch.where(unseen[..., None], S['wi_pos'] + 1000.0, S['wi_pos'])
        S3['wi_valid'] = S['wi_valid'] & ~unseen
        o3 = build_obs(S3, vis, st, mesh=mesh, audio=em)
        print('  legality (drop world items nobody sees): %s' % ('OK' if torch.equal(o3, obs0) else 'FAIL'), '(%d items dropped)' % int(unseen.sum()))
    S4 = {k: v.clone() for k, v in S.items()}
    at = S['attack_team']
    defenders = (torch.arange(10, device=DEV)[None, :] // 5) != at[:, None]
    held = S['bomb_state'] == 0
    bv = GO.point_visible(S, mesh, S['bomb_pos'][:, None], (S['bomb_state'] >= 1)[:, None], None, S['tick'])[:, :, 0]
    ctr = S['bomb_carrier'].clamp(min=0)
    cv = torch.gather(vis.any(-1), 2, ctr[:, None, None].expand(-1, 10, 1)).squeeze(-1) & (S['bomb_carrier'] >= 0)[:, None]
    known = torch.where(held[:, None], cv, bv) & defenders
    unknown_all = ~known.any(1)
    S4['bomb_pos'] = torch.where(unknown_all[:, None], S['bomb_pos'] + torch.tensor([300.0, -300.0, 50.0], dtype=torch.float64, device=DEV), S['bomb_pos'])
    S4['bomb_site'] = torch.where(unknown_all & (S['bomb_state'] < 2), torch.full_like(S['bomb_site'], -1), S['bomb_site'])
    o4 = build_obs(S4, vis, st, mesh=mesh, audio=em)
    bv2 = GO.point_visible(S4, mesh, S4['bomb_pos'][:, None], (S4['bomb_state'] >= 1)[:, None], None, S['tick'])[:, :, 0]
    cmp_rows = defenders & unknown_all[:, None] & torch.where(held[:, None], cv, bv2).logical_not()
    same4 = torch.equal(obs0[cmp_rows], o4[cmp_rows])
    print('  legality (move a bomb that no defender knows about; %d defender rows, %d envs): %s' % (
        int(cmp_rows.sum()), int(unknown_all.sum()), 'OK' if same4 else 'FAIL'))
    # held bomb: swap the (hidden) carrier for another attacker that no defender sees
    S5 = {k: v.clone() for k, v in S.items()}
    seen_by_def = (vis.any(-1) & defenders[:, :, None]).any(1)                      # [n,10] targets seen by some defender
    atk = ~defenders & S['alive'] & ~seen_by_def
    atk.scatter_(1, S['bomb_carrier'].clamp(min=0)[:, None], False)
    newc = atk.float().argmax(1); okc = atk.any(1) & held & unknown_all & (S['bomb_carrier'] >= 0)
    S5['bomb_carrier'] = torch.where(okc, newc, S['bomb_carrier'])
    hb = S['has_bomb'].clone()
    hb[okc] = False; hb[torch.nonzero(okc)[:, 0], newc[okc]] = True
    S5['has_bomb'] = hb
    o5 = build_obs(S5, vis, st, mesh=mesh, audio=em)
    rows5 = defenders & okc[:, None]
    print('  legality (swap the hidden carrier; %d defender rows): %s' % (int(rows5.sum()), 'OK' if torch.equal(obs0[rows5], o5[rows5]) else 'FAIL'))


def bench(S, vis, st, mesh, em):
    n = S['x'].shape[0]
    for E in (4096, 16384):
        idx = torch.randint(0, n, (E,), device=DEV)
        Sx = {k: v[idx] for k, v in S.items()}
        vx = vis[idx]
        emx = {k: v[idx] for k, v in em.items()}
        out = torch.empty(E, 10, OBS_DIM, dtype=torch.float32, device=DEV)
        torch.cuda.reset_peak_memory_stats(); m0 = torch.cuda.memory_allocated()
        build_obs(Sx, vx, st, mesh=mesh, audio=emx, out=out); sync()
        t0 = time.time()
        for _ in range(5): build_obs(Sx, vx, st, mesh=mesh, audio=emx, out=out)
        sync(); dt = (time.time() - t0) / 5
        peak = (torch.cuda.max_memory_allocated() - m0) / 2 ** 20
        t0 = time.time()
        for _ in range(5): build_obs(Sx, vx, st, mesh=None, audio=None, out=out)
        sync(); dt2 = (time.time() - t0) / 5
        print('  build_obs E=%d: %.1f ms (%.1f ms without audio / mesh queries), obs tensor %.0f MB, peak extra %.0f MB' % (
            E, dt * 1e3, dt2 * 1e3, out.numel() * 4 / 2 ** 20, peak), flush=True)


if __name__ == '__main__':
    main()
