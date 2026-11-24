"""Batched legal-observation tensors for training (GPU port of fpssim/obs.py, vectorized torch, no CUDA C++).

    from gpusim.obs import ObsStatic, GpuAudio, build_obs, decode_obs, SCHEMA, OBS_DIM
    st  = ObsStatic(static_map_dict, device)             # once per map (centre, zone landmarks, sites, map id)
    au  = GpuAudio(E, device)                            # sound emitter buffer (AudioEmitter of the CPU Arena)
    ...  every physics tick:  prev = au.snapshot(s) (before movement);  ... tick ...;  au.step(prev, s, events)
    ...  at every decision boundary (tick % 3 == 0):
    vis = gpusim.vis.visible(s, mesh, t, smoke_fn)       # bool [E,10 obs,10 tgt,3 parts]
    obs = build_obs(s, vis, st, mesh=mesh, audio=au.flush(fires), smoke_fn=smoke_fn)      # float32 [E,10,OBS_DIM]
    decode_obs(obs, e, seat, st)                          # readable nested dict (same names as fpssim.obs.build_observation)

Only legal information is read for the observer `seat` (rules: fpssim/obs.py, docs/obs.md, guide section 5): the observer's own
state, public round info, the 4 teammates' {alive, cash, zone, equipment}, enemies that `vis_parts` marks as visible (pose,
weapon id, parts only), bomb per side visibility, visible world items, and heard sounds.  Enemy hp / cash / ammo / inventory and
the position of non-visible enemies are never read into the tensor (tested by a perturbation test, gpusim/test_obs.py).

All normalisations are in the tables below (`SCHEMA` has name, start, length, description per field).  Times are encoded as the
remaining ticks relative to the current match tick, so the tensor is invariant to the absolute clock.

State keys read from `s` (SPEC names; optional keys in brackets default to the value shown):
  players [E,10]: x y z vx vy vz yaw pitch f64; alive crouch grounded bool; hp armor f64; helmet kit bool; cash; step_distance;
     movement_mode (0 ground,1 air,2 ladder,3 ladder_exit) ladder_id (-1 none) [reattach_until | mv_reattach_until 0]
     [stand_blocked | mv_stand_blocked False]
     blind_until throw_ready pickup_ready_tick; has_bomb bool; objective (0 none,1 plant,2 defuse) obj_progress;
     [walk_held | act_walk False] [nades [E,10,6] i32 (tables.UTIL_IDS order) 0]
  weapons [E,10,3]: w_wid w_ammo w_reserve w_shots w_shot_ready w_reload_until w_reload_commit w_last_shot (-1 none)
     w_reload_committed w_reload_shell w_pressed w_inacc w_rs[...,7]; active switch_until zoom [E,10]
  match [E]: tick [t0] round_index round_start_tick live_end post_end phase(0 buy 1 live 2 post 3 finished) attack_team;
     round_wins loss_level [E,2]; bomb_state (0 held 1 world 2 planted 3 defused 4 exploded) bomb_carrier (-1) bomb_pos [E,3]
     bomb_fuse_end (<=0 none) bomb_duration [bomb_site (-1 none; else nearest site)] [overtime_number] [win_target]
  world items [E,K]: [wi_valid wi_kind (index into tables.WEAPON_IDS) wi_pos [E,K,3]]  (absent -> no items)
"""
import math
import re
import numpy as np
import torch
import torch.nn.functional as F

from .tables import WEAPON_IDS, UTIL_IDS

W = len(WEAPON_IDS)
NU = len(UTIL_IDS)
KNIFE = WEAPON_IDS.index('knife')
MAP_IDS = ['dust2-cs2-v54', 'mirage-cs2-v54', 'borgo-r3-v54', 'molo-r3-v54', 'nevaio-r3-v54']

EYE_STAND, EYE_CROUCH = 1.62, 0.95
HALF_H, HALF_V = math.radians(55.0), math.radians(45.0)

# ---------------------------------------------------------------------------------------------- normalisation scales
POS = 100.0            # (p - map centre) / 100 m
VEL = 10.0             # m/s
REL = 50.0             # view-frame relative offsets, m
DIST = 100.0           # m
CASH = 16000.0
TICK = 200000.0        # match tick
RTICK = 7560.0         # round tick / phase remaining (buy 360 + live 6900 + post 300)
SEC = 60.0             # timers (shot_ready, reload, switch, throw_ready ...) in seconds
BLIND = 300.0          # blind ticks
FUSE = 2400.0          # bomb fuse ticks
PROG = 600.0           # plant / defuse progress and duration (ticks)
ANG = 10.0             # punch / view punch angles (rad) scale
INACC = 20.0           # inaccuracy (rad) scale
SHOTS = 30.0
SINCE = 600.0          # ticks since last shot (clipped at SINCE_CLIP ticks)
SINCE_CLIP = 6000
AMMO, RESERVE = 100.0, 200.0

NZ, NEN, NTM, NITEM, NSND, CAP = 4, 5, 4, 6, 8, 96
# sound kinds that can reach `sounds` (fpssim.obs.HEARD_BY_OTHERS | HEARD_BY_SELF); gunshots, draw, grenade_thrown, plant / defuse start never do
SOUND_KINDS = ['footstep', 'jump', 'land_soft', 'land_hard', 'reload', 'grenade_bounce', 'bomb_planted', 'bomb_defused', 'bomb_exploded',
               'bomb_beep', 'bomb_beep_fast', 'fire_started', 'fire_ambient']
DIRS = ['self', 'unknown', 'left', 'right', 'center']
BANDS = ['self', 'unknown', 'near', 'mid', 'far']
PHASES = ['buy', 'live', 'post', 'finished']
MODES = ['ground', 'air', 'ladder', 'ladder_exit']
BOMB_STATES = ['held', 'world', 'planted', 'defused', 'exploded', 'unknown']
NK = len(SOUND_KINDS)

# ---------------------------------------------------------------------------------------------- schema
_ROUND = [
    ('map', 5, 'one-hot map id (dust2, mirage, borgo, molo, nevaio)'),
    ('match_tick', 1, 'match_tick / 2e5'),
    ('round_index', 1, 'round_index / 24'),
    ('round_tick', 1, 'ticks since round start / 7560'),
    ('phase', 4, 'one-hot buy, live, post, finished'),
    ('phase_remaining', 1, 'phase_remaining_ticks / 7560'),
    ('team', 1, 'own team (0/1)'),
    ('seat', 5, 'one-hot own seat index inside the team'),
    ('side_attack', 1, '1 if own team attacks (side == attack)'),
    ('round_wins', 2, '(own, enemy) round wins / 13'),
    ('alive', 2, '(own, enemy) alive count / 5'),
    ('loss_level', 1, 'own loss_level / 5'),
    ('overtime', 1, '1 if period != regulation'),
    ('overtime_number', 1, 'overtime_number / 10'),
    ('win_target', 1, 'win_target / 13'),
    ('buy_remaining', 1, '(buy_deadline_tick - match_tick) / 1860 (negative once the deadline has passed)'),
]
_ME = [
    ('pos', 3, '(x,y,z) - map centre, / 100 m'),
    ('yaw', 2, '(sin, cos) of yaw'),
    ('pitch', 1, 'pitch / (pi/2)'),
    ('vel', 3, '(vx,vy,vz) / 10 m/s'),
    ('vel_view', 2, '(forward, right) horizontal velocity in the view frame / 10'),
    ('hp', 1, 'hp / 100'), ('armor', 1, 'armor / 100'), ('helmet', 1, 'helmet flag'), ('has_kit', 1, 'defuse kit flag'),
    ('cash', 1, 'cash / 16000'), ('alive', 1, 'alive'), ('crouch', 1, 'crouch'), ('grounded', 1, 'grounded'),
    ('walk', 1, 'walk held (= me.walk = me.quiet)'), ('stand_blocked', 1, 'stand_blocked'),
    ('active', 3, 'one-hot active_slot'), ('zoom', 3, 'one-hot zoom_level (0,1,2); ads_progress = zoom>0'),
    ('movement_mode', 4, 'one-hot ground, air, ladder, ladder_exit'),
    ('ladder_id', 1, '(ladder_id + 1) / 64 (0 = none)'),
    ('reattach_rem', 1, 'ticks until reattach_until / 60'),
    ('pickup_rem', 1, 'ticks until pickup_ready_tick / 60'), ('switch_rem', 1, 'ticks until switch_until / 60'),
    ('throw_rem', 1, 'ticks until throw_ready / 60'), ('blind_rem', 1, 'blind_ticks / 300'),
    ('throwables', NU, 'count of (hegrenade, flashbang, smokegrenade, molotov, incgrenade, decoy) / 2'),
    ('carrying_bomb', 1, 'carrying_bomb'),
]
_WEAP = [
    ('present', 1, 'slot holds a weapon'), ('id', W, 'one-hot weapon id (tables.WEAPON_IDS)'),
    ('ammo', 1, 'ammo / 100'), ('reserve', 1, 'reserve / 200'), ('shot_ready', 1, 'ticks until shot_ready / 60'),
    ('reloading', 1, 'reload_until is set'), ('reload_rem', 1, 'ticks until reload_until / 60'),
    ('commit_rem', 1, 'ticks until reload_state.commit / 60'), ('committed', 1, 'reload_state.committed'),
    ('shell', 1, 'reload_state.shell'), ('shots', 1, 'shots / 30'),
    ('has_last_shot', 1, 'last_shot is set'), ('since_shot', 1, 'min(ticks since last_shot, 6000) / 600'),
    ('punch_index', 1, 'aim_punch.index / 64'), ('punch', 2, '(pitch, yaw) aim punch * 10 rad (= recoil_pitch/yaw)'),
    ('punch_vel', 2, '(velPitch, velYaw) rad/s'), ('view_punch', 2, '(viewPitch, viewYaw) * 10 rad'),
    ('inaccuracy', 1, 'inaccuracy * 20 rad'), ('pressed', 1, 'trigger_state.pressed'),
]
_ENEMY = [
    ('valid', 1, 'slot holds a visible enemy (slots sorted by distance, nearest first)'),
    ('seat', 5, 'one-hot enemy seat index inside the enemy team'),
    ('pos', 3, 'enemy (x,y,z) - map centre, / 100'), ('rel', 3, '(forward, right, up) offset from own feet in own view frame / 50'),
    ('dist', 1, 'distance / 100'), ('yaw', 2, '(sin, cos) enemy yaw'), ('pitch', 1, 'enemy pitch / (pi/2)'),
    ('crouch', 1, 'enemy crouch'), ('weapon', W, 'one-hot appearance weapon id (active weapon)'),
    ('movement_mode', 4, 'one-hot enemy movement_mode'), ('parts', 3, 'visible parts (head, chest, abdomen)'),
]
_TEAM = [
    ('seat', 5, 'one-hot teammate seat index inside the team'), ('alive', 1, 'alive'), ('cash', 1, 'cash / 16000'),
    ('zone', NZ, 'one-hot nearest route landmark (ObsStatic.zone_names; zeros for none)'),
    ('weapons', W, 'multi-hot weapon ids carried (equipment, knife excluded; zeros when dead)'),
    ('throwables', NU, 'throwable counts / 2 (zeros when dead)'), ('armor', 1, 'armor > 0 (equipment)'),
    ('helmet', 1, 'helmet (equipment)'), ('kit', 1, 'defusekit (equipment)'),
]
_BOMB = [
    ('state', 6, 'one-hot shown state: held, world, planted, defused, exploded, unknown'),
    ('known', 1, 'id/carrier/position/site are given (own team attacks, or bomb / carrier visible)'),
    ('fuse_valid', 1, 'fuse_end_tick is not null'), ('fuse_rem', 1, '(fuse_end_tick - match_tick) / 2400'),
    ('carrier', 10, 'one-hot carrier seat (known and carried)'), ('carrier_me', 1, 'carrier is the observer'),
    ('has_pos', 1, 'position given'), ('pos', 3, 'bomb position - map centre, / 100'),
    ('rel', 3, 'bomb offset in own view frame / 50'), ('dist', 1, 'distance to bomb / 100'),
    ('site', NZ, 'one-hot site (ObsStatic.site_names), zeros when null'),
    ('is_planter', 1, 'observer is planting (plant_progress given)'), ('plant_progress', 1, 'plant progress ticks / 600'),
    ('is_defuser', 1, 'observer is defusing (defuse_progress/duration given)'), ('defuse_progress', 1, 'ticks / 600'),
    ('defuse_duration', 1, 'ticks / 600'),
]
_ITEM = [
    ('valid', 1, 'slot holds a visible world item (nearest first)'), ('kind', W, 'one-hot weapon id of the item'),
    ('pos', 3, 'item (x,y,z) - map centre, / 100'), ('rel', 3, 'offset in own view frame / 50'), ('dist', 1, 'distance / 100'),
]
_SOUND = [
    ('valid', 1, 'slot holds a heard sound (emission order)'), ('kind', NK, 'one-hot sound type (SOUND_KINDS)'),
    ('dir', 5, 'one-hot direction: self, unknown, left, right, center'),
    ('band', 5, 'one-hot distance_band: self, unknown, near, mid, far'),
]


class _Layout:
    def __init__(self):
        self.items = []
        self.off = {}
        self.n = 0

    def add(self, name, n, desc):
        self.off[name] = (self.n, n)
        self.items.append((name, self.n, n, desc))
        self.n += n

    def group(self, prefix, spec, count=1):
        for k in range(count):
            pre = prefix if count == 1 else '%s%d' % (prefix, k)
            for name, n, desc in spec:
                self.add('%s.%s' % (pre, name), n, desc)

    def sl(self, name):
        a, n = self.off[name]
        return slice(a, a + n)


_L = _Layout()
_L.group('rd', _ROUND)
_L.group('me', _ME)
for _k in range(3):
    _L.group('me.w%d' % _k, _WEAP)
_L.group('en', _ENEMY, NEN)
_L.group('tm', _TEAM, NTM)
_L.group('bomb', _BOMB)
_L.group('wi', _ITEM, NITEM)
_L.add('wi.count', 1, 'number of visible world items / 8 (slots hold the nearest %d)' % NITEM)
_L.group('snd', _SOUND, NSND)
_L.add('snd.count', 1, 'number of heard sounds / 8 (slots hold the first %d)' % NSND)
SCHEMA = [(n, a, l, d) for n, a, l, d in _L.items]
OBS_DIM = _L.n
OFF = _L.off


def _spec_len(spec):
    return sum(n for _, n, _ in spec)


def schema_table():
    """markdown table of SCHEMA (repeated groups collapsed to their first entry)."""
    rows = ['| range | name | len | description |', '|---|---|---|---|']
    skip = set()
    for name, a, n, d in SCHEMA:
        pre = name.split('.')[0]
        if pre[-1].isdigit() and pre[:-1] in ('en', 'tm', 'wi', 'snd') and not pre.endswith('0'):
            continue
        if re.match(r'me\.w[12]\.', name):
            continue
        rows.append('| %d-%d | %s | %d | %s |' % (a, a + n - 1, name, n, d))
    return '\n'.join(rows)


# ---------------------------------------------------------------------------------------------- static map data
class ObsStatic:
    """Per-map constants: normalisation centre, zone landmarks, sites, map id."""

    def __init__(self, static, device='cuda'):
        m = static.get('map', static)
        self.id = m.get('id', '')
        self.map_idx = MAP_IDS.index(self.id) if self.id in MAP_IDS else -1
        b = m.get('bounds') or {}
        lo = list(b.get('min', [0.0, 0.0, 0.0])) + [0.0] * 3
        hi = list(b.get('max', [0.0, 0.0, 0.0])) + [0.0] * 3
        self.center = torch.tensor([(lo[i] + hi[i]) / 2 for i in range(3)], dtype=torch.float64, device=device)
        lm = m.get('route_landmarks') or {}
        if not lm:
            lm = {s['id']: s['center'] for s in m.get('sites') or []}
        self.zone_names = list(lm)[:NZ]
        self.zone_pos = torch.tensor([lm[k] for k in self.zone_names] or np.zeros((0, 3)), dtype=torch.float64,
                                     device=device).reshape(-1, 3)
        sites = m.get('sites') or []
        self.site_names = [s['id'] for s in sites][:NZ]
        self.site_pos = torch.tensor([s.get('center', [0, 0, 0]) for s in sites][:NZ] or np.zeros((0, 3)),
                                     dtype=torch.float64, device=device).reshape(-1, 3)
        self.device = device


# ---------------------------------------------------------------------------------------------- small helpers
def _f(x):
    return x.float().unsqueeze(-1)


def _oh(idx, n, mask=None):
    o = F.one_hot(idx.clamp(0, n - 1).long(), n).float()
    if mask is not None:
        o = o * mask.unsqueeze(-1).float()
    return o


def _view_frame(dx, dy, dz, yaw):
    c, sn = torch.cos(yaw), torch.sin(yaw)
    return dx * c - dy * sn, -dx * sn - dy * c, dz            # forward (cos y, -sin y), right (-sin y, -cos y)


def _rel(d3, yaw):
    f, r, u = _view_frame(d3[..., 0], d3[..., 1], d3[..., 2], yaw)
    return torch.stack([f, r, u], -1)


def _eye(crouch):
    """eye height above the feet in float64 (1.62 standing, 0.95 crouched)."""
    return torch.where(crouch, torch.tensor(EYE_CROUCH, dtype=torch.float64, device=crouch.device),
                       torch.tensor(EYE_STAND, dtype=torch.float64, device=crouch.device))


def _g(s, k, default=None):
    v = s.get(k)
    return default if v is None else v


def point_visible(s, mesh, pts, valid, smoke_fn=None, tick=None, backoff=0.03):
    """fpssim.obs.VisModel.point_visible for many points: pts [E,K,3] f64, valid [E,K] -> bool [E,10 observers,K].
    Point inside the 55/45 degree cone of an alive observer and the segment eye -> point (pulled back 3 cm) is not blocked
    by the static mesh (nor by smoke if smoke_fn(p0, p1, env, tick) is given)."""
    E, K = valid.shape
    dev = pts.device
    ex = s['x'][:, :, None]
    ey = s['y'][:, :, None]
    ez = (s['z'] + _eye(s['crouch']))[:, :, None]
    dx = pts[:, None, :, 0] - ex
    dy = pts[:, None, :, 1] - ey
    dz = pts[:, None, :, 2] - ez
    dh = torch.hypot(dx, dy)
    da = torch.atan2(-dy, dx) - s['yaw'][:, :, None]
    wrap = torch.atan2(torch.sin(da), torch.cos(da))
    in0 = (dh < 1e-9) & (dz.abs() < 1e-9)
    cone = in0 | ((wrap.abs() <= HALF_H) & ((torch.atan2(dz, dh) - s['pitch'][:, :, None]).abs() <= HALF_V))
    cand = cone & valid[:, None, :] & s['alive'][:, :, None]
    out = torch.zeros(E, 10, K, dtype=torch.bool, device=dev)
    if mesh is None:
        return cand
    idx = cand.nonzero(as_tuple=False)
    if len(idx) == 0:
        return out
    e, o, k = idx.unbind(1)
    d = torch.stack([dx[e, o, k], dy[e, o, k], dz[e, o, k]], -1)
    n = d.norm(dim=-1)
    n = torch.where(n == 0, torch.ones_like(n), n)
    p0 = torch.stack([ex[e, o, 0], ey[e, o, 0], ez[e, o, 0]], -1)
    p1 = pts[e, k] - backoff * d / n[:, None]
    clear = ~mesh.blocked(p0, p1)
    if smoke_fn is not None:
        t = tick if tick is not None else s['tick']
        clear &= ~smoke_fn(p0, p1, e, t)
    out[e, o, k] = clear
    return out


# ---------------------------------------------------------------------------------------------- audio (AudioEmitter + _audible)
# internal emitter types -> public kind and per-type constants (fpssim.obs.AudioEmitter, HEARD_BY_OTHERS / HEARD_BY_SELF)
_IT = ['foot_r', 'foot_l', 'jump', 'land_soft', 'land_hard', 'reload', 'bounce', 'planted', 'defused', 'exploded', 'beep', 'beep_fast',
       'fire_started', 'fire_ambient']
_IT_KIND = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
_IT_RADIUS = [25.0, 25.0, 16.0, 25.0, 30.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, 24.0]    # -1 = no radius
_IT_DIRECT = [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
_IT_HASPOS = [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 1]
_IT_OTHERS = [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1]     # kind in HEARD_BY_OTHERS and not local_only (the local footstep copy / reload are own-only)
_IT_SELF = [0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]       # kind in HEARD_BY_SELF and not remote_only
_IT_LOCAL = [0, 1, 0, 0, 0, 1] + [0] * 8                    # local_only flags (for the emitter-parity test)
_IT_REMOTE = [1] + [0] * 13
IT_FOOT_R, IT_FOOT_L, IT_JUMP, IT_LAND_SOFT, IT_LAND_HARD, IT_RELOAD, IT_BOUNCE, IT_PLANTED, IT_DEFUSED, IT_EXPLODED, IT_BEEP, \
    IT_BEEP_FAST, IT_FIRE_STARTED, IT_FIRE_AMBIENT = range(14)
FOOT_INTERVAL, RUN_SPEED, SOFT_LAND, HARD_LAND = 1.85, 2.5, -1.5, -5.2
NEAR, MID, CENTER_PAN = 8.0, 20.0, 0.20
NB = 8        # bounce / fire_started events per tick
NFIRE = 12    # active fires considered for fire_ambient


def _it_tab(device):
    return {k: torch.tensor(v, device=device) for k, v in dict(
        kind=_IT_KIND, radius=_IT_RADIUS, direct=_IT_DIRECT, haspos=_IT_HASPOS, others=_IT_OTHERS, self_=_IT_SELF).items()}


class GpuAudio:
    """Per-env emitter buffer: AudioEmitter of the CPU Arena, restricted to the emitters that can be heard.  Call `step` once per physics
    tick and `flush` at each decision.  Like the platform (Arena.step) only the emitters of the decision tick itself are delivered:
    `step` drops whatever the previous tick left in the buffer.
    events (all optional, tensors): reload [E,10] bool (reload_start of the seat); bounce [E,NB] bool + bounce_pos [E,NB,3];
    fire_started [E,NB] bool + fire_started_pos [E,NB,3] (one per grenade `detonate` event of kind molotov / incgrenade in this tick,
    i.e. the CPU appends one for the plain 'detonate', one for 'fire_started' / 'fire_failed' and one for 'fire_extinguished');
    plant_complete defuse_complete bomb_explode [E] bool; beep [E] int (0 none, 1 beep, 2 fast beep, at s['bomb_pos']; default: the Arena
    schedule (fpssim.arena.BEEP_SLOW / BEEP_FAST, offsets since plant_complete) from bomb_state / bomb_fuse_end / bomb_planted_tick when
    those keys are in `s`).  fire / draw / throw / plant_start / defuse_start events are accepted and ignored (never audible).
    `flush(fires=(pos [E,F,3], valid [E,F]))` appends one `fire_ambient` per active fire (GpuGrenades fi_pos / fi_valid), like Arena.step."""

    SNAP = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'alive', 'crouch', 'grounded', 'step_distance')

    def __init__(self, E, device, cap=CAP):
        self.E, self.dev, self.cap = E, device, cap
        d = device
        self.n = torch.zeros(E, dtype=torch.long, device=d)
        self.itype = torch.zeros(E, cap, dtype=torch.int8, device=d)
        self.src = torch.full((E, cap), -1, dtype=torch.int8, device=d)
        self.pos = torch.zeros(E, cap, 3, dtype=torch.float64, device=d)
        self.last_step = torch.full((E, 10), float('nan'), dtype=torch.float64, device=d)
        self.acc = torch.zeros(E, 10, dtype=torch.float64, device=d)
        self._dropped = torch.zeros((), dtype=torch.long, device=d)     # overflowed emitters (device counter: no host sync per tick)
        # candidate columns of one tick: (itype, src)
        cols = []
        for it in (IT_FOOT_R, IT_FOOT_L, IT_JUMP, IT_LAND_SOFT, IT_RELOAD):          # one column per seat (land: soft / hard chosen per event)
            cols += [(it, s_) for s_ in range(10)]
        cols += [(IT_BOUNCE, -1)] * NB
        cols += [(IT_PLANTED, -1), (IT_DEFUSED, -1), (IT_EXPLODED, -1), (IT_BEEP, -1)]
        cols += [(IT_FIRE_STARTED, -1)] * NB
        self._cols = cols
        self._col_it = torch.tensor([c[0] for c in cols], dtype=torch.int8, device=d)
        self._col_src = torch.tensor([c[1] for c in cols], dtype=torch.int8, device=d)
        self.NC = len(cols)
        self._beep_tab = self._load_beep_tab(d)

    @staticmethod
    def _load_beep_tab(device):
        """bomb beep schedule: ticks since plant_complete -> 0 none / 1 beep / 2 fast beep (fpssim.arena.BEEP_SLOW / BEEP_FAST)."""
        from fpssim.arena import BEEP_SLOW, BEEP_FAST
        tab = torch.zeros(2401, dtype=torch.long, device=device)
        tab[torch.tensor(sorted(BEEP_SLOW), device=device)] = 1
        tab[torch.tensor(sorted(BEEP_FAST), device=device)] = 2
        return tab

    def snapshot(self, s):
        return {k: s[k].clone() for k in self.SNAP if k in s}

    def clear(self, env_mask=None):
        if env_mask is None:
            self.n.zero_()
        else:
            self.n.masked_fill_(env_mask.to(self.n.device).bool(), 0)

    def step(self, prev, s, events=None):
        ev = events or {}
        E, d = self.E, self.dev
        z = lambda *sh: torch.zeros(*sh, dtype=torch.bool, device=d)
        self.n.zero_()                                                     # only the decision tick's emitters are delivered
        # (state is updated in place everywhere: GpuSim replays this inside a CUDA graph)
        alive = s['alive']
        sd = s['step_distance']
        walk = _g(s, 'walk_held', _g(s, 'act_walk', torch.zeros_like(alive))).bool()
        speed = torch.hypot(s['vx'], s['vy'])
        # cs2-web footstep rule per 60 Hz tick (fpssim.obs.AudioEmitter): accumulate the horizontal step distance while grounded, not
        # crouching / walking and 2.5 <= speed <= 12; emit at >= 1.85 and keep the remainder; reset on crouch / walk, speed < 1,
        # and on teleports / respawn (step_distance decreases or jumps by > 2.5 m in one tick)
        old = self.last_step
        last = torch.where(torch.isnan(old), sd, old)
        move = sd - last
        self.last_step.copy_(torch.where(alive, sd, old))
        quiet = s['crouch'] | walk
        reset1 = (move < 0) | (move > 2.5)
        run = ~reset1 & s['grounded'] & ~quiet & (speed >= RUN_SPEED) & (speed <= 12.0)
        reset3 = ~reset1 & ~run & (quiet | (speed < 1.0))
        acc = torch.where(reset1 | reset3, torch.zeros_like(self.acc), torch.where(run, self.acc + move, self.acc))
        emit = alive & run & (acc >= FOOT_INTERVAL)
        acc = torch.where(emit, torch.remainder(acc, FOOT_INTERVAL), acc)
        self.acc.copy_(torch.where(alive, acc, self.acc))
        pg, pvz = prev['grounded'], prev['vz']
        jump = alive & pg & ~s['grounded'] & (s['vz'] > 1.5)
        land = alive & ~pg & s['grounded'] & (pvz <= SOFT_LAND)
        hard = pvz <= HARD_LAND
        if 'beep' in ev:
            beep = ev['beep']
        elif 'bomb_planted_tick' in s and 'bomb_state' in s:
            off = s['tick'].long() - s['bomb_planted_tick'].long()
            on = (s['bomb_state'] == 2) & (s['bomb_fuse_end'] > 0) & (off >= 0) & (off <= 2400)
            beep = torch.where(on, self._beep_tab[off.clamp(0, 2400)], 0)
        else:
            beep = torch.zeros(E, dtype=torch.long, device=d)
        mask = torch.cat([emit, emit, jump, land, ev.get('reload', z(E, 10)),
                          ev['bounce'] if 'bounce' in ev else z(E, NB),
                          torch.stack([ev.get(k, z(E)) for k in ('plant_complete', 'defuse_complete', 'bomb_explode')], 1),
                          (beep > 0)[:, None],
                          ev['fire_started'] if 'fire_started' in ev else z(E, NB)], 1)
        P = torch.stack([s['x'], s['y'], s['z']], -1)                      # [E,10,3] emitter position = player feet
        zer = lambda n: torch.zeros(E, n, 3, dtype=torch.float64, device=d)
        bpos = ev.get('bounce_pos', zer(NB))
        fpos = ev.get('fire_started_pos', zer(NB))
        bp = s.get('bomb_pos', torch.zeros(E, 3, dtype=torch.float64, device=d))
        pos = torch.cat([P] * 5 + [bpos, zer(3), bp[:, None], fpos], 1)
        it = self._col_it[None, :].expand(E, -1).clone()
        it[:, 30:40] = torch.where(hard, IT_LAND_HARD, IT_LAND_SOFT).to(torch.int8)   # land columns: land_hard where vz <= HARD_LAND
        it[:, 3 * 10 + 10 + 10 + NB + 3] = torch.where(beep == 2, IT_BEEP_FAST, IT_BEEP).to(torch.int8)     # beep column
        self._insert(mask, it, self._col_src[None, :].expand(E, -1), pos)

    @property
    def dropped(self):
        return int(self._dropped)

    def _insert(self, mask, it, src, pos):
        """append the masked columns in column order (no host sync: rejected / overflowing columns are scattered into a dump slot)."""
        csum = mask.long().cumsum(1)
        slot = self.n[:, None] + csum - 1
        ok = mask & (slot < self.cap)
        self._dropped += (mask & ~ok).sum()
        if not hasattr(self, '_flat'):
            E, cap = self.E, self.cap
            n = E * cap
            fi = torch.zeros(n + 1, dtype=torch.int8, device=self.dev); fs = torch.zeros(n + 1, dtype=torch.int8, device=self.dev)
            fp = torch.zeros(n + 1, 3, dtype=torch.float64, device=self.dev)
            fi[:n].view(E, cap).copy_(self.itype); fs[:n].view(E, cap).copy_(self.src); fp[:n].view(E, cap, 3).copy_(self.pos)
            self._flat = (fi, fs, fp)
            self.itype, self.src, self.pos = fi[:n].view(E, cap), fs[:n].view(E, cap), fp[:n].view(E, cap, 3)
            self._rowbase = torch.arange(E, device=self.dev)[:, None] * cap
        fi, fs, fp = self._flat
        k = torch.where(ok, self._rowbase + slot, torch.full_like(slot, self.E * self.cap)).reshape(-1)    # last element = dump slot
        fi.scatter_(0, k, it.to(torch.int8).reshape(-1))
        fs.scatter_(0, k, src.to(torch.int8).reshape(-1))
        fp.scatter_(0, k[:, None].expand(-1, 3), pos.to(torch.float64).reshape(-1, 3))
        self.n.copy_(torch.clamp(self.n + mask.long().sum(1), max=self.cap))

    def flush(self, fires=None, copy=True):
        """-> emitter table dict(valid [E,cap], itype, src, pos, n) and reset.  `fires = (pos [E,F,3] f64, valid [E,F] bool)` of the
        burning fires adds one fire_ambient emitter each (after the tick's emitters, like Arena.step).  copy=False returns views of the
        buffers (valid until the next step(); GpuSim.observe uses them right away)."""
        if fires is not None:
            fpos, fvalid = fires
            F_ = min(fpos.shape[1], NFIRE)
            E = self.E
            it = torch.full((E, F_), IT_FIRE_AMBIENT, dtype=torch.int8, device=self.dev)
            src = torch.full((E, F_), -1, dtype=torch.int8, device=self.dev)
            self._insert(fvalid[:, :F_].bool(), it, src, fpos[:, :F_])
        valid = torch.arange(self.cap, device=self.dev)[None, :] < self.n[:, None]
        if copy:
            out = dict(valid=valid, itype=self.itype.clone(), src=self.src.clone(), pos=self.pos.clone(), n=self.n.clone())
        else:
            out = dict(valid=valid, itype=self.itype, src=self.src, pos=self.pos, n=self.n.clone())
        self.n.zero_()
        return out


def hear(s, em):
    """fpssim.obs._audible for every (listener, emitter): -> kind [E,10,NSND] long (-1 none), dir, band, count [E,10].
    Platform rules (fit on our own training.jsonl): own sounds -> self / self (footstep, landing, reload only); others only for the
    whitelisted kinds; no position -> unknown / unknown; beyond the radius -> not heard; no occlusion; distance from the listener's FEET
    (near < 8, mid < 20, else far); direction mirrored (source on the right -> 'left'), 'center' when |sin(azimuth)| < 0.2,
    non-directional kinds 'unknown'; emitters are never merged; dead listeners hear nothing."""
    dev = s['x'].device
    cap = em['itype'].shape[1]
    tab = _it_tab(dev)
    it = em['itype'].long()
    src = em['src'].long()[:, None, :]                                                    # [E,1,C]
    is_src = src == torch.arange(10, device=dev)[None, :, None]
    valid = em['valid'][:, None, :] & s['alive'][:, :, None]
    selfhear = valid & is_src & tab['self_'][it].bool()[:, None, :]
    haspos = tab['haspos'][it].bool()[:, None, :]
    oth = valid & ~is_src & tab['others'][it].bool()[:, None, :]
    nopos = oth & ~haspos
    pos = em['pos']
    dx = pos[:, None, :, 0] - s['x'][:, :, None]
    dy = pos[:, None, :, 1] - s['y'][:, :, None]
    dz = pos[:, None, :, 2] - s['z'][:, :, None]
    d = torch.sqrt(dx * dx + dy * dy + dz * dz)
    rad = tab['radius'][it][:, None, :]
    cand = oth & haspos & ((rad < 0) | (d <= rad))
    sn = torch.sin(torch.atan2(-dy, dx) - s['yaw'][:, :, None])
    direct = tab['direct'][it].bool()[:, None, :]
    dcode = torch.where(direct, torch.where(sn.abs() < CENTER_PAN, 4, torch.where(sn > 0, 2, 3)), 1)
    bcode = torch.where(d < NEAR, 2, torch.where(d < MID, 3, 4))
    aud = selfhear | nopos | cand
    dcode = torch.where(selfhear, 0, torch.where(nopos, 1, dcode))
    bcode = torch.where(selfhear, 0, torch.where(nopos, 1, bcode))
    ar = torch.arange(cap, device=dev)[None, None, :]
    key = torch.where(aud, ar, cap + ar)
    k = min(NSND, cap)
    sel = torch.topk(key, k, dim=-1, largest=False, sorted=True).indices                  # first audible emitters in emission order
    ok = torch.gather(aud, 2, sel)
    knd = torch.where(ok, tab['kind'][torch.gather(it[:, None, :].expand(-1, 10, -1), 2, sel)], -1)
    dd = torch.where(ok, torch.gather(dcode, 2, sel), 0)
    bb = torch.where(ok, torch.gather(bcode, 2, sel), 0)
    if k < NSND:
        pad = NSND - k
        knd = F.pad(knd, (0, pad), value=-1); dd = F.pad(dd, (0, pad)); bb = F.pad(bb, (0, pad))
    return knd, dd, bb, aud.sum(-1)


# ---------------------------------------------------------------------------------------------- the builder
def _cat(spec, pieces):
    assert len(spec) == len(pieces), (len(spec), len(pieces))
    for (name, n, _), p in zip(spec, pieces):
        assert p.shape[-1] == n, (name, n, tuple(p.shape))
    return torch.cat(pieces, -1)


def build_obs(s, vis_parts, st, mesh=None, audio=None, smoke_fn=None, out=None, ctx=None, pv=None):
    """-> float32 [E,10,OBS_DIM].  s: state dict (module docstring), vis_parts bool [E,10 obs,10 tgt,3], st: ObsStatic,
    mesh: GpuMesh (bomb / world item point visibility; sounds have no occlusion), audio: GpuAudio.flush() table, smoke_fn(p0,p1,env,tick),
    out: optional preallocated float32 [E,10,OBS_DIM], ctx: optional scalars {'overtime_number': 0, 'win_target': 13} used when s lacks them.
    pv: optional precomputed point visibility bool [E,10,K+1] (gpusim.vis.trace points: world items, then the bomb).
    CUDA: the fused kernel (obs_fused; point visibility from gpusim.vis.trace, or point_visible for a foreign smoke_fn);
    CPU tensors or GPUSIM_OBS_TORCH=1: build_obs_torch."""
    import os
    if not s['x'].is_cuda or os.environ.get('GPUSIM_OBS_TORCH'):
        return build_obs_torch(s, vis_parts, st, mesh, audio, smoke_fn, out, ctx)
    if pv is None:
        pv = point_vis(s, mesh, smoke_fn)
    return obs_fused(s, vis_parts, st, pv, audio, out, ctx)


def point_vis(s, mesh, smoke_fn=None, tick=None):
    """point visibility of the world items and the bomb -> bool [E,10,K+1] (same values as the two point_visible calls of
    build_obs_torch)."""
    from . import vis as V
    tick = s['tick'] if tick is None else tick
    G = V.smoke_owner(smoke_fn)
    has_items = 'wi_valid' in s
    if V.fused_ok(s, mesh) and (smoke_fn is None or G is not None):
        pts = (s['wi_pos'] if has_items else None, s['wi_valid'] if has_items else None, s['bomb_pos'], s['bomb_state'])
        return V.trace(s, mesh, tick, G, points=pts, vis=False)[1]
    E = s['x'].shape[0]
    bvalid = (s['bomb_state'].long() >= 1)[:, None]
    bv = point_visible(s, mesh, s['bomb_pos'][:, None, :], bvalid, smoke_fn, tick)
    if not has_items:
        return bv
    iv = point_visible(s, mesh, s['wi_pos'], s['wi_valid'].bool(), smoke_fn, tick)
    return torch.cat([iv, bv], -1)


def build_obs_torch(s, vis_parts, st, mesh=None, audio=None, smoke_fn=None, out=None, ctx=None):
    """original vectorized torch implementation of build_obs (reference of the fused kernel)."""
    dev = s['x'].device
    E = s['x'].shape[0]
    ctx = ctx or {}
    ar10 = torch.arange(10, device=dev)
    team = ar10 // 5
    sit = ar10 % 5
    tick = s['tick'].long()
    tl = tick[:, None]
    alive = s['alive']
    x, y, z = s['x'], s['y'], s['z']
    yaw, pitch = s['yaw'], s['pitch']
    P3 = torch.stack([x, y, z], -1)                                                       # [E,10,3] f64
    cen = st.center
    nades = _g(s, 'nades', torch.zeros(E, 10, NU, dtype=torch.int32, device=dev)).long()
    walk = _g(s, 'walk_held', _g(s, 'act_walk', torch.zeros_like(alive))).bool()
    active = s['active'].long()
    wid = s['w_wid'].long()                                                               # [E,10,3]
    # ---------------------------------------------------------------- public per-player block (what enemies reveal)
    act_wid = torch.gather(wid, 2, active[..., None]).squeeze(-1)                         # [E,10]
    mm = s['movement_mode'].long()
    pub = torch.cat([((P3 - cen) / POS).float(), _f(torch.sin(yaw)), _f(torch.cos(yaw)), _f(pitch / (math.pi / 2)),
                     _f(s['crouch']), _oh(act_wid, W, act_wid >= 0), _oh(mm, 4)], -1)    # [E,10,3+2+1+1+W+4]
    # ---------------------------------------------------------------- round block
    team_e = team[None, :].expand(E, -1)
    at = s['attack_team'].long()[:, None]
    side_att = team_e == at
    rw = s['round_wins'].long()
    own = team_e
    rw_own = torch.gather(rw[:, None, :].expand(-1, 10, -1), 2, own[..., None]).squeeze(-1)
    rw_en = torch.gather(rw[:, None, :].expand(-1, 10, -1), 2, (1 - own)[..., None]).squeeze(-1)
    al_t = torch.stack([alive[:, :5].sum(1), alive[:, 5:].sum(1)], 1)                      # [E,2]
    al_own = torch.gather(al_t[:, None, :].expand(-1, 10, -1), 2, own[..., None]).squeeze(-1)
    al_en = torch.gather(al_t[:, None, :].expand(-1, 10, -1), 2, (1 - own)[..., None]).squeeze(-1)
    ll = s['loss_level'].long()
    ll_own = torch.gather(ll[:, None, :].expand(-1, 10, -1), 2, own[..., None]).squeeze(-1)
    phase = s['phase'].long()
    rs_t = s['round_start_tick'].long()
    fuse_end = s['bomb_fuse_end'].long()
    bstate = s['bomb_state'].long()
    live_end = torch.where(fuse_end > 0, fuse_end, s['live_end'].long())
    pend = torch.where(phase == 0, rs_t + 360, torch.where(phase == 1, live_end, torch.where(phase == 2, s['post_end'].long(), tick)))
    prem = (pend - tick).clamp(min=0)
    ot = _g(s, 'overtime_number', torch.full((E,), int(ctx.get('overtime_number', 0)), dtype=torch.long, device=dev)).long()
    wt = _g(s, 'win_target', torch.full((E,), int(ctx.get('win_target', 13)), dtype=torch.long, device=dev)).long()
    b2 = lambda v: v[:, None].expand(-1, 10)
    rd = _cat(_ROUND, [
        _oh(torch.full((E, 10), max(st.map_idx, 0), device=dev), 5, torch.full((E, 10), st.map_idx >= 0, device=dev)),
        _f(b2(tick - _g(s, 't0', torch.zeros_like(tick)).long()) / TICK), _f(b2(s['round_index'].long().clamp(min=0)) / 24.0), _f(b2(tick - rs_t) / RTICK), _oh(b2(phase), 4),
        _f(b2(prem) / RTICK), _f(team_e), _oh(sit[None, :].expand(E, -1), 5), _f(side_att),
        torch.stack([rw_own / 13.0, rw_en / 13.0], -1).float(), torch.stack([al_own / 5.0, al_en / 5.0], -1).float(),
        _f(ll_own / 5.0), _f(b2(ot > 0)), _f(b2(ot) / 10.0), _f(b2(wt) / 13.0), _f(b2(rs_t + 1860 - tick) / 1860.0)])
    # ---------------------------------------------------------------- me block
    vf, vr, _ = _view_frame(s['vx'], s['vy'], s['vz'], yaw)
    rem = lambda a: ((a.long() - tl).clamp(min=0).float() / SEC)
    zero10 = torch.zeros(E, 10, dtype=torch.long, device=dev)
    ladder = s['ladder_id'].long()
    me = _cat(_ME, [
        ((P3 - cen) / POS).float(), torch.stack([torch.sin(yaw), torch.cos(yaw)], -1).float(), _f(pitch / (math.pi / 2)),
        torch.stack([s['vx'], s['vy'], s['vz']], -1).float() / VEL, torch.stack([vf, vr], -1).float() / VEL,
        _f(s['hp'].clamp(0, 100) / 100.0), _f(s['armor'].clamp(0, 100) / 100.0), _f(s['helmet']), _f(s['kit']),
        _f(s['cash'].double().clamp(0, CASH) / CASH), _f(alive), _f(s['crouch']), _f(s['grounded']), _f(walk),
        _f(_g(s, 'stand_blocked', _g(s, 'mv_stand_blocked', torch.zeros_like(alive)))), _oh(active, 3), _oh(s['zoom'].long(), 3), _oh(mm, 4),
        _f((ladder + 1).clamp(min=0) / 64.0), _f(rem(_g(s, 'reattach_until', _g(s, 'mv_reattach_until', zero10)))),
        _f(rem(_g(s, 'pickup_ready_tick', zero10))), _f(rem(s['switch_until'])),
        _f(rem(_g(s, 'throw_ready', zero10))), _f((_g(s, 'blind_until', zero10).long() - tl).clamp(min=0) / BLIND),
        nades.float() / 2.0, _f(s['has_bomb'])])
    # weapons [E,10,3,*]
    t3 = tick[:, None, None]
    sh = lambda k: s[k].long()
    present = wid >= 0
    rs = s['w_rs']
    sr = (sh('w_shot_ready') - t3).clamp(min=0).float() / SEC
    ru = sh('w_reload_until')
    reloading = ru >= 0
    rc = sh('w_reload_commit')
    shell = s['w_reload_shell']
    ls = sh('w_last_shot')
    has_last = ls >= 0
    since = (t3 - ls).clamp(min=0, max=SINCE_CLIP).float() / SINCE
    rl = lambda v, m: torch.where(m, (v - t3).clamp(min=0).float() / SEC, torch.zeros(1, device=dev))
    wp = _cat(_WEAP, [
        _f(present), _oh(wid, W, present), _f(sh('w_ammo') / AMMO), _f(sh('w_reserve') / RESERVE), _f(sr),
        _f(reloading), _f(rl(ru, reloading)), _f(rl(torch.where(shell, ru, rc), reloading & (shell | (rc >= 0)))),
        _f(s['w_reload_committed'] & reloading & ~shell), _f(shell & reloading), _f(sh('w_shots') / SHOTS), _f(has_last),
        _f(torch.where(has_last, since, torch.zeros(1, device=dev))), _f(rs[..., 0] / 64.0),
        (rs[..., 1:3] * ANG).float(), rs[..., 3:5].float(), (rs[..., 5:7] * ANG).float(), _f(s['w_inacc'] * INACC),
        _f(s['w_pressed'])])                                                                # [E,10,3,Fw]
    wp = wp * present[..., None].float()                                                # empty slots read as zeros
    # ---------------------------------------------------------------- visible enemies
    enemy = (team[:, None] != team[None, :])[None]
    anyp = vis_parts.any(-1)
    vis = anyp & enemy & alive[:, :, None] & alive[:, None, :]
    d3 = P3[:, None, :, :] - P3[:, :, None, :]                                            # [E,obs,tgt,3]
    dist = d3.norm(dim=-1)
    key = torch.where(vis, dist, torch.full_like(dist, float('inf')))
    idx = torch.sort(key, dim=-1, stable=True).indices[..., :NEN]                         # [E,10,5]
    ev = torch.gather(vis, 2, idx)
    ee = torch.arange(E, device=dev)[:, None, None]
    pg = pub[ee, idx]                                                                     # [E,10,5,Fp]
    d3g = torch.gather(d3, 2, idx[..., None].expand(-1, -1, -1, 3))                       # [E,10,5,3]
    relv = _rel(d3g, yaw[:, :, None])
    distg = torch.gather(dist, 2, idx)
    prt = torch.gather(vis_parts, 2, idx[..., None].expand(-1, -1, -1, 3)).float()
    evf = ev.float()[..., None]
                                                                                          # pub layout: pos3 yaw2 pitch1 crouch1 weapon(W) mode4
    en = _cat(_ENEMY, [evf, _oh(idx % 5, 5, ev), pg[..., 0:3] * evf, (relv / REL).float() * evf,
                       (distg / DIST).float()[..., None] * evf, pg[..., 3:5] * evf, pg[..., 5:6] * evf, pg[..., 6:7] * evf,
                       pg[..., 7:7 + W] * evf, pg[..., 7 + W:7 + W + 4] * evf, prt * evf]).reshape(E, 10, -1)
    # ---------------------------------------------------------------- teammates
    # zone of every player
    if len(st.zone_pos):
        dz_ = ((P3[:, :, None, :] - st.zone_pos[None, None]) ** 2).sum(-1)
        zone = dz_.argmin(-1)
        zoh = _oh(zone, NZ)
    else:
        zoh = torch.zeros(E, 10, NZ, device=dev)
    wmh = (F.one_hot(wid.clamp(min=0), W).float() * (present & (wid != KNIFE))[..., None].float()).sum(2)      # [E,10,W] multi-hot
    al = alive[..., None].float()
    tpl = torch.cat([_f(alive), _f(s['cash'].double().clamp(0, CASH) / CASH), zoh, wmh * al, nades.float() / 2.0 * al,
                     _f((s['armor'] > 0) & alive), _f(s['helmet'] & alive), _f(s['kit'] & alive)], -1)   # [E,10,*]
    own_idx = (team[:, None] * 5 + torch.arange(5, device=dev)[None, :])                  # [10,5] seats of own team
    tm_idx = torch.stack([own_idx[i][own_idx[i] != i] for i in range(10)])                # [10,4]
    tg = tpl[:, tm_idx]                                                                   # [E,10,4,F]
    tm_seat = _oh(tm_idx % 5, 5)[None].expand(E, -1, -1, -1)
    tm = torch.cat([tm_seat, tg], -1).reshape(E, 10, -1)
    # ---------------------------------------------------------------- bomb
    bpos = s['bomb_pos']
    carrier = s['bomb_carrier'].long()
    ctr = carrier.clamp(min=0)
    cvis = torch.gather(vis, 2, ctr[:, None, None].expand(-1, 10, 1)).squeeze(-1) & (carrier >= 0)[:, None]   # enemy carrier seen
    bvalid = (bstate >= 1)[:, None].expand(-1, 10)
    bvis = point_visible(s, mesh, bpos[:, None, :], bvalid[:, :1], smoke_fn, tick)[:, :, 0]
    known = side_att | torch.where((bstate == 0)[:, None], cvis, bvis & bvalid)
    hidden_state = torch.where(bstate >= 2, bstate, torch.full_like(bstate, 5))[:, None].expand(-1, 10)
    shown = torch.where(known, bstate[:, None].expand(-1, 10), hidden_state)
    fuse_valid = (fuse_end > 0)[:, None] & (known | (hidden_state != 5))
    carried = known & (carrier >= 0)[:, None]
    dbm = bpos[:, None, :] - P3
    rb = _rel(dbm, yaw)
    site = _g(s, 'bomb_site', None)
    if site is None:
        if len(st.site_pos):
            site = ((bpos[:, None, :2] - st.site_pos[None, :, :2]) ** 2).sum(-1).argmin(-1)
        else:
            site = torch.full((E,), -1, dtype=torch.long, device=dev)
        site = torch.where(bstate >= 2, site, torch.full_like(site, -1))
    site = site.long()
    site_oh = _oh(site[:, None].expand(-1, 10), NZ, known & (site >= 0)[:, None])
    obj = s['objective'].long()
    prog = s['obj_progress'].double()
    isp = (obj == 1) & (prog > 0)
    isd = obj == 2
    kf = known.float()[..., None]
    bomb = _cat(_BOMB, [
        _oh(shown, 6), kf, _f(fuse_valid), _f(fuse_valid.float() * (fuse_end[:, None] - tl).float() / FUSE),
        _oh(carrier[:, None].expand(-1, 10), 10, carried), _f(carried & (carrier[:, None] == ar10[None, :])),
        kf, ((bpos[:, None, :] - cen) / POS).float() * kf, (rb / REL).float() * kf,
        (dbm.norm(dim=-1) / DIST).float()[..., None] * kf, site_oh,
        _f(isp), _f(torch.where(isp, prog, torch.zeros_like(prog)) / PROG), _f(isd), _f(torch.where(isd, prog, torch.zeros_like(prog)) / PROG),
        _f(torch.where(isd, _g(s, 'bomb_duration', torch.zeros(E, dtype=torch.long, device=dev)).double()[:, None].expand(-1, 10), torch.zeros_like(prog)) / PROG)])
    # ---------------------------------------------------------------- world items
    if 'wi_valid' in s:
        wv, wk, wpos = s['wi_valid'].bool(), s['wi_kind'].long(), s['wi_pos']
        if wv.shape[1] < NITEM:
            pad = NITEM - wv.shape[1]
            wv = F.pad(wv, (0, pad)); wk = F.pad(wk, (0, pad)); wpos = F.pad(wpos, (0, 0, 0, pad))
        K = wv.shape[1]
        ivis = point_visible(s, mesh, wpos, wv, smoke_fn, tick) & alive[:, :, None]       # [E,10,K]
        dd = wpos[:, None, :, :] - P3[:, :, None, :]
        di = dd.norm(dim=-1)
        ikey = torch.where(ivis, di, torch.full_like(di, float('inf')))
        iord = torch.sort(ikey, dim=-1, stable=True).indices[..., :NITEM]
        iv = torch.gather(ivis, 2, iord)
        ivf = iv.float()[..., None]
        ddg = torch.gather(dd, 2, iord[..., None].expand(-1, -1, -1, 3))
        kg = wk[ee, iord]
        pgi = wpos[ee, iord]
        wi = torch.cat([_cat(_ITEM, [ivf, _oh(kg, W, iv), ((pgi - cen) / POS).float() * ivf,
                                     (_rel(ddg, yaw[:, :, None]) / REL).float() * ivf,
                                     (ddg.norm(dim=-1) / DIST).float()[..., None] * ivf]).reshape(E, 10, -1),
                        _f(ivis.sum(-1) / 8.0)], -1)
    else:
        wi = torch.zeros(E, 10, NITEM * _spec_len(_ITEM) + 1, device=dev)
    # ---------------------------------------------------------------- sounds
    if audio is not None:
        knd, dcode, bcode, cnt = hear(s, audio)
        sv = knd >= 0
        sf = sv.float()[..., None]
        snd = torch.cat([_cat(_SOUND, [sf, _oh(knd, NK, sv), _oh(dcode, 5, sv), _oh(bcode, 5, sv)]).reshape(E, 10, -1),
                         _f(cnt / 8.0)], -1)
    else:
        snd = torch.zeros(E, 10, NSND * _spec_len(_SOUND) + 1, device=dev)
    # ---------------------------------------------------------------- assemble
    o = torch.cat([rd, me, wp.reshape(E, 10, -1), en, tm, bomb, wi, snd], -1) if out is None else None
    if o is None:
        parts = [rd, me, wp.reshape(E, 10, -1), en, tm, bomb, wi, snd]
        a = 0
        for p in parts:
            out[..., a:a + p.shape[-1]] = p
            a += p.shape[-1]
        o = out
    assert o.shape[-1] == OBS_DIM, (o.shape, OBS_DIM)
    return o


# ---------------------------------------------------------------------------------------------- fused CUDA builder
# build_obs on CUDA runs as one kernel (`_obs_ext().build`): 8 rows (match, seat) per 128-thread block, 16 lanes per row, every lane
# writes one part of the row into shared memory (lane 0 round + me, 1-3 weapon slots, 4-8 enemies, 9-12 teammates, 13 bomb, 14 world
# items, 15 sounds), then the block stores its 8 contiguous rows coalesced.  The arithmetic follows the torch reference
# (`build_obs_torch`) operation by operation: float64 maths, float32 rounding at the same places, division by a python scalar done as
# multiplication by its reciprocal (what torch does on CUDA), compiled with -fmad=false.  Point visibility (bomb / world items) comes from
# gpusim.vis.trace (fused) or, for a foreign smoke_fn, from point_visible.  GPUSIM_OBS_TORCH=1 forces the torch reference.
def _hexf(v):
    return float.hex(float(np.float32(v))) + 'f'


def _obs_src():
    L = []
    d = lambda n, v: L.append('#define %s %s' % (n, v))
    for pre, spec in (('R_', _ROUND), ('M_', _ME), ('W_', _WEAP), ('EN_', _ENEMY), ('TM_', _TEAM), ('B_', _BOMB), ('I_', _ITEM), ('S_', _SOUND)):
        a = 0
        for name, n, _ in spec:
            d(pre + name.upper(), a); a += n
        d(pre + 'LEN', a)
    o = lambda name: OFF[name][0]
    d('G_RD', o('rd.map')); d('G_ME', o('me.pos')); d('G_W0', o('me.w0.present')); d('G_EN0', o('en0.valid')); d('G_TM0', o('tm0.seat'))
    d('G_BOMB', o('bomb.state')); d('G_WI0', o('wi0.valid')); d('G_WICOUNT', o('wi.count')); d('G_SND0', o('snd0.valid'))
    d('G_SNDCOUNT', o('snd.count')); d('OBS_DIM', OBS_DIM); d('NW', W); d('NZ', NZ); d('NEN', NEN); d('NITEM', NITEM); d('NSND', NSND)
    d('KNIFE', KNIFE); d('NUTIL', NU)
    for n, v in (('TICK', TICK), ('24', 24.0), ('RTICK', RTICK), ('13', 13.0), ('5', 5.0), ('10', 10.0), ('1860', 1860.0), ('64', 64.0),
                 ('SEC', SEC), ('BLIND', BLIND), ('AMMO', AMMO), ('RES', RESERVE), ('SHOTS', SHOTS), ('SINCE', SINCE), ('FUSE', FUSE),
                 ('8', 8.0), ('VEL', VEL), ('2', 2.0)):
        d('IF_' + n, _hexf(np.float32(1.0) / np.float32(v)))            # torch: float32 tensor / python scalar = x * (1.0f / s)
    for n, v in (('POS', POS), ('PI2', math.pi / 2), ('100', 100.0), ('CASH', CASH), ('64', 64.0), ('REL', REL), ('DIST', DIST), ('PROG', PROG)):
        d('ID_' + n, repr(1.0 / v))                                      # float64 tensor / python scalar = x * (1.0 / s)
    d('ANG', repr(ANG)); d('INACC', repr(INACC)); d('SINCE_CLIP', SINCE_CLIP); d('NEARD', repr(NEAR)); d('MIDD', repr(MID))
    d('CPAN', repr(CENTER_PAN))
    tabs = '\n'.join('__constant__ int %s[14] = {%s};' % (n, ','.join(str(int(x)) for x in v)) for n, v in
                     (('IT_KIND', _IT_KIND), ('IT_DIRECT', _IT_DIRECT), ('IT_HASPOS', _IT_HASPOS), ('IT_OTHERS', _IT_OTHERS), ('IT_SELF', _IT_SELF)))
    tabs += '\n__constant__ double IT_RAD[14] = {%s};' % ','.join(repr(float(np.float32(x))) for x in _IT_RADIUS)
    return '\n'.join(L) + '\n' + tabs + '\n' + _OBS_BODY


_OBS_BODY = r"""
struct TF { const void* p; int t; };
__device__ __forceinline__ long long gi(const TF& f, long long i){
  switch(f.t){ case 0: return (long long)((const double*)f.p)[i]; case 1: return (long long)((const float*)f.p)[i];
    case 2: return ((const long long*)f.p)[i]; case 3: return ((const int*)f.p)[i]; case 4: return ((const short*)f.p)[i];
    case 5: return ((const signed char*)f.p)[i]; default: return ((const unsigned char*)f.p)[i]; } }
__device__ __forceinline__ long long gio(const TF& f, long long i, long long dflt){ return f.p ? gi(f,i) : dflt; }
__device__ __forceinline__ int ci(long long v, int lo, int hi){ return v<lo? lo : (v>hi? hi : (int)v); }
__device__ __forceinline__ double cd(double v, double lo, double hi){ return v<lo? lo : (v>hi? hi : v); }
__device__ __forceinline__ long long lmax0(long long v){ return v<0? 0 : v; }
// players [E,10]: float64 x y z vx vy vz yaw pitch hp armor; typed: the rest.  weapons [E,10,3] (+7 for w_rs).
struct OS {
  const double *x,*y,*z,*vx,*vy,*vz,*yaw,*pitch,*hp,*armor,*inacc,*rs,*bpos,*wpos,*empos,*cen,*zpos;
  TF alive,crouch,grounded,helmet,kit,has_bomb,cash,mm,ladder,reattach,walk,standb,pickup,switch_until,throw_ready,blind_until,nades,
     objective,obj_progress,active,zoom;
  TF wid,ammo,reserve,shots,shot_ready,reload_until,reload_commit,last_shot,committed,shell,pressed;
  TF tick,t0,round_index,round_start,live_end,post_end,phase,attack_team,round_wins,loss_level,bstate,bcarrier,fuse_end,bduration,bsite,ot,wt;
  TF wvalid,wkind,vp,pv,emn,emit,emsrc;
  int K, cap, Z, map_idx, has_items, has_audio; long long ot_def, wt_def;
};
__device__ __forceinline__ double norm3(double a, double b, double c){ return sqrt(NORM3(a,b,c)); }
__device__ __forceinline__ bool vpart(const OS& S, long long pi, int g, int k){ return gi(S.vp,(pi*10+g)*3+k)!=0; }

__device__ void lane_rd_me(const OS& S, int e, int i, float* R){
  long long pi=(long long)e*10+i;
  long long tick=gi(S.tick,e);
  int team=i/5, sit=i%5;
  float* r=R+G_RD;
  if(S.map_idx>=0) r[R_MAP+ci(S.map_idx,0,4)]=1.f;
  r[R_MATCH_TICK]=(float)(tick-gio(S.t0,e,0))*IF_TICK;
  long long ri=gi(S.round_index,e); if(ri<0) ri=0;
  r[R_ROUND_INDEX]=(float)ri*IF_24;
  long long rst=gi(S.round_start,e);
  r[R_ROUND_TICK]=(float)(tick-rst)*IF_RTICK;
  long long ph=gi(S.phase,e);
  r[R_PHASE+ci(ph,0,3)]=1.f;
  long long fe=gi(S.fuse_end,e);
  long long le= fe>0? fe : gi(S.live_end,e);
  long long pend= ph==0? rst+360 : (ph==1? le : (ph==2? gi(S.post_end,e) : tick));
  r[R_PHASE_REMAINING]=(float)lmax0(pend-tick)*IF_RTICK;
  r[R_TEAM]=(float)team;
  r[R_SEAT+sit]=1.f;
  bool side_att= (long long)team==gi(S.attack_team,e);
  r[R_SIDE_ATTACK]=side_att?1.f:0.f;
  r[R_ROUND_WINS]=(float)gi(S.round_wins,(long long)e*2+team)*IF_13;
  r[R_ROUND_WINS+1]=(float)gi(S.round_wins,(long long)e*2+1-team)*IF_13;
  long long a0=0,a1=0; for(int q=0;q<5;q++){ a0+=gi(S.alive,(long long)e*10+q)!=0; a1+=gi(S.alive,(long long)e*10+5+q)!=0; }
  r[R_ALIVE]=(float)(team==0?a0:a1)*IF_5; r[R_ALIVE+1]=(float)(team==0?a1:a0)*IF_5;
  r[R_LOSS_LEVEL]=(float)gi(S.loss_level,(long long)e*2+team)*IF_5;
  long long ot=gio(S.ot,e,S.ot_def), wt=gio(S.wt,e,S.wt_def);
  r[R_OVERTIME]=ot>0?1.f:0.f; r[R_OVERTIME_NUMBER]=(float)ot*IF_10; r[R_WIN_TARGET]=(float)wt*IF_13;
  r[R_BUY_REMAINING]=(float)(rst+1860-tick)*IF_1860;
  // ---- me
  float* m=R+G_ME;
  double x=S.x[pi], y=S.y[pi], z=S.z[pi], yaw=S.yaw[pi];
  m[M_POS]=(float)((x-S.cen[0])*ID_POS); m[M_POS+1]=(float)((y-S.cen[1])*ID_POS); m[M_POS+2]=(float)((z-S.cen[2])*ID_POS);
  double sy=sin(yaw), cy=cos(yaw);
  m[M_YAW]=(float)sy; m[M_YAW+1]=(float)cy;
  m[M_PITCH]=(float)(S.pitch[pi]*ID_PI2);
  double vx=S.vx[pi], vy=S.vy[pi], vz=S.vz[pi];
  m[M_VEL]=(float)vx*IF_VEL; m[M_VEL+1]=(float)vy*IF_VEL; m[M_VEL+2]=(float)vz*IF_VEL;
  double vf=vx*cy-vy*sy, vr=(-vx)*sy-vy*cy;
  m[M_VEL_VIEW]=(float)vf*IF_VEL; m[M_VEL_VIEW+1]=(float)vr*IF_VEL;
  m[M_HP]=(float)(cd(S.hp[pi],0.0,100.0)*ID_100);
  m[M_ARMOR]=(float)(cd(S.armor[pi],0.0,100.0)*ID_100);
  m[M_HELMET]=gi(S.helmet,pi)!=0; m[M_HAS_KIT]=gi(S.kit,pi)!=0;
  m[M_CASH]=(float)(cd((double)gi(S.cash,pi),0.0,16000.0)*ID_CASH);
  m[M_ALIVE]=gi(S.alive,pi)!=0; m[M_CROUCH]=gi(S.crouch,pi)!=0; m[M_GROUNDED]=gi(S.grounded,pi)!=0;
  m[M_WALK]=gio(S.walk,pi,0)!=0; m[M_STAND_BLOCKED]=gio(S.standb,pi,0)!=0;
  m[M_ACTIVE+ci(gi(S.active,pi),0,2)]=1.f; m[M_ZOOM+ci(gi(S.zoom,pi),0,2)]=1.f; m[M_MOVEMENT_MODE+ci(gi(S.mm,pi),0,3)]=1.f;
  m[M_LADDER_ID]=(float)lmax0(gi(S.ladder,pi)+1)*IF_64;
  m[M_REATTACH_REM]=(float)lmax0(gio(S.reattach,pi,0)-tick)*IF_SEC;
  m[M_PICKUP_REM]=(float)lmax0(gio(S.pickup,pi,0)-tick)*IF_SEC;
  m[M_SWITCH_REM]=(float)lmax0(gi(S.switch_until,pi)-tick)*IF_SEC;
  m[M_THROW_REM]=(float)lmax0(gio(S.throw_ready,pi,0)-tick)*IF_SEC;
  m[M_BLIND_REM]=(float)lmax0(gio(S.blind_until,pi,0)-tick)*IF_BLIND;
  for(int u=0;u<NUTIL;u++) m[M_THROWABLES+u]=(float)gio(S.nades,pi*NUTIL+u,0)*IF_2;
  m[M_CARRYING_BOMB]=gi(S.has_bomb,pi)!=0;
}
__device__ void lane_weapon(const OS& S, int e, int i, int w, float* R){
  long long pi=(long long)e*10+i, wi=pi*3+w;
  long long wid=gi(S.wid,wi); if(wid<0) return;
  long long tick=gi(S.tick,e);
  float* q=R+G_W0+w*W_LEN;
  q[W_PRESENT]=1.f; q[W_ID+ci(wid,0,NW-1)]=1.f;
  q[W_AMMO]=(float)gi(S.ammo,wi)*IF_AMMO; q[W_RESERVE]=(float)gi(S.reserve,wi)*IF_RES;
  q[W_SHOT_READY]=(float)lmax0(gi(S.shot_ready,wi)-tick)*IF_SEC;
  long long ru=gi(S.reload_until,wi), rc=gi(S.reload_commit,wi);
  bool rl=ru>=0, sh=gi(S.shell,wi)!=0;
  q[W_RELOADING]=rl?1.f:0.f;
  q[W_RELOAD_REM]= rl? (float)lmax0(ru-tick)*IF_SEC : 0.f;
  long long cv= sh? ru : rc;
  q[W_COMMIT_REM]= (rl && (sh || rc>=0))? (float)lmax0(cv-tick)*IF_SEC : 0.f;
  q[W_COMMITTED]= (gi(S.committed,wi)!=0 && rl && !sh)?1.f:0.f;
  q[W_SHELL]= (sh && rl)?1.f:0.f;
  q[W_SHOTS]=(float)gi(S.shots,wi)*IF_SHOTS;
  long long ls=gi(S.last_shot,wi); bool hl=ls>=0;
  q[W_HAS_LAST_SHOT]=hl?1.f:0.f;
  long long sn=tick-ls; sn= sn<0? 0 : (sn>SINCE_CLIP? SINCE_CLIP : sn);
  q[W_SINCE_SHOT]= hl? (float)sn*IF_SINCE : 0.f;
  const double* rs=S.rs+wi*7;
  q[W_PUNCH_INDEX]=(float)(rs[0]*ID_64);
  q[W_PUNCH]=(float)(rs[1]*ANG); q[W_PUNCH+1]=(float)(rs[2]*ANG);
  q[W_PUNCH_VEL]=(float)rs[3]; q[W_PUNCH_VEL+1]=(float)rs[4];
  q[W_VIEW_PUNCH]=(float)(rs[5]*ANG); q[W_VIEW_PUNCH+1]=(float)(rs[6]*ANG);
  q[W_INACCURACY]=(float)(S.inacc[wi]*INACC);
  q[W_PRESSED]=gi(S.pressed,wi)!=0;
}
__device__ __forceinline__ void put_rel(float* q, double dx, double dy, double dz, double c, double sn){
  double f=dx*c-dy*sn, r=(-dx)*sn-dy*c;
  q[0]=(float)(f*ID_REL); q[1]=(float)(r*ID_REL); q[2]=(float)(dz*ID_REL);
}
__device__ __forceinline__ void put_pos(const OS& S, float* q, double x, double y, double z){
  q[0]=(float)((x-S.cen[0])*ID_POS); q[1]=(float)((y-S.cen[1])*ID_POS); q[2]=(float)((z-S.cen[2])*ID_POS);
}
__device__ void lane_enemy(const OS& S, int e, int i, int j, float* R){
  long long pi=(long long)e*10+i;
  if(!gi(S.alive,pi)) return;
  int et=1-i/5;
  bool v[5]; double dd[5];
  double px=S.x[pi], py=S.y[pi], pz=S.z[pi];
  for(int jj=0;jj<5;jj++){
    int tg=et*5+jj; long long pt=(long long)e*10+tg;
    v[jj]=(vpart(S,pi,tg,0)||vpart(S,pi,tg,1)||vpart(S,pi,tg,2)) && gi(S.alive,pt)!=0;
    dd[jj]=norm3(S.x[pt]-px,S.y[pt]-py,S.z[pt]-pz);
  }
  if(!v[j]) return;
  int rank=0; for(int jj=0;jj<5;jj++) if(v[jj] && (dd[jj]<dd[j] || (dd[jj]==dd[j] && jj<j))) rank++;
  int tg=et*5+j; long long pt=(long long)e*10+tg;
  float* q=R+G_EN0+rank*EN_LEN;
  q[EN_VALID]=1.f; q[EN_SEAT+j]=1.f;
  put_pos(S,q+EN_POS,S.x[pt],S.y[pt],S.z[pt]);
  double yaw=S.yaw[pi];
  put_rel(q+EN_REL,S.x[pt]-px,S.y[pt]-py,S.z[pt]-pz,cos(yaw),sin(yaw));
  q[EN_DIST]=(float)(dd[j]*ID_DIST);
  double ty=S.yaw[pt];
  q[EN_YAW]=(float)sin(ty); q[EN_YAW+1]=(float)cos(ty);
  q[EN_PITCH]=(float)(S.pitch[pt]*ID_PI2);
  q[EN_CROUCH]=gi(S.crouch,pt)!=0;
  long long aw=gi(S.wid,pt*3+ci(gi(S.active,pt),0,2));
  if(aw>=0) q[EN_WEAPON+ci(aw,0,NW-1)]=1.f;
  q[EN_MOVEMENT_MODE+ci(gi(S.mm,pt),0,3)]=1.f;
  for(int k=0;k<3;k++) q[EN_PARTS+k]=vpart(S,pi,tg,k)?1.f:0.f;
}
__device__ void lane_team(const OS& S, int e, int i, int k, float* R){
  int team=i/5, sit=i%5;
  int tm=team*5+(k<sit? k : k+1);
  long long pt=(long long)e*10+tm;
  float* q=R+G_TM0+k*TM_LEN;
  q[TM_SEAT+tm%5]=1.f;
  bool al=gi(S.alive,pt)!=0;
  q[TM_ALIVE]=al?1.f:0.f;
  q[TM_CASH]=(float)(cd((double)gi(S.cash,pt),0.0,16000.0)*ID_CASH);
  if(S.Z>0){
    int best=0; double bd=0;
    for(int zz=0;zz<S.Z;zz++){
      double dx=S.x[pt]-S.zpos[3*zz], dy=S.y[pt]-S.zpos[3*zz+1], dz=S.z[pt]-S.zpos[3*zz+2];
      double d2=NORM3(dx,dy,dz);                 // torch .sum(-1) of the squares, same order as the norm
      if(zz==0 || d2<bd){ bd=d2; best=zz; }
    }
    q[TM_ZONE+ci(best,0,NZ-1)]=1.f;
  }
  if(!al) return;
  for(int w=0;w<3;w++){ long long wid=gi(S.wid,pt*3+w); if(wid>=0 && wid!=KNIFE) q[TM_WEAPONS+ci(wid,0,NW-1)]+=1.f; }
  for(int u=0;u<NUTIL;u++) q[TM_THROWABLES+u]=(float)gio(S.nades,pt*NUTIL+u,0)*IF_2;
  q[TM_ARMOR]=S.armor[pt]>0?1.f:0.f; q[TM_HELMET]=gi(S.helmet,pt)!=0; q[TM_KIT]=gi(S.kit,pt)!=0;
}
__device__ void lane_bomb(const OS& S, int e, int i, float* R){
  long long pi=(long long)e*10+i;
  long long tick=gi(S.tick,e);
  long long bst=gi(S.bstate,e), car=gi(S.bcarrier,e), fe=gi(S.fuse_end,e);
  int team=i/5;
  bool side_att=(long long)team==gi(S.attack_team,e);
  long long ctr= car<0? 0 : car;
  bool cvis=false;
  if(car>=0 && ctr<10 && (ctr/5)!=team){
    long long pc=(long long)e*10+ctr;
    cvis=(vpart(S,pi,(int)ctr,0)||vpart(S,pi,(int)ctr,1)||vpart(S,pi,(int)ctr,2)) && gi(S.alive,pi)!=0 && gi(S.alive,pc)!=0;
  }
  bool bvalid=bst>=1;
  bool bvis= S.pv.p ? gi(S.pv,pi*(S.K+1)+S.K)!=0 : false;
  bool known= side_att || (bst==0? cvis : (bvis && bvalid));
  long long hid= bst>=2? bst : 5;
  long long shown= known? bst : hid;
  bool fv= fe>0 && (known || hid!=5);
  bool carried= known && car>=0;
  float* q=R+G_BOMB;
  q[B_STATE+ci(shown,0,5)]=1.f;
  q[B_KNOWN]=known?1.f:0.f; q[B_FUSE_VALID]=fv?1.f:0.f;
  q[B_FUSE_REM]=((fv?1.f:0.f)*(float)(fe-tick))*IF_FUSE;
  if(carried) q[B_CARRIER+ci(car,0,9)]=1.f;
  q[B_CARRIER_ME]=(carried && car==i)?1.f:0.f;
  const double* bp=S.bpos+(long long)e*3;
  if(known){
    q[B_HAS_POS]=1.f;
    put_pos(S,q+B_POS,bp[0],bp[1],bp[2]);
    double dx=bp[0]-S.x[pi], dy=bp[1]-S.y[pi], dz=bp[2]-S.z[pi], yaw=S.yaw[pi];
    put_rel(q+B_REL,dx,dy,dz,cos(yaw),sin(yaw));
    q[B_DIST]=(float)(norm3(dx,dy,dz)*ID_DIST);
    long long site=gi(S.bsite,e);
    if(site>=0) q[B_SITE+ci(site,0,NZ-1)]=1.f;
  }
  long long ob=gi(S.objective,pi); double prog=(double)gi(S.obj_progress,pi);
  bool isp= ob==1 && prog>0, isd= ob==2;
  q[B_IS_PLANTER]=isp?1.f:0.f; q[B_PLANT_PROGRESS]=(float)((isp?prog:0.0)*ID_PROG);
  q[B_IS_DEFUSER]=isd?1.f:0.f; q[B_DEFUSE_PROGRESS]=(float)((isd?prog:0.0)*ID_PROG);
  q[B_DEFUSE_DURATION]=(float)((isd?(double)gio(S.bduration,e,0):0.0)*ID_PROG);
}
__device__ void lane_items(const OS& S, int e, int i, float* R){
  if(!S.has_items) return;
  long long pi=(long long)e*10+i;
  if(!gi(S.alive,pi) || !S.pv.p) return;
  int K=S.K; double px=S.x[pi], py=S.y[pi], pz=S.z[pi];
  int bk[NITEM]; double bdv[NITEM]; int nb=0, cnt=0;
  for(int k=0;k<K;k++){
    if(!gi(S.pv,pi*(K+1)+k)) continue;
    cnt++;
    const double* w=S.wpos+((long long)e*K+k)*3;
    double d=norm3(w[0]-px,w[1]-py,w[2]-pz);
    if(nb==NITEM && !(d<bdv[NITEM-1])) continue;          // ties keep the lower index (stable sort)
    int p= nb<NITEM? nb++ : NITEM-1;
    while(p>0 && d<bdv[p-1]){ bdv[p]=bdv[p-1]; bk[p]=bk[p-1]; p--; }
    bdv[p]=d; bk[p]=k;
  }
  double yaw=S.yaw[pi], c=cos(yaw), sn=sin(yaw);
  for(int r=0;r<nb;r++){
    int k=bk[r]; const double* w=S.wpos+((long long)e*K+k)*3;
    float* q=R+G_WI0+r*I_LEN;
    q[I_VALID]=1.f; q[I_KIND+ci(gi(S.wkind,(long long)e*K+k),0,NW-1)]=1.f;
    put_pos(S,q+I_POS,w[0],w[1],w[2]);
    double dx=w[0]-px, dy=w[1]-py, dz=w[2]-pz;
    put_rel(q+I_REL,dx,dy,dz,c,sn);
    q[I_DIST]=(float)(norm3(dx,dy,dz)*ID_DIST);
  }
  R[G_WICOUNT]=(float)cnt*IF_8;
}
__device__ void lane_sounds(const OS& S, int e, int i, float* R){
  if(!S.has_audio) return;
  long long pi=(long long)e*10+i;
  if(!gi(S.alive,pi)) return;
  long long n=gi(S.emn,e); if(n>S.cap) n=S.cap;
  double px=S.x[pi], py=S.y[pi], pz=S.z[pi], yaw=S.yaw[pi];
  int found=0, cnt=0;
  for(int c=0;c<n;c++){
    long long ix=(long long)e*S.cap+c;
    int it=(int)gi(S.emit,ix); long long src=gi(S.emsrc,ix);
    if(it<0 || it>13) continue;
    bool is_src= src==i;
    bool selfh= is_src && IT_SELF[it];
    bool oth= !is_src && IT_OTHERS[it];
    bool nopos= oth && !IT_HASPOS[it];
    const double* p=S.empos+ix*3;
    double dx=p[0]-px, dy=p[1]-py, dz=p[2]-pz;
    double d=sqrt((dx*dx+dy*dy)+dz*dz);
    double rad=IT_RAD[it];
    bool cand= oth && IT_HASPOS[it] && (rad<0 || d<=rad);
    if(!(selfh || nopos || cand)) continue;
    cnt++;
    if(found>=NSND) continue;
    int dc, bc;
    if(selfh){ dc=0; bc=0; }
    else if(nopos){ dc=1; bc=1; }
    else {
      if(IT_DIRECT[it]){ double sn=sin(atan2(-dy,dx)-yaw); dc= fabs(sn)<CPAN? 4 : (sn>0? 2 : 3); } else dc=1;
      bc= d<NEARD? 2 : (d<MIDD? 3 : 4);
    }
    float* q=R+G_SND0+found*S_LEN;
    q[S_VALID]=1.f; q[S_KIND+IT_KIND[it]]=1.f; q[S_DIR+dc]=1.f; q[S_BAND+bc]=1.f;
    found++;
  }
  R[G_SNDCOUNT]=(float)cnt*IF_8;
}
#define RPB 8
__global__ void __launch_bounds__(128) obs_k(OS S, float* __restrict__ out, int E){
  __shared__ __align__(16) float buf[RPB*OBS_DIM];
  // thread -> (row, task): the 8 rows of a task run side by side in one warp (4 tasks per warp, one heavy task per warp)
  const int TASK[16]={0,1,2,3, 14,4,5,6, 15,7,8,9, 13,10,11,12};
  int lr=threadIdx.x&7, lane=TASK[threadIdx.x>>3];
  long long row0=(long long)blockIdx.x*RPB, nrow=(long long)E*10;
  for(int k=threadIdx.x;k<RPB*OBS_DIM;k+=blockDim.x) buf[k]=0.f;
  __syncthreads();
  long long row=row0+lr;
  if(row<nrow){
    int e=(int)(row/10), i=(int)(row%10); float* R=buf+lr*OBS_DIM;
    if(lane==0) lane_rd_me(S,e,i,R);
    else if(lane<=3) lane_weapon(S,e,i,lane-1,R);
    else if(lane<=8) lane_enemy(S,e,i,lane-4,R);
    else if(lane<=12) lane_team(S,e,i,lane-9,R);
    else if(lane==13) lane_bomb(S,e,i,R);
    else if(lane==14) lane_items(S,e,i,R);
    else lane_sounds(S,e,i,R);
  }
  __syncthreads();
  long long nr= nrow-row0<RPB? nrow-row0 : RPB;
  float* o=out+row0*OBS_DIM;
  if(nr==RPB){
    float4* o4=(float4*)o; const float4* b4=(const float4*)buf;
    for(int k=threadIdx.x;k<RPB*OBS_DIM/4;k+=blockDim.x) o4[k]=b4[k];
  } else for(long long k=threadIdx.x;k<nr*OBS_DIM;k+=blockDim.x) o[k]=buf[k];
}
static TF tf(const torch::Tensor& t){
  TF f; f.p=t.numel()? t.data_ptr() : nullptr;
  auto d=t.scalar_type();
  f.t= d==torch::kFloat64?0: d==torch::kFloat32?1: d==torch::kInt64?2: d==torch::kInt32?3: d==torch::kInt16?4: d==torch::kInt8?5:6;
  return f;
}
static const double* dp(const torch::Tensor& t){ return t.numel()? t.data_ptr<double>() : nullptr; }
void build(std::vector<torch::Tensor> f, std::vector<torch::Tensor> ti, std::vector<int64_t> iv, torch::Tensor out){
  OS S;
  S.x=dp(f[0]); S.y=dp(f[1]); S.z=dp(f[2]); S.vx=dp(f[3]); S.vy=dp(f[4]); S.vz=dp(f[5]); S.yaw=dp(f[6]); S.pitch=dp(f[7]);
  S.hp=dp(f[8]); S.armor=dp(f[9]); S.inacc=dp(f[10]); S.rs=dp(f[11]); S.bpos=dp(f[12]); S.wpos=dp(f[13]); S.empos=dp(f[14]);
  S.cen=dp(f[15]); S.zpos=dp(f[16]);
  TF* T[]={&S.alive,&S.crouch,&S.grounded,&S.helmet,&S.kit,&S.has_bomb,&S.cash,&S.mm,&S.ladder,&S.reattach,&S.walk,&S.standb,&S.pickup,
    &S.switch_until,&S.throw_ready,&S.blind_until,&S.nades,&S.objective,&S.obj_progress,&S.active,&S.zoom,
    &S.wid,&S.ammo,&S.reserve,&S.shots,&S.shot_ready,&S.reload_until,&S.reload_commit,&S.last_shot,&S.committed,&S.shell,&S.pressed,
    &S.tick,&S.t0,&S.round_index,&S.round_start,&S.live_end,&S.post_end,&S.phase,&S.attack_team,&S.round_wins,&S.loss_level,&S.bstate,
    &S.bcarrier,&S.fuse_end,&S.bduration,&S.bsite,&S.ot,&S.wt,&S.wvalid,&S.wkind,&S.vp,&S.pv,&S.emn,&S.emit,&S.emsrc};
  int nT=(int)(sizeof(T)/sizeof(T[0]));
  TORCH_CHECK((int)ti.size()==nT, "obs build: wrong number of typed fields");
  for(int k=0;k<nT;k++) *T[k]=tf(ti[k]);
  S.K=(int)iv[0]; S.cap=(int)iv[1]; S.Z=(int)iv[2]; S.map_idx=(int)iv[3]; S.has_items=(int)iv[4]; S.has_audio=(int)iv[5];
  S.ot_def=iv[6]; S.wt_def=iv[7];
  int E=(int)f[0].size(0);
  long long nb=((long long)E*10+RPB-1)/RPB;
  if(nb) obs_k<<<(int)nb,128,0,at::cuda::getCurrentCUDAStream()>>>(S,out.data_ptr<float>(),E);
}
"""

_OBS_TYPED = ('alive', 'crouch', 'grounded', 'helmet', 'kit', 'has_bomb', 'cash', 'movement_mode', 'ladder_id', 'reattach', 'walk', 'standb',
              'pickup_ready_tick', 'switch_until', 'throw_ready', 'blind_until', 'nades', 'objective', 'obj_progress', 'active', 'zoom',
              'w_wid', 'w_ammo', 'w_reserve', 'w_shots', 'w_shot_ready', 'w_reload_until', 'w_reload_commit', 'w_last_shot',
              'w_reload_committed', 'w_reload_shell', 'w_pressed', 'tick', 't0', 'round_index', 'round_start_tick', 'live_end', 'post_end',
              'phase', 'attack_team', 'round_wins', 'loss_level', 'bomb_state', 'bomb_carrier', 'bomb_fuse_end', 'bomb_duration', 'bomb_site',
              'overtime_number', 'win_target', 'wi_valid', 'wi_kind', 'vp', 'pv', 'em_n', 'em_itype', 'em_src')
# torch.linalg.vector_norm / sum over a last dim of 3 (float64, CUDA) combine the squares as ((a^2 + c^2) + b^2) (two strided accumulators; checked on CUDA, torch 2.10)
NORM3 = '((a*a+c*c)+b*b)'
_omod = None


def _obs_ext():
    global _omod
    if _omod is None:
        import hashlib, os
        from torch.utils.cpp_extension import load_inline
        src = ('#include <torch/extension.h>\n#include <ATen/cuda/CUDAContext.h>\n#include <math.h>\n#define NORM3(a,b,c) %s\n' % NORM3 +
               _obs_src())
        key = hashlib.sha1(src.encode()).hexdigest()[:10]
        os.environ.setdefault('TORCH_EXTENSIONS_DIR', './fps_data/tmp/torch_ext')
        _omod = load_inline('fpsobs_' + key, cpp_sources='void build(std::vector<torch::Tensor> f, std::vector<torch::Tensor> ti, '
                            'std::vector<int64_t> iv, torch::Tensor out);', cuda_sources=src, functions=['build'],
                            extra_cuda_cflags=['-O3', '-fmad=false'], verbose=False)
    return _omod


def _site_of(s, st):
    """bomb_site when the state has none (as build_obs_torch)."""
    E = s['x'].shape[0]; dev = s['x'].device
    bpos, bstate = s['bomb_pos'], s['bomb_state'].long()
    if len(st.site_pos):
        site = ((bpos[:, None, :2] - st.site_pos[None, :, :2]) ** 2).sum(-1).argmin(-1)
    else:
        site = torch.full((E,), -1, dtype=torch.long, device=dev)
    return torch.where(bstate >= 2, site, torch.full_like(site, -1))


def obs_fused(s, vis_parts, st, pv, audio=None, out=None, ctx=None):
    """the fused kernel: pv = point visibility bool [E,10,K+1] (world items 0..K-1, bomb K; gpusim.vis.trace) or None."""
    dev = s['x'].device
    E = s['x'].shape[0]
    ctx = ctx or {}
    e0 = torch.zeros(0, device=dev)
    e64 = torch.zeros(0, dtype=torch.float64, device=dev)

    def d64(k, default=None):
        v = s.get(k, default)
        if v is None: return e64
        return v if (v.dtype == torch.float64 and v.is_contiguous()) else v.double().contiguous()

    def ty(k, *alt):
        for n in (k,) + alt:
            v = s.get(n)
            if v is not None: return v.contiguous()
        return e0
    has_items = 'wi_valid' in s
    K = s['wi_valid'].shape[1] if has_items else 0
    if pv is None:
        pv = torch.zeros(E, 10, K + 1, dtype=torch.bool, device=dev)
    if audio is not None:
        em_n, em_it, em_src, em_pos = audio['n'], audio['itype'], audio['src'], audio['pos']
        cap = em_it.shape[1]
    else:
        em_n = em_it = em_src = e0; em_pos = e64; cap = 0
    f = [d64('x'), d64('y'), d64('z'), d64('vx'), d64('vy'), d64('vz'), d64('yaw'), d64('pitch'), d64('hp'), d64('armor'),
         d64('w_inacc'), d64('w_rs'), d64('bomb_pos'), d64('wi_pos') if has_items else e64,
         em_pos.double().contiguous(), st.center.double().contiguous(), st.zone_pos.double().contiguous()]
    site = s.get('bomb_site')
    if site is None:
        site = _site_of(s, st)
    tv = dict(reattach=ty('reattach_until', 'mv_reattach_until'), walk=ty('walk_held', 'act_walk'), standb=ty('stand_blocked', 'mv_stand_blocked'),
              bomb_site=site.contiguous(), vp=vis_parts.contiguous(), pv=pv.contiguous(), em_n=em_n.contiguous(), em_itype=em_it.contiguous(),
              em_src=em_src.contiguous())
    ti = [tv[k] if k in tv else ty(k) for k in _OBS_TYPED]
    iv = [K, cap, len(st.zone_pos), st.map_idx, int(has_items), int(audio is not None), int(ctx.get('overtime_number', 0)),
          int(ctx.get('win_target', 13))]
    if out is None:
        out = torch.empty(E, 10, OBS_DIM, dtype=torch.float32, device=dev)
    assert out.is_contiguous() and out.dtype == torch.float32 and tuple(out.shape) == (E, 10, OBS_DIM)
    _obs_ext().build(f, ti, iv, out)
    return out


# ---------------------------------------------------------------------------------------------- decode (debug / parity)
def decode_obs(obs, e, seat, st):
    """One env / seat vector -> nested dict with fpssim.obs.build_observation field names where they exist.
    Fields stored as remaining ticks decode to absolute ticks (value <= current tick means 'already elapsed');
    sentinel fields (reload_until, last_shot, ladder_id, fuse_end_tick) decode to None when unset."""
    v = obs[e, seat].detach().double().cpu().numpy()
    g = lambda name: v[OFF[name][0]:OFF[name][0] + OFF[name][1]]
    G = lambda name: float(g(name)[0])
    cen = st.center.cpu().numpy()
    tick = int(round(G('rd.match_tick') * TICK))
    team = int(round(G('rd.team')))
    side = 'attack' if G('rd.side_attack') > 0.5 else 'defend'
    ph = PHASES[int(np.argmax(g('rd.phase')))]
    rw = [int(round(x * 13)) for x in g('rd.round_wins')]
    al = [int(round(x * 5)) for x in g('rd.alive')]
    rw_t = rw if team == 0 else rw[::-1]
    al_t = al if team == 0 else al[::-1]
    ang = lambda sn, cs: float(math.atan2(sn, cs))
    pos = lambda name: (g(name) * POS + cen).tolist()
    me = {}
    me['x'], me['y'], me['z'] = pos('me.pos')
    me['yaw'] = ang(*g('me.yaw'))
    me['pitch'] = G('me.pitch') * math.pi / 2
    vel = g('me.vel') * VEL
    me['vx'], me['vy'], me['vz'] = vel.tolist()
    me['hp'] = G('me.hp') * 100; me['armor'] = G('me.armor') * 100
    me['helmet'] = G('me.helmet') > .5; me['has_kit'] = G('me.has_kit') > .5
    me['cash'] = G('me.cash') * CASH; me['alive'] = G('me.alive') > .5; me['crouch'] = G('me.crouch') > .5
    me['grounded'] = G('me.grounded') > .5; me['walk'] = G('me.walk') > .5; me['quiet'] = me['walk']
    me['stand_blocked'] = G('me.stand_blocked') > .5
    me['active_slot'] = int(np.argmax(g('me.active'))); me['zoom_level'] = int(np.argmax(g('me.zoom')))
    me['ads_progress'] = 1.0 if me['zoom_level'] > 0 else 0.0
    me['movement_mode'] = MODES[int(np.argmax(g('me.movement_mode')))]
    lid = int(round(G('me.ladder_id') * 64)) - 1
    me['ladder_id'] = None if lid < 0 else lid
    me['reattach_until'] = tick + int(round(G('me.reattach_rem') * SEC))
    me['pickup_ready_tick'] = tick + int(round(G('me.pickup_rem') * SEC))
    me['switch_until'] = tick + int(round(G('me.switch_rem') * SEC))
    me['throw_ready'] = tick + int(round(G('me.throw_rem') * SEC))
    me['blind_ticks'] = int(round(G('me.blind_rem') * BLIND))
    cnt = np.rint(g('me.throwables') * 2).astype(int)
    me['throwables'] = [UTIL_IDS[i] for i in range(NU) for _ in range(cnt[i])]
    gm = {'smokegrenade': 'smoke', 'hegrenade': 'frag', 'flashbang': 'flash'}
    me['throwable_kinds'] = [gm[k] for k in ('smokegrenade', 'hegrenade', 'flashbang') if k in me['throwables']]
    me['carrying_bomb'] = G('me.carrying_bomb') > .5
    ws = []
    for k in range(3):
        pre = 'me.w%d.' % k
        if G(pre + 'present') < .5:
            ws.append(None); continue
        gw = lambda n: g(pre + n)
        Gw = lambda n: float(gw(n)[0])
        w = dict(id=WEAPON_IDS[int(np.argmax(gw('id')))], ammo=Gw('ammo') * AMMO, reserve=Gw('reserve') * RESERVE,
                 shot_ready=tick + Gw('shot_ready') * SEC, shots=Gw('shots') * SHOTS)
        w['reload_until'] = tick + int(round(Gw('reload_rem') * SEC)) if Gw('reloading') > .5 else None
        w['last_shot'] = tick - int(round(Gw('since_shot') * SINCE)) if Gw('has_last_shot') > .5 else None
        pu = gw('punch') / ANG; pv = gw('punch_vel'); vp = gw('view_punch') / ANG
        w['recoil_pitch'], w['recoil_yaw'] = float(pu[0]), float(pu[1])
        w['aim_punch'] = dict(index=Gw('punch_index') * 64, pitch=float(pu[0]), yaw=float(pu[1]), velPitch=float(pv[0]),
                              velYaw=float(pv[1]), viewPitch=float(vp[0]), viewYaw=float(vp[1]))
        w['inaccuracy'] = Gw('inaccuracy') / INACC
        w['trigger_state'] = dict(pressed=Gw('pressed') > .5)
        if Gw('reloading') > .5:
            w['reload_state'] = dict(commit=tick + int(round(Gw('commit_rem') * SEC)), finish=w['reload_until'],
                                     committed=Gw('committed') > .5, shell=Gw('shell') > .5)
        else:
            w['reload_state'] = None
        w['last_update_tick'] = float(tick)
        ws.append(w)
    me['weapons'] = ws
    aw = ws[me['active_slot']] if me['active_slot'] < 3 else None
    me['ammo'] = int(round(aw['ammo'])) if aw else 0
    me['shot_ready'] = aw['shot_ready'] if aw else 0
    # visible enemies
    vis = []
    for k in range(NEN):
        pre = 'en%d.' % k
        if G(pre + 'valid') < .5:
            continue
        sd = (1 - team) * 5 + int(np.argmax(g(pre + 'seat')))
        wv = g(pre + 'weapon')
        parts = [n for n, f in zip(('head', 'chest', 'abdomen'), g(pre + 'parts')) if f > .5]
        x_, y_, z_ = pos(pre + 'pos')
        vis.append(dict(seat=sd, team=sd // 5, x=x_, y=y_, z=z_, yaw=ang(*g(pre + 'yaw')), pitch=G(pre + 'pitch') * math.pi / 2,
                        crouch=G(pre + 'crouch') > .5, weapon=WEAPON_IDS[int(np.argmax(wv))] if wv.max() > .5 else None,
                        movement_mode=MODES[int(np.argmax(g(pre + 'movement_mode')))], climb_velocity=0.0, lean=0.0, parts=parts))
    # teammates
    tms = []
    own = [s_ for s_ in range(team * 5, team * 5 + 5) if s_ != seat]
    for k in range(NTM):
        pre = 'tm%d.' % k
        sd = team * 5 + int(np.argmax(g(pre + 'seat')))
        zv = g(pre + 'zone')
        eq = [WEAPON_IDS[i] for i in np.where(g(pre + 'weapons') > .5)[0]]
        thr = np.rint(g(pre + 'throwables') * 2).astype(int)
        eq += [UTIL_IDS[i] for i in range(NU) for _ in range(thr[i])]
        if G(pre + 'armor') > .5: eq.append('armor')
        if G(pre + 'helmet') > .5: eq.append('helmet')
        if G(pre + 'kit') > .5: eq.append('defusekit')
        tms.append(dict(seat=sd, alive=G(pre + 'alive') > .5, cash=G(pre + 'cash') * CASH,
                        zone=st.zone_names[int(np.argmax(zv))] if zv.max() > .5 else '', equipment=eq))
    # bomb
    sidx = int(np.argmax(g('bomb.state')))
    b = dict(state=BOMB_STATES[sidx], fuse_end_tick=(tick + int(round(G('bomb.fuse_rem') * FUSE))) if G('bomb.fuse_valid') > .5 else None)
    if G('bomb.known') > .5:
        c = g('bomb.carrier')
        b['id'] = 'r%d:bomb' % int(round(G('rd.round_index') * 24))
        b['carrier'] = int(np.argmax(c)) if c.max() > .5 else None
        b['position'] = pos('bomb.pos')
        sv = g('bomb.site')
        b['site'] = st.site_names[int(np.argmax(sv))] if sv.max() > .5 else None
    if G('bomb.is_planter') > .5:
        b['plant_progress'] = G('bomb.plant_progress') * PROG
    if G('bomb.is_defuser') > .5:
        b['defuse_progress'] = G('bomb.defuse_progress') * PROG
        if G('bomb.defuse_duration') > 0:
            b['defuse_duration'] = G('bomb.defuse_duration') * PROG
    # world items
    items = []
    for k in range(NITEM):
        pre = 'wi%d.' % k
        if G(pre + 'valid') < .5:
            continue
        items.append(dict(kind=WEAPON_IDS[int(np.argmax(g(pre + 'kind')))], position=pos(pre + 'pos')))
    # sounds
    snds = []
    for k in range(NSND):
        pre = 'snd%d.' % k
        if G(pre + 'valid') < .5:
            continue
        snds.append(dict(type=SOUND_KINDS[int(np.argmax(g(pre + 'kind')))], direction=DIRS[int(np.argmax(g(pre + 'dir')))],
                         distance_band=BANDS[int(np.argmax(g(pre + 'band')))]))
    return dict(
        rules_version='rsi-fps-5.5', match_tick=tick, round_index=int(round(G('rd.round_index') * 24)),
        round_tick=int(round(G('rd.round_tick') * RTICK)), phase=ph, phase_remaining_ticks=int(round(G('rd.phase_remaining') * RTICK)),
        seat=seat, team=team, side=side, role=0, round_wins=rw_t, alive_by_team=al_t, me=me, teammates=tms, visible=vis,
        sounds=snds, heard=[], bomb=b, world_items=items, map_id=st.id, attack_team=team if side == 'attack' else 1 - team,
        period='overtime' if G('rd.overtime') > .5 else 'regulation', overtime_number=int(round(G('rd.overtime_number') * 10)),
        win_target=int(round(G('rd.win_target') * 13)), buy_deadline_tick=tick + int(round(G('rd.buy_remaining') * 1860)),
        loss_level=int(round(G('rd.loss_level') * 5)), n_visible_items=int(round(G('wi.count') * 8)),
        n_heard=int(round(G('snd.count') * 8)))
