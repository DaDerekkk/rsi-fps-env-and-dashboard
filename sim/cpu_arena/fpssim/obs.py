"""Legal per-seat observation (rsi-fps-5.5 contract) and per-seat event views.

Everything here works on an *ArenaView*: a dict in the shape of one public `frames.jsonl` row (the platform's own
authoritative frame format), plus a few keys the frames do not carry.  Using the platform format means the builder is
validated directly on recorded frames (validate/obs.py) and the Arena only needs an adapter (`view_from_arena`).

ArenaView (dict) -- keys used:
  match_tick, round_tick, round_index, phase ('buy'|'live'|'post'|'finished'), phase_end_tick, attack_team,
  round_wins [2], fuse_end_tick, period, overtime_number, win_target, buy_deadline_tick
  players[10]: platform player dicts (frames.jsonl format: x,y,z,yaw,pitch,vx,vy,vz,hp,armor,helmet,cash,alive,crouch,
      stamina, lean_offset, active_slot, ads_progress, sprinting, grounded, slots{primary,secondary,knife,kit},
      throwables[] (handles 'item:N'), throwable_kinds[] (parallel kind names; optional, else from grenades{smoke,frag,flash}),
      weapons[3] (dict|None), pickup_ready_tick, switch_until, throw_ready, blind_until, ammo, reload, recoil, walk, quiet,
      duck, zoom_level, shot_ready, movement_mode, ladder_id, team, height ...)  [extra frame keys are ignored]
  bomb: {id,state,carrier,site,position,planter,plant_progress,defuser,defuse_progress,defuse_duration}
  world_items: [{id, kind, position, ...}]
  audio_events: emitter list since the previous decision (platform audio_events format, see AUDIO section)
  smokes: [{position, radius, created, expires}]          (only used by the default visibility model)
 extra keys (not in frames; optional):
  loss_level [2]          own-team loss level index is reported      (default [1,1])
  receipts {seat: [...]}  own action/economy receipts since the last decision, platform format (view_from_arena converts the Arena's)
  messages_by_seat {seat: [...]}  [{seat, match_tick, signal}] of the teammates (view_from_arena builds it from the held actions);
  messages {team: [...]}  legacy: the five seats' entries of a team, the own seat is filtered out
  Actor views never trust offline full-information item and bomb reveal overrides.
  smoke_fn(p0, p1, tick)  optional exact smoke test (GrenadeSystem.smoke_blocks) used by the point visibility
  static                  the map dict (static.json['map']); needed for the first observation and zones
  vis [10] int bitmask    full mutual visibility (bit s of vis[i] => i sees s, incl. teammates; self bit set).
                          If absent it is built from `visible` args / `los`.
  world_item_visible [10] list of visible world item ids per seat (frames carry it); else computed with `los`
  los                     callable(p0, p1) -> bool  (static-world line of sight, e.g. Arena.los)

Public API
  build_observation(arena_view, seat, visible, first=False) -> dict
        visible = [(enemy_seat, parts)] for this seat (Arena.visible_enemies()[seat]); None -> taken from view['vis'].
        arena_view may also be the Arena object itself (then view_from_arena(arena) is used, cached per tick).
  event_views(event, world) -> {seat: True | dict(redacted) }   (absent seat = not delivered)
        world: EventWorld (or anything with .vis, .alive, .point_visible(seat, xyz), .projectile_visible(seat, id))
  sounds_for(view, seat, events=None) -> list of {type, direction, distance_band}
  bomb_view(view, seat), visible_world_items(view, seat)
  VisModel: default mutual-visibility / point-visibility model (55 deg / 45 deg cone + LOS + smoke)
  AudioEmitter: generates platform-style audio_events from successive player states and events (see docs/obs.md)

Event-view rules were fitted against the `views` field of 26 public recordings (validate/obs.py prints the agreement numbers); the observation
fields against our own training.jsonl exports (tools/obscheck_diff.py, docs/obs.md).
"""
import copy
import hashlib
import json
import math
from numbers import Integral, Real
import weakref

RULES_VERSION = 'rsi-fps-5.5'
HALF_H = math.radians(55.0)
HALF_V = math.radians(45.0)
EYE = {False: 1.62, True: 0.95}
BODY_H = {False: 1.8, True: 1.1}
BODY_FRACS = (0.9, 0.75, 0.55)          # head / chest / abdomen sample heights (fraction of body height)
BASE_KEYS = ('type', 'event_id', 'match_tick', 'round_index')


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


# ======================================================================================= event views
GLOBAL_EVENTS = {'kill', 'round_start', 'round_end', 'plant_complete', 'bomb_explode', 'defuse_complete', 'side_swap',
                 'match_end'}
TEAM_EVENTS = {'cash', 'item_create', 'buy', 'consume'}            # team of e['seat']
HIDDEN_EVENTS = {'item_destroy', 'kill_reward', 'economy_round_start', 'round_economy'}
SELF_EVENTS = {'flash': 'target', 'fault': 'seat'}
# whole event delivered to the actor and to every seat that sees the actor
ACTOR_EVENTS = {'draw', 'reload_commit', 'reload_finish', 'reload_cancel', 'pickup', 'plant_start', 'plant_cancel',
                'defuse_start', 'defuse_cancel', 'defuse_complete_detail', 'throw'}
# actor events delivered with some keys always removed
ACTOR_DROP_KEYS = {'reload_start': ('duration',)}
# component events: key groups unlocked by (a) actor visible (b) point visible (c) target visible (d) being the target
COMPONENTS = {
    'fire': dict(actor=('player', 'weapon', 'from', 'pellet'), point=('to', 'pellet'), point_key='to'),
    'drop': dict(actor=('player', 'item_id'), point=('position',), point_key='position'),
    'plant_complete_detail': dict(actor=('player', 'item_id', 'site'), point=('position', 'site'), point_key='position'),
    'bomb_explode_detail': dict(actor=(), point=('position',), point_key='position'),
}
PROJECTILE_EVENTS = {'bounce', 'detonate'}                          # delivered whole when the grenade/point is visible


class EventWorld:
    """Minimal visibility context for event_views.
    vis: list of 10 int bitmasks (bit s of vis[i] -> seat i sees seat s; self bit set), taken at the start of the event
         tick (i.e. before the event's own state change -- a lethal hit still counts the victim as visible).
    alive: list of 10 bool (same instant).
    point_fn(seat, xyz) -> bool : point inside the seat's view cone and not occluded (VisModel.point_visible).
    proj_fn(seat, item_id, xyz) -> bool : grenade visible (defaults to point_fn on its position).
    alive_post: list of 10 bool after the event tick (optional; used by component events such as a death drop)."""

    def __init__(self, vis, alive, point_fn=None, proj_fn=None, alive_post=None):
        self.vis = vis
        self.alive = alive
        self.alive_post = alive_post      # alive right after the event tick (a player killed this tick is no longer seen)
        self._point = point_fn
        self._proj = proj_fn

    def sees(self, seat, other):
        if other is None:
            return False
        return seat == other or bool((self.vis[seat] >> other) & 1)

    def point_visible(self, seat, p):
        if p is None or self._point is None or not self.alive[seat]:
            return False
        return bool(self._point(seat, p))

    def projectile_visible(self, seat, item_id, p):
        if self._proj is not None:
            return bool(self._proj(seat, item_id, p))
        return self.point_visible(seat, p)


def _redact(e, keys):
    out = {k: e[k] for k in BASE_KEYS if k in e}
    for k in keys:
        if k in e:
            out[k] = e[k]
    return out


def _full_or(e, keys):
    """True if `keys` (plus base) covers every key of the event, else the redacted dict."""
    if all(k in keys or k in BASE_KEYS or k == 'views' for k in e):
        return True
    return _redact(e, keys)


def event_views(e, world):
    """-> {seat: True | redacted dict}.  Seats not in the result do not receive the event."""
    t = e['type']
    out = {}
    if t in GLOBAL_EVENTS:
        return {s: True for s in range(10)}
    if t in HIDDEN_EVENTS:
        return {}
    if t in SELF_EVENTS:
        s = e.get(SELF_EVENTS[t])
        return {s: True} if s is not None else {}
    if t in TEAM_EVENTS or (t in ('drop', 'pickup') and 'seat' in e and 'player' not in e):
        team = e['seat'] // 5
        return {s: True for s in range(team * 5, team * 5 + 5)}
    if t in ACTOR_EVENTS or t in ACTOR_DROP_KEYS:
        a = e.get('player')
        drop = ACTOR_DROP_KEYS.get(t, ())
        for s in range(10):
            if s == a or (world.alive[s] and world.sees(s, a)):
                out[s] = _redact(e, [k for k in e if k not in drop and k != 'views']) if drop else True
        return out
    if t == 'hit':
        a, tg = e.get('player'), e.get('target')
        for s in range(10):
            keys = []
            alive = world.alive[s]
            if a is not None and (s == a or (alive and world.sees(s, a))):
                keys += ['player', 'kind']
            if tg is not None and (s == tg or s == a or (alive and world.sees(s, tg))):
                keys += ['target', 'part', 'kind']
            if tg is not None and s == tg:
                keys += ['amount', 'armor_absorbed', 'hp_damage']
            if keys:
                out[s] = _full_or(e, set(keys))
        return out
    if t in COMPONENTS:
        c = COMPONENTS[t]
        a = e.get('player')
        p = e.get(c['point_key'])
        a_post = world.alive_post[a] if (a is not None and world.alive_post is not None) else True
        for s in range(10):
            ka = a is not None and bool(c['actor']) and (s == a or (a_post and world.alive[s] and world.sees(s, a)))
            kp = p is not None and world.point_visible(s, p)
            if ka and (kp or not c['point']) or (kp and not c['actor']):
                out[s] = True              # every component unlocked -> the whole event (incl. e.g. fire.materials)
            elif ka or kp:
                out[s] = _redact(e, (c['actor'] if ka else ()) + (c['point'] if kp else ()))
        return out
    if t in PROJECTILE_EVENTS:
        a = e.get('player')
        for s in range(10):
            if world.projectile_visible(s, e.get('item_id'), e.get('position')):
                out[s] = True
        return out
    # unknown event type: conservative -> actor only
    a = e.get('player', e.get('seat'))
    return {a: True} if isinstance(a, int) else {}


def deliver_events(events, world_at, seat):
    """Events since the previous decision as seen by `seat` (list of full or redacted dicts, original order).
    world_at(event) -> EventWorld for the event's tick."""
    out = []
    for e in events:
        v = e.get('views')
        if v is None:
            v = event_views(e, world_at(e))
        x = v.get(seat, v.get(str(seat)))
        if x is True:
            out.append({k: e[k] for k in e if k != 'views'})
        elif isinstance(x, dict):
            out.append(x)
    return out


# ======================================================================================= visibility model
class VisModel:
    """Mutual visibility (incl. teammates, as in the frames' `visible` lists) and point visibility.
    los(p0, p1) -> bool static-world line of sight; smokes: list of (x, y, z, r) spheres (or None)."""

    def __init__(self, los, smokes=None, half_h=HALF_H, half_v=HALF_V, smoke_fn=None, tick=None):
        self.los = los
        self.smokes = smokes or []
        self.half_h, self.half_v = half_h, half_v
        self.smoke_fn, self.tick = smoke_fn, tick          # optional exact smoke test smoke_fn(p0, p1, tick) -> bool (GrenadeSystem.smoke_blocks)

    @staticmethod
    def eye(p):
        return (p['x'], p['y'], p['z'] + EYE[bool(p.get('crouch'))])

    def in_cone(self, p, q):
        e = self.eye(p)
        dx, dy, dz = q[0] - e[0], q[1] - e[1], q[2] - e[2]
        dh = math.hypot(dx, dy)
        if dh < 1e-9 and abs(dz) < 1e-9:
            return True
        if abs(_wrap(math.atan2(-dy, dx) - p['yaw'])) > self.half_h:
            return False
        return abs(math.atan2(dz, dh) - p.get('pitch', 0.0)) <= self.half_v

    def _clear(self, a, b):
        if not self.los(a, b):
            return False
        if self.smoke_fn is not None and self.smoke_fn(a, b, self.tick):
            return False
        for (sx, sy, sz, r) in self.smokes:
            d = [b[i] - a[i] for i in range(3)]
            L2 = sum(x * x for x in d)
            t = 0.0 if L2 == 0 else max(0.0, min(1.0, ((sx - a[0]) * d[0] + (sy - a[1]) * d[1] + (sz - a[2]) * d[2]) / L2))
            c = [a[i] + d[i] * t for i in range(3)]
            if (c[0] - sx) ** 2 + (c[1] - sy) ** 2 + (c[2] - sz) ** 2 < r * r:
                return False
        return True

    def point_visible(self, p, q, backoff=0.03):
        """p: player dict, q: world point.  Point inside the 55/45 deg cone with a clear segment eye->point (the
        segment stops `backoff` metres short so surface points such as bullet impacts are not self-occluded)."""
        if not p.get('alive', True) or not self.in_cone(p, q):
            return False
        e = self.eye(p)
        d = [q[i] - e[i] for i in range(3)]
        n = math.sqrt(sum(x * x for x in d)) or 1.0
        q2 = [q[i] - backoff * d[i] / n for i in range(3)]
        return self._clear(e, q2)

    def parts(self, p, q):
        """visible parts of player q for viewer p ([] if not visible)."""
        if not p.get('alive', True) or not q.get('alive', True):
            return []
        if p.get('blind', False):
            return []
        h = BODY_H[bool(q.get('crouch'))]
        out = []
        e = self.eye(p)
        for frac, name in zip(BODY_FRACS, ('head', 'chest', 'abdomen')):
            tp = (q['x'], q['y'], q['z'] + h * frac)
            if self.in_cone(p, tp) and self._clear(e, tp):
                out.append(name)
        return out

    def mutual(self, players, blind=None):
        """-> (vis bitmasks [10], parts {(i, j): [...]}) over all pairs (teammates included)."""
        n = len(players)
        vis = [1 << i for i in range(n)]
        parts = {}
        for i, p in enumerate(players):
            if not p.get('alive', True) or (blind and blind[i]):
                continue
            for j, q in enumerate(players):
                if i == j:
                    continue
                pr = self.parts(p, q)
                if pr:
                    vis[i] |= 1 << j
                    parts[(i, j)] = pr
        return vis, parts


# ======================================================================================= sounds
NEAR, MID = 8.0, 20.0
CENTER_PAN = 0.20          # |sin(relative azimuth)| below this -> 'center' (fit on our own training.jsonl, 100% on clean pairs)
# Which emitters reach `sounds` (fit on 12, then 21 of our own training.jsonl exports vs the public frames' audio_events).
# gunshot: the only 2 samples in 21 matches (teammate shots at 59 m and 75 m, 'far') were delivered, so gunshots are heard map-wide
# (no radius); enemy and own gunshots have NO sample -- assumed heard like the other emitters with neither local_only nor remote_only
# (own jump / own landing are delivered self/self).
HEARD_BY_OTHERS = frozenset({'footstep', 'bomb_beep', 'bomb_beep_fast', 'grenade_bounce', 'bomb_planted', 'bomb_exploded', 'bomb_defused',
                             'land_hard', 'land_soft', 'jump', 'fire_ambient', 'fire_started', 'gunshot'})
HEARD_BY_SELF = frozenset({'footstep', 'land_hard', 'land_soft', 'reload', 'jump', 'gunshot'})   # draw / throws never appear.  jump: 1 own sample (self/self); gunshot: see below


def _pan(listener, pos):
    dx, dy = pos[0] - listener['x'], pos[1] - listener['y']
    dh = math.hypot(dx, dy)
    if dh < 1e-6:
        return 0.0
    yaw = listener['yaw']
    # right vector of forward (cos yaw, -sin yaw) in a z-up right-handed frame
    rx, ry = -math.sin(yaw), -math.cos(yaw)
    return (dx * rx + dy * ry) / dh


def _audible(a, listener, seat, los=None):
    """-> None (not heard) or (direction, distance_band).  Platform rules fit on our own training.jsonl (12 matches):
    only emitters of the decision tick itself are delivered (Arena drops the two intermediate ticks); kinds whitelisted
    above; own sounds -> self/self; no position -> unknown/unknown; beyond `radius` (footstep 25 m) -> not heard;
    NO occlusion effect; distance bands near < 8 m, mid < 20 m, else far (ear to emitter position);
    direction: the platform's stereo label is MIRRORED (source on the listener's right -> 'left'),
    'center' when |sin(relative azimuth)| < 0.2, 'unknown' for non-directional kinds."""
    src = a.get('source'); kind = a.get('kind')
    if src == seat:
        return ('self', 'self') if (kind in HEARD_BY_SELF and not a.get('remote_only')) else None
    if a.get('local_only') or kind not in HEARD_BY_OTHERS:
        return None
    pos = a.get('position')
    if pos is None:
        return ('unknown', 'unknown')
    ear = (listener['x'], listener['y'], listener['z'])      # distance is measured from the listener's FEET (fit: 0 band errors)
    d = math.dist(ear, pos)
    r = a.get('radius')
    if r is not None and d > r:
        return None
    if a.get('directional', True):
        dx, dy = pos[0] - ear[0], pos[1] - ear[1]
        rel = math.atan2(-dy, dx) - listener['yaw']          # > 0: source to the listener's right (platform yaw convention)
        sn = math.sin(rel)
        direction = 'center' if abs(sn) < CENTER_PAN else ('left' if sn > 0 else 'right')
    else:
        direction = 'unknown'
    band = 'near' if d < NEAR else ('mid' if d < MID else 'far')
    return (direction, band)


def sounds_for(view, seat, los=None):
    """Heard sounds for `seat` from view['audio_events'] (emitters accumulated since the previous decision).
    Every emitter is its own entry (the platform does NOT merge banks: a landing gives 2-3 identical entries)."""
    me = view['players'][seat]
    if not me.get('alive', True):
        return []
    out, seen = [], set()
    for a in view.get('audio_events') or []:
        r = _audible(a, me, seat, los)
        if r is None:
            continue
        out.append(dict(type=a.get('kind') or 'unknown', direction=r[0], distance_band=r[1]))
    return out


# ======================================================================================= bomb / items
def _side(view, seat):
    return 'attack' if seat // 5 == view['attack_team'] else 'defend'


def bomb_view(view, seat, vis=None, point_visible=None):
    """Bomb record for `seat` (reproduces frames' bomb_views; see validate/obs.py for agreement).
    attack side: id/carrier/position/site always.  defend side: state/fuse only, plus id/carrier/position/site when the
    bomb (world/planted) or its carrier (held) is visible.  plant_progress only to the planter, defuse_* only to the
    defuser."""
    b = view.get('bomb') or {}
    st = b.get('state')
    fuse = view.get('fuse_end_tick', b.get('fuse_end_tick'))
    if st is None:
        return dict(state='unknown', fuse_end_tick=None)
    pos = b.get('position')
    if pos is None and st == 'held' and b.get('carrier') is not None:
        c = view['players'][b['carrier']]
        pos = (c['x'], c['y'], c['z'])          # held bomb: reported at the carrier's feet (frames bomb_views)
    full = dict(state=st, fuse_end_tick=fuse, id=b.get('id'), carrier=b.get('carrier'),
                position=[float(x) for x in pos] if pos is not None else None, site=b.get('site'))
    attack = _side(view, seat) == 'attack'
    known = attack
    if not known:
        if st == 'held':
            c = b.get('carrier')
            known = c is not None and vis is not None and bool((vis[seat] >> c) & 1) and c // 5 != seat // 5
        elif st in ('world', 'planted', 'defused', 'exploded') and b.get('position') is not None:
            known = bool(point_visible(seat, b['position'])) if point_visible else False
    if known:
        out = full
        if st == 'held' and not attack:
            pass
    else:
        hidden_state = st if st in ('planted', 'defused', 'exploded') else 'unknown'
        out = dict(state=hidden_state, fuse_end_tick=fuse if hidden_state != 'unknown' else None)
    if b.get('planter') == seat and b.get('plant_progress') is not None:
        out['plant_progress'] = b['plant_progress']
    if b.get('defuser') == seat:
        if b.get('defuse_progress') is not None:
            out['defuse_progress'] = b['defuse_progress']
        if b.get('defuse_duration') is not None:
            out['defuse_duration'] = b['defuse_duration']
    return out


def visible_world_items(view, seat, point_visible=None):
    out = [dict(id=w['id'], kind=w['kind'], position=[_num(x) for x in w['position']])
           for w in (view.get('world_items') or ())
           if w.get('position') is not None and point_visible and point_visible(seat, w['position'])]
    return sorted(out, key=lambda w: _serial(w['id']))


# ======================================================================================= observation
MAP_EXCLUDE = ('render',)              # contract: no map.render in the policy input


def policy_map(static):
    return {k: v for k, v in static.items() if k not in MAP_EXCLUDE}


def _zone(static, p):
    lm = (static or {}).get('route_landmarks') or {}
    if not lm:
        lm = {s['id']: s['center'] for s in (static or {}).get('sites') or []}
    if not lm:
        return ''
    return min(lm, key=lambda k: (lm[k][0] - p['x']) ** 2 + (lm[k][1] - p['y']) ** 2 + (lm[k][2] - p['z']) ** 2)


# frame `grenades` counters (smoke/frag/flash only) -> platform throwable kind names (fallback when a player dict has no
# explicit `throwable_kinds`)
GRENADE_KIND = {'smoke': 'smokegrenade', 'frag': 'hegrenade', 'flash': 'flashbang'}


def throwable_kinds(p):
    """Kinds of the carried throwables, parallel to p['throwables'] (platform handles 'item:N', ascending = acquisition order)."""
    k = p.get('throwable_kinds')
    if isinstance(k, (list, tuple)):
        return [str(x) for x in k]
    gr = p.get('grenades') or {}
    return [GRENADE_KIND[g] for g in ('smoke', 'frag', 'flash') for _ in range(int(gr.get(g) or 0))]


def _equipment(p):
    """teammates[].equipment (fit on 1.7 M teammate entries: exact): [] when dead, else the ids of the carried guns/knife in
    slot order (primary, secondary, knife; empty slots skipped), then 'defusekit' if the kit is carried, then one entry per
    carried throwable (kind names, ascending handle order).  Armor/helmet/bomb are NOT listed."""
    if not p.get('alive', True):
        return []
    eq = [w['id'] for w in p.get('weapons') or [] if w and w.get('id')]
    if (p.get('slots') or {}).get('kit'):
        eq.append('defusekit')
    return eq + throwable_kinds(p)


def _num(v, d=0.0):
    """Project finite engine numbers to built-in JSON int/float; no coercion from text."""
    if callable(v):
        v = v()
    if v is None and d is not None:
        return d  # Explicit optional-field default only.
    if type(v) is int:
        return v
    if type(v) is float:
        out = v
    elif isinstance(v, Integral) and not isinstance(v, bool):
        return int(v)
    elif isinstance(v, Real) and not isinstance(v, bool):
        out = float(v)
    else:
        raise ValueError('Expected a real observation scalar')
    if not math.isfinite(out):
        raise ValueError('Expected a finite observation scalar')
    return out


def _int(v, d=0):
    return int(round(_num(v, d)))


def _clamp(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


# me.weapons[i] has exactly these keys in this order in every platform observation (no `last_update_tick`)
WEAPON_KEYS = ('id', 'ammo', 'reserve', 'shot_ready', 'reload_until', 'recoil_pitch', 'recoil_yaw', 'shots', 'last_shot',
               'item_id', 'aim_punch', 'inaccuracy', 'trigger_state', 'reload_state')


def _weapon(w, tick=None):
    if not w:
        return None
    o = {k: w.get(k) for k in WEAPON_KEYS if k in w}
    for k in ('ammo', 'reserve', 'shot_ready', 'shots'):
        o[k] = _int(o.get(k))
    for k in ('recoil_pitch', 'recoil_yaw', 'inaccuracy'):
        o[k] = _num(o.get(k))
    for k in ('reload_until', 'last_shot'):
        o[k] = None if o.get(k) is None else int(o[k])
    o['item_id'] = str(o.get('item_id') or '')
    for k in ('aim_punch', 'trigger_state'):
        o[k] = dict(o[k]) if isinstance(o.get(k), dict) else {}
    o['reload_state'] = dict(o['reload_state']) if isinstance(o.get('reload_state'), dict) else None
    return o


def build_me(view, seat):
    """me block, key order and types as in the platform observation (checked field by field on 5e5 own observations)."""
    p = view['players'][seat]
    tick = int(view['match_tick'])
    b = view.get('bomb') or {}
    slots = p.get('slots') or {}
    bl = p.get('blind_until') or 0
    me = dict(
        x=_num(p['x'], None), y=_num(p['y'], None), z=_num(p['z'], None), yaw=_num(p['yaw'], None), pitch=_num(p['pitch'], None),
        hp=_clamp(_num(p.get('hp')), 0.0, 100.0), armor=_clamp(_num(p.get('armor')), 0.0, 100.0),
        helmet=bool(p.get('helmet')), cash=_clamp(_int(p.get('cash')), 0, 16000), alive=bool(p.get('alive')),
        crouch=bool(p.get('crouch')), vx=_num(p.get('vx')), vy=_num(p.get('vy')), vz=_num(p.get('vz')),
        ammo=max(0, _int(p.get('ammo'))), reload=_num(p.get('reload')), recoil=_num(p.get('recoil')),
        stamina=_clamp(_num(p.get('stamina'), 100.0), 0.0, 100.0), lean_offset=_num(p.get('lean_offset')),
        active_slot=_int(p.get('active_slot'), 1), ads_progress=_clamp(_num(p.get('ads_progress')), 0.0, 1.0),
        sprinting=bool(p.get('sprinting')), grounded=bool(p.get('grounded', True)),
        slots=dict(primary=slots.get('primary'), secondary=slots.get('secondary'), knife=slots.get('knife'),
                   kit=slots.get('kit')),
        throwables=[str(t) for t in p.get('throwables') or []],
        weapons=[_weapon(w, tick) for w in (list(p.get('weapons') or []) + [None, None, None])[:3]],
        pickup_ready_tick=max(0, _int(p.get('pickup_ready_tick'))), switch_until=max(0, _int(p.get('switch_until'))),
        throw_ready=max(0, _int(p.get('throw_ready'))),
        throwable_kinds=throwable_kinds(p),
        blind_ticks=max(0, _int(bl) - tick),
        weapon_status=None,                    # null in all 5e5 recorded observations
        carrying_bomb=b.get('carrier') == seat and b.get('state') == 'held',
        has_kit=bool(slots.get('kit')),
        walk=bool(p.get('walk')), quiet=bool(p.get('quiet')), duck=bool(p.get('duck')),
        zoom_level=_int(p.get('zoom_level')),
        ladder_id=p.get('ladder_id'), movement_mode=p.get('movement_mode') or 'ground',
        shot_ready=_int(p.get('shot_ready')),
    )
    if me['movement_mode'] not in ('ground', 'air', 'ladder', 'ladder_exit'):
        me['movement_mode'] = 'ground'
    return me


def _enemy_entry(q, seat_q, parts):
    """visible[] entry: exactly seat/team/x/y/z/yaw/pitch/crouch/lean/parts/weapon; parts are listed alphabetically
    (abdomen, chest, head) as on the platform."""
    w = None
    ws = q.get('weapons') or []
    a = q.get('active_slot')
    if a is not None and 0 <= a < len(ws) and ws[a]:
        w = ws[a].get('id')
    w = q.get('weapon_id', w)
    return dict(seat=seat_q, team=seat_q // 5, x=_num(q['x'], None), y=_num(q['y'], None), z=_num(q['z'], None), yaw=_num(q['yaw'], None),
                pitch=_num(q['pitch'], None), crouch=bool(q.get('crouch')), lean=_num(q.get('lean_offset')),
                parts=sorted(parts or ['chest']), weapon=w)


def _messages(v, seat, team):
    """messages for `seat`: [{seat, match_tick, signal}] of the teammates (never the own seat).  View keys: `messages_by_seat`
    {seat: list} (preferred) or `messages` {team: list of all five seats' entries}."""
    mb = v.get('messages_by_seat')
    entries = mb.get(seat, ()) if mb is not None else (v.get('messages') or {}).get(team, ())
    return [dict(seat=m['seat'], match_tick=m['match_tick'], signal=m['signal']) for m in entries
            if type(m.get('seat')) is int and 0 <= m['seat'] < 10 and m['seat'] != seat
            and m['seat'] // 5 == team and 'match_tick' in m and 'signal' in m]


def simulated_static_hash(static):
    """Local content identity, explicitly NOT a claimed native descriptor hash."""
    data = json.dumps(policy_map(static), sort_keys=True, separators=(',', ':'), allow_nan=False).encode()
    return 'fpssim-legal-map-sha256:' + hashlib.sha256(data).hexdigest()


def build_observation(arena_view, seat, visible=None, first=False):
    """Legal observation dict for `seat` (key order and key set identical to the platform's recorded observation).
    arena_view: ArenaView dict (see module doc) or the Arena itself.   seat: int or an object with .seat.
    visible: [(enemy_seat, actual_visible_parts)]; absent or invalid entries never use a public mask.
    first: include the static `map` (first callback of the match)."""
    if not isinstance(arena_view, dict):
        arena_view = view_from_arena(arena_view)
    if not isinstance(seat, int):
        seat = seat.seat
    v = arena_view
    P = v['players']
    me_p = P[seat]
    team = seat // 5
    tick = int(v['match_tick'])
    static = v.get('static') or {}
    los = v.get('los')
    can_see = bool(me_p.get('alive', True) and not me_p.get('blind', False)
                   and _int(me_p.get('blind_until')) <= tick)
    actual_visible = []
    seen = set()
    for s, parts in (visible or []) if can_see else []:
        if type(s) is not int or not 0 <= s < len(P) or s // 5 == team or s in seen or not P[s].get('alive', True):
            continue
        parts = [p for p in (parts or []) if p in ('head', 'chest', 'abdomen')]
        if parts:
            actual_visible.append((s, parts)); seen.add(s)
    visible = actual_visible
    pv = None
    if los is not None:
        model = VisModel(los, [(s['position'][0], s['position'][1], s['position'][2], s.get('radius', 4.5))
                               for s in v.get('smokes') or [] if s.get('position')],
                         smoke_fn=v.get('smoke_fn'), tick=tick)
        pv = lambda s, q: can_see and s == seat and model.point_visible(P[s], q)
    # Only this observer's confirmed enemy entries authorize a visible carrier.
    vis = [0] * len(P)
    vis[seat] = (1 << seat) | sum(1 << s for s, _ in visible)
    alive = [sum(1 for s in range(t * 5, t * 5 + 5) if P[s].get('alive')) for t in (0, 1)]
    ll = v.get('loss_level') or [1, 1]
    o = dict(
        rules_version=RULES_VERSION, match_tick=tick, round_index=int(v['round_index']),
        round_tick=max(0, int(v.get('round_tick', 0))), phase=v['phase'] if v['phase'] in ('buy', 'live', 'post', 'finished') else 'buy',
        phase_remaining_ticks=max(0, int(round((v.get('phase_end_tick') or tick) - tick))),
        seat=seat, team=team, side=_side(v, seat), role=0,
        round_wins=[int(x) for x in v.get('round_wins', [0, 0])], alive_by_team=alive,
        me=build_me(v, seat),
        teammates=[dict(seat=s, alive=bool(P[s].get('alive')), cash=max(0, _int(P[s].get('cash'))),
                        zone=_zone(static, P[s]), equipment=_equipment(P[s]))
                   for s in range(team * 5, team * 5 + 5) if s != seat],
        visible=[_enemy_entry(P[s], s, parts) for s, parts in visible],
        heard=[],
        bomb=bomb_view(v, seat, vis, pv),
        world_items=visible_world_items(v, seat, pv) if can_see else [],
        map_id=static.get('id') or v.get('map_id', ''),
        static_schema_hash=(v['static_schema_hash'] if v.get('_is_sim_arena_view') is _SIM_VIEW_TOKEN
                            else simulated_static_hash(static)),
    )
    if first and static:
        o['map'] = copy.deepcopy(policy_map(static))
    o.update(
        messages=_messages(v, seat, team),
        receipts=list((v.get('receipts') or {}).get(seat, [])),
        attack_team=int(v['attack_team']),
        # current-contract top-level additions (doc section 5)
        period=v.get('period', 'regulation'), overtime_number=int(v.get('overtime_number', 0)),
        win_target=int(v.get('win_target', 13)), buy_deadline_tick=int(v.get('buy_deadline_tick', 1860)),
        loss_level=int(ll[team]),
        sounds=sounds_for(v, seat, los),
    )
    return o


# ======================================================================================= Arena adapter
GRENADE_KINDS = ('hegrenade', 'flashbang', 'smokegrenade', 'molotov', 'incgrenade', 'decoy')
GMAP = {'smokegrenade': 'smoke', 'hegrenade': 'frag', 'flashbang': 'flash'}
_ARENA_STATE = weakref.WeakKeyDictionary()      # arena -> bookkeeping (event mark, throwable handles)
_VIEW_CACHE = weakref.WeakKeyDictionary()
_SIM_VIEW_TOKEN = object()
SYNTH_BASE = 10 ** 6               # handle numbers used when the Arena has no item_create event for a throwable


def _arena_state(arena):
    """Per-arena bookkeeping that the Arena itself does not keep: (a) how far in arena.events the previous observation read,
    (b) the per-throwable handles.  The Arena stores grenades as per-kind counters; the platform lists them as handles
    ('item:N', ascending = acquisition order) next to a parallel kind list, so the handles are reconstructed from the
    `item_create` events and reconciled with the counters (throws / deaths remove the oldest handle of the kind)."""
    st = _ARENA_STATE.get(arena)
    if (st is None or arena.tick < st['tick'] or len(arena.events) < st['mark']
            or arena.events is not st['events'] or arena.players is not st['players']
            or arena.static is not st['static']):
        st = dict(tick=-1, mark=len(arena.events), ledger={s: [] for s in range(10)}, created={},
                  synth=SYNTH_BASE, events=arena.events, players=arena.players, static=arena.static,
                  sim_hash=simulated_static_hash(arena.static), receipt_events=[])
        _ARENA_STATE[arena] = st
    return st


def _sync_throwables(arena, st, new_events):
    for e in new_events:
        if e['type'] == 'item_create' and e.get('kind') in GRENADE_KINDS and e.get('seat') is not None:
            st['created'].setdefault((e['seat'], e['kind']), []).append(e['item_id'])
    for p in arena.players:
        want = {k: int(n) for k, n in (getattr(p, 'nades', None) or {}).items() if k in GRENADE_KINDS and n > 0}
        led = st['ledger'][p.seat]
        for k in {k for _, k in led} | set(want):
            have = [i for i, (_, kk) in enumerate(led) if kk == k]
            n = want.get(k, 0)
            for i in reversed(have[:max(0, len(have) - n)]):
                del led[i]                                  # oldest handles of that kind leave first
            for _ in range(max(0, n - len(have))):
                q = st['created'].get((p.seat, k))
                if q:
                    h = q.pop(0)
                else:
                    st['synth'] += 1
                    h = 'item:%d' % st['synth']
                led.append((h, k))
        led.sort(key=lambda hk: _serial(hk[0]))
    return st['ledger']


def _serial(h):
    try:
        return int(str(h).split(':')[1])
    except (IndexError, ValueError):
        return 10 ** 9


def _player_dict(arena, p, led=None):
    ps = p.ps
    ws = p.ws
    w = ws.weapon() if hasattr(ws, 'weapon') else None
    tick = arena.tick
    nades = getattr(p, 'nades', {}) or {}
    gr = {'smoke': 0, 'frag': 0, 'flash': 0}
    for k, n in nades.items():
        if k in GMAP:
            gr[GMAP[k]] += n
    if led is None:
        led = [('item:%d' % (SYNTH_BASE + i), k) for i, (k, n) in enumerate(nades.items()) for _ in range(int(n))]
    slots = [s for s in ws.slots]
    return dict(
        seat=p.seat, team=p.team, x=ps.x, y=ps.y, z=ps.z, yaw=ps.yaw, pitch=ps.pitch, vx=ps.vx, vy=ps.vy, vz=ps.vz,
        hp=_num(p.hp), armor=_num(p.armor), helmet=p.helmet, cash=p.cash, alive=p.alive, crouch=ps.crouch,
        grounded=ps.grounded, stamina=getattr(ps, 'stamina', 100.0), lean_offset=getattr(ps, 'lean_offset', 0.0),
        active_slot=ws.active, ads_progress=getattr(ws, 'ads_progress', 0.0), sprinting=getattr(ps, 'sprinting', False),
        slots=dict(primary=slots[0].item_id if slots[0] else None, secondary=slots[1].item_id if slots[1] else None,
                   knife=slots[2].item_id if slots[2] else None, kit=getattr(p, 'kit_item', None) if p.kit else None),
        throwables=[h for h, _ in led], throwable_kinds=[k for _, k in led], grenades=gr,
        weapons=[s.state(tick) if s else None for s in slots],
        pickup_ready_tick=p.pickup_ready_tick, switch_until=getattr(ws, 'switch_until', 0),
        throw_ready=getattr(p, 'throw_ready', 0), blind_until=getattr(p, 'blind_until', 0),
        ammo=(w.ammo if w else 0), reload=getattr(w, 'reload_progress', 0.0) if w else 0.0, recoil=0.0,
        walk=bool(p.held.get('walk')), quiet=bool(p.held.get('walk')), duck=bool(p.held.get('crouch') or p.held.get('duck')),
        zoom_level=getattr(ws, 'zoom_level', 0), shot_ready=(w.shot_ready if w else 0),
        movement_mode=ps.movement_mode, ladder_id=ps.ladder_id, climb_velocity=getattr(ps, 'climb_velocity', 0.0),
        reattach_until=ps.reattach_until, exit_target=getattr(ps, 'exit_target', None), stand_blocked=ps.stand_blocked,
        weapon_id=(w.id if w else None), height=ps.height, step_distance=ps.step_distance,
    )


# ---- receipts: Arena p.receipts (+ ledger events of the decision tick) -> platform receipt dicts
LEDGER_EVENTS = ('item_create', 'cash', 'buy', 'refund', 'item_destroy', 'drop')
RECEIPT_REASON = {None: 'ok', 'insufficient_cash': 'cash', 'buy_unavailable': 'unavailable', 'refund_unavailable': 'unavailable'}
TERMINAL = {'buy': 'buy', 'refund': 'refund', 'drop': 'drop', 'pickup': 'pickup'}


def _platform_event(e):
    d = dict(type=e['type'], tick=e['match_tick'] + 1, round_index=e['round_index'])     # platform stamps the tick after the decision
    d.update({k: v for k, v in e.items() if k not in ('type', 'event_id', 'match_tick', 'round_index')})
    return d


def platform_receipts(arena, p, events):
    """Receipts of seat p's action at the previous decision in the platform format
    {action, ok, reason, events[, item_id[, price]]}.  `events`: arena events emitted since the previous observation.
    Observed reasons: ok / cash (insufficient money) / unavailable (buy outside the window or zone, failed pickup); other
    refusal reasons of the Arena (wrong_side, already_owned, limit, ...) keep the Arena's wording (platform wording unknown)."""
    mine = [e for e in events if e['type'] in LEDGER_EVENTS and e.get('seat') == p.seat and e['match_tick'] == arena.tick - 3]
    pick = [e for e in events if e['type'] == 'pickup' and e.get('player') == p.seat and e['match_tick'] == arena.tick - 3]
    out = []
    for r in p.receipts:
        act, ok = r.get('action'), bool(r.get('ok'))
        reason = RECEIPT_REASON.get(r.get('reason'), r.get('reason'))
        if act == 'pickup' and not ok:
            reason = 'unavailable'
        evs = []
        if ok:
            term = TERMINAL.get(act)
            if act == 'pickup':
                evs = [dict(type='pickup', tick=e['match_tick'] + 1, round_index=e['round_index'],
                            transaction_id='economy:pickup%s' % str(e.get('event_id', '')).rsplit(':', 1)[-1],   # synthetic: the Arena books no ledger id
                            item_id=e.get('item_id'), seat=p.seat) for e in pick[:1]]
                pick = pick[1:]
            else:
                k = next((i for i, e in enumerate(mine) if e['type'] == term), None)
                if k is not None:
                    evs, mine = [_platform_event(e) for e in mine[:k + 1]], mine[k + 1:]
        rec = dict(action=act, ok=ok, reason=reason, events=evs)
        if ok and act in ('buy', 'refund', 'pickup'):
            rec['item_id'] = r.get('item_id')
            if act in ('buy', 'refund'):
                price = next((e.get('price') for e in evs if e['type'] in ('buy', 'refund')), None)
                if price is not None:
                    rec['price'] = price
        out.append(rec)
    return out


def team_messages(arena):
    """Per-seat `messages`: the teammates' `signal` of the previous decision, one entry per teammate in seat order (also for dead
    or silent ones, signal 0; a teammate who was already dead at that decision always reads 0), match_tick = the tick the signal
    was sent (now - 3).  The first observation has none."""
    if arena.tick <= 0:
        return {p.seat: [] for p in arena.players}
    out = {}
    for p in arena.players:
        lst = []
        for q in arena.players:
            if q.team == p.team and q.seat != p.seat:
                try:
                    sig = int((q.held or {}).get('signal', 0) or 0)
                except (TypeError, ValueError):
                    sig = 0
                dt = getattr(q, 'death_tick', None)
                if not q.alive and not (dt is not None and dt > arena.tick - 3):
                    sig = 0                  # 0 of 2.1e5 entries of dead teammates carry a signal (8.3e3 of the living ones do)
                lst.append(dict(seat=q.seat, match_tick=arena.tick - 3, signal=sig))
        out[p.seat] = lst
    return out




def _phase_end(arena):
    for k in ('phase_end_tick',):
        v = getattr(arena, k, None)
        if v is not None:
            return v
    ph = arena.phase
    if ph == 'buy':
        return getattr(arena, 'round_start_tick', 0) + 360
    if ph == 'live':
        b = arena.bomb or {}
        return b.get('fuse_end') or getattr(arena, 'live_end', arena.tick)
    if ph == 'post':
        return getattr(arena, 'post_end', None) or arena.tick
    return arena.tick


def view_from_arena(arena):
    """ArenaView from the coordinator's Arena (best effort; attribute names follow fpssim/arena.py as of this writing).
    Cache is per live arena episode and decision, not transferable between arenas."""
    st = _arena_state(arena)
    key = (arena.tick, getattr(arena, 'event_serial', 0), len(arena.events))
    c = _VIEW_CACHE.get(arena)
    if c is not None and c[0] is st and c[1] == key:
        return c[2]
    new_events = arena.events[st['mark']:]
    ledger = _sync_throwables(arena, st, new_events)
    # All ten seats reuse a view, while same-tick refreshes never consume earlier
    # receipt events. Across ticks the new causal window replaces the old one.
    if st['tick'] == arena.tick:
        st['receipt_events'].extend(new_events)
    else:
        st['receipt_events'] = new_events
    b = arena.bomb or {}
    objective = dict(planter=None, plant_progress=None, defuser=None,
                     defuse_progress=None, defuse_duration=None)
    actor = b.get('actor')
    if type(actor) is int and 0 <= actor < len(arena.players):
        p = arena.players[actor]
        if p.alive and p.objective == 'plant' and b.get('state') == 'held' and b.get('carrier') == actor:
            objective.update(planter=actor, plant_progress=p.obj_progress)
        elif p.alive and p.objective == 'defuse' and b.get('state') == 'planted':
            objective.update(defuser=actor, defuse_progress=p.obj_progress, defuse_duration=b.get('duration'))
    v = dict(
        match_tick=arena.tick, round_tick=arena.tick - getattr(arena, 'round_start_tick', 0),
        round_index=max(0, arena.round_index), phase=arena.phase, phase_end_tick=_phase_end(arena),
        attack_team=arena.attack_team, round_wins=list(arena.round_wins), fuse_end_tick=b.get('fuse_end'),
        period=getattr(arena, 'period', 'regulation'), overtime_number=getattr(arena, 'overtime_number', 0),
        win_target=getattr(arena, 'win_target', 13), buy_deadline_tick=getattr(arena, 'buy_deadline_tick', 1860),
        loss_level=list(getattr(arena, 'loss_level', [1, 1])),
        players=[_player_dict(arena, p, ledger[p.seat]) for p in arena.players],
        bomb=dict(id=b.get('id'), state=b.get('state'), carrier=b.get('carrier'), site=b.get('site'),
                  position=list(b['position']) if b.get('position') is not None else None, **objective),
        world_items=[dict(id=i['id'], kind=i['kind'], position=list(i['position']))
                     for i in (getattr(arena, 'world_items', None) or []) if i.get('position') is not None],
        smokes=[dict(position=list(s['position']), radius=s.get('radius', 4.5))
                for s in (getattr(arena, 'smokes_view', None) or []) if s.get('position') is not None],
        audio_events=list(getattr(arena, 'audio_events', None) or []),
        receipts={p.seat: (platform_receipts(arena, p, st['receipt_events']) if st['tick'] >= 0 else []) for p in arena.players},
        messages_by_seat=team_messages(arena),
        static=arena.static, static_schema_hash=st['sim_hash'], _is_sim_arena_view=_SIM_VIEW_TOKEN,
    )
    # Do not keep the weak dictionary's key alive in a value (Arena's bound LOS
    # and GrenadeSystem.damage_cb both reach back to the arena).
    owner = weakref.ref(arena)
    v['los'] = lambda a, b: owner().los(a, b)
    if hasattr(getattr(arena, 'grenades', None), 'smoke_blocks'):
        v['smoke_fn'] = lambda a, b, t: owner().grenades.smoke_blocks(a, b, t)
    st['mark'] = len(arena.events)
    st['tick'] = arena.tick
    _VIEW_CACHE[arena] = (st, key, v)
    return v


# ======================================================================================= audio emission
class AudioEmitter:
    """Platform-style audio_events from successive player states (fitted on the public frames; docs/obs.md).
    Call step(prev_players, players, events, tick) once per physics tick; collect .flush() at each decision."""
    FOOT_INTERVAL = 1.85        # cs2-web CS2_STEP_INTERVAL; platform step_distance cadence (validate/obs.py --audio)
    RUN_SPEED = 2.5             # below -> silent (cs2-web CS2_RUN_SPEED_THRESHOLD)
    SOFT_LAND, HARD_LAND = -1.5, -5.2

    def __init__(self):
        self.acc = {}
        self.buf = []
        self.last_step = {}

    def _ev(self, kind, pos, src, radius, level, curve='power', exponent=1.4, local_only=False, remote_only=False,
            directional=True, occludable=True, target_height=0.5, horizontal=False):
        self.buf.append(dict(kind=kind, position=list(pos) if pos is not None else None, source=src, curve=curve,
                             level=level, radius=radius, exponent=exponent, slope=0.0, local_only=local_only,
                             remote_only=remote_only, directional=directional, horizontal=horizontal,
                             occludable=occludable, target_height=target_height))

    def step(self, prev, cur, events=(), tick=0):
        for s, (a, b) in enumerate(zip(prev, cur)):
            if not b.get('alive'):
                continue
            pos = (b['x'], b['y'], b['z'])
            # cs2-web client/footstep-system.js remote rule, run per 60 Hz tick: accumulate the horizontal step distance while
            # grounded, not crouching/walking and 2.5 <= speed <= 12 m/s; emit when acc >= 1.85 and keep acc % 1.85; reset the
            # accumulator on crouch/walk or speed < 1 m/s, and on teleports (> 2.5 m in one tick). Fit on public frames
            # (tools/audio_fit.py): sampled-tick F1 0.67-0.76, recorded steps within +-1 tick 0.84-0.89 (rest = 24 Hz interpolation).
            sd = b.get('step_distance', 0.0)
            move = sd - self.last_step.get(s, sd)
            self.last_step[s] = sd
            speed = math.hypot(b.get('vx', 0.0), b.get('vy', 0.0))
            quiet = b.get('crouch') or b.get('walk')
            acc = self.acc.get(s, 0.0)
            if move < 0 or move > 2.5:
                acc = 0.0
            elif b.get('grounded') and not quiet and self.RUN_SPEED <= speed <= 12.0:
                acc += move
                if acc >= self.FOOT_INTERVAL:
                    acc %= self.FOOT_INTERVAL
                    self._ev('footstep', pos, s, 25.0, 0.28, exponent=1.5, remote_only=True, target_height=0.9)
                    self._ev('footstep', pos, s, 25.0, 0.28, exponent=1.5, local_only=True, directional=False,
                             occludable=False, target_height=0.0)
            elif quiet or speed < 1.0:
                acc = 0.0
            self.acc[s] = acc
            if a.get('grounded') and not b.get('grounded') and b.get('vz', 0.0) > 1.5:
                self._ev('jump', pos, s, 16.0, 0.18, exponent=1.6, occludable=False, target_height=0.0)
            if not a.get('grounded') and b.get('grounded') and a.get('vz', 0.0) <= self.SOFT_LAND:
                hard = a.get('vz', 0.0) <= self.HARD_LAND
                self._ev('land_hard' if hard else 'land_soft', pos, s, 30.0 if hard else 25.0, 0.3825 if hard else 0.45)
        glob = {'plant_start': 'bomb_plant', 'plant_complete': 'bomb_planted', 'defuse_start': 'bomb_defuse_start',
                'defuse_complete': 'bomb_defused', 'bomb_explode': 'bomb_exploded', 'throw': 'grenade_thrown'}
        for e in events:
            t = e.get('type')
            if t == 'fire' and e.get('pellet', 0) == 0:
                p = cur[e['player']]
                self._ev('gunshot', (p['x'], p['y'], p['z']), e['player'], None, 0.7, curve='shot', exponent=1.0,
                         occludable=False, target_height=0.0, horizontal=True)
            elif t == 'draw':
                p = cur[e['player']]
                self._ev('draw', (p['x'], p['y'], p['z']), e['player'], None, 0.68, curve='constant', local_only=True,
                         directional=False, occludable=False, target_height=0.0)
            elif t == 'reload_start':
                p = cur[e['player']]
                self._ev('reload', (p['x'], p['y'], p['z']), e['player'], None, 0.66, curve='constant', local_only=True,
                         directional=False, occludable=False, target_height=0.0)
            elif t == 'bounce':
                self._ev('grenade_bounce', e.get('position'), None, None, 0.5, curve='inverse', exponent=1.0,
                         directional=False, occludable=False, target_height=0.0)
            elif t in glob:
                self._ev(glob[t], None, None, None, 0.65, curve='constant', directional=False, occludable=False,
                         target_height=0.0)

    def bomb_beep(self, pos, fast=False):
        self._ev('bomb_beep_fast' if fast else 'bomb_beep', pos, None, None, 0.65, curve='inverse', exponent=1.0,
                 directional=False, occludable=False, target_height=0.0)

    def flush(self):
        out, self.buf = self.buf, []
        return out
