"""Weapon state machine: fire timing, ammo, magazine / shell reload, slot switch, zoom, knife cadence.

All times are integer physics ticks (60 Hz).  Rules are the cs2-web (commit e0e7e858) rules converted to ticks the
way the platform does it, validated tick-by-tick against recordings by validate/weapons_full.py (docs/weapons_full.md):

* fire:      allowed if t >= shot_ready, t >= switch_until, not reloading (shell reload with ammo > 0 is interrupted),
             ammo > 0, and (automatic or rising trigger edge).  shot_ready = t + ticks(fireInterval), last_shot = t,
             shots += 1, ammo -= 1.  Recoil/inaccuracy kick (fpssim.recoil, validated) is applied after the shot's
             direction inputs are captured in `punch_at_shot`.
* switch:    draw event at t, switch_until = t + 11 (cs2-web 180 ms; every observed weapon incl. awp/ssg08/knife).
             Switching cancels a running reload (ammo already committed stays committed) and resets zoom.
* reload:    reload_start at t: reload_until = t + ticks(reloadTime);
             commit (magazine enters) = t + round(frac * reloadTime * 60) where frac is the cs2-web reload-profile
             commitFraction (shared/reload-profiles.js), using the 'reloadEmpty' profile when the magazine is empty;
             an empty reload of a weapon without a reloadEmpty profile uses frac = 0.5 (platform rule, 247/247 cases).
             Transfer at commit: standard (ammo += min(mag - ammo, reserve)); cs2-web's clip-discarding rule is NOT used.
             reload_until clears at the finish tick.  Shell guns (nova/xm1014/sawedoff, UNVERIFIED, none in recordings):
             one shell per ticks(reloadTime), reload_until = next shell tick, firing with ammo > 0 interrupts.
* zoom:      zoom_level follows the requested level every tick (clamped to the weapon's zoomFovs), forced 0 while
             reloading; unzoomsAfterShot weapons (awp, ssg08) drop to 0 right after the shot and re-zoom on the next
             tick if the request is still held (cs2-web semantics; recordings show zoom held through AWP shots).
             ads_progress = 1.0 if zoomed else 0.0 (every recorded zoomed frame has ads 1.0, fov = zoomFovs[level]).
* knife:     light swing (fire) interval 0.5 s = 30 ticks, heavy (fire2) 1.0 s = 60 ticks; `knife_first` = more than
             60 ticks since this player's previous swing (cs2-web 1000 ms rule).  UNVERIFIED (no knife hits recorded).

Public interface kept for Arena: Weapon(wid, item_id), Weapon.state(tick), Weapon.punch_at_shot,
WeaponSystem.slots/active/switch_until/zoom_level, WeaponSystem.tick(...) -> list of (kind, Weapon) with kind in
shot / knife / draw / reload_start / reload_commit / reload_finish / reload_cancel.
New (backwards compatible): tick(..., fire2=False), cancel_reload(t) (Arena must call it on plant_start / defuse_start), WeaponSystem.knife_heavy / knife_first (valid after a 'knife'
output), WeaponSystem.zoom_fov / ads_progress (read-only properties), Weapon.from_state(dict), WeaponSystem.from_frame(player_dict),
and event-driven helpers do_switch / do_reload / do_fire / advance used by validate/weapons_full.py.
"""
import json, math, os
from .common import WEAPONS, HERE
from . import recoil as R

FIT = json.load(open(os.path.join(HERE, 'weapon_fit.json')))     # legacy fitted values (spread model used by combat)

TICK = 1.0 / 60.0
DRAW_TICKS = 11                       # switch_until = draw tick + 11 (validated, all weapons)
DRAW_TICKS_DEFAULT = DRAW_TICKS
KNIFE_FIRST_GAP = 60                  # cs2-web: first slash if > 1000 ms since previous swing
EMPTY_COMMIT_FRAC_DEFAULT = 0.5       # platform: empty reload without a reloadEmpty profile

# cs2-web shared/reload-profiles.js commitFraction: (reload, reloadEmpty or None)
RELOAD_COMMIT = {
    'ak47': (0.452055, None), 'm4a1': (0.445652, None), 'awp': (0.545455, None), 'pistol': (0.411765, 0.558824),
    'usp': (0.415385, 0.461538), 'elite': (0.778761, None), 'p250': (0.411765, 0.529412),
    'fiveseven': (0.426471, 0.529412), 'deagle': (0.348485, 0.469697), 'nova': (0.408163, None),
    'mag7': (0.432432, None), 'mp9': (0.40625, None), 'mp7': (0.457447, None), 'bizon': (0.465753, None),
    'scar20': (0.5, 0.478261), 'm4a4': (0.445652, None), 'ssg08': (0.540541, None), 'tec9': (0.558442, None),
    'xm1014': (0.596154, None), 'sawedoff': (0.384615, None), 'mac10': (0.467532, None), 'galilar': (0.384615, None),
    'sg553': (0.373494, None)}


def ticks(sec):
    """seconds -> ticks, ceil with float fuzz (10.2 -> 11, 9.0 -> 9; 2.266667 s -> 137 as on the platform)."""
    return int(math.ceil(sec * 60.0 - 1e-6))


def reload_commit_ticks(wid, empty):
    """Ticks from reload start until the magazine is transferred (exact on all 343 recorded reloads)."""
    meta = WEAPONS[wid]
    full, emp = RELOAD_COMMIT.get(wid, (1.0, None))
    frac = (emp if emp is not None else EMPTY_COMMIT_FRAC_DEFAULT) if empty else full
    return int(round(frac * meta['reloadTime'] * 60.0))


def _kick(wid, idx):
    """Legacy (pre-recoil.py) fitted kick table; kept for old tools only."""
    t = FIT['kick_table'].get(wid)
    if t:
        e = t[(idx - 1) % len(t)]
        return e[0], e[1]
    return FIT['kick_fallback_per_recoil'] * WEAPONS[wid]['recoil'], 0.0


class Weapon:
    __slots__ = ('id', 'ammo', 'reserve', 'shot_ready', 'reload_until', 'reload_commit', 'reload_committed', 'reload_shell',
                 'shots', 'last_shot', 'inaccuracy', 'rs', 'punch_at_shot', 'pressed', 'last_press', 'item_id',
                 'last_update_tick', 'since_shot', 'shell_next', 'reload_style')

    def __init__(self, wid, item_id=None):
        w = WEAPONS[wid]
        self.id = wid; self.item_id = item_id
        self.ammo = w['magazine']; self.reserve = w['reserve']
        self.shot_ready = 0; self.reload_until = None; self.reload_commit = None
        self.reload_committed = False; self.reload_shell = False
        self.shots = 0; self.last_shot = None
        self.inaccuracy = 0.0
        self.rs = R.new_state()
        self.pressed = False; self.last_press = None
        self.last_update_tick = 0; self.since_shot = 10 ** 6
        self.shell_next = None
        self.punch_at_shot = (0.0, 0.0, 0.0)
        self.reload_style = w.get('reloadStyle', 'magazine')

    @classmethod
    def from_state(cls, d):
        """Build from a frames / observation weapon dict (me.weapons[i])."""
        w = cls(d['id'], item_id=d.get('item_id'))
        w.ammo = d['ammo']; w.reserve = d['reserve']; w.shot_ready = d.get('shot_ready') or 0
        w.reload_until = d.get('reload_until'); w.shots = d.get('shots') or 0; w.last_shot = d.get('last_shot')
        w.inaccuracy = d.get('inaccuracy') or 0.0
        ap = d.get('aim_punch') or {}
        for k in w.rs: w.rs[k] = float(ap.get(k, 0.0) or 0.0)
        ts = d.get('trigger_state') or {}
        w.pressed = bool(ts.get('pressed')); w.last_press = ts.get('last_tick')
        rs = d.get('reload_state')
        if rs:
            w.reload_commit = rs.get('commit'); w.reload_committed = bool(rs.get('committed'))
            w.reload_shell = bool(rs.get('shell'))
            if w.reload_shell: w.shell_next = rs.get('finish', w.reload_until)
        return w

    # ---- per tick passive dynamics (cs2-web recoil-tables.js, per 60 Hz tick; validate/recoil.py: 99.85% tick-exact)
    def decay(self, crouch=False, in_air=False):
        rs = self.rs
        R.tick(rs)
        rt = R.recovery_time(self.id, crouch=crouch, in_air=in_air, recoil_index=rs['index'])
        self.inaccuracy = R.decay_penalty(self.inaccuracy, 1 / 60, rt)
        self.since_shot += 1

    def kick(self):
        R.kick(self.rs, self.id)
        self.inaccuracy = (self.inaccuracy or 0.0) + R.inaccuracy_fire(self.id)

    ap_pitch = property(lambda self: self.rs['pitch'])
    ap_yaw = property(lambda self: self.rs['yaw'])

    def reloading(self):
        return self.reload_until is not None

    def reload_state(self):
        if self.reload_until is None: return None
        if self.reload_shell:
            return {'commit': self.reload_until, 'finish': self.reload_until, 'committed': False, 'shell': True}
        return {'commit': self.reload_commit, 'finish': self.reload_until, 'committed': bool(self.reload_committed), 'shell': False}

    def state(self, tick):
        """dict in the shape of observation me.weapons[i] / frames weapons[i]."""
        ts = {} if self.id == 'knife' and self.last_press is None else {'pressed': bool(self.pressed)}
        if self.last_press is not None: ts['last_tick'] = self.last_press
        return dict(id=self.id, ammo=self.ammo, reserve=self.reserve, shot_ready=self.shot_ready,
                    reload_until=self.reload_until, recoil_pitch=self.rs['pitch'], recoil_yaw=self.rs['yaw'],
                    shots=self.shots, last_shot=self.last_shot, item_id=self.item_id,
                    aim_punch=dict(self.rs), inaccuracy=self.inaccuracy, trigger_state=ts,
                    reload_state=self.reload_state())


def spread_for(wid, crouch, speed, grounded, scoped):
    """Legacy catalogue base spread (radians); the exact model is fpssim.accuracy."""
    w = WEAPONS[wid]
    if scoped and w.get('zoomStyle'): return w['scopedSpread']
    return w['crouchingSpread'] if crouch else w['spread']


class WeaponSystem:
    """Per-player weapon inventory logic (three action slots: 0 primary, 1 secondary, 2 knife)."""

    def __init__(self):
        self.slots = [None, None, None]
        self.active = 1
        self.switch_until = 0
        self.switch_target = None
        self.zoom_level = 0
        self.events = []
        self.knife_heavy = False; self.knife_first = True; self.last_knife = None

    @classmethod
    def from_frame(cls, p):
        ws = cls()
        ws.slots = [Weapon.from_state(w) if w else None for w in (p.get('weapons') or [None, None, None])]
        while len(ws.slots) < 3: ws.slots.append(None)
        ws.active = p.get('active_slot', 1) if p.get('active_slot') is not None else 1
        ws.switch_until = p.get('switch_until') or 0
        ws.zoom_level = p.get('zoom_level') or 0
        return ws

    def weapon(self):
        return self.slots[self.active] if self.active is not None else None

    def meta(self):
        w = self.weapon()
        return WEAPONS[w.id] if w is not None else None

    @property
    def zoom_fov(self):
        m = self.meta()
        if m is None: return 90
        fovs = m.get('zoomFovs') or [90]
        return fovs[min(self.zoom_level, len(fovs) - 1)]

    @property
    def ads_progress(self):
        return 1.0 if self.zoom_level > 0 else 0.0

    def max_speed(self):
        """Movement speed cap (m/s): scopedMaxSpeed when zoomed (cs2-web weaponSpeedScale)."""
        m = self.meta()
        if m is None: return 6.0
        return m.get('scopedMaxSpeed', m['maxSpeed']) if self.zoom_level > 0 else m['maxSpeed']

    def set_zoom(self, level):
        m = self.meta()
        if m is None or self.weapon().reload_until is not None:
            self.zoom_level = 0; return
        n = len(m.get('zoomFovs') or [90]) - 1
        self.zoom_level = max(0, min(int(level or 0), n))

    # ------------------------------------------------------------------ event-level primitives
    def advance(self, t, crouch=False, in_air=False):
        """Passive part of tick t: decay every weapon, then reload commit / shell insert / finish. Returns outputs."""
        out = []
        for w in self.slots:
            if w is not None: w.decay(crouch, in_air)
        w = self.weapon()
        if w is None or w.reload_until is None: return out
        meta = WEAPONS[w.id]
        if w.reload_shell:
            while w.reload_until is not None and t >= w.reload_until:
                if w.reserve > 0 and w.ammo < meta['magazine']:
                    w.ammo += 1; w.reserve -= 1; out.append(('reload_commit', w))
                if w.reserve > 0 and w.ammo < meta['magazine']:
                    w.reload_until += ticks(meta['reloadTime'])
                else:
                    w.reload_until = None; w.reload_shell = False; out.append(('reload_finish', w))
            return out
        if not w.reload_committed and w.reload_commit is not None and t >= w.reload_commit:
            self._commit(w); out.append(('reload_commit', w))
        if t >= w.reload_until:
            if not w.reload_committed: self._commit(w); out.append(('reload_commit', w))
            w.reload_until = None; w.reload_commit = None; w.reload_committed = False
            out.append(('reload_finish', w))
        return out

    @staticmethod
    def _commit(w):
        meta = WEAPONS[w.id]
        take = min(meta['magazine'] - w.ammo, w.reserve)
        w.ammo += take; w.reserve -= take; w.reload_committed = True

    def can_switch(self, t, slot):
        return slot is not None and slot != self.active and 0 <= slot < 3 and self.slots[slot] is not None

    def do_switch(self, t, slot):
        out = self.cancel_reload(t)
        nw = self.slots[slot]                            # frames: the drawn weapon's trigger_state resets to {'pressed': False}
        nw.pressed = False; nw.last_press = None
        self.active = slot; self.zoom_level = 0
        self.switch_until = t + DRAW_TICKS
        out.append(('draw', self.weapon()))
        return out

    def cancel_reload(self, t=None):
        """Abort the active weapon's reload (ammo already committed stays).  The platform does this on switch and when
        an objective action starts (plant_start / defuse_start emit reload_cancel at the same tick) -> Arena must call
        it there.  Returns [('reload_cancel', w)] or []."""
        w = self.weapon()
        if w is None or w.reload_until is None: return []
        w.reload_until = None; w.reload_commit = None; w.reload_committed = False; w.reload_shell = False
        return [('reload_cancel', w)]

    def can_reload(self, t):
        w = self.weapon()
        if w is None: return False
        meta = WEAPONS[w.id]
        return w.reload_until is None and meta['magazine'] > 0 and w.ammo < meta['magazine'] and w.reserve > 0 \
            and t >= self.switch_until

    def do_reload(self, t):
        w = self.weapon(); meta = WEAPONS[w.id]
        dur = ticks(meta['reloadTime'])
        self.zoom_level = 0
        w.reload_until = t + dur; w.reload_committed = False
        if meta.get('reloadStyle') == 'shell':
            w.reload_shell = True; w.reload_commit = None
        else:
            w.reload_shell = False; w.reload_commit = t + reload_commit_ticks(w.id, w.ammo == 0)
        return [('reload_start', w)]

    def can_fire(self, t, edge=True, heavy=False):
        w = self.weapon()
        if w is None: return False
        meta = WEAPONS[w.id]
        if t < w.shot_ready or t < self.switch_until: return False
        if w.id == 'knife': return True
        if w.reload_until is not None and not (w.reload_shell and w.ammo > 0): return False
        return w.ammo > 0 and (meta['automatic'] or edge)

    def do_fire(self, t, heavy=False):
        w = self.weapon(); meta = WEAPONS[w.id]
        if w.id == 'knife':
            self.knife_heavy = bool(heavy)
            self.knife_first = self.last_knife is None or t - self.last_knife > KNIFE_FIRST_GAP
            self.last_knife = t; w.last_press = t
            w.shot_ready = t + ticks(1.0 if heavy else 0.5)
            return [('knife', w)]
        out = []
        if w.reload_until is not None:          # shell reload interrupted by a loaded shot
            w.reload_until = None; w.reload_shell = False; w.reload_commit = None
        w.ammo -= 1; w.shots += 1; w.last_shot = t; w.since_shot = 0
        w.last_press = t                         # frames: trigger_state.last_tick == tick of the last shot (survives round reset)
        w.shot_ready = t + ticks(meta['fireInterval'])
        w.punch_at_shot = (w.rs['pitch'], w.rs['yaw'], w.inaccuracy)   # pre-kick values drive this shot
        w.kick()
        out.append(('shot', w))
        if meta.get('unzoomsAfterShot'): self.zoom_level = 0
        return out

    # ------------------------------------------------------------------ input-driven tick (Arena)
    def tick(self, t, fire, reload, want_slot, can_act=True, crouch=False, zoom=None, in_air=False, fire2=False):
        """Advance one tick.  Returns list of (kind, Weapon).  `can_act` False in freeze/dead/objective.
        Order: decay + reload progress -> switch -> zoom -> reload request -> fire."""
        out = self.advance(t, crouch, in_air)
        if can_act and want_slot is not None and self.can_switch(t, want_slot):
            out += self.do_switch(t, want_slot)
        w = self.weapon()
        if w is None: return out
        if zoom is not None: self.set_zoom(zoom)
        if can_act and reload and self.can_reload(t):
            out += self.do_reload(t)
        trig = bool(fire) or (w.id == 'knife' and bool(fire2))
        edge = trig and not w.pressed
        w.pressed = trig
        if can_act and trig and self.can_fire(t, edge):
            out += self.do_fire(t, heavy=(w.id == 'knife' and bool(fire2)))
        return out
