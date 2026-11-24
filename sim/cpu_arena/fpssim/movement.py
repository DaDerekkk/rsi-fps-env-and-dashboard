"""60 Hz player movement (Source-engine style), fitted to authoritative recordings.

Fitted/verified on the public replays (see docs/movement.md):
  ground: friction 5.2/s with stopspeed 2.54 m/s (drop = max(speed,stop)*5.2*dt), accelerate 5.5*maxspeed*dt capped by
          addspeed = wishspeed - v.wishdir, wishspeed = min(|(fwd,right)|,1)*maxspeed*factor (walk 0.45), position += v_new*dt.
  air:    accel 12, wishspeed cap 0.762 m/s (30 u/s) for the addspeed test, gravity 20.32 m/s^2, jump v0 7.6708 m/s,
          z += (vz_old+vz_new)/2*dt.
  turn:   yaw 3.0 rad/s, pitch 2.0 rad/s at |rate|=1 (applied per tick before wish computed).
  hull:   0.3 m half-width square vs world, circles of radius 0.35 between players, height 1.8 / 1.1 (crouch).
Everything is one function call per physics tick; inputs are held between 20 Hz decisions by the caller (Arena).
"""
import math
from dataclasses import dataclass, field
from .common import WEAPONS
from .collision import HW, STAND_H, CROUCH_H, sweep_axis

TWO_PI = 2 * math.pi


@dataclass
class MoveParams:
    dt: float = 1 / 60
    friction: float = 5.2
    stopspeed: float = 2.54
    accel: float = 5.5
    airaccel: float = 12.0
    aircap: float = 0.762
    gravity: float = 20.32
    jump_v: float = 7.6708
    walk_factor: float = 0.45
    crouch_factor: float = 0.4773    # fitted: steady crouch-run 0.358*M at agent input 0.75 (tools/steady_speeds.py)
    crouch_walk_factor: float = 0.344  # steady 0.258*M at input 0.75
    yaw_rate: float = 3.0
    pitch_rate: float = 2.0
    pitch_limit: float = 1.4835   # observed extremes -1.48..1.15 (85 deg)
    step_height: float = 0.45
    player_radius: float = 0.35      # centre distance 0.7 between two players
    knife_speed: float = 6.0         # steady 0.945*6.35 on knife (assumes input 1.0)
    # ladders (provisional; map vertical_rules override)
    ladder_climb_speed: float = 5.0
    ladder_attach_reach: float = 1.4
    reattach_ticks: int = 12
    eye_stand: float = 1.62
    eye_crouch: float = 0.95


def wrap(a):
    return (a + math.pi) % TWO_PI - math.pi


@dataclass
class PlayerState:
    x: float = 0.0; y: float = 0.0; z: float = 0.0
    vx: float = 0.0; vy: float = 0.0; vz: float = 0.0
    yaw: float = 0.0; pitch: float = 0.0
    crouch: bool = False
    grounded: bool = True
    last_jump: bool = False
    movement_mode: str = 'ground'      # ground | air | ladder
    ladder_id: str = None
    reattach_until: int = 0
    stand_blocked: bool = False
    pinned: bool = False             # persistent player-contact pin (see Mover._touching)
    step_distance: float = 0.0
    alive: bool = True

    @property
    def air_phase(self):
        if self.movement_mode == 'ladder': return 'climbing'
        if self.grounded: return 'grounded'
        return 'rising' if self.vz > 0 else 'falling'

    @property
    def height(self):
        return CROUCH_H if self.crouch else STAND_H

    @property
    def speed(self):
        return math.hypot(self.vx, self.vy)


def max_speed(weapon_id, zoom_level=0, params=None):
    w = WEAPONS.get(weapon_id)
    if w is None or 'max_speed' not in w:
        return (params.knife_speed if params else 6.0)
    return w['scoped_max_speed'] if zoom_level else w['max_speed']


def speed_factor(walk, crouch, p):
    if crouch and walk: return p.crouch_walk_factor
    if crouch: return p.crouch_factor
    if walk: return p.walk_factor
    return 1.0


class Mover:
    def __init__(self, params=None):
        self.p = params or MoveParams()

    def step(self, s, a, M, world, others=(), tick=0, objective_locked=False):
        """Advance PlayerState s by one physics tick under held action dict a (platform action contract).
        M = weapon max speed (m/s). others: iterable of (x,y,z) of other alive players for separation."""
        p = self.p; dt = p.dt
        if not s.alive: return s
        get = a.get
        fwd = float(get('move_forward', 0.0)); rgt = float(get('move_right', 0.0))
        jump = bool(get('jump', False))
        want_crouch = bool(get('crouch', False) or get('duck', False))
        walk = bool(get('walk', False))
        if objective_locked:
            fwd = rgt = 0.0; jump = False
            s.vx = s.vy = 0.0
        # --- view ---
        s.yaw = wrap(s.yaw + float(get('yaw_rate', 0.0)) * p.yaw_rate * dt)
        s.pitch = max(-p.pitch_limit, min(p.pitch_limit, s.pitch + float(get('pitch_rate', 0.0)) * p.pitch_rate * dt))
        # --- stance ---
        if s.crouch and not want_crouch:
            ceil = world.ceiling(s.x - HW, s.x + HW, s.y - HW, s.y + HW, s.z + CROUCH_H)
            s.stand_blocked = ceil < s.z + STAND_H - 1e-6
            if not s.stand_blocked: s.crouch = False
        else:
            s.stand_blocked = False
            s.crouch = want_crouch
        if s.movement_mode == 'ladder':
            return self._ladder_step(s, a, world, tick)
        # ladder attach request
        if get('ladder', 'none') == 'attach' and tick >= s.reattach_until:
            if self._try_attach(s, world, tick): return s
        yaw = s.yaw; cy, sy = math.cos(yaw), math.sin(yaw)
        wx = fwd * cy + rgt * sy; wy = -fwd * sy + rgt * cy
        mag = math.hypot(wx, wy)
        fac = speed_factor(walk, s.crouch, p)
        ws = min(mag, 1.0) * M * fac
        if mag > 1e-9: wdx, wdy = wx / mag, wy / mag
        else: wdx = wdy = 0.0
        jumped = False
        if s.grounded:
            sp = math.hypot(s.vx, s.vy)
            if sp > 0:
                drop = max(sp, p.stopspeed) * p.friction * dt
                ns = max(sp - drop, 0.0)
                k = ns / sp; s.vx *= k; s.vy *= k
            add = ws - (s.vx * wdx + s.vy * wdy)
            if add > 0 and ws > 0:
                acc = min(p.accel * dt * M, add)
                s.vx += acc * wdx; s.vy += acc * wdy
            if jump and not s.last_jump:
                s.vz = p.jump_v; s.grounded = False; jumped = True
        else:
            wsc = min(ws, p.aircap)
            add = wsc - (s.vx * wdx + s.vy * wdy)
            if add > 0 and ws > 0:
                acc = min(p.airaccel * dt * ws, add)
                s.vx += acc * wdx; s.vy += acc * wdy
        s.last_jump = jump
        # --- integrate ---
        touching = bool(others) and self._touching(s, others)
        pinned = touching and ((s.grounded and not jumped) or s.pinned)
        s.pinned = pinned
        h = s.height
        step = p.step_height if s.grounded else 0.0
        x0, y0 = s.x, s.y
        zf = s.z
        if not pinned:
            nx, hit = sweep_axis(world, (s.x, s.y), 0, s.vx * dt, zf, zf + h, step)
            if hit: s.vx = 0.0
            s.x = nx
            ny, hit = sweep_axis(world, (s.x, s.y), 1, s.vy * dt, zf, zf + h, step)
            if hit: s.vy = 0.0
            s.y = ny
            self._separate(s, others, x0, y0)
        if s.grounded:
            sup = world.ground(s.x, s.y, s.z, HW, p.step_height)
            if sup >= s.z - p.step_height - 1e-6:
                s.z = sup; s.vz = 0.0
            else:
                s.grounded = False; s.vz = 0.0
                self._air_z(s, world, h)
        else:
            self._air_z(s, world, h)
        s.movement_mode = 'ground' if s.grounded else 'air'
        s.step_distance += math.hypot(s.x - x0, s.y - y0) if s.grounded else 0.0
        return s

    def _air_z(self, s, world, h):
        p = self.p; dt = p.dt
        vz0 = s.vz; vz1 = vz0 - p.gravity * dt
        znew = s.z + (vz0 + vz1) * 0.5 * dt
        if vz1 > 0 or vz0 > 0:
            ceil = world.ceiling(s.x - HW, s.x + HW, s.y - HW, s.y + HW, s.z + h - 1e-4)
            if znew + h > ceil:
                znew = ceil - h; vz1 = 0.0 if vz1 > 0 else vz1
        if vz1 <= 0 or znew < s.z:
            sup = world.ground(s.x, s.y, s.z, HW, 1e-3)   # only tops at/below current feet
            if znew <= sup:
                s.z = sup; s.vz = 0.0; s.grounded = True
                return
        s.z = znew; s.vz = vz1

    def _touching(self, s, others):
        """True if another player (z-overlapping) is within 0.7 m (+1e-4) at the start of the tick."""
        R2 = 2 * self.p.player_radius + 1e-4
        for (ox, oy, oz) in others:
            if abs(oz - s.z) < STAND_H and math.hypot(s.x - ox, s.y - oy) <= R2: return True
        return False

    def _separate(self, s, others, x0, y0):
        """Player-vs-player (fitted on the pinned-with-velocity statistics, 92 % per-block agreement):
        a player that starts the tick touching another one (d <= 0.7) does not move at all (x,y,z frozen, velocity
        and jump state keep evolving); a move that ends inside 0.7 m is pushed out radially onto the 0.7 circle."""
        R2 = 2 * self.p.player_radius
        for (ox, oy, oz) in others:
            if abs(oz - s.z) >= STAND_H: continue
            dx, dy = s.x - ox, s.y - oy
            d = math.hypot(dx, dy)
            if d < R2:
                if d < 1e-9: dx, dy, d = 1.0, 0.0, 1.0
                k = (R2 - d) / d
                s.x += dx * k; s.y += dy * k

    # --- ladders (fitted on borgo/nevaio recordings) ------------------------
    # attach: x,y snap to the descriptor's `rail` point (no offset), climbing starts the same tick;
    # climb: 5.0 m/s up (= 1/12 m per tick exactly), 7.5 m/s down, * climb input (a['climb'] if non-zero else move_forward), z clamped to
    # [bottom, top]; reaching `bottom` exits to bottom_exit on the ground; `ladder: detach` leaves into the air with zero
    # velocity (gravity resumes); after leaving, re-attach is blocked for 12 ticks (reattach_until = tick + 12).
    def _try_attach(self, s, world, tick):
        p = self.p
        vr = getattr(world, 'vertical', {}) or {}
        reach = vr.get('attach_reach', p.ladder_attach_reach)
        best = None
        for l in world.ladders:
            rx, ry = l['rail'][0], l['rail'][1]
            d = math.hypot(s.x - rx, s.y - ry)
            if d <= reach and l['bottom'] - 0.3 <= s.z <= l['top'] + 0.1:
                if best is None or d < best[0]: best = (d, l)
        if best is None: return False
        l = best[1]
        s.x, s.y = l['rail'][0], l['rail'][1]
        s.vx = s.vy = s.vz = 0.0
        s.movement_mode = 'ladder'; s.ladder_id = l['id']; s.grounded = False
        return True

    def _ladder_step(self, s, a, world, tick):
        p = self.p; dt = p.dt
        l = next((q for q in world.ladders if q['id'] == s.ladder_id), None)
        vr = getattr(world, 'vertical', {}) or {}
        cs = vr.get('climb_speed', p.ladder_climb_speed)
        climb = float(a.get('climb', 0.0)) or float(a.get('move_forward', 0.0))
        climb = max(-1.0, min(1.0, climb))
        s.vx = s.vy = 0.0
        if l is None or a.get('ladder') == 'detach':
            s.movement_mode = 'air'; s.ladder_id = None; s.grounded = False; s.vz = 0.0
            s.reattach_until = tick + p.reattach_ticks
            self._air_z(s, world, s.height)
            return s
        s.vz = climb * cs if climb >= 0 else climb * cs * 1.5     # descending is 1.5x faster (7.5 m/s), fitted
        s.z = min(l['top'], s.z + s.vz * dt)
        if s.z <= l['bottom'] + 1e-9:
            ex = l['bottom_exit']
            s.x, s.y, s.z = ex[0], ex[1], l['bottom'] + 0.00011
            s.movement_mode = 'ground'; s.ladder_id = None; s.vz = 0.0; s.grounded = True
            s.reattach_until = tick + p.reattach_ticks
        return s


def eye_z(s, p=None):
    """Eye height used by the observation/visibility code (stand 1.62, crouch 0.95 above the feet)."""
    p = p or MoveParams()
    return s.z + (p.eye_crouch if s.crouch else p.eye_stand)


def step_all(mover, states, actions, speeds, world, tick, locked=None):
    """Advance all players one physics tick.  Player-player contact is evaluated against the positions at the START of the
    tick (simultaneous update).  states: list[PlayerState]; actions: list[dict]; speeds: per-player max speed M;
    locked: per-player objective_locked flags."""
    snap = [(s.x, s.y, s.z, s.alive) for s in states]
    for i, s in enumerate(states):
        if not s.alive: continue
        others = [(x, y, z) for j, (x, y, z, al) in enumerate(snap) if j != i and al]
        mover.step(s, actions[i], speeds[i], world, others, tick, bool(locked[i]) if locked else False)
    return states
