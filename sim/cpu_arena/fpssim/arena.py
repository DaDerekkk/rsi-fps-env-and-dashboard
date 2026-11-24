"""Arena: 5v5 defusal referee (rsi-fps-5.5 contract) -- round/phase machine, economy, inventory, bomb, combat glue.

Clock: global match_tick at 60 Hz; the policy decides every 3 ticks (match_tick % 3 == 0).  `Arena.step(actions)` applies the
held actions during the next 3 ticks (event stamps tick+1..tick+3, exactly as in the public recordings) and returns the
10 legal observations of the next decision boundary.

Structure ported/aligned from cs2-web server/game.js (commit e0e7e858; buy/refund/drop/pickup/bomb logic) with the platform
differences in the rule guide (timings, five-tier loss bonus, objective win 3500, buy window, ledger events).
"""
import hashlib, json, math, os, random
from .common import WEAPONS, EQUIP, UTIL, KNIFE, HERE, MAP_IDS
from .weapons import Weapon, WeaponSystem
from . import combat
from . import movement as mv
from .movement import PlayerState, Mover, MoveParams
from .collision import make_world
from .geometry import MapGeom
from .grenades import GrenadeSystem
from . import accuracy as ACCM
from . import penetration as PEN
from .trimesh import TriMesh
build_observation = None
if os.environ.get('FPS_OBS', '1') == '1':  # fpssim/obs.py observation builder (FPS_OBS=0 -> minimal built-in obs)
    from .obs import build_observation

KIT_DIR = os.environ.get('FPS_KIT_DIR', './fps_data/fps_platform/kit/data')
STATIC_DIR = os.environ.get('FPS_STATIC_DIR', './fps_data/data/static')
ASSET_DIR = os.environ.get('FPS_ASSET_DIR', './fps_data/ref/assets')
MESH_FILES = {'dust2-cs2-v54': ('map_positions.f32', 'map_penetration-materials.u8'),
              'mirage-cs2-v54': ('mirage_positions.f32', 'mirage_penetration-materials.u8')}   # Mirage: Awpy rebuild, byte-identical to the platform mesh
VIS_H, VIS_V = math.radians(55.0), math.radians(45.0)      # validated on Dust2 (tools/vis_rule.py)
VIS_FRACS = (0.9, 0.75, 0.55, 0.3)
# Bomb beeps, ticks since plant_complete: complete deterministic schedule reconstructed from 140 recorded plants
# (tools/audio_beep.py; every offset 0..2400 was sampled at least once, so the table is exact). Fast kind from offset 1800.
BEEP_SLOW = frozenset([0, 60, 119, 177, 233, 288, 341, 393, 444, 493, 541, 588, 634, 679, 723, 765, 806, 846, 885, 923, 960, 996, 1032, 1067, 1101, 1134, 1166, 1197, 1228, 1258, 1287, 1315, 1343, 1370, 1396, 1422, 1447, 1471, 1495, 1518, 1541, 1563, 1584, 1605, 1625, 1645, 1664, 1683, 1701, 1719, 1737, 1754, 1771, 1787])
BEEP_FAST = frozenset([1803, 1818, 1833, 1848, 1862, 1876, 1890, 1903, 1916, 1929, 1941, 1953, 1965, 1976, 1987, 1998, 2009, 2019, 2029, 2039, 2049, 2058, 2067, 2076, 2085, 2093, 2101, 2109, 2117, 2125, 2133, 2141, 2149, 2157, 2165, 2173, 2181, 2189, 2197, 2205, 2213, 2221, 2229, 2237, 2245, 2253, 2261, 2269, 2277, 2285, 2293, 2301, 2309, 2317, 2325, 2333, 2341, 2349, 2357, 2365, 2373, 2381, 2389, 2397])
FLOOR_DIR = os.environ.get('FPS_FLOOR_DIR', './fps_data/data/floor')

BUY_END, LIVE_END_REL, POST = 360, 360 + 6900, 300
BUY_DEADLINE = 1860
LOSS = [1400, 1900, 2400, 2900, 3400]
CAP = 16000
GRENADES = ('hegrenade', 'flashbang', 'smokegrenade', 'molotov', 'incgrenade', 'decoy')
ACTION_KEYS = {'move_forward', 'move_right', 'yaw_rate', 'pitch_rate', 'climb', 'jump', 'crouch', 'duck', 'walk', 'fire',
               'fire_secondary', 'reload', 'sprint', 'ads', 'weapon_slot', 'zoom_level', 'ladder', 'buy', 'refund_item_id',
               'drop', 'pickup_id', 'use', 'throw', 'throw_state', 'throw_mode', 'signal', 'message', 'lean'}
NUM_KEYS = ('move_forward', 'move_right', 'yaw_rate', 'pitch_rate', 'climb')
BOOL_KEYS = ('jump', 'crouch', 'duck', 'walk', 'fire', 'fire_secondary', 'reload', 'sprint', 'ads')


def map_for_seed(seed):
    h = hashlib.sha256(f'rsi-fps-5.4:map:{seed}'.encode()).digest()
    return MAP_IDS[int.from_bytes(h, 'big') % 5]


def validate_action(a):
    """Strict action-contract check (extra keys / non-finite / bool-as-number / wrong enums are invalid)."""
    if not isinstance(a, dict): return False
    for k, v in a.items():
        if k not in ACTION_KEYS: return False
    for k in NUM_KEYS + ('lean',):
        if k in a:
            v = a[k]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or abs(v) > 1: return False
    for k in BOOL_KEYS:
        if k in a and not isinstance(a[k], bool): return False
    for k, lo, hi in (('weapon_slot', 0, 2), ('zoom_level', 0, 2), ('signal', 0, 15)):
        if k in a:
            v = a[k]
            if isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi: return False
    enums = {'ladder': ('none', 'attach', 'detach'), 'use': ('none', 'plant', 'defuse', 'pickup'),
             'throw': ('none',) + GRENADES, 'throw_state': ('idle', 'prime', 'release', 'cancel'),
             'throw_mode': ('full', 'lob', 'drop')}
    for k, vals in enums.items():
        if k in a and a[k] not in vals: return False
    if 'buy' in a and not isinstance(a['buy'], str): return False
    for k in ('drop', 'pickup_id', 'refund_item_id'):
        if k in a and a[k] is not None and (not isinstance(a[k], str) or len(a[k]) > 128): return False
    if 'message' in a:
        m = a['message']
        if not isinstance(m, dict) or set(m) != {'kind', 'zone', 'value'}: return False
    return True


class _GeomContact:
    """grenade-world adapter for maps without an exact mesh (Mirage): contact()/raycast_dist() on MapGeom.raycast."""
    def __init__(self, geom):
        self.g = geom

    def raycast_dist(self, o, d, L):
        hit, t, _, _ = self.g.raycast(o, (o[0] + d[0] * L, o[1] + d[1] * L, o[2] + d[2] * L))
        return t * L if hit else None

    def contact(self, o, d, L, idx=None):
        dist = self.raycast_dist(o, d, L)
        if dist is None: return None
        # approximate normal: opposite of the travel direction's dominant axis
        k = max(range(3), key=lambda i: abs(d[i])); n = [0.0, 0.0, 0.0]; n[k] = -1.0 if d[k] > 0 else 1.0
        return dist, tuple(n)


class Player:
    def __init__(self, seat):
        self.seat = seat; self.team = 0 if seat < 5 else 1
        self.ps = PlayerState()
        self.ws = WeaponSystem()
        self.cash = 0; self.hp = 100.0; self.armor = 0.0; self.helmet = False; self.kit = False
        self.alive = True
        self.nades = {}                # kind -> count
        self.nade_bought = {}          # per round purchase counts
        self.kit_item = None
        self.purchases = {}            # kind -> dict(price, item_id) for refunds (this round & life)
        self.has_bomb = False
        self.pickup_ready_tick = 0
        self.blind_until = 0; self.blind_start = 0
        self.held = {}                 # held action (applies between decisions)
        self.faulted = False
        self.objective = None          # 'plant' | 'defuse' | None
        self.obj_progress = 0
        self.throw_primed = None; self.throw_ready = 0
        self.receipts = []
        self.last_fire_tick = -10**9
        self.sounds = []
        self.death_tick = None
        self.kills = 0; self.deaths = 0

    @property
    def x(self): return self.ps.x
    @property
    def y(self): return self.ps.y
    @property
    def z(self): return self.ps.z


class Arena:
    def __init__(self, params=None, kit_dir=None):
        self.kit_dir = kit_dir or KIT_DIR
        self.mover = Mover(params)
        self.events = []
        self.map_id = None

    # ------------------------------------------------------------------ setup
    def reset(self, map_id=None, seed=0):
        self.seed = seed
        self.rng = random.Random(seed)
        self.map_id = map_id or map_for_seed(seed)
        self._load_map(self.map_id)
        self.events = []
        self.item_serial = 0; self.txn = 0; self.event_serial = 0
        self.tick = 0
        self.players = [Player(s) for s in range(10)]
        self.attack_team = self.rng.randrange(2)
        self.round_index = -1
        self.round_wins = [0, 0]
        self.loss_level = [1, 1]
        self.finished = False
        self.winner = None
        if self.mesh is not None or self.static.get('boxes'):
            gworld = self.mesh if self.mesh is not None else self.static
        else:                                 # Mirage: approximate voxel geometry (no public mesh) via an adapter
            gworld = _GeomContact(self.geom)
        self.grenades = GrenadeSystem(gworld, mesh=self.mesh, damage_cb=self._util_damage)
        from .visibility import LOS
        self.vis_los = LOS(self.mesh if self.mesh is not None else (self.static if self.static.get('boxes') else self.geom))
        self.world_items = []          # loose weapons / kits: dict(id,kind,position,weapon,owner,tick)
        self.bomb = None
        self.faults = [0] * 10
        from .obs import AudioEmitter
        self.audio = AudioEmitter(); self.audio_events = []
        for p in self.players:
            self._give(p, 'pistol' if self._side(p) == 'attack' else 'usp', default=True)
            self._give(p, 'knife', default=True)
        self._half_reset(initial=True)
        self._start_round()
        self.first_obs = True
        return self.observations()

    def _load_map(self, map_id):
        p = os.path.join(STATIC_DIR, f'static_{map_id}.json')
        d = json.load(open(p))
        self.static = d['map'] if 'map' in d else d
        from .obs import simulated_static_hash
        self.static_schema_hash = simulated_static_hash(self.static)
        self.mesh = None
        if map_id in MESH_FILES and self.static.get('collision'):
            col = self.static['collision'][0]
            pos, mat = MESH_FILES[map_id]
            self.mesh = TriMesh(os.path.join(ASSET_DIR, pos), col['basis'], col['offset'], os.path.join(ASSET_DIR, mat))
        self.geom = MapGeom.from_static(self.static, kit_dir=self.kit_dir)
        if self.mesh is not None:          # Dust2: exact triangle-mesh movement (fpssim/meshmove.py)
            from .meshmove import make_mesh_world, MeshMover
            self.world = make_mesh_world(self.static, ASSET_DIR, positions_name=MESH_FILES[map_id][0], cache_dir=FLOOR_DIR)
            self.mover = MeshMover()
        else:
            self.world = make_world(self.static, mapdata_dir=self.kit_dir, floor_cache_dir=FLOOR_DIR)
            self.mover = Mover()

    def los(self, a, b):
        return self.mesh.los(a, b) if self.mesh is not None else self.geom.los(a, b)

    # ------------------------------------------------------------------ helpers
    def _side(self, p):
        return 'attack' if p.team == self.attack_team else 'defend'

    def _emit(self, typ, **kw):
        self.event_serial += 1
        e = dict(type=typ, event_id=f'{self.seed}:{self.event_serial}', match_tick=self.tick, round_index=self.round_index)
        e.update(kw); self.events.append(e); return e

    def _txn(self):
        self.txn += 1; return f'economy:{self.txn - 1}'

    def _new_item(self):
        self.item_serial += 1; return f'item:{self.item_serial - 1}'

    def _cash(self, p, delta, reason, item_id=None):
        before = p.cash
        after = max(0, min(CAP, before + delta))
        self._emit('cash', transaction_id=self._txn(), item_id=item_id, seat=p.seat, reason=reason, before=before, after=after,
                   requested=delta, delta=after - before, overflow=max(0, before + delta - CAP) if delta > 0 else 0)
        p.cash = after

    def _give(self, p, kind, default=False, weapon=None):
        """Give a gun/knife to its catalogue slot (replacing the old weapon of that slot -> caller handles the drop)."""
        w = WEAPONS[kind]
        slot = {1: 0, 2: 1, 3: 2}[w['slot']]
        wp = weapon or Weapon(kind, item_id=self._new_item())
        if weapon is None:
            self._emit('item_create', transaction_id=self._txn(), item_id=wp.item_id, seat=p.seat, kind=kind, default=default)
        p.ws.slots[slot] = wp
        return wp

    # ------------------------------------------------------------------ round lifecycle
    def _half_reset(self, initial=False):
        for p in self.players:
            p.cash = 800
            if not initial:
                p.ws.slots = [None, None, None]
                p.armor = 0.0; p.helmet = False; p.kit = False; p.nades = {}
        self.loss_level = [1, 1]
        if not initial:
            for p in self.players:
                self._give(p, 'pistol' if self._side(p) == 'attack' else 'usp', default=True)
                self._give(p, 'knife', default=True)
            self._emit('cash_reset_marker') if False else None
        else:
            for p in self.players:
                self._emit('cash', transaction_id=self._txn(), item_id=None, seat=p.seat, reason='initial_cash', before=0, after=800,
                           requested=800, delta=800, overflow=0)

    def _start_round(self):
        self.round_index += 1
        if self.round_index == 12:
            self.attack_team = 1 - self.attack_team
            for p in self.players: self._swap_marker(p)
            self._emit('side_swap', attack_team=self.attack_team, defend_team=1 - self.attack_team)
            for p in self.players:
                self._emit('cash', transaction_id=self._txn(), item_id=None, seat=p.seat, reason='period_reset', before=p.cash,
                           after=800, requested=800, delta=800 - p.cash, overflow=0)
            self._half_reset()
        self.round_start_tick = self.tick
        self.buy_deadline_tick = self.round_start_tick + BUY_DEADLINE
        self.phase = 'buy'
        self.live_end = self.round_start_tick + LIVE_END_REL
        self.post_end = None
        self.round_over = False
        self.bomb = dict(state='held', carrier=None, position=None, site=None, planted_tick=None, fuse_end=None,
                         id=f'r{self.round_index}:bomb', drop_tick=None, dropper=None, actor=None, progress=0, duration=0, z0=0)
        self.world_items = []
        self.grenades.clear()
        self._emit('economy_round_start', transaction_id=self._txn(), item_id=None, attack_team=self.attack_team,
                   reset_money=800 if self.round_index in (0, 12) else None, loss_level=list(self.loss_level),
                   period='regulation', overtime_number=0)
        self._emit('round_start', attack_team=self.attack_team, defend_team=1 - self.attack_team,
                   buy_freeze_end_tick=self.round_start_tick + BUY_END, buy_deadline_tick=self.round_start_tick + BUY_DEADLINE,
                   live_end_tick=self.live_end, period='regulation', overtime_number=0, win_target=13,
                   reset_money=800 if self.round_index in (0, 12) else None)
        # spawn
        for side in ('attack', 'defend'):
            pads = list(self.geom.spawns('attack' if side == 'attack' else 'defend'))
            seats = [p for p in self.players if self._side(p) == side]
            idx = list(range(len(pads))); self.rng.shuffle(idx)
            for p, i in zip(seats, idx):
                pad = pads[i]
                self._respawn(p, pad)
        carriers = [p for p in self.players if self._side(p) == 'attack']
        c = self.rng.choice(carriers)
        c.has_bomb = True; self.bomb['carrier'] = c.seat; self.bomb['position'] = (c.x, c.y, c.z)

    def _swap_marker(self, p):
        pass

    def _respawn(self, p, pad):
        was_alive = p.alive
        if not was_alive:   # dead players lose everything
            p.ws.slots = [None, None, None]; p.armor = 0.0; p.helmet = False; p.kit = False; p.nades = {}
            self._give(p, 'pistol' if self._side(p) == 'attack' else 'usp', default=True)
            self._give(p, 'knife', default=True)
        else:
            # survivors keep gear, but guns refill and weapon readiness/recoil reset
            for i, w in enumerate(p.ws.slots):
                if w is not None:
                    nw = Weapon(w.id, item_id=w.item_id); p.ws.slots[i] = nw
        # an attacker that lost the bomb carrier role / side swap: team-restricted guns are dropped (side swap clears anyway)
        p.alive = True; p.hp = 100.0
        s = p.ps
        s.x, s.y, s.z = float(pad[0]), float(pad[1]), float(pad[2]) + (0.00011 if self.static.get('boxes') else 0.0)
        s.vx = s.vy = s.vz = 0.0; s.yaw = float(pad[3]); s.pitch = 0.0
        s.crouch = False; s.grounded = True; s.movement_mode = 'ground'; s.ladder_id = None; s.last_jump = False
        s.reattach_until = 0; s.pinned = False; s.stand_blocked = False; s.alive = True
        p.ws.active = 0 if p.ws.slots[0] else 1
        p.ws.switch_until = 0; p.ws.zoom_level = 0
        p.purchases = {}; p.nade_bought = {}
        p.has_bomb = False; p.objective = None; p.obj_progress = 0; p.blind_until = 0; p.held = {}
        p.throw_primed = None; p.receipts = []; p.death_tick = None; p.pickup_ready_tick = 0

    # ------------------------------------------------------------------ economy actions
    def _in_buy_region(self, p):
        if self.static.get('buy_zone') or self.static.get('buy_zones_by_side'):
            return self.geom.in_buy_zone(self._side(p), p.x, p.y, p.z)
        # whitebox maps publish no buy zone; ASSUMPTION: same union of 9 m spawn-pad circles (|dz| < 3) as Dust2/Mirage.
        # Every recorded buy happens during freeze on the pad, so the extent is unobserved.
        for pad in self.static['spawn_pads_by_side'][self._side(p)]:
            if math.hypot(p.x - pad[0], p.y - pad[1]) < 9.0 and abs(p.z - pad[2]) < 3.0:
                return True
        return False

    def _buy_ok(self, p):
        rt = self.tick - self.round_start_tick
        return p.alive and self.phase in ('buy', 'live') and rt <= BUY_DEADLINE and self._in_buy_region(p)

    def _receipt(self, p, action, ok, reason=None, **kw):
        r = dict(action=action, ok=ok, reason=reason, seat=p.seat, **kw)
        p.receipts.append(r); return r

    def _grenade_total(self, p):
        return sum(p.nades.values())

    def _buy(self, p, kind):
        if kind == 'none': return
        if not self._buy_ok(p):
            return self._receipt(p, 'buy', False, 'buy_unavailable', item=kind)
        eq = EQUIP.get(kind); gun = WEAPONS.get(kind)
        side_tag = 'T' if self._side(p) == 'attack' else 'CT'
        if eq is not None:
            if kind == 'c4' or not eq.get('teams') or side_tag not in eq['teams']:
                return self._receipt(p, 'buy', False, 'wrong_side', item=kind)
            price = eq['price']
            if kind == 'armor':
                if p.armor >= 100: return self._receipt(p, 'buy', False, 'already_owned', item=kind)
            elif kind == 'helmet':
                if p.armor >= 100 and p.helmet: return self._receipt(p, 'buy', False, 'already_owned', item=kind)
                price = 350 if p.armor >= 100 else 1000
            elif kind == 'defusekit':
                if p.kit: return self._receipt(p, 'buy', False, 'already_owned', item=kind)
            else:
                lim = eq.get('max_count', 1)
                if p.nades.get(kind, 0) >= lim or self._grenade_total(p) >= 4:
                    return self._receipt(p, 'buy', False, 'limit', item=kind)
                if p.nade_bought.get(kind, 0) >= {'flashbang': 2}.get(kind, 1):
                    return self._receipt(p, 'buy', False, 'round_limit', item=kind)
            if p.cash < price: return self._receipt(p, 'buy', False, 'insufficient_cash', item=kind)
            item_id = None
            if kind == 'armor': p.armor = 100.0
            elif kind == 'helmet': p.armor = 100.0; p.helmet = True
            elif kind == 'defusekit':
                p.kit = True; item_id = self._new_item()
                self._emit('item_create', transaction_id=self._txn(), item_id=item_id, seat=p.seat, kind=kind, default=False)
                p.kit_item = item_id
            else:
                item_id = self._new_item()
                self._emit('item_create', transaction_id=self._txn(), item_id=item_id, seat=p.seat, kind=kind, default=False)
                p.nades[kind] = p.nades.get(kind, 0) + 1; p.nade_bought[kind] = p.nade_bought.get(kind, 0) + 1
            self._cash(p, -price, 'buy', item_id)
            self._emit('buy', transaction_id=self._txn(), item_id=item_id, seat=p.seat, kind=kind, price=price)
            p.purchases[kind] = dict(price=price + (p.purchases.get(kind, {}).get('price', 0) if kind in ('helmet',) else 0),
                                     item_id=item_id, count=p.purchases.get(kind, {}).get('count', 0) + 1)
            return self._receipt(p, 'buy', True, None, item=kind, item_id=item_id)
        if gun is None or gun['slot'] not in (1, 2):
            return self._receipt(p, 'buy', False, 'unknown_item', item=kind)
        if side_tag not in gun['teams']:
            return self._receipt(p, 'buy', False, 'wrong_side', item=kind)
        if p.cash < gun['price']: return self._receipt(p, 'buy', False, 'insufficient_cash', item=kind)
        slot = {1: 0, 2: 1}[gun['slot']]
        old = p.ws.slots[slot]
        if old is not None: self._drop_weapon(p, slot, reason='replace')
        wp = Weapon(kind, item_id=self._new_item())
        self._emit('item_create', transaction_id=self._txn(), item_id=wp.item_id, seat=p.seat, kind=kind, default=False)
        p.ws.slots[slot] = wp
        self._cash(p, -gun['price'], 'buy', wp.item_id)
        self._emit('buy', transaction_id=self._txn(), item_id=wp.item_id, seat=p.seat, kind=kind, price=gun['price'])
        p.purchases[kind] = dict(price=gun['price'], item_id=wp.item_id, count=1)
        self._switch_to(p, slot)
        return self._receipt(p, 'buy', True, None, item=kind, item_id=wp.item_id)

    def _switch_to(self, p, slot):
        if p.ws.slots[slot] is None or p.ws.active == slot: return
        p.ws.active = slot; p.ws.zoom_level = 0
        p.ws.switch_until = self.tick + 11

    def _refund(self, p, ref):
        if ref is None: return
        if not self._buy_ok(p): return self._receipt(p, 'refund', False, 'refund_unavailable', item=ref)
        key = None
        for k, v in p.purchases.items():
            if ref in (k, v.get('item_id')): key = k; break
        if key is None: return self._receipt(p, 'refund', False, 'not_refundable', item=ref)
        v = p.purchases[key]
        if key in WEAPONS:
            slot = {1: 0, 2: 1}[WEAPONS[key]['slot']]
            w = p.ws.slots[slot]
            if w is None or w.id != key or w.shots or w.item_id != v['item_id']:
                return self._receipt(p, 'refund', False, 'not_refundable', item=ref)
            p.ws.slots[slot] = None
            self._emit('item_destroy', transaction_id=self._txn(), item_id=w.item_id)
            if p.ws.active == slot: p.ws.active = 0 if p.ws.slots[0] else 1
        elif key == 'armor' or key == 'helmet':
            if p.armor <= 0: return self._receipt(p, 'refund', False, 'not_refundable', item=ref)
            p.armor = 0.0; p.helmet = False
        elif key == 'defusekit':
            if not p.kit: return self._receipt(p, 'refund', False, 'not_refundable', item=ref)
            p.kit = False
        else:
            n = min(p.nades.get(key, 0), v.get('count', 1))
            if n <= 0: return self._receipt(p, 'refund', False, 'not_refundable', item=ref)
            p.nades[key] -= n
            if p.nades[key] == 0: del p.nades[key]
            p.nade_bought[key] = max(0, p.nade_bought.get(key, 0) - n)
            v = dict(v, price=v['price'] * n) if False else v
        price = v['price']
        del p.purchases[key]
        self._cash(p, price, 'refund', v.get('item_id'))
        self._emit('refund', transaction_id=self._txn(), item_id=v.get('item_id'), seat=p.seat, kind=key, price=price)
        return self._receipt(p, 'refund', True, None, item=ref)

    def _drop_weapon(self, p, slot, reason='drop', death=False):
        w = p.ws.slots[slot]
        if w is None: return None
        if w.id == 'knife' and not death: return None
        p.ws.slots[slot] = None
        pos = (p.x, p.y, p.z)
        it = dict(id=w.item_id, kind=w.id, position=pos, weapon=w, owner=p.seat, tick=self.tick)
        self.world_items.append(it)
        self._emit('drop', transaction_id=self._txn(), item_id=w.item_id, seat=p.seat, position=list(pos))
        if p.ws.active == slot:
            p.ws.active = 0 if p.ws.slots[0] else 1
            p.ws.switch_until = self.tick + 11
        p.purchases.pop(w.id, None)
        return it

    def _drop_request(self, p, what):
        if what in ('none', ''): return
        if not p.alive or self.phase == 'post': return self._receipt(p, 'drop', False, 'unavailable', item=what)
        if what == 'bomb':
            if p.has_bomb: self._drop_bomb(p, voluntary=True); return self._receipt(p, 'drop', True, None, item='bomb')
            return self._receipt(p, 'drop', False, 'no_bomb', item=what)
        slot = {'primary': 0, 'secondary': 1, '0': 0, '1': 1, 'slot0': 0, 'slot1': 1}.get(what)
        if slot is None:
            if what == p.ws.slots[0].id if p.ws.slots[0] else False: slot = 0
            elif p.ws.slots[1] and what == p.ws.slots[1].id: slot = 1
        if slot is None or p.ws.slots[slot] is None: return self._receipt(p, 'drop', False, 'nothing', item=what)
        self._drop_weapon(p, slot)
        return self._receipt(p, 'drop', True, None, item=what)

    def _drop_bomb(self, p, voluntary=False):
        if not p.has_bomb: return
        p.has_bomb = False
        b = self.bomb
        pos = (p.x, p.y, p.z)
        b.update(state='world', carrier=None, position=pos, drop_tick=self.tick, dropper=p.seat if voluntary else None)
        self._emit('drop', player=p.seat, item_id=b['id'], position=list(pos))

    def _pickup(self, p, pid):
        if not p.alive: return
        if pid is None: return self._receipt(p, 'pickup', False, 'no_id')
        if self.bomb['state'] == 'world' and pid == self.bomb['id'] and self._side(p) == 'attack':
            return self._try_pick_bomb(p, explicit=True)
        it = next((i for i in self.world_items if i['id'] == pid), None)
        if it is None: return self._receipt(p, 'pickup', False, 'not_found', item=pid)
        eye = (p.x, p.y, p.z + (0.95 if p.ps.crouch else 1.62))
        d = math.dist(eye, it['position'])
        if d > 2.8 and math.dist((p.x, p.y, p.z), it['position']) > 2.8:
            return self._receipt(p, 'pickup', False, 'too_far', item=pid)
        kind = it['kind']
        side_tag = 'T' if self._side(p) == 'attack' else 'CT'
        if kind not in WEAPONS: return self._receipt(p, 'pickup', False, 'unsupported', item=pid)
        slot = {1: 0, 2: 1, 3: 2}[WEAPONS[kind]['slot']]
        if slot == 2: return self._receipt(p, 'pickup', False, 'unsupported', item=pid)
        if p.ws.slots[slot] is not None:
            self._drop_weapon(p, slot)
        self.world_items.remove(it)
        p.ws.slots[slot] = it['weapon']
        self._switch_to(p, slot)
        self._emit('pickup', player=p.seat, item_id=it['id'])
        return self._receipt(p, 'pickup', True, None, item=pid, item_id=it['id'])

    def _try_pick_bomb(self, p, explicit=False):
        b = self.bomb
        eye = (p.x, p.y, p.z + (0.95 if p.ps.crouch else 1.62))
        pos = b['position']
        if math.dist((p.x, p.y, p.z), pos) >= 2.3 and math.dist(eye, pos) >= 2.3: return False
        if b['dropper'] == p.seat and self.tick - b['drop_tick'] < 90: return False
        if not self.geom.los(eye, (pos[0], pos[1], pos[2] + 0.4)): return False
        p.has_bomb = True; b.update(state='held', carrier=p.seat, position=pos)
        self._emit('pickup', player=p.seat, item_id=b['id'])
        if explicit: self._receipt(p, 'pickup', True, None, item=b['id'])
        return True

    # ------------------------------------------------------------------ objective
    def _obj_step(self):
        b = self.bomb
        if self.phase != 'live':
            for p in self.players:
                if p.objective: self._cancel_obj(p)
            return
        if b['state'] == 'planted' and self.tick >= b['fuse_end']:
            return
        # candidate actor selection
        if b['state'] == 'held':
            c = self.players[b['carrier']]
            b['position'] = (c.x, c.y, c.z)
            if not c.alive: self._drop_bomb(c)
        if b['state'] == 'world':
            for p in self.players:
                if p.alive and self._side(p) == 'attack' and not p.has_bomb:
                    if self._try_pick_bomb(p): break
        actor = None; kind = None
        if b['state'] == 'held':
            c = self.players[b['carrier']]
            if c.alive and c.held.get('use') == 'plant' and self._can_obj(c) and self.geom.in_plant_zone(c.x, c.y, c.z):
                actor, kind = c, 'plant'
        elif b['state'] == 'planted':
            cur = self.players[b['actor']] if b['actor'] is not None else None
            cands = [p for p in self.players if p.alive and self._side(p) == 'defend' and p.held.get('use') == 'defuse' and self._can_obj(p)
                     and math.dist((p.x, p.y, p.z), b['position']) < 2.8]
            if cur is not None and cur in cands: actor, kind = cur, 'defuse'
            else:
                for p in cands:
                    eye = (p.x, p.y, p.z + (0.95 if p.ps.crouch else 1.62))
                    if self.geom.los(eye, (b['position'][0], b['position'][1], b['position'][2] + 0.3)):
                        actor, kind = p, 'defuse'; break
        for p in self.players:
            if p.objective and p is not actor: self._cancel_obj(p)
        if actor is None:
            b['actor'] = None; return
        if actor.objective != kind:
            actor.objective = kind; actor.obj_progress = 0; b['actor'] = actor.seat
            dur = 180 if kind == 'plant' else (300 if actor.kit else 600)
            b['duration'] = dur
            if kind == 'plant':
                self._emit('plant_start', player=actor.seat, item_id=b['id'], site=self.geom.in_plant_zone(actor.x, actor.y, actor.z), duration_ticks=dur)
            else:
                self._emit('defuse_start', player=actor.seat, item_id=b['id'], duration_ticks=dur)
            if actor.ws.zoom_level: actor.ws.zoom_level = 0
            if hasattr(actor.ws, 'cancel_reload'): actor.ws.cancel_reload(self.tick)   # platform cancels reload on plant/defuse start
        actor.obj_progress += 1
        actor.ps.vx = actor.ps.vy = actor.ps.vz = 0.0
        if actor.obj_progress >= b['duration']:
            actor.objective = None
            if kind == 'plant':
                site = self.geom.in_plant_zone(actor.x, actor.y, actor.z)
                actor.has_bomb = False
                b.update(state='planted', carrier=None, position=(actor.x, actor.y, actor.z), site=site, planted_tick=self.tick,
                         fuse_end=self.tick + 2400, actor=None, planter=actor.seat)
                self._emit('plant_complete', item_id=b['id'], fuse_end_tick=b['fuse_end'])
                self._emit('plant_complete_detail', player=actor.seat, item_id=b['id'], site=site, position=list(b['position']))
                self._cash(actor, 300, 'plant')
            else:
                b['state'] = 'defused'
                self._emit('defuse_complete', item_id=b['id'], fuse_end_tick=b['fuse_end'])
                self._emit('defuse_complete_detail', player=actor.seat, item_id=b['id'])
                self._cash(actor, 300, 'defuse')
                self._end_round(1 - self.attack_team, 'defuse')

    def _can_obj(self, p):
        return p.alive and p.ps.grounded and math.hypot(p.ps.vx, p.ps.vy) <= 0.1 + 1e-9 and p.ws.zoom_level == 0 or \
            (p.alive and p.ps.grounded and p.objective is not None and math.hypot(p.ps.vx, p.ps.vy) <= 0.1 + 1e-9)

    def _cancel_obj(self, p):
        if p.objective == 'plant': self._emit('plant_cancel', player=p.seat, item_id=self.bomb['id'])
        elif p.objective == 'defuse': self._emit('defuse_cancel', player=p.seat, item_id=self.bomb['id'])
        p.objective = None; p.obj_progress = 0
        if self.bomb.get('actor') == p.seat: self.bomb['actor'] = None

    # ------------------------------------------------------------------ round end / economy settlement
    def _end_round(self, winner_team, reason):
        if self.round_over: return
        self.round_over = True
        self.phase = 'post'
        self.post_end = self.tick + POST
        for p in self.players:
            if p.objective: self._cancel_obj(p)
        planted_lost = (self.bomb['state'] in ('planted', 'defused')) and reason == 'defuse'
        objective_win = reason in ('bomb_explode', 'defuse')
        win_amt = 3500 if objective_win else 3250
        lvl_before = list(self.loss_level)
        loser = 1 - winner_team
        attackers_lost = loser == self.attack_team
        for p in self.players:
            if p.team == winner_team:
                self._cash(p, win_amt, 'round_win')
        for p in self.players:
            if p.team == loser:
                survive_timeout = (reason == 'timeout' and attackers_lost and p.alive)
                if not survive_timeout:
                    self._cash(p, LOSS[lvl_before[loser]], 'round_loss')
                if planted_lost and attackers_lost:
                    self._cash(p, 600, 'planted_loss')
        self.loss_level[loser] = min(4, self.loss_level[loser] + 1)
        self.loss_level[winner_team] = max(0, self.loss_level[winner_team] - 1)
        self.round_wins[winner_team] += 1
        self._emit('round_economy', transaction_id=self._txn(), item_id=None, winner=winner_team, reason=reason if reason != 'bomb_explode' else 'bomb_explode',
                   level_before=lvl_before, level_after=list(self.loss_level), round_wins=list(self.round_wins))
        self._emit('round_end', round_winner=winner_team, reason=reason, round_wins=list(self.round_wins), post_end_tick=self.post_end)
        done = max(self.round_wins) >= 13 or sum(self.round_wins) >= 24
        if done:
            self.finished = True
            if self.round_wins[0] == self.round_wins[1]: self.winner = None
            else: self.winner = int(self.round_wins[1] > self.round_wins[0])
            self._emit('match_end', match_winner=self.winner, reason='round_wins', round_wins=list(self.round_wins))

    # ------------------------------------------------------------------ kills / damage
    def _damage(self, victim, attacker, amount_hp, armor_abs, weapon, part=None):
        victim.armor = max(0.0, victim.armor - armor_abs)
        victim.hp -= amount_hp
        if victim.hp <= 1e-9 and victim.alive:
            victim.hp = 0.0
            self._kill(victim, attacker, weapon)

    def _kill(self, v, killer, weapon):
        v.alive = False; v.ps.alive = False; v.death_tick = self.tick; v.deaths += 1
        self._emit('kill', target=v.seat, attackers=[killer.seat] if killer is not None else [])
        if v.objective: self._cancel_obj(v)
        if v.has_bomb: self._drop_bomb(v)
        for slot in range(3):
            if v.ws.slots[slot] is not None and v.ws.slots[slot].id != 'knife' or (slot == 2 and v.ws.slots[2] is not None):
                self._drop_weapon(v, slot, death=True)
        if v.kit: v.kit = False
        v.nades = {}; v.armor = v.armor
        if killer is not None and killer.team != v.team and self.phase == 'live':
            reward = WEAPONS[weapon]['killReward'] if weapon in WEAPONS else 300
            self._cash(killer, reward, 'kill', killer.ws.slots[killer.ws.active].item_id if killer.ws.slots[killer.ws.active] else None)
            if self._side(killer) == 'defend':
                for p in self.players:
                    if self._side(p) == 'defend': self._cash(p, 50, 'ct_kill_bonus')
            self._emit('kill_reward', transaction_id=self._txn(), item_id=None, killer=killer.seat, victim_team=v.team, reward=reward)
            killer.kills += 1

    # ------------------------------------------------------------------ per-tick physics
    def _targets(self):
        return [dict(seat=q.seat, pos=(q.x, q.y, q.z), h=q.ps.height, alive=q.alive, armor=q.armor, helmet=q.helmet, team=q.team,
                     yaw=q.ps.yaw) for q in self.players]

    def _audio_state(self):
        return [dict(alive=p.alive, x=p.x, y=p.y, z=p.z, vx=p.ps.vx, vy=p.ps.vy, vz=p.ps.vz, grounded=p.ps.grounded,
                     crouch=p.ps.crouch, walk=bool(p.held.get('walk')), step_distance=p.ps.step_distance) for p in self.players]

    def _tick(self):
        prev = self._audio_state(); n0 = len(self.events)
        self._tick_inner()
        self.audio.step(prev, self._audio_state(), self.events[n0:], self.tick)
        b = self.bomb
        if b and b['state'] == 'planted' and b.get('fuse_end'):
            off = self.tick - b['planted_tick']
            if off in BEEP_SLOW or off in BEEP_FAST:
                self.audio.bomb_beep(b['position'], fast=off in BEEP_FAST)

    def _tick_inner(self):
        self.tick += 1
        t = self.tick
        live = self.phase == 'live'
        # phase transitions by clock
        if self.phase == 'buy' and t - self.round_start_tick >= BUY_END:
            self.phase = 'live'; live = True
        # ---- movement
        states = [p.ps for p in self.players]
        snap = [(p.x, p.y, p.z, p.alive) for p in self.players]
        for p in self.players:
            if not p.alive: continue
            a = p.held
            if self.phase == 'buy':
                a = dict(a); a.pop('jump', None)
                a['move_forward'] = 0.0; a['move_right'] = 0.0
                p.ps.vx = p.ps.vy = 0.0
            M = self._max_speed(p)
            others = [(x, y, z) for j, (x, y, z, al) in enumerate(snap) if j != p.seat and al]
            locked = p.objective is not None
            self.mover.step(p.ps, a, M, self.world, (), t, locked)
        self._resolve_players()
        # ---- weapons
        targets = self._targets()
        for p in self.players:
            if not p.alive: continue
            a = p.held
            can_act = live and p.objective is None and p.throw_primed is None
            want = a.get('weapon_slot', 1)
            if not live and want != p.ws.active and self.phase == 'buy':
                pass
            zoom = a.get('zoom_level', 0)
            if a.get('ads') and zoom == 0: zoom = 1
            out = p.ws.tick(t, bool(a.get('fire')) , bool(a.get('reload')), want if self.phase != 'post' else None,
                            can_act=can_act or (self.phase == 'buy' and False), crouch=p.ps.crouch, in_air=not p.ps.grounded, zoom=zoom if p.ws.weapon() and WEAPONS[p.ws.weapon().id].get('zoomStyle') else None)
            for kind, w in out:
                if kind == 'shot': self._do_shot(p, w, targets)
                elif kind == 'knife': self._do_knife(p, a, targets)
                elif kind in ('reload_start', 'reload_commit', 'reload_finish', 'reload_cancel'):
                    ev = dict(player=p.seat, weapon=w.id)
                    if kind == 'reload_start': ev.update(empty=w.ammo == 0, duration=WEAPONS[w.id]['reloadTime'])
                    self._emit(kind, **ev)
                elif kind == 'draw': self._emit('draw', player=p.seat, weapon=w.id)
        # ---- objective / bomb
        self._obj_step()
        # ---- grenades
        if self.phase == 'live':
            for p in self.players:
                if p.alive:
                    for ev in self.grenades.handle_throw(p, p.held, t):
                        self.events.append(dict(ev, event_id=f'{self.seed}:{self.event_serial + 1}', round_index=self.round_index))
                        self.event_serial += 1
            for ev in self.grenades.step(t, self.players):
                if ev.get('type') == 'detonate' and ev.get('kind') in ('molotov', 'incgrenade') and ev.get('position'):
                    self.audio.buf.append(dict(kind='fire_started', position=list(ev['position']), source=None, radius=None,
                                               directional=False, local_only=False, remote_only=False, occludable=False))
                self.events.append(dict(ev, event_id=f'{self.seed}:{self.event_serial + 1}', round_index=self.round_index))
                self.event_serial += 1
        # ---- bomb timer
        b = self.bomb
        if b['state'] == 'planted' and t >= b['fuse_end'] and not self.round_over:
            b['state'] = 'exploded'
            self._emit('bomb_explode', item_id=b['id'], fuse_end_tick=b['fuse_end'])
            self._emit('bomb_explode_detail', player=None, item_id=b['id'], position=list(b['position']))
            self._end_round(self.attack_team, 'bomb_explode')
        # ---- win conditions
        if self.phase == 'live' and not self.round_over:
            att_alive = sum(1 for p in self.players if p.alive and self._side(p) == 'attack')
            def_alive = sum(1 for p in self.players if p.alive and self._side(p) == 'defend')
            planted = b['state'] == 'planted'
            if def_alive == 0:
                self._end_round(self.attack_team, 'elimination')
            elif att_alive == 0 and not planted:
                self._end_round(1 - self.attack_team, 'elimination')
            elif t >= self.live_end and not planted:
                self._end_round(1 - self.attack_team, 'timeout')
        if self.phase == 'post' and t >= self.post_end and not self.finished:
            self._start_round()

    def _resolve_players(self):
        """cs2-web resolveAllPlayerCollisions (2 passes over all pairs): head standing ('boosting') and mutual radial
        push-out to 0.70 m with relative-velocity cancellation; objective-locked players are not pushed.  The cs2-web
        wall re-collision after a push is approximated by reverting a push that would end inside world geometry."""
        from .collision import HW
        live = [p for p in self.players if p.alive]
        R = 0.70
        for _ in range(2):
            for i in range(len(live)):
                for j in range(i + 1, len(live)):
                    a, b = live[i].ps, live[j].ps
                    dx, dy = a.x - b.x, a.y - b.y
                    d2 = dx * dx + dy * dy
                    if d2 >= R * R: continue
                    ta, tb = a.z + a.height, b.z + b.height
                    if a.z >= tb - 0.22 and a.vz <= 0.1:
                        a.z = tb; a.grounded = True; a.vz = max(0.0, a.vz); continue
                    if b.z >= ta - 0.22 and b.vz <= 0.1:
                        b.z = ta; b.grounded = True; b.vz = max(0.0, b.vz); continue
                    if min(ta, tb) - max(a.z, b.z) <= 0.05: continue
                    la, lb = live[i].objective is not None, live[j].objective is not None
                    if la and lb: continue
                    d = math.sqrt(d2); ov = R - d
                    if d > 1e-5: nx, ny = dx / d, dy / d
                    else: nx, ny = math.cos(a.yaw), -math.sin(a.yaw)
                    pa, pb, fa, fb = ov * .5, ov * .5, .5, .5
                    if la: pa, pb, fa, fb = 0.0, ov, 0.0, 1.0
                    elif lb: pa, pb, fa, fb = ov, 0.0, 1.0, 0.0
                    for st, sg, pu in ((a, 1, pa), (b, -1, pb)):
                        nxp, nyp = st.x + sg * nx * pu, st.y + sg * ny * pu
                        bl = getattr(self.world, 'blockers', None)
                        if bl is None or len(bl(nxp - HW, nxp + HW, nyp - HW, nyp + HW, st.z + 0.46, st.z + st.height)) == 0:
                            st.x, st.y = nxp, nyp
                    vn = (a.vx - b.vx) * nx + (a.vy - b.vy) * ny
                    if vn < 0:
                        a.vx -= vn * fa * nx; a.vy -= vn * fa * ny
                        b.vx += vn * fb * nx; b.vy += vn * fb * ny

    def _max_speed(self, p):
        w = p.ws.weapon()
        return mv.max_speed(w.id if w else 'knife', p.ws.zoom_level)

    def _eye(self, p):
        return (p.x, p.y, p.z + (0.95 if p.ps.crouch else 1.62))

    def _util_damage(self, victim, attacker_seat, hp, armor_abs, weapon):
        v = victim if isinstance(victim, Player) else self.players[victim]
        att = self.players[attacker_seat] if attacker_seat is not None else None
        self._damage(v, att, hp, armor_abs, weapon)

    def _do_shot(self, p, w, targets):
        """One trigger pull: cs2-web accuracy (pre-kick punch / penalty), two-disc spread, penetration trace on the
        mesh (Dust2) or first-hit world raycast (whitebox), cs2-web hit volumes, exact damage formula."""
        meta = WEAPONS[w.id]
        eye = self._eye(p)
        ap_pitch, ap_yaw, pen = getattr(w, 'punch_at_shot', (0.0, 0.0, w.inaccuracy))
        acc = ACCM.accuracy(meta, p.ps.speed, p.ps.crouch, p.ps.grounded, p.ps.vz, p.ws.zoom_level, pen)
        yaw = p.ps.yaw + 2.0 * ap_yaw
        pitch = max(-1.48, min(1.48, p.ps.pitch + 2.0 * ap_pitch))
        rng_ = meta.get('range') or 200.0
        for pel in range(meta.get('pellets', 1) or 1):
            d = ACCM.sample_direction(yaw, pitch, acc, self.rng.random)
            phits = []
            for q in self.players:
                if q.seat == p.seat or not q.alive or q.team == p.team: continue
                r = combat.ray_hit_player(eye, d, (q.x, q.y, q.z), q.ps.yaw, q.ps.crouch, rng_)
                if r is not None: phits.append((q.seat, r[0], r[1]))
            if self.mesh is not None:
                tr = PEN.trace_bullet(eye, d, meta, phits, PEN.surfaces(self.mesh, eye, d, rng_))
            else:
                end = (eye[0] + d[0] * rng_, eye[1] + d[1] * rng_, eye[2] + d[2] * rng_)
                hit, t, _, _ = self.geom.raycast(eye, end)
                wall = t * rng_ if hit else rng_
                first = min((h for h in phits if h[1] < wall), key=lambda h: h[1], default=None)
                tr = dict(end_distance=first[1] if first else wall, hit_world=hit and first is None,
                          hits=[(first[0], first[1], first[2], 1.0, False, 0)] if first else [], segments=[])
            to = tuple(eye[k] + d[k] * tr['end_distance'] for k in range(3))
            self.grenades.carve_bullet(eye, to, self.tick)
            self._emit('fire', player=p.seat, weapon=w.id, pellet=pel,
                       materials=[dict(entry=sg['entry'], exit=sg['exit'], material=sg['material'], thickness=sg['thickness'])
                                  for sg in tr['segments']], **{'from': list(eye)}, to=list(to))
            p.last_fire_tick = self.tick
            for seat, dist, part, scale, wallbang, npen in tr['hits']:
                v = self.players[seat]
                if not v.alive: continue
                amt, hp, ab = combat.damage(w.id, part, dist, v.armor, v.helmet)
                if scale != 1.0:
                    amt, hp, ab = amt * scale, hp * scale, ab * scale
                self._emit('hit', player=p.seat, target=v.seat, part=part, kind='bullet', amount=amt, hp_damage=hp, armor_absorbed=ab)
                self._damage(v, p, hp, ab, w.id, part)

    def _do_knife(self, p, a, targets):
        st = dict(seat=p.seat, eye=self._eye(p), yaw=p.ps.yaw, eye_h=0.95 if p.ps.crouch else 1.62)
        sec = bool(a.get('fire_secondary'))
        r = combat.knife_attack(st, targets, secondary=sec)
        self._emit('fire', player=p.seat, weapon='knife', pellet=0, materials=[], **{'from': list(self._eye(p))}, to=list(self._eye(p)))
        if r is None: return
        v = self.players[r['target']]
        self._emit('hit', player=p.seat, target=v.seat, part=r['part'], kind='knife', amount=r['amount'], hp_damage=r['hp_damage'],
                   armor_absorbed=r['armor_absorbed'])
        self._damage(v, p, r['hp_damage'], r['armor_absorbed'], 'knife', r['part'])

    # ------------------------------------------------------------------ public stepping
    def step(self, actions):
        """actions: list of 10 dict (platform action contract).  Returns (observations, events, done, info)."""
        n0 = len(self.events)
        for p, a in zip(self.players, actions):
            if p.faulted: a = {}
            elif not validate_action(a):
                p.faulted = True; self.faults[p.seat] += 1; a = {}
                self._emit('fault', player=p.seat, reason='invalid_action')
            p.receipts = []
            held = dict(a)
            p.held = held
            if p.alive and not self.finished and self.phase != 'post' or (p.alive and self.phase == 'post'):
                self._one_shots(p, held)
        for k in range(3):
            if self.finished: break
            self._tick()
            if k < 2:
                self.audio.flush()       # platform delivers only the decision tick's emitters (fit on our training.jsonl)
        self.audio_events = self.audio.flush()
        # burning fires: one looping 'fire_ambient' emitter per active fire at every decision (radius 24, non-directional)
        for f in getattr(self.grenades, 'fires', []):
            self.audio_events.append(dict(kind='fire_ambient', position=list(f['position']), source=f['id'], radius=24.0,
                                          directional=False, local_only=False, remote_only=False, occludable=False))
        obs = self.observations()
        ev = self.events[n0:]
        return obs, ev, self.finished, dict(tick=self.tick, round_wins=list(self.round_wins), faults=list(self.faults))

    def _one_shots(self, p, a):
        if self.phase == 'post': return
        if 'buy' in a and a['buy'] != 'none': self._buy(p, a['buy'])
        if a.get('refund_item_id'): self._refund(p, a['refund_item_id'])
        if a.get('drop', 'none') != 'none': self._drop_request(p, a['drop'])
        if a.get('use') == 'pickup': self._pickup(p, a.get('pickup_id'))

    # ------------------------------------------------------------------ observations
    def visible_enemies(self):
        """fpssim.visibility (validated 0.99997-0.999999 per enemy pair on the recordings): 55 deg horizontal, per-point 45 deg
        vertical, cs2-web hit-volume points (head/chest/abdomen), static LOS + smoke, blind window."""
        from .visibility import visible_enemies
        return visible_enemies(self.players, self.vis_los, self.grenades.smoke_blocks, self.tick)

    def observations(self):
        vis = self.visible_enemies()
        if build_observation is not None:
            obs = [build_observation(self, p, vis[p.seat], first=self.first_obs) for p in self.players]
        else:
            obs = [self._basic_obs(p, vis[p.seat]) for p in self.players]
        self.first_obs = False
        return obs

    def _basic_obs(self, p, vis):
        """Minimal contract-shaped observation (used until fpssim/obs.py lands)."""
        def wstate(w): return None if w is None else w.state(self.tick)
        me = dict(x=p.x, y=p.y, z=p.z, yaw=p.ps.yaw, pitch=p.ps.pitch, vx=p.ps.vx, vy=p.ps.vy, vz=p.ps.vz, hp=p.hp,
                  armor=p.armor, helmet=p.helmet, cash=p.cash, alive=p.alive, crouch=p.ps.crouch, grounded=p.ps.grounded,
                  active_slot=p.ws.active, weapons=[wstate(w) for w in p.ws.slots], carrying_bomb=p.has_bomb, has_kit=p.kit,
                  blind_ticks=max(0, p.blind_until - self.tick), ammo=(p.ws.weapon().ammo if p.ws.weapon() else 0),
                  reload=0.0, zoom_level=p.ws.zoom_level, throwables=dict(p.nades))
        b = self.bomb
        bomb = dict(state=b['state'] if (self._side(p) == 'attack' or b['state'] != 'held') else 'unknown',
                    fuse_end_tick=b.get('fuse_end'))
        if self._side(p) == 'attack' or b['state'] in ('planted', 'defused', 'exploded'):
            bomb.update(id=b['id'], carrier=b['carrier'], position=list(b['position']) if b['position'] else None, site=b.get('site'))
        o = dict(rules_version='rsi-fps-5.5', seat=p.seat, team=p.team, side=self._side(p), role=0, match_tick=self.tick,
                 round_index=self.round_index, round_tick=self.tick - self.round_start_tick, phase=self.phase,
                 round_wins=list(self.round_wins), attack_team=self.attack_team, me=me,
                 visible=[dict(seat=q, team=self.players[q].team, x=self.players[q].x, y=self.players[q].y, z=self.players[q].z,
                               yaw=self.players[q].ps.yaw, pitch=self.players[q].ps.pitch, crouch=self.players[q].ps.crouch,
                               parts=parts, weapon=(self.players[q].ws.weapon().id if self.players[q].ws.weapon() else None))
                          for q, parts in vis],
                 sounds=[], heard=[], bomb=bomb, world_items=[dict(id=i['id'], kind=i['kind'], position=list(i['position'])) for i in self.world_items],
                 messages=[], receipts=list(p.receipts), map_id=self.map_id, static_schema_hash='current',
                 teammates=[dict(seat=q.seat, alive=q.alive, cash=q.cash) for q in self.players if q.team == p.team and q is not p])
        if self.first_obs:
            o['map'] = self.static
        return o
