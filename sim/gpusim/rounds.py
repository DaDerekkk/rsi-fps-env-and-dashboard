"""Batched round machine / economy / bomb for E matches (port of fpssim.arena + fpssim.economy), see docs/gpu_rounds.md.

    from gpusim.rounds import Rounds
    R = Rounds(E, 'cuda', static_map_json, mesh=GpuMesh(...), K=48)
    R.reset(seed=1)                        # Arena.reset for every match (counter-based RNG, see u01_py)
    R.s                                    # dict of state tensors (SPEC names), shared with move / weapons / grenades
    R.one_shots()                          # decision boundary: buy / refund / drop / pickup, seats in order
    R.tick_begin(gtick); ...; R.obj_step(); R.end_tick()

The CUDA kernels (gpusim/rounds_cuda.py) work through one table of raw device pointers (`R.tab`): every state tensor is
allocated once here and must never be re-bound (weapons.tick re-binds a few keys: Rounds.stabilize() copies them back).
Time stamps are in the batch clock (`gtick`); `t0[e]` is the clock value at which match e (re)started.
"""
import json, math, os
import numpy as np
import torch

from fpssim.common import CATALOG, EQUIP
from . import tables as TB
from . import rounds_cuda as RC

NEVENT = len(RC.EVENTS)
EV = {n: i for i, n in enumerate(RC.EVENTS)}
NW = len(TB.WEAPON_IDS)
BUY_IDS = list(TB.WEAPON_IDS) + ['armor', 'helmet', 'defusekit'] + list(TB.UTIL_IDS) + ['c4']
BID = {k: i for i, k in enumerate(BUY_IDS)}
NB = len(BUY_IDS)
ID_ARMOR, ID_HELMET, ID_KIT, ID_UTIL0, ID_C4 = BID['armor'], BID['helmet'], BID['defusekit'], BID['hegrenade'], BID['c4']
PHASE = dict(buy=0, live=1, post=2, finished=3)
BOMB = dict(held=0, world=1, planted=2, defused=3, exploded=4)
MAXPAD, MAXZ, MAXV = 16, 4, 128
SNAP = ('w_ammo', 'w_reserve', 'w_shots', 'w_shot_ready', 'w_reload_until', 'w_reload_commit', 'w_last_shot', 'w_reload_committed',
        'w_reload_shell', 'w_pressed', 'w_inacc', 'w_rs', 'active', 'switch_until', 'zoom')
M64 = (1 << 64) - 1


# ------------------------------------------------------------------------------------------------------------ RNG twin
def _mix(z):
    z &= M64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M64
    return z ^ (z >> 31)


def u01_py(seed, purpose, a=0, b=0, c=0):
    """python twin of the CUDA u01(): uniform double in [0,1) from (seed, purpose, a, b, c).
    purposes: 1 spread (a = match tick, b = seat*16 + pellet, c = 0..3), 2 spawn (a = round index, b = side*16 + pad),
    3 bomb carrier (a = round index), 4 initial attack team."""
    h = _mix((seed & M64) ^ 0xA0761D6478BD642F)
    for v in (purpose, a, b, c):
        h = _mix((h + (v & M64) * 0x9E3779B97F4A7C15 + 0x632BE59BD9B4E019) & M64)
    return (h >> 11) * (1.0 / 9007199254740992.0)


# ------------------------------------------------------------------------------------------------------------ registry
def registry(E, K):
    """ordered list of (name, dtype, shape-after-E, init) for every per-match tensor.  init None = set by reset() itself."""
    f64, i32, i64, i8, b, i16 = torch.float64, torch.int32, torch.int64, torch.int8, torch.bool, torch.int16
    r = []
    a = lambda n, dt, sh=(), init=0: r.append((n, dt, tuple(sh), init))
    # ---- match
    a('tick', i32); a('t0', i32); a('round_index', i32, (), -1); a('round_start_tick', i32); a('live_end', i32); a('post_end', i32, (), -1)
    a('phase', i8); a('round_over', b); a('finished', b); a('winner', i8, (), -1); a('attack_team', i8)
    a('round_wins', i32, (2,)); a('loss_level', i32, (2,), 1)
    a('bomb_state', i8); a('bomb_carrier', i8, (), -1); a('bomb_pos', f64, (3,)); a('bomb_site', i8, (), -1)
    a('bomb_planted_tick', i32, (), -1); a('bomb_fuse_end', i32, (), -1); a('bomb_drop_tick', i32, (), -1)
    a('bomb_dropper', i8, (), -1); a('bomb_actor', i8, (), -1); a('bomb_duration', i32); a('bomb_planter', i8, (), -1)
    a('wi_count', i32); a('wi_overflow', i32); a('rng_seed', i64, (), None); a('ev_count', i32, (NEVENT,))
    a('hist_winner', i8, (32,), -1); a('hist_reason', i8, (32,), -1); a('hist_tick', i32, (32,), -1); a('round_started', b)
    # ---- world items
    a('wi_valid', i8, (K,)); a('wi_kind', i32, (K,)); a('wi_owner', i8, (K,)); a('wi_tick', i32, (K,)); a('wi_pos', f64, (K, 3))
    for n in ('wi_ammo', 'wi_reserve', 'wi_shots', 'wi_shot_ready', 'wi_reload_until', 'wi_reload_commit', 'wi_last_shot'):
        a(n, i32, (K,))
    for n in ('wi_committed', 'wi_shell', 'wi_pressed'):
        a(n, b, (K,))
    a('wi_inacc', f64, (K,)); a('wi_rs', f64, (K, 7))
    # ---- players: movement (gpusim.move state)
    P = (10,)
    for n in ('x', 'y', 'z', 'vx', 'vy', 'vz', 'yaw', 'pitch', 'mv_jump_buffer', 'step_distance'):
        a(n, f64, P)
    a('height', f64, P, 1.8)
    a('crouch', b, P); a('grounded', b, P, 1); a('last_jump', b, P); a('alive', b, P, 1); a('mv_stand_blocked', b, P)
    a('movement_mode', i8, P); a('ladder_id', i16, P, -1); a('mv_reattach_until', i32, P)
    # ---- players: game state
    a('hp', f64, P, 100.0); a('armor', f64, P); a('helmet', b, P); a('kit', b, P); a('cash', i32, P)
    a('has_bomb', b, P); a('objective', i8, P); a('obj_progress', i32, P); a('pickup_ready_tick', i32, P)
    a('death_tick', i32, P, -1); a('blind_until', i32, P); a('blind_start', i32, P); a('throw_ready', i32, P)
    a('nades', i32, (10, 6)); a('nade_bought', i32, (10, 6))
    a('pur_has', i8, (10, NB)); a('pur_price', i32, (10, NB)); a('pur_count', i32, (10, NB))
    a('kills', i32, P); a('deaths', i32, P); a('last_fire_tick', i32, P, -10 ** 9)
    a('rcpt', i32, (10, 4, 3))
    # ---- players: weapons (gpusim.weapons state)
    for n in ('w_wid',):
        a(n, i64, (10, 3), -1)
    for n in ('w_ammo', 'w_reserve', 'w_shots', 'w_shot_ready'):
        a(n, i64, (10, 3))
    for n in ('w_reload_until', 'w_reload_commit', 'w_last_shot'):
        a(n, i64, (10, 3), -1)
    for n in ('w_reload_committed', 'w_reload_shell', 'w_pressed'):
        a(n, b, (10, 3))
    a('w_inacc', f64, (10, 3)); a('w_rs', f64, (10, 3, 7))
    a('active', i64, P, 1); a('switch_until', i64, P); a('zoom', i64, P); a('last_knife', i64, P, -10 ** 6)
    # ---- held actions (SPEC action tensors, persistent) and per-tick derived inputs
    for n in ('act_move_forward', 'act_move_right', 'act_yaw_rate', 'act_pitch_rate', 'act_climb'):
        a(n, f64, P)
    for n in ('act_jump', 'act_crouch', 'act_walk', 'act_fire', 'act_fire2', 'act_reload', 'act_use_plant', 'act_use_defuse'):
        a(n, b, P)
    a('act_weapon_slot', i8, P, 1); a('act_zoom', i8, P); a('act_ladder', i8, P)
    a('act_buy', i32, P, -1); a('act_refund', i32, P, -1); a('act_drop', i8, P); a('act_pickup', i32, P, -1)
    a('act_throw_kind', i32, P, -1); a('act_throw_state', i32, P); a('act_throw_mode', i32, P)
    a('mv_fwd', f64, P); a('mv_right', f64, P); a('mv_jump', b, P); a('mv_maxspeed', f64, P, 6.0); a('mv_locked', b, P)
    a('alive_run', b, P); a('w_can_act', b, P); a('w_want', i64, P, -1); a('w_zoomreq', i64, P, -1)
    # pre-tick copy of the weapon state (Arena never ticks a player that an earlier seat killed in the same tick: its dropped
    # weapons must carry the state from before the batched weapons tick)
    d = {n: (dt, sh) for n, dt, sh, _ in r}
    for k in SNAP:
        a('ps_' + k, d[k][0], d[k][1])
    return r


def _teams_of(meta):
    return 'T' in meta['teams'], 'CT' in meta['teams']


class Rounds:
    def __init__(self, E, device, static, mesh=None, K=48, tables=None):
        """static: the map JSON ('map' dict); mesh: gpusim.ray.GpuMesh (LOS for bomb pickup / defuse) or None."""
        self.E, self.K, self.dev = E, K, torch.device(device)
        dv = self.dev
        static = static.get('map', static)
        self.static = static
        self.T = tables if tables is not None else TB.build(dv)
        T = self.T
        # ------------------------------------------------ per-match tensors
        self.reg = registry(E, K)
        self.s = {}
        self.init = {}
        for n, dt, sh, init in self.reg:
            self.s[n] = torch.zeros((E,) + sh, dtype=dt, device=dv)
            self.init[n] = init
        # ------------------------------------------------ static tables
        st = {}
        for k in ('magazine', 'reserve', 'slot', 'kill_reward', 'n_zoom'):
            st['T_' + k] = T[k].long().contiguous()
        for k in ('damage', 'range', 'range_mod', 'armor_ratio', 'head_mult', 'max_speed', 'scoped_max_speed', 'pen_power'):
            st['T_' + k] = T[k].double().contiguous()
        st['T_acc'] = T['acc'].double().reshape(-1).contiguous()
        st['T_acc_always1'] = T['acc_index_always1'].bool().contiguous()
        from fpssim import accuracy as ACCM
        st['T_acc_has'] = torch.tensor([w in ACCM.ACC for w in TB.WEAPON_IDS], device=dv)
        st['knife_id'] = torch.tensor([TB.KNIFE], dtype=torch.long, device=dv)
        price = np.zeros(NB); bt = np.zeros(NB, bool); bct = np.zeros(NB, bool); maxc = np.ones(NB)
        for i, w in enumerate(TB.WEAPON_IDS):
            m = CATALOG['weapons'][i]; price[i] = m['price']; bt[i], bct[i] = _teams_of(m)
        for k in ('armor', 'helmet', 'defusekit') + tuple(TB.UTIL_IDS) + ('c4',):
            eq = EQUIP[k]; i = BID[k]; price[i] = eq['price']; maxc[i] = eq.get('max_count', 1)
            bt[i], bct[i] = 'T' in eq.get('teams', []), 'CT' in eq.get('teams', [])
        st['buy_price'] = torch.tensor(price, dtype=torch.long, device=dv); st['buy_maxc'] = torch.tensor(maxc, dtype=torch.long, device=dv)
        st['buy_t'] = torch.tensor(bt, device=dv); st['buy_ct'] = torch.tensor(bct, device=dv)
        # map
        cfg_i = np.zeros(16, np.int64); cfg_d = np.zeros(8)
        pad = np.zeros((2, MAXPAD, 4))
        for side, name in enumerate(('attack', 'defend')):
            p = np.array(static['spawn_pads_by_side'][name], np.float64).reshape(-1, 4)
            assert len(p) <= MAXPAD
            pad[side, :len(p)] = p; cfg_i[1 + side] = len(p)
        if static.get('buy_zones_by_side'):
            raise NotImplementedError('polygon buy zones are not ported (all five maps use spawn-pad circles)')
        bz = static.get('buy_zone') or {}
        cfg_d[1] = bz.get('radius', 9.0); cfg_d[2] = bz.get('vertical_limit', 3.0)
        wb = bool(static.get('boxes'))
        cfg_i[5] = int(wb); cfg_d[0] = 0.00011 if wb else 0.0
        pz = static.get('plant_zones')
        pz_poly = np.zeros((MAXZ, MAXV, 2)); pz_n = np.zeros(MAXZ, np.int64); pz_z = np.zeros((MAXZ, 2)); pz_c = np.zeros((MAXZ, 4))
        if pz:
            self.site_names = list(pz.keys())
            assert len(pz) <= MAXZ
            for k, (sid, zone) in enumerate(pz.items()):
                poly = np.array(zone['polygon'], np.float64); assert len(poly) <= MAXV, len(poly)
                pz_poly[k, :len(poly)] = poly; pz_n[k] = len(poly); pz_z[k] = (zone['min_z'], zone['max_z'])
            cfg_i[3] = len(pz)
        else:
            self.site_names = [s['id'] for s in static['sites']]
            for k, sdef in enumerate(static['sites']):
                pz_c[k] = (sdef['center'][0], sdef['center'][1], sdef['radius'], 0)
            cfg_i[3] = len(static['sites']); cfg_i[6] = 1
        cfg_i[0] = K; cfg_i[4] = int(mesh is not None); cfg_i[7] = TB.WID['pistol']; cfg_i[8] = TB.WID['usp']
        st['cfg_i'] = torch.tensor(cfg_i, device=dv); st['cfg_d'] = torch.tensor(cfg_d, device=dv)
        st['pad'] = torch.tensor(pad.reshape(-1), device=dv)
        st['pz_poly'] = torch.tensor(pz_poly.reshape(-1), device=dv); st['pz_n'] = torch.tensor(pz_n, device=dv)
        st['pz_z'] = torch.tensor(pz_z.reshape(-1), device=dv); st['pz_circ'] = torch.tensor(pz_c.reshape(-1), device=dv)
        if mesh is not None:
            t = mesh.t
            st['bv_lo'] = t['node_lo'].contiguous(); st['bv_hi'] = t['node_hi'].contiguous(); st['bv_left'] = t['node_left'].contiguous()
            st['bv_start'] = t['node_start'].contiguous(); st['bv_count'] = t['node_count'].contiguous()
            st['bv_tri'] = mesh.tri64.contiguous()
        else:
            z = lambda dt: torch.zeros(3, dtype=dt, device=dv)
            st['bv_lo'] = z(torch.float32); st['bv_hi'] = z(torch.float32); st['bv_left'] = z(torch.int32)
            st['bv_start'] = z(torch.int32); st['bv_count'] = z(torch.int32); st['bv_tri'] = z(torch.float64)
        self.st = st
        self.mesh = mesh
        # ------------------------------------------------ pointer table + extension
        names = [n for n, *_ in self.reg] + list(st.keys())
        self._tensors = [self.s[n] for n, *_ in self.reg] + [st[k] for k in st]
        self.fields = [(n, t.dtype) for n, t in zip(names, self._tensors)]
        self.tab = torch.tensor([t.data_ptr() for t in self._tensors], dtype=torch.int64, device=dv)
        self.persist = dict(self.s)
        consts = dict(NB=NB, NW=NW, ID_ARMOR=ID_ARMOR, ID_HELMET=ID_HELMET, ID_KIT=ID_KIT, ID_UTIL0=ID_UTIL0, ID_C4=ID_C4)
        self.k = RC.ext(self.fields, consts)
        self.step_n = 0

    # ---------------------------------------------------------------------------------------------------------- reset
    def reset(self, mask=None, seed=0, gtick=0, epoch=0):
        """Arena.reset for the masked matches (all if None).  Match e uses RNG seed `seed + e + epoch * E` (+ a large odd
        multiplier), so partial resets get fresh streams."""
        E = self.E
        m = torch.ones(E, dtype=torch.bool, device=self.dev) if mask is None else mask.to(self.dev).bool()
        for n, dt, sh, init in self.reg:
            if init is None: continue
            t = self.s[n]
            if mask is None: t.fill_(init)
            else: t.masked_fill_(m.view((E,) + (1,) * len(sh)), init)        # no host sync (graph capturable)
        # seed / gtick / epoch: python ints or device scalars (GpuSim's auto-reset graph)
        ar = torch.arange(E, dtype=torch.int64, device=self.dev)
        sd = seed.long() if torch.is_tensor(seed) else int(seed)
        ep = epoch.long() if torch.is_tensor(epoch) else int(epoch)
        seeds = sd * 1000003 + ar + ep * E
        self.s['rng_seed'] = self.persist['rng_seed']
        self.persist['rng_seed'].copy_(torch.where(m, seeds, self.persist['rng_seed']))
        for k in ('tick', 't0'):
            t = self.s[k]
            if torch.is_tensor(gtick): t.copy_(torch.where(m, gtick.to(t.dtype), t))
            else: t.masked_fill_(m, int(gtick))
        self.k.init(self.tab, m, E)

    def match_seed(self, e):
        return int(self.s['rng_seed'][e].item())

    # ---------------------------------------------------------------------------------------------------- kernels
    def one_shots(self):
        self.k.one_shots(self.tab, self.E)

    def tick_begin(self, gtick):
        """gtick: int, or an int64 device scalar (read by the kernel: graph capturable)."""
        if torch.is_tensor(gtick): self.k.tick_begin_dev(self.tab, self.E, gtick)
        else: self.k.tick_begin(self.tab, self.E, int(gtick))

    def obj_step(self):
        self.k.obj_step(self.tab, self.E)

    def end_tick(self):
        self.k.end_tick(self.tab, self.E)

    def kill_pass(self, alive_before, killer):
        """Arena._kill for victims that grenade damage killed (alive_before [E,10] bool, killer [E,10] int32, -1 = none)."""
        self.k.kill_pass(self.tab, self.E, alive_before.contiguous(), killer.to(torch.int32).contiguous())

    def snapshot_weapons(self):
        """copy the weapon state before gpusim.weapons.tick (see registry): used to give players killed earlier in the same tick the
        CPU's un-ticked state."""
        if not hasattr(self, '_snap'):
            self._snap = ([self.s['ps_' + k] for k in SNAP], [self.persist[k] for k in SNAP])
        torch._foreach_copy_(*self._snap)

    def stabilize(self, keys=('active', 'zoom', 'switch_until', 'w_inacc', 'last_knife')):
        """gpusim.weapons.tick re-binds a few state keys: copy them back into the registered (pointer table) tensors."""
        for k in keys:
            p = self.persist[k]
            if self.s[k] is not p:
                p.copy_(self.s[k]); self.s[k] = p

    # ---------------------------------------------------------------------------------------------------- snapshots
    def snapshot(self, e):
        """python dict of match e (numpy / python scalars) for tests and debugging."""
        s = {k: v[e].cpu().numpy() for k, v in self.s.items() if v.dim() >= 1 and v.shape[0] == self.E}
        return s

    def events(self, e):
        c = self.s['ev_count'][e].cpu().numpy()
        return {n: int(c[i]) for i, n in enumerate(RC.EVENTS)}
