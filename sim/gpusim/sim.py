"""GpuSim: E batched 5v5 matches on one GPU, tick order identical to fpssim.arena.Arena._tick_inner (docs/gpu_sim.md).

    from gpusim.sim import GpuSim
    sim = GpuSim('dust2-cs2-v54', E=4096, device='cuda', seed=1)
    sim.reset()
    sim.step(actions)           # one 20 Hz decision = 3 physics ticks; actions: dict of [E,10] tensors (SPEC) or sim.actions_from_json(...)
    sim.s                       # state dict (SPEC names, see gpusim/rounds.py registry)

Per tick: rounds.tick_begin (clock, phase, buy-freeze, weapon inputs) -> GpuMover.step -> resolve_players -> weapons.tick ->
Damage.run (spread, bullets, damage, kills) -> rounds.obj_step (plant/defuse/pickup bomb) -> grenades (live only) + grenade
kills -> rounds.end_tick (bomb fuse, win conditions, next round).  The decision-boundary one-shots (buy / refund / drop /
pickup) run before the 3 ticks, seats in order.
"""
import json, os
import numpy as np
import torch

from . import tables as TB
from . import weapons as GW
from . import rounds as RD
from .rounds import Rounds, BUY_IDS, BID
from .damage import Damage

ROOT = os.environ.get('FPS_ROOT', './fps_data')
STATIC_DIR = os.environ.get('FPS_STATIC_DIR', ROOT + '/data/static')
ASSET_DIR = os.environ.get('FPS_ASSET_DIR', ROOT + '/ref/assets')
MESH_FILES = {'dust2-cs2-v54': ('map_positions.f32', 'map_penetration-materials.u8'),       # same table as fpssim.arena.MESH_FILES
              'mirage-cs2-v54': ('mirage_positions.f32', 'mirage_penetration-materials.u8')}
FLOOR_DIR = os.environ.get('FPS_FLOOR_DIR', ROOT + '/data/floor')
GRENADES = ('hegrenade', 'flashbang', 'smokegrenade', 'molotov', 'incgrenade', 'decoy')

# SPEC action keys -> persistent held-action tensors (Rounds registry names) and their defaults
ACTION_MAP = dict(move_forward='act_move_forward', move_right='act_move_right', yaw_rate='act_yaw_rate', pitch_rate='act_pitch_rate',
                  climb='act_climb', jump='act_jump', crouch='act_crouch', walk='act_walk', fire='act_fire', fire2='act_fire2',
                  reload='act_reload', use_plant='act_use_plant', use_defuse='act_use_defuse', weapon_slot='act_weapon_slot',
                  zoom='act_zoom', ladder='act_ladder', buy='act_buy', refund='act_refund', drop='act_drop', pickup='act_pickup',
                  throw_kind='act_throw_kind', throw_state='act_throw_state', throw_mode='act_throw_mode')
ACTION_DEFAULT = dict(weapon_slot=1, buy=-1, refund=-1, pickup=-1, throw_kind=-1)
DROP_CODE = {'none': 0, 'primary': 1, 'slot0': 1, '0': 1, 'secondary': 2, 'slot1': 2, '1': 2, 'bomb': 3}


def load_map(map_id, device, load_cpu_mesh=True):
    """(static dict, CPU TriMesh or None, GpuMesh or None, Shooter).  Dust2: platform mesh + materials; whitebox: triangulated boxes."""
    from fpssim.trimesh import TriMesh
    from .ray import GpuMesh
    from .shoot import Shooter
    d = json.load(open(os.path.join(STATIC_DIR, f'static_{map_id}.json')))
    static = d['map'] if 'map' in d else d
    if map_id in MESH_FILES and static.get('collision'):
        col = static['collision'][0]; pos, mat = MESH_FILES[map_id]
        M = TriMesh(os.path.join(ASSET_DIR, pos), col['basis'], col['offset'], os.path.join(ASSET_DIR, mat))
        G = GpuMesh(M.tri, device)
        return static, M, G, Shooter(device, G, M.tri, M.mat)
    if static.get('boxes'):
        from tools.meshmove_wb_validate import make_wb_world
        w = make_wb_world(static, './fps_data/tmp/meshcache')
        # MeshWorld.tri is in raw cs2-web coordinates (y up): bring it to platform world coordinates (world = basis @ p + offset)
        tri = np.einsum('ij,tkj->tki', w.B, np.asarray(w.tri, np.float64)) + w.o
        G = GpuMesh(tri, device)
        return static, None, G, Shooter(device, wb_tri=tri)
    raise NotImplementedError(f'{map_id}: no mesh')


class _TorchProxy:
    """stands in for `torch` inside gpusim.weapons while its tick is captured in a CUDA graph with a device tick counter:
    torch.full_like(x, t) with a tensor t would sync (.item()), everything else is forwarded."""
    def __getattr__(self, n):
        return getattr(torch, n)

    @staticmethod
    def full_like(x, v, **kw):
        return torch.zeros_like(x) + v if torch.is_tensor(v) else torch.full_like(x, v, **kw)


class GpuSim:
    def __init__(self, map_id, E, device='cuda', seed=0, K=48, auto_reset=False, grenades=True, collide=True, audio=False, graph=True):
        self.map_id, self.E, self.dev = map_id, E, torch.device(device)
        self.seed = seed; self.auto_reset = auto_reset
        dv = self.dev
        self.static, self.cpu_mesh, self.mesh, self.shooter = load_map(map_id, dv)
        self.R = Rounds(E, dv, self.static, mesh=self.mesh, K=K)
        self.s = self.R.s; self.T = self.R.T
        from .move import GpuMover
        world = None
        if map_id in MESH_FILES and self.static.get('collision'):         # same world the CPU Arena builds for mesh maps
            from fpssim.meshmove import make_mesh_world
            world = make_mesh_world(self.static, ASSET_DIR, positions_name=MESH_FILES[map_id][0], cache_dir=FLOOR_DIR)
        self.mover = GpuMover(self.static, ASSET_DIR, dv, world=world)
        self.mover.ensure_state(self.s)
        self.collide = collide
        self.grenades = None
        if grenades:
            try:
                from .grenades import GpuGrenades
                self.grenades = GpuGrenades(self.static, self.cpu_mesh, E, dv)
            except Exception as ex:                        # grenade module not delivered / not buildable yet: placeholder = no utility
                print('GpuSim: grenades unavailable (%s: %s); throws are ignored' % (type(ex).__name__, ex))
        self.damage = Damage(self.R, self.shooter, self.grenades)
        s = self.s
        self._mstate = {k: v for k, v in s.items() if k in
                        ('x', 'y', 'z', 'vx', 'vy', 'vz', 'yaw', 'pitch', 'height', 'step_distance', 'crouch', 'grounded', 'last_jump',
                         'movement_mode', 'ladder_id', 'mv_jump_buffer', 'mv_stand_blocked', 'mv_reattach_until')}
        self._mstate['alive'] = s['alive_run']
        self._mact = dict(move_forward=s['mv_fwd'], move_right=s['mv_right'], yaw_rate=s['act_yaw_rate'], pitch_rate=s['act_pitch_rate'],
                          climb=s['act_climb'], jump=s['mv_jump'], crouch=s['act_crouch'], walk=s['act_walk'], ladder=s['act_ladder'])
        self._gact = dict(throw_kind=s['act_throw_kind'], throw_state=s['act_throw_state'], throw_mode=s['act_throw_mode'])
        self.no_fire2 = torch.zeros(E, 10, dtype=torch.bool, device=dv)       # the CPU Arena never passes fire2 to the weapon tick
        self.gtick = 0
        self.vis = None
        self.graph = None            # weapons-only graph (fallback)
        self.dgraph = None           # whole-decision graph (one_shots + 3 ticks), auto-reset graph
        self.rgraph = None
        self.t_dev = torch.zeros((), dtype=torch.int64, device=self.dev)       # device copy of gtick (kernels read it in graphs)
        self._seed_dev = torch.zeros((), dtype=torch.int64, device=self.dev)
        self._epoch = torch.zeros((), dtype=torch.int64, device=self.dev)
        self.audio = None
        if audio:
            from .obs import GpuAudio, ObsStatic, NB as _NB
            self.audio = GpuAudio(E, self.dev); self._onb = _NB
            self.obs_static = ObsStatic(self.static, self.dev)
            self._au_ev = torch.tensor([RD.EV[k] for k in ('plant_complete', 'defuse_complete', 'bomb_explode')], device=self.dev)   # device index (graph capture)
        self.graph_mode = None
        if graph:
            try:
                self._build_decision_graph(); self.graph_mode = 'decision'
            except Exception as ex:      # e.g. a module that syncs inside its tick: fall back to the weapons-only graph
                print('GpuSim: whole-decision CUDA graph unavailable (%s: %s); using the weapons graph' % (type(ex).__name__, ex))
                self.dgraph = self.rgraph = None
                torch.cuda.synchronize()
                self._build_weapon_graph(); self.graph_mode = 'weapons'

    # ------------------------------------------------------------------------------------------------------------ CUDA graph
    def _weapon_body(self, t):
        R, s = self.R, self.s
        R.snapshot_weapons()
        out = GW.tick(s, self.T, t, s['alive_run'], s['w_can_act'], s['act_fire'], self.no_fire2, s['act_reload'], s['w_want'],
                      s['w_zoomreq'], s['crouch'], ~s['grounded'])
        R.stabilize()
        return out

    def _build_decision_graph(self):
        """capture one decision (one_shots + 3 ticks, device clock t_dev) and the auto-reset step in two CUDA graphs.  Done before the
        first reset() with every match marked finished, so the warm-up executions only touch state that reset() re-initialises."""
        s = self.s
        keep = list(self.R.persist.values())                     # every tensor the warm-up runs may touch, restored afterwards
        if self.grenades is not None: keep += list(self.grenades.g.values())
        if self.audio is not None:
            au = self.audio; au.flush(copy=False)
            keep += [au.n, au.itype, au.src, au.pos, au.last_step, au.acc, au._dropped]
        keep += [self.mover.err]
        saved = [t.clone() for t in keep]
        s['finished'].fill_(True); s['alive'].zero_(); s['alive_run'].zero_()
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        real = GW.torch; GW.torch = _TorchProxy()
        try:
            with torch.cuda.stream(st):
                for _ in range(2):
                    self._decision_body(); self._reset_body()
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._decision_body()
            gr = torch.cuda.CUDAGraph()
            with torch.cuda.graph(gr, pool=g.pool()):
                self._reset_body()
        finally:
            GW.torch = real
            torch.cuda.synchronize()
            for t, v in zip(keep, saved): t.copy_(v)
            torch.cuda.synchronize()
        self.dgraph, self.rgraph = g, gr

    def _decision_body(self):
        self.R.one_shots()
        for _ in range(3):
            self.t_dev += 1
            self._tick_body(self.t_dev)

    def _reset_body(self):
        """auto-reset of the finished matches without a host sync (the RNG epoch advances once per decision with a finished match,
        like reset(mask))."""
        done = self.s['finished'].clone()
        self._epoch += done.any().long()
        self._reset_masked(done, self.t_dev)

    def _build_weapon_graph(self):
        """capture snapshot + weapons.tick + stabilize in one CUDA graph (the tick number is a device scalar).  Done before the first
        reset(): every player is dead then, so the warm-up executions do not touch any state."""
        self.t_dev = torch.zeros((), dtype=torch.int64, device=self.dev)
        self.s['alive_run'].zero_()
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        real = GW.torch; GW.torch = _TorchProxy()
        try:
            with torch.cuda.stream(st):
                for _ in range(2): self._weapon_body(self.t_dev)
            torch.cuda.current_stream().wait_stream(st)
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g):
                self._wout = self._weapon_body(self.t_dev)
        finally:
            GW.torch = real
        self.graph = g

    # ------------------------------------------------------------------------------------------------------------ reset
    def reset(self, seed=None, mask=None):
        """Arena.reset for all matches (or the masked ones: their clock restarts, RNG gets a new epoch)."""
        if seed is not None: self.seed = seed
        self._seed_dev.fill_(int(self.seed))
        if mask is None:
            self.gtick = 0; self._epoch.zero_()
        else:
            self._epoch += 1
        self.t_dev.fill_(self.gtick)
        self._reset_masked(mask, self.t_dev if mask is not None else self.gtick)
        return self.s

    def _reset_masked(self, mask, gtick):
        self.R.reset(mask, self._seed_dev, gtick, self._epoch)
        if self.grenades is not None:
            self.grenades.clear(mask)
        if self.audio is not None:
            self.audio.clear(mask)

    @property
    def epoch(self):
        return int(self._epoch)

    # ------------------------------------------------------------------------------------------------------------ actions
    def set_actions(self, actions):
        """copy a dict of [E,10] tensors (SPEC action keys; missing keys = default) into the held-action tensors."""
        s = self.s
        for k, name in ACTION_MAP.items():
            v = actions.get(k)
            dst = s[name]
            if v is None:
                dst.fill_(ACTION_DEFAULT.get(k, 0))
            else:
                dst.copy_(v if torch.is_tensor(v) else torch.as_tensor(v))

    def actions_from_json(self, acts):
        """acts: list over matches of 10 platform action dicts (or a single list of 10 for E=1) -> dict of [E,10] tensors.
        Differences to the platform contract: `buy`/`refund_item_id` are catalogue kinds (items are not addressable by id),
        `pickup_id` is the world-item slot index (int) or 'bomb', `drop` codes see DROP_CODE; unknown strings = no action."""
        if acts and isinstance(acts[0], dict): acts = [acts]
        E = len(acts)
        f = lambda: np.zeros((E, 10)); i = lambda v=0: np.full((E, 10), v, np.int64)
        out = {k: (f() if k in ('move_forward', 'move_right', 'yaw_rate', 'pitch_rate', 'climb') else
                   np.zeros((E, 10), bool) if k in ('jump', 'crouch', 'walk', 'fire', 'fire2', 'reload', 'use_plant', 'use_defuse')
                   else i(ACTION_DEFAULT.get(k, 0))) for k in ACTION_MAP}
        K = self.R.K
        for e, row in enumerate(acts):
            for q, a in enumerate(row):
                g = a.get
                for k in ('move_forward', 'move_right', 'yaw_rate', 'pitch_rate', 'climb'): out[k][e, q] = float(g(k, 0.0) or 0.0)
                out['jump'][e, q] = bool(g('jump', False)); out['crouch'][e, q] = bool(g('crouch', False) or g('duck', False))
                out['walk'][e, q] = bool(g('walk', False)); out['fire'][e, q] = bool(g('fire', False))
                out['fire2'][e, q] = bool(g('fire_secondary', False)); out['reload'][e, q] = bool(g('reload', False))
                u = g('use', 'none'); out['use_plant'][e, q] = u == 'plant'; out['use_defuse'][e, q] = u == 'defuse'
                out['weapon_slot'][e, q] = int(g('weapon_slot', 1))
                z = int(g('zoom_level', 0)); out['zoom'][e, q] = 1 if (g('ads') and z == 0) else z
                out['ladder'][e, q] = {'none': 0, 'attach': 1, 'detach': 2}[g('ladder', 'none')]
                b = g('buy', 'none'); out['buy'][e, q] = BID.get(b, -1) if b != 'none' else -1
                r = g('refund_item_id'); out['refund'][e, q] = BID.get(r, -1) if isinstance(r, str) else -1
                d = g('drop', 'none')
                out['drop'][e, q] = DROP_CODE.get(d, 0) if d in DROP_CODE else {1: 1, 2: 2}.get(TB.WEAPONS[d]['slot'], 0) if d in TB.WEAPONS else 0
                if u == 'pickup':
                    p = g('pickup_id'); out['pickup'][e, q] = K if p == 'bomb' else (int(p) if isinstance(p, int) else -1)
                tk = g('throw', 'none'); out['throw_kind'][e, q] = GRENADES.index(tk) if tk in GRENADES else -1
                out['throw_state'][e, q] = ('idle', 'prime', 'release', 'cancel').index(g('throw_state', 'idle'))
                out['throw_mode'][e, q] = ('full', 'lob', 'drop').index(g('throw_mode', 'full'))
        return {k: torch.as_tensor(v, device=self.dev) for k, v in out.items()}

    # ------------------------------------------------------------------------------------------------------------ stepping
    def step(self, actions=None, observe=False):
        """One decision: apply `actions`, one-shots (buy / refund / drop / pickup), then 3 physics ticks.  Returns the state dict, or the
        observation tensor [E,10,OBS_DIM] with observe=True (needs audio=True).  The audio buffer only keeps the last tick's emitters
        (GpuAudio.step drops the previous ones: the platform delivers the decision tick's emitters only)."""
        if actions is not None: self.set_actions(actions)
        use_graph = self.dgraph is not None and not self.damage.keep
        if use_graph:
            self.t_dev.fill_(self.gtick)
            self.dgraph.replay()
            self.gtick += 3
        else:
            self.R.one_shots()
            for k in range(3):
                self.tick()
        obs = self.observe() if observe else None
        if self.auto_reset:                                    # no host sync: masked reset of the finished matches
            if use_graph: self.rgraph.replay()
            else:
                self.t_dev.fill_(self.gtick); self._reset_body()
        return obs if observe else self.s

    def tick(self):
        """one physics tick (eager path; step() replays the whole-decision graph instead when it exists)."""
        self.gtick += 1
        return self._tick_body(self.gtick)

    def _tick_body(self, g):
        """g: int (eager) or the int64 device clock (graphs)."""
        R, s, T = self.R, self.s, self.T
        au = self.audio
        if au is not None:
            prev = au.snapshot(s); c0 = s['ev_count'][:, self._au_ev].clone()      # Arena._tick: audio state before the tick
        R.tick_begin(g)
        dev_clock = torch.is_tensor(g)
        mtick = s['tick'][:, None].expand(-1, 10) if dev_clock else g             # alive matches: s['tick'] == the batch clock
        self.mover.step(self._mstate, self._mact, s['mv_maxspeed'], s['mv_locked'], mtick)
        if self.collide:
            self.mover.resolve_players(self._mstate, s['mv_locked'])
        if self.graph is not None and not dev_clock:
            self.t_dev.fill_(g); self.graph.replay(); out = self._wout
        else:
            out = self._weapon_body(g)
        self.damage.run(out, s['tick'] if dev_clock else g)
        R.obj_step()
        th = gev = None
        if self.grenades is not None:
            live = (s['phase'] == RD.PHASE['live']) & ~s['finished']
            th = self.grenades.handle_throw(s, self._gact, live)
            alive_before = s['alive'].clone()
            gev = self.grenades.step(s, None, live)
            R.kill_pass(alive_before, gev['ev_killer'])
        R.end_tick()
        if self.grenades is not None:
            self.grenades.clear(s['round_started'])
        if au is not None:
            d = (s['ev_count'][:, self._au_ev] - c0) > 0
            events = dict(reload=out['ev_reload_start'], plant_complete=d[:, 0], defuse_complete=d[:, 1], bomb_explode=d[:, 2])
            if gev is not None:
                nb = self._onb
                if 'ev_bounce' in gev:
                    events['bounce'], events['bounce_pos'] = self._compact(gev['ev_bounce'] > 0, gev['ev_bounce_pos'], nb)
                # Arena.step: one fire_started emitter per 'detonate' event of a molotov / incgrenade (kinds 3, 4)
                det = (gev['ev_det'] > 0) & ((gev['ev_det_kind'] == 3) | (gev['ev_det_kind'] == 4))
                events['fire_started'], events['fire_started_pos'] = self._compact(det, gev['ev_det_pos'], nb)
            au.step(prev, s, events)
        return out

    @staticmethod
    def _compact(mask, pos, nb):
        """[E,P] mask + [E,P,3] -> first nb set entries per match ([E,nb], [E,nb,3]) (GpuAudio keeps nb events per tick)."""
        if mask.shape[1] <= nb:
            pad = nb - mask.shape[1]
            return torch.nn.functional.pad(mask, (0, pad)), torch.nn.functional.pad(pos, (0, 0, 0, pad))
        idx = mask.int().topk(nb, dim=1).indices
        return mask.gather(1, idx), pos.gather(1, idx[..., None].expand(-1, -1, 3))

    def audio_emitters(self, copy=True):
        """GpuAudio.flush with the burning fires (one fire_ambient per fire and decision, like Arena.step) -> emitter table."""
        fires = None
        if self.grenades is not None:
            g = self.grenades.g
            fires = (g['fi_pos'], g['fi_valid'] > 0)
        return self.audio.flush(fires, copy=copy)

    def observe(self):
        """float32 [E,10,OBS_DIM] legal observations of the current decision boundary (gpusim.vis + gpusim.obs)."""
        from .obs import build_obs, point_vis
        from . import vis as V
        s = self.s
        fn = self.grenades.smoke_blocks if self.grenades is not None else None
        if V.fused_ok(s, self.mesh):            # enemy visibility + bomb / item points in one trace (no host sync)
            vis, pv = V.trace(s, self.mesh, self.gtick, self.grenades,
                              points=(s['wi_pos'], s['wi_valid'], s['bomb_pos'], s['bomb_state']), tick_p=s['tick'])
        else:
            vis = self.visibility(); pv = None
        em = self.audio_emitters(copy=False)
        return build_obs(s, vis, self.obs_static, mesh=self.mesh, audio=em, smoke_fn=fn, pv=pv)

    # ------------------------------------------------------------------------------------------------------------ queries
    def visibility(self):
        """bool [E,10 obs,10 tgt,3 parts] (gpusim.vis)."""
        from .vis import visible
        fn = self.grenades.smoke_blocks if self.grenades is not None else None
        return visible(self.s, self.mesh, self.gtick, fn)

    def events(self, e):
        return self.R.events(e)
