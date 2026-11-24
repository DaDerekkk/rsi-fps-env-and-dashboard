"""CUDA player movement: batched port of fpssim.meshmove.MeshMover (cs2-web triangle-mesh collision), see docs/gpu_move.md.

    from gpusim.move import GpuMover
    mv = GpuMover(map_static, assets_dir, 'cuda:0')      # one per map (Dust2/Mirage mesh, or triangulated whitebox)
    st = mv.new_state((E, 10))                           # dict of tensors (or build your own, see below)
    mv.step(st, action, maxspeed, locked, tick)          # advances st in place by one 60 Hz tick (alive players only)
    mv.resolve_players(st, locked)                       # player-vs-player pass (gpusim.collide), after step()
    mv.errors()                                          # sticky overflow flags (candidate list kmax, grid cells, contacts)

state: tensors of identical shape (e.g. [E,10]), platform WORLD coordinates (z up), updated in place.  Names follow gpusim/SPEC.md
    float64: x y z vx vy vz yaw pitch height step_distance
    bool:    alive crouch grounded last_jump
    int8:    movement_mode (0 ground, 1 air, 2 ladder)
    int16:   ladder_id (-1 = none, else the index into the map's ladder list, i.e. into map_static['ladders'])
  extra keys owned by this module (prefix `mv_`, created on first use by ensure_state() if missing):
    float64: mv_jump_buffer (cs2-web jump buffer, seconds)          int32: mv_reattach_until (absolute tick, ladder re-attach lock)
    bool:    mv_stand_blocked (last stand-up request was refused by a ceiling)
  `height` (1.8 / 1.1) is derived from `crouch` and written back for the other modules.
action: dict, each value broadcastable to the state shape (any float/bool/int dtype; converted once, use prep_action() to
  pre-convert the held action of a decision step and reuse it for its 3 ticks):
    move_forward move_right yaw_rate pitch_rate climb (float, [-1,1])   jump crouch walk (bool)
    ladder (0 none, 1 attach, 2 detach)                                   missing keys = 0 / False
maxspeed: weapon max speed M in m/s (tensor or float).  locked: objective lock (in plant/defuse): velocity 0, forced crouch.
tick: current match tick, int or an int tensor shaped [E] (per match) / like state (per player); only used for the 12-tick
  ladder re-attach lock.  freeze: optional bool (like Arena's buy phase): vx = vy = 0, no forward/right input, no jump.
Dead players (alive == False) are not touched.  Player-vs-player contact is NOT done here (MeshMover with others=()).
Memory: one candidate list of `kmax` int32 per player (kmax * E * 10 * 4 bytes, 2.7 GB at E = 16384 with the default 4096; the longest
list seen on Dust2 has 1244 triangles) plus the mesh (Dust2: 170 MB).  GPUMOVE_COUNT=1 builds a profiling variant (debug only).
"""
import math, os
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline

from fpssim import meshmove as MM
from fpssim.movement import MoveParams

KMAX = 4096          # default max candidate triangles per player per tick (error bit 1 if exceeded); GpuMover(kmax=...)
MAXC = 96            # max contacts kept per hullIntersect for collide() (error bit 4)
MAXCELLS = 64        # max broadphase cells per tick box (error bit 2)
BLOCK = 128          # threads per block of the step kernel (multiple of 32: a warp is the unit of cooperation)

STATE_F64 = ('x', 'y', 'z', 'vx', 'vy', 'vz', 'yaw', 'pitch', 'mv_jump_buffer', 'step_distance', 'height')
STATE_BOOL = ('crouch', 'grounded', 'last_jump', 'alive', 'mv_stand_blocked')
STATE_INT = (('movement_mode', torch.int8), ('ladder_id', torch.int16), ('mv_reattach_until', torch.int32))
ACT_F64 = ('move_forward', 'move_right', 'yaw_rate', 'pitch_rate', 'climb')
ACT_BOOL = ('jump', 'crouch', 'walk')
MODES = {'ground': 0, 'air': 1, 'ladder': 2}
LADDER = {'none': 0, 'attach': 1, 'detach': 2}

_mod = None
COUNT = bool(os.environ.get('GPUMOVE_COUNT'))     # profiling build with event counters (debug only)


def _consts():
    p = MoveParams()
    dt = p.dt
    n = max(3, math.ceil(dt * 180))
    c = dict(R=MM.PLAYER_RADIUS, STAND_H=MM.STAND_HEIGHT, CROUCH_H=MM.CROUCH_HEIGHT, GRAV=MM.GRAVITY,
             JUMP_V=MM.JUMP_SPEED, STEP_H=MM.STEP_HEIGHT, STEP_DROP=MM.STEP_HEIGHT + .06, JBUF_T=MM.JUMP_BUFFER_TIME,
             CJLIFT=MM.CROUCH_JUMP_LIFT, FRIC=MM.SV_FRICTION, STOP=MM.SV_STOPSPEED, ACC=MM.SV_ACCELERATE,
             AIRACC=MM.SV_AIRACCELERATE, AIRCAP=MM.SV_AIR_WISHSPEED_CAP, DT=dt, HSUB=dt / n,
             CROUCH_F=2.15 / 6.0, WALK_F=2.7 / 6.0, YAW_RATE=p.yaw_rate, PITCH_RATE=p.pitch_rate,
             PITCH_LIM=p.pitch_limit, HW=0.3, WALL_SHRINK=1e-6 - 1e-8)
    lines = [f'#define MV_{k} ({float(v)!r})' for k, v in c.items()]
    if COUNT: lines.append('#define MV_COUNT 1')
    lines += [f'#define MV_NSUB {n}', f'#define MV_REATTACH {int(p.reattach_ticks)}', f'#define MV_MAXC {MAXC}', f'#define MV_MAXCELLS {MAXCELLS}', f'#define MV_BLOCK {BLOCK}']
    # exact double-double constants for the correctly rounded sin/cos
    from fractions import Fraction
    PI = Fraction('3.14159265358979323846264338327950288419716939937510582097494459230781640628620899862803482534211706798214808651')

    def split(f, n=2):
        out = []
        for _ in range(n):
            h = float(f); out.append(h); f -= Fraction(h)
        return out
    NT = 15
    fact = lambda k: math.factorial(k)
    sin_c = [v for k in range(NT) for v in split(Fraction((-1) ** k, fact(2 * k + 1)))]
    cos_c = [v for k in range(NT) for v in split(Fraction((-1) ** k, fact(2 * k)))]
    arr = lambda xs: '{' + ', '.join(repr(float(x)) for x in xs) + '}'
    lines += [f'#define MV_NTAYLOR {NT}', f'#define MV_SIN_INIT {arr(sin_c)}', f'#define MV_COS_INIT {arr(cos_c)}',
              f'#define MV_PIO2_INIT {arr(split(PI / 2, 3))}']
    return '\n'.join(lines) + '\n'


def _ext():
    global _mod
    if _mod is None:
        os.environ.setdefault('TORCH_EXTENSIONS_DIR', './fps_data/tmp/torch_ext')
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'move_kernels.cu')).read()
        cpp = ('void step(std::vector<torch::Tensor> wt, std::vector<double> geo, std::vector<torch::Tensor> st, '
               'std::vector<torch::Tensor> ac, torch::Tensor tick_t, int64_t tick_div, torch::Tensor cbuf, torch::Tensor nbuf, torch::Tensor err, int64_t tick, int64_t kmax);\n'
               'void resolve(std::vector<torch::Tensor> wt, std::vector<double> geo, std::vector<torch::Tensor> st, torch::Tensor cbuf, '
               'torch::Tensor err, int64_t E, int64_t NP, bool wall, bool swept, int64_t kmax);\n'
               'torch::Tensor dbg_hypot(torch::Tensor a, torch::Tensor b);\ntorch::Tensor dbg_trig(torch::Tensor a);\n'
               + ('torch::Tensor dbg_counters(bool reset);\ntorch::Tensor dbg_warps(int64_t nw, bool reset);\ntorch::Tensor dbg_phases(int64_t nw);\n' if COUNT else '') +
               'torch::Tensor dbg_cand(std::vector<torch::Tensor> wt, std::vector<double> geo, torch::Tensor pv, torch::Tensor cbuf, torch::Tensor err, int64_t kmax);\n')
        _mod = load_inline(('fpsmove_cnt' if COUNT else 'fpsmove'), cpp_sources=cpp, cuda_sources=_consts() + src,
                           functions=['step', 'resolve', 'dbg_hypot', 'dbg_trig', 'dbg_cand'] + (['dbg_counters', 'dbg_warps', 'dbg_phases'] if COUNT else []),
                           extra_cuda_cflags=['-O3', '-fmad=false', '--prec-div=true', '--prec-sqrt=true'],
                           verbose=False)
    return _mod


def make_world(map_static, assets_dir, cache_dir='./fps_data/tmp/meshcache', positions_name=None):
    """CPU MeshWorld exactly as the validated CPU path builds it: real mesh for Dust2/Mirage (collision[0]), triangulated
    boxes (tools/meshmove_wb_validate.make_wb_world) for the whitebox maps (static map has boxes)."""
    if map_static.get('boxes'):
        from tools.meshmove_wb_validate import make_wb_world
        return make_wb_world(map_static, cache_dir)
    return MM.make_mesh_world(map_static, assets_dir, positions_name=positions_name, cache_dir=cache_dir)


def pack_world(w):
    """Triangle records [T, 84] float64 (see move_kernels.cu) + float32 AABBs + axis-0 data [T, 12] + grid + geometry vector + ladders."""
    T = len(w.tri)
    rec = np.zeros((T, 84), np.float64)
    rec[:, 0:3] = w.tlo; rec[:, 3:6] = w.thi; rec[:, 6:9] = w.llo; rec[:, 9:12] = w.lhi
    rec[:, 12:15] = w.surface
    rec[:, 15] = (w.valid.astype(np.int64) << np.arange(13)).sum(1)
    rec[:, 16:55] = w.axes.reshape(T, 39)
    rec[:, 55:68] = w.pmin; rec[:, 68:81] = w.pmax
    # the kernel tests the three world axes (1..3 = y, x, z) from tlo/thi: check that this is exact
    for k, i in ((1, 1), (2, 0), (3, 2)):
        assert np.array_equal(w.pmin[:, k], w.tlo[:, i]) and np.array_equal(w.pmax[:, k], w.thi[:, i]) and w.valid[:, k].all()
    # the float32 broad-phase filter is exact only because the triangle bounds are float32 values (mesh file is float32)
    tb = np.zeros((T, 3, 4), np.float32)
    tb[:, 0, :3] = w.tlo; tb[:, 1, :3] = w.thi
    assert np.array_equal(tb[:, 0, :3].astype(np.float64), w.tlo) and np.array_equal(tb[:, 1, :3].astype(np.float64), w.thi)
    # SAT axis 0 (surface normal) in float32 for the cheap separation test; an invalid (zero) normal never separates
    ok0 = w.valid[:, 0]
    tb[:, 0, 3] = w.axes[:, 0, 0]; tb[:, 1, 3] = w.axes[:, 0, 1]; tb[:, 2, 0] = w.axes[:, 0, 2]
    tb[:, 2, 1] = np.where(ok0, w.pmin[:, 0], -1e30); tb[:, 2, 2] = np.where(ok0, w.pmax[:, 0], 1e30)
    # the world <-> raw basis must be a signed permutation (collide.py transforms boxes to boxes)
    assert np.all(np.sort(np.abs(w.B).ravel())[-3:] == 1.0) and np.count_nonzero(w.B) == 3
    vr = w.vertical or {}
    p = MoveParams()
    geo = [float(w.nx), float(w.nz), float(w.gx0), float(w.gz0), float(w.cell)] + \
        [float(v) for v in w.B.reshape(-1)] + [float(v) for v in w.Bt.reshape(-1)] + [float(v) for v in w.o] + \
        [float(vr.get('attach_reach', p.ladder_attach_reach)), float(vr.get('climb_speed', p.ladder_climb_speed))]
    lad = np.array([[l['rail'][0], l['rail'][1], l['bottom'], l['top'], l['bottom_exit'][0], l['bottom_exit'][1]]
                    for l in w.ladders], np.float64).reshape(-1, 6)
    return rec, tb.reshape(T, 12), w._flat.astype(np.int32), w._start.astype(np.int32), geo, lad


class GpuMover:
    """Batched MeshMover for one map.  `GpuMover(map_static, assets_dir, device)`; `world=` takes a ready CPU MeshWorld instead.
    wall_check_default: what gpusim.collide uses when not told (True for whitebox maps, as in the CPU Arena, see collide.py)."""

    def __init__(self, map_static=None, assets_dir=None, device='cuda', world=None, whitebox=None,
                 cache_dir='./fps_data/tmp/meshcache', kmax=KMAX):
        self.world = world if world is not None else make_world(map_static, assets_dir, cache_dir)
        self.kmax = int(kmax)
        self.device = torch.device(device)
        if self.device.type == 'cuda' and self.device.index is None:
            self.device = torch.device('cuda', torch.cuda.current_device())
        self.whitebox = bool(map_static.get('boxes')) if (whitebox is None and map_static is not None) else bool(whitebox)
        self.wall_check_default = self.whitebox
        rec, tb, flat, start, geo, lad = pack_world(self.world)
        assert len(flat) < 2 ** 31
        dv = self.device
        self.tri = torch.as_tensor(rec, device=dv).contiguous()
        self.tbox = torch.as_tensor(tb, device=dv).contiguous()
        self.flat = torch.as_tensor(flat, device=dv).contiguous()
        self.start = torch.as_tensor(start, device=dv).contiguous()
        self.lad = torch.as_tensor(lad, device=dv).contiguous()
        self.geo = geo
        self.wt = [self.tri, self.tbox, self.flat, self.start, self.lad]
        self.ladder_ids = [l['id'] for l in self.world.ladders]
        self.err = torch.zeros(1, dtype=torch.int32, device=dv)
        self._cbuf = torch.empty(0, dtype=torch.int32, device=dv)
        self._none = torch.empty(0, dtype=torch.int32, device=dv)
        self._nbuf = torch.zeros(0, dtype=torch.int32, device=dv)
        _ext()

    # ------------------------------------------------------------------------------------------------------------------
    def new_state(self, shape):
        dv = self.device
        s = {k: torch.zeros(shape, dtype=torch.float64, device=dv) for k in STATE_F64}
        s.update({k: torch.zeros(shape, dtype=torch.bool, device=dv) for k in STATE_BOOL})
        s.update({k: torch.zeros(shape, dtype=dt, device=dv) for k, dt in STATE_INT})
        s['height'].fill_(MM.STAND_HEIGHT); s['grounded'].fill_(True); s['alive'].fill_(True); s['ladder_id'].fill_(-1)
        return s

    def ensure_state(self, state):
        """add the module-private keys (and height / step_distance) to a state dict that lacks them"""
        x = state['x']
        for k in STATE_F64:
            if k not in state:
                state[k] = torch.full_like(x, MM.STAND_HEIGHT if k == 'height' else 0.0)
        for k in STATE_BOOL:
            if k not in state: state[k] = torch.zeros_like(x, dtype=torch.bool)
        for k, dt in STATE_INT:
            if k not in state: state[k] = torch.full_like(x, -1 if k == 'ladder_id' else 0, dtype=dt)
        return state

    def prep_action(self, action, shape=None):
        """convert an action dict once to the kernel dtypes (step() then does no copies)"""
        dv = self.device
        out = {}
        for k in ACT_F64:
            if action.get(k) is not None: out[k] = self._cv(action[k], torch.float64, shape)
        for k in ACT_BOOL:
            if action.get(k) is not None: out[k] = self._cv(action[k], torch.bool, shape)
        if action.get('ladder') is not None: out['ladder'] = self._cv(action['ladder'], torch.int8, shape)
        return out

    def _cv(self, v, dt, shape):
        t = torch.as_tensor(v, device=self.device).to(dt)
        if shape is not None: t = t.expand(shape)
        return t.contiguous()

    def step(self, state, action, maxspeed, locked=None, tick=0, freeze=None):
        """advance every alive player of `state` by one tick, in place (see module docstring); returns state"""
        x = state['x']; shape = x.shape; dv = x.device
        P = x.numel()
        self.ensure_state(state)
        st = []
        for k in STATE_F64:
            t = state[k]
            assert t.dtype == torch.float64 and t.is_contiguous() and t.shape == shape, k
            st.append(t)
        for k in STATE_BOOL:
            t = state[k]; assert t.dtype == torch.bool and t.is_contiguous() and t.shape == shape, k; st.append(t)
        for k, dt in STATE_INT:
            t = state[k]; assert t.dtype == dt and t.is_contiguous() and t.shape == shape, k; st.append(t)

        def f64(v):
            if v is None: return torch.zeros(shape, dtype=torch.float64, device=dv)
            return self._cv(v, torch.float64, shape)

        def b(v):
            if v is None: return torch.zeros(shape, dtype=torch.bool, device=dv)
            return self._cv(v, torch.bool, shape)
        lad = action.get('ladder')
        ac = [f64(action.get(k)) for k in ACT_F64] + [f64(maxspeed)] + [b(action.get(k)) for k in ACT_BOOL] + [b(locked)]
        ac.append(torch.empty(0, dtype=torch.bool, device=dv) if freeze is None else b(freeze))
        ac.append(torch.zeros(shape, dtype=torch.int8, device=dv) if lad is None else self._cv(lad, torch.int8, shape))
        tick_t, tick_div, tick0 = self._none, 1, 0
        if isinstance(tick, (int, np.integer)): tick0 = int(tick)
        else:
            tt = torch.as_tensor(tick, device=dv).to(torch.int32).contiguous()
            if tt.numel() == 1: tick0 = int(tt.item())
            elif tt.numel() == P: tick_t, tick_div = tt, 1
            else:
                assert shape[-1] > 0 and tt.numel() * shape[-1] == P, 'tick must be a scalar, [E] or like the state'
                tick_t, tick_div = tt, shape[-1]
        if self._cbuf.numel() < self.kmax * P:
            self._cbuf = torch.empty(self.kmax * P, dtype=torch.int32, device=dv)
        if self._nbuf.numel() < P: self._nbuf = torch.zeros(P, dtype=torch.int32, device=dv)
        _ext().step(self.wt, self.geo, st, ac, tick_t, tick_div, self._cbuf, self._nbuf, self.err, tick0, self.kmax)
        return state

    def resolve_players(self, state, locked=None, wall_check=None, swept=False):
        """player-vs-player pass after step(), see gpusim.collide.resolve_players"""
        from .collide import resolve_players
        return resolve_players(state, locked, self, wall_check, swept)

    def candidates(self, x, y, z, vx, vy, vz):
        """debug: number of candidate triangles gather() finds for the given player states (same box as step())"""
        P = x.numel()
        if self._cbuf.numel() < self.kmax * P: self._cbuf = torch.empty(self.kmax * P, dtype=torch.int32, device=x.device)
        pv = torch.stack([x, y, z, vx, vy, vz], -1).reshape(-1, 6).contiguous()
        return _ext().dbg_cand(self.wt, self.geo, pv, self._cbuf, self.err, self.kmax)

    def errors(self):
        """bit 1: candidate overflow (KMAX), 2: too many broadphase cells, 4: contact overflow (MAXC).  Sticky; reset_errors()."""
        return int(self.err.item())

    def reset_errors(self):
        self.err.zero_()
